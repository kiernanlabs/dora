"""Main runner for the Kalshi market making bot."""

import sys
import os
import time
import asyncio
import signal
import base64
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization

from dora_bot.kalshi_client import KalshiHttpClient, Environment
from dora_bot.models import TargetOrder, Order, MarketConfig
from dora_bot.exchange_client import KalshiExchangeClient
from dora_bot.dynamo import DynamoDBClient
from dora_bot.state_manager import StateManager
from dora_bot.risk_manager import RiskManager
from dora_bot.strategy import MarketMaker
from dora_bot.structured_logger import (
    setup_structured_logging,
    get_logger,
    set_context,
    clear_context,
    generate_decision_id,
    get_run_id,
    get_version,
    EventType,
)

# Module-level logger - will be configured in main()
logger = get_logger(__name__)


class DoraBot:
    """Main trading bot orchestrator."""

    def __init__(self, use_demo: bool = True, aws_region: str = "us-east-1"):
        """Initialize the bot.

        Args:
            use_demo: If True, use demo environment
            aws_region: AWS region for DynamoDB
        """
        self.environment = "demo" if use_demo else "prod"
        self.bot_run_id = get_run_id()
        self.bot_version = get_version()

        logger.info("Initializing Dora Bot", extra={
            "event_type": EventType.STARTUP,
            "bot_version": self.bot_version,
            "bot_run_id": self.bot_run_id,
        })

        # Determine if running in container mode
        container_mode = os.getenv('CONTAINER_MODE', 'false').lower() == 'true'

        if container_mode:
            # Container mode: load credentials from environment variables
            logger.info("Running in container mode", extra={
                "event_type": EventType.STARTUP,
                "container_mode": True,
            })
            keyid = os.getenv('KALSHI_KEY_ID')
            private_key_b64 = os.getenv('KALSHI_PRIVATE_KEY')

            if not keyid or not private_key_b64:
                raise ValueError("Missing KALSHI_KEY_ID or KALSHI_PRIVATE_KEY environment variables")

            # Decode base64-encoded private key
            try:
                private_key_pem = base64.b64decode(private_key_b64)
                private_key = serialization.load_pem_private_key(
                    private_key_pem,
                    password=None
                )
            except Exception as e:
                raise ValueError(f"Failed to decode private key: {e}")
        else:
            # Local mode: load credentials from .env file
            kalshi_dir = os.path.join(os.path.dirname(__file__), '..')
            load_dotenv(os.path.join(kalshi_dir, '.env'))

            keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
            keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')

            if not keyid or not keyfile:
                raise ValueError("Missing API credentials in .env file")

            # Resolve keyfile path relative to kalshi directory
            keyfile_path = os.path.join(kalshi_dir, keyfile)

            if not os.path.exists(keyfile_path):
                raise FileNotFoundError(f"Private key file not found: {keyfile_path}")

            with open(keyfile_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None
                )

        # Initialize Kalshi client
        env = Environment.DEMO if use_demo else Environment.PROD

        kalshi_client = KalshiHttpClient(
            key_id=keyid,
            private_key=private_key,
            environment=env
        )

        # Initialize components
        self.exchange = KalshiExchangeClient(kalshi_client, bot_run_id=self.bot_run_id)
        self.dynamo = DynamoDBClient(region=aws_region, environment=self.environment)
        self.state = StateManager(self.dynamo, bot_run_id=self.bot_run_id)
        self.strategy = MarketMaker()

        # Load global config and initialize risk manager
        self.global_config = self.dynamo.get_global_config()
        self.risk = RiskManager(self.global_config, bot_run_id=self.bot_run_id)

        # Config refresh counter
        self.loop_count = 0
        self.config_refresh_interval = 10  # Reload configs every N loops

        logger.info("Bot initialized", extra={
            "event_type": EventType.STARTUP,
            "mode": "DEMO" if use_demo else "PROD",
            "aws_region": aws_region,
        })

    async def startup(self):
        """Perform startup sequence."""
        logger.info("Startup sequence beginning", extra={
            "event_type": EventType.STARTUP,
            "phase": "begin",
        })

        # 1. Verify exchange connectivity
        try:
            status = self.exchange.get_exchange_status()
            logger.info("Exchange connected", extra={
                "event_type": EventType.STARTUP,
                "exchange_active": status.get('exchange_active', 'unknown'),
            })
        except Exception as e:
            logger.error("Failed to connect to exchange", extra={
                "event_type": EventType.ERROR,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }, exc_info=True)
            raise

        # 2. Load state from DynamoDB
        if not self.state.load_from_dynamo():
            logger.warning("Failed to load state from DynamoDB, starting fresh", extra={
                "event_type": EventType.STATE_LOAD,
                "success": False,
            })

        # 3. Fetch current open orders from exchange
        exchange_orders = self.exchange.get_open_orders()
        logger.info("Fetched open orders from exchange", extra={
            "event_type": EventType.STARTUP,
            "open_orders_count": len(exchange_orders),
        })

        # 4. Reconcile state
        self.state.reconcile_with_exchange(exchange_orders)

        # 5. Cancel all orders on startup if configured
        if self.global_config.cancel_on_startup:
            logger.info("Cancelling all orders on startup", extra={
                "event_type": EventType.STARTUP,
                "cancel_on_startup": True,
            })
            cancelled = self.exchange.cancel_all_orders()
            logger.info("Cancelled orders on startup", extra={
                "event_type": EventType.ORDER_CANCEL,
                "cancelled_count": cancelled,
            })
            self.state.open_orders.clear()

        # 6. Fetch any fills we may have missed during downtime
        # Fetch fills from 24 hours before the last fill to now to catch any missed fills
        if self.state.risk_state.last_fill_timestamp:
            from datetime import timedelta, timezone

            # Fetch from 24 hours before the last fill to ensure we catch everything
            fetch_from = self.state.risk_state.last_fill_timestamp - timedelta(hours=24)
            logger.info("Fetching fills since last recorded", extra={
                "event_type": EventType.STARTUP,
                "fetch_from": fetch_from.isoformat(),
            })
            fills = self.exchange.get_fills(since=fetch_from)

            # The logged_fills set will prevent duplicate logging
            self.state.update_from_fills(fills)
            logger.info("Processed historical fills", extra={
                "event_type": EventType.STARTUP,
                "fills_count": len(fills),
            })
        else:
            logger.info("No previous fills, fetching all", extra={
                "event_type": EventType.STARTUP,
            })
            fills = self.exchange.get_fills()
            self.state.update_from_fills(fills)
            logger.info("Processed historical fills", extra={
                "event_type": EventType.STARTUP,
                "fills_count": len(fills),
            })

        # 7. Save positions to DynamoDB after reconciliation
        self.state.save_to_dynamo()

        # 8. Log current state
        summary = self.state.get_state_summary()
        logger.info("State reconciled", extra={
            "event_type": EventType.STARTUP,
            "state_summary": summary,
        })

        # 9. Check balance (optional, doesn't block startup if it fails)
        balance = self.exchange.get_balance()
        if balance:
            logger.info("Account balance fetched", extra={
                "event_type": EventType.STARTUP,
                "balance": balance.balance,
                "payout": balance.payout,
                "total": balance.total,
            })
        else:
            logger.warning("Could not fetch account balance", extra={
                "event_type": EventType.STARTUP,
            })

        logger.info("Startup complete", extra={
            "event_type": EventType.STARTUP,
            "phase": "complete",
            "bot_run_id": self.bot_run_id,
            "bot_version": self.bot_version,
        })

    async def run_loop(self):
        """Main event loop."""
        logger.info("Starting main event loop", extra={
            "event_type": EventType.STARTUP,
        })

        while True:
            loop_start = time.time()

            try:
                # Check if we should halt trading
                should_halt, halt_reason = self.risk.should_halt_trading(self.state)
                if should_halt:
                    logger.critical("Trading halted", extra={
                        "event_type": EventType.RISK_HALT,
                        "reason": halt_reason,
                        "daily_pnl": self.state.risk_state.daily_pnl,
                    })
                    # Cancel all orders and exit
                    self.exchange.cancel_all_orders()
                    self.state.save_to_dynamo()
                    break

                # Refresh configs periodically
                if self.loop_count % self.config_refresh_interval == 0:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)
                    logger.info("Refreshed market configs", extra={
                        "event_type": EventType.CONFIG_REFRESH,
                        "enabled_markets": len(market_configs),
                        "loop_count": self.loop_count,
                    })
                else:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)

                # Process each enabled market
                for market_id, config in market_configs.items():
                    # Generate decision_id for this market processing cycle
                    decision_id = generate_decision_id(self.bot_run_id, market_id, self.loop_count)
                    set_context(decision_id=decision_id, market=market_id)
                    await self.process_market(market_id, config, decision_id)
                    clear_context()

                # Process fills
                fills = self.exchange.get_fills(since=self.state.risk_state.last_fill_timestamp)
                if fills:
                    num_new_fills = self.state.update_from_fills(fills) or 0
                    if num_new_fills > 0:
                        logger.info("Processed fills", extra={
                            "event_type": EventType.FILL,
                            "new_fills": num_new_fills,
                            "daily_pnl": self.state.risk_state.daily_pnl,
                        })

                # Periodic state persistence (every 10 loops)
                if self.loop_count % 10 == 0:
                    self.state.save_to_dynamo()
                    # Emit heartbeat
                    logger.info("Heartbeat", extra={
                        "event_type": EventType.HEARTBEAT,
                        "loop_count": self.loop_count,
                        "markets_active": len(market_configs),
                        "open_orders_count": len(self.state.open_orders),
                        "daily_pnl": self.state.risk_state.daily_pnl,
                    })

                self.loop_count += 1

                # Sleep for configured interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.global_config.loop_interval_ms / 1000.0 - loop_duration)

                if loop_duration > self.global_config.loop_interval_ms / 1000.0:
                    logger.warning("Loop exceeded target duration", extra={
                        "event_type": EventType.LOG,
                        "loop_duration_ms": loop_duration * 1000,
                        "target_ms": self.global_config.loop_interval_ms,
                    })

                await asyncio.sleep(sleep_time)

                # Check for graceful shutdown signal (SIGTERM from container orchestrator)
                if shutdown_handler.shutdown_requested:
                    logger.info("Graceful shutdown requested", extra={
                        "event_type": EventType.SHUTDOWN,
                        "reason": "signal",
                    })
                    break

            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt", extra={
                    "event_type": EventType.SHUTDOWN,
                    "reason": "keyboard_interrupt",
                })
                break
            except Exception as e:
                logger.error("Error in main loop", extra={
                    "event_type": EventType.ERROR,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                }, exc_info=True)
                # Consider canceling all orders on unexpected errors
                time.sleep(5)  # Brief pause before retrying

        # Cleanup
        logger.info("Shutting down", extra={
            "event_type": EventType.SHUTDOWN,
            "phase": "cleanup",
        })
        self.state.save_to_dynamo()
        logger.info("Shutdown complete", extra={
            "event_type": EventType.SHUTDOWN,
            "phase": "complete",
            "graceful": True,
        })

    async def process_market(self, market_id: str, config: MarketConfig, decision_id: str):
        """Process a single market.

        Args:
            market_id: Market ticker
            config: Market configuration
            decision_id: Unique identifier for this decision cycle
        """
        try:
            # 1. Fetch order book
            order_book = self.exchange.get_order_book(market_id)

            # 2. Get current position
            position = self.state.get_inventory(market_id)

            # 3. Compute target quotes
            target_orders = self.strategy.compute_quotes(order_book, position, config)

            # Handle None or empty target_orders
            if target_orders is None:
                logger.warning("compute_quotes returned None", extra={
                    "event_type": EventType.LOG,
                    "market": market_id,
                    "decision_id": decision_id,
                })
                target_orders = []

            # Log decision event
            target_quotes = [
                {'side': t.side, 'price': t.price, 'size': t.size}
                for t in target_orders
            ] if target_orders else []

            logger.info("Decision made", extra={
                "event_type": EventType.DECISION_MADE,
                "market": market_id,
                "decision_id": decision_id,
                "best_bid": order_book.best_bid,
                "best_ask": order_book.best_ask,
                "spread": order_book.spread,
                "mid": order_book.mid_price,
                "net_yes_qty": position.net_yes_qty,
                "target_count": len(target_orders),
                "target_quotes": target_quotes,
            })

            # Log decision to DynamoDB (for historical record)
            self.dynamo.log_decision({
                'market_id': market_id,
                'decision_id': decision_id,
                'bot_run_id': self.bot_run_id,
                'bot_version': self.bot_version,
                'order_book_snapshot': {
                    'best_bid': order_book.best_bid,
                    'best_ask': order_book.best_ask,
                    'spread': order_book.spread,
                    'mid': order_book.mid_price
                },
                'inventory': {
                    'net_yes_qty': position.net_yes_qty
                },
                'target_quotes': target_quotes,
                'num_targets': len(target_orders) if target_orders else 0
            })

            if not target_orders:
                # No quotes desired, cancel any existing orders
                existing_orders = self.state.get_open_orders_for_market(market_id)
                for order in existing_orders:
                    self.exchange.cancel_order(order.order_id, decision_id=decision_id)
                    self.state.remove_order(order.order_id)
                return

            # 4. Diff with existing orders
            existing_orders = self.state.get_open_orders_for_market(market_id)
            to_cancel, to_place = self.diff_orders(existing_orders, target_orders)

            # 5. Cancel orders
            for order in to_cancel:
                if self.exchange.cancel_order(order.order_id, decision_id=decision_id):
                    self.state.remove_order(order.order_id)

            # 6. Place new orders (with risk checks)
            for target in to_place:
                allowed, reason = self.risk.check_order(target, position, config, self.state)

                if allowed:
                    placed_order = self.exchange.place_order(
                        market_id=target.market_id,
                        side=target.side,
                        price=target.price,
                        size=target.size,
                        decision_id=decision_id
                    )

                    if placed_order:
                        self.state.record_order(placed_order)
                else:
                    logger.debug("Order blocked by risk", extra={
                        "event_type": EventType.LOG,
                        "market": market_id,
                        "decision_id": decision_id,
                        "side": target.side,
                        "price": target.price,
                        "size": target.size,
                        "reason": reason,
                    })

        except Exception as e:
            logger.error("Error processing market", extra={
                "event_type": EventType.ERROR,
                "market": market_id,
                "decision_id": decision_id,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }, exc_info=True)

    def diff_orders(
        self,
        existing: List[Order],
        targets: List[TargetOrder]
    ) -> tuple[List[Order], List[TargetOrder]]:
        """Diff existing orders vs target orders.

        Args:
            existing: List of existing orders
            targets: List of target orders

        Returns:
            Tuple of (orders_to_cancel, orders_to_place)
        """
        to_cancel = []
        to_place = list(targets)  # Start with all targets

        for order in existing:
            # Check if this order matches any target
            matched = False
            for target in targets:
                if target.matches(order, tolerance=0.01):  # 1 tick tolerance
                    matched = True
                    to_place.remove(target)  # Don't need to place this
                    break

            if not matched:
                to_cancel.append(order)

        return to_cancel, to_place


class GracefulShutdown:
    """Handle graceful shutdown signals for container environments."""

    def __init__(self):
        self.shutdown_requested = False
        self.bot: Optional[DoraBot] = None

    def register_signals(self):
        """Register signal handlers for SIGTERM and SIGINT."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("Registered signal handlers", extra={
            "event_type": EventType.STARTUP,
        })

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        sig_name = signal.Signals(signum).name
        logger.info("Received shutdown signal", extra={
            "event_type": EventType.SHUTDOWN,
            "signal": sig_name,
        })
        self.shutdown_requested = True

    def set_bot(self, bot: DoraBot):
        """Set the bot instance for shutdown handling."""
        self.bot = bot


# Global shutdown handler
shutdown_handler = GracefulShutdown()


async def main():
    """Main entry point."""
    # Determine mode from environment or command line
    # Environment variable takes precedence (for container mode)
    use_demo_env = os.getenv('USE_DEMO', '').lower()
    if use_demo_env:
        use_demo = use_demo_env == 'true'
    else:
        # Fall back to command line args
        use_demo = True  # Default to demo
        if len(sys.argv) > 1 and sys.argv[1] == '--prod':
            use_demo = False

    env = "demo" if use_demo else "prod"

    # Initialize structured logging BEFORE creating bot
    bot_run_id = setup_structured_logging(
        service='dora-bot',
        env=env,
    )

    logger.info("Dora Bot starting", extra={
        "event_type": EventType.STARTUP,
        "mode": "DEMO" if use_demo else "PROD",
        "bot_run_id": bot_run_id,
    })

    # Register signal handlers
    shutdown_handler.register_signals()

    # Get AWS region from environment or default
    aws_region = os.getenv('AWS_REGION', 'us-east-1')

    # Create and run bot
    bot = DoraBot(use_demo=use_demo, aws_region=aws_region)
    shutdown_handler.set_bot(bot)

    try:
        await bot.startup()
        await bot.run_loop()
    except Exception as e:
        logger.critical("Fatal error", extra={
            "event_type": EventType.ERROR,
            "error_type": type(e).__name__,
            "error_msg": str(e),
        }, exc_info=True)
        # Attempt graceful cleanup
        if bot:
            try:
                logger.info("Attempting graceful cleanup after fatal error", extra={
                    "event_type": EventType.SHUTDOWN,
                })
                bot.exchange.cancel_all_orders()
                bot.state.save_to_dynamo()
            except Exception as cleanup_error:
                logger.error("Cleanup failed", extra={
                    "event_type": EventType.ERROR,
                    "error_type": type(cleanup_error).__name__,
                    "error_msg": str(cleanup_error),
                })
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
