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

from requests.exceptions import HTTPError

from collections import defaultdict
import secrets

from dora_bot.kalshi_client import KalshiHttpClient, Environment
from dora_bot.models import TargetOrder, Order, MarketConfig, OrderRequest, BatchCancelSummary, BatchPlaceSummary
from dora_bot.exchange_client import KalshiExchangeClient
from dora_bot.rate_limiter import RateLimiter
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
    log_decision_record,
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
        self.aws_region = aws_region
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
        self.exchange = KalshiExchangeClient(
            kalshi_client,
            bot_run_id=self.bot_run_id,
            environment=self.environment,
            aws_region=aws_region,
        )
        self.dynamo = DynamoDBClient(region=aws_region, environment=self.environment)
        self.state = StateManager(self.dynamo, bot_run_id=self.bot_run_id)
        self.strategy = MarketMaker()

        # Load global config and initialize risk manager
        self.global_config = self.dynamo.get_global_config()
        self.risk = RiskManager(self.global_config, bot_run_id=self.bot_run_id)

        # Config refresh counter
        self.loop_count = 0
        self.config_refresh_interval = 5  # Reload configs every N loops

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

        # 2. Load risk state and logged fills from DynamoDB
        # We'll rebuild positions from fills, but keep the risk state (daily PnL, etc.)
        self.state.risk_state = self.dynamo.get_risk_state()
        self.state.logged_fills = self.dynamo.get_all_fill_ids()
        logger.info("Loaded risk state and logged fills from DynamoDB", extra={
            "event_type": EventType.STATE_LOAD,
            "logged_fills_count": len(self.state.logged_fills),
            "daily_pnl": self.state.risk_state.daily_pnl,
            "trading_halted": self.state.risk_state.trading_halted,
        })

        # 3. Fetch current open orders from exchange
        exchange_orders = self.exchange.get_open_orders()
        logger.info("Fetched open orders from exchange", extra={
            "event_type": EventType.STARTUP,
            "open_orders_count": len(exchange_orders),
        })

        # 4. Reconcile open orders state
        self.state.reconcile_with_exchange(exchange_orders)

        # 5. Cancel all orders on startup if configured
        if self.global_config.cancel_on_startup and exchange_orders:
            startup_decision_id = f"{self.bot_run_id}:startup_cancel:{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            log_decision_record({
                "market_id": "ALL",
                "decision_id": startup_decision_id,
                "bot_run_id": self.bot_run_id,
                "bot_version": self.bot_version,
                "reason": "startup_cancel",
                "open_orders_count": len(exchange_orders),
                "target_quotes": [],
                "num_targets": 0,
            }, region=self.aws_region, environment=self.environment)
            logger.info("Cancelling all orders on startup", extra={
                "event_type": EventType.STARTUP,
                "cancel_on_startup": True,
                "decision_id": startup_decision_id,
                "open_orders_count": len(exchange_orders),
                "use_batch": self.global_config.use_batch_execution,
            })

            if self.global_config.use_batch_execution:
                # Use batch cancellation for efficiency
                cancelled = self._cancel_orders_batch_sync(exchange_orders, startup_decision_id)
            else:
                # Use sequential cancellation (old behavior)
                cancelled = self.exchange.cancel_all_orders(decision_id=startup_decision_id)

            logger.info("Cancelled orders on startup", extra={
                "event_type": EventType.ORDER_CANCEL,
                "cancelled_count": cancelled,
                "decision_id": startup_decision_id,
            })
            self.state.open_orders.clear()

        # 6. Fetch ALL fills and rebuild positions from scratch
        # This ensures positions are correct even if previous saves failed
        logger.info("Fetching all fills to reconcile positions", extra={
            "event_type": EventType.STARTUP,
        })
        fills = self.exchange.get_fills()
        logger.info("Fetched fills from exchange", extra={
            "event_type": EventType.STARTUP,
            "fills_count": len(fills),
        })

        # Reconcile positions by processing all fills from scratch
        # This ignores logged_fills for position calculation but won't duplicate logs
        processed_count = self.state.reconcile_positions_from_fills(fills)
        logger.info("Reconciled positions from fills", extra={
            "event_type": EventType.STARTUP,
            "fills_processed": processed_count,
            "positions_count": len(self.state.positions),
        })

        # 7. Save reconciled positions to DynamoDB
        self.state.save_to_dynamo()

        # 8. Log current state
        summary = self.state.get_state_summary()
        logger.info("State reconciled", extra={
            "event_type": EventType.STARTUP,
            "state_summary": summary,
        })

        # 9. Check balance
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
                    halt_decision_id = (
                        f"{self.bot_run_id}:halt_cancel:{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                    )
                    log_decision_record({
                        "market_id": "ALL",
                        "decision_id": halt_decision_id,
                        "bot_run_id": self.bot_run_id,
                        "bot_version": self.bot_version,
                        "reason": "risk_halt_cancel",
                        "halt_reason": halt_reason,
                        "daily_pnl": self.state.risk_state.daily_pnl,
                        "target_quotes": [],
                        "num_targets": 0,
                    }, region=self.aws_region, environment=self.environment)
                    logger.critical("Trading halted", extra={
                        "event_type": EventType.RISK_HALT,
                        "reason": halt_reason,
                        "daily_pnl": self.state.risk_state.daily_pnl,
                        "decision_id": halt_decision_id,
                    })
                    # Cancel all orders and exit
                    self.exchange.cancel_all_orders(decision_id=halt_decision_id)
                    self.state.save_to_dynamo()
                    break

                # Refresh configs and reconcile with exchange periodically
                if self.loop_count % self.config_refresh_interval == 0:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)
                    logger.info("Refreshed market configs", extra={
                        "event_type": EventType.CONFIG_REFRESH,
                        "enabled_markets": len(market_configs),
                        "loop_count": self.loop_count,
                    })

                    # Reconcile open orders with exchange to fix any state drift
                    all_exchange_orders = self.exchange.get_open_orders()
                    drift_stats = self.state.reconcile_with_exchange(all_exchange_orders, log_drift=True)

                    # Log if significant drift detected
                    if drift_stats["orders_only_local"] or drift_stats["orders_only_exchange"]:
                        logger.warning("Order state drift detected during reconciliation", extra={
                            "event_type": EventType.LOG,
                            "loop_count": self.loop_count,
                            "stale_local_orders": len(drift_stats["orders_only_local"]),
                            "untracked_exchange_orders": len(drift_stats["orders_only_exchange"]),
                            "matched_orders": drift_stats["orders_matched"],
                        })
                else:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)
                    # Fetch all exchange orders once for safeguard checks
                    all_exchange_orders = self.exchange.get_open_orders()

                # Group exchange orders by market for efficient lookup in process_market
                exchange_orders_by_market: Dict[str, List[Order]] = {}
                for order in all_exchange_orders:
                    if order.market_id not in exchange_orders_by_market:
                        exchange_orders_by_market[order.market_id] = []
                    exchange_orders_by_market[order.market_id].append(order)

                # Process each enabled market
                for market_id, config in market_configs.items():
                    # Generate decision_id for this market processing cycle
                    decision_id = generate_decision_id(self.bot_run_id, market_id, self.loop_count)
                    set_context(decision_id=decision_id, market=market_id)
                    market_orders = exchange_orders_by_market.get(market_id, [])
                    await self.process_market(market_id, config, decision_id, market_orders)
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

    async def process_market(self, market_id: str, config: MarketConfig, decision_id: str,
                             exchange_orders_for_market: List[Order]):
        """Process a single market.

        Args:
            market_id: Market ticker
            config: Market configuration
            decision_id: Unique identifier for this decision cycle
            exchange_orders_for_market: Pre-fetched list of open orders on exchange for this market
        """
        try:
            # 1. Fetch order book (pass pre-fetched orders to avoid extra API call)
            order_book = self.exchange.get_order_book(market_id, own_orders=exchange_orders_for_market)

            # 2. Get current position
            position = self.state.get_inventory(market_id)

            # 2b. Fetch recent trades for fair value
            trades = []
            try:
                trades = self.exchange.get_trades(market_id, limit=10)
            except Exception as e:
                logger.warning("Failed to fetch trades - falling back to mid price", extra={
                    "event_type": EventType.LOG,
                    "market": market_id,
                    "decision_id": decision_id,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                })

            # 3. Compute target quotes
            target_orders, price_calc = self.strategy.compute_quotes(order_book, position, config, trades)

            # Handle None or empty target_orders
            if target_orders is None:
                logger.warning("compute_quotes returned None", extra={
                    "event_type": EventType.LOG,
                    "market": market_id,
                    "decision_id": decision_id,
                })
                target_orders = []
            if price_calc is None:
                price_calc = {}

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
            log_decision_record({
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
                'price_calc': price_calc,
                'target_quotes': target_quotes,
                'num_targets': len(target_orders) if target_orders else 0
            }, region=self.aws_region, environment=self.environment)

            if not target_orders:
                # No quotes desired, cancel any existing orders
                existing_orders = self.state.get_open_orders_for_market(market_id)
                for order in existing_orders:
                    self.exchange.cancel_order(
                        order.order_id,
                        decision_id=decision_id,
                        market_id=order.market_id,
                        client_order_id=order.client_order_id,
                    )
                    self.state.remove_order(order.order_id)
                return

            # 4. Diff with existing orders
            existing_orders = self.state.get_open_orders_for_market(market_id)
            to_cancel, to_place = self.diff_orders(existing_orders, target_orders)

            # Debug: Log the full state before diff decision
            logger.debug("Order diff inputs", extra={
                "event_type": EventType.LOG,
                "market": market_id,
                "decision_id": decision_id,
                "existing_orders": [
                    {"order_id": o.order_id, "side": o.side, "price": o.price, "size": o.size}
                    for o in existing_orders
                ],
                "target_orders": [
                    {"side": t.side, "price": t.price, "size": t.size}
                    for t in target_orders
                ],
                "to_cancel_ids": [o.order_id for o in to_cancel],
                "to_place_count": len(to_place),
            })

            # Log order diff results
            if not to_cancel and not to_place:
                logger.info("No order changes needed - existing orders match targets", extra={
                    "event_type": EventType.LOG,
                    "market": market_id,
                    "decision_id": decision_id,
                    "existing_orders_count": len(existing_orders),
                    "target_orders_count": len(target_orders),
                })
            else:
                logger.info("Order diff complete", extra={
                    "event_type": EventType.LOG,
                    "market": market_id,
                    "decision_id": decision_id,
                    "to_cancel_count": len(to_cancel),
                    "to_place_count": len(to_place),
                })

            # 5. Cancel orders
            for order in to_cancel:
                if self.exchange.cancel_order(
                    order.order_id,
                    decision_id=decision_id,
                    market_id=order.market_id,
                    client_order_id=order.client_order_id,
                ):
                    self.state.remove_order(order.order_id)

            # 6. Place new orders (with risk checks)
            # SAFEGUARD: Verify we don't already have a bid or ask on the market
            # Constraint: Only one bid and one ask allowed per market at a time
            # Uses pre-fetched exchange_orders_for_market to avoid extra API calls
            if to_place:
                # Check if we already have a bid (side='yes') or ask (side='no') on exchange
                has_bid_on_exchange = any(o.side == 'yes' for o in exchange_orders_for_market)
                has_ask_on_exchange = any(o.side == 'no' for o in exchange_orders_for_market)

                # Sync local state with exchange orders we found
                for exchange_order in exchange_orders_for_market:
                    if exchange_order.order_id not in self.state.open_orders:
                        self.state.record_order(exchange_order)

                filtered_to_place = []
                for target in to_place:
                    # target.side is 'bid' or 'ask', exchange uses 'yes' or 'no'
                    is_bid = target.side == 'bid'

                    if is_bid and has_bid_on_exchange:
                        logger.warning("Skipping bid - already have bid order on exchange", extra={
                            "event_type": EventType.LOG,
                            "market": market_id,
                            "decision_id": decision_id,
                            "target_price": target.price,
                            "target_size": target.size,
                            "existing_bids": [{"order_id": o.order_id, "price": o.price, "size": o.size}
                                              for o in exchange_orders_for_market if o.side == 'yes'],
                        })
                    elif not is_bid and has_ask_on_exchange:
                        logger.warning("Skipping ask - already have ask order on exchange", extra={
                            "event_type": EventType.LOG,
                            "market": market_id,
                            "decision_id": decision_id,
                            "target_price": target.price,
                            "target_size": target.size,
                            "existing_asks": [{"order_id": o.order_id, "price": o.price, "size": o.size}
                                              for o in exchange_orders_for_market if o.side == 'no'],
                        })
                    else:
                        filtered_to_place.append(target)

                to_place = filtered_to_place

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
                    logger.warning("Order blocked by risk manager", extra={
                        "event_type": EventType.RISK_HALT,
                        "market": market_id,
                        "decision_id": decision_id,
                        "side": target.side,
                        "price": target.price,
                        "size": target.size,
                        "blocked_reason": reason,
                    })

        except HTTPError as e:
            # Track API errors by market
            status_code = e.response.status_code if e.response is not None else None
            error_code = None
            error_msg = str(e)

            # Try to extract error code from response
            try:
                if e.response is not None:
                    error_data = e.response.json()
                    if isinstance(error_data, dict):
                        error_code = error_data.get('error', {}).get('code')
            except:
                pass

            self.state.record_api_error(
                market_id=market_id,
                status_code=status_code,
                error_code=error_code,
                error_msg=error_msg
            )

            logger.error("API error processing market", extra={
                "event_type": EventType.ERROR,
                "market": market_id,
                "decision_id": decision_id,
                "error_type": type(e).__name__,
                "error_msg": error_msg,
                "status_code": status_code,
                "error_code": error_code,
                "api_error_count": self.state.get_api_error_count(market_id),
            }, exc_info=True)

        except Exception as e:
            # Track non-HTTP errors as well
            self.state.record_api_error(
                market_id=market_id,
                status_code=None,
                error_code=type(e).__name__,
                error_msg=str(e)
            )

            logger.error("Error processing market", extra={
                "event_type": EventType.ERROR,
                "market": market_id,
                "decision_id": decision_id,
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "api_error_count": self.state.get_api_error_count(market_id),
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
            # Iterate over to_place (not targets) so that each target can only match one order.
            # This ensures duplicate orders on exchange get cancelled (only one matches).
            matched = False
            for target in to_place:
                if target.matches(order, tolerance=0.005):  # 0.5 tick tolerance
                    matched = True
                    to_place.remove(target)  # Don't need to place this
                    break

            if not matched:
                to_cancel.append(order)

        return to_cancel, to_place

    # ============================================================
    # Batch-based execution methods (new architecture)
    # ============================================================

    async def run_loop_batch(self):
        """Main event loop using batch-based order execution.

        This is the new architecture that:
        1. Reconciles with exchange at start of each loop (single source of truth)
        2. Computes all targets across all markets before execution
        3. Executes cancellations and placements in batches of 20
        """
        logger.info("Starting batch-based main event loop", extra={
            "event_type": EventType.STARTUP,
            "mode": "batch",
        })

        # Initialize rate limiter
        rate_limiter = RateLimiter(
            requests_per_second=10.0,
            burst_limit=20,
            max_backoff_seconds=30.0,
        )

        # Load initial market configs
        market_configs = self.dynamo.get_all_market_configs(enabled_only=True)

        while True:
            loop_start = time.time()

            try:
                # 1. Check risk limits
                should_halt, halt_reason = self.risk.should_halt_trading(self.state)
                if should_halt:
                    halt_decision_id = (
                        f"{self.bot_run_id}:halt_cancel:{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                    )
                    log_decision_record({
                        "market_id": "ALL",
                        "decision_id": halt_decision_id,
                        "bot_run_id": self.bot_run_id,
                        "bot_version": self.bot_version,
                        "reason": "risk_halt_cancel",
                        "halt_reason": halt_reason,
                        "daily_pnl": self.state.risk_state.daily_pnl,
                        "target_quotes": [],
                        "num_targets": 0,
                    }, region=self.aws_region, environment=self.environment)
                    logger.critical("Trading halted", extra={
                        "event_type": EventType.RISK_HALT,
                        "reason": halt_reason,
                        "daily_pnl": self.state.risk_state.daily_pnl,
                        "decision_id": halt_decision_id,
                    })
                    self.exchange.cancel_all_orders(decision_id=halt_decision_id)
                    self.state.save_to_dynamo()
                    break

                # 2. Refresh configs periodically
                if self.loop_count % self.config_refresh_interval == 0:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)
                    logger.info("Refreshed market configs", extra={
                        "event_type": EventType.CONFIG_REFRESH,
                        "enabled_markets": len(market_configs),
                        "loop_count": self.loop_count,
                    })

                # 3. Reconcile order state with exchange (SINGLE SOURCE OF TRUTH)
                rate_limiter.acquire(1, endpoint="get_open_orders")
                exchange_orders = self.exchange.get_open_orders()
                drift_stats = self.state.reconcile_with_exchange(exchange_orders, log_drift=True)

                if drift_stats["orders_only_local"] or drift_stats["orders_only_exchange"]:
                    logger.warning("Order state drift detected", extra={
                        "event_type": EventType.LOG,
                        "loop_count": self.loop_count,
                        "stale_local_orders": len(drift_stats["orders_only_local"]),
                        "untracked_exchange_orders": len(drift_stats["orders_only_exchange"]),
                    })

                # 4. Process fills and update positions
                rate_limiter.acquire(1, endpoint="get_fills")
                fills = self.exchange.get_fills(since=self.state.risk_state.last_fill_timestamp)
                if fills:
                    num_new_fills = self.state.update_from_fills(fills) or 0
                    if num_new_fills > 0:
                        logger.info("Processed fills", extra={
                            "event_type": EventType.FILL,
                            "new_fills": num_new_fills,
                            "daily_pnl": self.state.risk_state.daily_pnl,
                        })

                # 5. Calculate targets for ALL markets (pure computation)
                all_targets = self._compute_all_targets(market_configs, exchange_orders, rate_limiter)

                # 6. Diff to get cancellations and placements
                to_cancel, to_place = self._diff_all_orders(exchange_orders, all_targets)

                logger.info("Batch diff complete", extra={
                    "event_type": EventType.LOG,
                    "loop_count": self.loop_count,
                    "to_cancel_count": len(to_cancel),
                    "to_place_count": len(to_place),
                    "markets_with_targets": len([m for m, t in all_targets.items() if t]),
                })

                # 7. Execute cancellations in batches
                if to_cancel:
                    cancel_summary = await self._batch_cancel(to_cancel, rate_limiter)
                    if cancel_summary.failed > 0:
                        logger.warning("Some cancellations failed", extra={
                            "event_type": EventType.LOG,
                            "succeeded": cancel_summary.succeeded,
                            "failed": cancel_summary.failed,
                        })

                # 8. Execute placements in batches
                # Add delay between cancel and place phases to avoid rate limits
                if to_cancel and to_place:
                    CANCEL_TO_PLACE_DELAY_SECONDS = 1.0
                    logger.debug("Rate limit pause between cancel and place phases", extra={
                        "event_type": EventType.RATE_LIMIT_BACKOFF,
                        "delay_seconds": CANCEL_TO_PLACE_DELAY_SECONDS,
                        "cancel_count": len(to_cancel),
                        "place_count": len(to_place),
                    })
                    await asyncio.sleep(CANCEL_TO_PLACE_DELAY_SECONDS)

                if to_place:
                    place_summary = await self._batch_place(to_place, rate_limiter)

                    # Record successfully placed orders in local state
                    for order in place_summary.placed_orders:
                        self.state.record_order(order)

                    if place_summary.failed > 0:
                        logger.warning("Some placements failed", extra={
                            "event_type": EventType.LOG,
                            "succeeded": place_summary.succeeded,
                            "failed": place_summary.failed,
                        })

                # 9. Save state every loop now 
                self.state.save_to_dynamo()
                logger.info("Heartbeat", extra={
                    "event_type": EventType.HEARTBEAT,
                    "loop_count": self.loop_count,
                    "markets_active": len(market_configs),
                    "open_orders_count": len(self.state.open_orders),
                    "daily_pnl": self.state.risk_state.daily_pnl,
                    "rate_limiter": rate_limiter.get_stats(),
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

                # Check for graceful shutdown
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
                logger.error("Error in batch main loop", extra={
                    "event_type": EventType.ERROR,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                }, exc_info=True)
                await asyncio.sleep(5)

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

    def _compute_all_targets(
        self,
        market_configs: Dict[str, MarketConfig],
        exchange_orders: List[Order],
        rate_limiter: RateLimiter,
    ) -> Dict[str, List[TargetOrder]]:
        """Compute target orders for all enabled markets.

        Args:
            market_configs: Dict of market_id -> MarketConfig
            exchange_orders: Current orders on exchange
            rate_limiter: Rate limiter for API calls

        Returns:
            Dict mapping market_id -> list of target orders
        """
        all_targets: Dict[str, List[TargetOrder]] = {}

        # Group exchange orders by market
        orders_by_market: Dict[str, List[Order]] = defaultdict(list)
        for order in exchange_orders:
            orders_by_market[order.market_id].append(order)

        for market_id, config in market_configs.items():
            decision_id = generate_decision_id(self.bot_run_id, market_id, self.loop_count)
            set_context(decision_id=decision_id, market=market_id)

            try:
                # Fetch order book (requires API call)
                rate_limiter.acquire(1, endpoint=f"get_order_book:{market_id}")
                market_orders = orders_by_market.get(market_id, [])
                order_book = self.exchange.get_order_book(market_id, own_orders=market_orders)

                # Get position
                position = self.state.get_inventory(market_id)

                # Fetch recent trades for fair value
                trades = []
                try:
                    rate_limiter.acquire(1, endpoint=f"get_trades:{market_id}")
                    trades = self.exchange.get_trades(market_id, limit=50)
                except Exception as e:
                    logger.warning("Failed to fetch trades - falling back to mid price", extra={
                        "event_type": EventType.LOG,
                        "market": market_id,
                        "decision_id": decision_id,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                    })

                # Compute quotes (pure strategy computation)
                target_orders, price_calc = self.strategy.compute_quotes(
                    order_book, position, config, trades
                )

                all_targets[market_id] = target_orders or []

                # Log decision
                target_quotes = [
                    {'side': t.side, 'price': t.price, 'size': t.size}
                    for t in all_targets[market_id]
                ]
                logger.info("Decision made", extra={
                    "event_type": EventType.DECISION_MADE,
                    "market": market_id,
                    "decision_id": decision_id,
                    "best_bid": order_book.best_bid,
                    "best_ask": order_book.best_ask,
                    "spread": order_book.spread,
                    "mid": order_book.mid_price,
                    "net_yes_qty": position.net_yes_qty,
                    "target_count": len(all_targets[market_id]),
                    "target_quotes": target_quotes,
                })

                # Log to DynamoDB
                log_decision_record({
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
                    'price_calc': price_calc or {},
                    'target_quotes': target_quotes,
                    'num_targets': len(all_targets[market_id])
                }, region=self.aws_region, environment=self.environment)

            except Exception as e:
                logger.error("Error computing targets for market", extra={
                    "event_type": EventType.ERROR,
                    "market": market_id,
                    "decision_id": decision_id,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                }, exc_info=True)
                all_targets[market_id] = []

            clear_context()

        return all_targets

    def _diff_all_orders(
        self,
        exchange_orders: List[Order],
        all_targets: Dict[str, List[TargetOrder]],
    ) -> tuple[List[Order], List[OrderRequest]]:
        """Diff current exchange orders against all targets.

        Args:
            exchange_orders: Current orders on exchange
            all_targets: Dict of market_id -> target orders

        Returns:
            Tuple of (orders_to_cancel, orders_to_place)
        """
        to_cancel: List[Order] = []
        to_place: List[OrderRequest] = []

        # Group exchange orders by market
        orders_by_market: Dict[str, List[Order]] = defaultdict(list)
        for order in exchange_orders:
            orders_by_market[order.market_id].append(order)

        # Get all markets (union of exchange orders and targets)
        all_markets = set(orders_by_market.keys()) | set(all_targets.keys())

        for market_id in all_markets:
            existing = orders_by_market.get(market_id, [])
            targets = all_targets.get(market_id, [])

            # Use existing diff logic
            market_cancel, market_place = self.diff_orders(existing, targets)

            to_cancel.extend(market_cancel)

            # Convert TargetOrders to OrderRequests
            decision_id = generate_decision_id(self.bot_run_id, market_id, self.loop_count)
            for target in market_place:
                # Convert bid/ask to yes/no
                kalshi_side = "yes" if target.side == "bid" else "no"
                to_place.append(OrderRequest(
                    market_id=market_id,
                    side=kalshi_side,
                    price=int(target.price * 100),  # Convert to cents
                    size=target.size,
                    client_order_id=secrets.token_hex(8),
                    decision_id=decision_id,
                ))

        return to_cancel, to_place

    async def _batch_cancel(
        self,
        orders_to_cancel: List[Order],
        rate_limiter: RateLimiter,
    ) -> BatchCancelSummary:
        """Cancel orders in batches of 10.

        Args:
            orders_to_cancel: List of orders to cancel
            rate_limiter: Rate limiter for API calls

        Returns:
            BatchCancelSummary with results
        """
        BATCH_SIZE = 10
        BATCH_DELAY_SECONDS = 1.0  # Delay between batches to avoid rate limits
        decision_id = generate_decision_id(self.bot_run_id, "BATCH_CANCEL", self.loop_count)

        total_succeeded = 0
        total_failed = 0
        all_failures: List[tuple[str, str]] = []

        for i in range(0, len(orders_to_cancel), BATCH_SIZE):
            batch = orders_to_cancel[i:i + BATCH_SIZE]
            order_ids = [o.order_id for o in batch]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(orders_to_cancel) + BATCH_SIZE - 1) // BATCH_SIZE

            # Add delay between batches (not before first batch)
            if batch_num > 1:
                logger.debug("Rate limit pause between cancel batches", extra={
                    "event_type": EventType.RATE_LIMIT_BACKOFF,
                    "delay_seconds": BATCH_DELAY_SECONDS,
                    "batch_number": batch_num,
                })
                await asyncio.sleep(BATCH_DELAY_SECONDS)

            logger.info("Executing cancel batch", extra={
                "event_type": EventType.BATCH_CANCEL,
                "decision_id": decision_id,
                "batch_number": batch_num,
                "total_batches": total_batches,
                "batch_size": len(batch),
                "total_to_cancel": len(orders_to_cancel),
            })

            # Acquire rate limit tokens for batch
            wait_time = rate_limiter.acquire(len(batch), endpoint="batch_cancel")
            if wait_time > 0.1:
                logger.info("Rate limiter enforced wait", extra={
                    "event_type": EventType.RATE_LIMIT_BACKOFF,
                    "wait_seconds": wait_time,
                    "batch_number": batch_num,
                    "endpoint": "batch_cancel",
                })

            result = self.exchange.batch_cancel_orders(order_ids, decision_id)
            total_succeeded += len(result.succeeded)
            total_failed += len(result.failed)

            # Remove successfully cancelled orders from local state
            for order_id in result.succeeded:
                self.state.remove_order(order_id)

            # Log individual failures and handle "not found" specially
            for order_id, error_msg in result.failed:
                order = next((o for o in batch if o.order_id == order_id), None)
                logger.warning("Cancel failed", extra={
                    "event_type": EventType.BATCH_CANCEL_FAILED,
                    "decision_id": decision_id,
                    "order_id": order_id,
                    "market": order.market_id if order else "unknown",
                    "error_msg": error_msg,
                    "batch_number": batch_num,
                })
                all_failures.append((order_id, error_msg))

                # If order "not found", it was already filled/cancelled - remove from local state
                if "not found" in error_msg.lower() or "not_found" in error_msg.lower():
                    self.state.remove_order(order_id)

        # Log summary
        logger.info("Batch cancel complete", extra={
            "event_type": EventType.BATCH_CANCEL_SUMMARY,
            "decision_id": decision_id,
            "total_succeeded": total_succeeded,
            "total_failed": total_failed,
            "total_requested": len(orders_to_cancel),
        })

        return BatchCancelSummary(
            succeeded=total_succeeded,
            failed=total_failed,
            failures=all_failures,
        )

    def _cancel_orders_batch_sync(
        self,
        orders_to_cancel: List[Order],
        decision_id: str,
    ) -> int:
        """Cancel orders in batches of 10 (synchronous version for startup).

        Args:
            orders_to_cancel: List of orders to cancel
            decision_id: Decision ID for logging

        Returns:
            Total number of successfully cancelled orders
        """
        BATCH_SIZE = 10
        BATCH_DELAY_SECONDS = 1.0  # Delay between batches to avoid rate limits
        total_succeeded = 0
        total_failed = 0

        for i in range(0, len(orders_to_cancel), BATCH_SIZE):
            batch = orders_to_cancel[i:i + BATCH_SIZE]
            order_ids = [o.order_id for o in batch]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(orders_to_cancel) + BATCH_SIZE - 1) // BATCH_SIZE

            # Add delay between batches (not before first batch)
            if batch_num > 1:
                logger.debug("Rate limit pause between startup cancel batches", extra={
                    "event_type": EventType.RATE_LIMIT_BACKOFF,
                    "delay_seconds": BATCH_DELAY_SECONDS,
                    "batch_number": batch_num,
                })
                time.sleep(BATCH_DELAY_SECONDS)

            logger.info("Executing startup cancel batch", extra={
                "event_type": EventType.BATCH_CANCEL,
                "decision_id": decision_id,
                "batch_number": batch_num,
                "total_batches": total_batches,
                "batch_size": len(batch),
                "total_to_cancel": len(orders_to_cancel),
            })

            result = self.exchange.batch_cancel_orders(order_ids, decision_id)
            total_succeeded += len(result.succeeded)
            total_failed += len(result.failed)

            # Remove successfully cancelled orders from local state
            for order_id in result.succeeded:
                self.state.remove_order(order_id)

            # Log individual failures
            for order_id, error_msg in result.failed:
                order = next((o for o in batch if o.order_id == order_id), None)
                logger.warning("Startup cancel failed", extra={
                    "event_type": EventType.BATCH_CANCEL_FAILED,
                    "decision_id": decision_id,
                    "order_id": order_id,
                    "market": order.market_id if order else "unknown",
                    "error_msg": error_msg,
                    "batch_number": batch_num,
                })

        # Log summary
        logger.info("Startup batch cancel complete", extra={
            "event_type": EventType.BATCH_CANCEL_SUMMARY,
            "decision_id": decision_id,
            "total_succeeded": total_succeeded,
            "total_failed": total_failed,
            "total_requested": len(orders_to_cancel),
        })

        return total_succeeded

    async def _batch_place(
        self,
        orders_to_place: List[OrderRequest],
        rate_limiter: RateLimiter,
    ) -> BatchPlaceSummary:
        """Place orders in batches of 10.

        Args:
            orders_to_place: List of order requests to place
            rate_limiter: Rate limiter for API calls

        Returns:
            BatchPlaceSummary with results
        """
        BATCH_SIZE = 10
        BATCH_DELAY_SECONDS = 1.0  # Delay between batches to avoid rate limits
        decision_id = generate_decision_id(self.bot_run_id, "BATCH_PLACE", self.loop_count)

        total_succeeded = 0
        total_failed = 0
        all_failures: List[tuple[OrderRequest, str]] = []
        placed_orders: List[Order] = []

        for i in range(0, len(orders_to_place), BATCH_SIZE):
            batch = orders_to_place[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(orders_to_place) + BATCH_SIZE - 1) // BATCH_SIZE

            # Add delay between batches (not before first batch)
            if batch_num > 1:
                logger.debug("Rate limit pause between place batches", extra={
                    "event_type": EventType.RATE_LIMIT_BACKOFF,
                    "delay_seconds": BATCH_DELAY_SECONDS,
                    "batch_number": batch_num,
                })
                await asyncio.sleep(BATCH_DELAY_SECONDS)

            logger.info("Executing place batch", extra={
                "event_type": EventType.BATCH_PLACE,
                "decision_id": decision_id,
                "batch_number": batch_num,
                "total_batches": total_batches,
                "batch_size": len(batch),
                "total_to_place": len(orders_to_place),
                "orders": [
                    {"market": o.market_id, "side": o.side, "price": o.price, "size": o.size}
                    for o in batch
                ],
            })

            # Acquire rate limit tokens for batch
            wait_time = rate_limiter.acquire(len(batch), endpoint="batch_place")
            if wait_time > 0.1:
                logger.info("Rate limiter enforced wait", extra={
                    "event_type": EventType.RATE_LIMIT_BACKOFF,
                    "wait_seconds": wait_time,
                    "batch_number": batch_num,
                    "endpoint": "batch_place",
                })

            result = self.exchange.batch_place_orders(batch, decision_id)
            total_succeeded += len(result.placed)
            total_failed += len(result.failed)
            placed_orders.extend(result.placed)

            # Log successful placements
            for order in result.placed:
                logger.info("Order placed", extra={
                    "event_type": EventType.ORDER_PLACED,
                    "decision_id": decision_id,
                    "order_id": order.order_id,
                    "market": order.market_id,
                    "side": order.side,
                    "price": order.price,
                    "size": order.size,
                    "client_order_id": order.client_order_id,
                    "batch_number": batch_num,
                })

            # Log individual failures
            for request, error_msg in result.failed:
                logger.warning("Place failed", extra={
                    "event_type": EventType.BATCH_PLACE_FAILED,
                    "decision_id": decision_id,
                    "market": request.market_id,
                    "side": request.side,
                    "price": request.price,
                    "size": request.size,
                    "client_order_id": request.client_order_id,
                    "error_msg": error_msg,
                    "batch_number": batch_num,
                })
                all_failures.append((request, error_msg))

        # Log summary
        logger.info("Batch place complete", extra={
            "event_type": EventType.BATCH_PLACE_SUMMARY,
            "decision_id": decision_id,
            "total_succeeded": total_succeeded,
            "total_failed": total_failed,
            "total_requested": len(orders_to_place),
        })

        return BatchPlaceSummary(
            succeeded=total_succeeded,
            failed=total_failed,
            failures=all_failures,
            placed_orders=placed_orders,
        )


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

    # Get AWS region from environment or default
    aws_region = os.getenv('AWS_REGION', 'us-east-1')

    # Initialize structured logging BEFORE creating bot
    bot_run_id = setup_structured_logging(
        service='dora-bot',
        env=env,
        aws_region=aws_region,
    )

    logger.info("Dora Bot starting", extra={
        "event_type": EventType.STARTUP,
        "mode": "DEMO" if use_demo else "PROD",
        "bot_run_id": bot_run_id,
    })

    # Register signal handlers
    shutdown_handler.register_signals()

    # Create and run bot
    bot = DoraBot(use_demo=use_demo, aws_region=aws_region)
    shutdown_handler.set_bot(bot)

    try:
        await bot.startup()
        # Select execution mode based on config
        if bot.global_config.use_batch_execution:
            logger.info("Using batch execution mode", extra={
                "event_type": EventType.STARTUP,
                "execution_mode": "batch",
            })
            await bot.run_loop_batch()
        else:
            logger.info("Using per-market execution mode", extra={
                "event_type": EventType.STARTUP,
                "execution_mode": "per_market",
            })
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
                cleanup_decision_id = (
                    f"{bot.bot_run_id}:fatal_cleanup_cancel:{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                )
                log_decision_record({
                    "market_id": "ALL",
                    "decision_id": cleanup_decision_id,
                    "bot_run_id": bot.bot_run_id,
                    "bot_version": bot.bot_version,
                    "reason": "fatal_error_cleanup_cancel",
                    "target_quotes": [],
                    "num_targets": 0,
                }, region=bot.aws_region, environment=bot.environment)
                bot.exchange.cancel_all_orders(decision_id=cleanup_decision_id)
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
