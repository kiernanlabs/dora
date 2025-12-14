"""Main runner for the Kalshi market making bot."""

import sys
import os
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization

# Add parent directory to path for kalshi starter code
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kalshi-starter-code-python-main'))
from clients import KalshiHttpClient, Environment

from models import TargetOrder, Order, MarketConfig
from exchange_client import KalshiExchangeClient
from dynamo import DynamoDBClient
from state_manager import StateManager
from risk_manager import RiskManager
from strategy import MarketMaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dora_bot.log')
    ]
)
logger = logging.getLogger(__name__)


class DoraBot:
    """Main trading bot orchestrator."""

    def __init__(self, use_demo: bool = True, aws_region: str = "us-east-1"):
        """Initialize the bot.

        Args:
            use_demo: If True, use demo environment
            aws_region: AWS region for DynamoDB
        """
        logger.info("Initializing Dora Bot...")

        # Get the parent kalshi directory (where .env is located)
        kalshi_dir = os.path.join(os.path.dirname(__file__), '..')

        # Load environment variables
        load_dotenv(os.path.join(kalshi_dir, '.env'))

        # Initialize Kalshi client
        env = Environment.DEMO if use_demo else Environment.PROD
        keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
        keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')

        if not keyid or not keyfile:
            raise ValueError(f"Missing API credentials in .env file")

        # Resolve keyfile path relative to kalshi directory
        keyfile_path = os.path.join(kalshi_dir, keyfile)

        if not os.path.exists(keyfile_path):
            raise FileNotFoundError(f"Private key file not found: {keyfile_path}")

        with open(keyfile_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )

        kalshi_client = KalshiHttpClient(
            key_id=keyid,
            private_key=private_key,
            environment=env
        )

        # Initialize components
        self.exchange = KalshiExchangeClient(kalshi_client)
        self.dynamo = DynamoDBClient(region=aws_region)
        self.state = StateManager(self.dynamo)
        self.strategy = MarketMaker()

        # Load global config and initialize risk manager
        self.global_config = self.dynamo.get_global_config()
        self.risk = RiskManager(self.global_config)

        # Config refresh counter
        self.loop_count = 0
        self.config_refresh_interval = 10  # Reload configs every N loops

        logger.info(f"Bot initialized in {'DEMO' if use_demo else 'PROD'} mode")

    async def startup(self):
        """Perform startup sequence."""
        logger.info("=== STARTUP SEQUENCE ===")

        # 1. Verify exchange connectivity
        try:
            status = self.exchange.get_exchange_status()
            logger.info(f"Exchange status: {status.get('exchange_active', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise

        # 2. Load state from DynamoDB
        if not self.state.load_from_dynamo():
            logger.warning("Failed to load state from DynamoDB, starting fresh")

        # 3. Fetch current open orders from exchange
        exchange_orders = self.exchange.get_open_orders()
        logger.info(f"Found {len(exchange_orders)} open orders on exchange")

        # 4. Reconcile state
        self.state.reconcile_with_exchange(exchange_orders)

        # 5. Cancel all orders on startup if configured
        if self.global_config.cancel_on_startup:
            logger.info("Cancelling all orders on startup (config: cancel_on_startup=True)")
            cancelled = self.exchange.cancel_all_orders()
            logger.info(f"Cancelled {cancelled} orders")
            self.state.open_orders.clear()

        # 6. Fetch any fills we may have missed during downtime
        # Fetch fills from 24 hours before the last fill to now to catch any missed fills
        if self.state.risk_state.last_fill_timestamp:
            from datetime import timedelta, timezone

            # Fetch from 24 hours before the last fill to ensure we catch everything
            fetch_from = self.state.risk_state.last_fill_timestamp - timedelta(hours=24)
            logger.info(f"Fetching fills from {fetch_from} (24hrs before last fill) to now")
            fills = self.exchange.get_fills(since=fetch_from)

            # The logged_fills set will prevent duplicate logging
            self.state.update_from_fills(fills)
            logger.info(f"Processed {len(fills)} fills")

        # 7. Save positions to DynamoDB after reconciliation
        logger.info("Saving reconciled state to DynamoDB...")
        self.state.save_to_dynamo()

        # 8. Log current state
        summary = self.state.get_state_summary()
        logger.info(f"State summary: {summary}")

        # 9. Check balance (optional, doesn't block startup if it fails)
        balance = self.exchange.get_balance()
        if balance:
            logger.info(f"Account balance: ${balance.balance:.2f}, Payout: ${balance.payout:.2f}, Total: ${balance.total:.2f}")
        else:
            logger.warning("Could not fetch account balance - continuing with state from DynamoDB")

        logger.info("=== STARTUP COMPLETE ===")

    async def run_loop(self):
        """Main event loop."""
        logger.info("Starting main event loop...")

        while True:
            loop_start = time.time()

            try:
                # Check if we should halt trading
                should_halt, halt_reason = self.risk.should_halt_trading(self.state)
                if should_halt:
                    logger.critical(f"Trading halted: {halt_reason}")
                    # Cancel all orders and exit
                    self.exchange.cancel_all_orders()
                    self.state.save_to_dynamo()
                    break

                # Refresh configs periodically
                if self.loop_count % self.config_refresh_interval == 0:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)
                    logger.info(f"Refreshed configs: {len(market_configs)} enabled markets")
                else:
                    market_configs = self.dynamo.get_all_market_configs(enabled_only=True)

                # Process each enabled market
                for market_id, config in market_configs.items():
                    await self.process_market(market_id, config)

                # Process fills
                fills = self.exchange.get_fills(since=self.state.risk_state.last_fill_timestamp)
                if fills:
                    new_fills = self.state.update_from_fills(fills)
                    logger.info(f"Processed {len(new_fills)} new fills, Daily PnL: ${self.state.risk_state.daily_pnl:.2f}")

                # Periodic state persistence (every 10 loops)
                if self.loop_count % 10 == 0:
                    self.state.save_to_dynamo()

                self.loop_count += 1

                # Sleep for configured interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.global_config.loop_interval_ms / 1000.0 - loop_duration)

                if loop_duration > self.global_config.loop_interval_ms / 1000.0:
                    logger.warning(f"Loop took {loop_duration*1000:.0f}ms (target: {self.global_config.loop_interval_ms}ms)")

                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Consider canceling all orders on unexpected errors
                time.sleep(5)  # Brief pause before retrying

        # Cleanup
        logger.info("Shutting down...")
        self.state.save_to_dynamo()
        logger.info("Shutdown complete")

    async def process_market(self, market_id: str, config: MarketConfig):
        """Process a single market.

        Args:
            market_id: Market ticker
            config: Market configuration
        """
        try:
            # 1. Fetch order book
            order_book = self.exchange.get_order_book(market_id)

            # 2. Get current position
            position = self.state.get_inventory(market_id)

            # 3. Compute target quotes
            target_orders = self.strategy.compute_quotes(order_book, position, config)

            # Log decision to DynamoDB
            self.dynamo.log_decision({
                'market_id': market_id,
                'order_book_snapshot': {
                    'best_bid': order_book.best_bid,
                    'best_ask': order_book.best_ask,
                    'spread': order_book.spread,
                    'mid': order_book.mid_price
                },
                'inventory': {
                    'yes_qty': position.yes_qty,
                    'no_qty': position.no_qty,
                    'net': position.net_position
                },
                'target_quotes': [
                    {'side': t.side, 'price': t.price, 'size': t.size}
                    for t in target_orders
                ],
                'num_targets': len(target_orders)
            })

            if not target_orders:
                # No quotes desired, cancel any existing orders
                existing_orders = self.state.get_open_orders_for_market(market_id)
                for order in existing_orders:
                    self.exchange.cancel_order(order.order_id)
                    self.state.remove_order(order.order_id)
                return

            # 4. Diff with existing orders
            existing_orders = self.state.get_open_orders_for_market(market_id)
            to_cancel, to_place = self.diff_orders(existing_orders, target_orders)

            # 5. Cancel orders
            for order in to_cancel:
                if self.exchange.cancel_order(order.order_id):
                    self.state.remove_order(order.order_id)

            # 6. Place new orders (with risk checks)
            for target in to_place:
                allowed, reason = self.risk.check_order(target, position, config, self.state)

                if allowed:
                    placed_order = self.exchange.place_order(
                        market_id=target.market_id,
                        side=target.side,
                        price=target.price,
                        size=target.size
                    )

                    if placed_order:
                        self.state.record_order(placed_order)
                else:
                    logger.debug(f"Order blocked by risk: {market_id} {target.side} {target.size}@{target.price:.2f} - {reason}")

        except Exception as e:
            logger.error(f"Error processing market {market_id}: {e}")

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
                    order_type = "BID" if order.side == "yes" else "ASK"
                    # Order.price is always YES price
                    logger.debug(f"Keeping existing order: {order_type} {order.size}@{order.price:.2f}")
                    break

            if not matched:
                # No matching target, cancel this order
                order_type = "BID" if order.side == "yes" else "ASK"
                # Order.price is always YES price

                # Find closest target to show price difference
                if targets:
                    # Convert order.side to target.side for comparison
                    order_target_side = "bid" if order.side == "yes" else "ask"
                    same_side_targets = [t for t in targets if t.side == order_target_side]

                    if same_side_targets:
                        closest_target = min(same_side_targets, key=lambda t: abs(t.price - order.price))
                        price_diff = abs(closest_target.price - order.price)
                        logger.info(f"Cancelling order: {order_type} {order.size}@{order.price:.2f} (target now {closest_target.price:.2f}, diff={price_diff:.3f})")
                    else:
                        logger.info(f"Cancelling order: {order_type} {order.size}@{order.price:.2f} (no target for this side)")
                else:
                    logger.info(f"Cancelling order: {order_type} {order.size}@{order.price:.2f} (no targets)")
                to_cancel.append(order)

        return to_cancel, to_place


async def main():
    """Main entry point."""
    # Parse command line args
    use_demo = True  # Default to demo
    if len(sys.argv) > 1 and sys.argv[1] == '--prod':
        use_demo = False
        logger.warning("Running in PRODUCTION mode!")

    # Create and run bot
    bot = DoraBot(use_demo=use_demo)

    try:
        await bot.startup()
        await bot.run_loop()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
