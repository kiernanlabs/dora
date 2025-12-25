"""State manager for tracking positions, orders, and risk state."""

from typing import Dict, List, Optional
from datetime import datetime, timezone
from collections import defaultdict

from dora_bot.models import Position, Order, Fill, RiskState
from dora_bot.dynamo import DynamoDBClient
from dora_bot.structured_logger import get_logger, EventType, log_execution_event

logger = get_logger(__name__)


class StateManager:
    """Manages in-memory state with DynamoDB persistence."""

    def __init__(self, dynamo_client: DynamoDBClient, bot_run_id: Optional[str] = None):
        """Initialize state manager.

        Args:
            dynamo_client: DynamoDB client for persistence
            bot_run_id: Bot run ID for log correlation
        """
        self.dynamo = dynamo_client
        self.bot_run_id = bot_run_id

        # In-memory state
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}  # order_id -> Order
        self.risk_state: RiskState = RiskState()

        # Track logged fills to avoid duplicates
        self.logged_fills: set = set()

        # API error tracking by market_id
        # Structure: {market_id: {count, last_error, last_error_code, last_status_code}}
        self.api_errors: Dict[str, Dict] = {}

        # Performance tracking
        self.last_sync_time: Optional[datetime] = None

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def load_from_dynamo(self) -> bool:
        """Load state from DynamoDB on startup.

        Returns:
            True if successful
        """
        try:
            logger.info("Loading state from DynamoDB", extra={
                "event_type": EventType.STATE_LOAD,
            })

            # Load positions
            self.positions = self.dynamo.get_positions()

            # Load risk state
            self.risk_state = self.dynamo.get_risk_state()

            # Load logged fills to prevent duplicate processing
            self.logged_fills = self.dynamo.get_all_fill_ids()

            # Load API error counts
            self.api_errors = self.dynamo.get_api_errors()

            logger.info("State loaded from DynamoDB", extra={
                "event_type": EventType.STATE_LOAD,
                "positions_count": len(self.positions),
                "daily_pnl": self.risk_state.daily_pnl,
                "trading_halted": self.risk_state.trading_halted,
                "logged_fills_count": len(self.logged_fills),
                "api_errors_markets": len(self.api_errors),
            })

            self.last_sync_time = datetime.utcnow()
            return True

        except Exception as e:
            logger.error("Error loading state from DynamoDB", extra={
                "event_type": EventType.ERROR,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
            return False

    def save_to_dynamo(self) -> bool:
        """Persist current state to DynamoDB.

        Returns:
            True if successful
        """
        try:
            # Clean up zero positions before saving
            self.cleanup_zero_positions()

            # Save positions
            if not self.dynamo.save_positions(self.positions):
                logger.error("Failed to save positions", extra={
                    "event_type": EventType.ERROR,
                })
                return False

            # Save risk state
            if not self.dynamo.save_risk_state(self.risk_state):
                logger.error("Failed to save risk state", extra={
                    "event_type": EventType.ERROR,
                })
                return False

            # Save open orders
            if not self.dynamo.save_open_orders(self.open_orders):
                logger.error("Failed to save open orders", extra={
                    "event_type": EventType.ERROR,
                })
                return False

            # Save API errors
            if not self.dynamo.save_api_errors(self.api_errors):
                logger.error("Failed to save API errors", extra={
                    "event_type": EventType.ERROR,
                })
                return False

            logger.debug("State saved to DynamoDB", extra={
                "event_type": EventType.STATE_SAVE,
                "positions_count": len(self.positions),
                "open_orders_count": len(self.open_orders),
                "daily_pnl": self.risk_state.daily_pnl,
                "api_errors_markets": len(self.api_errors),
            })

            self.last_sync_time = datetime.utcnow()
            return True

        except Exception as e:
            logger.error("Error saving state to DynamoDB", extra={
                "event_type": EventType.ERROR,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
            return False

    def reconcile_with_exchange(
        self,
        exchange_orders: List[Order],
        log_drift: bool = False,
        recent_order_grace_period_seconds: float = 60.0,
    ) -> dict:
        """Reconcile in-memory state with exchange reality.

        Orders that were placed recently (within grace period) are preserved even if
        they don't appear in the exchange response yet, to handle eventual consistency.

        Args:
            exchange_orders: List of open orders from exchange
            log_drift: If True, detect and log differences between local and exchange state
            recent_order_grace_period_seconds: Preserve local orders created within this many seconds

        Returns:
            Dictionary with drift statistics (only populated if log_drift=True)
        """
        now = datetime.now(timezone.utc)
        drift_stats = {
            "orders_only_local": [],
            "orders_only_exchange": [],
            "orders_matched": 0,
            "orders_preserved_recent": 0,
        }
        orders = {}

        # Build sets of order IDs for comparison
        local_order_ids = set(self.open_orders.keys())
        exchange_order_ids = {order.order_id for order in exchange_orders}

        # Identify orders only in local state
        only_local = local_order_ids - exchange_order_ids

        # Separate recent orders (preserve them) from stale orders (log them)
        recent_local_orders = {}
        stale_local_order_ids = []

        for order_id in only_local:
            order = self.open_orders.get(order_id)
            if order and order.created_at:
                age_seconds = (now - self._ensure_utc(order.created_at)).total_seconds()
                if age_seconds < recent_order_grace_period_seconds:
                    # This order was placed recently - keep it despite not being on exchange yet
                    recent_local_orders[order_id] = order
                    continue
            # This order is stale - it should have appeared on exchange by now
            stale_local_order_ids.append(order_id)

        if log_drift:
            orders['local'] = local_order_ids
            orders['exchange'] = exchange_order_ids

            # Log stale orders (not recent, should have been on exchange)
            if stale_local_order_ids:
                drift_stats["orders_only_local"] = stale_local_order_ids
                for order_id in stale_local_order_ids:
                    order = self.open_orders.get(order_id)
                    logger.warning("Order in local state but not on exchange (stale)", extra={
                        "event_type": EventType.LOG,
                        "order_id": order_id,
                        "market": order.market_id if order else "unknown",
                        "side": order.side if order else "unknown",
                        "price": order.price if order else 0,
                        "size": order.size if order else 0,
                    })

            # Log recent orders being preserved
            if recent_local_orders:
                drift_stats["orders_preserved_recent"] = len(recent_local_orders)
                for order_id, order in recent_local_orders.items():
                    age_seconds = (now - self._ensure_utc(order.created_at)).total_seconds() if order.created_at else 0
                    logger.info("Preserving recently placed order not yet on exchange", extra={
                        "event_type": EventType.LOG,
                        "order_id": order_id,
                        "market": order.market_id,
                        "side": order.side,
                        "price": order.price,
                        "size": order.size,
                        "age_seconds": round(age_seconds, 1),
                    })

            # Orders on exchange but not in local state (missed tracking)
            only_exchange = exchange_order_ids - local_order_ids
            if only_exchange:
                drift_stats["orders_only_exchange"] = list(only_exchange)
                exchange_orders_map = {o.order_id: o for o in exchange_orders}
                for order_id in only_exchange:
                    order = exchange_orders_map.get(order_id)
                    logger.warning("Order on exchange but not in local state (untracked)", extra={
                        "event_type": EventType.LOG,
                        "order_id": order_id,
                        "market": order.market_id if order else "unknown",
                        "side": order.side if order else "unknown",
                        "price": order.price if order else 0,
                        "size": order.size if order else 0,
                    })

            drift_stats["orders_matched"] = len(local_order_ids & exchange_order_ids)

        # Clear existing open orders
        self.open_orders.clear()

        # Add orders from exchange
        for order in exchange_orders:
            self.open_orders[order.order_id] = order

        # Preserve recently-placed local orders that haven't appeared on exchange yet
        for order_id, order in recent_local_orders.items():
            if order_id not in self.open_orders:
                self.open_orders[order_id] = order

        log_extra = {
            "event_type": EventType.STATE_LOAD,
            "open_orders_count": len(self.open_orders),
        }
        if log_drift:
            log_extra.update({
                "drift_only_local": len(drift_stats["orders_only_local"]),
                "drift_only_exchange": len(drift_stats["orders_only_exchange"]),
                "drift_matched": drift_stats["orders_matched"],
                "drift_preserved_recent": drift_stats["orders_preserved_recent"],
            })

        logger.info(f"State reconciled with exchange: order dump: {orders}", extra=log_extra)

        return drift_stats

    def reconcile_positions_from_fills(self, fills: List[Fill]) -> int:
        """Reconcile positions from fills, processing even already-logged fills.

        This method is used during startup to ensure positions are correct even if
        fills were logged but positions weren't saved (e.g., due to a crash).

        Note: This rebuilds positions from scratch but does NOT recalculate daily_pnl.
        The daily_pnl is preserved from DynamoDB as it should only track today's PnL.

        Fills that were not previously logged to DynamoDB (missed due to crash) will
        be logged during this reconciliation.

        Args:
            fills: List of fills to process

        Returns:
            Number of fills processed
        """
        if not fills:
            return 0

        # Sort fills by timestamp (oldest first)
        sorted_fills = sorted(fills, key=lambda f: f.timestamp)

        # Reset positions to ensure clean reconciliation
        self.positions.clear()

        processed_count = 0
        newly_logged_count = 0
        for fill in sorted_fills:
            # Update position
            if fill.market_id not in self.positions:
                self.positions[fill.market_id] = Position(market_id=fill.market_id)

            position = self.positions[fill.market_id]
            position.update_from_fill(fill)

            # Update last fill timestamp
            if self.risk_state.last_fill_timestamp is None or fill.timestamp > self.risk_state.last_fill_timestamp:
                self.risk_state.last_fill_timestamp = fill.timestamp

            # Log fills that weren't previously logged (missed due to crash)
            if fill.fill_id not in self.logged_fills:
                order_type = 'bid' if fill.side == 'yes' else 'ask'
                self.dynamo.log_trade({
                    'market_id': fill.market_id,
                    'side': fill.side,
                    'type': order_type,
                    'price': fill.price,
                    'size': fill.size,
                    'fill_price': fill.price,
                    'fill_size': fill.size,
                    'status': 'filled',
                    'pnl_realized': 0,  # Can't calculate PnL delta during reconciliation
                    'fees': fill.fees,
                    'order_id': fill.order_id,
                    'fill_id': fill.fill_id,
                    'fill_timestamp': fill.timestamp.isoformat(),
                    'reconciled_at_startup': True,  # Flag to indicate this was backfilled
                })
                logger.info("Backfilled missed fill during startup", extra={
                    "event_type": EventType.FILL,
                    "market": fill.market_id,
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "side": fill.side,
                    "price": fill.price,
                    "size": fill.size,
                    "fees": fill.fees,
                    "fill_timestamp": fill.timestamp.isoformat(),
                })
                newly_logged_count += 1

            # Ensure fill is in logged_fills
            self.logged_fills.add(fill.fill_id)
            processed_count += 1

        logger.info("Positions reconciled from fills", extra={
            "event_type": EventType.STATE_LOAD,
            "fills_processed": processed_count,
            "fills_newly_logged": newly_logged_count,
            "positions_count": len(self.positions),
            "daily_pnl": self.risk_state.daily_pnl,
        })

        return processed_count

    def update_from_fills(self, fills: List[Fill]) -> int:
        """Update positions and PnL from new fills.

        Args:
            fills: List of fills to process

        Returns:
            Number of new fills processed
        """
        if not fills:
            return 0

        # Filter out fills we've already processed
        new_fills = [f for f in fills if f.fill_id not in self.logged_fills]

        if not new_fills:
            return 0

        # Sort fills by timestamp (oldest first)
        new_fills = sorted(new_fills, key=lambda f: f.timestamp)

        for fill in new_fills:
            # Update position
            if fill.market_id not in self.positions:
                self.positions[fill.market_id] = Position(market_id=fill.market_id)

            position = self.positions[fill.market_id]
            old_realized_pnl = position.realized_pnl
            position.update_from_fill(fill)

            # Update daily PnL
            pnl_delta = position.realized_pnl - old_realized_pnl
            self.risk_state.daily_pnl += pnl_delta - fill.fees

            # Update last fill timestamp
            if self.risk_state.last_fill_timestamp is None or fill.timestamp > self.risk_state.last_fill_timestamp:
                self.risk_state.last_fill_timestamp = fill.timestamp

            order_type = 'bid' if fill.side == 'yes' else 'ask'
            order_meta = self.open_orders.get(fill.order_id)
            decision_id = order_meta.decision_id if order_meta else None
            client_order_id = order_meta.client_order_id if order_meta else None

            # Log fill event to structured logs
            logger.info("Fill processed", extra={
                "event_type": EventType.FILL,
                "market": fill.market_id,
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "decision_id": decision_id,
                "side": fill.side,
                "price": fill.price,
                "size": fill.size,
                "pnl_delta": pnl_delta,
                "fees": fill.fees,
                "daily_pnl": self.risk_state.daily_pnl,
                "net_position": position.net_yes_qty,
                "client_order_id": client_order_id,
            })
            if decision_id:
                log_execution_event({
                    "event_type": EventType.FILL,
                    "market": fill.market_id,
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "decision_id": decision_id,
                    "bot_run_id": self.bot_run_id,
                    "side": fill.side,
                    "price": fill.price,
                    "size": fill.size,
                    "pnl_delta": pnl_delta,
                    "fees": fill.fees,
                    "client_order_id": client_order_id,
                }, region=self.dynamo.region, environment=self.dynamo.environment)
            else:
                logger.warning("Skipping execution log for fill; missing decision_id", extra={
                    "event_type": EventType.LOG,
                    "market": fill.market_id,
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                })

            # Log trade to DynamoDB (only once per fill_id)
            self.dynamo.log_trade({
                'market_id': fill.market_id,
                'side': fill.side,
                'type': order_type,
                'price': fill.price,
                'size': fill.size,
                'fill_price': fill.price,
                'fill_size': fill.size,
                'status': 'filled',
                'pnl_realized': pnl_delta,
                'fees': fill.fees,
                'order_id': fill.order_id,
                'fill_id': fill.fill_id,
                'fill_timestamp': fill.timestamp.isoformat()
            })

            # Mark this fill as logged
            self.logged_fills.add(fill.fill_id)

        return len(new_fills)

    def get_inventory(self, market_id: str) -> Position:
        """Get position for a market.

        Args:
            market_id: Market ticker

        Returns:
            Position object (creates new if doesn't exist)
        """
        if market_id not in self.positions:
            self.positions[market_id] = Position(market_id=market_id)
        return self.positions[market_id]

    def record_order(self, order: Order) -> None:
        """Record a newly placed order.

        Args:
            order: Order to record
        """
        self.open_orders[order.order_id] = order

    def remove_order(self, order_id: str) -> None:
        """Remove an order (cancelled or filled).

        Args:
            order_id: Order ID to remove
        """
        if order_id in self.open_orders:
            del self.open_orders[order_id]

    def get_open_orders_for_market(self, market_id: str) -> List[Order]:
        """Get all open orders for a specific market.

        Args:
            market_id: Market ticker

        Returns:
            List of open orders
        """
        return [order for order in self.open_orders.values() if order.market_id == market_id]

    def get_total_exposure(self) -> int:
        """Calculate total exposure across all positions.

        Returns:
            Total exposure (sum of absolute positions)
        """
        return sum(pos.total_exposure for pos in self.positions.values())

    def record_api_error(self, market_id: str, status_code: int = None,
                         error_code: str = None, error_msg: str = None) -> None:
        """Record an API error for a market.

        Args:
            market_id: Market ticker (use 'global' for non-market-specific errors)
            status_code: HTTP status code (e.g., 404, 500)
            error_code: API error code (e.g., 'not_found')
            error_msg: Error message
        """
        if market_id not in self.api_errors:
            self.api_errors[market_id] = {
                'count': 0,
                'last_error': None,
                'last_error_code': None,
                'last_status_code': None,
                'last_error_msg': None,
            }

        self.api_errors[market_id]['count'] += 1
        self.api_errors[market_id]['last_error'] = datetime.utcnow().isoformat()
        if status_code is not None:
            self.api_errors[market_id]['last_status_code'] = status_code
        if error_code is not None:
            self.api_errors[market_id]['last_error_code'] = error_code
        if error_msg is not None:
            self.api_errors[market_id]['last_error_msg'] = error_msg

    def get_api_error_count(self, market_id: str) -> int:
        """Get the API error count for a market.

        Args:
            market_id: Market ticker

        Returns:
            Number of API errors recorded for this market
        """
        return self.api_errors.get(market_id, {}).get('count', 0)

    def clear_api_errors(self, market_id: str = None) -> None:
        """Clear API error counts.

        Args:
            market_id: If provided, clear only for this market. Otherwise clear all.
        """
        if market_id:
            if market_id in self.api_errors:
                del self.api_errors[market_id]
        else:
            self.api_errors = {}

    def cleanup_zero_positions(self) -> int:
        """Remove positions with zero quantity from state.

        This prevents old inactive markets from accumulating in the positions dict.

        Returns:
            Number of positions removed
        """
        zero_positions = [
            market_id for market_id, pos in self.positions.items()
            if pos.net_yes_qty == 0
        ]

        for market_id in zero_positions:
            del self.positions[market_id]

        if zero_positions:
            logger.info("Cleaned up zero positions", extra={
                "event_type": EventType.LOG,
                "removed_markets": zero_positions,
                "removed_count": len(zero_positions),
            })

        return len(zero_positions)

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL (call at start of new trading day)."""
        old_pnl = self.risk_state.daily_pnl
        self.risk_state.daily_pnl = 0.0
        logger.info("Daily PnL reset", extra={
            "event_type": EventType.LOG,
            "old_daily_pnl": old_pnl,
        })
        self.save_to_dynamo()

    def halt_trading(self, reason: str) -> None:
        """Halt trading with a reason.

        Args:
            reason: Reason for halting
        """
        logger.warning("Trading halted", extra={
            "event_type": EventType.RISK_HALT,
            "reason": reason,
            "daily_pnl": self.risk_state.daily_pnl,
        })
        self.risk_state.trading_halted = True
        self.risk_state.halt_reason = reason
        self.risk_state.last_updated = datetime.utcnow()
        self.save_to_dynamo()

    def resume_trading(self) -> None:
        """Resume trading after halt."""
        logger.info("Trading resumed", extra={
            "event_type": EventType.LOG,
        })
        self.risk_state.trading_halted = False
        self.risk_state.halt_reason = None
        self.risk_state.last_updated = datetime.utcnow()
        self.save_to_dynamo()

    def get_state_summary(self) -> Dict:
        """Get a summary of current state for logging.

        Returns:
            Dictionary with state summary
        """
        return {
            'num_positions': len(self.positions),
            'num_open_orders': len(self.open_orders),
            'total_exposure': self.get_total_exposure(),
            'daily_pnl': self.risk_state.daily_pnl,
            'trading_halted': self.risk_state.trading_halted,
            'last_fill': self.risk_state.last_fill_timestamp.isoformat() if self.risk_state.last_fill_timestamp else None
        }
