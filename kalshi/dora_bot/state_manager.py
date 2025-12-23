"""State manager for tracking positions, orders, and risk state."""

from typing import Dict, List, Optional
from datetime import datetime
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

        # Performance tracking
        self.last_sync_time: Optional[datetime] = None

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

            logger.info("State loaded from DynamoDB", extra={
                "event_type": EventType.STATE_LOAD,
                "positions_count": len(self.positions),
                "daily_pnl": self.risk_state.daily_pnl,
                "trading_halted": self.risk_state.trading_halted,
                "logged_fills_count": len(self.logged_fills),
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

            logger.debug("State saved to DynamoDB", extra={
                "event_type": EventType.STATE_SAVE,
                "positions_count": len(self.positions),
                "daily_pnl": self.risk_state.daily_pnl,
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

    def reconcile_with_exchange(self, exchange_orders: List[Order]) -> None:
        """Reconcile in-memory state with exchange reality.

        Args:
            exchange_orders: List of open orders from exchange
        """
        # Clear existing open orders
        self.open_orders.clear()

        # Add orders from exchange
        for order in exchange_orders:
            self.open_orders[order.order_id] = order

        logger.info("State reconciled with exchange", extra={
            "event_type": EventType.STATE_LOAD,
            "open_orders_count": len(self.open_orders),
        })

    def reconcile_positions_from_fills(self, fills: List[Fill]) -> int:
        """Reconcile positions from fills, processing even already-logged fills.

        This method is used during startup to ensure positions are correct even if
        fills were logged but positions weren't saved (e.g., due to a crash).

        Note: This rebuilds positions from scratch but does NOT recalculate daily_pnl.
        The daily_pnl is preserved from DynamoDB as it should only track today's PnL.

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
        for fill in sorted_fills:
            # Update position
            if fill.market_id not in self.positions:
                self.positions[fill.market_id] = Position(market_id=fill.market_id)

            position = self.positions[fill.market_id]
            position.update_from_fill(fill)

            # Update last fill timestamp
            if self.risk_state.last_fill_timestamp is None or fill.timestamp > self.risk_state.last_fill_timestamp:
                self.risk_state.last_fill_timestamp = fill.timestamp

            # Ensure fill is in logged_fills
            self.logged_fills.add(fill.fill_id)
            processed_count += 1

        logger.info("Positions reconciled from fills", extra={
            "event_type": EventType.STATE_LOAD,
            "fills_processed": processed_count,
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
