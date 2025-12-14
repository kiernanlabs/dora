"""State manager for tracking positions, orders, and risk state."""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from models import Position, Order, Fill, RiskState
from dynamo import DynamoDBClient

logger = logging.getLogger(__name__)


class StateManager:
    """Manages in-memory state with DynamoDB persistence."""

    def __init__(self, dynamo_client: DynamoDBClient):
        """Initialize state manager.

        Args:
            dynamo_client: DynamoDB client for persistence
        """
        self.dynamo = dynamo_client

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
            logger.info("Loading state from DynamoDB...")

            # Load positions
            self.positions = self.dynamo.get_positions()
            logger.info(f"Loaded {len(self.positions)} positions")

            # Load risk state
            self.risk_state = self.dynamo.get_risk_state()
            logger.info(f"Loaded risk state: daily_pnl={self.risk_state.daily_pnl:.2f}, halted={self.risk_state.trading_halted}")

            # Load logged fills to prevent duplicate processing
            self.logged_fills = self.dynamo.get_all_fill_ids()
            logger.info(f"Loaded {len(self.logged_fills)} logged fill IDs")

            self.last_sync_time = datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"Error loading state from DynamoDB: {e}")
            return False

    def save_to_dynamo(self) -> bool:
        """Persist current state to DynamoDB.

        Returns:
            True if successful
        """
        try:
            # Log what we're saving
            if self.positions:
                pos_summary = {mid: f"YES:{p.yes_qty} NO:{p.no_qty}" for mid, p in self.positions.items()}
                logger.info(f"Saving state: {len(self.positions)} positions - {pos_summary}")
            else:
                logger.debug("Saving state: no positions yet")

            # Save positions
            if not self.dynamo.save_positions(self.positions):
                logger.error("Failed to save positions")
                return False

            # Save risk state
            if not self.dynamo.save_risk_state(self.risk_state):
                logger.error("Failed to save risk state")
                return False

            self.last_sync_time = datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"Error saving state to DynamoDB: {e}")
            return False

    def reconcile_with_exchange(self, exchange_orders: List[Order]) -> None:
        """Reconcile in-memory state with exchange reality.

        Args:
            exchange_orders: List of open orders from exchange
        """
        logger.info(f"Reconciling state with {len(exchange_orders)} exchange orders")

        # Clear existing open orders
        self.open_orders.clear()

        # Add orders from exchange
        for order in exchange_orders:
            self.open_orders[order.order_id] = order

        logger.info(f"State reconciled: {len(self.open_orders)} open orders")

    def update_from_fills(self, fills: List[Fill]) -> None:
        """Update positions and PnL from new fills.

        Args:
            fills: List of fills to process
        """
        if not fills:
            return

        # Filter out fills we've already processed
        new_fills = [f for f in fills if f.fill_id not in self.logged_fills]

        if not new_fills:
            logger.debug(f"Skipping {len(fills)} already-processed fills")
            return

        logger.info(f"Processing {len(new_fills)} new fills (skipped {len(fills) - len(new_fills)} duplicates)")

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

            # Log trade to DynamoDB (only once per fill_id)
            self.dynamo.log_trade({
                'market_id': fill.market_id,
                'side': fill.side,
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

            logger.info(f"Fill processed: {fill.market_id} {fill.side} {fill.size}@{fill.price:.2f}, PnL delta: {pnl_delta:.2f}")

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
        logger.debug(f"Recorded order: {order.order_id}")

    def remove_order(self, order_id: str) -> None:
        """Remove an order (cancelled or filled).

        Args:
            order_id: Order ID to remove
        """
        if order_id in self.open_orders:
            del self.open_orders[order_id]
            logger.debug(f"Removed order: {order_id}")

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
        logger.info(f"Resetting daily PnL from {self.risk_state.daily_pnl:.2f} to 0")
        self.risk_state.daily_pnl = 0.0
        self.save_to_dynamo()

    def halt_trading(self, reason: str) -> None:
        """Halt trading with a reason.

        Args:
            reason: Reason for halting
        """
        logger.warning(f"HALTING TRADING: {reason}")
        self.risk_state.trading_halted = True
        self.risk_state.halt_reason = reason
        self.risk_state.last_updated = datetime.utcnow()
        self.save_to_dynamo()

    def resume_trading(self) -> None:
        """Resume trading after halt."""
        logger.info("Resuming trading")
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
