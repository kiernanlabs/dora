"""Risk management for the trading bot."""

import logging
from typing import Optional

from models import GlobalConfig, MarketConfig, TargetOrder, Position
from state_manager import StateManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk limits and trading rules."""

    def __init__(self, global_config: GlobalConfig):
        """Initialize risk manager.

        Args:
            global_config: Global configuration with risk limits
        """
        self.config = global_config

    def check_order(
        self,
        order: TargetOrder,
        position: Position,
        market_config: MarketConfig,
        state: StateManager
    ) -> tuple[bool, Optional[str]]:
        """Check if an order is allowed under risk limits.

        Args:
            order: Target order to check
            position: Current position in the market
            market_config: Market-specific configuration
            state: State manager with current positions

        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        """
        # Check if trading is enabled globally
        if not self.config.trading_enabled:
            return False, "Trading disabled globally"

        # Check if trading is halted
        if state.risk_state.trading_halted:
            return False, f"Trading halted: {state.risk_state.halt_reason}"

        # Check market-specific inventory limits
        new_position_yes = position.yes_qty
        new_position_no = position.no_qty

        # TargetOrder.side is "bid" or "ask"
        # bid = buying YES, ask = buying NO (selling YES)
        if order.side == "bid":
            new_position_yes += order.size
        else:  # order.side == "ask"
            new_position_no += order.size

        # Check YES inventory limit
        if abs(new_position_yes) > market_config.max_inventory_yes:
            return False, f"YES inventory limit exceeded: {abs(new_position_yes)} > {market_config.max_inventory_yes}"

        # Check NO inventory limit
        if abs(new_position_no) > market_config.max_inventory_no:
            return False, f"NO inventory limit exceeded: {abs(new_position_no)} > {market_config.max_inventory_no}"

        # Check global exposure limit
        # Calculate what total exposure would be if this order fills
        total_exposure = state.get_total_exposure()
        additional_exposure = order.size  # Simplified: assumes order adds to exposure

        if total_exposure + additional_exposure > self.config.max_total_exposure:
            return False, f"Global exposure limit exceeded: {total_exposure + additional_exposure} > {self.config.max_total_exposure}"

        # All checks passed
        return True, None

    def should_halt_trading(self, state: StateManager) -> tuple[bool, Optional[str]]:
        """Check if trading should be halted based on current state.

        Args:
            state: State manager with current positions and PnL

        Returns:
            Tuple of (should_halt, reason_if_yes)
        """
        # Check daily loss limit
        if state.risk_state.daily_pnl < -self.config.max_daily_loss:
            return True, f"Daily loss limit breached: ${state.risk_state.daily_pnl:.2f} < -${self.config.max_daily_loss:.2f}"

        # Check if already halted
        if state.risk_state.trading_halted:
            return True, state.risk_state.halt_reason

        return False, None

    def check_market_limits(
        self,
        market_id: str,
        position: Position,
        market_config: MarketConfig
    ) -> tuple[bool, Optional[str]]:
        """Check if we should continue quoting in a market.

        Args:
            market_id: Market ticker
            position: Current position
            market_config: Market configuration

        Returns:
            Tuple of (should_quote, reason_if_not)
        """
        # Check if market is enabled
        if not market_config.enabled:
            return False, "Market disabled in config"

        # Check if position is near limits (warning, not hard stop)
        yes_util = abs(position.yes_qty) / market_config.max_inventory_yes if market_config.max_inventory_yes > 0 else 0
        no_util = abs(position.no_qty) / market_config.max_inventory_no if market_config.max_inventory_no > 0 else 0

        if yes_util > 0.9:
            logger.warning(f"{market_id}: YES inventory at {yes_util*100:.0f}% of limit")

        if no_util > 0.9:
            logger.warning(f"{market_id}: NO inventory at {no_util*100:.0f}% of limit")

        return True, None

    def get_exposure(self, state: StateManager) -> float:
        """Calculate total notional exposure.

        Args:
            state: State manager

        Returns:
            Total notional exposure
        """
        return float(state.get_total_exposure())

    def get_max_order_size(
        self,
        position: Position,
        market_config: MarketConfig,
        side: str
    ) -> int:
        """Calculate maximum allowed order size given current position.

        Args:
            position: Current position
            market_config: Market configuration
            side: 'yes' or 'no'

        Returns:
            Maximum allowed size
        """
        if side == "yes":
            remaining = market_config.max_inventory_yes - abs(position.yes_qty)
        else:
            remaining = market_config.max_inventory_no - abs(position.no_qty)

        return max(0, remaining)

    def adjust_quote_size_for_risk(
        self,
        base_size: int,
        position: Position,
        market_config: MarketConfig,
        side: str
    ) -> int:
        """Adjust quote size based on risk and inventory.

        Args:
            base_size: Desired base size
            position: Current position
            market_config: Market configuration
            side: 'yes' or 'no'

        Returns:
            Adjusted size (0 if should not quote)
        """
        max_size = self.get_max_order_size(position, market_config, side)

        if max_size == 0:
            return 0

        # Calculate inventory utilization
        if side == "yes":
            utilization = abs(position.yes_qty) / market_config.max_inventory_yes
        else:
            utilization = abs(position.no_qty) / market_config.max_inventory_no

        # Reduce size as we approach limits
        if utilization > 0.8:
            # Reduce size linearly from 100% at 80% utilization to 0% at 100%
            size_factor = (1.0 - utilization) / 0.2
            adjusted_size = int(base_size * size_factor)
            return min(adjusted_size, max_size)

        return min(base_size, max_size)

    def emergency_stop(self, state: StateManager, reason: str) -> None:
        """Trigger emergency stop (halt trading).

        Args:
            state: State manager
            reason: Reason for emergency stop
        """
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        state.halt_trading(reason)
