"""Risk management for the trading bot."""

from typing import Optional

from dora_bot.models import GlobalConfig, MarketConfig, TargetOrder, Position
from dora_bot.state_manager import StateManager
from dora_bot.structured_logger import get_logger, EventType

logger = get_logger(__name__)


class RiskManager:
    """Enforces risk limits and trading rules."""

    def __init__(self, global_config: GlobalConfig, bot_run_id: Optional[str] = None):
        """Initialize risk manager.

        Args:
            global_config: Global configuration with risk limits
            bot_run_id: Bot run ID for log correlation
        """
        self.config = global_config
        self.bot_run_id = bot_run_id

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
        # TargetOrder.side is "bid" or "ask"
        # bid = buying YES (increases net_yes_qty), ask = selling YES (decreases net_yes_qty)
        if order.side == "bid":
            new_net_position = position.net_yes_qty + order.size
        else:  # order.side == "ask"
            new_net_position = position.net_yes_qty - order.size

        # Check inventory limit (applies to both long and short)
        if abs(new_net_position) > market_config.max_inventory_yes:
            return False, f"Inventory limit exceeded: {abs(new_net_position)} > {market_config.max_inventory_yes}"

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
        utilization = abs(position.net_yes_qty) / market_config.max_inventory_yes if market_config.max_inventory_yes > 0 else 0

        if utilization > 0.9:
            logger.warning("Inventory near limit", extra={
                "event_type": EventType.LOG,
                "market": market_id,
                "utilization_pct": utilization * 100,
                "net_yes_qty": position.net_yes_qty,
                "max_inventory": market_config.max_inventory_yes,
            })

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
            side: 'bid' or 'ask' (bid=buy YES, ask=sell YES)

        Returns:
            Maximum allowed size
        """
        if side == "bid":
            # Buying YES - check how much room before hitting long limit
            if position.net_yes_qty >= 0:
                remaining = market_config.max_inventory_yes - position.net_yes_qty
            else:
                # Short position, buying brings us toward 0 then long
                remaining = market_config.max_inventory_yes + abs(position.net_yes_qty)
        else:  # ask
            # Selling YES - check how much room before hitting short limit
            if position.net_yes_qty <= 0:
                remaining = market_config.max_inventory_yes - abs(position.net_yes_qty)
            else:
                # Long position, selling brings us toward 0 then short
                remaining = market_config.max_inventory_yes + position.net_yes_qty

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
            side: 'bid' or 'ask' (bid=buy YES, ask=sell YES)

        Returns:
            Adjusted size (0 if should not quote)
        """
        max_size = self.get_max_order_size(position, market_config, side)

        if max_size == 0:
            return 0

        # Calculate inventory utilization based on net position
        utilization = abs(position.net_yes_qty) / market_config.max_inventory_yes if market_config.max_inventory_yes > 0 else 0

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
        logger.critical("Emergency stop triggered", extra={
            "event_type": EventType.RISK_HALT,
            "reason": reason,
            "daily_pnl": state.risk_state.daily_pnl,
        })
        state.halt_trading(reason)
