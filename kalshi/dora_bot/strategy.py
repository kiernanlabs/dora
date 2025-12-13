"""Market making strategy logic."""

import logging
from typing import List
import math

from models import OrderBook, Position, MarketConfig, TargetOrder

logger = logging.getLogger(__name__)


class MarketMaker:
    """Market making strategy implementation."""

    TICK_SIZE = 0.01  # Kalshi tick size in decimal

    def compute_quotes(
        self,
        order_book: OrderBook,
        position: Position,
        config: MarketConfig
    ) -> List[TargetOrder]:
        """Compute target quotes for a market.

        Args:
            order_book: Current order book state
            position: Current position in the market
            config: Market configuration

        Returns:
            List of target orders to place
        """
        if not config.enabled:
            return []

        # Check if order book has valid data
        if order_book.best_bid is None or order_book.best_ask is None:
            logger.warning(f"{config.market_id}: Invalid order book (missing bid/ask)")
            return []

        # Check if spread is wide enough
        if order_book.spread < config.min_spread:
            logger.info(f"{config.market_id}: Spread too narrow ({order_book.spread:.3f} < {config.min_spread:.3f}) - NO QUOTE")
            return []

        # Calculate fair value (mid price)
        fair_value = order_book.mid_price

        # Calculate inventory skew
        # Positive net position = long YES, should encourage selling (widen bid, tighten ask)
        # Negative net position = short YES, should encourage buying (tighten bid, widen ask)
        net_position = position.net_position
        skew = self._calculate_skew(net_position, config)

        # Log order book state
        logger.info(f"{config.market_id} Order Book: bid={order_book.best_bid:.3f} ask={order_book.best_ask:.3f} mid={fair_value:.3f} spread={order_book.spread:.3f}")
        logger.info(f"{config.market_id} Position: YES={position.yes_qty} NO={position.no_qty} net={net_position}")
        logger.info(f"{config.market_id} Skew calculation: net_pos={net_position} → skew={skew:.4f}")

        # Determine target bid/ask prices
        target_bid, target_ask = self._calculate_target_prices(
            fair_value=fair_value,
            min_spread=config.min_spread,
            skew=skew,
            best_bid=order_book.best_bid,
            best_ask=order_book.best_ask
        )

        logger.info(f"{config.market_id} Target prices (before rounding): bid={target_bid:.3f} ask={target_ask:.3f}")

        # Determine sizes based on inventory
        bid_size = self._calculate_size(
            base_size=config.quote_size,
            position_qty=position.yes_qty,
            max_inventory=config.max_inventory_yes,
            side="bid"
        )

        ask_size = self._calculate_size(
            base_size=config.quote_size,
            position_qty=position.yes_qty,
            max_inventory=config.max_inventory_yes,
            side="ask"
        )

        logger.info(f"{config.market_id} Sizes: bid={bid_size} ask={ask_size}")

        # Build target orders
        # NOTE: We store everything as YES prices internally
        # The exchange_client will handle converting to Kalshi API format
        targets = []

        if bid_size > 0 and target_bid > 0.01:  # Min price is 0.01
            rounded_bid = self._round_price(target_bid)
            targets.append(TargetOrder(
                market_id=config.market_id,
                side="bid",  # Buying YES
                price=rounded_bid,
                size=bid_size
            ))
            logger.info(f"{config.market_id} → BID: {bid_size}@{rounded_bid:.2f}")

        if ask_size > 0 and target_ask < 0.99:  # Max price is 0.99
            rounded_ask = self._round_price(target_ask)
            targets.append(TargetOrder(
                market_id=config.market_id,
                side="ask",  # Selling YES
                price=rounded_ask,
                size=ask_size
            ))
            logger.info(f"{config.market_id} → ASK: {ask_size}@{rounded_ask:.2f}")

        return targets

    def _calculate_skew(self, net_position: int, config: MarketConfig) -> float:
        """Calculate inventory skew adjustment.

        Args:
            net_position: Net position (positive = long YES)
            config: Market configuration

        Returns:
            Skew in price units (positive = widen bid, negative = tighten bid)
        """
        if net_position == 0:
            return 0.0

        # Normalize position to [-1, 1]
        max_pos = max(config.max_inventory_yes, config.max_inventory_no)
        normalized_pos = net_position / max_pos if max_pos > 0 else 0

        # Skew proportional to position and skew factor
        # If long (positive), skew > 0 → widen bid, tighten ask to encourage selling
        # If short (negative), skew < 0 → tighten bid, widen ask to encourage buying
        skew = normalized_pos * config.inventory_skew_factor * config.min_spread

        return skew

    def _calculate_target_prices(
        self,
        fair_value: float,
        min_spread: float,
        skew: float,
        best_bid: float,
        best_ask: float
    ) -> tuple[float, float]:
        """Calculate target bid and ask prices.

        Args:
            fair_value: Fair value (mid price)
            min_spread: Minimum required spread
            skew: Inventory skew adjustment
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            Tuple of (target_bid, target_ask)
        """
        # Base quotes: place at half the min spread from fair value
        half_spread = min_spread / 2.0

        # Apply skew
        # Skew > 0: widen bid (lower), tighten ask (higher toward fair)
        # Skew < 0: tighten bid (higher toward fair), widen ask (lower)
        target_bid = fair_value - half_spread - skew
        target_ask = fair_value + half_spread - skew

        # Try to be one tick inside the best quotes (join the best bid/ask)
        # But only if it doesn't violate our minimum spread
        inside_bid = best_bid + self.TICK_SIZE
        inside_ask = best_ask - self.TICK_SIZE

        # Use inside bid if it maintains spread and is better than our target
        if inside_bid < target_ask - min_spread and inside_bid > target_bid:
            target_bid = inside_bid

        # Use inside ask if it maintains spread and is better than our target
        if inside_ask > target_bid + min_spread and inside_ask < target_ask:
            target_ask = inside_ask

        return target_bid, target_ask

    def _calculate_size(
        self,
        base_size: int,
        position_qty: int,
        max_inventory: int,
        side: str
    ) -> int:
        """Calculate order size based on inventory.

        Args:
            base_size: Base quote size from config
            position_qty: Current position quantity
            max_inventory: Maximum allowed inventory
            side: 'bid' or 'ask'

        Returns:
            Adjusted size (0 if at limit)
        """
        if max_inventory == 0:
            return 0

        # Calculate utilization (0 to 1)
        utilization = abs(position_qty) / max_inventory

        # If we're long and trying to buy more, or short and trying to sell more, reduce size
        if side == "bid":
            # Bidding (buying YES) - reduce if long
            if position_qty > 0:
                # Scale size down as we approach limit
                if utilization > 0.8:
                    size_factor = (1.0 - utilization) / 0.2
                    return int(base_size * size_factor)
        else:
            # Asking (selling YES) - reduce if short
            if position_qty < 0:
                if utilization > 0.8:
                    size_factor = (1.0 - utilization) / 0.2
                    return int(base_size * size_factor)

        # Check we don't exceed limits
        remaining = max_inventory - abs(position_qty)
        return int(min(base_size, remaining))

    def _round_price(self, price: float) -> float:
        """Round price to tick size.

        Args:
            price: Raw price

        Returns:
            Rounded price
        """
        return round(price / self.TICK_SIZE) * self.TICK_SIZE

    def should_cancel_order(
        self,
        existing_price: float,
        target_price: float,
        tolerance_ticks: int = 1
    ) -> bool:
        """Determine if an existing order should be cancelled.

        Args:
            existing_price: Price of existing order
            target_price: Desired target price
            tolerance_ticks: Number of ticks tolerance

        Returns:
            True if order should be cancelled
        """
        price_diff = abs(existing_price - target_price)
        return price_diff > (tolerance_ticks * self.TICK_SIZE)
