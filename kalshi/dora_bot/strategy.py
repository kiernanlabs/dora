"""Market making strategy logic."""

import logging
from typing import Any, Dict, List, Optional
import math
from datetime import datetime

from dora_bot.models import OrderBook, Position, MarketConfig, TargetOrder

logger = logging.getLogger(__name__)


class MarketMaker:
    """Market making strategy implementation."""

    TICK_SIZE = 0.01  # Kalshi tick size in decimal

    def compute_quotes(
        self,
        order_book: OrderBook,
        position: Position,
        config: MarketConfig,
        trades: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[TargetOrder], Optional[Dict[str, Any]]]:
        """Compute target quotes for a market.

        Args:
            order_book: Current order book state
            position: Current position in the market
            config: Market configuration
            trades: Recent trades from the API

        Returns:
            List of target orders to place and pricing metadata (if calculated).
        """
        if not config.enabled:
            return [], None

        # Check if order book has valid data
        if order_book.best_bid is None or order_book.best_ask is None:
            logger.warning("Invalid order book", extra={
                "market": config.market_id,
                "reason": "missing_bid_ask"
            })
            return [], None

        min_spread = False
        # Check if spread is wide enough
        if order_book.spread < config.min_spread:
            logger.info("Spread too narrow - proceeding, but should single-side quote if skewed", extra={
                "market": config.market_id,
                "spread": order_book.spread,
                "min_spread": config.min_spread
            })
            min_spread = True
            # return [], None

        trades = trades or []
        last_trade_price, last_trade_time = self._get_last_trade_info(trades)
        if last_trade_price is not None:
            fair_value = last_trade_price
            fv_source = "last_trade"
        else:
            fair_value = order_book.mid_price
            fv_source = "mid"
        using_config_fv = False

        # Calculate inventory skew
        net_position = position.net_position
        skew = self._calculate_skew(net_position, config)

        # Determine target bid/ask prices
        target_bid, target_ask, price_calc = self._calculate_target_prices(
            fair_value=fair_value,
            min_spread=config.min_spread,
            skew=skew,
            best_bid=order_book.best_bid,
            best_ask=order_book.best_ask
        )

        # Determine sizes based on inventory
        bid_size = 0
        ask_size = 0

        # if we are below min spread threshold, exit market by selling position
        if min_spread:
            if net_position > 0:
                bid_size = 0
                ask_size = net_position
            else:
                bid_size = -net_position
                ask_size = 0
        else:
            bid_size = self._calculate_size(
                base_size=config.quote_size,
                position_qty=position.net_yes_qty,
                max_inventory=config.max_inventory_yes,
                side="bid"
            )

            ask_size = self._calculate_size(
                base_size=config.quote_size,
                position_qty=position.net_yes_qty,
                max_inventory=config.max_inventory_yes,
                side="ask"
            )                
        

        # Log consolidated quote calculation with flattened fields for CloudWatch
        logic_msg = (
            "Quote calculation: fv_source={fv_source} fv={fv} last_trade={last_trade} "
            "trades_count={trades_count} min_spread={min_spread} "
            "skew={skew} net_yes={net_yes} max_yes={max_yes} max_no={max_no} "
            "target_bid={target_bid} target_ask={target_ask} "
            "bid_size={bid_size} ask_size={ask_size} "
            "price_calc={price_calc}"
        ).format(
            fv_source=fv_source,
            fv=fair_value,
            last_trade=last_trade_price,
            trades_count=len(trades),
            min_spread=config.min_spread,
            skew=skew,
            net_yes=position.net_yes_qty,
            max_yes=config.max_inventory_yes,
            max_no=config.max_inventory_no,
            target_bid=target_bid,
            target_ask=target_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            price_calc=price_calc
        )
        logger.info(logic_msg, extra={
            "market": config.market_id,
            "best_bid": order_book.best_bid,
            "best_ask": order_book.best_ask,
            "mid": order_book.mid_price,
            "spread": order_book.spread,
            "net_yes_qty": position.net_yes_qty,
            "net_position": net_position,
            "fair_value": fair_value,
            "using_config_fv": using_config_fv,
            "fv_source": fv_source,
            "last_trade_price": last_trade_price,
            "last_trade_time": last_trade_time,
            "trades_count": len(trades),
            "min_spread": config.min_spread,
            "skew": skew,
            "max_inventory_yes": config.max_inventory_yes,
            "max_inventory_no": config.max_inventory_no,
            "target_bid": target_bid,
            "target_ask": target_ask,
            "bid_size": bid_size,
            "ask_size": ask_size
        })

        # Build target orders
        # NOTE: We store everything as YES prices internally
        # The exchange_client will handle converting to Kalshi API format
        targets = []

        if bid_size > 0 and target_bid is not None and target_bid >= 0.01:  # Min price is 0.01
            rounded_bid = self._round_price(target_bid)
            targets.append(TargetOrder(
                market_id=config.market_id,
                side="bid",  # Buying YES
                price=rounded_bid,
                size=bid_size
            ))

        if ask_size > 0 and target_ask is not None and target_ask <= 0.99:  # Max price is 0.99
            rounded_ask = self._round_price(target_ask)
            targets.append(TargetOrder(
                market_id=config.market_id,
                side="ask",  # Selling YES
                price=rounded_ask,
                size=ask_size
            ))

        # Log final quotes with detail (removed - redundant with "Decision made" log in main.py)
        if not targets:
            logger.info("No quotes generated - size or price constraints", extra={
                "market": config.market_id,
                "reason": "size_or_price_constraints"
            })

        return targets, price_calc

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

        normalized_pos = max(-1.0, min(1.0, normalized_pos))

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
    ) -> tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Calculate target bid and ask prices.

        Strategy: Place just inside the current spread, then apply skew.
        Fall back to min_spread from fair_value only if inside prices
        would violate minimum spread requirements.

        Args:
            fair_value: Fair value (last trade or mid price fallback)
            min_spread: Minimum required spread
            skew: Inventory skew adjustment
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            Tuple of (target_bid, target_ask, price_calc); either side may be None to skip quoting.
        """
        # Start by placing one tick inside the current spread if spread is wide, if not, place at market
        inside_bid = best_bid + self.TICK_SIZE
        inside_ask = best_ask - self.TICK_SIZE

        if (best_ask - best_bid) < 6:
            inside_bid = best_bid
            inside_ask = best_ask

        # Apply skew adjustment
        # Skew > 0 (long): lower bid, lower ask to encourage selling
        # Skew < 0 (short): raise bid, raise ask to encourage buying
        market_bid = max(inside_bid - skew, best_bid)
        market_ask = min(inside_ask - skew, best_ask)

        # Hard cap at fair value so we never cross it
        fair_value_bid = fair_value - 0.01
        fair_value_ask = fair_value + 0.01

        target_bid = min(market_bid, fair_value_bid)
        target_ask = max(market_ask, fair_value_ask)

        if market_bid <= fair_value_bid:
            bid_logic = "market"
        else:
            bid_logic = "fair value cap"

        if market_ask >= fair_value_ask:
            ask_logic = "market"
        else:
            ask_logic = "fair value cap"

        if bid_logic == "fair value cap" and ask_logic == "fair value cap":
            logic = "fair value cap"
        else:
            logic = "market"

        # Ensure we maintain minimum spread
        current_target_spread = target_ask - target_bid
        if current_target_spread < min_spread:
            # Fall back to single-sided quoting based on inventory skew.
            logic = "minimum spread fallback"
            if skew < 0:
                target_bid = min(inside_bid, fair_value)
                target_ask = None
            elif skew > 0:
                target_bid = None
                target_ask = max(inside_ask, fair_value)
            else:
                target_bid = None
                target_ask = None

        # Safety: ensure bid doesn't exceed ask
        if target_bid is not None and target_ask is not None and target_bid >= target_ask:
            half_spread = min_spread / 2.0
            target_bid = fair_value - half_spread
            target_ask = fair_value + half_spread
            logic = "minimum spread fallback"

        if target_bid is not None:
            target_bid = max(target_bid, 0.01)
        if target_ask is not None:
            target_ask = min(target_ask, 0.99)

        price_calc = {
            "skew": skew,
            "best_bid": best_bid,
            "market_bid": market_bid,
            "fair_value_bid": fair_value_bid,
            "best_ask": best_ask,
            "market_ask": market_ask,
            "fair_value_ask": fair_value_ask,
            "fair_value": fair_value,
            "logic": logic,
        }

        return target_bid, target_ask, price_calc

    def _get_last_trade_info(
        self,
        trades: List[Dict[str, Any]],
    ) -> tuple[Optional[float], Optional[str]]:
        if not trades:
            return None, None

        latest_trade = None
        latest_timestamp = None
        latest_timestamp_raw = None
        latest_price = None

        for trade in trades:
            quantity_raw = trade.get("quantity")
            if quantity_raw is not None:
                try:
                    quantity = float(quantity_raw)
                except (TypeError, ValueError):
                    quantity = None
                if quantity == 1:
                    continue

            price = self._extract_trade_price(trade)
            if price is None:
                continue

            timestamp_raw = trade.get("created_time") or trade.get("created_at") or trade.get("timestamp")
            timestamp = self._parse_trade_timestamp(timestamp_raw) if timestamp_raw else None

            if timestamp is None:
                if latest_trade is None:
                    latest_trade = trade
                    latest_timestamp_raw = timestamp_raw
                    latest_price = price
                continue

            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_trade = trade
                latest_timestamp = timestamp
                latest_timestamp_raw = timestamp_raw
                latest_price = price

        if latest_trade is None or latest_price is None:
            return None, None

        return latest_price, latest_timestamp_raw

    def _extract_trade_price(self, trade: Dict[str, Any]) -> Optional[float]:
        raw_price = trade.get("yes_price")
        if raw_price is None:
            raw_price = trade.get("price")
        if raw_price is None:
            return None

        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return None

        # logger.info(f"extracting price: {price}")

        # price returned from API in whole cents
        price = price / 100.0
        return price

    def _parse_trade_timestamp(self, timestamp: Optional[str]) -> Optional[datetime]:
        if not timestamp:
            return None
        try:
            timestamp = timestamp.replace("Z", "+00:00")
            if "." in timestamp and "+" in timestamp:
                date_part, rest = timestamp.split(".")
                microseconds, timezone = rest.split("+")
                microseconds = microseconds.ljust(6, "0")[:6]
                timestamp = f"{date_part}.{microseconds}+{timezone}"
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None

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
                if utilization >= 1.0:
                    return 0
                # Scale size down as we approach limit
                if utilization > 0.8:
                    size_factor = (1.0 - utilization) / 0.2
                    return int(base_size * size_factor)
            
                # If not close to limit, return base or remaining capacity (if position is positive)
                return int(min(base_size, max_inventory - position_qty))        
            
            # otherwise return max base_size - position quantity (so if we are short 17, then set quantity to 17)
            return int(max(base_size, -position_qty))

        else:
            # Asking (selling YES) - reduce if short
            if position_qty < 0:
                if utilization >= 1.0:
                    return 0
                if utilization > 0.8:
                    size_factor = (1.0 - utilization) / 0.2
                    return int(base_size * size_factor)
                 
                # If not close to limit, return base or remaining capacity (if position is negative)
                return int(min(base_size, abs(max_inventory + position_qty)))
            
            # otherwise return max base_size, position quantity (so if we are long 17, then set quantity to 17)
            return int(max(base_size, position_qty))            

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
