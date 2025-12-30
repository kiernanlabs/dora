"""
P&L and Trade Calculations for Dora Manager
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


def calculate_fill_fee(price: float, count: int) -> float:
    """Calculate the fee for a fill using Kalshi's fee formula.

    Formula: 0.0175 * C * P * (1-P)
    where P = price in dollars (0.50 for 50 cents)
          C = number of contracts

    Args:
        price: Price per contract in dollars (e.g., 0.50 for 50 cents)
        count: Number of contracts traded

    Returns:
        Fee amount in dollars
    """
    return 0.0175 * count * price * (1 - price)


@dataclass
class PositionState:
    """Track position state for a market."""
    net_yes_qty: int = 0
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class MarketSummary:
    """Summary statistics for a single market."""
    market_id: str
    event_ticker: str = ""

    # Position info
    net_position: int = 0
    avg_cost: Optional[float] = None

    # P&L metrics
    realized_pnl_all_time: float = 0.0
    realized_pnl_window: float = 0.0
    unrealized_pnl_worst: Optional[float] = None
    unrealized_pnl_best: Optional[float] = None

    # Trade metrics in window
    trade_count: int = 0
    contracts_traded: int = 0
    fees_paid: float = 0.0

    # Order info
    our_bid_price: Optional[float] = None
    our_ask_price: Optional[float] = None
    is_bid_active: bool = False
    is_ask_active: bool = False
    market_best_bid: Optional[float] = None
    market_best_ask: Optional[float] = None

    # Flags
    flagged_for_deactivation: bool = False
    deactivation_reason: str = ""


@dataclass
class TradingSummary:
    """Overall trading summary."""
    # Time window info
    window_hours: int = 3
    report_timestamp: str = ""
    environment: str = ""

    # Aggregate P&L
    total_realized_pnl_all_time: float = 0.0
    total_realized_pnl_window: float = 0.0
    total_unrealized_pnl_worst: float = 0.0
    total_unrealized_pnl_best: float = 0.0

    # Aggregate trade stats
    total_trade_count: int = 0
    total_contracts_traded: int = 0
    total_fees_paid: float = 0.0

    # Exposure
    total_exposure: int = 0
    active_bids_count: int = 0
    active_bids_qty: int = 0
    active_asks_count: int = 0
    active_asks_qty: int = 0

    # Market counts
    active_markets_count: int = 0
    markets_with_trades: int = 0
    markets_flagged_count: int = 0

    # Market details
    market_summaries: List[MarketSummary] = field(default_factory=list)
    markets_with_window_trades: List[MarketSummary] = field(default_factory=list)
    flagged_markets: List[MarketSummary] = field(default_factory=list)


class TradingCalculator:
    """Calculate trading statistics and P&L."""

    def __init__(
        self,
        positions: Dict[str, Dict],
        market_configs: List[Dict],
        all_trades: List[Dict],
        window_trades: List[Dict],
        decisions_by_market: Dict[str, Dict],
        open_orders_by_market: Dict[str, List[Dict]],
        window_hours: int = 3,
        min_pnl_threshold: float = -3.0,
    ):
        """
        Initialize calculator.

        Args:
            positions: Current positions from DynamoDB
            market_configs: Enabled market configurations
            all_trades: All trades for cost basis calculation
            window_trades: Trades in the reporting window
            decisions_by_market: Most recent decision per market
            open_orders_by_market: Open orders grouped by market
            window_hours: Hours for the reporting window
            min_pnl_threshold: P&L threshold for flagging markets (default -$3)
        """
        self.positions = positions
        self.market_configs = market_configs
        self.all_trades = all_trades
        self.window_trades = window_trades
        self.decisions_by_market = decisions_by_market
        self.open_orders_by_market = open_orders_by_market
        self.window_hours = window_hours
        self.min_pnl_threshold = min_pnl_threshold

        # Calculate cutoff time once to ensure consistency across all calculations
        self.cutoff_time = datetime.now(timezone.utc) - timedelta(hours=window_hours)

        # Index trades by market
        self.trades_by_market: Dict[str, List[Dict]] = {}
        for trade in all_trades:
            market_id = trade.get('market_id')
            if market_id:
                if market_id not in self.trades_by_market:
                    self.trades_by_market[market_id] = []
                self.trades_by_market[market_id].append(trade)

        # Index window trades by market
        self.window_trades_by_market: Dict[str, List[Dict]] = {}
        for trade in window_trades:
            market_id = trade.get('market_id')
            if market_id:
                if market_id not in self.window_trades_by_market:
                    self.window_trades_by_market[market_id] = []
                self.window_trades_by_market[market_id].append(trade)

    def calculate_realized_pnl_from_trades(
        self,
        trades: List[Dict]
    ) -> Tuple[float, Dict[str, PositionState]]:
        """
        Calculate realized P&L from a list of trades using proper cost basis tracking.

        Returns:
            Tuple of (total_realized_pnl, position_states_by_market)
        """
        positions: Dict[str, PositionState] = {}

        # Sort trades chronologically
        sorted_trades = sorted(
            trades,
            key=lambda t: t.get('fill_timestamp') or t.get('timestamp', '')
        )

        for trade in sorted_trades:
            market_id = trade.get('market_id')
            if not market_id:
                continue

            if market_id not in positions:
                positions[market_id] = PositionState()

            pos = positions[market_id]
            side = trade.get('side', '')
            price = trade.get('price', 0.0)
            size = trade.get('size', 0)
            fees = trade.get('fees', 0.0) or 0.0

            # Subtract fees from realized P&L on every fill
            pos.realized_pnl -= fees

            # Update position using same logic as Position.update_from_fill()
            if side in ['buy', 'yes']:
                # Bid fill - buying YES contracts
                if pos.net_yes_qty >= 0:
                    # Adding to long position
                    total_cost = pos.avg_buy_price * pos.net_yes_qty + price * size
                    pos.net_yes_qty += size
                    pos.avg_buy_price = total_cost / pos.net_yes_qty if pos.net_yes_qty > 0 else 0
                else:
                    # Closing short position - realize P&L
                    close_qty = min(abs(pos.net_yes_qty), size)
                    realized = (pos.avg_sell_price - price) * close_qty
                    pos.realized_pnl += realized
                    pos.net_yes_qty += size

                    if pos.net_yes_qty > 0:
                        pos.avg_buy_price = price
            else:
                # Ask fill - selling YES contracts
                if pos.net_yes_qty <= 0:
                    # Adding to short position
                    total_cost = pos.avg_sell_price * abs(pos.net_yes_qty) + price * size
                    pos.net_yes_qty -= size
                    pos.avg_sell_price = total_cost / abs(pos.net_yes_qty) if pos.net_yes_qty != 0 else 0
                else:
                    # Closing long position - realize P&L
                    close_qty = min(pos.net_yes_qty, size)
                    realized = (price - pos.avg_buy_price) * close_qty
                    pos.realized_pnl += realized
                    pos.net_yes_qty -= size

                    if pos.net_yes_qty < 0:
                        pos.avg_sell_price = price

        total_pnl = sum(p.realized_pnl for p in positions.values())
        return total_pnl, positions

    def calculate_window_pnl_for_market(self, market_id: str) -> float:
        """Calculate realized P&L for a market in the reporting window."""
        all_market_trades = self.trades_by_market.get(market_id, [])
        if not all_market_trades:
            return 0.0

        # Sort trades chronologically
        sorted_trades = sorted(
            all_market_trades,
            key=lambda t: t.get('fill_timestamp') or t.get('timestamp', '')
        )

        # Use the cutoff time calculated in __init__ for consistency
        cutoff = self.cutoff_time

        # Track position state
        pos = PositionState()
        realized_pnl_before_window = 0.0

        for trade in sorted_trades:
            ts_str = trade.get('fill_timestamp') or trade.get('timestamp', '')
            if not ts_str:
                continue

            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception:
                continue

            side = trade.get('side', '')
            price = trade.get('price', 0.0)
            size = trade.get('size', 0)
            fees = trade.get('fees', 0.0) or 0.0

            # Subtract fees
            pos.realized_pnl -= fees

            # Update position
            if side in ['buy', 'yes']:
                if pos.net_yes_qty >= 0:
                    total_cost = pos.avg_buy_price * pos.net_yes_qty + price * size
                    pos.net_yes_qty += size
                    pos.avg_buy_price = total_cost / pos.net_yes_qty if pos.net_yes_qty > 0 else 0
                else:
                    close_qty = min(abs(pos.net_yes_qty), size)
                    realized = (pos.avg_sell_price - price) * close_qty
                    pos.realized_pnl += realized
                    pos.net_yes_qty += size
                    if pos.net_yes_qty > 0:
                        pos.avg_buy_price = price
            else:
                if pos.net_yes_qty <= 0:
                    total_cost = pos.avg_sell_price * abs(pos.net_yes_qty) + price * size
                    pos.net_yes_qty -= size
                    pos.avg_sell_price = total_cost / abs(pos.net_yes_qty) if pos.net_yes_qty != 0 else 0
                else:
                    close_qty = min(pos.net_yes_qty, size)
                    realized = (price - pos.avg_buy_price) * close_qty
                    pos.realized_pnl += realized
                    pos.net_yes_qty -= size
                    if pos.net_yes_qty < 0:
                        pos.avg_sell_price = price

            # Track P&L at cutoff
            # Use < instead of <= to match get_trades_in_window() which uses >= cutoff
            if ts < cutoff:
                realized_pnl_before_window = pos.realized_pnl

        return pos.realized_pnl - realized_pnl_before_window

    def calculate_market_summary(self, config: Dict) -> MarketSummary:
        """Calculate summary for a single market."""
        market_id = config.get('market_id', '')
        summary = MarketSummary(
            market_id=market_id,
            event_ticker=config.get('event_ticker', '')
        )

        # Get position info
        position = self.positions.get(market_id, {})
        summary.net_position = int(position.get('net_yes_qty', 0))
        summary.realized_pnl_all_time = position.get('realized_pnl', 0.0)

        if summary.net_position > 0:
            summary.avg_cost = position.get('avg_buy_price')
        elif summary.net_position < 0:
            summary.avg_cost = position.get('avg_sell_price')

        # Get order book from decision
        decision = self.decisions_by_market.get(market_id, {})
        order_book = decision.get('order_book_snapshot', {})
        summary.market_best_bid = order_book.get('best_bid')
        summary.market_best_ask = order_book.get('best_ask')

        # Get our live orders
        market_orders = self.open_orders_by_market.get(market_id, [])
        live_bids = [o for o in market_orders if o.get('side') == 'yes']
        live_asks = [o for o in market_orders if o.get('side') == 'no']
        live_bids.sort(key=lambda x: x.get('price', 0) or 0, reverse=True)
        live_asks.sort(key=lambda x: x.get('price', 0) or 0)

        if live_bids:
            summary.our_bid_price = live_bids[0].get('price', 0)
        if live_asks:
            summary.our_ask_price = live_asks[0].get('price', 0)

        # Check if orders are competitive
        if summary.our_bid_price and summary.market_best_bid:
            summary.is_bid_active = summary.our_bid_price >= (summary.market_best_bid - 0.0001)
        elif summary.our_bid_price:
            summary.is_bid_active = True

        if summary.our_ask_price and summary.market_best_ask:
            summary.is_ask_active = summary.our_ask_price <= (summary.market_best_ask + 0.0001)
        elif summary.our_ask_price:
            summary.is_ask_active = True

        # Calculate unrealized P&L
        if summary.net_position != 0 and summary.avg_cost is not None:
            # Worst case - exit at market best
            if summary.net_position > 0 and summary.market_best_bid:
                exit_fee = calculate_fill_fee(summary.market_best_bid, summary.net_position)
                summary.unrealized_pnl_worst = (
                    (summary.market_best_bid - summary.avg_cost) * summary.net_position - exit_fee
                )
            elif summary.net_position < 0 and summary.market_best_ask:
                exit_fee = calculate_fill_fee(summary.market_best_ask, abs(summary.net_position))
                summary.unrealized_pnl_worst = (
                    (summary.avg_cost - summary.market_best_ask) * abs(summary.net_position) - exit_fee
                )

            # Best case - exit at our competitive orders
            if summary.net_position > 0:
                if summary.our_ask_price and summary.is_ask_active:
                    exit_fee = calculate_fill_fee(summary.our_ask_price, summary.net_position)
                    summary.unrealized_pnl_best = (
                        (summary.our_ask_price - summary.avg_cost) * summary.net_position - exit_fee
                    )
                elif summary.market_best_bid:
                    exit_fee = calculate_fill_fee(summary.market_best_bid, summary.net_position)
                    summary.unrealized_pnl_best = (
                        (summary.market_best_bid - summary.avg_cost) * summary.net_position - exit_fee
                    )
            elif summary.net_position < 0:
                if summary.our_bid_price and summary.is_bid_active:
                    exit_fee = calculate_fill_fee(summary.our_bid_price, abs(summary.net_position))
                    summary.unrealized_pnl_best = (
                        (summary.avg_cost - summary.our_bid_price) * abs(summary.net_position) - exit_fee
                    )
                elif summary.market_best_ask:
                    exit_fee = calculate_fill_fee(summary.market_best_ask, abs(summary.net_position))
                    summary.unrealized_pnl_best = (
                        (summary.avg_cost - summary.market_best_ask) * abs(summary.net_position) - exit_fee
                    )

        # Calculate window metrics
        window_trades = self.window_trades_by_market.get(market_id, [])
        summary.trade_count = len(window_trades)
        summary.contracts_traded = sum(t.get('size', 0) for t in window_trades)
        summary.fees_paid = sum(t.get('fees', 0.0) or 0.0 for t in window_trades)

        # Calculate window P&L
        summary.realized_pnl_window = self.calculate_window_pnl_for_market(market_id)

        # Check if market should be flagged for deactivation
        # Only flag if there were trades in the window AND P&L is below threshold
        if summary.trade_count > 0 and summary.realized_pnl_window < self.min_pnl_threshold:
            summary.flagged_for_deactivation = True
            summary.deactivation_reason = (
                f"Realized P&L of ${summary.realized_pnl_window:.2f} in last {self.window_hours}hrs "
                f"is below threshold of ${self.min_pnl_threshold:.2f}"
            )

        return summary

    def calculate_summary(self) -> TradingSummary:
        """Calculate full trading summary."""
        summary = TradingSummary(
            window_hours=self.window_hours,
            report_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Calculate all-time realized P&L
        total_all_time_pnl, _ = self.calculate_realized_pnl_from_trades(self.all_trades)
        summary.total_realized_pnl_all_time = total_all_time_pnl

        # Aggregate window trade stats
        summary.total_trade_count = len(self.window_trades)
        summary.total_contracts_traded = sum(t.get('size', 0) for t in self.window_trades)
        summary.total_fees_paid = sum(t.get('fees', 0.0) or 0.0 for t in self.window_trades)

        summary.active_markets_count = len(self.market_configs)

        # Calculate per-market summaries
        for config in self.market_configs:
            market_summary = self.calculate_market_summary(config)
            summary.market_summaries.append(market_summary)

            # Aggregate window P&L from individual market calculations
            # This is the correct approach because calculate_window_pnl_for_market()
            # uses all trades to establish proper cost basis before the window
            summary.total_realized_pnl_window += market_summary.realized_pnl_window

            # Aggregate exposure
            summary.total_exposure += abs(market_summary.net_position)

            # Aggregate unrealized P&L
            if market_summary.unrealized_pnl_worst is not None:
                summary.total_unrealized_pnl_worst += market_summary.unrealized_pnl_worst
            if market_summary.unrealized_pnl_best is not None:
                summary.total_unrealized_pnl_best += market_summary.unrealized_pnl_best

            # Track active orders
            if market_summary.is_bid_active and market_summary.our_bid_price:
                summary.active_bids_count += 1
                bid_orders = [o for o in self.open_orders_by_market.get(market_summary.market_id, [])
                             if o.get('side') == 'yes']
                summary.active_bids_qty += sum(o.get('size', 0) for o in bid_orders)

            if market_summary.is_ask_active and market_summary.our_ask_price:
                summary.active_asks_count += 1
                ask_orders = [o for o in self.open_orders_by_market.get(market_summary.market_id, [])
                             if o.get('side') == 'no']
                summary.active_asks_qty += sum(o.get('size', 0) for o in ask_orders)

            # Track markets with trades in window
            if market_summary.trade_count > 0:
                summary.markets_with_trades += 1

            # Include markets with trades OR open positions in the table
            if market_summary.trade_count > 0 or market_summary.net_position != 0:
                summary.markets_with_window_trades.append(market_summary)

            # Track flagged markets
            if market_summary.flagged_for_deactivation:
                summary.markets_flagged_count += 1
                summary.flagged_markets.append(market_summary)

        # Sort markets with trades by P&L (worst first)
        summary.markets_with_window_trades.sort(key=lambda m: m.realized_pnl_window)

        return summary
