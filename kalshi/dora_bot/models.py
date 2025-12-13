"""Data models for the Kalshi market making bot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, List, Tuple
from decimal import Decimal


@dataclass
class OrderBook:
    """Order book snapshot for a market with depth."""
    market_id: str
    # Top of book (for backwards compatibility)
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_size: int = 0
    ask_size: int = 0
    # Order book depth (up to 3 levels)
    # Each level is (price, size)
    bid_levels: List[Tuple[float, int]] = field(default_factory=list)
    ask_levels: List[Tuple[float, int]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def spread(self) -> Optional[float]:
        """Calculate the spread between best bid and ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate the mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def total_bid_depth(self) -> int:
        """Total size across all bid levels."""
        return sum(size for _, size in self.bid_levels)

    @property
    def total_ask_depth(self) -> int:
        """Total size across all ask levels."""
        return sum(size for _, size in self.ask_levels)


@dataclass
class Order:
    """Represents an order on the exchange.

    NOTE: Uses Kalshi's API format:
    - side: 'yes' or 'no' (Kalshi's format, not bid/ask)
    - price: Always the YES price (what Kalshi stores in yes_price field)

    Examples:
    - Buying YES at $0.33: side='yes', price=0.33
    - Selling YES at $0.33 (buying NO): side='no', price=0.33 (yes_price in API)
    """
    order_id: str
    market_id: str
    side: Literal["yes", "no"]  # Kalshi format: yes = buying YES, no = buying NO (selling YES)
    price: float  # Always YES price (0.01 to 0.99) regardless of side
    size: int
    filled_size: int = 0
    status: Literal["pending", "resting", "filled", "cancelled"] = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    tif: str = "gtc"  # time in force: gtc, ioc, fok

    @property
    def remaining_size(self) -> int:
        """Calculate remaining unfilled size."""
        return self.size - self.filled_size

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in ["pending", "resting"] and self.remaining_size > 0


@dataclass
class Fill:
    """Represents a fill/trade execution."""
    fill_id: str
    order_id: str
    market_id: str
    side: Literal["yes", "no"]
    price: float
    size: int
    timestamp: datetime
    fees: float = 0.0

    @property
    def notional(self) -> float:
        """Calculate notional value."""
        return self.price * self.size


@dataclass
class Position:
    """Position in a specific market."""
    market_id: str
    yes_qty: int = 0  # Positive = long, negative = short
    no_qty: int = 0
    avg_cost_yes: float = 0.0
    avg_cost_no: float = 0.0
    realized_pnl: float = 0.0

    @property
    def net_position(self) -> int:
        """Net position (yes_qty - no_qty)."""
        return self.yes_qty - self.no_qty

    @property
    def total_exposure(self) -> int:
        """Total exposure (absolute value)."""
        return abs(self.yes_qty) + abs(self.no_qty)

    def update_from_fill(self, fill: Fill):
        """Update position from a fill."""
        if fill.side == "yes":
            # Buying YES
            if self.yes_qty >= 0:
                # Adding to long or opening long
                total_cost = self.avg_cost_yes * self.yes_qty + fill.price * fill.size
                self.yes_qty += fill.size
                self.avg_cost_yes = total_cost / self.yes_qty if self.yes_qty > 0 else 0
            else:
                # Closing short position
                realized = (-self.avg_cost_yes - fill.price) * min(abs(self.yes_qty), fill.size)
                self.realized_pnl += realized
                self.yes_qty += fill.size
                if self.yes_qty > 0:
                    self.avg_cost_yes = fill.price
        else:
            # Selling YES (buying NO)
            if self.yes_qty <= 0:
                # Adding to short or opening short
                total_cost = self.avg_cost_yes * abs(self.yes_qty) + fill.price * fill.size
                self.yes_qty -= fill.size
                self.avg_cost_yes = total_cost / abs(self.yes_qty) if self.yes_qty != 0 else 0
            else:
                # Closing long position
                realized = (fill.price - self.avg_cost_yes) * min(self.yes_qty, fill.size)
                self.realized_pnl += realized
                self.yes_qty -= fill.size
                if self.yes_qty < 0:
                    self.avg_cost_yes = fill.price


@dataclass
class MarketConfig:
    """Per-market configuration."""
    market_id: str
    enabled: bool = False
    max_inventory_yes: int = 100
    max_inventory_no: int = 100
    min_spread: float = 0.06  # Minimum spread required to quote
    quote_size: int = 10
    inventory_skew_factor: float = 0.5  # How aggressively to skew quotes based on inventory
    toxicity_score: Optional[float] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GlobalConfig:
    """Global bot configuration."""
    max_total_exposure: int = 500
    max_daily_loss: float = 100.0
    loop_interval_ms: int = 5000  # 5 seconds for testing
    trading_enabled: bool = True
    risk_aversion_k: float = 0.5
    cancel_on_startup: bool = True  # Whether to cancel all orders on bot startup


@dataclass
class RiskState:
    """Current risk state."""
    daily_pnl: float = 0.0
    last_fill_timestamp: Optional[datetime] = None
    trading_halted: bool = False
    halt_reason: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TargetOrder:
    """Desired order that strategy wants to place.

    NOTE: price is always in YES terms, side is 'bid' or 'ask'
    - bid = buying YES at this price
    - ask = selling YES at this price
    """
    market_id: str
    side: Literal["bid", "ask"]
    price: float  # Always YES price
    size: int

    def matches(self, order: Order, tolerance: float = 0.01) -> bool:
        """Check if this target matches an existing order.

        Order.side is "yes" or "no" (Kalshi format), Order.price is always YES price
        TargetOrder.side is "bid" or "ask", TargetOrder.price is always YES price
        - bid (buy YES at X) matches order: side="yes", price=X
        - ask (sell YES at X) matches order: side="no", price=X (same YES price!)
        """
        # Check market and size
        if order.market_id != self.market_id or order.size < self.size:
            return False

        # Match bid/ask to yes/no
        if self.side == "bid":
            # Buying YES at price X - should match YES order at price X
            return order.side == "yes" and abs(order.price - self.price) <= tolerance
        else:  # self.side == "ask"
            # Selling YES at price X - should match NO order with same YES price X
            # (NO order with yes_price=X means buying NO at X, which is selling YES at X)
            return order.side == "no" and abs(order.price - self.price) <= tolerance


@dataclass
class Balance:
    """Account balance information."""
    balance: float
    payout: float

    @property
    def total(self) -> float:
        """Total available funds."""
        return self.balance + self.payout
