"""
User Evaluation Service

Business logic for analyzing individual wallet performance on Polymarket.
Fetches trades for a wallet, calculates current position values using live prices,
and provides performance metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import requests


@dataclass
class WalletTrade:
    """Represents a single trade from a wallet's history."""
    timestamp: int
    datetime: datetime
    wallet: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    outcome: str
    token_id: str
    condition_id: str
    market_title: str
    transaction_hash: str


@dataclass
class TradeWithCurrentValue:
    """A trade with its current unwind value calculated."""
    timestamp: int
    datetime: datetime
    side: str
    price: float
    size: float
    outcome: str
    token_id: str
    market_title: str
    current_price: float
    current_value: float
    pnl: float
    transaction_hash: str


class UserEvaluationService:
    """Service for evaluating individual wallet performance."""

    DATA_API_BASE = "https://data-api.polymarket.com"
    PRICE_API_BASE = "https://clob.polymarket.com"

    def __init__(self):
        self.session = requests.Session()

    def get_wallet_trades(self, wallet: str, limit: int = 1000) -> list[WalletTrade]:
        """
        Fetch recent trades for a wallet.

        Args:
            wallet: The wallet address to fetch trades for
            limit: Maximum number of trades to retrieve (default 1000)

        Returns:
            List of WalletTrade objects
        """
        url = f"{self.DATA_API_BASE}/trades"
        params = {"user": wallet, "limit": limit}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        trades = []
        for trade_data in response.json():
            ts = trade_data.get("timestamp")
            if not ts:
                continue

            trades.append(WalletTrade(
                timestamp=int(ts),
                datetime=datetime.fromtimestamp(int(ts)),
                wallet=wallet,
                side=trade_data.get("side", "BUY"),
                price=float(trade_data.get("price", 0) or 0),
                size=float(trade_data.get("size", 0) or 0),
                outcome=trade_data.get("outcome", "Unknown"),
                token_id=trade_data.get("asset", "") or "",
                condition_id=trade_data.get("conditionId", "") or "",
                market_title=trade_data.get("title") or trade_data.get("question") or "Unknown Market",
                transaction_hash=trade_data.get("transactionHash", "")
            ))

        return trades

    def get_current_price(self, token_id: str, side: str) -> Optional[float]:
        """
        Get current price to unwind a position.

        Args:
            token_id: The token ID (asset) from the trade
            side: "SELL" to unwind a BUY position, "BUY" to unwind a SELL position

        Returns:
            Current price as float, or None if unavailable
        """
        if not token_id or side not in {"BUY", "SELL"}:
            return None

        url = f"{self.PRICE_API_BASE}/price"
        params = {"token_id": token_id, "side": side}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            price_str = data.get("price")
            return float(price_str) if price_str else None
        except Exception:
            return None

    def evaluate_trades_with_current_prices(
        self,
        trades: list[WalletTrade]
    ) -> list[TradeWithCurrentValue]:
        """
        Calculate current value for each trade by fetching unwind prices.

        For BUY trades: fetches current SELL price (price to exit the position)
        For SELL trades: fetches current BUY price (price to close the short)

        Args:
            trades: List of WalletTrade objects

        Returns:
            List of TradeWithCurrentValue objects with P&L calculations
        """
        trades_with_values = []

        for trade in trades:
            # Determine the side needed to unwind this position
            unwind_side = "SELL" if trade.side == "BUY" else "BUY"

            # Fetch current price to unwind
            current_price = self.get_current_price(trade.token_id, unwind_side)

            # If we can't get current price, use the trade price as fallback
            if current_price is None:
                current_price = trade.price

            # Calculate current value and P&L
            if trade.side == "BUY":
                # BUY trade: profit if current price > entry price
                current_value = trade.size * current_price
                pnl = (current_price - trade.price) * trade.size
            else:
                # SELL trade: profit if current price < entry price
                current_value = trade.size * current_price
                pnl = (trade.price - current_price) * trade.size

            trades_with_values.append(TradeWithCurrentValue(
                timestamp=trade.timestamp,
                datetime=trade.datetime,
                side=trade.side,
                price=trade.price,
                size=trade.size,
                outcome=trade.outcome,
                token_id=trade.token_id,
                market_title=trade.market_title,
                current_price=current_price,
                current_value=current_value,
                pnl=pnl,
                transaction_hash=trade.transaction_hash
            ))

        return trades_with_values

    def analyze_wallet(self, wallet: str, limit: int = 1000) -> dict:
        """
        Complete analysis of a wallet's trading performance.

        Args:
            wallet: The wallet address to analyze
            limit: Maximum number of trades to retrieve

        Returns:
            Dictionary containing:
            - trades: List of TradeWithCurrentValue objects
            - summary: Summary statistics
        """
        # Fetch trades
        trades = self.get_wallet_trades(wallet, limit)

        if not trades:
            return {
                "trades": [],
                "summary": {
                    "total_trades": 0,
                    "total_pnl": 0.0,
                    "total_volume": 0.0,
                    "profitable_trades": 0,
                    "losing_trades": 0
                }
            }

        # Evaluate with current prices
        trades_with_values = self.evaluate_trades_with_current_prices(trades)

        # Calculate summary statistics
        total_pnl = sum(t.pnl for t in trades_with_values)
        total_volume = sum(t.size * t.price for t in trades_with_values)
        profitable_trades = sum(1 for t in trades_with_values if t.pnl > 0)
        losing_trades = sum(1 for t in trades_with_values if t.pnl < 0)

        return {
            "trades": trades_with_values,
            "summary": {
                "total_trades": len(trades_with_values),
                "total_pnl": total_pnl,
                "total_volume": total_volume,
                "profitable_trades": profitable_trades,
                "losing_trades": losing_trades,
                "win_rate": (profitable_trades / len(trades_with_values) * 100) if trades_with_values else 0.0
            }
        }
