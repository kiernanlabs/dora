"""
Polymarket Service Layer

Business logic for fetching and analyzing Polymarket trade data.
This module is designed to be backend-agnostic and can be used with
Streamlit, Lambda, FastAPI, or any other framework.
"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests


@dataclass
class Market:
    """Represents a Polymarket market."""
    id: str
    condition_id: str
    slug: str
    question: str
    description: str
    outcomes: list
    outcome_prices: list
    volume: float
    active: bool
    closed: bool


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: int
    datetime: datetime
    wallet: str
    wallet_name: str  # Display name if available
    side: str
    price: float
    size: float
    outcome: str
    transaction_hash: str


@dataclass
class WalletTrade:
    """Represents a trade pulled for a specific wallet with market metadata."""
    timestamp: int
    datetime: datetime
    wallet: str
    wallet_name: str
    side: str
    price: float
    size: float
    outcome: str
    outcome_index: Optional[int]
    asset: str
    condition_id: str
    market_slug: str
    event_slug: str
    market_title: str
    icon: str
    bio: str
    profile_image: str
    transaction_hash: str


@dataclass
class WalletMarketSummary:
    """Aggregated stats for a wallet within a market."""
    market_title: str
    market_slug: str
    event_slug: str
    icon: str
    total_trades: int
    total_buy_dollars: float
    trade_counts_by_outcome: dict
    avg_trade_size: float
    avg_trade_dollars: float
    avg_time_between_trades_seconds: float
    pnl: float
    last_trade_time: datetime
    outcome_details: list
    trade_details: list


@dataclass
class WalletClassification:
    """Classification result for a wallet."""
    wallet: str
    classification: str
    trade_count: int
    total_size: float
    buy_count: int
    sell_count: int


@dataclass
class ExposureData:
    """Cumulative exposure data for a classification by outcome."""
    timestamps: list
    exposure_by_outcome: dict  # {outcome: [cumulative exposure over time]}
    cumulative_by_outcome: dict  # {outcome: current cumulative exposure}


@dataclass
class AnalysisResult:
    """Complete analysis result for a market."""
    trades: list[Trade]
    wallet_classifications: dict[str, str]
    exposure_by_classification: dict[str, ExposureData]
    outcomes_data: dict
    summary: dict


class PolymarketService:
    """Service class for Polymarket API operations and analysis."""

    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    DATA_API_BASE = "https://data-api.polymarket.com"

    CLASSIFICATIONS = [
        "Micro-Bots",
        "Directional Whales",
        "Conviction Retail",
        "Light Market Makers",
        "Noise/Casuals"
    ]

    def __init__(self):
        self.session = requests.Session()
        self.price_session = requests.Session()

    def get_event_markets(self, event_slug: str) -> list[Market]:
        """Fetch all markets for an event using the event slug."""
        url = f"{self.GAMMA_API_BASE}/events/slug/{event_slug}"
        response = self.session.get(url)
        response.raise_for_status()
        event_data = response.json()

        markets = []
        for market_data in event_data.get("markets", []):
            markets.append(Market(
                id=market_data.get("id", ""),
                condition_id=market_data.get("conditionId", ""),
                slug=market_data.get("slug", ""),
                question=market_data.get("question", ""),
                description=market_data.get("description", ""),
                outcomes=market_data.get("outcomes", "").split(",") if market_data.get("outcomes") else [],
                outcome_prices=market_data.get("outcomePrices", "").split(",") if market_data.get("outcomePrices") else [],
                volume=float(market_data.get("volume", 0) or 0),
                active=market_data.get("active", False),
                closed=market_data.get("closed", False)
            ))

        return markets

    def get_market_by_slug(self, slug: str) -> Market:
        """Fetch a single market by its slug."""
        url = f"{self.GAMMA_API_BASE}/markets/slug/{slug}"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()

        return Market(
            id=data.get("id", ""),
            condition_id=data.get("conditionId", ""),
            slug=data.get("slug", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            outcomes=data.get("outcomes", "").split(",") if data.get("outcomes") else [],
            outcome_prices=data.get("outcomePrices", "").split(",") if data.get("outcomePrices") else [],
            volume=float(data.get("volume", 0) or 0),
            active=data.get("active", False),
            closed=data.get("closed", False)
        )

    def get_trades(self, condition_id: str, limit: int = 100) -> list[Trade]:
        """Fetch recent trades for a market."""
        url = f"{self.DATA_API_BASE}/trades"
        params = {"market": condition_id, "limit": limit}
        response = self.session.get(url, params=params)
        response.raise_for_status()

        trades = []
        for trade_data in response.json():
            ts = trade_data.get("timestamp")
            if ts:
                wallet = trade_data.get("proxyWallet", "unknown")
                # Use name if available, otherwise use truncated wallet address
                wallet_name = trade_data.get("name") or trade_data.get("pseudonym") or ""
                trades.append(Trade(
                    timestamp=int(ts),
                    datetime=datetime.fromtimestamp(int(ts)),
                    wallet=wallet,
                    wallet_name=wallet_name,
                    side=trade_data.get("side", "BUY"),
                    price=float(trade_data.get("price", 0)),
                    size=float(trade_data.get("size", 0)),
                    outcome=trade_data.get("outcome", "Unknown"),
                    transaction_hash=trade_data.get("transactionHash", "")
                ))

        return trades

    def get_wallet_trades(self, wallet: str, limit: int = 400) -> list[WalletTrade]:
        """Fetch recent trades for a wallet across markets."""
        url = f"{self.DATA_API_BASE}/trades"
        params = {"user": wallet, "limit": limit}
        response = self.session.get(url, params=params)
        response.raise_for_status()

        trades: list[WalletTrade] = []
        for trade_data in response.json():
            ts = trade_data.get("timestamp")
            if not ts:
                continue

            trades.append(WalletTrade(
                timestamp=int(ts),
                datetime=datetime.fromtimestamp(int(ts)),
                wallet=wallet,
                wallet_name=trade_data.get("name") or trade_data.get("pseudonym") or "",
                side=trade_data.get("side", "BUY"),
                price=float(trade_data.get("price", 0) or 0),
                size=float(trade_data.get("size", 0) or 0),
                outcome=trade_data.get("outcome", "Unknown"),
                outcome_index=trade_data.get("outcomeIndex"),
                asset=trade_data.get("asset", "") or "",
                condition_id=trade_data.get("conditionId", "") or "",
                market_slug=trade_data.get("slug") or trade_data.get("marketSlug") or "",
                event_slug=trade_data.get("eventSlug", "") or "",
                market_title=trade_data.get("title") or trade_data.get("question") or "",
                icon=trade_data.get("icon", "") or "",
                bio=trade_data.get("bio", "") or "",
                profile_image=trade_data.get("profileImage") or trade_data.get("profileImageOptimized") or "",
                transaction_hash=trade_data.get("transactionHash", "")
            ))

        return trades

    def get_token_price(self, token_id: str, side: str) -> Optional[float]:
        """Fetch current price for a token and side via the clob price endpoint."""
        if not token_id or side not in {"BUY", "SELL"}:
            return None
        url = "https://clob.polymarket.com/price"
        params = {"token_id": token_id, "side": side}
        resp = self.price_session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price_str = data.get("price")
        try:
            return float(price_str)
        except (TypeError, ValueError):
            return None

    def get_wallet_market_summaries(
        self,
        wallet: str,
        trade_limit: int = 400
    ) -> dict:
        """
        Return aggregated stats for the most recent markets this wallet traded in.

        Includes profile metadata and per-market P/L using current outcome prices
        (from gamma API) when available.
        """
        trades = self.get_wallet_trades(wallet, trade_limit)
        if not trades:
            return {"profile": {}, "markets": []}

        # Preserve most recent market ordering by last trade timestamp
        trades_by_market: dict[str, list[WalletTrade]] = {}
        for trade in trades:
            key = trade.market_slug or trade.condition_id or "unknown"
            trades_by_market.setdefault(key, []).append(trade)

        ordered_markets = sorted(
            trades_by_market.items(),
            key=lambda item: max(t.timestamp for t in item[1]),
            reverse=True
        )

        # Preload price data for up to 10 markets to keep requests small
        price_maps: dict[str, dict] = {}
        token_price_cache: dict[tuple[str, str], Optional[float]] = {}
        for key, market_trades in ordered_markets[:10]:
            slug = market_trades[0].market_slug
            if not slug or slug in price_maps:
                price_maps.setdefault(slug or key, {})
                continue
            try:
                market = self.get_market_by_slug(slug)
                price_map = {}
                if market.outcomes and market.outcome_prices:
                    for name, price in zip(market.outcomes, market.outcome_prices):
                        try:
                            price_map[name] = float(price)
                        except (TypeError, ValueError):
                            continue
                price_maps[slug] = price_map
            except Exception:
                price_maps[slug] = {}

        summaries: list[WalletMarketSummary] = []
        for key, market_trades in ordered_markets[:10]:
            sorted_trades = sorted(market_trades, key=lambda t: t.timestamp)
            counts: dict = {}
            holdings: dict[str, float] = {}
            cash_flow: dict[str, float] = {}
            deltas: list[float] = []
            last_ts: Optional[int] = None
            last_price_seen: dict[str, float] = {}
            total_buy_dollars = 0.0
            asset_by_outcome: dict[str, str] = {}
            outcome_buy_qty: dict[str, float] = {}
            outcome_sell_qty: dict[str, float] = {}
            outcome_buy_dollars: dict[str, float] = {}
            outcome_sell_dollars: dict[str, float] = {}
            trade_details: list[dict] = []

            for trade in sorted_trades:
                counts.setdefault(trade.outcome, {"BUY": 0, "SELL": 0})
                counts[trade.outcome][trade.side] = counts[trade.outcome].get(trade.side, 0) + 1

                if last_ts is not None:
                    deltas.append(trade.timestamp - last_ts)
                last_ts = trade.timestamp

                holdings[trade.outcome] = holdings.get(trade.outcome, 0.0) + (trade.size if trade.side == "BUY" else -trade.size)
                cash_flow[trade.outcome] = cash_flow.get(trade.outcome, 0.0) + (
                    -trade.price * trade.size if trade.side == "BUY" else trade.price * trade.size
                )
                if trade.side == "BUY":
                    total_buy_dollars += trade.price * trade.size
                    outcome_buy_qty[trade.outcome] = outcome_buy_qty.get(trade.outcome, 0.0) + trade.size
                    outcome_buy_dollars[trade.outcome] = outcome_buy_dollars.get(trade.outcome, 0.0) + trade.price * trade.size
                else:
                    outcome_sell_qty[trade.outcome] = outcome_sell_qty.get(trade.outcome, 0.0) + trade.size
                    outcome_sell_dollars[trade.outcome] = outcome_sell_dollars.get(trade.outcome, 0.0) + trade.price * trade.size
                last_price_seen[trade.outcome] = trade.price
                if trade.asset:
                    asset_by_outcome[trade.outcome] = trade.asset

                # Per-trade exit price and P/L
                trade_exit_side = "SELL" if trade.side == "BUY" else "BUY"
                trade_exit_price = None
                cache_key = (trade.asset, trade_exit_side)
                if trade.asset:
                    if cache_key not in token_price_cache:
                        try:
                            token_price_cache[cache_key] = self.get_token_price(trade.asset, trade_exit_side)
                        except Exception:
                            token_price_cache[cache_key] = None
                    trade_exit_price = token_price_cache.get(cache_key)
                if trade_exit_price is None:
                    # Fallback to price map / last seen
                    trade_exit_price = price_maps.get(trade.market_slug or key, {}).get(trade.outcome, trade.price)

                trade_pnl = (trade_exit_price - trade.price) * trade.size if trade.side == "BUY" else (trade.price - trade_exit_price) * trade.size
                trade_return = (trade_pnl / (trade.price * trade.size)) * 100 if trade.price and trade.size else 0.0
                trade_details.append({
                    "timestamp": trade.timestamp,
                    "datetime": trade.datetime,
                    "outcome": trade.outcome,
                    "side": trade.side,
                    "price": trade.price,
                    "size": trade.size,
                    "exit_price": trade_exit_price,
                    "pnl": trade_pnl,
                    "return_pct": trade_return
                })

            avg_size = sum(t.size for t in sorted_trades) / len(sorted_trades) if sorted_trades else 0.0
            avg_dollars = sum(t.size * t.price for t in sorted_trades) / len(sorted_trades) if sorted_trades else 0.0
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

            # Current price per outcome from gamma (fallback to last seen trade price)
            slug = sorted_trades[-1].market_slug or key
            price_map = price_maps.get(slug, {}) if slug else {}
            pnl = 0.0
            outcome_details = []
            for outcome, shares in holdings.items():
                current_price = price_map.get(outcome, last_price_seen.get(outcome, 0.0))
                token_id = asset_by_outcome.get(outcome, "")
                if token_id:
                    exit_side = "SELL" if shares >= 0 else "BUY"
                    cache_key = (token_id, exit_side)
                    if cache_key not in token_price_cache:
                        try:
                            token_price_cache[cache_key] = self.get_token_price(token_id, exit_side)
                        except Exception:
                            token_price_cache[cache_key] = None
                    fetched_price = token_price_cache.get(cache_key)
                    if fetched_price is not None:
                        current_price = fetched_price

                dollars_bought = outcome_buy_dollars.get(outcome, 0.0)
                dollars_sold = outcome_sell_dollars.get(outcome, 0.0)
                pnl_outcome = cash_flow.get(outcome, 0.0) + shares * current_price
                pnl += pnl_outcome

                outcome_details.append({
                    "outcome": outcome,
                    "token_id": token_id,
                    "exit_side": "SELL" if shares >= 0 else "BUY",
                    "net_shares": shares,
                    "dollars_bought": dollars_bought,
                    "dollars_sold": dollars_sold,
                    "avg_buy_price": (dollars_bought / outcome_buy_qty[outcome]) if outcome_buy_qty.get(outcome) else 0.0,
                    "avg_sell_price": (dollars_sold / outcome_sell_qty[outcome]) if outcome_sell_qty.get(outcome) else 0.0,
                    "exit_price": current_price,
                    "pnl": pnl_outcome
                })

            latest_trade = max(sorted_trades, key=lambda t: t.timestamp)
            summaries.append(WalletMarketSummary(
                market_title=latest_trade.market_title or latest_trade.market_slug or latest_trade.condition_id,
                market_slug=latest_trade.market_slug,
                event_slug=latest_trade.event_slug,
                icon=latest_trade.icon,
                total_trades=len(sorted_trades),
                total_buy_dollars=total_buy_dollars,
                trade_counts_by_outcome=counts,
                avg_trade_size=avg_size,
                avg_trade_dollars=avg_dollars,
                avg_time_between_trades_seconds=avg_delta,
                pnl=pnl,
                last_trade_time=datetime.fromtimestamp(latest_trade.timestamp),
                outcome_details=outcome_details,
                trade_details=trade_details
            ))

        profile = {
            "bio": next((t.bio for t in trades if t.bio), ""),
            "profile_image": next((t.profile_image for t in trades if t.profile_image), ""),
            "display_name": next((t.wallet_name for t in trades if t.wallet_name), "")
        }

        return {"profile": profile, "markets": summaries}

    def classify_wallets(self, trades: list[Trade]) -> dict[str, str]:
        """
        Classify wallets based on their trading behavior patterns.

        Classifications:
        - Micro-Bots/Parametric: 1.0101/2.0404 repeating sizes, automated cadence
        - Directional Whales: 100-400 size, single move, price-moving
        - Conviction Retail: 10-40 size manual bets, 1 trade
        - Light Market Makers: Multiple trades, both sides, 5-20 size
        - Noise/Casuals: 1 trade, small size, no pattern
        """
        # Aggregate wallet behavior
        wallet_stats: dict = {}

        for trade in trades:
            wallet = trade.wallet
            if wallet not in wallet_stats:
                wallet_stats[wallet] = {
                    "sizes": [],
                    "buy_count": 0,
                    "sell_count": 0
                }

            wallet_stats[wallet]["sizes"].append(trade.size)
            if trade.side == "BUY":
                wallet_stats[wallet]["buy_count"] += 1
            else:
                wallet_stats[wallet]["sell_count"] += 1

        # Classify each wallet
        wallet_classifications = {}

        for wallet, stats in wallet_stats.items():
            sizes = stats["sizes"]
            trade_count = len(sizes)
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0
            has_both_sides = stats["buy_count"] > 0 and stats["sell_count"] > 0

            # Check for parametric/bot patterns (only for small sizes typical of bots)
            is_parametric = self._check_parametric_pattern(sizes)

            # Classification logic - order matters! Check whales first
            if max_size >= 100 and trade_count <= 5:
                classification = "Directional Whales"
            elif is_parametric or (trade_count >= 3 and all(1 <= s <= 3 for s in sizes)):
                classification = "Micro-Bots"
            elif 10 <= avg_size <= 40 and trade_count == 1:
                classification = "Conviction Retail"
            elif trade_count >= 2 and has_both_sides and 5 <= avg_size <= 20:
                classification = "Light Market Makers"
            else:
                classification = "Noise/Casuals"

            wallet_classifications[wallet] = classification

        return wallet_classifications

    def _check_parametric_pattern(self, sizes: list[float]) -> bool:
        """Check if sizes contain parametric/bot patterns.

        Only considers small sizes (< 10) typical of bot activity.
        Ignores round numbers like X.0000 which are common for any size.
        """
        for size in sizes:
            # Only check small sizes typical of bots
            if size >= 10:
                continue

            size_str = f"{size:.4f}"
            if '.' in size_str:
                decimal_part = size_str.split('.')[1]
                if len(decimal_part) >= 4:
                    # Skip round numbers (0000) - not indicative of bots
                    if decimal_part == "0000":
                        continue
                    # Check for repeating patterns like 0101, 0202, 0404
                    if (decimal_part[:2] == decimal_part[2:4] or
                        decimal_part in ["0101", "0202", "0404", "1010", "2020"]):
                        return True
        return False

    def calculate_exposure(
        self,
        trades: list[Trade],
        wallet_classifications: dict[str, str]
    ) -> dict[str, ExposureData]:
        """Calculate cumulative exposure over time by classification and outcome.

        BUY increases exposure to the outcome, SELL decreases it.
        """
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Collect all unique outcomes
        outcomes = set(t.outcome for t in trades)

        exposure_data = {
            cls: ExposureData(
                timestamps=[],
                exposure_by_outcome={outcome: [] for outcome in outcomes},
                cumulative_by_outcome={outcome: 0.0 for outcome in outcomes}
            )
            for cls in self.CLASSIFICATIONS
        }

        for trade in sorted_trades:
            classification = wallet_classifications.get(trade.wallet, "Noise/Casuals")
            exp = exposure_data[classification]

            # BUY increases exposure, SELL decreases exposure
            if trade.side == "BUY":
                exp.cumulative_by_outcome[trade.outcome] += trade.size
            else:
                exp.cumulative_by_outcome[trade.outcome] -= trade.size

            exp.timestamps.append(trade.datetime)

            # Record current exposure for all outcomes at this timestamp
            for outcome in outcomes:
                exp.exposure_by_outcome[outcome].append(
                    exp.cumulative_by_outcome[outcome]
                )

        return exposure_data

    def parse_outcomes(self, trades: list[Trade]) -> dict:
        """Parse trades into lists for plotting, grouped by outcome."""
        outcomes: dict = {}

        for trade in trades:
            outcome = trade.outcome
            if outcome not in outcomes:
                outcomes[outcome] = {
                    "timestamps": [],
                    "prices": [],
                    "sizes": [],
                    "sides": [],
                    "wallets": [],
                    "wallet_names": []
                }

            outcomes[outcome]["timestamps"].append(trade.datetime)
            outcomes[outcome]["prices"].append(trade.price)
            outcomes[outcome]["sizes"].append(trade.size)
            outcomes[outcome]["sides"].append(trade.side)
            outcomes[outcome]["wallets"].append(trade.wallet)
            outcomes[outcome]["wallet_names"].append(trade.wallet_name)

        return outcomes

    def analyze_market(
        self,
        condition_id: str,
        limit: int = 100
    ) -> Optional[AnalysisResult]:
        """
        Perform complete analysis of a market.

        Returns AnalysisResult with trades, classifications, exposure data,
        and summary statistics.
        """
        trades = self.get_trades(condition_id, limit)

        if not trades:
            return None

        wallet_classifications = self.classify_wallets(trades)
        exposure_data = self.calculate_exposure(trades, wallet_classifications)
        outcomes_data = self.parse_outcomes(trades)

        # Build summary
        summary = {
            "total_trades": len(trades),
            "unique_wallets": len(wallet_classifications),
            "by_outcome": {},
            "by_classification": {}
        }

        # Summary by outcome
        for outcome, data in outcomes_data.items():
            if data["prices"]:
                summary["by_outcome"][outcome] = {
                    "trade_count": len(data["prices"]),
                    "avg_price": sum(data["prices"]) / len(data["prices"]),
                    "total_size": sum(data["sizes"])
                }

        # Summary by classification
        classification_counts: dict = {}
        for cls in wallet_classifications.values():
            classification_counts[cls] = classification_counts.get(cls, 0) + 1

        for cls in self.CLASSIFICATIONS:
            exp = exposure_data.get(cls)
            summary["by_classification"][cls] = {
                "wallet_count": classification_counts.get(cls, 0),
                "exposure_by_outcome": exp.cumulative_by_outcome if exp else {}
            }

        return AnalysisResult(
            trades=trades,
            wallet_classifications=wallet_classifications,
            exposure_by_classification=exposure_data,
            outcomes_data=outcomes_data,
            summary=summary
        )

    def generate_trade_log_csv(
        self,
        trades: list[Trade],
        wallet_classifications: dict[str, str]
    ) -> str:
        """Generate CSV content for trade log."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            "timestamp",
            "datetime",
            "wallet",
            "classification",
            "side",
            "price",
            "size",
            "outcome",
            "transaction_hash"
        ])

        # Sort and write trades
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        for trade in sorted_trades:
            classification = wallet_classifications.get(trade.wallet, "Noise/Casuals")
            writer.writerow([
                trade.timestamp,
                trade.datetime.isoformat(),
                trade.wallet,
                classification,
                trade.side,
                trade.price,
                trade.size,
                trade.outcome,
                trade.transaction_hash
            ])

        return output.getvalue()
