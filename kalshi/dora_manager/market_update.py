#!/usr/bin/env python3
"""
Market Update Script for Dora Bot.

This script analyzes market performance and generates recommended config updates:
1. Markets to SCALE DOWN (poor performance):
   - (A) P&L < -$0.50 over lookback period, OR
   - (B) No fills in lookback period
   Scale down: cut quote_size and max_inventory by 50%, double min_spread
   If quote_size is already <=5: exit (disable or set min_spread=0.50 if has position)

2. Markets to EXPAND:
   - Positive P&L in lookback period
   For expand markets: double quote_size, max_inventory_yes, max_inventory_no (capped at 25)

3. Sibling market ACTIVATION (when expanding):
   - For markets being expanded, fetch all sibling markets from the same event
   - Activate any sibling markets not already in config with default settings
   - Uses OpenAI to assess information risk for new sibling markets

Usage:
    python -m dora_bot.market_update --env prod
    python -m dora_bot.market_update --env prod --pnl-lookback 48 --volume-lookback 72
    python -m dora_bot.market_update --env demo --dry-run
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from openai import OpenAI

from db_client import DynamoDBClient

# Public API endpoint (no auth required for market data)
KALSHI_API_BASE = "https://api.elections.kalshi.com"
EVENTS_ENDPOINT = "/trade-api/v2/events"

# Default settings for newly activated sibling markets
DEFAULT_QUOTE_SIZE = 5
DEFAULT_MAX_INVENTORY_YES = 5
DEFAULT_MAX_INVENTORY_NO = 5
DEFAULT_MIN_SPREAD = 0.05

# Expansion caps
MAX_QUOTE_SIZE = 25
MAX_INVENTORY = 25

# Scale down threshold - if quote_size is at or below this, exit instead of scaling down
MIN_QUOTE_SIZE_FOR_SCALE_DOWN = 5

# Information risk threshold - markets with >25% info risk should be scaled down
MAX_INFO_RISK = 25

# Thresholds for "highly active" markets that get web search
HIGH_ACTIVITY_FILL_COUNT = 5  # fills in volume lookback period
HIGH_ACTIVITY_VOLUME_24H = 1000  # 24hr volume for potential siblings

# New market protection period - don't recommend exit for markets added within this period
NEW_MARKET_PROTECTION_HOURS = 48


@dataclass
class MarketAnalysis:
    """Analysis results for a single market."""
    market_id: str
    event_ticker: Optional[str]  # Event this market belongs to
    pnl_24h: float
    fill_count_24h: int
    fill_count_48h: int
    last_fill_time: Optional[datetime]
    median_fill_size: Optional[float]
    current_quote_size: int
    current_max_inventory_yes: int
    current_max_inventory_no: int
    current_min_spread: float
    current_enabled: bool
    has_position: bool
    position_qty: int
    created_at: Optional[datetime] = None  # When the market config was created
    # Enriched Kalshi metadata for AI model input
    event_title: Optional[str] = None  # Full event title
    market_title: Optional[str] = None  # Full market title
    volume_24h_trades: int = 0  # Total number of trades in 24hr
    volume_24h_contracts: int = 0  # Total number of contracts traded in 24hr
    buy_volume_trades: int = 0  # Buy side trade count
    buy_volume_contracts: int = 0  # Buy side contract count
    sell_volume_trades: int = 0  # Sell side trade count
    sell_volume_contracts: int = 0  # Sell side contract count
    current_spread: Optional[float] = None  # Current bid-ask spread
    spread_24h_ago: Optional[float] = None  # Spread 24hrs ago
    # Orderbook data (in cents)
    yes_bid: Optional[int] = None  # Current yes bid price
    yes_ask: Optional[int] = None  # Current yes ask price
    previous_yes_bid: Optional[int] = None  # Yes bid 24hrs ago
    previous_yes_ask: Optional[int] = None  # Yes ask 24hrs ago


@dataclass
class RecommendedAction:
    """A recommended action for a market."""
    market_id: str
    event_ticker: Optional[str]  # Event this market belongs to
    event_name: Optional[str]  # Plain text name of the event
    action: str  # 'exit', 'scale_down', 'expand', 'activate_sibling', 'reset_defaults', or 'no_action'
    reason: str
    new_enabled: Optional[bool]
    new_min_spread: Optional[float]
    new_quote_size: Optional[int]
    new_max_inventory_yes: Optional[int]
    new_max_inventory_no: Optional[int]
    # Current settings (for comparison)
    current_quote_size: Optional[int]
    current_min_spread: Optional[float]
    # Context for review
    pnl_24h: float
    fill_count_24h: int
    fill_count_48h: int
    last_fill_time: Optional[datetime]
    has_position: bool
    position_qty: int
    # Information risk assessment (for activate_sibling actions)
    info_risk_probability: Optional[float] = None
    info_risk_rationale: Optional[str] = None
    # Market creation date (for protection period tracking)
    created_at: Optional[datetime] = None
    # Enriched Kalshi metadata for AI model input
    event_title: Optional[str] = None
    market_title: Optional[str] = None
    volume_24h_trades: int = 0
    volume_24h_contracts: int = 0
    buy_volume_trades: int = 0
    buy_volume_contracts: int = 0
    sell_volume_trades: int = 0
    sell_volume_contracts: int = 0
    current_spread: Optional[float] = None
    spread_24h_ago: Optional[float] = None
    # Orderbook data (in cents)
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    previous_yes_bid: Optional[int] = None
    previous_yes_ask: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'market_id': self.market_id,
            'event_ticker': self.event_ticker,
            'event_name': self.event_name,
            'action': self.action,
            'reason': self.reason,
            'new_enabled': self.new_enabled,
            'new_min_spread': self.new_min_spread,
            'new_quote_size': self.new_quote_size,
            'new_max_inventory_yes': self.new_max_inventory_yes,
            'new_max_inventory_no': self.new_max_inventory_no,
            'current_quote_size': self.current_quote_size,
            'current_min_spread': self.current_min_spread,
            'pnl_24h': self.pnl_24h,
            'fill_count_24h': self.fill_count_24h,
            'fill_count_48h': self.fill_count_48h,
            'last_fill_time': self.last_fill_time.isoformat() if self.last_fill_time else None,
            'has_position': self.has_position,
            'position_qty': self.position_qty,
            'info_risk_probability': self.info_risk_probability,
            'info_risk_rationale': self.info_risk_rationale,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'event_title': self.event_title,
            'market_title': self.market_title,
            'volume_24h_trades': self.volume_24h_trades,
            'volume_24h_contracts': self.volume_24h_contracts,
            'buy_volume_trades': self.buy_volume_trades,
            'buy_volume_contracts': self.buy_volume_contracts,
            'sell_volume_trades': self.sell_volume_trades,
            'sell_volume_contracts': self.sell_volume_contracts,
            'current_spread': self.current_spread,
            'spread_24h_ago': self.spread_24h_ago,
            'yes_bid': self.yes_bid,
            'yes_ask': self.yes_ask,
            'previous_yes_bid': self.previous_yes_bid,
            'previous_yes_ask': self.previous_yes_ask,
        }


@dataclass
class MarketConfig:
    """Market configuration from DynamoDB."""
    market_id: str
    enabled: bool = True
    quote_size: int = DEFAULT_QUOTE_SIZE
    max_inventory_yes: int = DEFAULT_MAX_INVENTORY_YES
    max_inventory_no: int = DEFAULT_MAX_INVENTORY_NO
    min_spread: float = DEFAULT_MIN_SPREAD
    inventory_skew_factor: float = 0.5
    event_ticker: Optional[str] = None
    created_at: Optional[datetime] = None
    disabled_reason: Optional[str] = None
    disabled_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketConfig':
        """Create MarketConfig from DynamoDB dictionary."""
        # Parse datetime fields if they exist
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

        disabled_at = data.get('disabled_at')
        if disabled_at and isinstance(disabled_at, str):
            disabled_at = datetime.fromisoformat(disabled_at.replace('Z', '+00:00'))

        return cls(
            market_id=data['market_id'],
            enabled=data.get('enabled', True),
            quote_size=int(data.get('quote_size', DEFAULT_QUOTE_SIZE)),
            max_inventory_yes=int(data.get('max_inventory_yes', DEFAULT_MAX_INVENTORY_YES)),
            max_inventory_no=int(data.get('max_inventory_no', DEFAULT_MAX_INVENTORY_NO)),
            min_spread=float(data.get('min_spread', DEFAULT_MIN_SPREAD)),
            inventory_skew_factor=float(data.get('inventory_skew_factor', 0.5)),
            event_ticker=data.get('event_ticker'),
            created_at=created_at,
            disabled_reason=data.get('disabled_reason'),
            disabled_at=disabled_at
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MarketConfig to DynamoDB dictionary."""
        data = {
            'market_id': self.market_id,
            'enabled': self.enabled,
            'quote_size': self.quote_size,
            'max_inventory_yes': self.max_inventory_yes,
            'max_inventory_no': self.max_inventory_no,
            'min_spread': self.min_spread,
            'inventory_skew_factor': self.inventory_skew_factor,
        }

        if self.event_ticker:
            data['event_ticker'] = self.event_ticker
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.disabled_reason:
            data['disabled_reason'] = self.disabled_reason
        if self.disabled_at:
            data['disabled_at'] = self.disabled_at.isoformat()

        return data


def get_fills_for_date_range(
    dynamodb_resource,
    table_name: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict]:
    """Query fills from the trade log for a date range.

    Args:
        dynamodb_resource: boto3 DynamoDB resource
        table_name: Name of the trade log table
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

    Returns:
        List of fill records
    """
    table = dynamodb_resource.Table(table_name)
    fills = []

    # Generate all dates in the range
    current = start_date.date()
    end = end_date.date()

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        try:
            # Query all items for this date
            response = table.query(
                KeyConditionExpression=Key('date').eq(date_str)
            )

            for item in response.get('Items', []):
                fills.append(_serialize_decimal(item))

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    fills.append(_serialize_decimal(item))

        except ClientError as e:
            print(f"Warning: Error querying fills for {date_str}: {e}")

        current += timedelta(days=1)

    return fills


def _serialize_decimal(obj):
    """Convert Decimal to float for processing."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialize_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_decimal(item) for item in obj]
    return obj


def assess_information_risk(
    market_title: str,
    current_price: float,
    market_subtitle: Optional[str] = None,
    rules: Optional[str] = None,
    use_web_search: bool = False,
) -> Dict[str, Any]:
    """Assess the likelihood of market-moving information being released in the next 7 days.

    Uses OpenAI API to evaluate information risk for a prediction market.

    Args:
        market_title: The title of the market
        current_price: Current market price (0-100)
        market_subtitle: Optional subtitle providing additional context
        rules: Optional resolution rules for the market
        use_web_search: If True, enable web search tool for highly active markets

    Returns:
        Dictionary containing:
        - probability: Likelihood percentage (0-100)
        - rationale: 2-3 sentence explanation
        - error: Error message if API call fails
    """
    try:
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "probability": None,
                "rationale": "OpenAI API key not configured",
                "error": "Missing OPENAI_API_KEY in environment variables",
            }

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Build context for the prompt
        context = f"The market is: {market_title}"
        if market_subtitle:
            context += f"\n{market_subtitle}"
        if rules:
            context += f"\n\nResolution rules: {rules[:500]}"  # Limit rules length
        context += f"\n\nThe current price is ~{current_price:.0f}%"

        # Create the prompt
        prompt = f"""You are a market risk assessment expert for prediction markets on Kalshi. Your job is to evaluate the likelihood that market moving information will be released in the next 7 days that would move the current pricing more than 20ppts in either direction (e.g. 5% --> 25% or 45% --> 25%).

If the outcome of the market will be decided within the next 7 days, please return a 100% chance of market moving news.  Today is {datetime.now().strftime("%Y-%m-%d")}

Please return your assessment in the form of a likelihood percentage (number from 0-100%) and 2-3 sentence rationale.

{context}

Your response should be only a JSON dictionary e.g. {{"probability": "XX%", "rationale": "XXXX"}}"""

        # Build API call kwargs
        api_kwargs = {
            "model": "gpt-5-mini",
            "reasoning": {"effort": "medium"},
            "input": prompt,
        }

        # Add web search tool for highly active markets
        if use_web_search:
            api_kwargs["tools"] = [{"type": "web_search"}]

        response = client.responses.create(**api_kwargs)

        # Parse the response (Responses API uses output_text)
        response_text = getattr(response, "output_text", None)
        if response_text is None:
            return {
                "probability": None,
                "rationale": "No output_text in response",
                "error": "Empty API response",
            }
        response_text = response_text.strip()

        # Try to extract JSON if wrapped in markdown code blocks
        if response_text.startswith("```"):
            # Remove markdown code blocks
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        return {
            "probability": result.get("probability", "N/A"),
            "rationale": result.get("rationale", "No rationale provided"),
            "error": None,
        }

    except json.JSONDecodeError as e:
        return {
            "probability": None,
            "rationale": f"Failed to parse API response: {str(e)}",
            "error": "JSON parsing error",
        }
    except Exception as e:
        return {
            "probability": None,
            "rationale": f"Error calling OpenAI API: {str(e)}",
            "error": str(e),
        }


def fetch_event_markets(event_ticker: str) -> List[Dict[str, Any]]:
    """Fetch all markets for an event from the Kalshi API.

    Args:
        event_ticker: Event ticker to fetch markets for

    Returns:
        List of market dictionaries from the API
    """
    try:
        url = f"{KALSHI_API_BASE}/trade-api/v2/events/{event_ticker}"
        print(f"    Fetching: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # The API returns 'markets' at the top level, alongside 'event'
        markets = data.get('markets', [])
        print(f"    Response: found {len(markets)} markets in event {event_ticker}")
        return markets
    except requests.exceptions.HTTPError as e:
        print(f"    HTTP Error fetching {event_ticker}: {e.response.status_code} - {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"    Error fetching {event_ticker}: {type(e).__name__}: {e}")
        return []


def fetch_event_tickers(market_ids: List[str]) -> Dict[str, Optional[str]]:
    """Fetch event_ticker for each market from the public Kalshi API.

    Args:
        market_ids: List of market IDs to fetch

    Returns:
        Dictionary mapping market_id to event_ticker (or None if not found)
    """
    event_tickers = {}

    for market_id in market_ids:
        try:
            url = f"{KALSHI_API_BASE}/trade-api/v2/markets/{market_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # The API returns market info with 'market' key containing the data
            market_data = data.get('market', data)
            event_ticker = market_data.get('event_ticker')
            event_tickers[market_id] = event_ticker
        except Exception as e:
            print(f"Warning: Could not fetch event_ticker for {market_id}: {e}")
            event_tickers[market_id] = None

    return event_tickers


def fetch_market_details(market_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch market details (title, subtitle, rules, price) for each market from the public Kalshi API.

    Args:
        market_ids: List of market IDs to fetch

    Returns:
        Dictionary mapping market_id to market details dict
    """
    market_details = {}

    for market_id in market_ids:
        try:
            url = f"{KALSHI_API_BASE}/trade-api/v2/markets/{market_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            market_data = data.get('market', data)

            # Get mid-price for current price estimate
            yes_bid = market_data.get('yes_bid', 0) or 0
            yes_ask = market_data.get('yes_ask', 100) or 100
            current_price = (yes_bid + yes_ask) / 2

            # Get previous bid/ask for historical spread calculation
            previous_yes_bid = market_data.get('previous_yes_bid')
            previous_yes_ask = market_data.get('previous_yes_ask')

            market_details[market_id] = {
                'title': market_data.get('title', ''),
                'subtitle': market_data.get('subtitle', ''),
                'rules_primary': market_data.get('rules_primary', ''),
                'current_price': current_price,
                'event_ticker': market_data.get('event_ticker'),
                'yes_bid': yes_bid,
                'yes_ask': yes_ask,
                'previous_yes_bid': previous_yes_bid,
                'previous_yes_ask': previous_yes_ask,
            }
        except Exception as e:
            print(f"Warning: Could not fetch details for {market_id}: {e}")
            market_details[market_id] = None

    return market_details


def fetch_event_names(event_tickers: List[str]) -> Dict[str, str]:
    """Fetch event names (titles) for a list of event tickers from the Kalshi API.

    Args:
        event_tickers: List of event ticker strings

    Returns:
        Dictionary mapping event_ticker to event title (name)
    """
    event_names = {}

    for event_ticker in event_tickers:
        if not event_ticker:
            continue
        try:
            url = f"{KALSHI_API_BASE}{EVENTS_ENDPOINT}/{event_ticker}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            event_data = data.get('event', {})
            event_names[event_ticker] = event_data.get('title', '')
        except Exception as e:
            print(f"  Warning: Could not fetch event name for {event_ticker}: {e}")
            event_names[event_ticker] = ''

    return event_names


def fetch_trade_history(
    market_id: str,
    limit: int = 100,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fetch trade history for a market from Kalshi API with exponential backoff.

    Args:
        market_id: Market ticker
        limit: Maximum number of trades to fetch
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        List of trade dictionaries
    """
    import time
    import random

    url = f"{KALSHI_API_BASE}/trade-api/v2/markets/trades"
    params = {"ticker": market_id, "limit": limit}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    print(f"  Warning: Rate limited for {market_id} after {max_retries} retries")
                    return []

            response.raise_for_status()
            data = response.json()
            return data.get("trades", [])

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            print(f"  Warning: Could not fetch trades for {market_id}: {e}")
            return []
        except Exception as e:
            print(f"  Warning: Could not fetch trades for {market_id}: {e}")
            return []

    return []


def calculate_side_volumes(trades: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """Calculate total volume on buy and sell sides from trade history.

    Args:
        trades: List of trade dictionaries

    Returns:
        Tuple of (buy_volume_trades, buy_volume_contracts, sell_volume_trades, sell_volume_contracts)
    """
    buy_volume_trades = 0
    buy_volume_contracts = 0
    sell_volume_trades = 0
    sell_volume_contracts = 0

    for trade in trades:
        # Kalshi trades have 'taker_side' which is 'yes' or 'no'
        # 'yes' means the taker bought YES, 'no' means the taker sold YES (bought NO)
        count = trade.get("count", 0) or 0
        taker_side = trade.get("taker_side", "")

        if taker_side == "yes":
            buy_volume_trades += 1
            buy_volume_contracts += count
        elif taker_side == "no":
            sell_volume_trades += 1
            sell_volume_contracts += count

    return buy_volume_trades, buy_volume_contracts, sell_volume_trades, sell_volume_contracts


def filter_trades_by_time(trades: List[Dict[str, Any]], hours: int = 24) -> List[Dict[str, Any]]:
    """Filter trades to only include those from the last N hours.

    Args:
        trades: List of trade dictionaries
        hours: Number of hours to look back

    Returns:
        Filtered list of trades
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    filtered_trades = []

    for trade in trades:
        trade_time_str = trade.get("created_time")
        if trade_time_str:
            try:
                # Handle both ISO format with and without 'Z'
                if trade_time_str.endswith('Z'):
                    trade_time_str = trade_time_str[:-1] + '+00:00'
                trade_time = datetime.fromisoformat(trade_time_str)
                if trade_time.tzinfo is None:
                    trade_time = trade_time.replace(tzinfo=timezone.utc)

                if trade_time >= cutoff:
                    filtered_trades.append(trade)
            except (ValueError, TypeError):
                # If we can't parse the timestamp, skip it
                continue

    return filtered_trades


def analyze_markets(
    db_client: DynamoDBClient = None,
    dynamo: DynamoDBClient = None,
    environment: str = None,
    pnl_lookback_hours: int = 24,
    volume_lookback_hours: int = 48,
    enabled_only: bool = False
) -> Dict[str, MarketAnalysis]:
    """Analyze all markets based on fills and positions.

    Args:
        db_client: DynamoDB client (Lambda parameter name)
        dynamo: DynamoDB client (CLI parameter name, for backwards compatibility)
        environment: 'demo' or 'prod' (auto-detected from db_client if not provided)
        pnl_lookback_hours: Hours to look back for P&L calculation (default 24)
        volume_lookback_hours: Hours to look back for fill count (default 48)
        enabled_only: If True, only analyze enabled markets (default False)

    Returns:
        Dictionary mapping market_id to MarketAnalysis
    """
    # Support both parameter names for flexibility
    if db_client is None and dynamo is None:
        raise ValueError("Either db_client or dynamo parameter must be provided")

    dynamo = db_client or dynamo

    # Auto-detect environment from db_client if not provided
    if environment is None:
        environment = dynamo.environment
    now = datetime.now(timezone.utc)
    cutoff_pnl = now - timedelta(hours=pnl_lookback_hours)
    cutoff_volume = now - timedelta(hours=volume_lookback_hours)

    # Get market configs (filtered by enabled_only parameter)
    configs_list = dynamo.get_all_market_configs(enabled_only=enabled_only)

    # Convert list to dict of MarketConfig objects
    configs: Dict[str, MarketConfig] = {}
    for config_dict in configs_list:
        config = MarketConfig.from_dict(config_dict)
        configs[config.market_id] = config

    # Fetch event_tickers for all markets from public API
    print(f"Fetching event info for {len(configs)} markets...")
    event_tickers = fetch_event_tickers(list(configs.keys()))

    # Fetch enriched metadata for AI model input
    print(f"Fetching market details for {len(configs)} markets...")
    market_details = fetch_market_details(list(configs.keys()))

    # Get unique event tickers and fetch their names
    unique_event_tickers = list(set(ticker for ticker in event_tickers.values() if ticker))
    print(f"Fetching event names for {len(unique_event_tickers)} events...")
    event_names = fetch_event_names(unique_event_tickers)

    # Get current positions
    positions = dynamo.get_positions()

    # Get fills for the lookback period (use the longer of the two windows)
    max_lookback = max(pnl_lookback_hours, volume_lookback_hours)
    cutoff_max = now - timedelta(hours=max_lookback)

    suffix = '_demo' if environment == 'demo' else '_prod'
    table_name = f"dora_trade_log{suffix}"

    fills = get_fills_for_date_range(
        dynamo.dynamodb,
        table_name,
        cutoff_max,
        now
    )

    # Group fills by market
    fills_by_market: Dict[str, List[Dict]] = defaultdict(list)
    for fill in fills:
        market_id = fill.get('market_id')
        if market_id:
            fills_by_market[market_id].append(fill)

    # Analyze each market
    analyses = {}
    for market_id, config in configs.items():
        market_fills = fills_by_market.get(market_id, [])

        # Parse fill timestamps and filter by time window
        fills_pnl = []  # For P&L lookback
        fills_volume = []  # For volume/fill count lookback
        last_fill_time = None

        for fill in market_fills:
            # Get timestamp from the fill
            ts_str = fill.get('fill_timestamp') or fill.get('timestamp')
            if not ts_str:
                # Try to extract from the sort key
                sort_key = fill.get('timestamp#order_id', '')
                if '#' in sort_key:
                    ts_str = sort_key.split('#')[0]

            if ts_str:
                try:
                    # Handle both ISO format with and without 'Z'
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1] + '+00:00'
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    # Track last fill time
                    if last_fill_time is None or ts > last_fill_time:
                        last_fill_time = ts

                    # Categorize by time window
                    if ts >= cutoff_volume:
                        fills_volume.append(fill)
                    if ts >= cutoff_pnl:
                        fills_pnl.append(fill)
                except (ValueError, TypeError):
                    # If we can't parse the timestamp, include in both windows
                    fills_volume.append(fill)
                    fills_pnl.append(fill)

        # Calculate P&L for the P&L lookback period
        pnl_lookback = 0.0
        fill_sizes = []
        for fill in fills_pnl:
            # pnl_realized already includes fees (subtracted in Position.update_from_fill)
            pnl = fill.get('pnl_realized', 0.0) or 0.0
            pnl_lookback += pnl

            # Track fill sizes for median calculation
            size = fill.get('size') or fill.get('fill_size', 0)
            if size:
                fill_sizes.append(size)

        # Calculate median fill size
        median_fill_size = statistics.median(fill_sizes) if fill_sizes else None

        # Get position info
        position = positions.get(market_id)
        has_position = position is not None and position.get('net_yes_qty', 0) != 0
        position_qty = position.get('net_yes_qty', 0) if position else 0

        # Fetch enriched Kalshi metadata
        market_detail = market_details.get(market_id) or {}
        # Use event_ticker from market_detail (already fetched), fallback to event_tickers dict
        event_ticker = market_detail.get('event_ticker') or event_tickers.get(market_id)
        event_title = event_names.get(event_ticker, '') if event_ticker else None
        market_title = market_detail.get('title')

        # Calculate current spread from orderbook
        yes_bid = market_detail.get('yes_bid', 0) or 0
        yes_ask = market_detail.get('yes_ask', 100) or 100
        current_spread = (yes_ask - yes_bid) / 100 if yes_ask and yes_bid else None

        # Calculate spread from 24 hours ago
        previous_yes_bid = market_detail.get('previous_yes_bid')
        previous_yes_ask = market_detail.get('previous_yes_ask')
        if previous_yes_bid is not None and previous_yes_ask is not None:
            spread_24h_ago = (previous_yes_ask - previous_yes_bid) / 100
        else:
            spread_24h_ago = None

        # Fetch trade history for volume calculations (24h window)
        trades = fetch_trade_history(market_id, limit=200)
        trades_24h = filter_trades_by_time(trades, hours=24)

        # Calculate volume statistics
        volume_24h_trades = len(trades_24h)
        volume_24h_contracts = sum(trade.get('count', 0) or 0 for trade in trades_24h)
        buy_volume_trades, buy_volume_contracts, sell_volume_trades, sell_volume_contracts = calculate_side_volumes(trades_24h)

        analyses[market_id] = MarketAnalysis(
            market_id=market_id,
            event_ticker=event_ticker,
            pnl_24h=pnl_lookback,  # Named pnl_24h for backward compat but uses configured lookback
            fill_count_24h=len(fills_pnl),  # Fills in P&L lookback period
            fill_count_48h=len(fills_volume),  # Fills in volume lookback period
            last_fill_time=last_fill_time,
            median_fill_size=median_fill_size,
            current_quote_size=config.quote_size,
            current_max_inventory_yes=config.max_inventory_yes,
            current_max_inventory_no=config.max_inventory_no,
            current_min_spread=config.min_spread,
            current_enabled=config.enabled,
            has_position=has_position,
            position_qty=position_qty,
            created_at=config.created_at,
            # Enriched Kalshi metadata
            event_title=event_title,
            market_title=market_title,
            volume_24h_trades=volume_24h_trades,
            volume_24h_contracts=volume_24h_contracts,
            buy_volume_trades=buy_volume_trades,
            buy_volume_contracts=buy_volume_contracts,
            sell_volume_trades=sell_volume_trades,
            sell_volume_contracts=sell_volume_contracts,
            current_spread=current_spread,
            spread_24h_ago=spread_24h_ago,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            previous_yes_bid=previous_yes_bid,
            previous_yes_ask=previous_yes_ask,
        )

    return analyses


def assess_info_risk_for_markets(
    market_ids: List[str],
    analyses: Dict[str, MarketAnalysis],
) -> Dict[str, Dict[str, Any]]:
    """Assess information risk for a list of active markets.

    Args:
        market_ids: List of market IDs to assess
        analyses: Dictionary of market analyses (for fill counts)

    Returns:
        Dictionary mapping market_id to info risk assessment result
    """
    if not market_ids:
        return {}

    print(f"\nFetching market details for {len(market_ids)} active markets...")
    market_details = fetch_market_details(market_ids)

    print(f"Assessing information risk for {len(market_ids)} active markets...")
    results = {}

    # Increase workers for faster parallel execution (10 concurrent requests)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for market_id in market_ids:
            details = market_details.get(market_id)
            if not details:
                continue

            # Check if this is a highly active market (>50 fills in volume lookback)
            analysis = analyses.get(market_id)
            is_highly_active = analysis and analysis.fill_count_48h >= HIGH_ACTIVITY_FILL_COUNT

            future = executor.submit(
                assess_information_risk,
                details.get('title', ''),
                details.get('current_price', 50),
                details.get('subtitle', ''),
                details.get('rules_primary', ''),
                is_highly_active,  # use_web_search
            )
            futures[future] = market_id

        for future in as_completed(futures):
            market_id = futures[future]
            ir_result = future.result()

            # Parse probability from result
            prob_str = ir_result.get("probability", "N/A")
            info_risk = None
            if prob_str and prob_str != "N/A":
                try:
                    info_risk = float(str(prob_str).replace("%", "").strip())
                except (ValueError, AttributeError):
                    pass

            analysis = analyses.get(market_id)
            is_highly_active = analysis and analysis.fill_count_48h >= HIGH_ACTIVITY_FILL_COUNT
            web_search_str = " (web_search)" if is_highly_active else ""
            print(f"  {market_id}: info_risk={info_risk}%{web_search_str}")

            results[market_id] = {
                'info_risk_probability': info_risk,
                'info_risk_rationale': ir_result.get('rationale'),
                'error': ir_result.get('error'),
            }

    return results


def generate_recommendations(
    analyses: Dict[str, MarketAnalysis],
    existing_configs: Dict[str, MarketConfig] = None,
    event_names: Dict[str, str] = None,
    skip_info_risk: bool = False,
) -> List[RecommendedAction]:
    """Generate recommended actions based on market analysis.

    Args:
        analyses: Dictionary of market analyses
        existing_configs: Dictionary of existing market configs (optional, will be derived from analyses)
        event_names: Dictionary mapping event_ticker to event name/title (optional, will be fetched if needed)
        skip_info_risk: Skip OpenAI information risk assessment (default False)

    Returns:
        List of recommended actions
    """
    # Build existing_configs from analyses if not provided
    if existing_configs is None:
        existing_configs = {}
        for market_id, analysis in analyses.items():
            # Create a MarketConfig from the analysis current_ fields
            config = MarketConfig(
                market_id=market_id,
                enabled=analysis.current_enabled,
                quote_size=analysis.current_quote_size,
                max_inventory_yes=analysis.current_max_inventory_yes,
                max_inventory_no=analysis.current_max_inventory_no,
                min_spread=analysis.current_min_spread,
                event_ticker=analysis.event_ticker,
                created_at=analysis.created_at,
            )
            existing_configs[market_id] = config

    # Build event_names if not provided
    if event_names is None:
        event_names = {}
        # Extract unique event tickers from analyses
        event_tickers = set()
        for analysis in analyses.values():
            if analysis.event_ticker:
                event_tickers.add(analysis.event_ticker)
        # Fetch event names if we have tickers
        if event_tickers:
            event_names = fetch_event_names(list(event_tickers))
    recommendations = []

    # Build maps for event-based analysis
    all_markets_by_event: Dict[str, List[str]] = defaultdict(list)
    poor_performance_candidates: Dict[str, str] = {}  # market_id -> reason

    # Track events that have expand candidates (for sibling activation)
    expand_events: set = set()

    # Collect all active market IDs for info risk assessment
    active_market_ids = []
    for market_id, analysis in analyses.items():
        if analysis.current_enabled:
            active_market_ids.append(market_id)

    # Assess information risk for top 25 active markets (unless skipped)
    # Ranked by: 1) abs(net_position), 2) trade count in volume window
    info_risk_results = {}
    if not skip_info_risk and active_market_ids:
        # Sort by priority: abs(position) DESC, then fill_count DESC
        active_with_priority = []
        for market_id in active_market_ids:
            analysis = analyses.get(market_id)
            if analysis:
                priority = (
                    abs(analysis.position_qty),  # Primary: absolute position
                    analysis.fill_count_48h,     # Secondary: trade count
                )
                active_with_priority.append((market_id, priority))

        # Sort descending and take top 25
        active_with_priority.sort(key=lambda x: x[1], reverse=True)
        top_25_market_ids = [m[0] for m in active_with_priority[:25]]

        print(f"Assessing info risk for top {len(top_25_market_ids)} active markets (out of {len(active_market_ids)} total)")
        info_risk_results = assess_info_risk_for_markets(top_25_market_ids, analyses)

    # First pass: identify poor performance candidates and build event map
    for market_id, analysis in analyses.items():
        # Skip disabled markets without positions (already exited)
        if not analysis.current_enabled and not analysis.has_position:
            continue

        # Track all enabled markets by event
        if analysis.current_enabled and analysis.event_ticker:
            all_markets_by_event[analysis.event_ticker].append(market_id)

        # Check POOR PERFORMANCE conditions (scale down or exit)
        poor_reason = None

        # (A) P&L < $0 over lookback period
        if analysis.pnl_24h < 0:
            poor_reason = f"P&L < $0 (${analysis.pnl_24h:.2f})"

        # (B) No fills in lookback period (only for enabled markets)
        elif analysis.current_enabled and analysis.fill_count_48h == 0:
            poor_reason = "No fills in lookback period"

        # (C) High information risk (>25%) for enabled markets
        elif analysis.current_enabled:
            ir_result = info_risk_results.get(market_id, {})
            ir_prob = ir_result.get('info_risk_probability')
            if ir_prob is not None and ir_prob > MAX_INFO_RISK:
                poor_reason = f"High info risk ({ir_prob:.0f}% > {MAX_INFO_RISK}%)"

        if poor_reason:
            poor_performance_candidates[market_id] = poor_reason

        # Check EXPAND conditions (for tracking events with expanding markets)
        # Only allow expansion if info risk is <= MAX_INFO_RISK
        ir_result = info_risk_results.get(market_id, {})
        ir_prob = ir_result.get('info_risk_probability')
        info_risk_ok = ir_prob is None or ir_prob <= MAX_INFO_RISK

        if (analysis.current_enabled and
            analysis.pnl_24h > 0 and
            analysis.event_ticker and
            info_risk_ok):
            expand_events.add(analysis.event_ticker)

    # Second pass: generate recommendations
    for market_id, analysis in analyses.items():
        # Skip disabled markets without positions (already exited)
        if not analysis.current_enabled and not analysis.has_position:
            continue

        # Get info risk data for this market
        ir_result = info_risk_results.get(market_id, {})
        ir_prob = ir_result.get('info_risk_probability')
        ir_rationale = ir_result.get('info_risk_rationale')

        poor_reason = poor_performance_candidates.get(market_id)
        action_taken = False  # Track if we added a recommendation

        if poor_reason:
            event_ticker = analysis.event_ticker

            # Check if market is within protection period (recently created)
            is_protected = False
            if analysis.created_at:
                now = datetime.now(timezone.utc)
                market_age = now - analysis.created_at
                is_protected = market_age < timedelta(hours=NEW_MARKET_PROTECTION_HOURS)

            # Check if a sibling market in the same event is expanding
            has_expanding_sibling = event_ticker and event_ticker in expand_events

            # Check if we can scale down or need to exit
            # If quote_size > MIN_QUOTE_SIZE_FOR_SCALE_DOWN, scale down instead of exit
            # But if a sibling is expanding, reset to defaults instead
            if has_expanding_sibling:
                # Sibling is expanding: reset to defaults instead of scaling down/exiting
                # If already at defaults, add no_action
                if (analysis.current_quote_size == DEFAULT_QUOTE_SIZE and
                    analysis.current_max_inventory_yes == DEFAULT_MAX_INVENTORY_YES and
                    analysis.current_max_inventory_no == DEFAULT_MAX_INVENTORY_NO and
                    analysis.current_min_spread == DEFAULT_MIN_SPREAD):
                    recommendations.append(RecommendedAction(
                        market_id=market_id,
                        event_ticker=event_ticker,
                        event_name=event_names.get(event_ticker, ''),
                        action='no_action',
                        reason=f"{poor_reason} - sibling expanding, already at defaults",
                        new_enabled=None,
                        new_min_spread=None,
                        new_quote_size=None,
                        new_max_inventory_yes=None,
                        new_max_inventory_no=None,
                        current_quote_size=analysis.current_quote_size,
                        current_min_spread=analysis.current_min_spread,
                        pnl_24h=analysis.pnl_24h,
                        fill_count_24h=analysis.fill_count_24h,
                        fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                        has_position=analysis.has_position,
                        position_qty=analysis.position_qty,
                        info_risk_probability=ir_prob,
                        info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                    ))
                else:
                    recommendations.append(RecommendedAction(
                        market_id=market_id,
                        event_ticker=event_ticker,
                        event_name=event_names.get(event_ticker, ''),
                        action='reset_defaults',
                        reason=f"{poor_reason} (sibling expanding in {event_ticker}, resetting to defaults)",
                        new_enabled=True,
                        new_min_spread=DEFAULT_MIN_SPREAD,
                        new_quote_size=DEFAULT_QUOTE_SIZE,
                        new_max_inventory_yes=DEFAULT_MAX_INVENTORY_YES,
                        new_max_inventory_no=DEFAULT_MAX_INVENTORY_NO,
                        current_quote_size=analysis.current_quote_size,
                        current_min_spread=analysis.current_min_spread,
                        pnl_24h=analysis.pnl_24h,
                        fill_count_24h=analysis.fill_count_24h,
                        fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                        has_position=analysis.has_position,
                        position_qty=analysis.position_qty,
                        info_risk_probability=ir_prob,
                        info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                    ))
                action_taken = True
            elif analysis.current_quote_size > MIN_QUOTE_SIZE_FOR_SCALE_DOWN:
                # Scale down: cut quote_size and max_inventory by 50%, double min_spread
                new_quote_size = max(1, analysis.current_quote_size // 2)
                new_max_inv_yes = max(1, analysis.current_max_inventory_yes // 2)
                new_max_inv_no = max(1, analysis.current_max_inventory_no // 2)
                new_min_spread = min(0.50, analysis.current_min_spread * 2)

                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    event_name=event_names.get(event_ticker, ''),
                    action='scale_down',
                    reason=f"{poor_reason} - scaling down (quote {analysis.current_quote_size}->{new_quote_size})",
                    new_enabled=None,
                    new_min_spread=new_min_spread,
                    new_quote_size=new_quote_size,
                    new_max_inventory_yes=new_max_inv_yes,
                    new_max_inventory_no=new_max_inv_no,
                    current_quote_size=analysis.current_quote_size,
                    current_min_spread=analysis.current_min_spread,
                    pnl_24h=analysis.pnl_24h,
                    fill_count_24h=analysis.fill_count_24h,
                    fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                    has_position=analysis.has_position,
                    position_qty=analysis.position_qty,
                    info_risk_probability=ir_prob,
                    info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                ))
                action_taken = True
            elif is_protected:
                # Protected market at minimum: reset to defaults instead of exiting
                # If already at defaults, add no_action
                if (analysis.current_quote_size == DEFAULT_QUOTE_SIZE and
                    analysis.current_max_inventory_yes == DEFAULT_MAX_INVENTORY_YES and
                    analysis.current_max_inventory_no == DEFAULT_MAX_INVENTORY_NO and
                    analysis.current_min_spread == DEFAULT_MIN_SPREAD):
                    recommendations.append(RecommendedAction(
                        market_id=market_id,
                        event_ticker=event_ticker,
                        event_name=event_names.get(event_ticker, ''),
                        action='no_action',
                        reason=f"{poor_reason} - within {NEW_MARKET_PROTECTION_HOURS}h protection, already at defaults",
                        new_enabled=None,
                        new_min_spread=None,
                        new_quote_size=None,
                        new_max_inventory_yes=None,
                        new_max_inventory_no=None,
                        current_quote_size=analysis.current_quote_size,
                        current_min_spread=analysis.current_min_spread,
                        pnl_24h=analysis.pnl_24h,
                        fill_count_24h=analysis.fill_count_24h,
                        fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                        has_position=analysis.has_position,
                        position_qty=analysis.position_qty,
                        info_risk_probability=ir_prob,
                        info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                    ))
                else:
                    recommendations.append(RecommendedAction(
                        market_id=market_id,
                        event_ticker=event_ticker,
                        event_name=event_names.get(event_ticker, ''),
                        action='reset_defaults',
                        reason=f"{poor_reason} (within {NEW_MARKET_PROTECTION_HOURS}h protection, resetting to defaults)",
                        new_enabled=True,
                        new_min_spread=DEFAULT_MIN_SPREAD,
                        new_quote_size=DEFAULT_QUOTE_SIZE,
                        new_max_inventory_yes=DEFAULT_MAX_INVENTORY_YES,
                        new_max_inventory_no=DEFAULT_MAX_INVENTORY_NO,
                        current_quote_size=analysis.current_quote_size,
                        current_min_spread=analysis.current_min_spread,
                        pnl_24h=analysis.pnl_24h,
                        fill_count_24h=analysis.fill_count_24h,
                        fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                        has_position=analysis.has_position,
                        position_qty=analysis.position_qty,
                        info_risk_probability=ir_prob,
                        info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                    ))
                action_taken = True
            elif analysis.has_position:
                # Quote size at minimum and has position: set min_spread to 0.50 to exit
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    event_name=event_names.get(event_ticker, ''),
                    action='exit',
                    reason=f"{poor_reason} (quote_size at minimum, has position)",
                    new_enabled=True,  # Keep enabled to allow position exit
                    new_min_spread=0.50,
                    new_quote_size=None,
                    new_max_inventory_yes=None,
                    new_max_inventory_no=None,
                    current_quote_size=analysis.current_quote_size,
                    current_min_spread=analysis.current_min_spread,
                    pnl_24h=analysis.pnl_24h,
                    fill_count_24h=analysis.fill_count_24h,
                    fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                    has_position=analysis.has_position,
                    position_qty=analysis.position_qty,
                    info_risk_probability=ir_prob,
                    info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                ))
                action_taken = True
            else:
                # Quote size at minimum and no position: disable the market
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    event_name=event_names.get(event_ticker, ''),
                    action='exit',
                    reason=f"{poor_reason} (quote_size at minimum)",
                    new_enabled=False,
                    new_min_spread=None,
                    new_quote_size=None,
                    new_max_inventory_yes=None,
                    new_max_inventory_no=None,
                    current_quote_size=analysis.current_quote_size,
                    current_min_spread=analysis.current_min_spread,
                    pnl_24h=analysis.pnl_24h,
                    fill_count_24h=analysis.fill_count_24h,
                    fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                    has_position=analysis.has_position,
                    position_qty=analysis.position_qty,
                    info_risk_probability=ir_prob,
                    info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                ))
                action_taken = True

        # Check EXPAND conditions (only for enabled markets with fills and acceptable info risk)
        if not action_taken:
            info_risk_ok = ir_prob is None or ir_prob <= MAX_INFO_RISK
            if (analysis.current_enabled and
                analysis.pnl_24h > 0 and
                info_risk_ok):

                # Cap expansion at MAX_QUOTE_SIZE and MAX_INVENTORY
                new_quote_size = min(MAX_QUOTE_SIZE, analysis.current_quote_size * 2)
                new_max_inv_yes = min(MAX_INVENTORY, analysis.current_max_inventory_yes * 2)
                new_max_inv_no = min(MAX_INVENTORY, analysis.current_max_inventory_no * 2)

                # Reduce min_spread by $0.01, but not below 0.04
                new_min_spread = max(0.04, analysis.current_min_spread - 0.01)

                # If already at max for size/inventory and min_spread can't be reduced, no_action
                if (new_quote_size == analysis.current_quote_size and
                    new_max_inv_yes == analysis.current_max_inventory_yes and
                    new_max_inv_no == analysis.current_max_inventory_no and
                    new_min_spread == analysis.current_min_spread):
                    recommendations.append(RecommendedAction(
                        market_id=market_id,
                        event_ticker=analysis.event_ticker,
                        event_name=event_names.get(analysis.event_ticker, '') if analysis.event_ticker else '',
                        action='no_action',
                        reason=f"Positive P&L (${analysis.pnl_24h:.2f}) but already at max settings",
                        new_enabled=None,
                        new_min_spread=None,
                        new_quote_size=None,
                        new_max_inventory_yes=None,
                        new_max_inventory_no=None,
                        current_quote_size=analysis.current_quote_size,
                        current_min_spread=analysis.current_min_spread,
                        pnl_24h=analysis.pnl_24h,
                        fill_count_24h=analysis.fill_count_24h,
                        fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                        has_position=analysis.has_position,
                        position_qty=analysis.position_qty,
                        info_risk_probability=ir_prob,
                        info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                    ))
                else:
                    recommendations.append(RecommendedAction(
                        market_id=market_id,
                        event_ticker=analysis.event_ticker,
                        event_name=event_names.get(analysis.event_ticker, '') if analysis.event_ticker else '',
                        action='expand',
                        reason=f"Positive P&L (${analysis.pnl_24h:.2f})",
                        new_enabled=None,
                        new_min_spread=new_min_spread,
                        new_quote_size=new_quote_size,
                        new_max_inventory_yes=new_max_inv_yes,
                        new_max_inventory_no=new_max_inv_no,
                        current_quote_size=analysis.current_quote_size,
                        current_min_spread=analysis.current_min_spread,
                        pnl_24h=analysis.pnl_24h,
                        fill_count_24h=analysis.fill_count_24h,
                        fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                        has_position=analysis.has_position,
                        position_qty=analysis.position_qty,
                        info_risk_probability=ir_prob,
                        info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
                    ))
                action_taken = True

        # If no action was taken, add a no_action recommendation
        if not action_taken:
            # Determine the reason for no action
            if analysis.pnl_24h == 0:
                no_action_reason = "Neutral P&L ($0.00)"
            elif analysis.pnl_24h > 0:
                no_action_reason = f"Positive P&L (${analysis.pnl_24h:.2f}) but already at max settings"
            else:
                no_action_reason = "No action criteria met"

            recommendations.append(RecommendedAction(
                market_id=market_id,
                event_ticker=analysis.event_ticker,
                event_name=event_names.get(analysis.event_ticker, '') if analysis.event_ticker else '',
                action='no_action',
                reason=no_action_reason,
                new_enabled=None,
                new_min_spread=None,
                new_quote_size=None,
                new_max_inventory_yes=None,
                new_max_inventory_no=None,
                current_quote_size=analysis.current_quote_size,
                current_min_spread=analysis.current_min_spread,
                pnl_24h=analysis.pnl_24h,
                fill_count_24h=analysis.fill_count_24h,
                fill_count_48h=analysis.fill_count_48h,
                        last_fill_time=analysis.last_fill_time,
                has_position=analysis.has_position,
                position_qty=analysis.position_qty,
                info_risk_probability=ir_prob,
                info_risk_rationale=ir_rationale,
                        created_at=analysis.created_at,
                        # Enriched Kalshi metadata
                        event_title=analysis.event_title,
                        market_title=analysis.market_title,
                        volume_24h_trades=analysis.volume_24h_trades,
                        volume_24h_contracts=analysis.volume_24h_contracts,
                        buy_volume_trades=analysis.buy_volume_trades,
                        buy_volume_contracts=analysis.buy_volume_contracts,
                        sell_volume_trades=analysis.sell_volume_trades,
                        sell_volume_contracts=analysis.sell_volume_contracts,
                        current_spread=analysis.current_spread,
                        spread_24h_ago=analysis.spread_24h_ago,
                        yes_bid=analysis.yes_bid,
                        yes_ask=analysis.yes_ask,
                        previous_yes_bid=analysis.previous_yes_bid,
                        previous_yes_ask=analysis.previous_yes_ask,
            ))

    # Third pass: For expand events, activate sibling markets that are not already active
    if expand_events:
        print(f"\nFetching sibling markets for {len(expand_events)} expanding events...")
        sibling_recommendations = generate_sibling_activations(
            expand_events, existing_configs, all_markets_by_event, event_names
        )
        recommendations.extend(sibling_recommendations)

    return recommendations


def generate_sibling_activations(
    expand_events: set,
    existing_configs: Dict[str, MarketConfig],
    active_markets_by_event: Dict[str, List[str]],
    event_names: Dict[str, str],
) -> List[RecommendedAction]:
    """Generate recommendations to activate sibling markets for expanding events.

    Args:
        expand_events: Set of event tickers with expanding markets
        existing_configs: Dictionary of existing market configs in DynamoDB
        active_markets_by_event: Dictionary of currently active markets by event
        event_names: Dictionary mapping event_ticker to event name/title

    Returns:
        List of activate_sibling recommendations
    """
    recommendations = []
    existing_market_ids = set(existing_configs.keys())

    for event_ticker in expand_events:
        # Fetch all markets for this event from Kalshi API
        event_markets = fetch_event_markets(event_ticker)

        if not event_markets:
            print(f"  {event_ticker}: No markets found")
            continue

        # Find markets that are not already in our config
        active_markets = set(active_markets_by_event.get(event_ticker, []))
        new_markets = []

        for market in event_markets:
            market_id = market.get('ticker')
            if not market_id:
                continue

            # Skip if already active or already in config (even if disabled)
            if market_id in active_markets or market_id in existing_market_ids:
                continue

            # Skip closed/settled markets
            status = market.get('status', '')
            if status in ('closed', 'settled'):
                continue

            # Skip markets with low volume (< 10 in last 24hrs)
            volume_24h = market.get('volume_24h', 0) or 0
            if volume_24h < 10:
                continue

            new_markets.append(market)

        if not new_markets:
            print(f"  {event_ticker}: No new sibling markets to activate")
            continue

        # Sort by 24hr volume (descending) and take top 5
        new_markets.sort(key=lambda m: m.get('volume_24h', 0) or 0, reverse=True)
        top_markets = new_markets[:5]

        print(f"  {event_ticker}: Found {len(new_markets)} new sibling markets, assessing top {len(top_markets)} by volume...")

        # Assess information risk for each top market (in parallel for efficiency)
        # Increased workers from 5 to 10 for faster execution
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for market in top_markets:
                market_id = market.get('ticker')
                # Get mid-price for current price estimate
                yes_bid = market.get('yes_bid', 0) or 0
                yes_ask = market.get('yes_ask', 100) or 100
                current_price = (yes_bid + yes_ask) / 2

                # Use web search for highly active sibling markets (>1000 volume)
                volume_24h = market.get('volume_24h', 0) or 0
                use_web_search = volume_24h >= HIGH_ACTIVITY_VOLUME_24H

                future = executor.submit(
                    assess_information_risk,
                    market.get('title', ''),
                    current_price,
                    market.get('subtitle', ''),
                    market.get('rules_primary', ''),
                    use_web_search,
                )
                futures[future] = (market_id, market)

            for future in as_completed(futures):
                market_id, market = futures[future]
                ir_result = future.result()

                # Parse probability from result
                prob_str = ir_result.get("probability", "N/A")
                info_risk = None
                if prob_str and prob_str != "N/A":
                    try:
                        info_risk = float(str(prob_str).replace("%", "").strip())
                    except (ValueError, AttributeError):
                        pass

                volume_24h = market.get('volume_24h', 0) or 0
                is_highly_active = volume_24h >= HIGH_ACTIVITY_VOLUME_24H
                web_search_str = " (web_search)" if is_highly_active else ""

                # Filter out siblings with high info risk (>25%)
                if info_risk is not None and info_risk > MAX_INFO_RISK:
                    print(f"    {market_id}: volume_24h={volume_24h}, info_risk={info_risk}%{web_search_str} - SKIPPED (>{MAX_INFO_RISK}%)")
                    continue

                print(f"    {market_id}: volume_24h={volume_24h}, info_risk={info_risk}%{web_search_str} - OK")

                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    event_name=event_names.get(event_ticker, ''),
                    action='activate_sibling',
                    reason=f"Sibling of expanding market in event {event_ticker}",
                    new_enabled=True,
                    new_min_spread=DEFAULT_MIN_SPREAD,
                    new_quote_size=DEFAULT_QUOTE_SIZE,
                    new_max_inventory_yes=DEFAULT_MAX_INVENTORY_YES,
                    new_max_inventory_no=DEFAULT_MAX_INVENTORY_NO,
                    current_quote_size=None,  # New market, no current settings
                    current_min_spread=None,  # New market, no current settings
                    pnl_24h=0.0,
                    fill_count_24h=0,
                    fill_count_48h=0,
                    last_fill_time=None,  # New market, no fills yet
                    has_position=False,
                    position_qty=0,
                    info_risk_probability=info_risk,
                    info_risk_rationale=ir_result.get("rationale"),
                    created_at=None,  # New market, not yet in config
                    # Enriched Kalshi metadata - will be populated on next analysis
                    event_title=None,
                    market_title=None,
                    volume_24h_trades=0,
                    volume_24h_contracts=0,
                    buy_volume_trades=0,
                    buy_volume_contracts=0,
                    sell_volume_trades=0,
                    sell_volume_contracts=0,
                    current_spread=None,
                    spread_24h_ago=None,
                    yes_bid=None,
                    yes_ask=None,
                    previous_yes_bid=None,
                    previous_yes_ask=None,
                ))

    return recommendations


def write_recommendations_csv(
    recommendations: List[RecommendedAction],
    output_path: Path
) -> None:
    """Write recommendations to a CSV file for review.

    Args:
        recommendations: List of recommended actions
        output_path: Path to write CSV file
    """
    fieldnames = [
        'market_id',
        'event_ticker',
        'event_name',
        'action',
        'reason',
        'current_quote_size',
        'current_min_spread',
        'new_enabled',
        'new_min_spread',
        'new_quote_size',
        'new_max_inventory_yes',
        'new_max_inventory_no',
        'pnl_24h',
        'fill_count_24h',
        'fill_count_48h',
        'has_position',
        'position_qty',
        'info_risk_probability',
        'info_risk_rationale',
        'approve'  # User can set to 'yes' or 'no'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in recommendations:
            # Default approve based on action type
            # skip_event_active actions default to 'no' since no changes needed
            default_approve = 'no' if rec.action == 'skip_event_active' else 'yes'

            writer.writerow({
                'market_id': rec.market_id,
                'event_ticker': rec.event_ticker or '',
                'event_name': rec.event_name or '',
                'action': rec.action,
                'reason': rec.reason,
                'current_quote_size': int(rec.current_quote_size) if rec.current_quote_size is not None else '',
                'current_min_spread': f'{rec.current_min_spread:.2f}' if rec.current_min_spread is not None else '',
                'new_enabled': rec.new_enabled if rec.new_enabled is not None else '',
                'new_min_spread': rec.new_min_spread if rec.new_min_spread is not None else '',
                'new_quote_size': int(rec.new_quote_size) if rec.new_quote_size is not None else '',
                'new_max_inventory_yes': int(rec.new_max_inventory_yes) if rec.new_max_inventory_yes is not None else '',
                'new_max_inventory_no': int(rec.new_max_inventory_no) if rec.new_max_inventory_no is not None else '',
                'pnl_24h': f'{rec.pnl_24h:.2f}',
                'fill_count_24h': rec.fill_count_24h,
                'fill_count_48h': rec.fill_count_48h,
                'has_position': rec.has_position,
                'position_qty': int(rec.position_qty) if rec.position_qty else 0,
                'info_risk_probability': f'{rec.info_risk_probability:.0f}' if rec.info_risk_probability is not None else '',
                'info_risk_rationale': rec.info_risk_rationale or '',
                'approve': default_approve,
            })

    print(f"Wrote {len(recommendations)} recommendations to {output_path}")


def read_approved_recommendations(csv_path: Path) -> List[Dict]:
    """Read and filter approved recommendations from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of approved recommendation dictionaries
    """
    approved = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('approve', '').lower() in ('yes', 'y', 'true', '1'):
                approved.append(row)

    return approved


def execute_updates(
    dynamo: DynamoDBClient,
    approved: List[Dict]
) -> Tuple[int, int]:
    """Execute approved market config updates.

    Args:
        dynamo: DynamoDB client
        approved: List of approved recommendation dictionaries

    Returns:
        Tuple of (success_count, failure_count)
    """
    success = 0
    failure = 0

    for rec in approved:
        market_id = rec['market_id']
        action = rec.get('action', '')

        # Get current config or create new one for activate_sibling
        config = dynamo.get_market_config(market_id)

        if config is None:
            if action == 'activate_sibling':
                # Create new config for sibling market
                config = MarketConfig(
                    market_id=market_id,
                    enabled=True,
                    max_inventory_yes=DEFAULT_MAX_INVENTORY_YES,
                    max_inventory_no=DEFAULT_MAX_INVENTORY_NO,
                    min_spread=DEFAULT_MIN_SPREAD,
                    quote_size=DEFAULT_QUOTE_SIZE,
                    inventory_skew_factor=0.5,
                    event_ticker=rec.get('event_ticker') or None,
                    created_at=datetime.now(timezone.utc),
                )
            else:
                print(f"Warning: Market config not found for {market_id}, skipping")
                failure += 1
                continue
        else:
            # Apply updates to existing config
            if rec.get('new_enabled'):
                enabled_str = rec['new_enabled'].lower()
                if enabled_str in ('true', 'yes', '1'):
                    config.enabled = True
                elif enabled_str in ('false', 'no', '0'):
                    config.enabled = False

            if rec.get('new_min_spread'):
                try:
                    config.min_spread = float(rec['new_min_spread'])
                except ValueError:
                    pass

            if rec.get('new_quote_size'):
                try:
                    # Handle float strings like "20.0" from CSV
                    config.quote_size = int(float(rec['new_quote_size']))
                except (ValueError, TypeError):
                    pass

            if rec.get('new_max_inventory_yes'):
                try:
                    # Handle float strings like "40.0" from CSV
                    config.max_inventory_yes = int(float(rec['new_max_inventory_yes']))
                except (ValueError, TypeError):
                    pass

            if rec.get('new_max_inventory_no'):
                try:
                    # Handle float strings like "40.0" from CSV
                    config.max_inventory_no = int(float(rec['new_max_inventory_no']))
                except (ValueError, TypeError):
                    pass

        # Save updated/new config
        if dynamo.put_market_config(config.to_dict()):
            if action == 'activate_sibling':
                print(f"✓ Created {market_id}: {action} (info_risk={rec.get('info_risk_probability', 'N/A')}%)")
            else:
                print(f"✓ Updated {market_id}: {action}")
            success += 1
        else:
            print(f"✗ Failed to update {market_id}")
            failure += 1

    return success, failure


def save_execution_timestamp(dynamo: DynamoDBClient) -> bool:
    """Save the timestamp of the last market_update execution to the state table.

    Args:
        dynamo: DynamoDB client

    Returns:
        True if successful
    """
    try:
        dynamo.state_table.put_item(Item={
            'key': 'market_update_last_run',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        return True
    except Exception as e:
        print(f"Warning: Failed to save execution timestamp: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate and apply market config updates based on performance analysis.'
    )
    parser.add_argument(
        '--env',
        choices=['demo', 'prod'],
        required=True,
        help='Environment to analyze (demo or prod)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV path (default: market_updates_YYYYMMDD_HHMMSS.csv)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate CSV but do not prompt for execution'
    )
    parser.add_argument(
        '--execute',
        type=Path,
        default=None,
        help='Skip analysis and execute from an existing CSV file'
    )
    parser.add_argument(
        '--pnl-lookback',
        type=int,
        default=24,
        help='Hours to look back for P&L calculation (default: 24)'
    )
    parser.add_argument(
        '--volume-lookback',
        type=int,
        default=48,
        help='Hours to look back for fill count/volume (default: 48)'
    )

    args = parser.parse_args()

    # Initialize DynamoDB client
    dynamo = DynamoDBClient(region=args.region, environment=args.env)

    if args.execute:
        # Execute mode: apply updates from existing CSV
        if not args.execute.exists():
            print(f"Error: CSV file not found: {args.execute}")
            sys.exit(1)

        approved = read_approved_recommendations(args.execute)
        if not approved:
            print("No approved recommendations found in CSV")
            sys.exit(0)

        print(f"\nFound {len(approved)} approved updates to apply...")
        success, failure = execute_updates(dynamo, approved)
        print(f"\nCompleted: {success} succeeded, {failure} failed")

        # Save execution timestamp if any updates succeeded
        if success > 0:
            save_execution_timestamp(dynamo)

        sys.exit(0 if failure == 0 else 1)

    # Analysis mode
    print(f"Analyzing markets in {args.env} environment...")
    print(f"  P&L lookback: {args.pnl_lookback} hours")
    print(f"  Volume lookback: {args.volume_lookback} hours")
    analyses = analyze_markets(
        dynamo, args.env,
        pnl_lookback_hours=args.pnl_lookback,
        volume_lookback_hours=args.volume_lookback
    )
    print(f"Analyzed {len(analyses)} markets")

    # Get all existing configs (for sibling activation check)
    existing_configs = dynamo.get_all_market_configs(enabled_only=False)

    # Fetch event names for all analyzed markets
    unique_event_tickers = list(set(
        a.event_ticker for a in analyses.values() if a.event_ticker
    ))
    print(f"\nFetching event names for {len(unique_event_tickers)} events...")
    event_names = fetch_event_names(unique_event_tickers)

    # Generate recommendations
    recommendations = generate_recommendations(analyses, existing_configs, event_names)

    if not recommendations:
        print("\nNo recommended actions at this time.")
        sys.exit(0)

    # Summary
    exits = [r for r in recommendations if r.action == 'exit']
    scale_downs = [r for r in recommendations if r.action == 'scale_down']
    expands = [r for r in recommendations if r.action == 'expand']
    siblings = [r for r in recommendations if r.action == 'activate_sibling']
    print(f"\nRecommendations: {len(exits)} exits, {len(scale_downs)} scale-downs, {len(expands)} expansions, {len(siblings)} sibling activations")

    # Write CSV
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"market_updates_{args.env}_{timestamp}.csv")

    write_recommendations_csv(recommendations, output_path)

    if args.dry_run:
        print("\nDry run mode: Exiting without applying changes.")
        print(f"Review the CSV at: {output_path}")
        sys.exit(0)

    # Interactive approval flow
    print("\n" + "="*60)
    print("Please review and edit the CSV file if needed.")
    print(f"File: {output_path}")
    print("\nSet 'approve' column to 'yes' or 'no' for each row.")
    print("="*60)

    while True:
        response = input("\nPress Enter when ready to apply approved updates (or 'q' to quit): ")
        if response.lower() in ('q', 'quit', 'exit'):
            print("Aborted.")
            sys.exit(0)

        # Re-read the CSV to get user edits
        approved = read_approved_recommendations(output_path)

        if not approved:
            print("No approved recommendations found. Edit the CSV and try again.")
            continue

        print(f"\nReady to apply {len(approved)} updates:")
        for rec in approved:
            print(f"  - {rec['market_id']}: {rec['action']}")

        confirm = input("\nConfirm? (y/n): ")
        if confirm.lower() in ('y', 'yes'):
            break
        else:
            print("Edit the CSV and press Enter when ready.")

    # Execute updates
    print("\nApplying updates...")
    success, failure = execute_updates(dynamo, approved)
    print(f"\nCompleted: {success} succeeded, {failure} failed")

    # Save execution timestamp if any updates succeeded
    if success > 0:
        save_execution_timestamp(dynamo)

    sys.exit(0 if failure == 0 else 1)


if __name__ == '__main__':
    main()
