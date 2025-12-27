"""
Simplified market screener for Kalshi markets.

This module fetches all markets from the Kalshi API and outputs them to a CSV file.
Designed to be fast and simple - can be extended later to filter/score markets
and update market_config in DynamoDB for the dora_bot.

Usage:
    python market_screener.py [--output markets.csv] [--status open] [--mve-filter exclude] [--limit 1000]

By default, fetches only open markets and excludes MVE (multi-variate event) markets.

Filters applied:
    1. 24hr volume > 100 contracts
    2. Close time > 7 days away
    3. Current AND previous spread both > 5 cents
    4. Event ticker prefix not in restricted_markets.csv
    5. 24hr midpoint change <= 20%
    6. Information risk < 25% (likelihood of market-moving news in next 7 days)
"""
import argparse
import csv
import json
import os
import sys
import requests
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

from dora_bot.dynamo import DynamoDBClient
from dora_bot.models import MarketConfig


# Public API endpoint (no auth required for market data)
KALSHI_API_BASE = "https://api.elections.kalshi.com"
MARKETS_ENDPOINT = "/trade-api/v2/markets"
EVENTS_ENDPOINT = "/trade-api/v2/events"

# Filter thresholds
MIN_VOLUME_24H = 100
MIN_SPREAD = 5  # cents
MAX_INFO_RISK = 25  # percent - maximum acceptable information risk
MAX_MIDPOINT_CHANGE = 20  # percent - maximum 24hr midpoint change allowed
MIN_DAYS_UNTIL_CLOSE = 7  # minimum days until market closes
DEFAULT_THREADS = 10  # default number of parallel threads for API calls
MIN_SIDE_VOLUME = 20  # minimum volume on each side (buy/sell) from trade history

# Default market config values
DEFAULT_QUOTE_SIZE = 5
DEFAULT_MAX_INVENTORY = 5
DEFAULT_MIN_SPREAD = 0.04  # 4 cents

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def load_restricted_prefixes(filepath: str) -> Set[str]:
    """Load restricted market prefixes from CSV file.

    Args:
        filepath: Path to restricted_markets.csv

    Returns:
        Set of restricted event ticker prefixes
    """
    prefixes = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                prefix = line.strip()
                if prefix:
                    prefixes.add(prefix)
    return prefixes


def calculate_spread(ask: Optional[int], bid: Optional[int]) -> int:
    """Calculate spread between ask and bid prices.

    Args:
        ask: Ask price in cents
        bid: Bid price in cents

    Returns:
        Spread in cents, or 0 if either value is None
    """
    if ask is None or bid is None:
        return 0
    return ask - bid


def calculate_midpoint(ask: Optional[int], bid: Optional[int]) -> Optional[float]:
    """Calculate midpoint between ask and bid prices.

    Args:
        ask: Ask price in cents
        bid: Bid price in cents

    Returns:
        Midpoint in cents, or None if either value is None
    """
    if ask is None or bid is None:
        return None
    return (ask + bid) / 2


def is_closing_soon(close_time: Optional[str], min_days: int = MIN_DAYS_UNTIL_CLOSE) -> bool:
    """Check if a market closes within the minimum days threshold.

    Args:
        close_time: ISO format datetime string for when market closes
        min_days: Minimum days until close (default: MIN_DAYS_UNTIL_CLOSE)

    Returns:
        True if market closes within min_days, False otherwise (or if close_time is None)
    """
    if not close_time:
        return False

    try:
        # Parse ISO format datetime (e.g., "2025-01-15T12:00:00Z")
        close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        now = datetime.now(close_dt.tzinfo)
        days_until_close = (close_dt - now).days
        return days_until_close < min_days
    except (ValueError, TypeError):
        # If we can't parse the date, don't filter it out
        return False


def calculate_midpoint_change(
    current_ask: Optional[int],
    current_bid: Optional[int],
    previous_ask: Optional[int],
    previous_bid: Optional[int],
) -> Optional[float]:
    """Calculate percentage change in midpoint over 24 hours.

    Args:
        current_ask: Current ask price in cents
        current_bid: Current bid price in cents
        previous_ask: Previous ask price in cents
        previous_bid: Previous bid price in cents

    Returns:
        Absolute percentage change, or None if cannot be calculated
    """
    current_mid = calculate_midpoint(current_ask, current_bid)
    previous_mid = calculate_midpoint(previous_ask, previous_bid)

    if current_mid is None or previous_mid is None or previous_mid == 0:
        return None

    return abs(current_mid - previous_mid)


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


def filter_markets(
    markets: List[Dict[str, Any]],
    restricted_prefixes: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter markets based on volume, spread, and restricted tickers.

    Args:
        markets: List of market dictionaries from API
        restricted_prefixes: Set of restricted event ticker prefixes

    Returns:
        Tuple of (filtered list of markets, filter stats)
    """
    filtered = []
    stats = {
        "volume_24h": 0,
        "closing_soon": 0,
        "spread": 0,
        "restricted_prefix": 0,
        "midpoint_change": 0,
    }

    for market in markets:
        # Filter 1: 24hr volume > 100
        volume_24h = market.get("volume_24h", 0) or 0
        if volume_24h <= MIN_VOLUME_24H:
            stats["volume_24h"] += 1
            continue

        # Filter 2: Market doesn't close within MIN_DAYS_UNTIL_CLOSE days
        if is_closing_soon(market.get("close_time")):
            stats["closing_soon"] += 1
            continue

        # Filter 3: Current AND previous spread both > 5
        current_spread = calculate_spread(
            market.get("yes_ask"),
            market.get("yes_bid"),
        )
        previous_spread = calculate_spread(
            market.get("previous_yes_ask"),
            market.get("previous_yes_bid"),
        )

        if current_spread <= MIN_SPREAD or previous_spread <= MIN_SPREAD:
            stats["spread"] += 1
            continue

        # Filter 4: Event ticker prefix not in restricted list
        event_ticker = market.get("event_ticker", "")
        prefix = event_ticker[:5] if event_ticker else ""
        if prefix in restricted_prefixes:
            stats["restricted_prefix"] += 1
            continue

        # Filter 5: Midpoint change < 20% over 24hrs
        midpoint_change = calculate_midpoint_change(
            market.get("yes_ask"),
            market.get("yes_bid"),
            market.get("previous_yes_ask"),
            market.get("previous_yes_bid"),
        )
        if midpoint_change is not None and midpoint_change > MAX_MIDPOINT_CHANGE:
            stats["midpoint_change"] += 1
            continue

        # Add computed fields for convenience
        market["current_spread"] = current_spread
        market["previous_spread"] = previous_spread
        market["midpoint_change_24h"] = midpoint_change

        filtered.append(market)

    return filtered, stats


def fetch_all_markets(
    status: Optional[str] = "open",
    mve_filter: Optional[str] = "exclude",
    page_size: int = 1000,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch all markets from Kalshi API with pagination.

    Args:
        status: Filter by market status (default: 'open').
                Valid values: unopened, open, paused, closed, settled
        mve_filter: MVE filter mode - 'exclude', 'only', or None (default: 'exclude')
        page_size: Number of markets per page (max 1000)
        max_pages: Optional limit on number of pages to fetch (for testing)

    Returns:
        List of market dictionaries
    """
    all_markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page_count = 0

    print(f"Fetching markets from Kalshi API (page_size={page_size}, status={status}, mve_filter={mve_filter})...")

    while True:
        # Build request params
        params: Dict[str, Any] = {"limit": min(page_size, 1000)}
        if status:
            params["status"] = status
        if mve_filter:
            params["mve_filter"] = mve_filter
        if cursor:
            params["cursor"] = cursor

        # Make request
        url = f"{KALSHI_API_BASE}{MARKETS_ENDPOINT}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract markets
        markets = data.get("markets", [])
        all_markets.extend(markets)
        page_count += 1

        print(f"  Page {page_count}: fetched {len(markets)} markets (total: {len(all_markets)})")

        # Check for next page
        cursor = data.get("cursor")
        if not cursor:
            break

        # Check page limit
        if max_pages and page_count >= max_pages:
            print(f"  Reached max pages limit ({max_pages})")
            break

    return all_markets


def markets_to_dataframe(markets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of market dicts to a pandas DataFrame.

    Args:
        markets: List of market dictionaries from API

    Returns:
        DataFrame with all market fields, priority columns first
    """
    if not markets:
        return pd.DataFrame()

    # Flatten nested structures if needed
    flattened = []
    for market in markets:
        row = {}
        for key, value in market.items():
            if isinstance(value, dict):
                # Flatten nested dicts with prefix
                for nested_key, nested_value in value.items():
                    row[f"{key}_{nested_key}"] = nested_value
            elif isinstance(value, list):
                # Convert lists to JSON strings
                row[key] = str(value) if value else None
            else:
                row[key] = value
        flattened.append(row)

    df = pd.DataFrame(flattened)

    # Add activate column defaulting to 0
    df.insert(0, "activate", 0)

    # Define priority columns to appear first (after activate)
    priority_columns = [
        "activate",
        "ticker",
        "title",
        "rules_primary",
        "volume_24h",
        "yes_bid",
        "yes_ask",
        "info_risk_probability",
        "info_risk_rationale",
        "info_risk_error",
    ]

    # Reorder columns: priority columns first, then the rest
    existing_priority = [col for col in priority_columns if col in df.columns]
    other_columns = [col for col in df.columns if col not in priority_columns]
    df = df[existing_priority + other_columns]

    return df


def assess_information_risk(
    market_title: str,
    current_price: float,
    market_subtitle: Optional[str] = None,
    rules: Optional[str] = None,
) -> Dict[str, Any]:
    """Assess the likelihood of market-moving information being released in the next 7 days.

    Uses OpenAI API to evaluate information risk for a prediction market.

    Args:
        market_title: The title of the market
        current_price: Current market price (0-100)
        market_subtitle: Optional subtitle providing additional context
        rules: Optional resolution rules for the market

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
        prompt = f"""You are a market risk assessment expert for prediction markets on Kalshi. Your job is to evaluate the likelihood that market moving information will be released in the next 7 days that would move the current pricing more than 20% in either direction.

If the outcome of the market will be decided within the next 7 days, please return a 100% chance of market moving news.  Today is {datetime.now().strftime("%Y-%m-%d")}

Please return your assessment in the form of a likelihood percentage (number from 0-100%) and 2-3 sentence rationale.

{context}

Your response should be only a JSON dictionary e.g. {{"probability": "XX%", "rationale": "XXXX"}}"""

        response = client.responses.create(
                model="gpt-5-mini",
                reasoning={"effort": "medium"},
                input=prompt,
            )

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


def _assess_info_risk_single(
    market: Dict[str, Any],
    index: int,
    total: int,
) -> Tuple[Dict[str, Any], bool, str]:
    """Assess information risk for a single market (worker function for threading).

    Args:
        market: Market dictionary
        index: Market index (1-based)
        total: Total number of markets

    Returns:
        Tuple of (market with results, passed_filter, status_message)
    """
    ticker = market.get("ticker", "N/A")
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")
    rules = market.get("rules_primary", "")

    # Get mid-price for current price estimate
    yes_bid = market.get("yes_bid", 0) or 0
    yes_ask = market.get("yes_ask", 100) or 100
    current_price = (yes_bid + yes_ask) / 2

    # Call OpenAI to assess information risk
    result = assess_information_risk(
        market_title=title,
        current_price=current_price,
        market_subtitle=subtitle,
        rules=rules,
    )

    # Parse probability from result
    prob_str = result.get("probability", "N/A")
    error = result.get("error")

    if error:
        market["info_risk_probability"] = None
        market["info_risk_rationale"] = result.get("rationale", "")
        market["info_risk_error"] = error
        return market, True, f"[{index}/{total}] {ticker}: ERROR - {error}"

    # Parse percentage string to number
    try:
        prob_value = float(str(prob_str).replace("%", "").strip())
    except (ValueError, AttributeError):
        market["info_risk_probability"] = None
        market["info_risk_rationale"] = result.get("rationale", "")
        market["info_risk_error"] = f"Could not parse probability: {prob_str}"
        return market, True, f"[{index}/{total}] {ticker}: PARSE ERROR - '{prob_str}'"

    market["info_risk_probability"] = prob_value
    market["info_risk_rationale"] = result.get("rationale", "")
    market["info_risk_error"] = None

    if prob_value <= MAX_INFO_RISK:
        return market, True, f"[{index}/{total}] {ticker}: {prob_value:.0f}% - PASS"
    else:
        return market, False, f"[{index}/{total}] {ticker}: {prob_value:.0f}% - FILTERED OUT"


def assess_information_risk_for_markets(
    markets: List[Dict[str, Any]],
    max_workers: int = DEFAULT_THREADS,
) -> List[Dict[str, Any]]:
    """Assess information risk for each market and filter out high-risk markets.

    Args:
        markets: List of market dictionaries
        max_workers: Maximum number of parallel threads (default: 10)

    Returns:
        List of markets with info_risk_probability < MAX_INFO_RISK
    """
    filtered = []
    total = len(markets)

    print(f"\nAssessing information risk for {total} markets ({max_workers} threads)...")
    print(f"  (Filtering out markets with >{MAX_INFO_RISK}% chance of market-moving news)\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_assess_info_risk_single, market, i, total): i
            for i, market in enumerate(markets, 1)
        }

        # Process results as they complete
        for future in as_completed(futures):
            market, passed, message = future.result()
            print(f"  {message}")
            if passed:
                filtered.append(market)

    print(f"\nMarkets after information risk filter: {len(filtered)}/{total}")
    return filtered


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


def calculate_side_volumes(trades: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Calculate total volume on buy and sell sides from trade history.

    Args:
        trades: List of trade dictionaries

    Returns:
        Tuple of (buy_volume, sell_volume) - volume of YES buying and YES selling
    """
    buy_volume = 0
    sell_volume = 0

    for trade in trades:
        # Kalshi trades have 'taker_side' which is 'yes' or 'no'
        # 'yes' means the taker bought YES, 'no' means the taker sold YES (bought NO)
        count = trade.get("count", 0) or 0
        taker_side = trade.get("taker_side", "")

        if taker_side == "yes":
            buy_volume += count
        elif taker_side == "no":
            sell_volume += count

    return buy_volume, sell_volume


def _check_side_volume_single(
    market: Dict[str, Any],
    index: int,
    total: int,
) -> Tuple[Dict[str, Any], bool, str]:
    """Check side volume for a single market (worker function for threading).

    Args:
        market: Market dictionary
        index: Market index (1-based)
        total: Total number of markets

    Returns:
        Tuple of (market with results, passed_filter, status_message)
    """
    ticker = market.get("ticker", "N/A")

    # Fetch trade history
    trades = fetch_trade_history(ticker)

    if not trades:
        market["buy_volume"] = 0
        market["sell_volume"] = 0
        return market, False, f"[{index}/{total}] {ticker}: No trades found - FILTERED OUT"

    # Calculate side volumes
    buy_volume, sell_volume = calculate_side_volumes(trades)
    market["buy_volume"] = buy_volume
    market["sell_volume"] = sell_volume

    # Check if both sides have minimum volume
    if buy_volume >= MIN_SIDE_VOLUME and sell_volume >= MIN_SIDE_VOLUME:
        return market, True, f"[{index}/{total}] {ticker}: buy={buy_volume}, sell={sell_volume} - PASS"
    else:
        return market, False, f"[{index}/{total}] {ticker}: buy={buy_volume}, sell={sell_volume} - FILTERED OUT (need {MIN_SIDE_VOLUME} on each side)"


def filter_by_side_volume(
    markets: List[Dict[str, Any]],
    max_workers: int = 3,  # Lower default to avoid rate limiting
) -> List[Dict[str, Any]]:
    """Filter markets by requiring minimum volume on both buy and sell sides.

    Args:
        markets: List of market dictionaries
        max_workers: Maximum number of parallel threads (default 3 to avoid rate limits)

    Returns:
        List of markets with sufficient volume on both sides
    """
    filtered = []
    total = len(markets)

    print(f"\nChecking trade history for {total} markets ({max_workers} threads)...")
    print(f"  (Filtering out markets with <{MIN_SIDE_VOLUME} volume on either side)\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_check_side_volume_single, market, i, total): i
            for i, market in enumerate(markets, 1)
        }

        # Process results as they complete
        for future in as_completed(futures):
            market, passed, message = future.result()
            print(f"  {message}")
            if passed:
                filtered.append(market)

    print(f"\nMarkets after side volume filter: {len(filtered)}/{total}")
    return filtered


def write_candidates_csv(
    markets: List[Dict[str, Any]],
    output_path: str,
    event_names: Dict[str, str],
) -> None:
    """Write candidate markets to a CSV file for review.

    Args:
        markets: List of market dictionaries
        output_path: Path to write CSV file
        event_names: Dictionary mapping event_ticker to event name/title
    """
    fieldnames = [
        'market_id',
        'event_ticker',
        'event_name',
        'title',
        'volume_24h',
        'buy_volume',
        'sell_volume',
        'yes_bid',
        'yes_ask',
        'fair_value',
        'fair_value_rationale',
        'info_risk_probability',
        'new_quote_size',
        'new_max_inventory_yes',
        'new_max_inventory_no',
        'new_min_spread',
        'approve'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for market in markets:
            event_ticker = market.get('event_ticker', '')
            writer.writerow({
                'market_id': market.get('ticker', ''),
                'event_ticker': event_ticker,
                'event_name': event_names.get(event_ticker, ''),
                'title': market.get('title', ''),
                'volume_24h': market.get('volume_24h', 0),
                'buy_volume': market.get('buy_volume', 0),
                'sell_volume': market.get('sell_volume', 0),
                'yes_bid': market.get('yes_bid', 0),
                'yes_ask': market.get('yes_ask', 0),
                'fair_value': f"{market.get('fair_value', 0):.1f}" if market.get('fair_value') else '',
                'fair_value_rationale': market.get('fair_value_rationale', ''),
                'info_risk_probability': f"{market.get('info_risk_probability', 0):.0f}" if market.get('info_risk_probability') else '',
                'new_quote_size': DEFAULT_QUOTE_SIZE,
                'new_max_inventory_yes': DEFAULT_MAX_INVENTORY,
                'new_max_inventory_no': DEFAULT_MAX_INVENTORY,
                'new_min_spread': DEFAULT_MIN_SPREAD,
                'approve': 'yes',  # Default to yes
            })

    print(f"Wrote {len(markets)} candidates to {output_path}")


def read_approved_markets(csv_path: str) -> List[Dict]:
    """Read and filter approved markets from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of approved market dictionaries
    """
    approved = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('approve', '').lower() in ('yes', 'y', 'true', '1'):
                approved.append(row)

    return approved


def upload_to_market_config(
    approved: List[Dict],
    environment: str = "prod",
    region: str = "us-east-1",
) -> Tuple[int, int]:
    """Upload approved markets to DynamoDB market_config table.

    Args:
        approved: List of approved market dictionaries
        environment: 'demo' or 'prod'
        region: AWS region

    Returns:
        Tuple of (success_count, failure_count)
    """
    dynamo = DynamoDBClient(region=region, environment=environment)

    success = 0
    failure = 0

    for market in approved:
        market_id = market['market_id']

        try:
            existing = dynamo.get_market_config(market_id)
            if existing and existing.enabled:
                print(f"↷ Skipped {market_id}: already active")
                continue

            # Parse values from CSV
            quote_size = int(float(market.get('new_quote_size', DEFAULT_QUOTE_SIZE)))
            max_inv_yes = int(float(market.get('new_max_inventory_yes', DEFAULT_MAX_INVENTORY)))
            max_inv_no = int(float(market.get('new_max_inventory_no', DEFAULT_MAX_INVENTORY)))
            min_spread = float(market.get('new_min_spread', DEFAULT_MIN_SPREAD))

            # Parse fair value if present
            fair_value = None
            if market.get('fair_value'):
                try:
                    fair_value = float(market['fair_value']) / 100.0  # Convert % to decimal
                except (ValueError, TypeError):
                    pass

            config = MarketConfig(
                market_id=market_id,
                enabled=True,
                max_inventory_yes=max_inv_yes,
                max_inventory_no=max_inv_no,
                min_spread=min_spread,
                quote_size=quote_size,
                inventory_skew_factor=0.5,
                fair_value=fair_value,
                event_ticker=market.get('event_ticker') or None,
                created_at=datetime.now(timezone.utc),
            )

            if dynamo.put_market_config(config):
                print(f"✓ Created {market_id}")
                success += 1
            else:
                print(f"✗ Failed to create {market_id}")
                failure += 1

        except Exception as e:
            print(f"✗ Failed to create {market_id}: {e}")
            failure += 1

    return success, failure


def print_top_markets(markets: List[Dict[str, Any]], n: int = 5) -> None:
    """Print top N markets by 24hr volume.

    Args:
        markets: List of filtered market dictionaries
        n: Number of markets to print
    """
    print(f"\n{'='*80}")
    print(f"TOP {n} CANDIDATE MARKETS (by 24hr volume)")
    print(f"{'='*80}")

    for i, market in enumerate(markets[:n], 1):
        ticker = market.get("ticker", "N/A")
        title = market.get("title", "N/A")
        volume_24h = market.get("volume_24h", 0)
        current_spread = market.get("current_spread", 0)
        previous_spread = market.get("previous_spread", 0)
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 0)
        info_risk = market.get("info_risk_probability")
        info_rationale = market.get("info_risk_rationale", "")
        fair_value = market.get("fair_value")
        fair_value_rationale = market.get("fair_value_rationale", "")

        print(f"\n{i}. {ticker}")
        print(f"   Title: {title[:60]}{'...' if len(title) > 60 else ''}")
        print(f"   24hr Volume: {volume_24h:,} contracts")
        print(f"   Current Bid/Ask: {yes_bid}/{yes_ask} (spread: {current_spread})")
        print(f"   Previous Spread: {previous_spread}")
        if info_risk is not None:
            print(f"   Info Risk: {info_risk:.0f}%")
            if info_rationale:
                # Truncate rationale for display
                rationale_display = info_rationale[:80] + "..." if len(info_rationale) > 80 else info_rationale
                print(f"   Rationale: {rationale_display}")
        if fair_value is not None:
            market_mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else 0
            edge = fair_value - market_mid
            print(f"   Fair Value: {fair_value:.0f}% (edge: {edge:+.0f}%)")
            if fair_value_rationale:
                # Truncate rationale for display
                fv_rationale_display = fair_value_rationale[:80] + "..." if len(fair_value_rationale) > 80 else fair_value_rationale
                print(f"   FV Rationale: {fv_rationale_display}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Screen Kalshi markets and add to market_config.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: screener_candidates_TIMESTAMP.csv)",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="open",
        choices=["unopened", "open", "paused", "closed", "settled", "none"],
        help="Filter by market status (default: 'open'). Use 'none' to disable filter.",
    )
    parser.add_argument(
        "--mve-filter",
        type=str,
        default="exclude",
        choices=["exclude", "only", "none"],
        help="MVE filter: 'exclude' (default), 'only', or 'none' to disable",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Page size for API requests (default: 1000, max: 1000)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to fetch (for testing)",
    )
    parser.add_argument(
        "--restricted-file",
        type=str,
        default=None,
        help="Path to restricted markets CSV (default: restricted_markets.csv in same directory)",
    )
    parser.add_argument(
        "--skip-info-risk",
        action="store_true",
        help="Skip information risk assessment (faster, but no AI filtering)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of parallel threads for API calls (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top markets to output (default: 10)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="prod",
        choices=["demo", "prod"],
        help="Environment for uploading to DynamoDB (default: prod)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate CSV but do not prompt for upload",
    )
    parser.add_argument(
        "--execute",
        type=str,
        default=None,
        help="Skip screening and execute from an existing CSV file",
    )
    args = parser.parse_args()

    # Execute mode: upload from existing CSV
    if args.execute:
        if not os.path.exists(args.execute):
            print(f"Error: CSV file not found: {args.execute}")
            sys.exit(1)

        approved = read_approved_markets(args.execute)
        if not approved:
            print("No approved markets found in CSV")
            sys.exit(0)

        print(f"\nFound {len(approved)} approved markets to upload...")
        success, failure = upload_to_market_config(approved, environment=args.env)
        print(f"\nCompleted: {success} succeeded, {failure} failed")
        sys.exit(0 if failure == 0 else 1)

    start_time = datetime.now()
    print(f"Market Screener started at {start_time.isoformat()}")

    # Load restricted prefixes
    script_dir = os.path.dirname(os.path.abspath(__file__))
    restricted_file = args.restricted_file or os.path.join(script_dir, "restricted_markets.csv")
    restricted_prefixes = load_restricted_prefixes(restricted_file)
    print(f"Loaded {len(restricted_prefixes)} restricted market prefixes")

    # Convert 'none' to None for filters
    mve_filter = None if args.mve_filter == "none" else args.mve_filter
    status = None if args.status == "none" else args.status

    # Fetch markets
    markets = fetch_all_markets(
        status=status,
        mve_filter=mve_filter,
        page_size=args.limit,
        max_pages=args.max_pages,
    )

    print(f"\nTotal markets fetched: {len(markets)}")

    # Apply filters
    filtered_markets, filter_stats = filter_markets(markets, restricted_prefixes)
    print(f"Markets after filtering: {len(filtered_markets)}")
    print(f"  - Volume > {MIN_VOLUME_24H} (filtered out {filter_stats['volume_24h']})")
    print(f"  - Close time > {MIN_DAYS_UNTIL_CLOSE} days away (filtered out {filter_stats['closing_soon']})")
    print(f"  - Current & previous spread > {MIN_SPREAD} (filtered out {filter_stats['spread']})")
    print(
        f"  - Excluded {len(restricted_prefixes)} restricted prefixes "
        f"(filtered out {filter_stats['restricted_prefix']})"
    )
    print(f"  - 24hr midpoint change <= {MAX_MIDPOINT_CHANGE}% (filtered out {filter_stats['midpoint_change']})")

    if not filtered_markets:
        print("No markets passed filters.")
        return

    # Sort by 24hr volume (descending)
    filtered_markets.sort(key=lambda x: x.get("volume_24h", 0) or 0, reverse=True)

    # Apply information risk filter using OpenAI
    if not args.skip_info_risk:
        filtered_markets = assess_information_risk_for_markets(
            filtered_markets, max_workers=args.threads
        )

    if not filtered_markets:
        print("No markets passed information risk filter.")
        return

    # Re-sort after filtering (in case order changed)
    filtered_markets.sort(key=lambda x: x.get("volume_24h", 0) or 0, reverse=True)

    # Filter by trade history - require volume on both sides
    filtered_markets = filter_by_side_volume(filtered_markets, max_workers=args.threads)

    if not filtered_markets:
        print("No markets passed side volume filter.")
        return

    # Re-sort after filtering
    filtered_markets.sort(key=lambda x: x.get("volume_24h", 0) or 0, reverse=True)

    # Take top N markets
    top_markets = filtered_markets[:args.top_n]
    print(f"\nSelected top {len(top_markets)} markets by 24hr volume")

    # Print top candidates
    print_top_markets(top_markets, n=len(top_markets))

    # Fetch event names for the selected markets
    unique_event_tickers = list(set(
        m.get('event_ticker', '') for m in top_markets if m.get('event_ticker')
    ))
    print(f"\nFetching event names for {len(unique_event_tickers)} events...")
    event_names = fetch_event_names(unique_event_tickers)

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"screener_candidates_{timestamp}.csv"

    # Write candidates CSV
    write_candidates_csv(top_markets, output_path, event_names)

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nScreening completed in {duration:.1f} seconds")

    if args.dry_run:
        print("\nDry run mode: Exiting without uploading.")
        print(f"Review the CSV at: {output_path}")
        sys.exit(0)

    # Interactive approval flow
    print("\n" + "="*60)
    print("Please review and edit the CSV file if needed.")
    print(f"File: {output_path}")
    print("\nSet 'approve' column to 'yes' or 'no' for each row.")
    print("="*60)

    while True:
        response = input("\nPress Enter when ready to upload approved markets (or 'q' to quit): ")
        if response.lower() in ('q', 'quit', 'exit'):
            print("Aborted.")
            sys.exit(0)

        # Re-read the CSV to get user edits
        approved = read_approved_markets(output_path)

        if not approved:
            print("No approved markets found. Edit the CSV and try again.")
            continue

        print(f"\nReady to upload {len(approved)} markets to {args.env}:")
        for market in approved:
            print(f"  - {market['market_id']}")

        confirm = input("\nConfirm? (y/n): ")
        if confirm.lower() in ('y', 'yes'):
            break
        else:
            print("Edit the CSV and press Enter when ready.")

    # Upload to DynamoDB
    print("\nUploading to market_config...")
    success, failure = upload_to_market_config(approved, environment=args.env)
    print(f"\nCompleted: {success} succeeded, {failure} failed")
    sys.exit(0 if failure == 0 else 1)


if __name__ == "__main__":
    main()
