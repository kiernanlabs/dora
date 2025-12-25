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

Additional analysis:
    - Fair value assessment (AI estimate of YES probability without market reference)
"""
import argparse
import json
import os
import requests
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI


# Public API endpoint (no auth required for market data)
KALSHI_API_BASE = "https://api.elections.kalshi.com"
MARKETS_ENDPOINT = "/trade-api/v2/markets"

# Filter thresholds
MIN_VOLUME_24H = 100
MIN_SPREAD = 5  # cents
MAX_INFO_RISK = 25  # percent - maximum acceptable information risk
MAX_MIDPOINT_CHANGE = 20  # percent - maximum 24hr midpoint change allowed
MIN_DAYS_UNTIL_CLOSE = 7  # minimum days until market closes
DEFAULT_THREADS = 10  # default number of parallel threads for API calls

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


def filter_markets(
    markets: List[Dict[str, Any]],
    restricted_prefixes: Set[str],
) -> List[Dict[str, Any]]:
    """Filter markets based on volume, spread, and restricted tickers.

    Args:
        markets: List of market dictionaries from API
        restricted_prefixes: Set of restricted event ticker prefixes

    Returns:
        Filtered list of markets
    """
    filtered = []

    for market in markets:
        # Filter 1: 24hr volume > 100
        volume_24h = market.get("volume_24h", 0) or 0
        if volume_24h <= MIN_VOLUME_24H:
            continue

        # Filter 2: Market doesn't close within MIN_DAYS_UNTIL_CLOSE days
        if is_closing_soon(market.get("close_time")):
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
            continue

        # Filter 4: Event ticker prefix not in restricted list
        event_ticker = market.get("event_ticker", "")
        prefix = event_ticker[:5] if event_ticker else ""
        if prefix in restricted_prefixes:
            continue

        # Filter 5: Midpoint change < 20% over 24hrs
        midpoint_change = calculate_midpoint_change(
            market.get("yes_ask"),
            market.get("yes_bid"),
            market.get("previous_yes_ask"),
            market.get("previous_yes_bid"),
        )
        if midpoint_change is not None and midpoint_change > MAX_MIDPOINT_CHANGE:
            continue

        # Add computed fields for convenience
        market["current_spread"] = current_spread
        market["previous_spread"] = previous_spread
        market["midpoint_change_24h"] = midpoint_change

        filtered.append(market)

    return filtered


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
        "fair_value",
        "fair_value_rationale",
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
                tools=[{"type": "web_search"}],
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


def assess_fair_value(
    market_title: str,
    market_subtitle: Optional[str] = None,
    rules: Optional[str] = None,
) -> Dict[str, Any]:
    """Assess the fair value (likelihood of resolving YES) for a market.

    Uses OpenAI API to evaluate the probability without referencing prediction markets.

    Args:
        market_title: The title of the market
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
        context = f"The question is: {market_title}"
        if market_subtitle:
            context += f"\n{market_subtitle}"
        if rules:
            context += f"\n\nResolution criteria: {rules[:500]}"  # Limit rules length

        # Create the prompt
        prompt = f"""You are an expert analyst tasked with estimating probabilities for real-world events. Your job is to estimate the likelihood that the following event will resolve to YES.

Do NOT reference prediction markets, betting odds, or market prices. Base your estimate solely on your knowledge of the subject matter, historical precedents, and logical reasoning.

Today is {datetime.now().strftime("%Y-%m-%d")}

{context}

Please return your assessment in the form of a likelihood percentage (number from 0-100%) and 2-3 sentence rationale explaining your reasoning.

Your response should be only a JSON dictionary e.g. {{"probability": "XX%", "rationale": "XXXX"}}"""

        response = client.responses.create(
                model="gpt-5-mini",
                reasoning={"effort": "medium"},
                tools=[{"type": "web_search"}],
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


def _assess_fair_value_single(
    market: Dict[str, Any],
    index: int,
    total: int,
) -> Tuple[Dict[str, Any], str]:
    """Assess fair value for a single market (worker function for threading).

    Args:
        market: Market dictionary
        index: Market index (1-based)
        total: Total number of markets

    Returns:
        Tuple of (market with results, status_message)
    """
    ticker = market.get("ticker", "N/A")
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")
    rules = market.get("rules_primary", "")

    # Call OpenAI to assess fair value
    result = assess_fair_value(
        market_title=title,
        market_subtitle=subtitle,
        rules=rules,
    )

    # Parse probability from result
    prob_str = result.get("probability", "N/A")
    error = result.get("error")

    if error:
        market["fair_value"] = None
        market["fair_value_rationale"] = result.get("rationale", "")
        market["fair_value_error"] = error
        return market, f"[{index}/{total}] {ticker}: ERROR - {error}"

    # Parse percentage string to number
    try:
        prob_value = float(str(prob_str).replace("%", "").strip())
    except (ValueError, AttributeError):
        market["fair_value"] = None
        market["fair_value_rationale"] = result.get("rationale", "")
        market["fair_value_error"] = f"Could not parse probability: {prob_str}"
        return market, f"[{index}/{total}] {ticker}: PARSE ERROR - '{prob_str}'"

    market["fair_value"] = prob_value
    market["fair_value_rationale"] = result.get("rationale", "")
    market["fair_value_error"] = None

    # Calculate edge vs current market price
    yes_bid = market.get("yes_bid", 0) or 0
    yes_ask = market.get("yes_ask", 100) or 100
    market_mid = (yes_bid + yes_ask) / 2
    edge = prob_value - market_mid

    return market, f"[{index}/{total}] {ticker}: {prob_value:.0f}% (market: {market_mid:.0f}%, edge: {edge:+.0f}%)"


def assess_fair_value_for_markets(
    markets: List[Dict[str, Any]],
    max_workers: int = DEFAULT_THREADS,
) -> List[Dict[str, Any]]:
    """Assess fair value for each market.

    Args:
        markets: List of market dictionaries
        max_workers: Maximum number of parallel threads (default: 10)

    Returns:
        Same list of markets with fair_value fields added
    """
    total = len(markets)

    print(f"\nAssessing fair value for {total} markets ({max_workers} threads)...")
    print(f"  (Estimating YES probability without market reference)\n")

    # We need to preserve market order, so collect results and rebuild list
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_assess_fair_value_single, market, i, total): i
            for i, market in enumerate(markets, 1)
        }

        # Process results as they complete
        for future in as_completed(futures):
            index = futures[future]
            market, message = future.result()
            print(f"  {message}")
            results[index] = market

    # Rebuild list in original order
    markets = [results[i] for i in range(1, total + 1)]

    print(f"\nFair value assessment complete for {total} markets")
    return markets


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
    parser = argparse.ArgumentParser(description="Fetch all Kalshi markets and output to CSV.")
    parser.add_argument(
        "--output",
        type=str,
        default="markets.csv",
        help="Output CSV file path (default: markets.csv)",
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
        "--skip-fair-value",
        action="store_true",
        help="Skip fair value assessment",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of parallel threads for API calls (default: {DEFAULT_THREADS})",
    )
    args = parser.parse_args()

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
    filtered_markets = filter_markets(markets, restricted_prefixes)
    print(f"Markets after filtering: {len(filtered_markets)}")
    print(f"  - Volume > {MIN_VOLUME_24H}")
    print(f"  - Close time > {MIN_DAYS_UNTIL_CLOSE} days away")
    print(f"  - Current & previous spread > {MIN_SPREAD}")
    print(f"  - Excluded {len(restricted_prefixes)} restricted prefixes")
    print(f"  - 24hr midpoint change <= {MAX_MIDPOINT_CHANGE}%")

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

    # Assess fair value for remaining markets using OpenAI
    if not args.skip_fair_value:
        filtered_markets = assess_fair_value_for_markets(
            filtered_markets, max_workers=args.threads
        )

    # Print top 5 candidates
    print_top_markets(filtered_markets, n=5)

    # Convert to DataFrame
    df = markets_to_dataframe(filtered_markets)

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} filtered markets to {args.output}")

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nCompleted in {duration:.1f} seconds")


if __name__ == "__main__":
    main()
