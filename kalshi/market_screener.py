"""
Simplified market screener for Kalshi markets.

This module fetches all markets from the Kalshi API and outputs them to a CSV file.
Designed to be fast and simple - can be extended later to filter/score markets
and update market_config in DynamoDB for the dora_bot.

Usage:
    python market_screener.py [--output markets.csv] [--status open] [--mve-filter exclude] [--limit 1000]

By default, fetches only open markets and excludes MVE (multi-variate event) markets.
"""
import argparse
import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime


# Public API endpoint (no auth required for market data)
KALSHI_API_BASE = "https://api.elections.kalshi.com"
MARKETS_ENDPOINT = "/trade-api/v2/markets"


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
        DataFrame with all market fields
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

    return pd.DataFrame(flattened)


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
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Market Screener started at {start_time.isoformat()}")

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

    # Convert to DataFrame
    df = markets_to_dataframe(markets)

    if df.empty:
        print("No markets found.")
        return

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} markets to {args.output}")

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nCompleted in {duration:.1f} seconds")

    # Print column info
    print(f"\nColumns in output ({len(df.columns)} total):")
    for col in sorted(df.columns):
        print(f"  - {col}")


if __name__ == "__main__":
    main()
