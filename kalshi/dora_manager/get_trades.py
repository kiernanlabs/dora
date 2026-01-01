#!/usr/bin/env python3
"""
Simple script to fetch recent trades for a Kalshi market.

Usage:
    python get_trades.py [MARKET_TICKER] [--limit N] [--output FILE.csv]
"""
import argparse
import requests
import json
import csv
from datetime import datetime

KALSHI_API_BASE = "https://api.elections.kalshi.com"


def fetch_trades(ticker: str, limit: int = 20):
    """Fetch recent trades for a market."""
    url = f"{KALSHI_API_BASE}/trade-api/v2/markets/trades"
    params = {"ticker": ticker, "limit": limit}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("trades", [])
    except Exception as e:
        print(f"Error fetching trades: {e}")
        return []


def save_to_csv(trades, ticker, output_path):
    """Save trades to a CSV file."""
    if not trades:
        print(f"No trades to save for {ticker}")
        return

    fieldnames = ["timestamp", "ticker", "taker_side", "yes_price", "count", "trade_id"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trade in trades:
            created_time = trade.get("created_time", "")
            if created_time:
                try:
                    dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    timestamp = created_time
            else:
                timestamp = ""

            writer.writerow({
                "timestamp": timestamp,
                "ticker": ticker,
                "taker_side": trade.get("taker_side", ""),
                "yes_price": trade.get("yes_price", 0),
                "count": trade.get("count", 0),
                "trade_id": trade.get("trade_id", ""),
            })

    print(f"Saved {len(trades)} trades to {output_path}")


def display_trades(trades, ticker):
    """Display trades in a readable format."""
    if not trades:
        print(f"No trades found for {ticker}")
        return

    print(f"\n{'='*80}")
    print(f"Last {len(trades)} trades for {ticker}")
    print(f"{'='*80}\n")

    for i, trade in enumerate(trades, 1):
        created_time = trade.get("created_time", "")
        if created_time:
            # Parse and format timestamp
            try:
                dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except:
                time_str = created_time
        else:
            time_str = "N/A"

        count = trade.get("count", 0)
        yes_price = trade.get("yes_price", 0)
        taker_side = trade.get("taker_side", "N/A")

        print(f"{i:3}. {time_str}")
        print(f"     Side: {taker_side:3}  |  Price: {yes_price}¢  |  Size: {count} contracts")
        print()


def main():
    parser = argparse.ArgumentParser(description="Fetch recent trades for a Kalshi market")
    parser.add_argument("ticker", nargs="?", default="KXCABLEAVE-25-26FEB", help="Market ticker")
    parser.add_argument("--limit", type=int, default=20, help="Number of trades to fetch (default: 20)")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file path")
    parser.add_argument("--no-display", action="store_true", help="Don't display trades (only save to CSV)")
    args = parser.parse_args()

    trades = fetch_trades(args.ticker, args.limit)

    if args.output:
        save_to_csv(trades, args.ticker, args.output)

    if not args.no_display:
        display_trades(trades, args.ticker)


if __name__ == "__main__":
    main()
