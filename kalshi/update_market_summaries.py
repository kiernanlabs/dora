"""
Recompute spread_sufficient_flag and backfill 7d volume/profit fields for an existing market_summaries.csv.

Rules:
- spread_sufficient_flag: if current spread is <= 2¢, flag = 0. Else flag = 1 only if both
  7d and 60d required spreads are <= 50% of current spread. If any needed value is missing,
  flag is left blank.
- Backfills:
  - volume_7d_buy_yes / volume_7d_sell_yes from API when missing
  - volume_7d_min_buy_sell = min(volume_7d_buy_yes, volume_7d_sell_yes)
  - expected_weekly_profit_cents = volume_7d_min_buy_sell * current_spread (falls back to 7d_current_spread)

Usage:
    python update_market_summaries.py [--csv market_summaries.csv] [--prod] [--trade-limit 1000]
"""
import argparse
import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from kalshi_service import KalshiService


def safe(val):
    return None if (val is None or (isinstance(val, float) and math.isnan(val))) else val


def compute_flag(row) -> float:
    current_spread = safe(row.get('current_spread')) or safe(row.get('7d_current_spread'))
    req_7 = safe(row.get('7d_required_spread'))
    req_60 = safe(row.get('60d_required_spread'))

    if current_spread is None or req_7 is None or req_60 is None:
        return None

    if current_spread <= 2.0:
        return 0

    return int((req_7 <= 0.5 * current_spread) and (req_60 <= 0.5 * current_spread))


def compute_min_buy_sell(row):
    buy = safe(row.get('volume_7d_buy_yes'))
    sell = safe(row.get('volume_7d_sell_yes'))
    if buy is None or sell is None:
        # fall back to existing value if present
        return safe(row.get('volume_7d_min_buy_sell'))
    return min(buy, sell)


def compute_expected_weekly_profit(row):
    min_vol = safe(row.get('volume_7d_min_buy_sell'))
    if min_vol is None:
        min_vol = compute_min_buy_sell(row)
    spread = safe(row.get('current_spread')) or safe(row.get('7d_current_spread'))
    if min_vol is None or spread is None:
        return None
    return min_vol * spread


def fetch_7d_volumes(ticker: str, service: KalshiService, trade_limit: int = 1000):
    """Fetch trades and compute 7d buy/sell volumes for a ticker."""
    df = service.get_trades_dataframe(ticker=ticker, limit=trade_limit)
    if df is None or df.empty:
        return None, None

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    if 'timestamp' in df.columns:
        df = df[df['timestamp'] >= cutoff]

    if df.empty or 'action' not in df.columns or 'count' not in df.columns:
        return None, None

    buy_yes = df[df['action'] == 'Buy Yes']['count'].sum()
    sell_yes = df[df['action'] == 'Sell Yes']['count'].sum()
    return buy_yes, sell_yes


def main():
    parser = argparse.ArgumentParser(description="Update spread_sufficient_flag and backfill volumes for existing CSV.")
    parser.add_argument('--csv', type=str, default='market_summaries.csv', help='Path to CSV to update in-place.')
    parser.add_argument('--prod', action='store_true', help='Use production environment (default: demo).')
    parser.add_argument('--trade-limit', type=int, default=1000, help='Number of trades to fetch per market when backfilling volumes.')
    parser.add_argument('--batch-size', type=int, default=100, help='Write progress to CSV after this many rows.')
    args = parser.parse_args()

    service = KalshiService(use_demo=not args.prod)

    df = pd.read_csv(args.csv)
    # Ensure new columns exist
    for col in ['volume_7d_buy_yes', 'volume_7d_sell_yes', 'volume_7d_min_buy_sell', 'expected_weekly_profit_cents']:
        if col not in df.columns:
            df[col] = None

    total_rows = len(df)

    # Backfill missing 7d volumes from API
    for idx, row in df.iterrows():
        human_idx = idx + 1
        buy = safe(row.get('volume_7d_buy_yes'))
        sell = safe(row.get('volume_7d_sell_yes'))
        ticker = row.get('market_ticker')
        if ticker and (buy is None or sell is None):
            try:
                print(f"[{human_idx}/{total_rows}] Backfilling 7d volumes for {ticker}...")
                buy_val, sell_val = fetch_7d_volumes(ticker, service, trade_limit=args.trade_limit)
                if buy_val is not None:
                    df.at[idx, 'volume_7d_buy_yes'] = buy_val
                if sell_val is not None:
                    df.at[idx, 'volume_7d_sell_yes'] = sell_val
            except Exception as exc:
                # Leave values as-is on error
                print(f"Warning: failed to backfill volumes for {ticker}: {exc}")
        else:
            print(f"[{human_idx}/{total_rows}] Skipping {ticker or 'unknown ticker'} (volumes present or ticker missing)")

        # Per-row recompute for interim writes
        row_series = df.loc[idx]
        df.at[idx, 'volume_7d_min_buy_sell'] = compute_min_buy_sell(row_series)
        df.at[idx, 'expected_weekly_profit_cents'] = compute_expected_weekly_profit(row_series)
        df.at[idx, 'spread_sufficient_flag'] = compute_flag(row_series)

        # Batch write progress
        if human_idx % args.batch_size == 0:
            df.to_csv(args.csv, index=False)
            print(f"Checkpoint: wrote progress through row {human_idx} to {args.csv}")

    # Recompute fields
    df['volume_7d_min_buy_sell'] = df.apply(compute_min_buy_sell, axis=1)
    df['expected_weekly_profit_cents'] = df.apply(compute_expected_weekly_profit, axis=1)
    df['spread_sufficient_flag'] = df.apply(compute_flag, axis=1)
    df.to_csv(args.csv, index=False)
    print(f"Updated spread_sufficient_flag and backfilled 7d min/expected profit for {len(df)} rows in {args.csv}")


if __name__ == "__main__":
    main()
