"""
Recompute spread_sufficient_flag and backfill new columns for an existing market_summaries.csv.

Rules:
- spread_sufficient_flag: if current spread is <= 2¢, flag = 0. Else flag = 1 only if both
  7d and 60d required spreads are <= 50% of current spread. If any needed value is missing,
  flag is left blank.
- Backfills:
  - volume_7d_min_buy_sell = min(volume_7d_buy_yes, volume_7d_sell_yes) when available
  - expected_weekly_profit_cents = volume_7d_min_buy_sell * current_spread (falls back to 7d_current_spread)

Usage:
    python update_spread_flag.py [--csv market_summaries.csv]
"""
import argparse
import pandas as pd
import math


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


def main():
    parser = argparse.ArgumentParser(description="Update spread_sufficient_flag for existing CSV.")
    parser.add_argument('--csv', type=str, default='market_summaries.csv', help='Path to CSV to update in-place.')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Ensure new columns exist
    for col in ['volume_7d_buy_yes', 'volume_7d_sell_yes', 'volume_7d_min_buy_sell', 'expected_weekly_profit_cents']:
        if col not in df.columns:
            df[col] = None

    # Recompute fields
    df['volume_7d_min_buy_sell'] = df.apply(compute_min_buy_sell, axis=1)
    df['expected_weekly_profit_cents'] = df.apply(compute_expected_weekly_profit, axis=1)
    df['spread_sufficient_flag'] = df.apply(compute_flag, axis=1)
    df.to_csv(args.csv, index=False)
    print(f"Updated spread_sufficient_flag and backfilled 7d min/expected profit for {len(df)} rows in {args.csv}")


if __name__ == "__main__":
    main()
