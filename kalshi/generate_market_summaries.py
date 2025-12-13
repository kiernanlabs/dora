"""
Generate market summary statistics for open events/markets.

This script fetches open events, iterates through their open markets,
calculates the same summary metrics used by the Streamlit app (profits at
current vs. optimal spreads and required spreads over 7d/60d), and writes the
results to a CSV.

Usage:
    python generate_market_summaries.py [--prod] [--trade-limit 1000] [--risk-k 0.25] [--output market_summaries.csv] [--include-info-risk] [--max-markets 100]
"""
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import math

from kalshi_service import KalshiService

MAKER_FEE_RATE = 0.0175
TAKER_FEE_RATE = 0.07

def filter_df_by_days(df: pd.DataFrame, days: float) -> pd.DataFrame:
    """Return dataframe filtered to rows within the last N days."""
    if days <= 0 or 'timestamp' not in df.columns:
        return df
    cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    return df[df['timestamp'] >= cutoff_time]


def extract_spread(orderbook: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], List, List]:
    """Return best bid/ask, current spread, and orderbook levels."""
    orderbook_data = orderbook.get('orderbook', {}) if orderbook else {}
    yes_bids = orderbook_data.get('yes', [])
    no_bids = orderbook_data.get('no', [])

    if not yes_bids or not no_bids:
        return None, None, None, yes_bids, no_bids

    best_bid = max([x[0] for x in yes_bids]) if yes_bids else None
    yes_asks_from_no = [100 - x[0] for x in no_bids] if no_bids else []
    best_ask = min(yes_asks_from_no) if yes_asks_from_no else None
    current_spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None

    return best_bid, best_ask, current_spread, yes_bids, no_bids


def fee_per_contract_cents(price_cents: float, rate: float = 0.07) -> float:
    """Kalshi fee per contract in cents using the provided rate (taker default).

    Uses the exact calculated fee (no rounding up) since rounding is reimbursed.
    """
    price_dollars = price_cents / 100 if price_cents is not None else 0.5
    fee_cents = rate * price_dollars * (1 - price_dollars) * 100
    return fee_cents


def calculate_liquidation_price(volume: int, side: str, orderbook_levels: List[List[float]]) -> Tuple[float, float]:
    """Average price and total value to liquidate a position through the orderbook."""
    if volume <= 0 or not orderbook_levels:
        return 0.0, 0.0

    remaining = volume
    total_value = 0.0

    for price, available_qty in orderbook_levels:
        if remaining <= 0:
            break
        fill_qty = min(remaining, available_qty)
        total_value += fill_qty * price
        remaining -= fill_qty

    filled = volume - remaining
    avg_price = total_value / filled if filled > 0 else 0.0
    return avg_price, total_value


def calculate_profit_at_spread(
    df_filtered: pd.DataFrame,
    bid_price: float,
    ask_price: float,
    sorted_bids: List[List[float]],
    sorted_asks: List[List[float]],
) -> Optional[float]:
    """Compute total P&L (cents) for a given bid/ask using historical fills and orderbook depth."""
    if bid_price is None or ask_price is None or df_filtered is None or df_filtered.empty:
        return None

    if 'action' not in df_filtered.columns or 'adjusted_price' not in df_filtered.columns or 'count' not in df_filtered.columns:
        return None

    bid_fills = df_filtered[(df_filtered['action'] == 'Sell Yes') & (df_filtered['adjusted_price'] <= bid_price)]
    bid_volume = bid_fills['count'].sum() if len(bid_fills) > 0 else 0

    ask_fills = df_filtered[(df_filtered['action'] == 'Buy Yes') & (df_filtered['adjusted_price'] >= ask_price)]
    ask_volume = ask_fills['count'].sum() if len(ask_fills) > 0 else 0

    net_position = bid_volume - ask_volume
    matched_volume = min(bid_volume, ask_volume)
    realized_pnl_gross = matched_volume * (ask_price - bid_price) if matched_volume > 0 else 0
    maker_fee_bid = fee_per_contract_cents(bid_price, rate=MAKER_FEE_RATE) * matched_volume
    maker_fee_ask = fee_per_contract_cents(ask_price, rate=MAKER_FEE_RATE) * matched_volume
    realized_pnl = realized_pnl_gross - (maker_fee_bid + maker_fee_ask)

    if net_position > 0:
        position_cost = net_position * bid_price
        liquidation_avg_price, liquidation_value = calculate_liquidation_price(net_position, 'sell', sorted_bids)
        taker_fee = fee_per_contract_cents(liquidation_avg_price, rate=TAKER_FEE_RATE) * net_position
        unrealized_pnl = (liquidation_value - taker_fee) - position_cost
    elif net_position < 0:
        position_value = abs(net_position) * ask_price
        liquidation_avg_price, liquidation_cost = calculate_liquidation_price(abs(net_position), 'buy', sorted_asks)
        taker_fee = fee_per_contract_cents(liquidation_avg_price, rate=TAKER_FEE_RATE) * abs(net_position)
        unrealized_pnl = position_value - (liquidation_cost + taker_fee)
    else:
        unrealized_pnl = 0

    return realized_pnl + unrealized_pnl


def optimize_profit(df_filtered: pd.DataFrame, sorted_bids: List[List[float]], sorted_asks: List[List[float]]) -> Dict[str, Any]:
    """Search for profit-maximizing bid/ask pair (returns cents)."""
    best_profit = float('-inf')
    best_bid_opt = None
    best_ask_opt = None

    for test_bid in range(0, 100, 1):
        for test_ask in range(test_bid + 1, 101, 1):
            profit_cents = calculate_profit_at_spread(df_filtered, test_bid, test_ask, sorted_bids, sorted_asks)
            if profit_cents is None:
                continue
            if profit_cents > best_profit:
                best_profit = profit_cents
                best_bid_opt = test_bid
                best_ask_opt = test_ask

    if best_profit == float('-inf'):
        return {}

    return {
        'profit_cents': best_profit,
        'bid': best_bid_opt,
        'ask': best_ask_opt
    }


def compute_summary_metrics(df: pd.DataFrame, orderbook: Dict[str, Any], risk_aversion_k: float, service: KalshiService) -> Dict[int, Dict[str, Any]]:
    """Compute summary metrics for 7/60 day horizons."""
    best_bid, best_ask, current_spread, yes_bids, no_bids = extract_spread(orderbook)
    if best_bid is None or best_ask is None or current_spread is None:
        return {}

    sorted_bids = sorted(yes_bids, key=lambda x: x[0], reverse=True)
    sorted_asks = sorted([[100 - x[0], x[1]] for x in no_bids], key=lambda x: x[0]) if no_bids else []

    summary: Dict[int, Dict[str, Any]] = {}
    for lookback in (7, 60):
        df_filtered = filter_df_by_days(df, lookback)

        risk_metrics = service.calculate_trade_risk_metrics(df_filtered, risk_aversion_k=risk_aversion_k)
        required_spread = risk_metrics.get('required_full_spread')

        current_profit_cents = calculate_profit_at_spread(df_filtered, best_bid, best_ask, sorted_bids, sorted_asks)
        optimal = optimize_profit(df_filtered, sorted_bids, sorted_asks)

        midpoint_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else (df_filtered['adjusted_price'].mean() if 'adjusted_price' in df_filtered.columns else 50)
        maker_fee_cents = fee_per_contract_cents(midpoint_price, rate=MAKER_FEE_RATE)
        roundtrip_fee_cents = maker_fee_cents * 2  # assume maker both legs for required spread

        summary[lookback] = {
            'current_profit_cents': current_profit_cents,
            'optimal_profit_cents': optimal.get('profit_cents') if optimal else None,
            'optimal_bid': optimal.get('bid') if optimal else None,
            'optimal_ask': optimal.get('ask') if optimal else None,
            'current_spread': current_spread,
            'required_full_spread': required_spread,
            'required_full_spread_with_fees': (required_spread + roundtrip_fee_cents) if required_spread is not None else None,
            'roundtrip_fee_cents': roundtrip_fee_cents,
            'maker_fee_cents': maker_fee_cents
        }

    return summary


def fetch_events(service: KalshiService, status: str = "open", page_size: int = 200) -> List[Dict[str, Any]]:
    """Fetch events with pagination."""
    events: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params = {'status': status, 'limit': page_size}
        if cursor:
            params['cursor'] = cursor

        data = service.client.get("/trade-api/v2/events", params=params)
        events.extend(data.get('events', []))

        cursor = data.get('cursor') or data.get('next_cursor')
        if not cursor:
            break

    return events


def fetch_markets_for_event(service: KalshiService, event_ticker: str, status: str = "open", page_size: int = 200) -> List[Dict[str, Any]]:
    """Fetch markets for a given event with pagination."""
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params = {'event_ticker': event_ticker, 'status': status, 'limit': page_size}
        if cursor:
            params['cursor'] = cursor

        data = service.client.get("/trade-api/v2/markets", params=params)
        markets.extend(data.get('markets', []))

        cursor = data.get('cursor') or data.get('next_cursor')
        if not cursor:
            break

    return markets


def summarize_market(
    service: KalshiService,
    market: Dict[str, Any],
    orderbook: Dict[str, Any],
    trade_limit: int,
    risk_aversion_k: float,
    include_info_risk: bool,
) -> Dict[str, Any]:
    """Compute summary metrics for a single market."""
    ticker = market.get('ticker')
    market_title = market.get('title')

    trades_df = service.get_trades_dataframe(ticker=ticker, limit=trade_limit)
    if trades_df.empty:
        return {
            'market_ticker': ticker,
            'market_title': market_title,
            'status': 'no trades'
        }

    summary_metrics = compute_summary_metrics(trades_df, orderbook, risk_aversion_k, service)

    # 7d volume
    volume_7d_contracts = None
    volume_7d_orders = None
    df_7d = filter_df_by_days(trades_df, 7)
    buy_yes_7d = None
    sell_yes_7d = None
    min_buy_sell_7d = None
    if not df_7d.empty:
        if 'count' in df_7d.columns:
            volume_7d_contracts = df_7d['count'].sum()
            if 'action' in df_7d.columns:
                buy_yes_7d = df_7d[df_7d['action'] == 'Buy Yes']['count'].sum()
                sell_yes_7d = df_7d[df_7d['action'] == 'Sell Yes']['count'].sum()
                min_buy_sell_7d = min(buy_yes_7d, sell_yes_7d)
        volume_7d_orders = len(df_7d)

    info_risk_prob = None
    info_risk_rationale = None
    if include_info_risk:
        yes_bid = market.get('yes_bid', 0)
        yes_ask = market.get('yes_ask', 0)
        current_price = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else 50

        info_risk = service.assess_information_risk(
            market_title=market_title or '',
            current_price=current_price,
            market_subtitle=market.get('subtitle'),
            rules=market.get('rules')
        )
        info_risk_prob = info_risk.get('probability')
        info_risk_rationale = info_risk.get('rationale')

    row = {
        'market_ticker': ticker,
        'market_title': market_title,
        'status': market.get('status'),
        'current_spread': summary_metrics.get(7, {}).get('current_spread')
    }

    row['volume_7d_contracts'] = volume_7d_contracts
    row['volume_7d_orders'] = volume_7d_orders
    row['volume_7d_buy_yes'] = buy_yes_7d
    row['volume_7d_sell_yes'] = sell_yes_7d
    row['volume_7d_min_buy_sell'] = min_buy_sell_7d

    current_spread_value = summary_metrics.get(7, {}).get('current_spread')
    req_7 = summary_metrics.get(7, {}).get('required_full_spread')
    req_60 = summary_metrics.get(60, {}).get('required_full_spread')
    spread_sufficient = None

    for lookback in (7, 60):
        data = summary_metrics.get(lookback, {})
        row.update({
            f'{lookback}d_profit_current_usd': (data.get('current_profit_cents') / 100) if data.get('current_profit_cents') is not None else None,
            f'{lookback}d_profit_optimal_usd': (data.get('optimal_profit_cents') / 100) if data.get('optimal_profit_cents') is not None else None,
            f'{lookback}d_optimal_bid': data.get('optimal_bid'),
            f'{lookback}d_optimal_ask': data.get('optimal_ask'),
            f'{lookback}d_current_spread': data.get('current_spread'),
            f'{lookback}d_required_spread': data.get('required_full_spread'),
            f'{lookback}d_required_spread_with_fees': data.get('required_full_spread_with_fees'),
            f'{lookback}d_roundtrip_fee_cents': data.get('roundtrip_fee_cents'),
        })

    if current_spread_value is not None and req_7 is not None and req_60 is not None:
        # Immediately fail if current spread is 2¢ or tighter
        if current_spread_value <= 2.0:
            spread_sufficient = 0
        else:
            # 1 if both 7d and 60d required spreads are <= 50% of current spread
            spread_sufficient = int(
                (req_7 <= 0.5 * current_spread_value) and (req_60 <= 0.5 * current_spread_value)
            )

    row['spread_sufficient_flag'] = spread_sufficient
    if current_spread_value is not None and min_buy_sell_7d is not None:
        row['expected_weekly_profit_cents'] = min_buy_sell_7d * current_spread_value
    else:
        row['expected_weekly_profit_cents'] = None

    if include_info_risk:
        row['info_risk_probability'] = info_risk_prob
        row['info_risk_rationale'] = info_risk_rationale

    return row


def main():
    parser = argparse.ArgumentParser(description="Generate market summary CSV for open events.")
    parser.add_argument('--prod', action='store_true', help='Use production environment (default: demo).')
    parser.add_argument('--trade-limit', type=int, default=1000, help='Number of trades to fetch per market.')
    parser.add_argument('--risk-k', type=float, default=0.25, help='Risk aversion constant k.')
    parser.add_argument('--output', type=str, default='market_summaries.csv', help='Output CSV path.')
    parser.add_argument('--include-info-risk', action='store_true', help='Call OpenAI for information risk per market.')
    parser.add_argument('--max-markets', type=int, default=0, help='Maximum number of markets to process (0 = no limit).')
    parser.add_argument('--batch-size', type=int, default=100, help='Write progress to CSV after this many new markets.')
    args = parser.parse_args()

    use_demo = not args.prod
    service = KalshiService(use_demo=use_demo)
    max_markets = args.max_markets if args.max_markets and args.max_markets > 0 else None
    existing_df = None
    existing_keys = set()

    if os.path.exists(args.output):
        try:
            existing_df = pd.read_csv(args.output)
            if 'market_ticker' in existing_df.columns:
                existing_keys = set(
                    existing_df.apply(
                        lambda r: f"{r.get('event_ticker', '')}/{r.get('market_ticker', '')}", axis=1
                    )
                )
                print(f"Loaded existing file {args.output} with {len(existing_df)} rows; will skip duplicates.")
        except Exception as exc:
            print(f"Warning: could not read existing output file {args.output}: {exc}")

    print(f"Fetching open events ({'demo' if use_demo else 'prod'} environment)...")
    events = fetch_events(service, status="open")
    print(f"Found {len(events)} events")

    rows = []
    processed_markets = 0
    skipped_markets = 0
    total_events = len(events)
    if existing_keys:
        print(f"Previously processed markets: {len(existing_keys)}")
    for event in events:
        event_ticker = event.get('ticker') or event.get('event_ticker')
        event_title = event.get('title')
        if not event_ticker:
            continue

        markets = fetch_markets_for_event(service, event_ticker=event_ticker, status="open")
        print(f"- {event_ticker} ({event_title}): {len(markets)} markets")

        for market in markets:
            key = f"{event_ticker}/{market.get('ticker')}"
            if key in existing_keys:
                skipped_markets += 1
                continue

            try:
                orderbook = service.get_orderbook(market.get('ticker'))
                summary_row = summarize_market(
                    service=service,
                    market=market,
                    orderbook=orderbook,
                    trade_limit=args.trade_limit,
                    risk_aversion_k=args.risk_k,
                    include_info_risk=args.include_info_risk,
                )
                summary_row['event_ticker'] = event_ticker
                summary_row['event_title'] = event_title
                rows.append(summary_row)
            except Exception as exc:
                rows.append({
                    'event_ticker': event_ticker,
                    'event_title': event_title,
                    'market_ticker': market.get('ticker'),
                    'market_title': market.get('title'),
                    'status': f'error: {exc}'
                })

            processed_markets += 1
            current_index = processed_markets + skipped_markets
            print(f"Processed {current_index} markets so far ({processed_markets} new, {skipped_markets} skipped) across {total_events} events")

            # Batch write progress
            if args.batch_size > 0 and processed_markets % args.batch_size == 0:
                temp_new_df = pd.DataFrame(rows) if rows else pd.DataFrame()
                if existing_df is not None:
                    combined_df = pd.concat([existing_df, temp_new_df], ignore_index=True) if not temp_new_df.empty else existing_df
                else:
                    combined_df = temp_new_df
                combined_df.to_csv(args.output, index=False)
                print(f"Checkpoint: wrote progress through {processed_markets} new markets to {args.output}")
            if max_markets and processed_markets >= max_markets:
                print(f"Reached max markets limit ({max_markets}); stopping early.")
                break

        if max_markets and processed_markets >= max_markets:
            break

    if not rows and existing_df is not None:
        print("No new rows to write; existing file left unchanged.")
        return

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not new_df.empty else existing_df
    else:
        combined_df = new_df

    combined_df.to_csv(args.output, index=False)
    print(f"Wrote {len(new_df)} new rows; total rows now {len(combined_df)} in {args.output}")


if __name__ == "__main__":
    main()
