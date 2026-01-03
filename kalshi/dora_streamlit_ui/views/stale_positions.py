"""
Stale Positions Page - Identifies markets with positions but no recent fills
"""
import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import db_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db_client import ReadOnlyDynamoDBClient


def calculate_fill_fee(price: float, count: int) -> float:
    """Calculate the fee for a fill using Kalshi's fee formula.

    Formula: 0.0175 × C × P × (1-P)
    where P = price in dollars (0.50 for 50 cents)
          C = number of contracts
    """
    return 0.0175 * count * price * (1 - price)


def get_time_ago(timestamp_str: Optional[str]) -> str:
    """Get just the 'time ago' string from a timestamp."""
    if not timestamp_str:
        return 'N/A'
    try:
        # Parse ISO timestamp
        if 'Z' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        elif '+' in timestamp_str or (('-' in timestamp_str) and timestamp_str.rfind('-') > 10):
            dt = datetime.fromisoformat(timestamp_str)
        else:
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

        # Calculate time ago
        now = datetime.now(timezone.utc)
        delta = now - dt

        if delta.total_seconds() < 0:
            return "just now"
        elif delta.total_seconds() < 60:
            return f"{int(delta.total_seconds())} seconds ago"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)} minutes ago"
        elif delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)} hours ago"
        else:
            return f"{int(delta.total_seconds() / 86400)} days ago"
    except Exception:
        return 'N/A'


def render_stale_positions_table(
    positions: Dict,
    market_configs: List[Dict],
    trades: List[Dict],
    decisions_by_market: Dict[str, Dict],
    open_orders_by_market: Dict[str, List[Dict]],
    stale_hours: int = 48
):
    """Render stale positions table with markets that have positions but no recent fills.

    Args:
        positions: Pre-fetched positions dict
        market_configs: Pre-fetched market configs
        trades: Pre-fetched trades
        decisions_by_market: Pre-fetched most recent decision per market
        open_orders_by_market: Pre-fetched open orders grouped by market_id
        stale_hours: Number of hours to consider a position stale (default: 48)
    """
    st.subheader("Stale Positions")

    # Calculate stale cutoff
    stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=stale_hours)

    # Index trades by market_id
    trades_by_market: Dict[str, List[Dict]] = {}
    for t in trades:
        mid = t.get('market_id')
        if mid:
            if mid not in trades_by_market:
                trades_by_market[mid] = []
            trades_by_market[mid].append(t)

    # Build table data for markets with stale positions
    table_data = []
    markets_with_exit_orders = 0
    total_stale_exposure = 0

    for config in market_configs:
        market_id = config.get('market_id')
        position = positions.get(market_id, {})
        net_qty = position.get('net_yes_qty', 0)

        # Skip if no position
        if net_qty == 0:
            continue

        # Find most recent fill for this market
        market_trades = trades_by_market.get(market_id, [])
        most_recent_fill_ts = None
        most_recent_fill_dt = None

        for t in market_trades:
            ts_str = t.get('fill_timestamp') or t.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if most_recent_fill_dt is None or ts > most_recent_fill_dt:
                        most_recent_fill_dt = ts
                        most_recent_fill_ts = ts_str
                except Exception:
                    pass

        # Skip if last fill is not stale (less than stale_hours ago)
        if most_recent_fill_dt and most_recent_fill_dt > stale_cutoff:
            continue

        # Skip if no fills at all (position might be from initial state)
        if most_recent_fill_dt is None:
            # Include these as they're also stale
            pass

        # This is a stale position - include it
        event_tag = config.get('event_ticker') or 'N/A'

        # Get order book from most recent decision
        decision = decisions_by_market.get(market_id)
        order_book = decision.get('order_book_snapshot', {}) if decision else {}

        # Get live orders
        market_orders = open_orders_by_market.get(market_id, [])
        live_bids = [o for o in market_orders if o.get('side') == 'yes']
        live_asks = [o for o in market_orders if o.get('side') == 'no']
        live_bids.sort(key=lambda x: x.get('price', 0) or 0, reverse=True)
        live_asks.sort(key=lambda x: x.get('price', 0) or 0)

        # Format our bids/asks
        our_bid = 'N/A'
        our_ask = 'N/A'
        our_bid_price = None
        our_ask_price = None
        is_bid_active = False
        is_ask_active = False
        market_best_bid = order_book.get('best_bid', 0)
        market_best_ask = order_book.get('best_ask', 0)

        if live_bids:
            best_live_bid = live_bids[0]
            our_bid_price = best_live_bid.get('price', 0)
            our_bid_size = best_live_bid.get('size', 0)

            if market_best_bid:
                is_bid_active = our_bid_price >= (market_best_bid - 0.0001)
                indicator = "🟢 ✓" if is_bid_active else "🔴 ✗"
            else:
                is_bid_active = True
                indicator = "🟢 ✓"
            our_bid = f"{indicator} ${our_bid_price:.3f} ({our_bid_size})"

        if live_asks:
            best_live_ask = live_asks[0]
            our_ask_price = best_live_ask.get('price', 0)
            our_ask_size = best_live_ask.get('size', 0)

            if market_best_ask:
                is_ask_active = our_ask_price <= (market_best_ask + 0.0001)
                indicator = "🟢 ✓" if is_ask_active else "🔴 ✗"
            else:
                is_ask_active = True
                indicator = "🟢 ✓"
            our_ask = f"{indicator} ${our_ask_price:.3f} ({our_ask_size})"

        # Check if we have an active exit order for this position
        has_exit_order = False
        if net_qty > 0:  # Long position - need active ask to exit
            has_exit_order = is_ask_active and our_ask_price is not None
        elif net_qty < 0:  # Short position - need active bid to exit
            has_exit_order = is_bid_active and our_bid_price is not None

        if has_exit_order:
            markets_with_exit_orders += 1

        # Add to total stale exposure
        total_stale_exposure += abs(net_qty)

        # Calculate average cost
        avg_cost = None
        if net_qty > 0:
            avg_cost = position.get('avg_buy_price', 0)
        elif net_qty < 0:
            avg_cost = position.get('avg_sell_price', 0)

        avg_cost_display = 'N/A'
        if avg_cost is not None:
            is_profitable_active = False
            if net_qty > 0 and our_ask_price is not None:
                is_profitable_active = is_ask_active and our_ask_price > avg_cost
            elif net_qty < 0 and our_bid_price is not None:
                is_profitable_active = is_bid_active and our_bid_price < avg_cost
            check_mark = " ✅" if is_profitable_active else ""
            avg_cost_display = f"${avg_cost:.3f}{check_mark}"

        # Calculate unrealized P&L (worst case)
        unrealized_pnl_worst = None
        if net_qty > 0 and avg_cost is not None:
            best_bid = order_book.get('best_bid')
            if best_bid:
                exit_fee = calculate_fill_fee(best_bid, net_qty)
                unrealized_pnl_worst = (best_bid - avg_cost) * net_qty - exit_fee
        elif net_qty < 0 and avg_cost is not None:
            best_ask = order_book.get('best_ask')
            if best_ask:
                exit_fee = calculate_fill_fee(best_ask, abs(net_qty))
                unrealized_pnl_worst = (avg_cost - best_ask) * abs(net_qty) - exit_fee

        # Calculate unrealized P&L (best case)
        unrealized_pnl_best = None
        if net_qty > 0 and avg_cost is not None:
            if our_ask_price and is_ask_active:
                exit_fee = calculate_fill_fee(our_ask_price, net_qty)
                unrealized_pnl_best = (our_ask_price - avg_cost) * net_qty - exit_fee
            elif order_book.get('best_bid'):
                exit_fee = calculate_fill_fee(order_book.get('best_bid'), net_qty)
                unrealized_pnl_best = (order_book.get('best_bid') - avg_cost) * net_qty - exit_fee
        elif net_qty < 0 and avg_cost is not None:
            if our_bid_price and is_bid_active:
                exit_fee = calculate_fill_fee(our_bid_price, abs(net_qty))
                unrealized_pnl_best = (avg_cost - our_bid_price) * abs(net_qty) - exit_fee
            elif order_book.get('best_ask'):
                exit_fee = calculate_fill_fee(order_book.get('best_ask'), abs(net_qty))
                unrealized_pnl_best = (avg_cost - order_book.get('best_ask')) * abs(net_qty) - exit_fee

        fill_time_ago = get_time_ago(most_recent_fill_ts) if most_recent_fill_ts else 'Never'

        # Calculate days since last fill
        days_since_fill = 'N/A'
        if most_recent_fill_dt:
            delta = datetime.now(timezone.utc) - most_recent_fill_dt
            days_since_fill = f"{delta.days}d {delta.seconds // 3600}h"

        table_data.append({
            'Event': event_tag,
            'Market': market_id,
            'Net Position': int(round(net_qty)),
            'Abs Position': abs(int(round(net_qty))),
            'Avg Cost': avg_cost_display,
            'Best Bid': f"${order_book.get('best_bid', 0.0):.3f}" if order_book.get('best_bid') else 'N/A',
            'Best Ask': f"${order_book.get('best_ask', 0.0):.3f}" if order_book.get('best_ask') else 'N/A',
            'Our Bid': our_bid,
            'Our Ask': our_ask,
            'Has Exit Order': '✅ Yes' if has_exit_order else '❌ No',
            'Unrealized P&L (Worst)': f"${unrealized_pnl_worst:+.2f}" if unrealized_pnl_worst is not None else 'N/A',
            'Unrealized P&L (Best)': f"${unrealized_pnl_best:+.2f}" if unrealized_pnl_best is not None else 'N/A',
            'Last Fill': fill_time_ago,
            'Time Since Fill': days_since_fill,
            'Realized P&L': f"${position.get('realized_pnl', 0.0):.2f}",
        })

    # Sort by absolute position (descending)
    table_data.sort(key=lambda x: x['Abs Position'], reverse=True)

    # Display summary metrics
    num_stale_markets = len(table_data)
    pct_with_exit_orders = (markets_with_exit_orders / num_stale_markets * 100) if num_stale_markets > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stale Markets", num_stale_markets,
                  help=f"Markets with positions but no fills in {stale_hours}+ hours")
    with col2:
        st.metric("Total Stale Exposure", f"{total_stale_exposure} contracts",
                  help="Sum of absolute position sizes across all stale markets")
    with col3:
        st.metric("With Active Exit Orders", f"{markets_with_exit_orders} ({pct_with_exit_orders:.1f}%)",
                  help="Markets with competitive orders that would exit the position")
    with col4:
        st.metric("Without Exit Orders", f"{num_stale_markets - markets_with_exit_orders}",
                  help="Stale positions without active exit orders")

    st.markdown("---")

    if not table_data:
        st.success(f"No stale positions found! All markets with positions have had fills within the last {stale_hours} hours.")
        return

    # Create DataFrame and display
    df = pd.DataFrame(table_data)

    # Drop the Abs Position column (only used for sorting)
    df = df.drop(columns=['Abs Position'])

    table_height = max(400, int((len(df) + 1) * 35))

    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        height=table_height,
        column_config={
            'Event': st.column_config.TextColumn('Event', width='small'),
            'Market': st.column_config.TextColumn('Market', width='medium'),
            'Net Position': st.column_config.NumberColumn('Net Position', width='small'),
            'Avg Cost': st.column_config.TextColumn('Avg Cost', width='small'),
            'Best Bid': st.column_config.TextColumn('Best Bid', width='small'),
            'Best Ask': st.column_config.TextColumn('Best Ask', width='small'),
            'Our Bid': st.column_config.TextColumn('Our Bid', width='medium'),
            'Our Ask': st.column_config.TextColumn('Our Ask', width='medium'),
            'Has Exit Order': st.column_config.TextColumn('Exit Order', width='small'),
            'Unrealized P&L (Worst)': st.column_config.TextColumn('Unreal (W)', width='small'),
            'Unrealized P&L (Best)': st.column_config.TextColumn('Unreal (B)', width='small'),
            'Last Fill': st.column_config.TextColumn('Last Fill', width='medium'),
            'Time Since Fill': st.column_config.TextColumn('Time Since', width='small'),
            'Realized P&L': st.column_config.TextColumn('P&L', width='small'),
        }
    )


def render(environment: str, region: str):
    """Render the stale positions page."""
    st.title("⏰ Stale Positions")
    st.markdown(f"**Environment:** {environment.upper()} | **Region:** {region}")
    st.markdown("Markets with positions but no recent fills (48+ hours)")
    st.markdown("---")

    # Initialize DynamoDB client
    db_client = ReadOnlyDynamoDBClient(region=region, environment=environment)

    # Add refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()

    # Fetch data
    with st.spinner("Loading data from DynamoDB..."):
        positions = db_client.get_positions()
        market_configs = db_client.get_all_market_configs(enabled_only=True)
        trades = db_client.get_recent_trades(days=30)

        # Pre-fetch decisions and orders
        market_ids = [c.get('market_id') for c in market_configs if c.get('market_id')]
        decisions_by_market = db_client.get_most_recent_decision_per_market(market_ids=market_ids)
        open_orders_by_market = db_client.get_open_orders_by_market()

    # Render the stale positions table
    render_stale_positions_table(
        positions,
        market_configs,
        trades,
        decisions_by_market,
        open_orders_by_market,
        stale_hours=48
    )
