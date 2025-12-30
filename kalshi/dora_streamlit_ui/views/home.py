"""
Home Page - Dashboard Overview
"""
import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import sys
import os
import time
import logging

# Add parent directory to path to import db_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db_client import ReadOnlyDynamoDBClient

logger = logging.getLogger(__name__)


def calculate_fill_fee(price: float, count: int) -> float:
    """Calculate the fee for a fill using Kalshi's fee formula.

    Formula: 0.0175 × C × P × (1-P)
    where P = price in dollars (0.50 for 50 cents)
          C = number of contracts

    Args:
        price: Price per contract in dollars (e.g., 0.50 for 50 cents)
        count: Number of contracts traded

    Returns:
        Fee amount in dollars
    """
    return 0.0175 * count * price * (1 - price)


def to_local_time(timestamp_str: str) -> str:
    """Convert ISO timestamp to local time string."""
    try:
        if 'Z' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        elif '+' in timestamp_str or ('-' in timestamp_str.split('T')[1] if 'T' in timestamp_str else False):
            dt = datetime.fromisoformat(timestamp_str)
        else:
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

        local_dt = dt.astimezone()
        return local_dt.strftime('%Y-%m-%d %I:%M:%S %p')
    except Exception:
        return timestamp_str


def get_kalshi_market_url(market_id: str) -> str:
    """Generate Kalshi market URL from market ID."""
    return f"https://kalshi.com/markets/{market_id}"


def render_pnl_chart(pnl_data: List[Dict], positions: Dict, trades: List[Dict] = None,
                     unrealized_worst: float = 0.0, unrealized_best: float = 0.0,
                     active_bids_count: int = 0, active_bids_qty: int = 0,
                     active_asks_count: int = 0, active_asks_qty: int = 0,
                     fees_today: float = 0.0, balance_data: Dict = None):
    """Render P&L over time chart."""
    st.subheader("P&L Over Time")

    if not pnl_data:
        st.info("No P&L data available")
        return

    # Create DataFrame
    df = pd.DataFrame(pnl_data)

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add cumulative P&L line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_pnl'],
        name='Cumulative P&L',
        mode='lines+markers',
        line=dict(color='#00CC96', width=2),
        marker=dict(size=6)
    ))

    # Add daily P&L bars
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['daily_pnl'],
        name='Daily P&L',
        marker=dict(color=['#EF553B' if x < 0 else '#00CC96' for x in df['daily_pnl']])
    ))

    # Update layout
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, width='stretch')

    # Display summary metrics - Row 1: P&L and balance metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        # Display cash balance
        if balance_data:
            cash_balance = balance_data.get('balance', 0.0)
            payout = balance_data.get('payout', 0.0)
            total_balance = cash_balance + payout
            st.metric("Cash Balance", f"${total_balance:.2f}",
                      help=f"Balance: ${cash_balance:.2f} | Payout: ${payout:.2f}")
        else:
            st.metric("Cash Balance", "N/A", help="Balance not available")
    with col2:
        total_pnl = df['cumulative_pnl'].iloc[-1] if len(df) > 0 else 0
        st.metric("Total Realized P&L", f"${total_pnl:.2f}")
    with col3:
        daily_pnl = df['daily_pnl'].iloc[-1] if len(df) > 0 else 0
        st.metric("Today's P&L", f"${daily_pnl:.2f}")
    with col4:
        st.metric("Fees Today", f"${fees_today:.2f}",
                  help="Total fees paid on trades today")
    with col5:
        st.metric("Unrealized (Worst)", f"${unrealized_worst:+.2f}",
                  help="If we exit all positions at current market best bid/ask")
    with col6:
        st.metric("Unrealized (Best)", f"${unrealized_best:+.2f}",
                  help="If we exit all positions at our active orders (if competitive) or market best")

    # Row 2: Exposure metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_exposure = sum(abs(p.get('net_yes_qty', 0)) for p in positions.values())
        st.metric("Total Exposure", f"{total_exposure} contracts")
    with col2:
        st.metric(f"Active Bids ({active_bids_qty} cts)", f"{active_bids_count} markets",
                  help="Markets where our bid is at or above market best bid")
    with col3:
        st.metric(f"Active Asks ({active_asks_qty} cts)", f"{active_asks_count} markets",
                  help="Markets where our ask is at or below market best ask")
    with col4:
        st.metric(f"Total Active ({active_bids_qty + active_asks_qty} cts)", f"{active_bids_count + active_asks_count} markets")

    # Debug expander showing trade-by-trade P&L breakdown
    if trades:
        with st.expander("Debug: P&L Breakdown by Trade", expanded=False):
            # Recalculate P&L with detailed tracking
            trades_sorted = sorted(trades, key=lambda t: t.get('fill_timestamp') or t.get('timestamp', ''))

            # Track position state per market
            positions_state = {}  # market_id -> {net_yes_qty, avg_buy_price, avg_sell_price, realized_pnl}
            trade_details = []
            global_cumulative_pnl = 0.0  # Track cumulative P&L across ALL markets

            for trade in trades_sorted:
                market_id = trade.get('market_id')
                if not market_id:
                    continue

                # Initialize position for this market if needed
                if market_id not in positions_state:
                    positions_state[market_id] = {
                        'net_yes_qty': 0,
                        'avg_buy_price': 0.0,
                        'avg_sell_price': 0.0,
                        'realized_pnl': 0.0
                    }

                pos = positions_state[market_id]
                side = trade.get('side', '')
                price = trade.get('price', 0.0)
                size = trade.get('size', 0)
                fees = trade.get('fees', 0.0) or 0.0
                date_str = trade.get('date', '')
                timestamp = trade.get('fill_timestamp') or trade.get('timestamp', '')

                # Track state before update
                pnl_before = pos['realized_pnl']
                qty_before = pos['net_yes_qty']
                avg_buy_before = pos['avg_buy_price']
                avg_sell_before = pos['avg_sell_price']

                # Subtract fees from realized P&L on every fill (same as Position.update_from_fill)
                pos['realized_pnl'] -= fees

                # Update position using same logic as Position.update_from_fill()
                if side in ['buy', 'yes']:
                    # Bid fill - buying YES contracts
                    if pos['net_yes_qty'] >= 0:
                        # Adding to long position
                        total_cost = pos['avg_buy_price'] * pos['net_yes_qty'] + price * size
                        pos['net_yes_qty'] += size
                        pos['avg_buy_price'] = total_cost / pos['net_yes_qty'] if pos['net_yes_qty'] > 0 else 0
                    else:
                        # Closing short position - realize P&L
                        close_qty = min(abs(pos['net_yes_qty']), size)
                        realized = (pos['avg_sell_price'] - price) * close_qty
                        pos['realized_pnl'] += realized
                        pos['net_yes_qty'] += size

                        if pos['net_yes_qty'] > 0:
                            pos['avg_buy_price'] = price
                else:
                    # Ask fill - selling YES contracts
                    if pos['net_yes_qty'] <= 0:
                        # Adding to short position
                        total_cost = pos['avg_sell_price'] * abs(pos['net_yes_qty']) + price * size
                        pos['net_yes_qty'] -= size
                        pos['avg_sell_price'] = total_cost / abs(pos['net_yes_qty']) if pos['net_yes_qty'] != 0 else 0
                    else:
                        # Closing long position - realize P&L
                        close_qty = min(pos['net_yes_qty'], size)
                        realized = (price - pos['avg_buy_price']) * close_qty
                        pos['realized_pnl'] += realized
                        pos['net_yes_qty'] -= size

                        if pos['net_yes_qty'] < 0:
                            pos['avg_sell_price'] = price

                # Calculate P&L change and update global cumulative
                pnl_change = pos['realized_pnl'] - pnl_before
                global_cumulative_pnl += pnl_change

                # Check if P&L change is fee-only (no position was closed)
                # This happens when pnl_change equals -fees (within floating point tolerance)
                is_fee_only = abs(pnl_change - (-fees)) < 0.0001 and fees > 0

                # Format cost basis based on position - only show relevant values
                qty_after = pos['net_yes_qty']
                if qty_before > 0:
                    # Had long position - show avg buy
                    avg_buy_before_str = f"${avg_buy_before:.3f}"
                    avg_sell_before_str = 'N/A'
                elif qty_before < 0:
                    # Had short position - show avg sell
                    avg_buy_before_str = 'N/A'
                    avg_sell_before_str = f"${avg_sell_before:.3f}"
                else:
                    # Was flat
                    avg_buy_before_str = 'N/A'
                    avg_sell_before_str = 'N/A'

                if qty_after > 0:
                    # Now long - show avg buy
                    avg_buy_after_str = f"${pos['avg_buy_price']:.3f}"
                    avg_sell_after_str = 'N/A'
                elif qty_after < 0:
                    # Now short - show avg sell
                    avg_buy_after_str = 'N/A'
                    avg_sell_after_str = f"${pos['avg_sell_price']:.3f}"
                else:
                    # Now flat
                    avg_buy_after_str = 'N/A'
                    avg_sell_after_str = 'N/A'

                # Convert side to bid/ask terminology
                side_display = 'bid' if side in ['buy', 'yes'] else 'ask'

                # Add asterisk if P&L change is fee-only
                pnl_change_str = f"${pnl_change:+.2f}{'*' if is_fee_only else ''}"

                trade_details.append({
                    'Date': date_str,
                    'Timestamp': timestamp,
                    'Market': market_id,
                    'Side': side_display,
                    'P&L Change': pnl_change_str,
                    'Cumulative P&L': f"${global_cumulative_pnl:.2f}",
                    'Price': f"${price:.3f}",
                    'Size': size,
                    'Qty Before': qty_before,
                    'Qty After': qty_after,
                    'Avg Buy Before': avg_buy_before_str,
                    'Avg Sell Before': avg_sell_before_str,
                    'Avg Buy After': avg_buy_after_str,
                    'Avg Sell After': avg_sell_after_str,
                })

            # Reverse to show newest first
            trade_details = trade_details[::-1]

            st.caption(f"Showing {len(trade_details)} trades (newest first). * = fee-only (no position closed)")

            if trade_details:
                # Display as dataframe
                trade_df = pd.DataFrame(trade_details)
                st.dataframe(
                    trade_df,
                    hide_index=True,
                    width='stretch',
                    height=400,
                )

                # Also show daily aggregation
                st.markdown("#### Daily P&L Aggregation")
                daily_breakdown = {}  # date -> {'pnl': float, 'trade_count': int, 'contracts': int}
                for detail in trade_details:
                    date = detail['Date']
                    pnl_str = detail['P&L Change'].replace('$', '').replace('+', '').replace('*', '')
                    size = detail.get('Size', 0)
                    try:
                        pnl_val = float(pnl_str)
                        if date not in daily_breakdown:
                            daily_breakdown[date] = {'pnl': 0.0, 'trade_count': 0, 'contracts': 0}
                        daily_breakdown[date]['pnl'] += pnl_val
                        daily_breakdown[date]['trade_count'] += 1
                        daily_breakdown[date]['contracts'] += size
                    except ValueError:
                        pass

                if daily_breakdown:
                    daily_df = pd.DataFrame([
                        {
                            'Date': date,
                            'Daily P&L': f"${data['pnl']:.2f}",
                            'Trade Count': data['trade_count'],
                            'Contracts': data['contracts']
                        }
                        for date, data in sorted(daily_breakdown.items())
                    ])
                    st.dataframe(daily_df, hide_index=True, width='stretch')
            else:
                st.info("No trade details available")


def render_exposure_chart(positions: Dict, market_configs: List[Dict]):
    """Render exposure by market chart."""
    st.subheader("Exposure by Market")

    if not positions:
        st.info("No positions found")
        return

    enabled_market_ids = set()
    event_by_market: Dict[str, str] = {}
    for config in market_configs:
        market_id = config.get('market_id')
        if not market_id:
            continue
        if config.get('enabled', True):
            enabled_market_ids.add(market_id)
        event_by_market[market_id] = config.get('event_ticker') or 'N/A'

    # Create DataFrame
    markets = []
    exposures = []
    events = []
    for market_id, position in positions.items():
        if enabled_market_ids and market_id not in enabled_market_ids:
            continue
        markets.append(market_id)
        exposures.append(abs(position.get('net_yes_qty', 0)))
        events.append(event_by_market.get(market_id, 'N/A'))

    df = pd.DataFrame({'event': events, 'market': markets, 'exposure': exposures})
    df = df[df['exposure'] > 0]
    if df.empty:
        st.info("No exposure found in enabled markets")
        return

    event_totals = (
        df.groupby('event', as_index=False)['exposure']
        .sum()
        .sort_values('exposure', ascending=False)
    )
    event_order = event_totals['event'].tolist()
    df['event'] = pd.Categorical(df['event'], categories=event_order, ordered=True)
    df = df.sort_values(['event', 'exposure'], ascending=[True, False])

    # Create horizontal bar chart, clustered by event via ordering
    fig = go.Figure(go.Bar(
        x=df['exposure'],
        y=df['market'],
        orientation='h',
        customdata=df['event'],
        marker=dict(color='#636EFA'),
        width=0.45,
        hovertemplate="Event: %{customdata}<br>Market: %{y}<br>Exposure: %{x}<extra></extra>",
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Exposure (contracts)",
        yaxis_title="Market",
        bargap=0.45,
        showlegend=False,
        yaxis=dict(
            categoryorder="array",
            categoryarray=df['market'].tolist()[::-1],
            tickfont=dict(size=9),
            automargin=True,
        ),
    )

    st.plotly_chart(fig, width='stretch')


def format_timestamp_with_ago(timestamp_str: str) -> tuple[str, str]:
    """Format timestamp as local time and calculate time ago."""
    try:
        # Parse ISO timestamp - handle both with and without timezone
        if 'Z' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        elif '+' in timestamp_str or timestamp_str.endswith(('00:00', '00')):
            # Already has timezone info
            dt = datetime.fromisoformat(timestamp_str)
        else:
            # No timezone info - assume UTC
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

        # Convert to local timezone
        local_dt = dt.astimezone()
        local_time_str = local_dt.strftime('%Y-%m-%d %I:%M:%S %p %Z')

        # Calculate time ago
        now = datetime.now(timezone.utc)
        delta = now - dt

        if delta.total_seconds() < 0:
            ago_str = "just now"
        elif delta.total_seconds() < 60:
            ago_str = f"{int(delta.total_seconds())} seconds ago"
        elif delta.total_seconds() < 3600:
            ago_str = f"{int(delta.total_seconds() / 60)} minutes ago"
        elif delta.total_seconds() < 86400:
            ago_str = f"{int(delta.total_seconds() / 3600)} hours ago"
        else:
            ago_str = f"{int(delta.total_seconds() / 86400)} days ago"

        return local_time_str, ago_str
    except Exception as e:
        # For debugging, show the error
        return timestamp_str, f"Parse error: {str(e)}"


def get_time_ago(timestamp_str: Optional[str]) -> str:
    """Get just the 'time ago' string from a timestamp."""
    if not timestamp_str:
        return 'N/A'
    try:
        _, ago_str = format_timestamp_with_ago(timestamp_str)
        return ago_str
    except Exception:
        return 'N/A'


def render_recent_logs(db_client: ReadOnlyDynamoDBClient, trades: List[Dict]):
    """Render most recent decision and fill details."""
    st.subheader("Recent Activity")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Most Recent Decision")
        decision = db_client.get_most_recent_decision_log()
        if decision:
            timestamp = decision.get('timestamp', '')
            local_time, time_ago = format_timestamp_with_ago(timestamp)

            st.markdown(f"**Market:** {decision.get('market_id')}")
            st.markdown(f"**Time:** {local_time}")
            st.markdown(f"*{time_ago}*")

            with st.expander("📋 View Details", expanded=False):
                st.json({
                    'decision_id': decision.get('decision_id'),
                    'inventory': decision.get('inventory'),
                    'order_book': decision.get('order_book_snapshot'),
                    'target_quotes': decision.get('target_quotes'),
                })
        else:
            st.info("No recent decision logs")

    with col2:
        st.markdown("#### Most Recent Fill")
        most_recent_trade = None
        most_recent_ts = None
        for trade in trades:
            ts_str = trade.get('fill_timestamp') or trade.get('timestamp')
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception:
                continue
            if most_recent_ts is None or ts > most_recent_ts:
                most_recent_ts = ts
                most_recent_trade = trade

        if most_recent_trade and most_recent_ts:
            timestamp_str = most_recent_trade.get('fill_timestamp') or most_recent_trade.get('timestamp', '')
            local_time, time_ago = format_timestamp_with_ago(timestamp_str)

            st.markdown(f"**Market:** {most_recent_trade.get('market_id')}")
            st.markdown(f"**Side:** {most_recent_trade.get('side')}")
            st.markdown(f"**Price:** {most_recent_trade.get('price')}")
            st.markdown(f"**Size:** {most_recent_trade.get('size')}")
            st.markdown(f"**Time:** {local_time}")
            st.markdown(f"*{time_ago}*")

            with st.expander("📋 View Details", expanded=False):
                st.json({
                    'fill_id': most_recent_trade.get('fill_id'),
                    'order_id': most_recent_trade.get('order_id'),
                    'client_order_id': most_recent_trade.get('client_order_id'),
                    'status': most_recent_trade.get('status'),
                    'pnl_realized': most_recent_trade.get('pnl_realized'),
                    'fees': most_recent_trade.get('fees'),
                })
        else:
            st.info("No recent fills")


def calculate_24h_change(trades: List[Dict], market_id: str, metric: str, current_position: Dict = None) -> float:
    """Calculate 24-hour change for position or P&L.

    For P&L calculation, we need ALL trades (not just last 24h) to properly track cost basis,
    then calculate the realized P&L change in the last 24 hours.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    # For position change, only look at recent trades
    if metric == 'position':
        recent_trades = [
            t for t in trades
            if t.get('market_id') == market_id
            and (t.get('fill_timestamp') or t.get('timestamp'))
            and datetime.fromisoformat((t.get('fill_timestamp') or t.get('timestamp', '')).replace('Z', '+00:00')) > cutoff
        ]

        change = 0
        for trade in recent_trades:
            side = trade.get('side', '')
            size = trade.get('size', 0)
            if side in ['buy', 'yes']:
                change += size
            else:
                change -= size
        return change

    elif metric == 'pnl':
        # Get ALL trades for this market to properly calculate realized P&L
        all_market_trades = [
            t for t in trades
            if t.get('market_id') == market_id
        ]

        # Sort trades chronologically
        all_market_trades.sort(key=lambda t: t.get('fill_timestamp') or t.get('timestamp', ''))

        # Track position state
        net_yes_qty = 0
        avg_buy_price = 0.0
        avg_sell_price = 0.0
        realized_pnl = 0.0
        realized_pnl_24h_ago = 0.0

        for trade in all_market_trades:
            timestamp_str = trade.get('fill_timestamp') or trade.get('timestamp', '')
            if not timestamp_str:
                continue

            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            side = trade.get('side', '')
            price = trade.get('price', 0.0)
            size = trade.get('size', 0)
            fees = trade.get('fees', 0.0) or 0.0

            # Subtract fees from realized P&L on every fill (same as Position.update_from_fill)
            realized_pnl -= fees

            # Update position using same logic as Position.update_from_fill()
            if side in ['buy', 'yes']:
                # Bid fill - buying YES contracts
                if net_yes_qty >= 0:
                    # Adding to long position
                    total_cost = avg_buy_price * net_yes_qty + price * size
                    net_yes_qty += size
                    avg_buy_price = total_cost / net_yes_qty if net_yes_qty > 0 else 0
                else:
                    # Closing short position - realize P&L
                    close_qty = min(abs(net_yes_qty), size)
                    realized = (avg_sell_price - price) * close_qty
                    realized_pnl += realized
                    net_yes_qty += size

                    if net_yes_qty > 0:
                        avg_buy_price = price
            else:
                # Ask fill - selling YES contracts
                if net_yes_qty <= 0:
                    # Adding to short position
                    total_cost = avg_sell_price * abs(net_yes_qty) + price * size
                    net_yes_qty -= size
                    avg_sell_price = total_cost / abs(net_yes_qty) if net_yes_qty != 0 else 0
                else:
                    # Closing long position - realize P&L
                    close_qty = min(net_yes_qty, size)
                    realized = (price - avg_buy_price) * close_qty
                    realized_pnl += realized
                    net_yes_qty -= size

                    if net_yes_qty < 0:
                        avg_sell_price = price
            
            # Save realized P&L from 24h ago
            if timestamp <= cutoff:
                realized_pnl_24h_ago = realized_pnl


        # Return the change in realized P&L over the last 24 hours
        return realized_pnl - realized_pnl_24h_ago

    return 0.0


def render_active_markets_table(
    db_client: ReadOnlyDynamoDBClient,
    positions: Dict,
    market_configs: List[Dict],
    trades: List[Dict],
    decisions: List[Dict],
    open_orders_by_market: Dict[str, List[Dict]],
    open_orders_last_updated: Optional[str] = None,
):
    """Render active markets table with positions and metrics.

    Args:
        db_client: DynamoDB client (not used for per-market queries anymore)
        positions: Pre-fetched positions dict
        market_configs: Pre-fetched market configs
        trades: Pre-fetched trades (last 30 days)
        decisions: Pre-fetched decision logs (last 1 day)
        open_orders_by_market: Pre-fetched open orders grouped by market_id (from DynamoDB state)
        open_orders_last_updated: Timestamp when open orders were last synced with exchange
    """
    st.subheader("Active Markets")

    if not market_configs:
        st.info("No active markets found")
        return

    # Pre-index data by market_id for O(1) lookups
    # Index decisions by market_id (most recent first, already sorted)
    decisions_by_market: Dict[str, List[Dict]] = {}
    for d in decisions:
        mid = d.get('market_id')
        if mid:
            if mid not in decisions_by_market:
                decisions_by_market[mid] = []
            decisions_by_market[mid].append(d)

    # Index trades by market_id
    trades_by_market: Dict[str, List[Dict]] = {}
    for t in trades:
        mid = t.get('market_id')
        if mid:
            if mid not in trades_by_market:
                trades_by_market[mid] = []
            trades_by_market[mid].append(t)

    # Pre-calculate 24h cutoff
    cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)

    # Display open orders sync timestamp if available
    if open_orders_last_updated:
        sync_time_ago = get_time_ago(open_orders_last_updated)
        st.caption(f"Open orders last synced with exchange: {sync_time_ago}")

    # Build table data
    table_data = []
    markets_with_mismatch = []  # Track markets where decision targets don't match live orders

    for config in market_configs:
        market_id = config.get('market_id')
        position = positions.get(market_id, {})
        event_tag = config.get('event_ticker') or 'N/A'

        # Get most recent decision for order book snapshot (from pre-fetched data)
        market_decisions = decisions_by_market.get(market_id, [])
        decision = market_decisions[0] if market_decisions else None
        order_book = decision.get('order_book_snapshot', {}) if decision else {}

        # Get target quotes from decision (what bot intended to quote)
        target_quotes = decision.get('target_quotes', []) if decision else []
        target_bids = [q for q in target_quotes if q.get('side') == 'bid']
        target_asks = [q for q in target_quotes if q.get('side') == 'ask']
        target_bids.sort(key=lambda x: x.get('price', 0), reverse=True)
        target_asks.sort(key=lambda x: x.get('price', 0))

        # Get ACTUAL live orders from DynamoDB state (synced from exchange)
        # Orders have side='yes' (bid) or side='no' (ask) in Kalshi format
        market_orders = open_orders_by_market.get(market_id, [])

        # Separate live orders into bids and asks
        # side='yes' means buying YES = bid, side='no' means buying NO = selling YES = ask
        live_bids = [o for o in market_orders if o.get('side') == 'yes']
        live_asks = [o for o in market_orders if o.get('side') == 'no']
        live_bids.sort(key=lambda x: x.get('price', 0) or 0, reverse=True)
        live_asks.sort(key=lambda x: x.get('price', 0) or 0)

        # Check for mismatch between decision targets and actual live orders
        has_mismatch = False
        mismatch_details = []
        # Compare target vs live - if decision has targets but no live orders, that's a mismatch
        if target_bids and not live_bids:
            has_mismatch = True
            mismatch_details.append(f"Target bid: ${target_bids[0].get('price', 0):.3f} but no live bid")
        if target_asks and not live_asks:
            has_mismatch = True
            mismatch_details.append(f"Target ask: ${target_asks[0].get('price', 0):.3f} but no live ask")

        if has_mismatch:
            markets_with_mismatch.append({
                'market_id': market_id,
                'details': mismatch_details,
                'decision_id': decision.get('decision_id') if decision else None,
            })

        # Format our bids/asks based on LIVE orders (from state table)
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

            # For bids: higher is better (we're at or above market best bid)
            if market_best_bid:
                is_bid_active = our_bid_price >= (market_best_bid - 0.0001)
                indicator = "🟢 ✓" if is_bid_active else "🔴 ✗"
            else:
                is_bid_active = True
                indicator = "🟢 ✓"  # No competition
            our_bid = f"{indicator} ${our_bid_price:.3f} ({our_bid_size})"

        if live_asks:
            best_live_ask = live_asks[0]
            our_ask_price = best_live_ask.get('price', 0)
            our_ask_size = best_live_ask.get('size', 0)

            # For asks: lower is better (we're at or below market best ask)
            if market_best_ask:
                is_ask_active = our_ask_price <= (market_best_ask + 0.0001)
                indicator = "🟢 ✓" if is_ask_active else "🔴 ✗"
            else:
                is_ask_active = True
                indicator = "🟢 ✓"  # No competition
            our_ask = f"{indicator} ${our_ask_price:.3f} ({our_ask_size})"

        # Calculate 24h changes (using pre-fetched trades)
        market_trades = trades_by_market.get(market_id, [])
        pos_change_24h = calculate_24h_change(market_trades, market_id, 'position')
        pnl_change_24h = calculate_24h_change(market_trades, market_id, 'pnl')

        # Calculate 24h filled contracts from pre-fetched trades
        filled_24h = 0
        most_recent_fill_ts = None
        most_recent_fill_dt = None
        for t in market_trades:
            ts_str = t.get('fill_timestamp') or t.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    # Track the most recent fill timestamp
                    if most_recent_fill_dt is None or ts > most_recent_fill_dt:
                        most_recent_fill_dt = ts
                        most_recent_fill_ts = ts_str
                    if ts > cutoff_24h:
                        filled_24h += t.get('size', 0)
                except Exception:
                    pass
        fill_time_ago = get_time_ago(most_recent_fill_ts)

        # Calculate average cost for net position
        avg_cost = None
        net_qty = position.get('net_yes_qty', 0)
        if net_qty > 0:
            avg_cost = position.get('avg_buy_price', 0)
        elif net_qty < 0:
            avg_cost = position.get('avg_sell_price', 0)

        net_qty_display = int(round(net_qty))
        max_position_value = config.get('max_position')
        if max_position_value is None:
            max_yes = config.get('max_inventory_yes')
            max_no = config.get('max_inventory_no')
            if max_yes is not None and max_no is not None and max_yes != max_no:
                max_position_value = f"{int(max_yes)}/{int(max_no)}"
            else:
                max_position_value = max_yes if max_yes is not None else max_no
        if max_position_value is not None and not isinstance(max_position_value, str):
            max_position_value = str(int(max_position_value))

        created_at_value = config.get('created_at')
        if created_at_value:
            if isinstance(created_at_value, datetime):
                created_at_dt = created_at_value
            else:
                try:
                    created_at_dt = datetime.fromisoformat(str(created_at_value).replace('Z', '+00:00'))
                except ValueError:
                    created_at_dt = None
            if created_at_dt:
                if created_at_dt.tzinfo is None:
                    created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
                days_ago = max(0, (datetime.now(timezone.utc) - created_at_dt).days)
                created_at_display = f"{days_ago} days ago"
            else:
                created_at_display = 'N/A'
        else:
            created_at_display = 'N/A'

        avg_cost_display = 'N/A'
        if avg_cost is not None:
            is_profitable_active = False
            if net_qty > 0 and our_ask_price is not None:
                is_profitable_active = is_ask_active and our_ask_price > avg_cost
            elif net_qty < 0 and our_bid_price is not None:
                is_profitable_active = is_bid_active and our_bid_price < avg_cost
            check_mark = " ✅" if is_profitable_active else ""
            avg_cost_display = f"${avg_cost:.3f}{check_mark}"

        # Calculate unrealized P&L (worst case - exit at market best, minus exit fees)
        # For long positions (net_qty > 0): we'd sell at best_bid, so unrealized = (best_bid - avg_buy_price) * qty - fee
        # For short positions (net_qty < 0): we'd buy at best_ask, so unrealized = (avg_sell_price - best_ask) * abs(qty) - fee
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

        # Calculate unrealized P&L (best case - exit at our active orders if competitive, minus exit fees)
        unrealized_pnl_best = None
        if net_qty > 0 and avg_cost is not None:  # Long position - need to sell
            if our_ask_price and is_ask_active:
                # Can exit at our ask
                exit_fee = calculate_fill_fee(our_ask_price, net_qty)
                unrealized_pnl_best = (our_ask_price - avg_cost) * net_qty - exit_fee
            elif order_book.get('best_bid'):
                # Our ask isn't competitive, have to hit the bid
                exit_fee = calculate_fill_fee(order_book.get('best_bid'), net_qty)
                unrealized_pnl_best = (order_book.get('best_bid') - avg_cost) * net_qty - exit_fee
        elif net_qty < 0 and avg_cost is not None:  # Short position - need to buy
            if our_bid_price and is_bid_active:
                # Can exit at our bid
                exit_fee = calculate_fill_fee(our_bid_price, abs(net_qty))
                unrealized_pnl_best = (avg_cost - our_bid_price) * abs(net_qty) - exit_fee
            elif order_book.get('best_ask'):
                # Our bid isn't competitive, have to hit the ask
                exit_fee = calculate_fill_fee(order_book.get('best_ask'), abs(net_qty))
                unrealized_pnl_best = (avg_cost - order_book.get('best_ask')) * abs(net_qty) - exit_fee

        spread_value = order_book.get('spread')
        spread_display = f"${spread_value:.3f}" if spread_value else 'N/A'
        min_spread_value = config.get('min_spread')
        min_spread_display = f"${min_spread_value:.3f}" if min_spread_value is not None else 'N/A'

        table_data.append({
            'Event': event_tag,
            'Market': market_id,
            'Best Bid': f"${order_book.get('best_bid', 0.0):.3f}" if order_book.get('best_bid') else 'N/A',
            'Best Ask': f"${order_book.get('best_ask', 0.0):.3f}" if order_book.get('best_ask') else 'N/A',
            'Our Bid': our_bid,
            'Our Ask': our_ask,
            'Spread': spread_display,
            'Minimum Spread': min_spread_display,
            'Max Position': max_position_value if max_position_value is not None else 'N/A',
            'Config Created': created_at_display,
            'Net Position': net_qty_display,
            'Avg Cost': avg_cost_display,
            'Unrealized P&L (Worst)': f"${unrealized_pnl_worst:+.2f}" if unrealized_pnl_worst is not None else 'N/A',
            'Unrealized P&L (Best)': f"${unrealized_pnl_best:+.2f}" if unrealized_pnl_best is not None else 'N/A',
            'Position 24h Δ': f"{pos_change_24h:+.0f}",
            'Filled 24h': f"{filled_24h} ({fill_time_ago})",
            'Realized P&L': f"${position.get('realized_pnl', 0.0):.2f}",
            'P&L 24h Δ': f"${pnl_change_24h:+.2f}",
        })

    # Display warning if there are mismatches between decision targets and live orders
    if markets_with_mismatch:
        st.error(f"**Order Execution Mismatch Detected** - {len(markets_with_mismatch)} market(s) have target quotes that are not reflected in live orders. This may indicate orders were rejected or the bot is halted.")
        with st.expander("View Mismatch Details", expanded=True):
            for mismatch in markets_with_mismatch:
                st.markdown(f"**{mismatch['market_id']}**: {', '.join(mismatch['details'])}")

    def _event_sort_key(event_value: str) -> tuple[bool, str]:
        return (event_value == 'N/A', event_value or '')

    table_data.sort(
        key=lambda row: (
            _event_sort_key(row.get('Event', 'N/A')),
            -row.get('Net Position', 0),
            row.get('Market', ''),
        )
    )

    # Calculate totals for the summary row
    total_net_position = sum(row['Net Position'] for row in table_data)

    # Sum unrealized P&L (worst) (parse from formatted string)
    total_unrealized_pnl_worst = 0.0
    for row in table_data:
        if row['Unrealized P&L (Worst)'] != 'N/A':
            # Parse "$+1.23" or "$-1.23" format
            val_str = row['Unrealized P&L (Worst)'].replace('$', '').replace('+', '')
            try:
                total_unrealized_pnl_worst += float(val_str)
            except ValueError:
                pass

    # Sum unrealized P&L (best) (parse from formatted string)
    total_unrealized_pnl_best = 0.0
    for row in table_data:
        if row['Unrealized P&L (Best)'] != 'N/A':
            # Parse "$+1.23" or "$-1.23" format
            val_str = row['Unrealized P&L (Best)'].replace('$', '').replace('+', '')
            try:
                total_unrealized_pnl_best += float(val_str)
            except ValueError:
                pass

    # Sum position changes (parse from formatted string)
    total_pos_change = 0.0
    for row in table_data:
        val_str = row['Position 24h Δ'].replace('+', '')
        try:
            total_pos_change += float(val_str)
        except ValueError:
            pass

    # Sum filled contracts (extract number from "X (time ago)" format)
    total_filled = 0
    for row in table_data:
        filled_str = row['Filled 24h'].split('(')[0].strip()
        try:
            total_filled += int(filled_str)
        except ValueError:
            pass

    # Sum realized P&L (parse from formatted string)
    total_realized_pnl = 0.0
    for row in table_data:
        val_str = row['Realized P&L'].replace('$', '')
        try:
            total_realized_pnl += float(val_str)
        except ValueError:
            pass

    # Sum P&L changes (parse from formatted string)
    total_pnl_change = 0.0
    for row in table_data:
        val_str = row['P&L 24h Δ'].replace('$', '').replace('+', '')
        try:
            total_pnl_change += float(val_str)
        except ValueError:
            pass

    # Add total row
    total_row = {
        'Event': '',
        'Market': '**TOTAL**',
        'Best Bid': '',
        'Best Ask': '',
        'Our Bid': '',
        'Our Ask': '',
        'Spread': '',
        'Minimum Spread': '',
        'Max Position': '',
        'Config Created': '',
        'Net Position': total_net_position,
        'Avg Cost': '',
        'Unrealized P&L (Worst)': f"${total_unrealized_pnl_worst:+.2f}" if total_unrealized_pnl_worst != 0 else '$0.00',
        'Unrealized P&L (Best)': f"${total_unrealized_pnl_best:+.2f}" if total_unrealized_pnl_best != 0 else '$0.00',
        'Position 24h Δ': f"{total_pos_change:+.0f}",
        'Filled 24h': f"{total_filled}",
        'Realized P&L': f"${total_realized_pnl:.2f}",
        'P&L 24h Δ': f"${total_pnl_change:+.2f}",
    }
    table_data.append(total_row)

    with st.expander("Debug: Active Markets Totals", expanded=False):
        st.write("Rows before total:", len(table_data) - 1)
        st.write("Totals computed:", {
            "net_position": total_net_position,
            "unrealized_pnl_worst": total_unrealized_pnl_worst,
            "unrealized_pnl_best": total_unrealized_pnl_best,
            "position_change_24h": total_pos_change,
            "filled_24h": total_filled,
            "realized_pnl": total_realized_pnl,
            "pnl_change_24h": total_pnl_change,
        })
        st.write("Total row:", total_row)

    # Create DataFrame and display
    df = pd.DataFrame(table_data)

    def _parse_money_value(value: str) -> Optional[float]:
        if not isinstance(value, str):
            return None
        if not value or value == 'N/A':
            return None
        try:
            return float(value.replace('$', ''))
        except ValueError:
            return None

    def highlight_spread(row: pd.Series) -> List[str]:
        spread_val = _parse_money_value(row.get('Spread', ''))
        min_spread_val = _parse_money_value(row.get('Minimum Spread', ''))
        if spread_val is None or min_spread_val is None:
            return [''] * len(row)
        if spread_val < min_spread_val:
            color = '#F8D7DA'
        elif spread_val <= (min_spread_val + 0.01):
            color = '#E2E3E5'
        else:
            color = '#D4EDDA'
        styles = [''] * len(row)
        styles[row.index.get_loc('Spread')] = f'background-color: {color}'
        return styles

    styled_df = df.style.apply(highlight_spread, axis=1)
    table_height = max(400, int((len(df) + 1) * 32))
    column_order = [
        'Event',
        'Market',
        'Realized P&L',
        'P&L 24h Δ',
        'Unrealized P&L (Worst)',
        'Unrealized P&L (Best)',
        'Net Position',
        'Max Position',
        'Avg Cost',
        'Best Bid',
        'Best Ask',
        'Our Bid',
        'Our Ask',
        'Spread',
        'Minimum Spread',
        'Filled 24h',
        'Config Created',
    ]
    for col in df.columns:
        if col not in column_order:
            column_order.append(col)

    # Style the dataframe
    st.dataframe(
        styled_df,
        width='stretch',
        hide_index=True,
        height=table_height,
        column_order=column_order,
        column_config={
            'Event': st.column_config.TextColumn('Event', width='medium', pinned=True),
            'Market': st.column_config.TextColumn('Market', width='medium', pinned=True),
            'Best Bid': st.column_config.TextColumn('Best Bid', width='small'),
            'Best Ask': st.column_config.TextColumn('Best Ask', width='small'),
            'Our Bid': st.column_config.TextColumn('Our Bid', width='medium'),
            'Our Ask': st.column_config.TextColumn('Our Ask', width='medium'),
            'Spread': st.column_config.TextColumn('Spread', width='small'),
            'Minimum Spread': st.column_config.TextColumn('Min Spread', width='small'),
            'Max Position': st.column_config.TextColumn('Max Pos', width='small'),
            'Config Created': st.column_config.TextColumn('Created', width='medium'),
            'Net Position': st.column_config.NumberColumn('Net Position', width='small'),
            'Avg Cost': st.column_config.TextColumn('Avg Cost', width='small'),
            'Unrealized P&L (Worst)': st.column_config.TextColumn('Unreal (W)', width='small'),
            'Unrealized P&L (Best)': st.column_config.TextColumn('Unreal (B)', width='small'),
            'Position 24h Δ': st.column_config.TextColumn('Pos Δ 24h', width='small'),
            'Filled 24h': st.column_config.TextColumn('Filled 24h', width='medium'),
            'Realized P&L': st.column_config.TextColumn('P&L', width='small'),
            'P&L 24h Δ': st.column_config.TextColumn('P&L Δ 24h', width='small'),
        }
    )
    st.caption(f"Debug: active markets table rows = {len(df)}")

    # Add click functionality using selectbox
    st.markdown("---")
    st.markdown("#### View Market Details")
    selected_market = st.selectbox(
        "Select a market to view detailed logs:",
        options=[row['Market'] for row in table_data if row['Market'] != '**TOTAL**'],
        key='market_selector'
    )

    if selected_market:
        if st.button("View Market Deep Dive", type="primary"):
            st.session_state['selected_market'] = selected_market
            st.session_state['navigate_to_deep_dive'] = True
            st.rerun()


def render(environment: str, region: str):
    """Render the home page."""
    st.title("🧭 Dora Bot Dashboard")
    st.markdown(f"**Environment:** {environment.upper()} | **Region:** {region}")
    st.markdown("---")

    # Initialize DynamoDB client
    db_client = ReadOnlyDynamoDBClient(region=region, environment=environment)

    # Add refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()

    # Fetch data - bulk load everything upfront to avoid N+1 queries
    with st.spinner("Loading data from DynamoDB..."):
        timings: List[tuple[str, float]] = []

        def _timed(label: str, func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timings.append((label, elapsed))
            return result

        overall_start = time.perf_counter()
        positions = _timed("get_positions", db_client.get_positions)
        market_configs = _timed("get_all_market_configs", db_client.get_all_market_configs, enabled_only=True)
        pnl_data = _timed("get_pnl_over_time", db_client.get_pnl_over_time, days=30)
        # Load all trades (at least 30 days) to properly calculate realized P&L with cost basis
        trades = _timed("get_recent_trades", db_client.get_recent_trades, days=30)
        # Pre-fetch decision logs for the active markets table
        decisions = _timed("get_recent_decision_logs", db_client.get_recent_decision_logs, days=1)
        # Fetch open orders from DynamoDB state (synced from exchange by bot)
        open_orders_data = _timed("get_open_orders", db_client.get_open_orders)
        open_orders_by_market = _timed("get_open_orders_by_market", db_client.get_open_orders_by_market)
        open_orders_last_updated = open_orders_data.get('last_updated')
        # Fetch account balance
        balance_data = _timed("get_balance", db_client.get_balance)

        overall_elapsed = time.perf_counter() - overall_start
        timings_str = ", ".join(f"{label}={elapsed:.3f}s" for label, elapsed in timings)
        logger.info("Home page fetch timings: %s (total=%.3fs)", timings_str, overall_elapsed)

    if timings:
        with st.expander("Debug: Load Timings", expanded=False):
            st.write(f"Total load: {overall_elapsed:.3f}s")
            st.dataframe(
                pd.DataFrame(timings, columns=["call", "seconds"]).sort_values("seconds", ascending=False),
                hide_index=True,
                width='stretch',
            )

    # Calculate unrealized P&L metrics (worst case and best case)
    total_unrealized_worst = 0.0
    total_unrealized_best = 0.0

    # Track active bids/asks across all markets
    active_bids_count = 0
    active_bids_qty = 0
    active_asks_count = 0
    active_asks_qty = 0

    # Calculate total fees paid today
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    fees_today = 0.0
    for trade in trades:
        if trade.get('date') == today_str:
            fees_today += trade.get('fees', 0.0) or 0.0

    # Pre-index decisions by market_id
    decisions_by_market: Dict[str, List[Dict]] = {}
    for d in decisions:
        mid = d.get('market_id')
        if mid:
            if mid not in decisions_by_market:
                decisions_by_market[mid] = []
            decisions_by_market[mid].append(d)

    for config in market_configs:
        market_id = config.get('market_id')
        position = positions.get(market_id, {})
        net_qty = position.get('net_yes_qty', 0)

        # Get order book from most recent decision
        market_decisions = decisions_by_market.get(market_id, [])
        decision = market_decisions[0] if market_decisions else None
        order_book = decision.get('order_book_snapshot', {}) if decision else {}

        market_best_bid = order_book.get('best_bid', 0)
        market_best_ask = order_book.get('best_ask', 0)

        # Get our live orders
        market_orders = open_orders_by_market.get(market_id, [])
        live_bids = [o for o in market_orders if o.get('side') == 'yes']
        live_asks = [o for o in market_orders if o.get('side') == 'no']
        live_bids.sort(key=lambda x: x.get('price', 0) or 0, reverse=True)
        live_asks.sort(key=lambda x: x.get('price', 0) or 0)

        our_bid_price = live_bids[0].get('price', 0) if live_bids else None
        our_ask_price = live_asks[0].get('price', 0) if live_asks else None

        # Check if our orders are competitive
        is_bid_active = False
        is_ask_active = False
        if our_bid_price and market_best_bid:
            is_bid_active = our_bid_price >= (market_best_bid - 0.0001)
        elif our_bid_price:
            is_bid_active = True
        if our_ask_price and market_best_ask:
            is_ask_active = our_ask_price <= (market_best_ask + 0.0001)
        elif our_ask_price:
            is_ask_active = True

        # Track active bids/asks for summary metrics
        if is_bid_active and live_bids:
            active_bids_count += 1
            active_bids_qty += sum(o.get('size', 0) for o in live_bids)
        if is_ask_active and live_asks:
            active_asks_count += 1
            active_asks_qty += sum(o.get('size', 0) for o in live_asks)

        # Skip unrealized P&L calculation if no position
        if net_qty == 0:
            continue

        # Get average cost
        avg_cost = None
        if net_qty > 0:
            avg_cost = position.get('avg_buy_price', 0)
        elif net_qty < 0:
            avg_cost = position.get('avg_sell_price', 0)

        if avg_cost is None:
            continue

        # Calculate worst case unrealized P&L (exit at market best, minus exit fees)
        if net_qty > 0:  # Long position
            if market_best_bid:
                exit_fee = calculate_fill_fee(market_best_bid, net_qty)
                unrealized_worst = (market_best_bid - avg_cost) * net_qty - exit_fee
                total_unrealized_worst += unrealized_worst
        else:  # Short position (net_qty < 0)
            if market_best_ask:
                exit_fee = calculate_fill_fee(market_best_ask, abs(net_qty))
                unrealized_worst = (avg_cost - market_best_ask) * abs(net_qty) - exit_fee
                total_unrealized_worst += unrealized_worst

        # Calculate best case unrealized P&L (exit at our active orders if competitive, minus exit fees)
        if net_qty > 0:  # Long position - need to sell
            if our_ask_price and is_ask_active:
                # Can exit at our ask
                exit_fee = calculate_fill_fee(our_ask_price, net_qty)
                unrealized_best = (our_ask_price - avg_cost) * net_qty - exit_fee
                total_unrealized_best += unrealized_best
            elif market_best_bid:
                # Our ask isn't competitive, have to hit the bid
                exit_fee = calculate_fill_fee(market_best_bid, net_qty)
                unrealized_best = (market_best_bid - avg_cost) * net_qty - exit_fee
                total_unrealized_best += unrealized_best
        else:  # Short position (net_qty < 0) - need to buy
            if our_bid_price and is_bid_active:
                # Can exit at our bid
                exit_fee = calculate_fill_fee(our_bid_price, abs(net_qty))
                unrealized_best = (avg_cost - our_bid_price) * abs(net_qty) - exit_fee
                total_unrealized_best += unrealized_best
            elif market_best_ask:
                # Our bid isn't competitive, have to hit the ask
                exit_fee = calculate_fill_fee(market_best_ask, abs(net_qty))
                unrealized_best = (avg_cost - market_best_ask) * abs(net_qty) - exit_fee
                total_unrealized_best += unrealized_best

    # Layout: Top row - Charts
    col1, col2 = st.columns([2, 1])

    with col1:
        render_pnl_chart(pnl_data, positions, trades, total_unrealized_worst, total_unrealized_best,
                         active_bids_count, active_bids_qty, active_asks_count, active_asks_qty,
                         fees_today, balance_data)

    with col2:
        render_exposure_chart(positions, market_configs)

    st.markdown("---")

    # Middle row - Recent logs
    render_recent_logs(db_client, trades)

    st.markdown("---")

    # Bottom row - Active markets table
    render_active_markets_table(
        db_client,
        positions,
        market_configs,
        trades,
        decisions,
        open_orders_by_market,
        open_orders_last_updated,
    )

    # Display risk state
    st.markdown("---")
    with st.expander("⚠️ Risk State"):
        risk_state = db_client.get_risk_state()
        if risk_state:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily P&L", f"${risk_state.get('daily_pnl', 0.0):.2f}")
            with col2:
                halted = risk_state.get('trading_halted', False)
                st.metric("Trading Status", "HALTED" if halted else "ACTIVE")
            with col3:
                if risk_state.get('halt_reason'):
                    st.warning(f"Halt Reason: {risk_state['halt_reason']}")
        else:
            st.info("No risk state data available")
