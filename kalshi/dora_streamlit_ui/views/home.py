"""
Home Page - Dashboard Overview
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import db_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db_client import ReadOnlyDynamoDBClient


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


def render_pnl_chart(pnl_data: List[Dict], positions: Dict):
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

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_pnl = df['cumulative_pnl'].iloc[-1] if len(df) > 0 else 0
        st.metric("Total Realized P&L", f"${total_pnl:.2f}")
    with col2:
        daily_pnl = df['daily_pnl'].iloc[-1] if len(df) > 0 else 0
        st.metric("Today's P&L", f"${daily_pnl:.2f}")
    with col3:
        # Calculate total exposure
        total_exposure = sum(abs(p.get('net_yes_qty', 0)) for p in positions.values())
        st.metric("Total Exposure", f"{total_exposure} contracts")


def render_exposure_chart(positions: Dict):
    """Render exposure by market chart."""
    st.subheader("Exposure by Market")

    if not positions:
        st.info("No positions found")
        return

    # Create DataFrame
    markets = []
    exposures = []
    for market_id, position in positions.items():
        markets.append(market_id)
        exposures.append(abs(position.get('net_yes_qty', 0)))

    df = pd.DataFrame({'market': markets, 'exposure': exposures})
    df = df.sort_values('exposure', ascending=True)

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=df['exposure'],
        y=df['market'],
        orientation='h',
        marker=dict(color='#636EFA')
    ))

    fig.update_layout(
        height=max(300, len(df) * 30),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Exposure (contracts)",
        yaxis_title="Market"
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


def render_recent_logs(db_client: ReadOnlyDynamoDBClient):
    """Render most recent decision and execution logs."""
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
        st.markdown("#### Most Recent Execution")
        execution = db_client.get_most_recent_execution_log()
        if execution:
            timestamp = execution.get('event_ts', '')
            local_time, time_ago = format_timestamp_with_ago(timestamp)

            st.markdown(f"**Market:** {execution.get('market')}")
            st.markdown(f"**Event:** {execution.get('event_type')}")
            st.markdown(f"**Time:** {local_time}")
            st.markdown(f"*{time_ago}*")

            with st.expander("📋 View Details", expanded=False):
                st.json({
                    'decision_id': execution.get('decision_id'),
                    'order_id': execution.get('order_id'),
                    'side': execution.get('side'),
                    'price': execution.get('price'),
                    'size': execution.get('size'),
                    'status': execution.get('status'),
                })
        else:
            st.info("No recent execution logs")


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
    executions: List[Dict]
):
    """Render active markets table with positions and metrics.

    Args:
        db_client: DynamoDB client (not used for per-market queries anymore)
        positions: Pre-fetched positions dict
        market_configs: Pre-fetched market configs
        trades: Pre-fetched trades (last 30 days)
        decisions: Pre-fetched decision logs (last 1 day)
        executions: Pre-fetched execution logs (last 1 day)
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

    # Index executions by market_id
    executions_by_market: Dict[str, List[Dict]] = {}
    for e in executions:
        mid = e.get('market')
        if mid:
            if mid not in executions_by_market:
                executions_by_market[mid] = []
            executions_by_market[mid].append(e)

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

    # Build table data
    table_data = []
    for config in market_configs:
        market_id = config.get('market_id')
        position = positions.get(market_id, {})

        # Get most recent decision for order book snapshot (from pre-fetched data)
        market_decisions = decisions_by_market.get(market_id, [])
        decision = market_decisions[0] if market_decisions else None
        order_book = decision.get('order_book_snapshot', {}) if decision else {}

        # Get active quotes from decision (no extra DB call)
        active_quotes = None
        if decision:
            target_quotes = decision.get('target_quotes', [])
            bids = [q for q in target_quotes if q.get('side') == 'bid']
            asks = [q for q in target_quotes if q.get('side') == 'ask']
            bids.sort(key=lambda x: x.get('price', 0), reverse=True)
            asks.sort(key=lambda x: x.get('price', 0))
            active_quotes = {'bids': bids, 'asks': asks}

        # Format our bids/asks with indicators for best bid/ask
        our_bid = 'N/A'
        our_ask = 'N/A'
        if active_quotes:
            if active_quotes['bids']:
                our_best_bid = active_quotes['bids'][0]
                our_bid_price = our_best_bid.get('price', 0)
                market_best_bid = order_book.get('best_bid', 0)

                # For bids: higher is better (we're at or above market best bid)
                # Green check if our bid >= market best bid, red X otherwise
                if market_best_bid:
                    is_best_or_better = our_bid_price >= (market_best_bid - 0.0001)
                    indicator = "🟢 ✓" if is_best_or_better else "🔴 ✗"
                else:
                    indicator = "🟢 ✓"  # No competition
                our_bid = f"{indicator} ${our_bid_price:.3f} ({our_best_bid.get('size', 0)})"

            if active_quotes['asks']:
                our_best_ask = active_quotes['asks'][0]
                our_ask_price = our_best_ask.get('price', 0)
                market_best_ask = order_book.get('best_ask', 0)

                # For asks: lower is better (we're at or below market best ask)
                # Green check if our ask <= market best ask, red X otherwise
                if market_best_ask:
                    is_best_or_better = our_ask_price <= (market_best_ask + 0.0001)
                    indicator = "🟢 ✓" if is_best_or_better else "🔴 ✗"
                else:
                    indicator = "🟢 ✓"  # No competition
                our_ask = f"{indicator} ${our_ask_price:.3f} ({our_best_ask.get('size', 0)})"

        # Calculate 24h changes (using pre-fetched trades)
        market_trades = trades_by_market.get(market_id, [])
        pos_change_24h = calculate_24h_change(market_trades, market_id, 'position')
        pnl_change_24h = calculate_24h_change(market_trades, market_id, 'pnl')

        # Calculate 24h filled contracts from pre-fetched trades
        filled_24h = 0
        most_recent_fill_ts = None
        for t in market_trades:
            ts_str = t.get('fill_timestamp') or t.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if most_recent_fill_ts is None:
                        most_recent_fill_ts = ts_str  # First one is most recent (sorted)
                    if ts > cutoff_24h:
                        filled_24h += t.get('size', 0)
                except Exception:
                    pass
        fill_time_ago = get_time_ago(most_recent_fill_ts)

        # Calculate 24h execution count from pre-fetched executions
        market_executions = executions_by_market.get(market_id, [])
        execution_count_24h = 0
        most_recent_exec_ts = None
        for e in market_executions:
            ts_str = e.get('event_ts')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if most_recent_exec_ts is None:
                        most_recent_exec_ts = ts_str  # First one is most recent (sorted)
                    if ts > cutoff_24h:
                        execution_count_24h += 1
                except Exception:
                    pass
        exec_time_ago = get_time_ago(most_recent_exec_ts)

        # Calculate average cost for net position
        avg_cost = None
        net_qty = position.get('net_yes_qty', 0)
        if net_qty > 0:
            avg_cost = position.get('avg_buy_price', 0)
        elif net_qty < 0:
            avg_cost = position.get('avg_sell_price', 0)

        table_data.append({
            'Market': market_id,
            'Best Bid': f"${order_book.get('best_bid', 0.0):.3f}" if order_book.get('best_bid') else 'N/A',
            'Best Ask': f"${order_book.get('best_ask', 0.0):.3f}" if order_book.get('best_ask') else 'N/A',
            'Our Bid': our_bid,
            'Our Ask': our_ask,
            'Spread': f"${order_book.get('spread', 0.0):.3f}" if order_book.get('spread') else 'N/A',
            'Net Position': net_qty,
            'Avg Cost': f"${avg_cost:.3f}" if avg_cost is not None else 'N/A',
            'Position 24h Δ': f"{pos_change_24h:+.0f}",
            'Filled 24h': f"{filled_24h} ({fill_time_ago})",
            'Order Executions 24h': f"{execution_count_24h} ({exec_time_ago})",
            'Realized P&L': f"${position.get('realized_pnl', 0.0):.2f}",
            'P&L 24h Δ': f"${pnl_change_24h:+.2f}",
        })

    # Create DataFrame and display
    df = pd.DataFrame(table_data)

    # Style the dataframe
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            'Market': st.column_config.TextColumn('Market', width='medium'),
            'Best Bid': st.column_config.TextColumn('Best Bid', width='small'),
            'Best Ask': st.column_config.TextColumn('Best Ask', width='small'),
            'Our Bid': st.column_config.TextColumn('Our Bid', width='medium'),
            'Our Ask': st.column_config.TextColumn('Our Ask', width='medium'),
            'Spread': st.column_config.TextColumn('Spread', width='small'),
            'Net Position': st.column_config.NumberColumn('Net Position', width='small'),
            'Avg Cost': st.column_config.TextColumn('Avg Cost', width='small'),
            'Position 24h Δ': st.column_config.TextColumn('Pos Δ 24h', width='small'),
            'Filled 24h': st.column_config.TextColumn('Filled 24h', width='medium'),
            'Order Executions 24h': st.column_config.TextColumn('Execs 24h', width='medium'),
            'Realized P&L': st.column_config.TextColumn('P&L', width='small'),
            'P&L 24h Δ': st.column_config.TextColumn('P&L Δ 24h', width='small'),
        }
    )

    # Add click functionality using selectbox
    st.markdown("---")
    st.markdown("#### View Market Details")
    selected_market = st.selectbox(
        "Select a market to view detailed logs:",
        options=[row['Market'] for row in table_data],
        key='market_selector'
    )

    if selected_market:
        if st.button("View Market Deep Dive", type="primary"):
            st.session_state['selected_market'] = selected_market
            st.session_state['navigate_to_deep_dive'] = True
            st.rerun()


def render(environment: str, region: str):
    """Render the home page."""
    st.title("🏠 Dora Bot Dashboard")
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
        positions = db_client.get_positions()
        market_configs = db_client.get_all_market_configs(enabled_only=True)
        pnl_data = db_client.get_pnl_over_time(days=30)
        # Load all trades (at least 30 days) to properly calculate realized P&L with cost basis
        trades = db_client.get_recent_trades(days=30)
        # Pre-fetch decision logs and execution logs for the active markets table
        decisions = db_client.get_recent_decision_logs(days=1)
        executions = db_client.get_recent_execution_logs(days=1)

    # Layout: Top row - Charts
    col1, col2 = st.columns([2, 1])

    with col1:
        render_pnl_chart(pnl_data, positions)

    with col2:
        render_exposure_chart(positions)

    st.markdown("---")

    # Middle row - Recent logs
    render_recent_logs(db_client)

    st.markdown("---")

    # Bottom row - Active markets table
    render_active_markets_table(db_client, positions, market_configs, trades, decisions, executions)

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
