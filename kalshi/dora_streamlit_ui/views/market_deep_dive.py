"""
Market Deep Dive Page - Detailed view of decision and execution logs
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


def parse_iso_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse an ISO timestamp into a timezone-aware datetime."""
    if not timestamp_str:
        return None
    try:
        if 'Z' in timestamp_str:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if '+' in timestamp_str or ('-' in timestamp_str.split('T')[1] if 'T' in timestamp_str else False):
            return datetime.fromisoformat(timestamp_str)
        return datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def get_kalshi_market_url(market_id: str) -> str:
    """Generate Kalshi market URL from market ID."""
    return f"https://kalshi.com/markets/{market_id}"


def build_live_order_debug(executions: List[Dict]) -> Dict[str, Dict]:
    """Mirror live order detection logic and return intermediate debug data."""
    accepted_orders: Dict[str, Dict] = {}
    for e in executions:
        event_type = e.get('event_type', '')
        order_id = e.get('order_id')
        status = e.get('status')

        if not order_id:
            continue

        if event_type == 'ORDER_RESULT' and status == 'ACCEPTED':
            accepted_orders[order_id] = {
                'side': e.get('side'),
                'price': e.get('price'),
                'size': e.get('size'),
                'event_ts': e.get('event_ts', ''),
            }

    termination_events: List[Dict] = []
    for e in executions:
        event_type = e.get('event_type', '')
        order_id = e.get('order_id')
        status = e.get('status')

        if not order_id:
            continue

        if event_type == 'ORDER_RESULT' and status in ('CANCELLED', 'ALREADY_GONE'):
            termination_events.append({
                'order_id': order_id,
                'event_type': event_type,
                'status': status,
                'event_ts': e.get('event_ts', ''),
            })
        elif event_type == 'FILL':
            termination_events.append({
                'order_id': order_id,
                'event_type': event_type,
                'status': status,
                'event_ts': e.get('event_ts', ''),
            })

    terminated_order_ids = {e['order_id'] for e in termination_events}

    most_recent_by_side: Dict[str, Dict] = {}
    for order_id, details in accepted_orders.items():
        side = details.get('side')
        event_ts = details.get('event_ts', '')

        if not side:
            continue

        if side not in most_recent_by_side:
            most_recent_by_side[side] = {'order_id': order_id, 'event_ts': event_ts}
        else:
            current_ts = most_recent_by_side[side]['event_ts']
            if event_ts > current_ts:
                most_recent_by_side[side] = {'order_id': order_id, 'event_ts': event_ts}

    live_orders: Dict[str, Dict] = {}
    for side, details in most_recent_by_side.items():
        order_id = details['order_id']
        if order_id not in terminated_order_ids:
            live_orders[order_id] = accepted_orders[order_id]

    return {
        'accepted_orders': accepted_orders,
        'termination_events': termination_events,
        'most_recent_by_side': most_recent_by_side,
        'terminated_order_ids': terminated_order_ids,
        'live_orders': live_orders,
    }


def render_decision_logs(db_client: ReadOnlyDynamoDBClient, market_id: str, days: int):
    """Render decision logs for a specific market."""
    st.subheader(f"Decision Logs for {market_id}")

    with st.spinner("Loading decision logs..."):
        decisions = db_client.get_recent_decision_logs(days=days, market_id=market_id)

    if not decisions:
        st.info(f"No decision logs found for {market_id} in the last {days} days")
        return

    st.markdown(f"**Found {len(decisions)} decision logs**")

    # Create summary table
    table_data = []
    for decision in decisions:
        order_book = decision.get('order_book_snapshot', {})
        inventory = decision.get('inventory', {})
        target_quotes = decision.get('target_quotes', [])

        # Extract target bid and ask
        target_bids = [q for q in target_quotes if q.get('side') == 'bid']
        target_asks = [q for q in target_quotes if q.get('side') == 'ask']

        target_bid_str = 'N/A'
        target_ask_str = 'N/A'
        if target_bids:
            best_bid = max(target_bids, key=lambda x: x.get('price', 0))
            target_bid_str = f"${best_bid.get('price', 0):.3f} ({best_bid.get('size', 0)})"
        if target_asks:
            best_ask = min(target_asks, key=lambda x: x.get('price', 999))
            target_ask_str = f"${best_ask.get('price', 0):.3f} ({best_ask.get('size', 0)})"

        table_data.append({
            'Timestamp': to_local_time(decision.get('timestamp', '')),
            'Decision ID': decision.get('decision_id', '')[:30] + '...',
            'Best Bid': f"${order_book.get('best_bid', 0.0):.3f}",
            'Best Ask': f"${order_book.get('best_ask', 0.0):.3f}",
            'Target Bid': target_bid_str,
            'Target Ask': target_ask_str,
            'Spread': f"${order_book.get('spread', 0.0):.3f}",
            'Mid': f"${order_book.get('mid', 0.0):.3f}",
            'Position': inventory.get('net_yes_qty', 0),
            'Num Quotes': decision.get('num_targets', 0),
        })

    df = pd.DataFrame(table_data)

    # Display table
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        height=400
    )

    # Allow drilling into specific decision
    st.markdown("---")
    st.markdown("#### Decision Details")

    selected_idx = st.selectbox(
        "Select a decision to view details:",
        options=range(len(decisions)),
        format_func=lambda i: f"{decisions[i].get('timestamp')} - {decisions[i].get('decision_id', '')[:50]}",
        key=f'decision_selector_{market_id}'
    )

    if selected_idx is not None:
        selected_decision = decisions[selected_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Order Book Snapshot**")
            st.json(selected_decision.get('order_book_snapshot', {}))

            st.markdown("**Inventory**")
            st.json(selected_decision.get('inventory', {}))

        with col2:
            st.markdown("**Target Quotes**")
            target_quotes = selected_decision.get('target_quotes', [])
            if target_quotes:
                quotes_df = pd.DataFrame(target_quotes)
                st.dataframe(quotes_df, width='stretch', hide_index=True)
            else:
                st.info("No target quotes")

            st.markdown("**Metadata**")
            st.json({
                'decision_id': selected_decision.get('decision_id'),
                'bot_run_id': selected_decision.get('bot_run_id'),
                'bot_version': selected_decision.get('bot_version'),
                'timestamp': selected_decision.get('timestamp'),
            })


def render_execution_logs(db_client: ReadOnlyDynamoDBClient, market_id: str, days: int, days_lookback: int = 7):
    """Render execution logs for a specific market."""
    st.subheader(f"Execution Logs for {market_id}")

    with st.spinner("Loading execution logs..."):
        executions = db_client.get_recent_execution_logs(days=days, market_id=market_id)

    if not executions:
        st.info(f"No execution logs found for {market_id} in the last {days} days")
        return

    st.markdown(f"**Found {len(executions)} execution events**")

    # Create summary table
    table_data = []
    for execution in executions:
        # Handle numeric fields properly to avoid Arrow serialization errors
        size_val = execution.get('size')
        latency_val = execution.get('latency_ms')

        table_data.append({
            'Timestamp': to_local_time(execution.get('event_ts', '')),
            'Event Type': execution.get('event_type', ''),
            'Decision ID': execution.get('decision_id', '')[:30] + '...' if execution.get('decision_id') else 'N/A',
            'Order ID': execution.get('order_id', '')[:20] + '...' if execution.get('order_id') else 'N/A',
            'Side': execution.get('side', '-'),
            'Price': f"${execution.get('price', 0.0):.3f}" if execution.get('price') else '-',
            'Size': int(size_val) if size_val is not None else None,
            'Status': execution.get('status', ''),
            'Latency (ms)': int(latency_val) if latency_val is not None else None,
        })

    df = pd.DataFrame(table_data)

    # Add color coding for event types
    def highlight_event_type(row):
        event_type = row['Event Type']
        if event_type == 'FILL':
            return ['background-color: #d4edda'] * len(row)
        elif event_type == 'ORDER_RESULT' and row['Status'] == 'REJECTED':
            return ['background-color: #f8d7da'] * len(row)
        elif event_type == 'ORDER_PLACE':
            return ['background-color: #d1ecf1'] * len(row)
        elif event_type == 'ORDER_CANCEL':
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    # Display table with styling
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        height=400
    )

    # Event type legend
    st.markdown("""
    **Event Types:**
    - 🟦 ORDER_PLACE: Order placement attempt
    - 🟩 FILL: Trade execution
    - 🟨 ORDER_CANCEL: Order cancellation
    - 🟥 ORDER_RESULT (REJECTED): Failed order
    """)

    with st.expander("🔎 Live Order Debug", expanded=False):
        debug_data = build_live_order_debug(executions)

        accepted_orders = debug_data['accepted_orders']
        termination_events = debug_data['termination_events']
        most_recent_by_side = debug_data['most_recent_by_side']
        terminated_order_ids = debug_data['terminated_order_ids']
        live_orders = debug_data['live_orders']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accepted Orders", len(accepted_orders))
        with col2:
            st.metric("Terminated Orders", len(terminated_order_ids))
        with col3:
            st.metric("Most Recent per Side", len(most_recent_by_side))
        with col4:
            st.metric("Live Orders", len(live_orders))

        st.markdown("**Accepted Orders (ORDER_RESULT = ACCEPTED)**")
        if accepted_orders:
            accepted_df = pd.DataFrame(
                [
                    {
                        'Order ID': order_id,
                        'Side': details.get('side'),
                        'Price': details.get('price'),
                        'Size': details.get('size'),
                        'Event TS': to_local_time(details.get('event_ts', '')),
                    }
                    for order_id, details in accepted_orders.items()
                ]
            )
            st.dataframe(accepted_df, width='stretch', hide_index=True)
        else:
            st.info("No accepted orders found.")

        st.markdown("**Termination Events (CANCELLED / ALREADY_GONE / FILL)**")
        if termination_events:
            termination_df = pd.DataFrame(
                [
                    {
                        'Order ID': e.get('order_id'),
                        'Event Type': e.get('event_type'),
                        'Status': e.get('status'),
                        'Event TS': to_local_time(e.get('event_ts', '')),
                    }
                    for e in termination_events
                ]
            )
            st.dataframe(termination_df, width='stretch', hide_index=True)
        else:
            st.info("No termination events found.")

        st.markdown("**Most Recent Accepted Order by Side**")
        if most_recent_by_side:
            most_recent_df = pd.DataFrame(
                [
                    {
                        'Side': side,
                        'Order ID': details.get('order_id'),
                        'Event TS': to_local_time(details.get('event_ts', '')),
                        'Terminated': details.get('order_id') in terminated_order_ids,
                    }
                    for side, details in most_recent_by_side.items()
                ]
            )
            st.dataframe(most_recent_df, width='stretch', hide_index=True)
        else:
            st.info("No recent orders by side found.")

        st.markdown("**Final Live Orders**")
        if live_orders:
            live_orders_df = pd.DataFrame(
                [
                    {
                        'Order ID': order_id,
                        'Side': details.get('side'),
                        'Price': details.get('price'),
                        'Size': details.get('size'),
                        'Event TS': to_local_time(details.get('event_ts', '')),
                    }
                    for order_id, details in live_orders.items()
                ]
            )
            st.dataframe(live_orders_df, width='stretch', hide_index=True)
        else:
            st.info("No live orders detected.")

    # Allow drilling into specific execution
    st.markdown("---")
    st.markdown("#### Execution Details")

    selected_idx = st.selectbox(
        "Select an execution to view details:",
        options=range(len(executions)),
        format_func=lambda i: f"{executions[i].get('event_ts')} - {executions[i].get('event_type')} - {executions[i].get('order_id', '')[:30]}",
        key=f'execution_selector_{market_id}'
    )

    if selected_idx is not None:
        selected_execution = executions[selected_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Event Details**")
            st.json({
                'event_type': selected_execution.get('event_type'),
                'event_ts': selected_execution.get('event_ts'),
                'decision_id': selected_execution.get('decision_id'),
                'order_id': selected_execution.get('order_id'),
                'client_order_id': selected_execution.get('client_order_id'),
            })

        with col2:
            st.markdown("**Order Details**")
            st.json({
                'market': selected_execution.get('market'),
                'side': selected_execution.get('side'),
                'price': selected_execution.get('price'),
                'size': selected_execution.get('size'),
                'status': selected_execution.get('status'),
                'latency_ms': selected_execution.get('latency_ms'),
            })

        # Show error details if present
        if selected_execution.get('error_type') or selected_execution.get('error_msg'):
            st.markdown("---")
            st.markdown("**Error Information**")
            st.error(f"**Type:** {selected_execution.get('error_type')}\n\n**Message:** {selected_execution.get('error_msg')}")

        # Show associated decision log
        decision_id = selected_execution.get('decision_id')
        if decision_id:
            st.markdown("---")
            st.markdown("**Associated Decision Log**")

            # Get decision logs and find the matching one
            decisions = db_client.get_recent_decision_logs(days=days_lookback, market_id=market_id)
            matching_decision = next((d for d in decisions if d.get('decision_id') == decision_id), None)

            if matching_decision:
                with st.expander("📊 View Decision Details", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Order Book Snapshot**")
                        st.json(matching_decision.get('order_book_snapshot', {}))

                        st.markdown("**Inventory**")
                        st.json(matching_decision.get('inventory', {}))

                    with col2:
                        st.markdown("**Target Quotes**")
                        target_quotes = matching_decision.get('target_quotes', [])
                        if target_quotes:
                            quotes_df = pd.DataFrame(target_quotes)
                            st.dataframe(quotes_df, width='stretch', hide_index=True)
                        else:
                            st.info("No target quotes")

                        st.markdown("**Metadata**")
                        st.json({
                            'timestamp': matching_decision.get('timestamp'),
                            'bot_run_id': matching_decision.get('bot_run_id'),
                            'bot_version': matching_decision.get('bot_version'),
                        })
            else:
                st.info(f"Decision log not found for decision_id: {decision_id}")


def render_execution_timeline(executions: List[Dict]):
    """Render timeline visualization of execution events over time."""
    if not executions:
        return

    st.subheader("Execution Timeline")

    # Prepare data for timeline
    timeline_data = []
    for execution in executions:
        timeline_data.append({
            'timestamp': execution.get('event_ts'),
            'event_type': execution.get('event_type', 'UNKNOWN'),
            'order_id': execution.get('order_id', 'N/A')[:20],
            'price': execution.get('price'),
            'size': execution.get('size'),
            'side': execution.get('side'),
            'status': execution.get('status')
        })

    df = pd.DataFrame(timeline_data)

    # Define color mapping for event types
    color_map = {
        'ORDER_PLACE': '#636EFA',
        'ORDER_RESULT': '#00CC96',
        'ORDER_CANCEL': '#FFA15A',
        'FILL': '#19D3F3',
        'UNKNOWN': '#B6E880'
    }

    # Create scatter plot with event types
    fig = go.Figure()

    for event_type in df['event_type'].unique():
        df_filtered = df[df['event_type'] == event_type]

        # Create hover text with details
        hover_texts = []
        for _, row in df_filtered.iterrows():
            hover_text = f"Event: {row['event_type']}<br>"
            hover_text += f"Time: {row['timestamp']}<br>"
            if row['order_id']:
                hover_text += f"Order: {row['order_id']}<br>"
            if row['price']:
                hover_text += f"Price: ${row['price']:.3f}<br>"
            if row['size']:
                hover_text += f"Size: {row['size']}<br>"
            if row['side']:
                hover_text += f"Side: {row['side']}<br>"
            if row['status']:
                hover_text += f"Status: {row['status']}"
            hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=df_filtered['timestamp'],
            y=[event_type] * len(df_filtered),
            mode='markers',
            name=event_type,
            marker=dict(
                size=10,
                color=color_map.get(event_type, '#B6E880'),
                line=dict(width=1, color='white')
            ),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time",
        yaxis_title="Event Type",
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')


def render_latency_chart(executions: List[Dict]):
    """Render latency distribution chart."""
    st.subheader("Order Latency Distribution")

    # Filter executions with latency data
    latency_data = [
        {
            'timestamp': e.get('event_ts'),
            'latency_ms': e.get('latency_ms'),
            'event_type': e.get('event_type')
        }
        for e in executions
        if e.get('latency_ms') is not None
    ]

    if not latency_data:
        st.info("No latency data available")
        return

    df = pd.DataFrame(latency_data)

    # Create scatter plot
    fig = go.Figure()

    for event_type in df['event_type'].unique():
        df_filtered = df[df['event_type'] == event_type]
        fig.add_trace(go.Scatter(
            x=df_filtered['timestamp'],
            y=df_filtered['latency_ms'],
            mode='markers',
            name=event_type,
            marker=dict(size=6)
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Timestamp",
        yaxis_title="Latency (ms)",
        hovermode='closest'
    )

    st.plotly_chart(fig, width='stretch')

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Latency", f"{df['latency_ms'].mean():.1f} ms")
    with col2:
        st.metric("P50 Latency", f"{df['latency_ms'].median():.1f} ms")
    with col3:
        st.metric("P95 Latency", f"{df['latency_ms'].quantile(0.95):.1f} ms")


def render_fill_logs(db_client: ReadOnlyDynamoDBClient, market_id: str, days: int):
    """Render fill logs (trades) for a specific market."""
    st.subheader(f"Fill Logs for {market_id}")

    with st.spinner("Loading fill logs..."):
        trades = db_client.get_recent_trades(days=days, market_id=market_id)

    if not trades:
        st.info(f"No fills found for {market_id} in the last {days} days")
        return

    st.markdown(f"**Found {len(trades)} fills**")

    # Sort fills by timestamp (most recent first)
    fallback_ts = datetime.min.replace(tzinfo=timezone.utc)

    def _fill_ts(trade: Dict) -> datetime:
        ts_str = trade.get('fill_timestamp') or trade.get('timestamp', '')
        return parse_iso_timestamp(ts_str) or fallback_ts

    trades_sorted = sorted(trades, key=_fill_ts, reverse=True)

    # Create summary table
    table_data = []
    for trade in trades_sorted:
        # Use fill_timestamp if available, otherwise fall back to timestamp
        timestamp = trade.get('fill_timestamp') or trade.get('timestamp', '')

        table_data.append({
            'Timestamp': to_local_time(timestamp) if timestamp else 'N/A',
            'Fill ID': trade.get('fill_id', '')[:20] + '...' if trade.get('fill_id') else 'N/A',
            'Order ID': trade.get('order_id', '')[:20] + '...' if trade.get('order_id') else 'N/A',
            'Side': trade.get('side', ''),
            'Price': f"${trade.get('price', 0.0):.3f}" if trade.get('price') else 'N/A',
            'Size': trade.get('size', 0),
            'Fees': f"${trade.get('fees', 0.0):.2f}" if trade.get('fees') is not None else 'N/A',
            'Total Value': f"${(trade.get('price', 0.0) * trade.get('size', 0)):.2f}",
        })

    df = pd.DataFrame(table_data)

    # Display table
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        height=400
    )

    # Summary statistics
    if trades_sorted:
        st.markdown("---")
        st.markdown("#### Summary Statistics")

        total_volume = sum(t.get('size', 0) for t in trades_sorted)
        buy_volume = sum(t.get('size', 0) for t in trades_sorted if t.get('side') in ['buy', 'yes'])
        sell_volume = sum(t.get('size', 0) for t in trades_sorted if t.get('side') in ['sell', 'no'])
        total_fees = sum(t.get('fees', 0) or 0 for t in trades_sorted)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fills", len(trades))
        with col2:
            st.metric("Total Volume", f"{total_volume} contracts")
        with col3:
            st.metric("Buy/Sell Split", f"{buy_volume}/{sell_volume}")
        with col4:
            st.metric("Total Fees", f"${total_fees:.2f}")

    # Allow drilling into specific fill
    st.markdown("---")
    st.markdown("#### Fill Details")

    selected_idx = st.selectbox(
        "Select a fill to view details:",
        options=range(len(trades_sorted)),
        format_func=lambda i: f"{trades_sorted[i].get('fill_timestamp') or trades_sorted[i].get('timestamp', 'N/A')} - {trades_sorted[i].get('side')} {trades_sorted[i].get('size')} @ ${trades_sorted[i].get('price', 0):.3f}",
        key=f'fill_selector_{market_id}'
    )

    if selected_idx is not None:
        selected_fill = trades_sorted[selected_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fill Information**")
            st.json({
                'fill_id': selected_fill.get('fill_id'),
                'order_id': selected_fill.get('order_id'),
                'fill_timestamp': selected_fill.get('fill_timestamp'),
                'date': selected_fill.get('date'),
            })

        with col2:
            st.markdown("**Trade Details**")
            st.json({
                'market_id': selected_fill.get('market_id'),
                'side': selected_fill.get('side'),
                'price': selected_fill.get('price'),
                'size': selected_fill.get('size'),
                'fees': selected_fill.get('fees'),
                'total_value': selected_fill.get('price', 0) * selected_fill.get('size', 0),
            })


def render(environment: str, region: str):
    """Render the market deep dive page."""
    st.title("🔍 Market Deep Dive")
    st.markdown(f"**Environment:** {environment.upper()} | **Region:** {region}")
    st.markdown("---")

    # Initialize DynamoDB client
    db_client = ReadOnlyDynamoDBClient(region=region, environment=environment)

    # Market selector
    with st.spinner("Loading markets..."):
        market_configs = db_client.get_all_market_configs(enabled_only=False)

    if not market_configs:
        st.warning("No markets found in the database")
        return

    # Get market from session state or dropdown
    market_options = [config.get('market_id') for config in market_configs]

    # Check if there's a selected market from the home page
    if 'selected_market' in st.session_state and st.session_state['selected_market'] in market_options:
        default_idx = market_options.index(st.session_state['selected_market'])
    else:
        default_idx = 0

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        selected_market = st.selectbox(
            "Select Market:",
            options=market_options,
            index=default_idx,
            key='deep_dive_market_selector'
        )

    with col2:
        days_lookback = st.selectbox(
            "Lookback Period:",
            options=[1, 3, 7, 14, 30],
            index=2,  # Default to 7 days
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
            key='days_lookback'
        )

    with col3:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()

    if selected_market:
        # Store selected market in session state
        st.session_state['selected_market'] = selected_market

        # Display market config
        with st.expander("⚙️ Market Configuration", expanded=False):
            config = db_client.get_market_config(selected_market)
            if config:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Enabled", "✅ Yes" if config.get('enabled') else "❌ No")
                    st.metric("Quote Size", config.get('quote_size'))
                with col2:
                    fair_value = config.get('fair_value')
                    if fair_value is not None:
                        st.metric("Fair Value", f"${fair_value:.3f}")
                    else:
                        st.metric("Fair Value", "N/A")
                    st.metric("Min Spread", f"${config.get('min_spread', 0.0):.3f}")
                with col3:
                    st.metric("Inventory Skew Factor", f"{config.get('inventory_skew_factor', 0.0):.2f}")
                    st.metric("Max Inventory (YES)", config.get('max_inventory_yes'))
                with col4:
                    st.metric("Max Inventory (NO)", config.get('max_inventory_no'))
            else:
                st.warning("Market configuration not found")

        st.markdown("---")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Decision Logs", "⚡ Execution Logs", "💰 Fill Logs", "📈 Analytics"])

        with tab1:
            render_decision_logs(db_client, selected_market, days_lookback)

        with tab2:
            executions = db_client.get_recent_execution_logs(days=days_lookback, market_id=selected_market)
            render_execution_logs(db_client, selected_market, days_lookback)

        with tab3:
            render_fill_logs(db_client, selected_market, days_lookback)

        with tab4:
            executions = db_client.get_recent_execution_logs(days=days_lookback, market_id=selected_market)
            if executions:
                render_execution_timeline(executions)
                st.markdown("---")
                render_latency_chart(executions)
            else:
                st.info(f"No execution data available for {selected_market}")
