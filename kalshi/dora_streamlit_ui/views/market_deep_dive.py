"""
Market Deep Dive Page - Detailed view of fill logs
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
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


def format_fill_side(side: Optional[str]) -> str:
    """Map fill side to bid/ask framing."""
    if not side:
        return 'N/A'
    normalized = side.lower()
    if normalized in ('yes', 'buy', 'bid'):
        return 'bid'
    if normalized in ('no', 'sell', 'ask'):
        return 'ask'
    return side


def get_kalshi_market_url(market_id: str) -> str:
    """Generate Kalshi market URL from market ID."""
    return f"https://kalshi.com/markets/{market_id}"


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
            'Side': format_fill_side(trade.get('side')),
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
        bid_volume = sum(t.get('size', 0) for t in trades_sorted if format_fill_side(t.get('side')) == 'bid')
        ask_volume = sum(t.get('size', 0) for t in trades_sorted if format_fill_side(t.get('side')) == 'ask')
        total_fees = sum(t.get('fees', 0) or 0 for t in trades_sorted)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fills", len(trades))
        with col2:
            st.metric("Total Volume", f"{total_volume} contracts")
        with col3:
            st.metric("Bid/Ask Split", f"{bid_volume}/{ask_volume}")
        with col4:
            st.metric("Total Fees", f"${total_fees:.2f}")

    # Allow drilling into specific fill
    st.markdown("---")
    st.markdown("#### Fill Details")

    selected_idx = st.selectbox(
        "Select a fill to view details:",
        options=range(len(trades_sorted)),
        format_func=lambda i: f"{trades_sorted[i].get('fill_timestamp') or trades_sorted[i].get('timestamp', 'N/A')} - {format_fill_side(trades_sorted[i].get('side'))} {trades_sorted[i].get('size')} @ ${trades_sorted[i].get('price', 0):.3f}",
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
                'side': format_fill_side(selected_fill.get('side')),
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

        # Display fill logs only
        render_fill_logs(db_client, selected_market, days_lookback)
