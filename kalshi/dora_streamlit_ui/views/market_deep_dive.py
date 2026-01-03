"""
Market Deep Dive Page - Detailed view of fill logs
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path to import db_client and kalshi client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dora_bot'))
from db_client import ReadOnlyDynamoDBClient

# Import Kalshi client for fetching market trades
try:
    from kalshi_client import KalshiHttpClient, Environment
    from cryptography.hazmat.primitives import serialization
    import base64
    KALSHI_CLIENT_AVAILABLE = True
except ImportError:
    KALSHI_CLIENT_AVAILABLE = False


NY_TZ = ZoneInfo("America/New_York")


def get_kalshi_client(environment: str) -> Optional['KalshiHttpClient']:
    """Create a Kalshi HTTP client for API calls."""
    if not KALSHI_CLIENT_AVAILABLE:
        st.error("Kalshi client not available. Please install required dependencies.")
        return None

    try:
        # Try to get credentials from Streamlit secrets first
        if hasattr(st, 'secrets'):
            if environment == 'demo':
                keyid = st.secrets.get('DEMO_KEYID')
                key_b64 = st.secrets.get('DEMO_KEY_B64')
            else:
                keyid = st.secrets.get('PROD_KEYID')
                key_b64 = st.secrets.get('PROD_KEY_B64')

            if keyid and key_b64:
                # Decode base64-encoded private key
                private_key_pem = base64.b64decode(key_b64)
                private_key = serialization.load_pem_private_key(private_key_pem, password=None)

                env = Environment.DEMO if environment == 'demo' else Environment.PROD
                return KalshiHttpClient(key_id=keyid, private_key=private_key, environment=env)

        # Fall back to environment variables
        from dotenv import load_dotenv
        kalshi_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dora_bot', '..')
        load_dotenv(os.path.join(kalshi_dir, '.env'))

        keyid = os.getenv('DEMO_KEYID') if environment == 'demo' else os.getenv('PROD_KEYID')
        keyfile = os.getenv('DEMO_KEYFILE') if environment == 'demo' else os.getenv('PROD_KEYFILE')

        if keyid and keyfile:
            keyfile_path = os.path.join(kalshi_dir, keyfile)
            if os.path.exists(keyfile_path):
                with open(keyfile_path, "rb") as key_file:
                    private_key = serialization.load_pem_private_key(key_file.read(), password=None)

                env = Environment.DEMO if environment == 'demo' else Environment.PROD
                return KalshiHttpClient(key_id=keyid, private_key=private_key, environment=env)

        return None
    except Exception as e:
        st.error(f"Failed to create Kalshi client: {e}")
        return None


def to_local_time(timestamp_str: str) -> str:
    """Convert ISO timestamp to New York time string."""
    try:
        if 'Z' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        elif '+' in timestamp_str or ('-' in timestamp_str.split('T')[1] if 'T' in timestamp_str else False):
            dt = datetime.fromisoformat(timestamp_str)
        else:
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

        local_dt = dt.astimezone(NY_TZ)
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


def render_most_recent_decision(db_client: ReadOnlyDynamoDBClient, market_id: str):
    """Render the most recent decision log for a market."""
    st.subheader(f"📊 Most Recent Decision Log")

    with st.spinner("Loading most recent decision..."):
        decision = db_client.get_most_recent_decision_log(market_id=market_id)

    if not decision:
        st.info(f"No recent decision logs found for {market_id}")
        return

    # Show decision timestamp
    decision_ts = decision.get('timestamp', '')
    if decision_ts:
        st.info(f"**Decision Timestamp:** {to_local_time(decision_ts)}")

    # Extract data
    price_calc = decision.get('price_calc', {})
    order_book = decision.get('order_book_snapshot', {})
    inventory_data = decision.get('inventory', {})
    target_quotes = decision.get('target_quotes', [])

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fair_value = decision.get('fair_value') or price_calc.get('fair_value')
        st.metric("Fair Value", f"${fair_value:.3f}" if fair_value is not None else 'N/A')
    with col2:
        mid_price = decision.get('mid_price') or order_book.get('mid')
        st.metric("Mid Price", f"${mid_price:.3f}" if mid_price is not None else 'N/A')
    with col3:
        net_yes_qty = inventory_data.get('net_yes_qty') if isinstance(inventory_data, dict) else inventory_data
        st.metric("Inventory (YES)", net_yes_qty if net_yes_qty is not None else 'N/A')
    with col4:
        fv_source = price_calc.get('fv_source', 'N/A')
        st.metric("FV Source", fv_source)
    with col5:
        trade_std_dev = price_calc.get('trade_std_dev')
        st.metric("Trade StdDev", f"${trade_std_dev:.4f}" if trade_std_dev is not None else 'N/A')

    # WMA metrics if available
    if price_calc.get('wma_price') is not None:
        st.markdown("**Weighted Moving Average Metrics:**")
        wma_col1, wma_col2, wma_col3 = st.columns(3)
        with wma_col1:
            st.metric("WMA Price", f"${price_calc.get('wma_price'):.3f}")
        with wma_col2:
            st.metric("WMA Trades Count", price_calc.get('wma_trades_count', 0))
        with wma_col3:
            wma_time = price_calc.get('wma_trade_time', '')
            st.metric("WMA Last Trade", to_local_time(wma_time) if wma_time else 'N/A')

    # Target quotes
    if target_quotes:
        st.markdown("**Target Quotes:**")
        quote_col1, quote_col2 = st.columns(2)
        with quote_col1:
            st.markdown("*Bids*")
            bids = [q for q in target_quotes if q.get('side') == 'bid']
            if bids:
                for bid in sorted(bids, key=lambda x: x.get('price', 0), reverse=True):
                    st.text(f"${bid.get('price', 0):.3f} x {bid.get('size', 0)}")
            else:
                st.text("No bids")
        with quote_col2:
            st.markdown("*Asks*")
            asks = [q for q in target_quotes if q.get('side') == 'ask']
            if asks:
                for ask in sorted(asks, key=lambda x: x.get('price', 0)):
                    st.text(f"${ask.get('price', 0):.3f} x {ask.get('size', 0)}")
            else:
                st.text("No asks")

    # Full decision log
    with st.expander("Full Decision Log Details", expanded=False):
        st.json(decision)


def render_historical_trade_chart(
    db_client: ReadOnlyDynamoDBClient,
    market_id: str,
    environment: str,
    days: int = 3
):
    """Render a historical trade chart with Kalshi market trades and our fills overlaid."""
    st.subheader(f"📈 Historical Trade Chart (Last {days} Days)")

    # Create Kalshi client
    kalshi_client = get_kalshi_client(environment)
    if not kalshi_client:
        st.warning("Unable to load Kalshi client. Chart requires API access.")
        return

    with st.spinner("Loading trade data from Kalshi API..."):
        try:
            # Calculate time range
            now = datetime.now(timezone.utc)
            min_ts = int((now - timedelta(days=days)).timestamp() * 1000)  # Convert to milliseconds
            max_ts = int(now.timestamp() * 1000)

            # Fetch market trades from Kalshi
            trades_response = kalshi_client.get_trades(
                ticker=market_id,
                min_ts=min_ts,
                max_ts=max_ts,
                limit=1000  # Get up to 1000 trades
            )

            market_trades = trades_response.get('trades', [])

            if not market_trades:
                st.warning(f"No market trades found for {market_id} in the last {days} days")
                return

            # Process market trades
            trade_times = []
            trade_prices = []
            for trade in market_trades:
                # Kalshi returns yes_price in cents
                price = trade.get('yes_price', 0) / 100.0
                # created_time is ISO format
                timestamp_str = trade.get('created_time', '')
                if timestamp_str:
                    ts = parse_iso_timestamp(timestamp_str)
                    if ts:
                        trade_times.append(ts)
                        trade_prices.append(price)

            # Get our fills from DynamoDB
            our_fills = db_client.get_recent_trades(days=days, market_id=market_id)

            # Process our fills and get decision context for each
            fill_times = []
            fill_prices = []
            fill_hover_texts = []
            fill_sizes = []

            for fill in our_fills:
                fill_timestamp = fill.get('fill_timestamp') or fill.get('timestamp')
                fill_price = fill.get('price')

                if fill_timestamp and fill_price:
                    ts = parse_iso_timestamp(fill_timestamp)
                    if ts:
                        fill_times.append(ts)
                        fill_prices.append(fill_price)
                        fill_sizes.append(fill.get('size', 0))

                        # Get decision context for this fill
                        decision_context = db_client.get_decision_context_for_fill(
                            market_id=market_id,
                            fill_timestamp=fill_timestamp,
                            days=1  # Look within 1 day for decision
                        )

                        # Build hover text
                        hover_parts = [
                            f"<b>Fill @ {to_local_time(fill_timestamp)}</b>",
                            f"Price: ${fill_price:.3f}",
                            f"Size: {fill.get('size', 0)}",
                            f"Side: {format_fill_side(fill.get('side'))}",
                            f"P&L: ${fill.get('pnl_realized', 0):+.2f}"
                        ]

                        if decision_context and decision_context.get('decision'):
                            decision = decision_context['decision']
                            price_calc = decision.get('price_calc', {})
                            inventory = decision.get('inventory', {})
                            target_quotes = decision.get('target_quotes', [])

                            hover_parts.append("<br><b>Decision Context:</b>")

                            # Inventory
                            net_yes = inventory.get('net_yes_qty') if isinstance(inventory, dict) else inventory
                            if net_yes is not None:
                                hover_parts.append(f"Inventory: {net_yes}")

                            # Fair value and source
                            fv = decision.get('fair_value') or price_calc.get('fair_value')
                            fv_source = price_calc.get('fv_source')
                            if fv is not None:
                                hover_parts.append(f"Fair Value: ${fv:.3f} ({fv_source})")

                            # Volatility
                            std_dev = price_calc.get('trade_std_dev')
                            if std_dev is not None:
                                hover_parts.append(f"Trade StdDev: ${std_dev:.4f}")

                            # Target quotes
                            if target_quotes:
                                bids = [q for q in target_quotes if q.get('side') == 'bid']
                                asks = [q for q in target_quotes if q.get('side') == 'ask']
                                if bids:
                                    bid_str = ', '.join([f"${b.get('price', 0):.3f}x{b.get('size', 0)}" for b in bids])
                                    hover_parts.append(f"Target Bids: {bid_str}")
                                if asks:
                                    ask_str = ', '.join([f"${a.get('price', 0):.3f}x{a.get('size', 0)}" for a in asks])
                                    hover_parts.append(f"Target Asks: {ask_str}")

                        fill_hover_texts.append("<br>".join(hover_parts))

            # Create plotly chart
            fig = go.Figure()

            # Add market trades as line
            fig.add_trace(go.Scatter(
                x=trade_times,
                y=trade_prices,
                mode='lines',
                name='Market Trades',
                line=dict(color='lightblue', width=1),
                hovertemplate='<b>Market Trade</b><br>Time: %{x}<br>Price: $%{y:.3f}<extra></extra>'
            ))

            # Add our fills as scatter points
            if fill_times:
                fig.add_trace(go.Scatter(
                    x=fill_times,
                    y=fill_prices,
                    mode='markers',
                    name='Our Fills',
                    marker=dict(
                        size=[min(10 + s, 30) for s in fill_sizes],  # Size based on fill size
                        color='red',
                        symbol='circle',
                        line=dict(color='darkred', width=2)
                    ),
                    text=fill_hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ))

            # Update layout
            fig.update_layout(
                title=f"Trade History: {market_id}",
                xaxis_title="Time (NY)",
                yaxis_title="Price ($)",
                hovermode='closest',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Display chart
            st.plotly_chart(fig, width='stretch')

            # Display summary stats
            st.markdown("**Chart Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Market Trades", len(market_trades))
            with col2:
                st.metric("Our Fills", len(our_fills))
            with col3:
                if trade_prices:
                    st.metric("Avg Market Price", f"${sum(trade_prices)/len(trade_prices):.3f}")
            with col4:
                if fill_prices:
                    st.metric("Avg Fill Price", f"${sum(fill_prices)/len(fill_prices):.3f}")

        except Exception as e:
            st.error(f"Error loading trade chart: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


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
            'Timestamp': timestamp or 'N/A',
            'Local Time': to_local_time(timestamp) if timestamp else 'N/A',
            'Fill ID': trade.get('fill_id', '')[:20] + '...' if trade.get('fill_id') else 'N/A',
            'Order ID': trade.get('order_id', '')[:20] + '...' if trade.get('order_id') else 'N/A',
            'Side': format_fill_side(trade.get('side')),
            'Price': f"${trade.get('price', 0.0):.3f}" if trade.get('price') else 'N/A',
            'Size': trade.get('size', 0),
            'Fees': f"${trade.get('fees', 0.0):.2f}" if trade.get('fees') is not None else 'N/A',
            'Realized P&L': f"${trade.get('pnl_realized', 0.0):+.2f}" if trade.get('pnl_realized') is not None else 'N/A',
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
        total_realized_pnl = sum(t.get('pnl_realized', 0) or 0 for t in trades_sorted)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Fills", len(trades))
        with col2:
            st.metric("Total Volume", f"{total_volume} contracts")
        with col3:
            st.metric("Bid/Ask Split", f"{bid_volume}/{ask_volume}")
        with col4:
            st.metric("Total Fees", f"${total_fees:.2f}")
        with col5:
            st.metric("Total Realized P&L", f"${total_realized_pnl:+.2f}")

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

        # Add button to load decision context
        st.markdown("---")
        fill_timestamp = selected_fill.get('fill_timestamp') or selected_fill.get('timestamp')

        if fill_timestamp:
            if st.button("Load Decision Context", key=f'load_decision_{market_id}_{selected_idx}', type="primary"):
                with st.spinner("Loading decision context..."):
                    decision_context = db_client.get_decision_context_for_fill(
                        market_id=market_id,
                        fill_timestamp=fill_timestamp,
                        days=days
                    )

                if decision_context and decision_context.get('decision'):
                    st.success("Decision context loaded successfully!")

                    # Display decision log data
                    st.markdown("**Decision Log (Most Recent Before Fill)**")
                    decision = decision_context['decision']

                    # Show decision timestamp
                    decision_ts = decision.get('timestamp', '')
                    if decision_ts:
                        st.info(f"Decision Timestamp: {to_local_time(decision_ts)}")

                    # Show key decision metrics
                    # Extract price_calc once for reuse
                    price_calc = decision.get('price_calc', {})

                    dec_col1, dec_col2, dec_col3, dec_col4 = st.columns(4)
                    with dec_col1:
                        # Fair value can be at top level or in price_calc
                        fair_value = decision.get('fair_value')
                        if fair_value is None:
                            fair_value = price_calc.get('fair_value')
                        st.metric("Fair Value", f"${fair_value:.3f}" if fair_value is not None else 'N/A')
                    with dec_col2:
                        # Try to get mid_price from order_book_snapshot if not at top level
                        mid_price = decision.get('mid_price')
                        if mid_price is None:
                            order_book_snapshot = decision.get('order_book_snapshot', {})
                            mid_price = order_book_snapshot.get('mid')
                        st.metric("Mid Price", f"${mid_price:.3f}" if mid_price is not None else 'N/A')
                    with dec_col3:
                        # Inventory is stored as a dict with net_yes_qty
                        inventory_data = decision.get('inventory', {})
                        if isinstance(inventory_data, dict):
                            net_yes_qty = inventory_data.get('net_yes_qty')
                        else:
                            net_yes_qty = inventory_data
                        st.metric("Inventory (YES)", net_yes_qty if net_yes_qty is not None else 'N/A')
                    with dec_col4:
                        # Skew is in price_calc (note: field is 'skew', not 'inventory_skew')
                        skew = decision.get('inventory_skew')
                        if skew is None:
                            skew = price_calc.get('skew')
                        if skew is None:
                            skew = price_calc.get('inventory_skew')
                        st.metric("Inventory Skew", f"${skew:.4f}" if skew is not None else 'N/A')

                    # Show order book depth if available
                    order_book = decision.get('order_book_snapshot', {})
                    top_bids = order_book.get('top_bids', [])
                    top_asks = order_book.get('top_asks', [])

                    if top_bids or top_asks:
                        st.markdown("---")
                        st.markdown("**Order Book Depth at Decision Time**")

                        ob_col1, ob_col2 = st.columns(2)
                        with ob_col1:
                            st.markdown("*Bids (Best 3)*")
                            if top_bids:
                                for bid in top_bids:
                                    st.text(f"${bid.get('price', 0):.3f} x {bid.get('size', 0)}")
                            else:
                                st.text("No bid data")

                        with ob_col2:
                            st.markdown("*Asks (Best 3)*")
                            if top_asks:
                                for ask in top_asks:
                                    st.text(f"${ask.get('price', 0):.3f} x {ask.get('size', 0)}")
                            else:
                                st.text("No ask data")

                    # Show recent trades if available
                    recent_trades = decision.get('recent_trades', [])
                    if recent_trades:
                        st.markdown("---")
                        st.markdown("**Recent Trades at Decision Time**")

                        # Create a table for recent trades
                        trades_data = []
                        for trade in recent_trades:
                            timestamp = trade.get('timestamp', '')
                            trades_data.append({
                                'Price': f"${trade.get('price', 0):.3f}",
                                'Size': trade.get('size', 0),
                                'Time': to_local_time(timestamp) if timestamp else 'N/A'
                            })

                        trades_df = pd.DataFrame(trades_data)
                        st.dataframe(trades_df, hide_index=True, width='stretch')

                    # Show target quotes if available
                    target_quotes = decision.get('target_quotes', [])
                    if target_quotes:
                        st.markdown("**Target Quotes at Decision Time**")
                        bids = [q for q in target_quotes if q.get('side') == 'bid']
                        asks = [q for q in target_quotes if q.get('side') == 'ask']

                        quote_col1, quote_col2 = st.columns(2)
                        with quote_col1:
                            st.markdown("*Bids*")
                            if bids:
                                for bid in sorted(bids, key=lambda x: x.get('price', 0), reverse=True):
                                    st.text(f"${bid.get('price', 0):.3f} x {bid.get('size', 0)}")
                            else:
                                st.text("No bids")

                        with quote_col2:
                            st.markdown("*Asks*")
                            if asks:
                                for ask in sorted(asks, key=lambda x: x.get('price', 0)):
                                    st.text(f"${ask.get('price', 0):.3f} x {ask.get('size', 0)}")
                            else:
                                st.text("No asks")

                    # Show full decision log in expander
                    with st.expander("Full Decision Log Details", expanded=False):
                        st.json(decision)

                else:
                    st.error(f"Could not find decision context for this fill")
        else:
            st.warning("No timestamp available for this fill")


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

        # Display most recent decision log
        render_most_recent_decision(db_client, selected_market)

        st.markdown("---")

        # Historical trade chart button
        if st.button("📈 Load Historical Trade Chart", type="primary", key='load_chart_button'):
            render_historical_trade_chart(db_client, selected_market, environment, days=3)

        st.markdown("---")

        # Display fill logs
        render_fill_logs(db_client, selected_market, days_lookback)
