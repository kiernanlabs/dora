import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from kalshi_service import KalshiService
import traceback
import math

MAKER_FEE_RATE = 0.0175
TAKER_FEE_RATE = 0.07

def log_event(message: str):
    """Append user-facing activity messages for later display."""
    st.session_state.setdefault('activity_log', []).append(message)

def filter_df_by_days(df: pd.DataFrame, days: float) -> pd.DataFrame:
    """Return dataframe filtered to rows within the last N days."""
    if days <= 0 or 'timestamp' not in df.columns:
        return df
    cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    return df[df['timestamp'] >= cutoff_time]

def extract_spread(orderbook):
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

def fee_per_contract_cents(price_cents: float, rate: float) -> float:
    """Kalshi fee per contract in cents using the provided rate.

    Uses the exact calculated fee (no rounding up) since rounding is reimbursed.
    """
    price_dollars = price_cents / 100 if price_cents is not None else 0.5
    fee_cents = rate * price_dollars * (1 - price_dollars) * 100
    return fee_cents

def calculate_profit_at_spread(df_filtered: pd.DataFrame, bid_price: float, ask_price: float, sorted_bids, sorted_asks) -> float:
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
    # Maker fees on executed legs (realized)
    maker_fee_bid = fee_per_contract_cents(bid_price, rate=MAKER_FEE_RATE) * matched_volume
    maker_fee_ask = fee_per_contract_cents(ask_price, rate=MAKER_FEE_RATE) * matched_volume
    realized_pnl = realized_pnl_gross - (maker_fee_bid + maker_fee_ask)

    if net_position > 0:
        position_cost = net_position * bid_price
        liquidation_avg_price, liquidation_value, _ = calculate_liquidation_price(net_position, 'sell', sorted_bids)
        taker_fee = fee_per_contract_cents(liquidation_avg_price, rate=TAKER_FEE_RATE) * net_position
        unrealized_pnl = (liquidation_value - taker_fee) - position_cost
    elif net_position < 0:
        position_value = abs(net_position) * ask_price
        liquidation_avg_price, liquidation_cost, _ = calculate_liquidation_price(abs(net_position), 'buy', sorted_asks)
        taker_fee = fee_per_contract_cents(liquidation_avg_price, rate=TAKER_FEE_RATE) * abs(net_position)
        unrealized_pnl = position_value - (liquidation_cost + taker_fee)
    else:
        unrealized_pnl = 0

    return realized_pnl + unrealized_pnl

def optimize_profit(df_filtered: pd.DataFrame, sorted_bids, sorted_asks) -> dict:
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

def compute_summary_metrics(df: pd.DataFrame, orderbook, risk_aversion_k: float, use_demo: bool) -> dict:
    """Compute summary metrics for 7/60 day horizons."""
    best_bid, best_ask, current_spread, yes_bids, no_bids = extract_spread(orderbook)
    if best_bid is None or best_ask is None or current_spread is None:
        return {}

    sorted_bids = sorted(yes_bids, key=lambda x: x[0], reverse=True)
    sorted_asks = sorted([[100 - x[0], x[1]] for x in no_bids], key=lambda x: x[0]) if no_bids else []

    service = KalshiService(use_demo=use_demo)

    summary = {}
    for lookback in (7, 60):
        df_filtered = filter_df_by_days(df, lookback)

        risk_metrics = service.calculate_trade_risk_metrics(df_filtered, risk_aversion_k=risk_aversion_k)
        required_spread = risk_metrics.get('required_full_spread')

        current_profit_cents = calculate_profit_at_spread(df_filtered, best_bid, best_ask, sorted_bids, sorted_asks)
        optimal = optimize_profit(df_filtered, sorted_bids, sorted_asks)

        midpoint_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else (df_filtered['adjusted_price'].mean() if 'adjusted_price' in df_filtered.columns else 50)
        maker_fee_cents = fee_per_contract_cents(midpoint_price, rate=MAKER_FEE_RATE)
        roundtrip_fee_cents = maker_fee_cents * 2  # assume maker on both legs for required spread

        summary[lookback] = {
            'current_profit_cents': current_profit_cents,
            'optimal_profit_cents': optimal.get('profit_cents') if optimal else None,
            'optimal_bid': optimal.get('bid') if optimal else None,
            'optimal_ask': optimal.get('ask') if optimal else None,
            'current_spread': current_spread,
            'required_full_spread': required_spread,
            'required_full_spread_with_fees': (required_spread + roundtrip_fee_cents) if required_spread is not None else None,
            'roundtrip_fee_cents': roundtrip_fee_cents,
            'maker_fee_cents': maker_fee_cents,
            'midpoint_price': midpoint_price
        }

    return summary

def run_information_risk_auto(force: bool = False):
    """Run information risk assessment once per ticker/environment unless forced."""
    if not st.session_state.market_info:
        return None

    market = st.session_state.market_info.get('market', {})
    ticker = market.get('ticker')
    key = (ticker, st.session_state.use_demo)

    if not force:
        if st.session_state.info_risk and st.session_state.last_info_risk_key == key:
            return st.session_state.info_risk

    try:
        service = KalshiService(use_demo=st.session_state.use_demo)

        yes_bid = market.get('yes_bid', 0)
        yes_ask = market.get('yes_ask', 0)
        current_price = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else 50

        info_risk = service.assess_information_risk(
            market_title=market.get('title', ''),
            current_price=current_price,
            market_subtitle=market.get('subtitle'),
            rules=market.get('rules')
        )

        st.session_state.info_risk = info_risk
        st.session_state.last_info_risk_key = key
        return info_risk
    except Exception as e:
        st.session_state.info_risk = {'error': str(e), 'rationale': str(e)}
        st.session_state.last_info_risk_key = key
        return st.session_state.info_risk

def calculate_liquidation_price(volume, side, orderbook_levels):
    """
    Calculate the average price to liquidate a position through the orderbook.

    Args:
        volume: Number of contracts to liquidate (positive number)
        side: 'buy' or 'sell' - the side we need to trade on to liquidate
        orderbook_levels: List of [price, quantity] pairs sorted appropriately

    Returns:
        Tuple of (average_price, total_cost, breakdown_text)
    """
    if volume <= 0 or not orderbook_levels:
        return 0, 0, "No position to liquidate"

    remaining = volume
    total_cost = 0
    filled_levels = []

    for price, available_qty in orderbook_levels:
        if remaining <= 0:
            break

        fill_qty = min(remaining, available_qty)
        cost = fill_qty * price
        total_cost += cost
        filled_levels.append((price, fill_qty))
        remaining -= fill_qty

    if remaining > 0:
        # Not enough liquidity in orderbook
        avg_price = total_cost / (volume - remaining) if (volume - remaining) > 0 else 0
        breakdown = f"⚠️ Only {volume - remaining:,}/{volume:,} contracts available in orderbook"
    else:
        avg_price = total_cost / volume
        # Create breakdown text
        if len(filled_levels) == 1:
            breakdown = f"{filled_levels[0][1]:,} @ ¢{filled_levels[0][0]:.2f}"
        else:
            breakdown = " + ".join([f"{qty:,}@¢{price:.2f}" for price, qty in filled_levels[:3]])
            if len(filled_levels) > 3:
                breakdown += f" + {len(filled_levels) - 3} more levels"

    return avg_price, total_cost, breakdown

# Page configuration
st.set_page_config(
    page_title="Kalshi Market Risk Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""
if 'use_demo' not in st.session_state:
    st.session_state.use_demo = False
if 'market_info' not in st.session_state:
    st.session_state.market_info = None
if 'orderbook' not in st.session_state:
    st.session_state.orderbook = None
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = None
if 'info_risk' not in st.session_state:
    st.session_state.info_risk = None
if 'last_info_risk_key' not in st.session_state:
    st.session_state.last_info_risk_key = None
if 'auto_info_risk' not in st.session_state:
    st.session_state.auto_info_risk = True

# Title
st.title("📊 Kalshi Market Risk Analysis Dashboard")
st.markdown("Analyze trade price and volume patterns for Kalshi markets")

# Sidebar for configuration
st.sidebar.header("Configuration")
use_demo = st.sidebar.checkbox("Use Demo Environment", value=st.session_state.use_demo)
ticker = st.sidebar.text_input(
    "Market Ticker",
    value=st.session_state.ticker,
    placeholder="e.g., INXD-25JAN31-T4850"
)
auto_info_risk = st.sidebar.checkbox(
    "Run Information Risk Assessment",
    value=st.session_state.auto_info_risk,
    help="Automatically call the AI information risk assessment when loading a market"
)
st.session_state.auto_info_risk = auto_info_risk
limit = st.sidebar.slider("Number of Trades", min_value=100, max_value=2000, value=1000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Parameters")
risk_aversion_k = st.sidebar.number_input(
    "Risk Aversion Constant (k)",
    min_value=0.0,
    max_value=5.0,
    value=0.25,
    step=0.05,
    help="Multiplier for inventory risk calculation. Higher values = more conservative spread requirements."
)

# Add explanatory text
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to use:
1. Enter a market ticker symbol
2. Select number of recent trades to fetch
3. Click "Fetch Data" to load and visualize

### What you'll see:
- **Trade Price Over Time**: Shows Buy Yes and Sell Yes prices (Sell Yes = 100 - Buy No price)
- **Signed Volume Over Time**: Positive values = Buy Yes trades, Negative values = Sell Yes trades
""")

# Main content
if st.sidebar.button("Fetch Data", type="primary"):
    if not ticker:
        st.error("Please enter a market ticker")
    else:
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                # Initialize service
                service = KalshiService(use_demo=use_demo)

                # Get market metadata
                market_info = service.get_market(ticker=ticker)

                # Get orderbook
                orderbook = service.get_orderbook(ticker=ticker)

                # Get trades as DataFrame
                df = service.get_trades_dataframe(ticker=ticker, limit=limit)

                if df.empty:
                    st.warning(f"No trades found for ticker: {ticker}")
                    st.session_state.df = None
                    st.session_state.market_info = None
                    st.session_state.orderbook = None
                    st.session_state.risk_metrics = None
                    st.session_state.info_risk = None
                    st.session_state.last_info_risk_key = None
                else:
                    # Calculate risk metrics
                    risk_metrics = service.calculate_trade_risk_metrics(df, risk_aversion_k=risk_aversion_k)
                    risk_metrics['risk_aversion_k'] = risk_aversion_k  # Store the k value used

                    # Store in session state (use df with metrics)
                    st.session_state.df = risk_metrics['df_with_metrics']
                    st.session_state.ticker = ticker
                    st.session_state.use_demo = use_demo
                    st.session_state.market_info = market_info
                    st.session_state.orderbook = orderbook
                    st.session_state.risk_metrics = risk_metrics
                    st.session_state.info_risk = None
                    st.session_state.last_info_risk_key = None

        except FileNotFoundError as e:
            st.error(f"Configuration Error: {str(e)}")
            st.info("Make sure you have set up your .env file with DEMO_KEYID and DEMO_KEYFILE (or PROD_* for production)")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

# Display data if available in session state
if st.session_state.df is not None:
    # Check if risk_aversion_k has changed and recalculate if needed
    if st.session_state.risk_metrics is not None:
        current_k = risk_aversion_k
        stored_k = st.session_state.risk_metrics.get('risk_aversion_k', 0.5)

        # If k has changed, recalculate risk metrics
        if abs(current_k - stored_k) > 0.001:
            service = KalshiService(use_demo=st.session_state.use_demo)
            # Get the original DataFrame without the metrics columns
            df_original = st.session_state.df[['timestamp', 'price', 'action', 'count', 'signed_volume',
                                                 'yes_price', 'taker_side', 'created_time', 'adjusted_price']].copy() if all(col in st.session_state.df.columns for col in ['yes_price', 'taker_side', 'created_time', 'adjusted_price']) else st.session_state.df
            risk_metrics = service.calculate_trade_risk_metrics(df_original, risk_aversion_k=current_k)
            risk_metrics['risk_aversion_k'] = current_k  # Store the k value used
            st.session_state.df = risk_metrics['df_with_metrics']
            st.session_state.risk_metrics = risk_metrics

    df = st.session_state.df

    # Auto-run information risk assessment once data is available
    if st.session_state.market_info and st.session_state.info_risk is None and st.session_state.get('auto_info_risk', False):
        with st.spinner("Running information risk assessment..."):
            run_information_risk_auto(force=False)

    # Quick summary (top-line numbers)
    summary_metrics = {}
    try:
        if st.session_state.orderbook is not None:
            summary_metrics = compute_summary_metrics(
                df,
                st.session_state.orderbook,
                risk_aversion_k=risk_aversion_k,
                use_demo=st.session_state.use_demo
            )
    except Exception as e:
        log_event(f"Summary metrics unavailable: {e}")

    if summary_metrics:
        st.subheader("📌 Summary")
        info_risk = st.session_state.get('info_risk', {})
        info_prob = "Not run"
        if info_risk:
            if info_risk.get('error'):
                info_prob = info_risk.get('rationale') or info_risk.get('error')
            else:
                info_prob = info_risk.get('probability')

        summary_col_info = st.columns(1)
        summary_col_info[0].metric("Likelihood of 20% Move (7d)", info_prob)

        for lookback in (7, 60):
            data = summary_metrics.get(lookback, {})
            if not data:
                continue

            st.caption(f"Last {lookback} Days")
            col_a, col_b, col_c, col_d, col_e = st.columns(5)

            current_profit = data.get('current_profit_cents')
            optimal_profit = data.get('optimal_profit_cents')
            current_spread = data.get('current_spread')
            required_spread = data.get('required_full_spread')
            required_spread_with_fees = data.get('required_full_spread_with_fees')
            roundtrip_fee_cents = data.get('roundtrip_fee_cents')
            maker_fee_cents = data.get('maker_fee_cents')
            midpoint_price = data.get('midpoint_price')

            col_a.metric(
                f"{lookback}d Profit (Current Spread)",
                f"${current_profit/100:,.2f}" if current_profit is not None else "N/A"
            )
            col_b.metric(
                f"{lookback}d Profit (Optimal Spread)",
                f"${optimal_profit/100:,.2f}" if optimal_profit is not None else "N/A"
            )

            col_c.metric(
                f"{lookback}d Current Spread",
                f"¢{current_spread:.2f}" if current_spread is not None else "N/A"
            )
            col_d.metric(
                f"{lookback}d Required Spread",
                f"¢{required_spread:.2f}" if required_spread is not None else "N/A",
                delta=(f"{current_spread - required_spread:+.2f}¢" if required_spread is not None and current_spread is not None else None),
                delta_color="inverse"
            )
            col_e.metric(
                f"{lookback}d Required Spread + Fees",
                f"¢{required_spread_with_fees:.2f}" if required_spread_with_fees is not None else "N/A",
                help=(f"Includes estimated roundtrip maker fees of ¢{roundtrip_fee_cents:.2f}" if roundtrip_fee_cents is not None else None)
            )

            if roundtrip_fee_cents is not None and maker_fee_cents is not None and midpoint_price is not None:
                st.caption(
                    f"Fee calc: midpoint ¢{midpoint_price:.2f} → maker per leg ¢{maker_fee_cents:.4f} (applied to realized P&L), "
                    f"roundtrip ¢{roundtrip_fee_cents:.4f}; unrealized P&L assumes taker fee on the liquidation leg."
                )

        st.markdown("---")

    # Display Market Information
    if st.session_state.market_info is not None:
        st.header("📋 Market Information")

        market = st.session_state.market_info.get('market', {})

        # Main market details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(market.get('title', 'N/A'))

            # Display subtitle if available
            if market.get('subtitle'):
                st.markdown(f"*{market.get('subtitle')}*")

            # Display market ticker
            st.markdown(f"**Ticker:** `{market.get('ticker', 'N/A')}`")

            # Display resolution criteria
            if market.get('rules'):
                with st.expander("📜 Resolution Criteria", expanded=False):
                    st.markdown(market.get('rules'))

        with col2:
            # Key dates and metrics
            st.markdown("### Key Information")

            # Market status
            status = market.get('status', 'N/A')
            status_emoji = {'open': '🟢', 'closed': '🔴', 'settled': '✅'}.get(status.lower(), '⚪')
            st.markdown(f"**Status:** {status_emoji} {status}")

            # Important dates
            if market.get('close_time'):
                st.markdown(f"**Close Time:** {market.get('close_time')}")
            if market.get('expiration_time'):
                st.markdown(f"**Expiration:** {market.get('expiration_time')}")
            if market.get('expected_expiration_time'):
                st.markdown(f"**Expected Expiration:** {market.get('expected_expiration_time')}")

            # Current price
            if market.get('yes_bid'):
                st.markdown(f"**Yes Bid:** ¢{market.get('yes_bid', 0)}")
            if market.get('yes_ask'):
                st.markdown(f"**Yes Ask:** ¢{market.get('yes_ask', 0)}")

            # Volume information
            if market.get('volume'):
                st.markdown(f"**Total Volume:** {market.get('volume', 0):,} contracts")
            if market.get('open_interest'):
                st.markdown(f"**Open Interest:** {market.get('open_interest', 0):,} contracts")

            # Calculate 7-day volume from trade data
            if 'df' in st.session_state and st.session_state.df is not None and not st.session_state.df.empty:
                df_temp = st.session_state.df

                # Filter for last 7 days
                if 'timestamp' in df_temp.columns:
                    cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
                    df_7d = df_temp[df_temp['timestamp'] >= cutoff_time]

                    if len(df_7d) > 0:
                        # Calculate 7-day volume in contracts
                        if 'count' in df_7d.columns:
                            volume_7d_contracts = df_7d['count'].sum()
                            volume_7d_orders = len(df_7d)
                            st.markdown(f"**Volume (7d):** {volume_7d_contracts:,} contracts / {volume_7d_orders:,} orders")
                        else:
                            st.markdown(f"**Volume (7d):** {len(df_7d):,} orders")
                    else:
                        st.markdown(f"**Volume (7d):** 0 contracts / 0 orders")

                # Total trades fetched
                total_trades = len(df_temp)
                st.markdown(f"**Total Trades (fetched):** {total_trades:,} orders")

                # Calculate average volume per trade
                if 'count' in df_temp.columns:
                    avg_volume_per_trade = df_temp['count'].mean()
                    st.markdown(f"**Avg Volume/Trade:** {avg_volume_per_trade:.1f} contracts")

            if market.get('liquidity'):
                st.markdown(f"**Liquidity:** {market.get('liquidity', 0):,} contracts")

        st.markdown("---")

    # Display Order Book
    if st.session_state.orderbook is not None:
        st.header("📊 Order Book")

        orderbook_data = st.session_state.orderbook.get('orderbook', {})
        yes_bids = orderbook_data.get('yes', [])
        no_bids = orderbook_data.get('no', [])

        # Convert No bids to Yes asks
        # A No bid at price X is equivalent to a Yes ask at (100-X)
        yes_asks = [[100 - price, quantity] for price, quantity in no_bids] if no_bids else []

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Yes Bids")
            if yes_bids and len(yes_bids) > 0:
                yes_bid_df = pd.DataFrame(yes_bids, columns=['Price (¢)', 'Quantity'])
                yes_bid_df = yes_bid_df.sort_values('Price (¢)', ascending=False)
                st.dataframe(yes_bid_df, hide_index=True, width='stretch')

                # Summary stats
                total_yes_bid_quantity = sum([x[1] for x in yes_bids])
                best_yes_bid = max([x[0] for x in yes_bids]) if yes_bids else 0
                st.caption(f"Total Quantity: {total_yes_bid_quantity:,} | Best Bid: ¢{best_yes_bid}")
            else:
                st.info("No Yes bids available")

        with col2:
            st.subheader("Yes Asks")
            if yes_asks and len(yes_asks) > 0:
                yes_ask_df = pd.DataFrame(yes_asks, columns=['Price (¢)', 'Quantity'])
                yes_ask_df = yes_ask_df.sort_values('Price (¢)', ascending=True)
                st.dataframe(yes_ask_df, hide_index=True, width='stretch')

                # Summary stats
                total_yes_ask_quantity = sum([x[1] for x in yes_asks])
                best_yes_ask = min([x[0] for x in yes_asks]) if yes_asks else 0
                st.caption(f"Total Quantity: {total_yes_ask_quantity:,} | Best Ask: ¢{best_yes_ask}")
            else:
                st.info("No Yes asks available")

        # Helpful explanation
        st.info("💡 **Note:** Yes asks are derived from No bids (a No bid at price X = Yes ask at 100-X)")

        st.markdown("---")

    # Debug info (collapsible)
    with st.expander("🔍 Debug Info - Available Data Fields"):
        st.write("**DataFrame columns:**", list(df.columns))
        if not df.empty:
            st.write("**First trade sample:**")
            st.json(df.iloc[0].to_dict(), expanded=False)
        if st.session_state.market_info:
            st.write("**Market Info:**")
            st.json(st.session_state.market_info, expanded=False)
        if st.session_state.orderbook:
            st.write("**Orderbook:**")
            st.json(st.session_state.orderbook, expanded=False)

    # Display summary metrics
    st.subheader("Market Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", len(df))

    with col2:
        if 'price' in df.columns:
            latest_price = df.iloc[-1]['price']
            st.metric("Latest Price", f"¢{latest_price:.2f}")

    with col3:
        if 'count' in df.columns:
            total_volume = df['count'].sum()
            st.metric("Total Volume", f"{total_volume:,}")

    with col4:
        if 'signed_volume' in df.columns:
            net_volume = df['signed_volume'].sum()
            st.metric("Net Volume", f"{net_volume:,}")

    # Display risk metrics and market attractiveness
    if st.session_state.risk_metrics is not None:
        st.markdown("---")
        st.subheader("💰 Market Attractiveness")

        # Time period selector
        lookback_period = st.selectbox(
            "Analysis Period",
            options=[7, 60],
            format_func=lambda x: f"Last {x} Days",
            index=0
        )

        # Filter dataframe by lookback period
        df_filtered = df.copy()
        if 'timestamp' in df.columns:
            cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_period)
            df_filtered = df_filtered[df_filtered['timestamp'] >= cutoff_time]

        # Recalculate risk metrics for filtered data if needed
        if len(df_filtered) != len(df):
            service = KalshiService(use_demo=st.session_state.use_demo)
            risk_metrics_filtered = service.calculate_trade_risk_metrics(df_filtered, risk_aversion_k=risk_aversion_k)
            risk_metrics_filtered['risk_aversion_k'] = risk_aversion_k
        else:
            risk_metrics_filtered = st.session_state.risk_metrics

        risk_metrics = risk_metrics_filtered

        current_k = risk_metrics.get('risk_aversion_k', 0.25)
        st.info(f"""
        **Risk Analysis (Last {lookback_period} Days):**
        - Analyzed {risk_metrics['num_trades_with_exits']:,} trades with exit prices (out of {risk_metrics['num_trades_total']:,} total trades)
        - Exit prices are determined by finding the first same-side trade within 24 hours (excluding first 10 minutes)
        - P&L calculated from the maker's perspective
        - Risk aversion constant (k): {current_k:.2f}
        """)

        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

        with risk_col1:
            st.metric(
                "Adverse Selection (per unit)",
                f"¢{risk_metrics['adverse_selection_per_unit']:.4f}",
                help="Average P&L across all trades - represents expected loss from being picked off by informed traders"
            )

        with risk_col2:
            st.metric(
                "Inventory Risk (per unit)",
                f"¢{risk_metrics['inventory_risk_per_unit']:.4f}",
                help="Standard deviation of P&L × risk aversion constant (0.5) - represents risk from holding inventory"
            )

        with risk_col3:
            st.metric(
                "Required Half-Spread",
                f"¢{risk_metrics['required_half_spread']:.4f}",
                help="Required half-spread to cover adverse selection and inventory risk"
            )

        with risk_col4:
            st.metric(
                "Required Full Spread",
                f"¢{risk_metrics['required_full_spread']:.4f}",
                help="Required full bid-ask spread (2× half-spread) for sustainable market making (excludes fees)"
            )

        # Additional insights
        if risk_metrics['required_full_spread'] > 0:
            current_market = st.session_state.market_info.get('market', {})
            yes_bid = current_market.get('yes_bid', 0)
            yes_ask = current_market.get('yes_ask', 0)
            if yes_bid and yes_ask:
                current_spread = yes_ask - yes_bid
                midpoint_price = (yes_bid + yes_ask) / 2
                maker_fee_cents = fee_per_contract_cents(midpoint_price, rate=MAKER_FEE_RATE)
                roundtrip_fee_cents = maker_fee_cents * 2
                required_with_fees = risk_metrics['required_full_spread'] + roundtrip_fee_cents
                st.markdown(f"""
                **Spread Comparison:**
                - Current market spread: ¢{current_spread:.2f}
                - Required spread (risk only): ¢{risk_metrics['required_full_spread']:.4f}
                - Required spread + est. roundtrip maker fees (¢{roundtrip_fee_cents:.2f}): ¢{required_with_fees:.4f}
                - {'✅ Current spread is sufficient' if current_spread >= required_with_fees else '⚠️ Current spread may be too narrow after fees'}
                """)
        else:
            st.info("Insufficient data to calculate risk metrics (no trades with exit prices found)")

        # Information Risk Assessment
        st.markdown("---")
        st.markdown("### 📰 Information Risk Assessment")
        st.caption("AI-powered assessment of likelihood that market-moving information will be released in the next 7 days")

        if st.button("🤖 Assess Information Risk", help="Uses OpenAI API to evaluate likelihood of upcoming market-moving news"):
            with st.spinner("Analyzing market information risk..."):
                info_risk = run_information_risk_auto(force=True)

                if info_risk.get('error'):
                    st.error(f"⚠️ {info_risk.get('rationale', info_risk.get('error'))}")
                    if info_risk.get('error') and 'OPENAI_API_KEY' in info_risk.get('error', ''):
                        st.info("💡 To use this feature, add `OPENAI_API_KEY=your-api-key` to your .env file")
                else:
                    risk_col1, risk_col2 = st.columns([1, 3])

                    with risk_col1:
                        st.metric(
                            "Information Risk",
                            info_risk['probability'],
                            help="Likelihood that market-moving information will be released in next 7 days"
                        )

                    with risk_col2:
                        st.markdown("**Rationale:**")
                        st.write(info_risk['rationale'])

        # Display cached assessment if available
        elif 'info_risk' in st.session_state and st.session_state.info_risk:
            info_risk = st.session_state.info_risk
            if not info_risk.get('error'):
                risk_col1, risk_col2 = st.columns([1, 3])

                with risk_col1:
                    st.metric(
                        "Information Risk",
                        info_risk['probability'],
                        help="Likelihood that market-moving information will be released in next 7 days"
                    )

                with risk_col2:
                    st.markdown("**Rationale:**")
                    st.write(info_risk['rationale'])
            else:
                st.error(f"⚠️ {info_risk.get('rationale', info_risk.get('error'))}")
                if info_risk.get('error') and 'OPENAI_API_KEY' in info_risk.get('error', ''):
                    st.info("💡 To use this feature, add `OPENAI_API_KEY=your-api-key` to your .env file")

        # Simulated Profit Analysis
        st.markdown("---")
        st.markdown("### 💵 Simulated Profit Analysis")

        # Get current best bid/ask from orderbook
        if st.session_state.orderbook is not None:
            orderbook_data = st.session_state.orderbook.get('orderbook', {})
            yes_bids = orderbook_data.get('yes', [])
            no_bids = orderbook_data.get('no', [])

            if yes_bids and no_bids:
                current_best_bid = max([x[0] for x in yes_bids]) if yes_bids else 0
                # Convert No bids to Yes asks (No bid at X = Yes ask at 100-X)
                yes_asks_from_no = [100 - x[0] for x in no_bids]
                current_best_ask = min(yes_asks_from_no) if yes_asks_from_no else 100

                st.write(f"**Current Market:** Bid = ¢{current_best_bid:.2f}, Ask = ¢{current_best_ask:.2f}")

                # Calculate fills at current bid/ask over the historical period
                if 'action' in df_filtered.columns and 'adjusted_price' in df_filtered.columns and 'count' in df_filtered.columns:
                    # Your bid gets filled when someone sells Yes at or below your bid
                    bid_fills = df_filtered[(df_filtered['action'] == 'Sell Yes') & (df_filtered['adjusted_price'] <= current_best_bid)]
                    bid_volume = bid_fills['count'].sum() if len(bid_fills) > 0 else 0

                    # Your ask gets filled when someone buys Yes at or above your ask
                    ask_fills = df_filtered[(df_filtered['action'] == 'Buy Yes') & (df_filtered['adjusted_price'] >= current_best_ask)]
                    ask_volume = ask_fills['count'].sum() if len(ask_fills) > 0 else 0

                    # Net position
                    net_position = bid_volume - ask_volume

                    # Calculate realized P&L (from matched trades)
                    matched_volume = min(bid_volume, ask_volume)
                    realized_pnl = matched_volume * (current_best_ask - current_best_bid) if matched_volume > 0 else 0

                    # Calculate unrealized P&L (from net position)
                    # Cost to acquire the net position
                    unrealized_calc_text = ""
                    if net_position > 0:
                        # Net long: bought at bid price
                        position_cost = net_position * current_best_bid
                        # To liquidate long position, we need to sell (hit bids in orderbook)
                        # Sort bids by price descending (best prices first)
                        sorted_bids = sorted(yes_bids, key=lambda x: x[0], reverse=True)
                        liquidation_avg_price, liquidation_value, breakdown = calculate_liquidation_price(
                            net_position, 'sell', sorted_bids
                        )
                        unrealized_pnl = liquidation_value - position_cost
                        unrealized_calc_text = f"Long {net_position:,} @ ¢{current_best_bid:.2f} → Sell ({breakdown}) @ avg ¢{liquidation_avg_price:.2f} = ${unrealized_pnl/100:,.2f}"
                    elif net_position < 0:
                        # Net short: sold at ask price
                        position_value = abs(net_position) * current_best_ask
                        # To close short position, we need to buy (lift asks in orderbook)
                        # Sort asks by price ascending (best prices first)
                        sorted_asks = sorted([[100 - x[0], x[1]] for x in no_bids], key=lambda x: x[0])
                        liquidation_avg_price, liquidation_cost, breakdown = calculate_liquidation_price(
                            abs(net_position), 'buy', sorted_asks
                        )
                        unrealized_pnl = position_value - liquidation_cost
                        unrealized_calc_text = f"Short {abs(net_position):,} @ ¢{current_best_ask:.2f} → Buy ({breakdown}) @ avg ¢{liquidation_avg_price:.2f} = ${unrealized_pnl/100:,.2f}"
                    else:
                        unrealized_pnl = 0
                        unrealized_calc_text = "No net position"

                    total_pnl = realized_pnl + unrealized_pnl

                    # Display current bid/ask results
                    sim_col1, sim_col2, sim_col3 = st.columns(3)

                    with sim_col1:
                        st.metric("Bid Volume Filled", f"{bid_volume:,}")
                        st.metric("Ask Volume Filled", f"{ask_volume:,}")

                    with sim_col2:
                        st.metric("Net Position", f"{net_position:+,}")
                        st.metric("Realized P&L", f"${realized_pnl/100:,.2f}")
                        st.caption(f"{matched_volume:,} matched × (¢{current_best_ask:.2f} - ¢{current_best_bid:.2f})")

                    with sim_col3:
                        st.metric("Unrealized P&L", f"${unrealized_pnl/100:,.2f}")
                        st.caption(unrealized_calc_text)
                        st.metric("Total P&L", f"${total_pnl/100:,.2f}", delta=None)

                    # Optimize bid/ask combinations
                    st.markdown("---")
                    st.markdown("**🎯 Profit-Maximizing Bid/Ask Optimization**")

                    with st.spinner("Finding optimal bid/ask combination..."):
                        best_profit = float('-inf')
                        best_bid_opt = 0
                        best_ask_opt = 0
                        best_details = {}

                        # Prepare sorted orderbook levels for liquidation calculations
                        sorted_bids = sorted(yes_bids, key=lambda x: x[0], reverse=True)
                        sorted_asks = sorted([[100 - x[0], x[1]] for x in no_bids], key=lambda x: x[0])

                        # Search through price combinations
                        for test_bid in range(0, 100, 1):  # Test every 1 cent
                            for test_ask in range(test_bid + 1, 101, 1):  # Ask must be higher than bid
                                # Calculate fills for this combination
                                test_bid_fills = df_filtered[(df_filtered['action'] == 'Sell Yes') & (df_filtered['adjusted_price'] <= test_bid)]
                                test_ask_fills = df_filtered[(df_filtered['action'] == 'Buy Yes') & (df_filtered['adjusted_price'] >= test_ask)]

                                if len(test_bid_fills) == 0 and len(test_ask_fills) == 0:
                                    continue

                                test_bid_volume = test_bid_fills['count'].sum() if len(test_bid_fills) > 0 else 0
                                test_ask_volume = test_ask_fills['count'].sum() if len(test_ask_fills) > 0 else 0

                                if test_bid_volume == 0 and test_ask_volume == 0:
                                    continue

                                test_net_position = test_bid_volume - test_ask_volume

                                # Calculate realized P&L
                                test_matched_volume = min(test_bid_volume, test_ask_volume)
                                test_realized_pnl = test_matched_volume * (test_ask - test_bid) if test_matched_volume > 0 else 0

                                # Calculate unrealized P&L based on liquidating through orderbook
                                if test_net_position > 0:
                                    # Net long: bought at test_bid, liquidate by selling through orderbook
                                    test_position_cost = test_net_position * test_bid
                                    _, test_liquidation_value, _ = calculate_liquidation_price(
                                        test_net_position, 'sell', sorted_bids
                                    )
                                    test_unrealized_pnl = test_liquidation_value - test_position_cost
                                elif test_net_position < 0:
                                    # Net short: sold at test_ask, close by buying through orderbook
                                    test_position_value = abs(test_net_position) * test_ask
                                    _, test_liquidation_cost, _ = calculate_liquidation_price(
                                        abs(test_net_position), 'buy', sorted_asks
                                    )
                                    test_unrealized_pnl = test_position_value - test_liquidation_cost
                                else:
                                    test_unrealized_pnl = 0

                                test_total_pnl = test_realized_pnl + test_unrealized_pnl

                                if test_total_pnl > best_profit:
                                    best_profit = test_total_pnl
                                    best_bid_opt = test_bid
                                    best_ask_opt = test_ask
                                    best_details = {
                                        'bid_volume': test_bid_volume,
                                        'ask_volume': test_ask_volume,
                                        'net_position': test_net_position,
                                        'realized_pnl': test_realized_pnl,
                                        'unrealized_pnl': test_unrealized_pnl,
                                        'total_pnl': test_total_pnl
                                    }

                        # Display optimal results
                        if best_profit > float('-inf'):
                            st.success(f"**Optimal Strategy:** Bid = ¢{best_bid_opt:.2f}, Ask = ¢{best_ask_opt:.2f}")

                            # Calculate optimal unrealized calc text with orderbook depth
                            opt_matched_volume = min(best_details['bid_volume'], best_details['ask_volume'])
                            opt_unrealized_calc_text = ""
                            if best_details['net_position'] > 0:
                                opt_liquidation_avg_price, _, opt_breakdown = calculate_liquidation_price(
                                    best_details['net_position'], 'sell', sorted_bids
                                )
                                opt_unrealized_calc_text = f"Long {best_details['net_position']:,} @ ¢{best_bid_opt:.2f} → Sell ({opt_breakdown}) @ avg ¢{opt_liquidation_avg_price:.2f} = ${best_details['unrealized_pnl']/100:,.2f}"
                            elif best_details['net_position'] < 0:
                                opt_liquidation_avg_price, _, opt_breakdown = calculate_liquidation_price(
                                    abs(best_details['net_position']), 'buy', sorted_asks
                                )
                                opt_unrealized_calc_text = f"Short {abs(best_details['net_position']):,} @ ¢{best_ask_opt:.2f} → Buy ({opt_breakdown}) @ avg ¢{opt_liquidation_avg_price:.2f} = ${best_details['unrealized_pnl']/100:,.2f}"
                            else:
                                opt_unrealized_calc_text = "No net position"

                            opt_col1, opt_col2, opt_col3 = st.columns(3)

                            with opt_col1:
                                st.metric("Optimal Bid Volume", f"{best_details['bid_volume']:,}")
                                st.metric("Optimal Ask Volume", f"{best_details['ask_volume']:,}")

                            with opt_col2:
                                st.metric("Optimal Net Position", f"{best_details['net_position']:+,}")
                                st.metric("Optimal Realized P&L", f"${best_details['realized_pnl']/100:,.2f}")
                                st.caption(f"{opt_matched_volume:,} matched × (¢{best_ask_opt:.2f} - ¢{best_bid_opt:.2f})")

                            with opt_col3:
                                st.metric("Optimal Unrealized P&L", f"${best_details['unrealized_pnl']/100:,.2f}")
                                st.caption(opt_unrealized_calc_text)
                                st.metric("Optimal Total P&L", f"${best_details['total_pnl']/100:,.2f}", delta=f"+${(best_details['total_pnl']-total_pnl)/100:,.2f}" if best_details['total_pnl'] > total_pnl else None)

                            # Comparison
                            st.info(f"""
                            **Comparison:**
                            - Current market P&L: ${total_pnl/100:,.2f}
                            - Optimal P&L: ${best_details['total_pnl']/100:,.2f}
                            - Improvement: ${(best_details['total_pnl']-total_pnl)/100:+,.2f}
                            """)
                        else:
                            st.warning("No profitable bid/ask combination found for this period.")
            else:
                st.warning("Orderbook data not available or incomplete.")
        else:
            st.warning("No orderbook data available for simulated profit analysis.")

    # Create visualizations
    st.subheader("Market Activity Visualization")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Trade Price Over Time', 'Signed Volume Over Time'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    # Plot 1: Trade Price Over Time (separate lines for Buy Yes and Sell Yes)
    if 'timestamp' in df.columns and 'adjusted_price' in df.columns and 'action' in df.columns:
        # Buy Yes trades
        buy_yes_df_chart = df[df['action'] == 'Buy Yes']
        if len(buy_yes_df_chart) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_yes_df_chart['timestamp'],
                    y=buy_yes_df_chart['adjusted_price'],
                    mode='lines+markers',
                    name='Buy Yes Price',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=5, symbol='triangle-up'),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Buy Yes Price:</b> ¢%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Sell Yes trades
        sell_yes_df_chart = df[df['action'] == 'Sell Yes']
        if len(sell_yes_df_chart) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_yes_df_chart['timestamp'],
                    y=sell_yes_df_chart['adjusted_price'],
                    mode='lines+markers',
                    name='Sell Yes Price',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=5, symbol='triangle-down'),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Sell Yes Price:</b> ¢%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

    # Plot 2: Signed Volume Over Time
    if 'timestamp' in df.columns and 'signed_volume' in df.columns:
        # Color bars based on positive/negative
        colors = ['green' if x >= 0 else 'red' for x in df['signed_volume']]

        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['signed_volume'],
                name='Signed Volume',
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                hovertemplate='<b>Time:</b> %{x}<br><b>Volume:</b> %{y}<br><extra></extra>'
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (¢)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (+ Buy Yes, - Sell Yes)", row=2, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    st.plotly_chart(fig, width="stretch")

    # Display raw data table
    with st.expander("View Raw Trade Data"):
        # Select relevant columns for display
        display_columns = []
        if 'timestamp' in df.columns:
            display_columns.append('timestamp')
        if 'action' in df.columns:
            display_columns.append('action')
        if 'adjusted_price' in df.columns:
            display_columns.append('adjusted_price')
        if 'count' in df.columns:
            display_columns.append('count')
        if 'signed_volume' in df.columns:
            display_columns.append('signed_volume')
        if 'exit_price' in df.columns:
            display_columns.append('exit_price')
        if 'pnl' in df.columns:
            display_columns.append('pnl')

        if display_columns:
            display_df = df[display_columns].sort_values('timestamp', ascending=False).copy()
            # Format price columns
            if 'adjusted_price' in display_df.columns:
                display_df['adjusted_price'] = display_df['adjusted_price'].apply(lambda x: f"¢{x:.2f}")
                display_df.rename(columns={'adjusted_price': 'price'}, inplace=True)
            if 'exit_price' in display_df.columns:
                display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"¢{x:.2f}" if pd.notna(x) else "N/A")
            if 'pnl' in display_df.columns:
                display_df['pnl'] = display_df['pnl'].apply(lambda x: f"¢{x:+.4f}" if pd.notna(x) else "N/A")
            st.dataframe(display_df, width="stretch")
        else:
            st.dataframe(df, width="stretch")

    # Additional statistics
    st.subheader("Trade Statistics")

    # Calculate buy/sell statistics
    if 'action' in df.columns and 'count' in df.columns and 'adjusted_price' in df.columns:
        buy_yes_df = df[df['action'] == 'Buy Yes']
        sell_yes_df = df[df['action'] == 'Sell Yes']

        total_buy_yes = buy_yes_df['count'].sum() if len(buy_yes_df) > 0 else 0
        total_sell_yes = sell_yes_df['count'].sum() if len(sell_yes_df) > 0 else 0

        avg_price_buy_yes = buy_yes_df['adjusted_price'].mean() if len(buy_yes_df) > 0 else 0
        avg_price_sell_yes = sell_yes_df['adjusted_price'].mean() if len(sell_yes_df) > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'price' in df.columns:
            st.markdown("**Price Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Min Price', 'Max Price', 'Mean Price', 'Median Price', 'Std Dev'],
                'Value': [
                    f"¢{df['price'].min():.2f}",
                    f"¢{df['price'].max():.2f}",
                    f"¢{df['price'].mean():.2f}",
                    f"¢{df['price'].median():.2f}",
                    f"¢{df['price'].std():.2f}"
                ]
            })
            st.table(stats_df)

    with col2:
        if 'action' in df.columns:
            st.markdown("**Buy Yes Statistics**")
            buy_yes_stats_df = pd.DataFrame({
                'Metric': ['Total Volume', 'Avg Price', 'Number of Trades'],
                'Value': [
                    f"{total_buy_yes:,}",
                    f"¢{avg_price_buy_yes:.2f}",
                    f"{len(buy_yes_df):,}"
                ]
            })
            st.table(buy_yes_stats_df)

    with col3:
        if 'action' in df.columns:
            st.markdown("**Sell Yes Statistics**")
            sell_yes_stats_df = pd.DataFrame({
                'Metric': ['Total Volume', 'Avg Price', 'Number of Trades'],
                'Value': [
                    f"{total_sell_yes:,}",
                    f"¢{avg_price_sell_yes:.2f}",
                    f"{len(sell_yes_df):,}"
                ]
            })
            st.table(sell_yes_stats_df)

    # Mock Order Calculator
    st.subheader("Mock Order Calculator")
    st.markdown("Calculate what would have happened if you had standing bid/ask orders at specific prices")

    calc_col1, calc_col2, calc_col3 = st.columns([1, 1, 1])

    # Initialize session state for bid/ask if not exists
    # We need to set default values BEFORE the widgets are created
    if 'bid_price_value' not in st.session_state:
        st.session_state.bid_price_value = 45.0
    if 'ask_price_value' not in st.session_state:
        st.session_state.ask_price_value = 55.0
    if 'is_optimized' not in st.session_state:
        st.session_state.is_optimized = False

    log_event(f"Loaded mock order defaults (bid ¢{st.session_state.bid_price_value:.2f}, ask ¢{st.session_state.ask_price_value:.2f})")

    with calc_col1:
        bid_price = st.number_input(
            "Your Bid Price (¢) - You buy Yes at this price",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.bid_price_value,
            step=0.5,
            help="Price at which you would buy Yes (gets filled when someone sells Yes at or below this price)",
            key="bid_price_input",
            on_change=lambda: setattr(st.session_state, 'is_optimized', False)
        )

    with calc_col2:
        ask_price = st.number_input(
            "Your Ask Price (¢) - You sell Yes at this price",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.ask_price_value,
            step=0.5,
            help="Price at which you would sell Yes (gets filled when someone buys Yes at or above this price)",
            key="ask_price_input",
            on_change=lambda: setattr(st.session_state, 'is_optimized', False)
        )

    with calc_col3:
        lookback_days = st.number_input(
            "Lookback Period (days)",
            min_value=0.0,
            max_value=365.0,
            value=7.0,
            step=1.0,
            help="Only consider trades from the last N days for optimization (0 = use all trades)",
            key="lookback_days_input"
        )

        if st.button("🎯 Optimize for Market → 0", help="Find bid/ask prices that maximize realized P&L minus absolute position cost"):
            log_event(f"Starting optimization with lookback {lookback_days:.0f} days (bid ¢{bid_price:.2f}, ask ¢{ask_price:.2f})")

            # Filter dataframe by lookback period if specified
            df_filtered = df.copy()
            if lookback_days > 0 and 'timestamp' in df.columns:
                import pandas as pd
                # Get current time with UTC timezone to match the dataframe timestamps
                cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
                df_filtered = df_filtered[df_filtered['timestamp'] >= cutoff_time]
                log_event(f"Filtered trades to {len(df_filtered)} rows for last {lookback_days:.0f} days (from {len(df)})")
            else:
                log_event(f"Using all {len(df)} trades for optimization")

            # Find optimal bid/ask that maximizes realized P&L - abs(position cost)
            best_pnl = float('-inf')
            best_bid = 0
            best_ask = 0

            # Search through price combinations
            for test_bid in range(0, 100, 1):  # Test every 1 cent
                for test_ask in range(test_bid + 1, 101, 1):  # Ask must be higher than bid
                    # Calculate fills for this combination using filtered data
                    test_bid_fills = df_filtered[(df_filtered['action'] == 'Sell Yes') & (df_filtered['adjusted_price'] <= test_bid)]
                    test_ask_fills = df_filtered[(df_filtered['action'] == 'Buy Yes') & (df_filtered['adjusted_price'] >= test_ask)]

                    if len(test_bid_fills) == 0 and len(test_ask_fills) == 0:
                        continue

                    test_bid_volume = test_bid_fills['count'].sum() if len(test_bid_fills) > 0 else 0
                    test_ask_volume = test_ask_fills['count'].sum() if len(test_ask_fills) > 0 else 0

                    if test_bid_volume == 0 and test_ask_volume == 0:
                        continue

                    test_bid_avg_price = test_bid_fills['adjusted_price'].mean() if len(test_bid_fills) > 0 else 0
                    test_ask_avg_price = test_ask_fills['adjusted_price'].mean() if len(test_ask_fills) > 0 else 0

                    test_net_position = test_bid_volume - test_ask_volume

                    # Calculate realized P&L
                    test_matched_volume = min(test_bid_volume, test_ask_volume)
                    test_realized_pnl = test_matched_volume * (test_ask_avg_price - test_bid_avg_price) if test_matched_volume > 0 else 0

                    # Calculate optimization metric: realized P&L - abs(position cost)
                    # This penalizes having a net position (either long or short)
                    if test_net_position > 0:
                        test_position_cost = test_net_position * test_bid_avg_price
                    elif test_net_position < 0:
                        test_position_cost = abs(test_net_position) * test_ask_avg_price
                    else:
                        test_position_cost = 0

                    test_optimization_metric = test_realized_pnl - abs(test_position_cost)

                    if test_optimization_metric > best_pnl:
                        best_pnl = test_optimization_metric
                        best_bid = test_bid
                        best_ask = test_ask

            log_event(f"Optimization complete → bid ¢{best_bid:.2f}, ask ¢{best_ask:.2f}, score {best_pnl:.2f}")

            # Update the value storage variables (these will be used on next rerun)
            st.session_state.bid_price_value = float(best_bid)
            st.session_state.ask_price_value = float(best_ask)
            st.session_state.is_optimized = True

            # Delete the widget keys to force them to recreate with new values
            if 'bid_price_input' in st.session_state:
                del st.session_state.bid_price_input
            if 'ask_price_input' in st.session_state:
                del st.session_state.ask_price_input

            st.rerun()

    # Show optimization banner if optimized values were just set
    if st.session_state.get('is_optimized', False):
        st.info(f"🎯 Using optimized prices: Bid = ¢{bid_price:.2f}, Ask = ¢{ask_price:.2f} (maximizes realized P&L - |position cost|)")
        if st.button("Reset to Defaults"):
            log_event("Reset mock order inputs to defaults")
            st.session_state.bid_price_value = 45.0
            st.session_state.ask_price_value = 55.0
            st.session_state.is_optimized = False

            # Delete the widget keys to force them to recreate with new values
            if 'bid_price_input' in st.session_state:
                del st.session_state.bid_price_input
            if 'ask_price_input' in st.session_state:
                del st.session_state.ask_price_input

            st.rerun()

    if st.session_state.get('activity_log'):
        with st.expander("Activity Log"):
            for entry in st.session_state.activity_log[-20:]:
                st.write(entry)

    if bid_price >= ask_price:
        st.warning("⚠️ Bid price should be lower than ask price for a valid market-making spread")

    if 'action' in df.columns and 'adjusted_price' in df.columns and 'count' in df.columns:
        # Calculate fills
        # Your bid gets filled when someone sells Yes (action='Sell Yes') at or below your bid price
        bid_fills = df[(df['action'] == 'Sell Yes') & (df['adjusted_price'] <= bid_price)]
        bid_volume = bid_fills['count'].sum() if len(bid_fills) > 0 else 0
        bid_avg_price = bid_fills['adjusted_price'].mean() if len(bid_fills) > 0 else 0

        # Your ask gets filled when someone buys Yes (action='Buy Yes') at or above your ask price
        ask_fills = df[(df['action'] == 'Buy Yes') & (df['adjusted_price'] >= ask_price)]
        ask_volume = ask_fills['count'].sum() if len(ask_fills) > 0 else 0
        ask_avg_price = ask_fills['adjusted_price'].mean() if len(ask_fills) > 0 else 0

        # Calculate P&L
        # Assume market ends at 50¢ (or use latest price as proxy)
        settlement_price = df['adjusted_price'].iloc[-1] if len(df) > 0 else 50.0

        # Net position
        net_position = bid_volume - ask_volume

        # Calculate realized and unrealized P&L
        # Realized P&L: profit from matched buy/sell pairs
        matched_volume = min(bid_volume, ask_volume)
        realized_pnl = matched_volume * (ask_avg_price - bid_avg_price) if matched_volume > 0 else 0

        # Unrealized P&L: mark-to-market on net position
        if net_position > 0:
            # Net long: bought more than sold
            # Cost basis is the average buy price
            unrealized_pnl = net_position * (settlement_price - bid_avg_price)
            position_cost = net_position * bid_avg_price
            position_value = net_position * settlement_price
        elif net_position < 0:
            # Net short: sold more than bought
            # Sold at average ask price, mark to market at settlement
            unrealized_pnl = abs(net_position) * (ask_avg_price - settlement_price)
            position_cost = abs(net_position) * settlement_price  # What we'd pay to close
            position_value = abs(net_position) * ask_avg_price    # What we received
        else:
            unrealized_pnl = 0
            position_cost = 0
            position_value = 0

        # Total P&L
        total_pnl = realized_pnl + unrealized_pnl

        # Display results
        st.markdown("---")
        st.markdown("### Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.markdown("**Bid Fills (You Bought Yes)**")
            bid_results_df = pd.DataFrame({
                'Metric': ['Volume Filled', 'Avg Fill Price', 'Number of Fills'],
                'Value': [
                    f"{bid_volume:,}",
                    f"¢{bid_avg_price:.2f}" if bid_volume > 0 else "N/A",
                    f"{len(bid_fills):,}"
                ]
            })
            st.table(bid_results_df)

        with result_col2:
            st.markdown("**Ask Fills (You Sold Yes)**")
            ask_results_df = pd.DataFrame({
                'Metric': ['Volume Filled', 'Avg Fill Price', 'Number of Fills'],
                'Value': [
                    f"{ask_volume:,}",
                    f"¢{ask_avg_price:.2f}" if ask_volume > 0 else "N/A",
                    f"{len(ask_fills):,}"
                ]
            })
            st.table(ask_results_df)

        with result_col3:
            st.markdown("**Position & P&L**")

            # Build the metrics list dynamically
            metrics = []
            values = []

            # Net position
            metrics.append('Net Position')
            values.append(f"{net_position:+,}")

            # Settlement price
            metrics.append('Settlement Price')
            values.append(f"¢{settlement_price:.2f}")

            # Realized P&L
            metrics.append('Realized P&L')
            values.append(f"${realized_pnl/100:+,.2f}")

            # Unrealized P&L breakdown
            if net_position != 0:
                metrics.append('Position Cost')
                values.append(f"${position_cost/100:,.2f}")

                metrics.append('Position Value')
                values.append(f"${position_value/100:,.2f}")

                metrics.append('Unrealized P&L')
                values.append(f"${unrealized_pnl/100:+,.2f}")

            # Total P&L
            metrics.append('Total P&L')
            values.append(f"${total_pnl/100:+,.2f}")

            # Total P&L if market resolves to 0
            # If market goes to 0, long positions lose their cost, short positions gain
            if net_position > 0:
                pnl_at_zero = realized_pnl - position_cost
            elif net_position < 0:
                pnl_at_zero = realized_pnl + position_value
            else:
                pnl_at_zero = realized_pnl

            metrics.append('P&L if Market → 0')
            values.append(f"${pnl_at_zero/100:+,.2f}")

            position_df = pd.DataFrame({
                'Metric': metrics,
                'Value': values
            })
            st.table(position_df)

            if total_pnl >= 0:
                st.success(f"💰 Total Profit: ${total_pnl/100:,.2f}")
            else:
                st.error(f"📉 Total Loss: ${total_pnl/100:,.2f}")

        # Additional insights
        st.markdown("**Trading Insights:**")
        insights = []

        if bid_volume > 0 and ask_volume > 0:
            spread_captured = ask_avg_price - bid_avg_price
            insights.append(f"✓ You captured an average spread of ¢{spread_captured:.2f}")

        if net_position > 0:
            insights.append(f"⚠️ You have a net long position of {net_position:,} contracts")
        elif net_position < 0:
            insights.append(f"⚠️ You have a net short position of {abs(net_position):,} contracts")
        else:
            insights.append(f"✓ You have a balanced position (net zero exposure)")

        if bid_volume == 0:
            insights.append(f"ℹ️ No bid fills - market never traded at or below ¢{bid_price:.2f}")

        if ask_volume == 0:
            insights.append(f"ℹ️ No ask fills - market never traded at or above ¢{ask_price:.2f}")

        for insight in insights:
            st.write(insight)

        st.info(f"**Note:** P&L calculation uses the latest trade price (¢{settlement_price:.2f}) as a proxy for settlement. Actual settlement depends on market outcome.")

else:
    # Show placeholder
    st.info("👈 Enter a market ticker in the sidebar and click 'Fetch Data' to get started")

    # Example tickers
    st.subheader("Example Market Tickers")
    st.markdown("""
    Try these example tickers (if available in demo environment):
    - `INXD-25JAN31-T4850` - S&P 500 index market
    - Or any valid market ticker from Kalshi

    **Note:** Make sure the ticker exists in your selected environment (Demo/Prod)
    """)
