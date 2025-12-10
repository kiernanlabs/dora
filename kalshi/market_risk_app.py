import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from kalshi_service import KalshiService
import traceback

# Page configuration
st.set_page_config(
    page_title="Kalshi Market Risk Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Kalshi Market Risk Analysis Dashboard")
st.markdown("Analyze trade price and volume patterns for Kalshi markets")

# Sidebar for configuration
st.sidebar.header("Configuration")
use_demo = st.sidebar.checkbox("Use Demo Environment", value=False)
ticker = st.sidebar.text_input(
    "Market Ticker",
    value="",
    placeholder="e.g., INXD-25JAN31-T4850"
)
limit = st.sidebar.slider("Number of Trades", min_value=100, max_value=2000, value=1000, step=100)

# Add explanatory text
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to use:
1. Enter a market ticker symbol
2. Select number of recent trades to fetch
3. Click "Fetch Data" to load and visualize

### What you'll see:
- **Trade Price Over Time**: Shows the yes price for each trade
- **Signed Volume Over Time**: Positive values = Yes/Buy trades, Negative values = No/Sell trades
""")

# Main content
if st.sidebar.button("Fetch Data", type="primary"):
    if not ticker:
        st.error("Please enter a market ticker")
    else:
        try:
            with st.spinner(f"Fetching {limit} trades for {ticker}..."):
                # Initialize service
                service = KalshiService(use_demo=use_demo)

                # Get trades as DataFrame
                df = service.get_trades_dataframe(ticker=ticker, limit=limit)

                if df.empty:
                    st.warning(f"No trades found for ticker: {ticker}")
                else:
                    # Debug info (collapsible)
                    with st.expander("🔍 Debug Info - Available Data Fields"):
                        st.write("**DataFrame columns:**", list(df.columns))
                        if not df.empty:
                            st.write("**First trade sample:**")
                            st.json(df.iloc[0].to_dict(), expanded=False)
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

                    # Create visualizations
                    st.subheader("Market Activity Visualization")

                    # Create subplots
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Trade Price Over Time', 'Signed Volume Over Time'),
                        vertical_spacing=0.12,
                        row_heights=[0.5, 0.5]
                    )

                    # Plot 1: Trade Price Over Time
                    if 'timestamp' in df.columns and 'price' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df['timestamp'],
                                y=df['price'],
                                mode='lines+markers',
                                name='Trade Price',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=4),
                                hovertemplate='<b>Time:</b> %{x}<br><b>Price:</b> ¢%{y:.2f}<extra></extra>'
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
                    fig.update_yaxes(title_text="Volume (+ Yes/Buy, - No/Sell)", row=2, col=1)

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
                        if 'price' in df.columns:
                            display_columns.append('price')
                        if 'count' in df.columns:
                            display_columns.append('count')
                        if 'signed_volume' in df.columns:
                            display_columns.append('signed_volume')

                        if display_columns:
                            display_df = df[display_columns].sort_values('timestamp', ascending=False).copy()
                            # Format price column
                            if 'price' in display_df.columns:
                                display_df['price'] = display_df['price'].apply(lambda x: f"¢{x:.2f}")
                            st.dataframe(display_df, width="stretch")
                        else:
                            st.dataframe(df, width="stretch")

                    # Additional statistics
                    st.subheader("Trade Statistics")

                    # Calculate buy/sell statistics
                    if 'action' in df.columns and 'count' in df.columns and 'price' in df.columns:
                        buy_yes_df = df[df['action'] == 'Buy Yes']
                        buy_no_df = df[df['action'] == 'Buy No']

                        total_buy_yes = buy_yes_df['count'].sum() if len(buy_yes_df) > 0 else 0
                        total_buy_no = buy_no_df['count'].sum() if len(buy_no_df) > 0 else 0

                        avg_price_buy_yes = buy_yes_df['price'].mean() if len(buy_yes_df) > 0 else 0
                        avg_price_buy_no = buy_no_df['price'].mean() if len(buy_no_df) > 0 else 0

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
                            st.markdown("**Buy No Statistics**")
                            buy_no_stats_df = pd.DataFrame({
                                'Metric': ['Total Volume', 'Avg Price', 'Number of Trades'],
                                'Value': [
                                    f"{total_buy_no:,}",
                                    f"¢{avg_price_buy_no:.2f}",
                                    f"{len(buy_no_df):,}"
                                ]
                            })
                            st.table(buy_no_stats_df)

        except FileNotFoundError as e:
            st.error(f"Configuration Error: {str(e)}")
            st.info("Make sure you have set up your .env file with DEMO_KEYID and DEMO_KEYFILE (or PROD_* for production)")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

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
