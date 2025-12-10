"""
Polymarket Wallet Analytics - Streamlit App

UI layer for analyzing individual wallet performance on Polymarket.
Shows all trades with current market prices and P&L calculations.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from user_evaluation_service import UserEvaluationService


# Initialize service
@st.cache_resource
def get_service():
    return UserEvaluationService()


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a currency value with a dollar sign and fixed decimals."""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a percentage value."""
    return f"{value:,.{decimals}f}%"


def render_wallet_input():
    """Render the wallet address input section."""
    st.header("Wallet Analysis")

    col1, col2 = st.columns([4, 1])
    with col1:
        wallet = st.text_input(
            "Wallet Address",
            placeholder="e.g., 0x1234567890abcdef...",
            help="Enter the wallet address to analyze"
        )
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze Wallet", type="primary")

    if analyze_btn and wallet:
        with st.spinner("Fetching wallet trades and current prices..."):
            try:
                service = get_service()
                result = service.analyze_wallet(wallet, limit=1000)

                if result and result["trades"]:
                    st.session_state["wallet_result"] = result
                    st.session_state["wallet_address"] = wallet
                    st.success(f"Analyzed {result['summary']['total_trades']} trades")
                else:
                    st.warning("No trades found for this wallet")
            except Exception as e:
                st.error(f"Error analyzing wallet: {e}")

    # Display results if available
    if "wallet_result" in st.session_state and st.session_state["wallet_result"]:
        render_wallet_results(
            st.session_state["wallet_address"],
            st.session_state["wallet_result"]
        )


def render_wallet_results(wallet: str, result: dict):
    """Render the wallet analysis results."""
    st.divider()

    summary = result["summary"]
    trades = result["trades"]

    # Display wallet address
    st.subheader(f"Wallet: {wallet[:10]}...{wallet[-8:]}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", summary["total_trades"])
    with col2:
        pnl_color = "normal" if summary["total_pnl"] >= 0 else "inverse"
        st.metric(
            "Total P&L",
            format_currency(summary["total_pnl"]),
            delta=format_currency(summary["total_pnl"]) if summary["total_pnl"] != 0 else None
        )
    with col3:
        st.metric("Total Volume", format_currency(summary["total_volume"]))
    with col4:
        st.metric("Win Rate", format_percentage(summary["win_rate"]))

    # Additional metrics
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Profitable Trades", summary["profitable_trades"])
    with col6:
        st.metric("Losing Trades", summary["losing_trades"])

    # Trades table
    st.subheader("Recent Trades with Current Values")

    # Convert trades to dataframe
    trades_data = []
    for trade in sorted(trades, key=lambda t: t.timestamp, reverse=True):
        trades_data.append({
            "Datetime": trade.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "Market": trade.market_title[:50] + "..." if len(trade.market_title) > 50 else trade.market_title,
            "Outcome": trade.outcome,
            "Side": trade.side,
            "Entry Price": format_currency(trade.price),
            "Current Price": format_currency(trade.current_price),
            "Size": f"{trade.size:.2f}",
            "Entry Value": format_currency(trade.size * trade.price),
            "Current Value": format_currency(trade.current_value),
            "P&L": format_currency(trade.pnl),
            "Return %": f"{(trade.pnl / (trade.size * trade.price) * 100):.2f}%" if trade.size and trade.price else "0.00%"
        })

    df = pd.DataFrame(trades_data)

    # Apply color styling to P&L column
    def color_pnl(val):
        """Color P&L values: green for profit, red for loss."""
        if isinstance(val, str) and val.startswith("$"):
            num_val = float(val.replace("$", "").replace(",", ""))
            if num_val > 0:
                return "background-color: #d4edda; color: #155724"
            elif num_val < 0:
                return "background-color: #f8d7da; color: #721c24"
        return ""

    # Display styled dataframe
    styled_df = df.style.applymap(color_pnl, subset=["P&L"])

    st.dataframe(
        styled_df,
        width=None,
        hide_index=True,
        use_container_width=True
    )

    # Download section
    st.subheader("Export Data")
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Trades (CSV)",
        data=csv_data,
        file_name=f"wallet_analysis_{wallet[:8]}.csv",
        mime="text/csv"
    )


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Polymarket Wallet Analytics",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
        <style>
        /* Modern color scheme and typography */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            font-weight: 700;
            background: linear-gradient(120deg, #10b981 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        h2 {
            color: #1e293b;
            font-weight: 600;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        h3 {
            color: #334155;
            font-weight: 600;
        }
        /* Card-like containers */
        div[data-testid="stDataFrame"] {
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
        }
        /* Better metric styling */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #10b981;
        }
        /* Improved button styling */
        .stButton > button {
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        /* Info/Warning/Success boxes */
        div.stAlert {
            border-radius: 8px;
            border-left: 4px solid;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Polymarket Wallet Analytics")
    st.markdown(
        "💼 **Analyze individual wallet performance with real-time position values**"
    )

    # Render main section
    render_wallet_input()

    # Footer
    st.divider()
    st.caption("Data sourced from Polymarket APIs. Current prices fetched from CLOB API.")


if __name__ == "__main__":
    main()
