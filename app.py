"""
Polymarket Trade Pattern Analyzer - Streamlit App

UI layer for analyzing Polymarket trade patterns.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from polymarket_service import PolymarketService, AnalysisResult


# Initialize service
@st.cache_resource
def get_service():
    return PolymarketService()


def render_market_lookup():
    """Render the event/market lookup section."""
    st.header("1. Find Markets")

    col1, col2 = st.columns([3, 1])
    with col1:
        event_slug = st.text_input(
            "Event Slug",
            placeholder="e.g., 2024-presidential-election",
            help="Enter the event slug from a Polymarket URL"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        get_markets_btn = st.button("Get Markets", type="primary")

    if get_markets_btn and event_slug:
        with st.spinner("Fetching markets..."):
            try:
                service = get_service()
                markets = service.get_event_markets(event_slug)

                if markets:
                    st.session_state["markets"] = markets
                    st.success(f"Found {len(markets)} markets")
                else:
                    st.warning("No markets found for this event")
            except Exception as e:
                st.error(f"Error fetching markets: {e}")

    # Display markets if available
    if "markets" in st.session_state and st.session_state["markets"]:
        st.subheader("Available Markets")

        markets_df = pd.DataFrame([
            {
                "Question": m.question[:80] + "..." if len(m.question) > 80 else m.question,
                "Condition ID": m.condition_id,
                "Volume": f"${m.volume:,.0f}",
                "Active": "Yes" if m.active else "No"
            }
            for m in st.session_state["markets"]
        ])

        st.dataframe(
            markets_df,
            width="stretch",
            hide_index=True
        )

        # Allow copying condition ID
        st.info("Copy a Condition ID from the table above to use in the analysis section below.")


def render_market_analysis():
    """Render the market analysis section."""
    st.header("2. Analyze Market")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        market_id = st.text_input(
            "Market / Condition ID",
            placeholder="e.g., 0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917",
            help="Enter the condition ID or market ID"
        )
    with col2:
        trade_limit = st.number_input(
            "Trade Limit",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        analyze_btn = st.button("Analyze Market", type="primary")

    if analyze_btn and market_id:
        with st.spinner("Analyzing trades..."):
            try:
                service = get_service()
                result = service.analyze_market(market_id, trade_limit)

                if result:
                    st.session_state["analysis_result"] = result
                    st.session_state["market_id"] = market_id
                    st.success(f"Analyzed {result.summary['total_trades']} trades from {result.summary['unique_wallets']} wallets")
                else:
                    st.warning("No trades found for this market")
            except Exception as e:
                st.error(f"Error analyzing market: {e}")

    # Display analysis if available
    if "analysis_result" in st.session_state and st.session_state["analysis_result"]:
        render_analysis_results(st.session_state["analysis_result"])


def render_analysis_results(result: AnalysisResult):
    """Render the analysis results including summaries and charts."""
    st.divider()

    # Calculate wallet stats from trades
    wallet_stats = {}
    for trade in result.trades:
        wallet = trade.wallet
        if wallet not in wallet_stats:
            wallet_stats[wallet] = {
                "name": trade.wallet_name,
                "classification": result.wallet_classifications.get(wallet, "Unknown"),
                "total_volume": 0.0,
                "trade_count": 0
            }
        wallet_stats[wallet]["total_volume"] += trade.size
        wallet_stats[wallet]["trade_count"] += 1
        # Update name if we find one (some trades might have name, others not)
        if trade.wallet_name and not wallet_stats[wallet]["name"]:
            wallet_stats[wallet]["name"] = trade.wallet_name

    # Summary metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary by Outcome")
        outcome_data = []
        for outcome, stats in result.summary["by_outcome"].items():
            outcome_data.append({
                "Outcome": outcome,
                "Trades": stats["trade_count"],
                "Avg Price": f"${stats['avg_price']:.3f}",
                "Total Size": f"{stats['total_size']:.2f}"
            })
        st.dataframe(pd.DataFrame(outcome_data), width="stretch", hide_index=True)

    with col2:
        st.subheader("Summary by Classification")
        class_data = []
        for cls, stats in result.summary["by_classification"].items():
            row = {
                "Classification": cls,
                "Wallets": stats["wallet_count"],
            }
            # Add exposure for each outcome
            for outcome, exposure in stats.get("exposure_by_outcome", {}).items():
                row[f"{outcome} Exposure"] = f"{exposure:.2f}"
            class_data.append(row)
        st.dataframe(pd.DataFrame(class_data), width="stretch", hide_index=True)

    # Top Wallets Table
    st.subheader("Top Wallets by Volume")

    # Sort wallets by total volume descending
    sorted_wallets = sorted(wallet_stats.items(), key=lambda x: x[1]["total_volume"], reverse=True)

    top_wallets_data = []
    for wallet, stats in sorted_wallets[:20]:  # Show top 20
        # Display name: use wallet_name if available, otherwise truncated address
        if stats["name"]:
            display_name = stats["name"]
        else:
            display_name = f"{wallet[:8]}...{wallet[-6:]}" if len(wallet) > 16 else wallet

        top_wallets_data.append({
            "Trader": display_name,
            "Classification": stats["classification"],
            "Total Volume": f"{stats['total_volume']:,.2f}",
            "Trades": stats["trade_count"]
        })

    st.dataframe(pd.DataFrame(top_wallets_data), width="stretch", hide_index=True)

    # Wallet filter
    st.subheader("Trade Charts")

    # Build wallet options with name/classification info, sorted by volume
    wallet_options = ["All Wallets"]
    wallet_list = []
    for wallet, stats in sorted_wallets:
        # Use wallet name if available, otherwise truncated address
        if stats["name"]:
            display_name = stats["name"]
        else:
            display_name = f"{wallet[:8]}...{wallet[-6:]}" if len(wallet) > 16 else wallet

        label = f"{display_name} ({stats['classification']}) - {stats['total_volume']:,.0f} shares"
        wallet_options.append(label)
        wallet_list.append(wallet)

    selected_option = st.selectbox(
        "Highlight wallet trades:",
        options=wallet_options,
        index=0,
        help="Select a wallet to highlight its trades on the charts"
    )

    # Get the actual wallet address from selection
    selected_wallet = None
    if selected_option != "All Wallets":
        idx = wallet_options.index(selected_option) - 1  # -1 for "All Wallets"
        selected_wallet = wallet_list[idx]

    render_interactive_charts(result, selected_wallet)

    # Download section
    st.subheader("Export Data")
    service = get_service()
    csv_data = service.generate_trade_log_csv(
        result.trades,
        result.wallet_classifications
    )
    st.download_button(
        label="Download Trade Log (CSV)",
        data=csv_data,
        file_name="trade_log.csv",
        mime="text/csv"
    )


def render_interactive_charts(result: AnalysisResult, selected_wallet: str = None):
    """Render interactive Plotly charts for the analysis results."""
    outcomes_data = result.outcomes_data
    exposure_data = result.exposure_by_classification
    wallet_classifications = result.wallet_classifications
    trades = result.trades

    # Outcome colors
    outcome_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Classification colors
    classification_colors = {
        "Micro-Bots": "#e41a1c",
        "Directional Whales": "#377eb8",
        "Conviction Retail": "#4daf4a",
        "Light Market Makers": "#984ea3",
        "Noise/Casuals": "#ff7f00"
    }

    # Highlight color for selected wallet
    highlight_color = "#FFD700"  # Gold

    def build_last_price_series():
        """Build running last-price series per outcome for overlaying on exposure charts."""
        series = {}
        for outcome, data in outcomes_data.items():
            if not data["timestamps"]:
                continue

            sorted_indices = sorted(range(len(data["timestamps"])),
                                    key=lambda i: data["timestamps"][i])
            timestamps = [data["timestamps"][i] for i in sorted_indices]
            prices = [data["prices"][i] for i in sorted_indices]

            last_prices = []
            current_price = None
            for price in prices:
                current_price = price
                last_prices.append(current_price)

            series[outcome] = {
                "timestamps": timestamps,
                "last_prices": last_prices
            }
        return series

    last_price_series = build_last_price_series()

    # Helper function to get display name for wallet
    def get_display_name(wallet: str, wallet_name: str) -> str:
        """Return wallet name if available, otherwise truncated address."""
        if wallet_name:
            return wallet_name
        return f"{wallet[:8]}...{wallet[-6:]}" if len(wallet) > 16 else wallet

    # Chart 1: Trade Prices Over Time
    fig_prices = go.Figure()

    for idx, (outcome, data) in enumerate(outcomes_data.items()):
        if not data["timestamps"]:
            continue

        color = outcome_colors[idx % len(outcome_colors)]

        sorted_indices = sorted(range(len(data["timestamps"])),
                                key=lambda i: data["timestamps"][i])
        timestamps = [data["timestamps"][i] for i in sorted_indices]
        prices = [data["prices"][i] for i in sorted_indices]
        sizes = [data["sizes"][i] for i in sorted_indices]
        sides = [data["sides"][i] for i in sorted_indices]
        wallets = [data["wallets"][i] for i in sorted_indices]
        wallet_names = [data["wallet_names"][i] for i in sorted_indices] if "wallet_names" in data else [""] * len(wallets)
        # Get classification for each wallet
        classifications = [wallet_classifications.get(w, "Unknown") for w in wallets]
        # Get display names (wallet name if available, otherwise truncated address)
        display_names = [get_display_name(w, wn) for w, wn in zip(wallets, wallet_names)]

        # Add line trace for this outcome (connecting all prices over time)
        fig_prices.add_trace(go.Scatter(
            x=timestamps, y=prices,
            mode='lines',
            line=dict(color=color, width=1, dash='solid'),
            name=f"{outcome} (price line)",
            showlegend=False,
            hoverinfo='skip'
        ))

        # BUY trades (non-highlighted)
        buy_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                    if side == "BUY" and (selected_wallet is None or w != selected_wallet)]
        if buy_data:
            buy_ts, buy_prices, buy_sizes, buy_wallets, buy_wnames, buy_classes, buy_displays = zip(*buy_data)
            fig_prices.add_trace(go.Scatter(
                x=buy_ts, y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color=color, opacity=0.4 if selected_wallet else 0.7),
                name=f"{outcome} (BUY)",
                hovertemplate="<b>%{customdata[2]}</b><br>" +
                              f"<b>{outcome} BUY</b><br>" +
                              "Price: %{y:.3f}<br>" +
                              "Size: %{customdata[0]:.2f}<br>" +
                              "Trader: %{customdata[1]}<extra></extra>",
                customdata=list(zip(buy_sizes, buy_displays, buy_classes))
            ))

        # SELL trades (non-highlighted)
        sell_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                     if side == "SELL" and (selected_wallet is None or w != selected_wallet)]
        if sell_data:
            sell_ts, sell_prices, sell_sizes, sell_wallets, sell_wnames, sell_classes, sell_displays = zip(*sell_data)
            fig_prices.add_trace(go.Scatter(
                x=sell_ts, y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=color, opacity=0.4 if selected_wallet else 0.7),
                name=f"{outcome} (SELL)",
                hovertemplate="<b>%{customdata[2]}</b><br>" +
                              f"<b>{outcome} SELL</b><br>" +
                              "Price: %{y:.3f}<br>" +
                              "Size: %{customdata[0]:.2f}<br>" +
                              "Trader: %{customdata[1]}<extra></extra>",
                customdata=list(zip(sell_sizes, sell_displays, sell_classes))
            ))

    # Add highlighted wallet trades on top (for price chart)
    if selected_wallet:
        for idx, (outcome, data) in enumerate(outcomes_data.items()):
            if not data["timestamps"]:
                continue

            sorted_indices = sorted(range(len(data["timestamps"])),
                                    key=lambda i: data["timestamps"][i])
            timestamps = [data["timestamps"][i] for i in sorted_indices]
            prices = [data["prices"][i] for i in sorted_indices]
            sizes = [data["sizes"][i] for i in sorted_indices]
            sides = [data["sides"][i] for i in sorted_indices]
            wallets = [data["wallets"][i] for i in sorted_indices]
            wallet_names = [data["wallet_names"][i] for i in sorted_indices] if "wallet_names" in data else [""] * len(wallets)
            classifications = [wallet_classifications.get(w, "Unknown") for w in wallets]
            display_names = [get_display_name(w, wn) for w, wn in zip(wallets, wallet_names)]

            # Highlighted BUY trades
            buy_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                        if side == "BUY" and w == selected_wallet]
            if buy_data:
                buy_ts, buy_prices, buy_sizes, buy_wallets, buy_wnames, buy_classes, buy_displays = zip(*buy_data)
                fig_prices.add_trace(go.Scatter(
                    x=buy_ts, y=buy_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=14, color=highlight_color,
                               line=dict(color='black', width=2)),
                    name=f"★ {outcome} (BUY)",
                    hovertemplate="<b>★ SELECTED WALLET</b><br>" +
                                  "<b>%{customdata[2]}</b><br>" +
                                  f"<b>{outcome} BUY</b><br>" +
                                  "Price: %{y:.3f}<br>" +
                                  "Size: %{customdata[0]:.2f}<br>" +
                                  "Trader: %{customdata[1]}<extra></extra>",
                    customdata=list(zip(buy_sizes, buy_displays, buy_classes))
                ))

            # Highlighted SELL trades
            sell_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                         if side == "SELL" and w == selected_wallet]
            if sell_data:
                sell_ts, sell_prices, sell_sizes, sell_wallets, sell_wnames, sell_classes, sell_displays = zip(*sell_data)
                fig_prices.add_trace(go.Scatter(
                    x=sell_ts, y=sell_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=14, color=highlight_color,
                               line=dict(color='black', width=2)),
                    name=f"★ {outcome} (SELL)",
                    hovertemplate="<b>★ SELECTED WALLET</b><br>" +
                                  "<b>%{customdata[2]}</b><br>" +
                                  f"<b>{outcome} SELL</b><br>" +
                                  "Price: %{y:.3f}<br>" +
                                  "Size: %{customdata[0]:.2f}<br>" +
                                  "Trader: %{customdata[1]}<extra></extra>",
                    customdata=list(zip(sell_sizes, sell_displays, sell_classes))
                ))

    fig_prices.update_layout(
        title="Trade Prices Over Time (▲ = BUY, ▼ = SELL)" + (f" - Highlighting: {selected_wallet[:8]}..." if selected_wallet else ""),
        xaxis_title="Time",
        yaxis_title="Price ($)",
        yaxis=dict(range=[0, 1.05]),
        height=400,
        hovermode='closest'
    )
    st.plotly_chart(fig_prices, key="prices_chart")

    # Chart 2: Trade Sizes Over Time
    fig_sizes = go.Figure()

    for idx, (outcome, data) in enumerate(outcomes_data.items()):
        if not data["timestamps"]:
            continue

        color = outcome_colors[idx % len(outcome_colors)]

        sorted_indices = sorted(range(len(data["timestamps"])),
                                key=lambda i: data["timestamps"][i])
        timestamps = [data["timestamps"][i] for i in sorted_indices]
        prices = [data["prices"][i] for i in sorted_indices]
        sizes = [data["sizes"][i] for i in sorted_indices]
        sides = [data["sides"][i] for i in sorted_indices]
        wallets = [data["wallets"][i] for i in sorted_indices]
        wallet_names = [data["wallet_names"][i] for i in sorted_indices] if "wallet_names" in data else [""] * len(wallets)
        classifications = [wallet_classifications.get(w, "Unknown") for w in wallets]
        display_names = [get_display_name(w, wn) for w, wn in zip(wallets, wallet_names)]

        # BUY trades (non-highlighted)
        buy_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                    if side == "BUY" and (selected_wallet is None or w != selected_wallet)]
        if buy_data:
            buy_ts, buy_prices, buy_sizes, buy_wallets, buy_wnames, buy_classes, buy_displays = zip(*buy_data)
            fig_sizes.add_trace(go.Scatter(
                x=buy_ts, y=buy_sizes,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color=color, opacity=0.4 if selected_wallet else 0.7),
                name=f"{outcome} (BUY)",
                hovertemplate="<b>%{customdata[2]}</b><br>" +
                              f"<b>{outcome} BUY</b><br>" +
                              "Size: %{y:.2f}<br>" +
                              "Price: %{customdata[0]:.3f}<br>" +
                              "Trader: %{customdata[1]}<extra></extra>",
                customdata=list(zip(buy_prices, buy_displays, buy_classes))
            ))

        # SELL trades (non-highlighted)
        sell_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                     if side == "SELL" and (selected_wallet is None or w != selected_wallet)]
        if sell_data:
            sell_ts, sell_prices, sell_sizes, sell_wallets, sell_wnames, sell_classes, sell_displays = zip(*sell_data)
            fig_sizes.add_trace(go.Scatter(
                x=sell_ts, y=sell_sizes,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=color, opacity=0.4 if selected_wallet else 0.7),
                name=f"{outcome} (SELL)",
                hovertemplate="<b>%{customdata[2]}</b><br>" +
                              f"<b>{outcome} SELL</b><br>" +
                              "Size: %{y:.2f}<br>" +
                              "Price: %{customdata[0]:.3f}<br>" +
                              "Trader: %{customdata[1]}<extra></extra>",
                customdata=list(zip(sell_prices, sell_displays, sell_classes))
            ))

    # Add highlighted wallet trades on top (for size chart)
    if selected_wallet:
        for idx, (outcome, data) in enumerate(outcomes_data.items()):
            if not data["timestamps"]:
                continue

            sorted_indices = sorted(range(len(data["timestamps"])),
                                    key=lambda i: data["timestamps"][i])
            timestamps = [data["timestamps"][i] for i in sorted_indices]
            prices = [data["prices"][i] for i in sorted_indices]
            sizes = [data["sizes"][i] for i in sorted_indices]
            sides = [data["sides"][i] for i in sorted_indices]
            wallets = [data["wallets"][i] for i in sorted_indices]
            wallet_names = [data["wallet_names"][i] for i in sorted_indices] if "wallet_names" in data else [""] * len(wallets)
            classifications = [wallet_classifications.get(w, "Unknown") for w in wallets]
            display_names = [get_display_name(w, wn) for w, wn in zip(wallets, wallet_names)]

            # Highlighted BUY trades
            buy_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                        if side == "BUY" and w == selected_wallet]
            if buy_data:
                buy_ts, buy_prices, buy_sizes, buy_wallets, buy_wnames, buy_classes, buy_displays = zip(*buy_data)
                fig_sizes.add_trace(go.Scatter(
                    x=buy_ts, y=buy_sizes,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=14, color=highlight_color,
                               line=dict(color='black', width=2)),
                    name=f"★ {outcome} (BUY)",
                    hovertemplate="<b>★ SELECTED WALLET</b><br>" +
                                  "<b>%{customdata[2]}</b><br>" +
                                  f"<b>{outcome} BUY</b><br>" +
                                  "Size: %{y:.2f}<br>" +
                                  "Price: %{customdata[0]:.3f}<br>" +
                                  "Trader: %{customdata[1]}<extra></extra>",
                    customdata=list(zip(buy_prices, buy_displays, buy_classes))
                ))

            # Highlighted SELL trades
            sell_data = [(ts, p, s, w, wn, c, dn) for ts, p, s, side, w, wn, c, dn in zip(timestamps, prices, sizes, sides, wallets, wallet_names, classifications, display_names)
                         if side == "SELL" and w == selected_wallet]
            if sell_data:
                sell_ts, sell_prices, sell_sizes, sell_wallets, sell_wnames, sell_classes, sell_displays = zip(*sell_data)
                fig_sizes.add_trace(go.Scatter(
                    x=sell_ts, y=sell_sizes,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=14, color=highlight_color,
                               line=dict(color='black', width=2)),
                    name=f"★ {outcome} (SELL)",
                    hovertemplate="<b>★ SELECTED WALLET</b><br>" +
                                  "<b>%{customdata[2]}</b><br>" +
                                  f"<b>{outcome} SELL</b><br>" +
                                  "Size: %{y:.2f}<br>" +
                                  "Price: %{customdata[0]:.3f}<br>" +
                                  "Trader: %{customdata[1]}<extra></extra>",
                    customdata=list(zip(sell_prices, sell_displays, sell_classes))
                ))

    fig_sizes.update_layout(
        title="Trade Sizes Over Time" + (f" - Highlighting: {selected_wallet[:8]}..." if selected_wallet else ""),
        xaxis_title="Time",
        yaxis_title="Trade Size",
        height=400,
        hovermode='closest'
    )
    st.plotly_chart(fig_sizes, key="sizes_chart")

    exposure_tab, wallet_tab = st.tabs(["Cumulative Category Exposure", "Wallet Intelligence"])

    with exposure_tab:
        # Get all outcomes for exposure charts
        all_outcomes = set()
        for exp in exposure_data.values():
            all_outcomes.update(exp.exposure_by_outcome.keys())
        outcomes_list = sorted(all_outcomes)

        # Chart 3: First outcome exposure
        if len(outcomes_list) >= 1:
            outcome1 = outcomes_list[0]
            fig_exp1 = make_subplots(specs=[[{"secondary_y": True}]])

            for cls, exp in exposure_data.items():
                if not exp.timestamps:
                    continue
                color = classification_colors.get(cls, "#999999")
                exposure_values = exp.exposure_by_outcome.get(outcome1, [])
                if exposure_values:
                    fig_exp1.add_trace(go.Scatter(
                        x=exp.timestamps, y=exposure_values,
                        mode='lines',
                        line=dict(color=color, width=2, shape='hv'),  # step function: horizontal then vertical
                        name=cls,
                        hovertemplate=f"<b>{cls}</b><br>Exposure: %{{y:.2f}}<extra></extra>"
                    ), secondary_y=False)

            # Add zero line
            fig_exp1.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

            # Overlay last price on secondary y-axis
            price_series1 = last_price_series.get(outcome1)
            if price_series1:
                fig_exp1.add_trace(go.Scatter(
                    x=price_series1["timestamps"],
                    y=price_series1["last_prices"],
                    mode='lines',
                    line=dict(color="#2c3e50", width=1.5, dash='dot'),
                    name=f"{outcome1} Last Price",
                    hovertemplate=f"<b>{outcome1} Last Price</b><br>Price: %{{y:.3f}}<extra></extra>"
                ), secondary_y=True)

            fig_exp1.update_layout(
                title=f"Net '{outcome1}' Exposure by Wallet Classification",
                xaxis_title="Time",
                height=400,
                hovermode='x unified'
            )
            fig_exp1.update_yaxes(title_text=f"Net {outcome1} Exposure", secondary_y=False)
            fig_exp1.update_yaxes(title_text=f"{outcome1} Last Price", secondary_y=True, range=[0, 1.05])
            st.plotly_chart(fig_exp1, key="exp1_chart")

        # Chart 4: Second outcome exposure
        if len(outcomes_list) >= 2:
            outcome2 = outcomes_list[1]
            fig_exp2 = make_subplots(specs=[[{"secondary_y": True}]])

            for cls, exp in exposure_data.items():
                if not exp.timestamps:
                    continue
                color = classification_colors.get(cls, "#999999")
                exposure_values = exp.exposure_by_outcome.get(outcome2, [])
                if exposure_values:
                    fig_exp2.add_trace(go.Scatter(
                        x=exp.timestamps, y=exposure_values,
                        mode='lines',
                        line=dict(color=color, width=2, shape='hv'),  # step function: horizontal then vertical
                        name=cls,
                        hovertemplate=f"<b>{cls}</b><br>Exposure: %{{y:.2f}}<extra></extra>"
                    ), secondary_y=False)

            # Add zero line
            fig_exp2.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

            # Overlay last price on secondary y-axis
            price_series2 = last_price_series.get(outcome2)
            if price_series2:
                fig_exp2.add_trace(go.Scatter(
                    x=price_series2["timestamps"],
                    y=price_series2["last_prices"],
                    mode='lines',
                    line=dict(color="#2c3e50", width=1.5, dash='dot'),
                    name=f"{outcome2} Last Price",
                    hovertemplate=f"<b>{outcome2} Last Price</b><br>Price: %{{y:.3f}}<extra></extra>"
                ), secondary_y=True)

            fig_exp2.update_layout(
                title=f"Net '{outcome2}' Exposure by Wallet Classification",
                xaxis_title="Time",
                height=400,
                hovermode='x unified'
            )
            fig_exp2.update_yaxes(title_text=f"Net {outcome2} Exposure", secondary_y=False)
            fig_exp2.update_yaxes(title_text=f"{outcome2} Last Price", secondary_y=True, range=[0, 1.05])
            st.plotly_chart(fig_exp2, key="exp2_chart")

    with wallet_tab:
        # Table: selected wallet trades with running outcome exposure
        st.subheader("Selected Wallet Trade Log")
        if not selected_wallet:
            st.info("Select a wallet above to view a detailed trade log.")
        else:
            wallet_trades = [t for t in trades if t.wallet == selected_wallet]
            if not wallet_trades:
                st.info("No trades found for the selected wallet.")
            else:
                wallet_trades = sorted(wallet_trades, key=lambda t: t.timestamp)
                running_totals = {}
                dollars_bought = {}
                dollars_sold = {}
                rows_by_outcome = {}

                for t in wallet_trades:
                    outcome = t.outcome
                    running_totals.setdefault(outcome, 0.0)
                    dollars_bought.setdefault(outcome, 0.0)
                    dollars_sold.setdefault(outcome, 0.0)
                    rows_by_outcome.setdefault(outcome, [])

                    delta = t.size if t.side == "BUY" else -t.size
                    running_totals[outcome] += delta
                    amount_dollars = t.size * t.price
                    if t.side == "BUY":
                        dollars_bought[outcome] += amount_dollars
                    else:
                        dollars_sold[outcome] += amount_dollars

                    rows_by_outcome[outcome].append({
                        "Datetime": t.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "Outcome": outcome,
                        "Side": t.side,
                        "Price": t.price,
                        "Size": t.size,
                        "Amount (number of shares)": t.size,
                        "Amount (dollar value)": amount_dollars,
                        "Cumulative Shares (outcome)": running_totals[outcome]
                    })

                # Compute P/L by outcome and total
                pnl_by_outcome = {}
                total_pnl = 0.0
                for outcome, shares in running_totals.items():
                    last_price_info = last_price_series.get(outcome, {})
                    current_price = last_price_info.get("last_prices", [0])[-1] if last_price_info else 0
                    current_value = shares * current_price
                    pnl = current_value + dollars_sold.get(outcome, 0.0) - dollars_bought.get(outcome, 0.0)
                    pnl_by_outcome[outcome] = {
                        "current_price": current_price,
                        "shares": shares,
                        "current_value": current_value,
                        "dollars_bought": dollars_bought.get(outcome, 0.0),
                        "dollars_sold": dollars_sold.get(outcome, 0.0),
                        "pnl": pnl
                    }
                    total_pnl += pnl

                # Wallet intelligence summary
                wallet_display_name = wallet_trades[0].wallet_name or f"{selected_wallet[:8]}...{selected_wallet[-6:]}"
                st.markdown(f"**Wallet:** `{selected_wallet}` ({wallet_display_name})  |  "
                            f"**Classification:** {wallet_classifications.get(selected_wallet, 'Unknown')}  |  "
                            f"**Total P/L:** {total_pnl:,.2f}")
                st.caption(f"Trades: {len(wallet_trades)} | First trade: {wallet_trades[0].datetime} | Last trade: {wallet_trades[-1].datetime}")

                for outcome, rows in rows_by_outcome.items():
                    st.markdown(f"**{outcome} Trades**  — P/L: {pnl_by_outcome[outcome]['pnl']:,.2f} | "
                                f"Shares: {pnl_by_outcome[outcome]['shares']:,.2f} @ "
                                f"Price {pnl_by_outcome[outcome]['current_price']:.3f}")
                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        width="stretch",
                        hide_index=True
                    )

def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Polymarket Trade Analyzer",
        page_icon="📊",
        layout="wide"
    )

    st.title("Polymarket Trade Pattern Analyzer")
    st.markdown(
        "Analyze trading patterns and wallet behavior in Polymarket prediction markets."
    )

    # Render sections
    render_market_lookup()
    st.divider()
    render_market_analysis()

    # Footer
    st.divider()
    st.caption("Data sourced from Polymarket APIs. Wallet classifications are heuristic-based.")


if __name__ == "__main__":
    main()
