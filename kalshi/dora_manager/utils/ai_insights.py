"""
AI-Powered Insights for Market and Event Analysis

Uses OpenAI Responses API to generate strategic recommendations for:
- Event-level insights for market_update emails
- Market-level insights for market_screener emails
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_event_insights(
    event_ticker: str,
    event_markets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate AI insights for an event (for market_update emails).

    Analyzes all markets within an event to provide:
    - Overall insights on profitability drivers
    - Recommendation: "Expand", "Scale back", or "Fully Exit"
    - Rationale for recommendation

    Args:
        event_ticker: Event ticker (e.g., "KXBTC-24DEC31")
        event_markets: List of market proposals for this event, each containing:
            - market_id: Market ticker
            - action: Proposed action
            - metadata: Dict with pnl_24h, fill_count, position_qty, etc.
            - current_config: Current market configuration
            - proposed_changes: Proposed configuration changes

    Returns:
        Dictionary containing:
        - insights: Overall profitability drivers (2-3 sentences)
        - recommendation: One of ["Expand", "Scale back", "Fully Exit"]
        - rationale: 2-3 sentence rationale for recommendation
        - error: Error message if API call fails (None if successful)
    """
    try:
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "insights": "OpenAI API key not configured",
                "recommendation": None,
                "rationale": "Missing OPENAI_API_KEY in environment variables",
                "error": "Missing API key",
            }

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Calculate aggregate statistics for the event
        total_pnl = sum(m.get('metadata', {}).get('pnl_24h', 0) or 0 for m in event_markets)
        total_fills = sum(m.get('metadata', {}).get('fill_count', 0) or 0 for m in event_markets)
        markets_with_fills = sum(1 for m in event_markets if (m.get('metadata', {}).get('fill_count', 0) or 0) > 0)
        fill_rate = (markets_with_fills / len(event_markets) * 100) if event_markets else 0
        total_position = sum(abs(m.get('metadata', {}).get('position_qty', 0) or 0) for m in event_markets)

        # Count actions
        action_counts = {}
        for m in event_markets:
            action = m.get('action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1

        # Get top 3 profitable and top 3 unprofitable markets
        markets_sorted = sorted(
            event_markets,
            key=lambda m: m.get('metadata', {}).get('pnl_24h', 0) or 0,
            reverse=True
        )
        top_3_profitable = markets_sorted[:3]
        top_3_unprofitable = markets_sorted[-3:]

        # Build context for the prompt
        context = f"""Event: {event_ticker}
Number of markets: {len(event_markets)}
Total P&L (24h): ${total_pnl:+,.2f}
Total fills: {total_fills}
Markets with fills: {markets_with_fills}/{len(event_markets)} ({fill_rate:.0f}%)
Total position exposure: {total_position} contracts

Proposed actions:
"""
        for action, count in sorted(action_counts.items()):
            context += f"  - {action}: {count}\n"

        context += "\nTop 3 most profitable markets:\n"
        for m in top_3_profitable:
            pnl = m.get('metadata', {}).get('pnl_24h', 0) or 0
            fills = m.get('metadata', {}).get('fill_count', 0) or 0
            context += f"  - {m['market_id']}: ${pnl:+,.2f} P&L, {fills} fills\n"

        context += "\nTop 3 least profitable markets:\n"
        for m in top_3_unprofitable:
            pnl = m.get('metadata', {}).get('pnl_24h', 0) or 0
            fills = m.get('metadata', {}).get('fill_count', 0) or 0
            context += f"  - {m['market_id']}: ${pnl:+,.2f} P&L, {fills} fills\n"

        # Create the prompt
        prompt = f"""You are a market risk and profitability analyst for prediction market trading. Your job is to analyze the performance of all markets within an event and provide strategic insights.

Today's date: {datetime.now().strftime("%Y-%m-%d")}

{context}

Please provide:
1. Overall insights on profitability drivers for this event (2-3 sentences explaining what's driving performance - good or bad)
2. Recommendation on path forward - one of: "Expand", "Scale back", or "Fully Exit"
3. Rationale for your recommendation (2-3 sentences)

Guidelines for recommendations:
- "Expand": Event is highly profitable with good fill rates and low risk. Recommend increasing quote sizes or inventory limits.
- "Scale back": Event shows mixed performance or concerning trends. Recommend reducing exposure via smaller quote sizes or tighter inventory limits.
- "Fully Exit": Event is consistently unprofitable or high risk. Recommend disabling all markets in this event.

Your response should be only a JSON dictionary with this exact format:
{{"insights": "...", "recommendation": "Expand|Scale back|Fully Exit", "rationale": "..."}}"""

        # Call OpenAI Responses API
        response = client.responses.create(
            model="gpt-5.2",
            reasoning={"effort": "medium"},
            input=prompt,
        )

        # Parse the response
        response_text = getattr(response, "output_text", None)
        if response_text is None:
            return {
                "insights": "No output_text in response",
                "recommendation": None,
                "rationale": "Empty API response",
                "error": "Empty response",
            }
        response_text = response_text.strip()

        # Try to extract JSON if wrapped in markdown code blocks
        if response_text.startswith("```"):
            # Remove markdown code blocks
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        return {
            "insights": result.get("insights", "No insights provided"),
            "recommendation": result.get("recommendation", "N/A"),
            "rationale": result.get("rationale", "No rationale provided"),
            "error": None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response for event {event_ticker}: {e}")
        return {
            "insights": f"Failed to parse AI response: {str(e)}",
            "recommendation": None,
            "rationale": "",
            "error": "JSON parsing error",
        }
    except Exception as e:
        logger.error(f"Error generating event insights for {event_ticker}: {e}")
        return {
            "insights": f"Error calling OpenAI API: {str(e)}",
            "recommendation": None,
            "rationale": "",
            "error": str(e),
        }


def generate_market_insights(
    market_id: str,
    market_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate AI insights for a new market candidate (for market_screener emails).

    Analyzes a single market to provide:
    - Recommendation: "strong recommendation to enter", "enter with caution", or "do not enter"
    - Rationale for recommendation

    Args:
        market_id: Market ticker
        market_data: Market metadata containing:
            - title: Market title
            - volume_24h: 24hr trading volume
            - yes_bid/yes_ask: Current prices
            - info_risk_probability: Information risk assessment (if available)
            - buy_volume/sell_volume: Volume on each side
            - etc.

    Returns:
        Dictionary containing:
        - recommendation: One of ["strong recommendation to enter", "enter with caution", "do not enter"]
        - rationale: 2-3 sentence rationale for recommendation
        - error: Error message if API call fails (None if successful)
    """
    try:
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "recommendation": None,
                "rationale": "OpenAI API key not configured",
                "error": "Missing API key",
            }

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Extract key metrics
        title = market_data.get('title', '')
        volume_24h = market_data.get('volume_24h', 0)
        yes_bid = market_data.get('yes_bid', 0)
        yes_ask = market_data.get('yes_ask', 0)
        spread = yes_ask - yes_bid if yes_bid and yes_ask else 0
        info_risk = market_data.get('info_risk_probability')
        buy_volume = market_data.get('buy_volume', 0)
        sell_volume = market_data.get('sell_volume', 0)
        bid_depth = market_data.get('bid_depth_5c', 0)
        ask_depth = market_data.get('ask_depth_5c', 0)

        # Build context for the prompt
        context = f"""Market: {market_id}
Title: {title}

Key metrics:
- 24hr volume: {volume_24h:,} contracts
- Current bid/ask: {yes_bid}/{yes_ask} (spread: {spread} cents)
- Buy volume: {buy_volume:,} contracts
- Sell volume: {sell_volume:,} contracts
- Order book depth (±5¢): {bid_depth:,} contracts (bid), {ask_depth:,} contracts (ask)
"""

        if info_risk is not None:
            context += f"- Information risk: {info_risk:.0f}% (chance of market-moving news in next 7 days)\n"

        # Create the prompt
        prompt = f"""You are a market selection analyst for prediction market trading. Your job is to evaluate new market candidates and provide entry recommendations.

Today's date: {datetime.now().strftime("%Y-%m-%d")}

{context}

Please provide:
1. Recommendation on path forward - one of: "strong recommendation to enter", "enter with caution", or "do not enter"
2. Rationale for your recommendation (2-3 sentences)

Guidelines for recommendations:
- "strong recommendation to enter": High volume, tight spread, good liquidity on both sides, low information risk. Market shows strong trading characteristics.
- "enter with caution": Decent volume and liquidity but some concerning factors (wider spread, higher info risk, unbalanced volume). Worth trying but monitor closely.
- "do not enter": Low volume, poor liquidity, very wide spread, or very high information risk. Not worth the risk.

Your response should be only a JSON dictionary with this exact format:
{{"recommendation": "strong recommendation to enter|enter with caution|do not enter", "rationale": "..."}}"""

        # Call OpenAI Responses API
        response = client.responses.create(
            model="gpt-5.2",
            reasoning={"effort": "medium"},
            input=prompt,
        )

        # Parse the response
        response_text = getattr(response, "output_text", None)
        if response_text is None:
            return {
                "recommendation": None,
                "rationale": "Empty API response",
                "error": "Empty response",
            }
        response_text = response_text.strip()

        # Try to extract JSON if wrapped in markdown code blocks
        if response_text.startswith("```"):
            # Remove markdown code blocks
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        return {
            "recommendation": result.get("recommendation", "N/A"),
            "rationale": result.get("rationale", "No rationale provided"),
            "error": None,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response for market {market_id}: {e}")
        return {
            "recommendation": None,
            "rationale": f"Failed to parse AI response: {str(e)}",
            "error": "JSON parsing error",
        }
    except Exception as e:
        logger.error(f"Error generating market insights for {market_id}: {e}")
        return {
            "recommendation": None,
            "rationale": f"Error calling OpenAI API: {str(e)}",
            "error": str(e),
        }
