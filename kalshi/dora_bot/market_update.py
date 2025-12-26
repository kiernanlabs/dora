#!/usr/bin/env python3
"""
Market Update Script for Dora Bot.

This script analyzes market performance and generates recommended config updates:
1. Markets to SCALE DOWN (poor performance):
   - (A) P&L < -$0.50 over lookback period, OR
   - (B) No fills in lookback period
   Scale down: cut quote_size and max_inventory by 50%, double min_spread
   If quote_size is already <=5: exit (disable or set min_spread=0.50 if has position)

2. Markets to EXPAND:
   - Positive P&L in lookback period AND median fill size = current quote size
   For expand markets: double quote_size, max_inventory_yes, max_inventory_no (capped at 25)

3. Sibling market ACTIVATION (when expanding):
   - For markets being expanded, fetch all sibling markets from the same event
   - Activate any sibling markets not already in config with default settings
   - Uses OpenAI to generate fair value estimates for new sibling markets

Usage:
    python -m dora_bot.market_update --env prod
    python -m dora_bot.market_update --env prod --pnl-lookback 48 --volume-lookback 72
    python -m dora_bot.market_update --env demo --dry-run
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from openai import OpenAI

from dora_bot.dynamo import DynamoDBClient
from dora_bot.models import MarketConfig, Position

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Public API endpoint (no auth required for market data)
KALSHI_API_BASE = "https://api.elections.kalshi.com"

# Default settings for newly activated sibling markets
DEFAULT_QUOTE_SIZE = 5
DEFAULT_MAX_INVENTORY_YES = 5
DEFAULT_MAX_INVENTORY_NO = 5
DEFAULT_MIN_SPREAD = 0.04

# Expansion caps
MAX_QUOTE_SIZE = 25
MAX_INVENTORY = 25

# Scale down threshold - if quote_size is at or below this, exit instead of scaling down
MIN_QUOTE_SIZE_FOR_SCALE_DOWN = 5


@dataclass
class MarketAnalysis:
    """Analysis results for a single market."""
    market_id: str
    event_ticker: Optional[str]  # Event this market belongs to
    pnl_24h: float
    fill_count_24h: int
    fill_count_48h: int
    last_fill_time: Optional[datetime]
    median_fill_size: Optional[float]
    current_quote_size: int
    current_max_inventory_yes: int
    current_max_inventory_no: int
    current_min_spread: float
    current_enabled: bool
    has_position: bool
    position_qty: int


@dataclass
class RecommendedAction:
    """A recommended action for a market."""
    market_id: str
    event_ticker: Optional[str]  # Event this market belongs to
    action: str  # 'exit', 'scale_down', 'expand', or 'activate_sibling'
    reason: str
    new_enabled: Optional[bool]
    new_min_spread: Optional[float]
    new_quote_size: Optional[int]
    new_max_inventory_yes: Optional[int]
    new_max_inventory_no: Optional[int]
    # Context for review
    pnl_24h: float
    fill_count_24h: int
    fill_count_48h: int
    has_position: bool
    position_qty: int
    # Fair value assessment (for activate_sibling actions)
    fair_value: Optional[float] = None
    fair_value_rationale: Optional[str] = None


def get_fills_for_date_range(
    dynamodb_resource,
    table_name: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict]:
    """Query fills from the trade log for a date range.

    Args:
        dynamodb_resource: boto3 DynamoDB resource
        table_name: Name of the trade log table
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

    Returns:
        List of fill records
    """
    table = dynamodb_resource.Table(table_name)
    fills = []

    # Generate all dates in the range
    current = start_date.date()
    end = end_date.date()

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        try:
            # Query all items for this date
            response = table.query(
                KeyConditionExpression=Key('date').eq(date_str)
            )

            for item in response.get('Items', []):
                fills.append(_serialize_decimal(item))

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    fills.append(_serialize_decimal(item))

        except ClientError as e:
            print(f"Warning: Error querying fills for {date_str}: {e}")

        current += timedelta(days=1)

    return fills


def _serialize_decimal(obj):
    """Convert Decimal to float for processing."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialize_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_decimal(item) for item in obj]
    return obj


def assess_fair_value(
    market_title: str,
    market_subtitle: Optional[str] = None,
    rules: Optional[str] = None,
) -> Dict[str, Any]:
    """Assess the fair value (likelihood of resolving YES) for a market.

    Uses OpenAI API to evaluate the probability without referencing prediction markets.

    Args:
        market_title: The title of the market
        market_subtitle: Optional subtitle providing additional context
        rules: Optional resolution rules for the market

    Returns:
        Dictionary containing:
        - probability: Likelihood percentage (0-100)
        - rationale: 2-3 sentence explanation
        - error: Error message if API call fails
    """
    try:
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "probability": None,
                "rationale": "OpenAI API key not configured",
                "error": "Missing OPENAI_API_KEY in environment variables",
            }

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Build context for the prompt
        context = f"The question is: {market_title}"
        if market_subtitle:
            context += f"\n{market_subtitle}"
        if rules:
            context += f"\n\nResolution criteria: {rules[:500]}"  # Limit rules length

        # Create the prompt
        prompt = f"""You are an expert analyst tasked with estimating probabilities for real-world events. Your job is to estimate the likelihood that the following event will resolve to YES.

Do NOT reference prediction markets, betting odds, or market prices. Base your estimate solely on your knowledge of the subject matter, historical precedents, and logical reasoning.

Today is {datetime.now().strftime("%Y-%m-%d")}

{context}

Please return your assessment in the form of a likelihood percentage (number from 0-100%) and 2-3 sentence rationale explaining your reasoning.

Your response should be only a JSON dictionary e.g. {{"probability": "XX%", "rationale": "XXXX"}}"""

        response = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "medium"},
            tools=[{"type": "web_search"}],
            input=prompt,
        )

        # Parse the response (Responses API uses output_text)
        response_text = getattr(response, "output_text", None)
        if response_text is None:
            return {
                "probability": None,
                "rationale": "No output_text in response",
                "error": "Empty API response",
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
            "probability": result.get("probability", "N/A"),
            "rationale": result.get("rationale", "No rationale provided"),
            "error": None,
        }

    except json.JSONDecodeError as e:
        return {
            "probability": None,
            "rationale": f"Failed to parse API response: {str(e)}",
            "error": "JSON parsing error",
        }
    except Exception as e:
        return {
            "probability": None,
            "rationale": f"Error calling OpenAI API: {str(e)}",
            "error": str(e),
        }


def fetch_event_markets(event_ticker: str) -> List[Dict[str, Any]]:
    """Fetch all markets for an event from the Kalshi API.

    Args:
        event_ticker: Event ticker to fetch markets for

    Returns:
        List of market dictionaries from the API
    """
    try:
        url = f"{KALSHI_API_BASE}/trade-api/v2/events/{event_ticker}"
        print(f"    Fetching: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # The API returns 'markets' at the top level, alongside 'event'
        markets = data.get('markets', [])
        print(f"    Response: found {len(markets)} markets in event {event_ticker}")
        return markets
    except requests.exceptions.HTTPError as e:
        print(f"    HTTP Error fetching {event_ticker}: {e.response.status_code} - {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"    Error fetching {event_ticker}: {type(e).__name__}: {e}")
        return []


def fetch_event_tickers(market_ids: List[str]) -> Dict[str, Optional[str]]:
    """Fetch event_ticker for each market from the public Kalshi API.

    Args:
        market_ids: List of market IDs to fetch

    Returns:
        Dictionary mapping market_id to event_ticker (or None if not found)
    """
    event_tickers = {}

    for market_id in market_ids:
        try:
            url = f"{KALSHI_API_BASE}/trade-api/v2/markets/{market_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # The API returns market info with 'market' key containing the data
            market_data = data.get('market', data)
            event_ticker = market_data.get('event_ticker')
            event_tickers[market_id] = event_ticker
        except Exception as e:
            print(f"Warning: Could not fetch event_ticker for {market_id}: {e}")
            event_tickers[market_id] = None

    return event_tickers


def analyze_markets(
    dynamo: DynamoDBClient,
    environment: str,
    pnl_lookback_hours: int = 24,
    volume_lookback_hours: int = 48
) -> Dict[str, MarketAnalysis]:
    """Analyze all markets based on fills and positions.

    Args:
        dynamo: DynamoDB client
        environment: 'demo' or 'prod'
        pnl_lookback_hours: Hours to look back for P&L calculation (default 24)
        volume_lookback_hours: Hours to look back for fill count (default 48)

    Returns:
        Dictionary mapping market_id to MarketAnalysis
    """
    now = datetime.now(timezone.utc)
    cutoff_pnl = now - timedelta(hours=pnl_lookback_hours)
    cutoff_volume = now - timedelta(hours=volume_lookback_hours)

    # Get all market configs (including disabled ones)
    configs = dynamo.get_all_market_configs(enabled_only=False)

    # Fetch event_tickers for all markets from public API
    print(f"Fetching event info for {len(configs)} markets...")
    event_tickers = fetch_event_tickers(list(configs.keys()))

    # Get current positions
    positions = dynamo.get_positions()

    # Get fills for the lookback period (use the longer of the two windows)
    max_lookback = max(pnl_lookback_hours, volume_lookback_hours)
    cutoff_max = now - timedelta(hours=max_lookback)

    suffix = '_demo' if environment == 'demo' else '_prod'
    table_name = f"dora_trade_log{suffix}"

    fills = get_fills_for_date_range(
        dynamo.dynamodb,
        table_name,
        cutoff_max,
        now
    )

    # Group fills by market
    fills_by_market: Dict[str, List[Dict]] = defaultdict(list)
    for fill in fills:
        market_id = fill.get('market_id')
        if market_id:
            fills_by_market[market_id].append(fill)

    # Analyze each market
    analyses = {}
    for market_id, config in configs.items():
        market_fills = fills_by_market.get(market_id, [])

        # Parse fill timestamps and filter by time window
        fills_pnl = []  # For P&L lookback
        fills_volume = []  # For volume/fill count lookback
        last_fill_time = None

        for fill in market_fills:
            # Get timestamp from the fill
            ts_str = fill.get('fill_timestamp') or fill.get('timestamp')
            if not ts_str:
                # Try to extract from the sort key
                sort_key = fill.get('timestamp#order_id', '')
                if '#' in sort_key:
                    ts_str = sort_key.split('#')[0]

            if ts_str:
                try:
                    # Handle both ISO format with and without 'Z'
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1] + '+00:00'
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    # Track last fill time
                    if last_fill_time is None or ts > last_fill_time:
                        last_fill_time = ts

                    # Categorize by time window
                    if ts >= cutoff_volume:
                        fills_volume.append(fill)
                    if ts >= cutoff_pnl:
                        fills_pnl.append(fill)
                except (ValueError, TypeError):
                    # If we can't parse the timestamp, include in both windows
                    fills_volume.append(fill)
                    fills_pnl.append(fill)

        # Calculate P&L for the P&L lookback period
        pnl_lookback = 0.0
        fill_sizes = []
        for fill in fills_pnl:
            # P&L is calculated as: (sell_price - buy_price) * size - fees
            # For individual fills, we use pnl_realized if available
            pnl = fill.get('pnl_realized', 0.0) or 0.0
            fees = fill.get('fees', 0.0) or 0.0
            pnl_lookback += pnl - fees

            # Track fill sizes for median calculation
            size = fill.get('size') or fill.get('fill_size', 0)
            if size:
                fill_sizes.append(size)

        # Calculate median fill size
        median_fill_size = statistics.median(fill_sizes) if fill_sizes else None

        # Get position info
        position = positions.get(market_id)
        has_position = position is not None and position.net_yes_qty != 0
        position_qty = position.net_yes_qty if position else 0

        analyses[market_id] = MarketAnalysis(
            market_id=market_id,
            event_ticker=event_tickers.get(market_id),
            pnl_24h=pnl_lookback,  # Named pnl_24h for backward compat but uses configured lookback
            fill_count_24h=len(fills_pnl),  # Fills in P&L lookback period
            fill_count_48h=len(fills_volume),  # Fills in volume lookback period
            last_fill_time=last_fill_time,
            median_fill_size=median_fill_size,
            current_quote_size=config.quote_size,
            current_max_inventory_yes=config.max_inventory_yes,
            current_max_inventory_no=config.max_inventory_no,
            current_min_spread=config.min_spread,
            current_enabled=config.enabled,
            has_position=has_position,
            position_qty=position_qty,
        )

    return analyses


def generate_recommendations(
    analyses: Dict[str, MarketAnalysis],
    existing_configs: Dict[str, MarketConfig]
) -> List[RecommendedAction]:
    """Generate recommended actions based on market analysis.

    Args:
        analyses: Dictionary of market analyses
        existing_configs: Dictionary of existing market configs in DynamoDB

    Returns:
        List of recommended actions
    """
    recommendations = []

    # Build maps for event-based analysis
    all_markets_by_event: Dict[str, List[str]] = defaultdict(list)
    poor_performance_candidates: Dict[str, str] = {}  # market_id -> reason

    # Track events that have expand candidates (for sibling activation)
    expand_events: set = set()

    # First pass: identify poor performance candidates and build event map
    for market_id, analysis in analyses.items():
        # Skip disabled markets without positions (already exited)
        if not analysis.current_enabled and not analysis.has_position:
            continue

        # Track all enabled markets by event
        if analysis.current_enabled and analysis.event_ticker:
            all_markets_by_event[analysis.event_ticker].append(market_id)

        # Check POOR PERFORMANCE conditions (scale down or exit)
        poor_reason = None

        # (A) P&L < -$0.50 over lookback period
        if analysis.pnl_24h < -0.50:
            poor_reason = f"P&L < -$0.50 (${analysis.pnl_24h:.2f})"

        # (B) No fills in lookback period (only for enabled markets)
        elif analysis.current_enabled and analysis.fill_count_48h == 0:
            poor_reason = "No fills in lookback period"

        if poor_reason:
            poor_performance_candidates[market_id] = poor_reason

        # Check EXPAND conditions (for tracking events with expanding markets)
        if (analysis.current_enabled and
            analysis.pnl_24h > 0 and
            analysis.median_fill_size is not None and
            analysis.median_fill_size == analysis.current_quote_size and
            analysis.event_ticker):
            expand_events.add(analysis.event_ticker)

    # Second pass: generate recommendations
    for market_id, analysis in analyses.items():
        # Skip disabled markets without positions (already exited)
        if not analysis.current_enabled and not analysis.has_position:
            continue

        poor_reason = poor_performance_candidates.get(market_id)

        if poor_reason:
            event_ticker = analysis.event_ticker

            # Check if we can scale down or need to exit
            # If quote_size > MIN_QUOTE_SIZE_FOR_SCALE_DOWN, scale down instead of exit
            if analysis.current_quote_size > MIN_QUOTE_SIZE_FOR_SCALE_DOWN:
                # Scale down: cut quote_size and max_inventory by 50%, double min_spread
                new_quote_size = max(1, analysis.current_quote_size // 2)
                new_max_inv_yes = max(1, analysis.current_max_inventory_yes // 2)
                new_max_inv_no = max(1, analysis.current_max_inventory_no // 2)
                new_min_spread = min(0.50, analysis.current_min_spread * 2)

                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='scale_down',
                    reason=f"{poor_reason} - scaling down (quote {analysis.current_quote_size}->{new_quote_size})",
                    new_enabled=None,
                    new_min_spread=new_min_spread,
                    new_quote_size=new_quote_size,
                    new_max_inventory_yes=new_max_inv_yes,
                    new_max_inventory_no=new_max_inv_no,
                    pnl_24h=analysis.pnl_24h,
                    fill_count_24h=analysis.fill_count_24h,
                    fill_count_48h=analysis.fill_count_48h,
                    has_position=analysis.has_position,
                    position_qty=analysis.position_qty,
                ))
            elif analysis.has_position:
                # Quote size at minimum and has position: set min_spread to 0.50 to exit
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='exit',
                    reason=f"{poor_reason} (quote_size at minimum, has position)",
                    new_enabled=True,  # Keep enabled to allow position exit
                    new_min_spread=0.50,
                    new_quote_size=None,
                    new_max_inventory_yes=None,
                    new_max_inventory_no=None,
                    pnl_24h=analysis.pnl_24h,
                    fill_count_24h=analysis.fill_count_24h,
                    fill_count_48h=analysis.fill_count_48h,
                    has_position=analysis.has_position,
                    position_qty=analysis.position_qty,
                ))
            else:
                # Quote size at minimum and no position: disable the market
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='exit',
                    reason=f"{poor_reason} (quote_size at minimum)",
                    new_enabled=False,
                    new_min_spread=None,
                    new_quote_size=None,
                    new_max_inventory_yes=None,
                    new_max_inventory_no=None,
                    pnl_24h=analysis.pnl_24h,
                    fill_count_24h=analysis.fill_count_24h,
                    fill_count_48h=analysis.fill_count_48h,
                    has_position=analysis.has_position,
                    position_qty=analysis.position_qty,
                ))
            continue

        # Check EXPAND conditions (only for enabled markets with fills)
        if (analysis.current_enabled and
            analysis.pnl_24h > 0 and
            analysis.median_fill_size is not None and
            analysis.median_fill_size == analysis.current_quote_size):

            # Cap expansion at MAX_QUOTE_SIZE and MAX_INVENTORY
            new_quote_size = min(MAX_QUOTE_SIZE, analysis.current_quote_size * 2)
            new_max_inv_yes = min(MAX_INVENTORY, analysis.current_max_inventory_yes * 2)
            new_max_inv_no = min(MAX_INVENTORY, analysis.current_max_inventory_no * 2)

            # Skip if already at max
            if (new_quote_size == analysis.current_quote_size and
                new_max_inv_yes == analysis.current_max_inventory_yes and
                new_max_inv_no == analysis.current_max_inventory_no):
                continue

            recommendations.append(RecommendedAction(
                market_id=market_id,
                event_ticker=analysis.event_ticker,
                action='expand',
                reason=f"Positive P&L (${analysis.pnl_24h:.2f}) and median fill = quote size ({analysis.current_quote_size})",
                new_enabled=None,
                new_min_spread=None,
                new_quote_size=new_quote_size,
                new_max_inventory_yes=new_max_inv_yes,
                new_max_inventory_no=new_max_inv_no,
                pnl_24h=analysis.pnl_24h,
                fill_count_24h=analysis.fill_count_24h,
                fill_count_48h=analysis.fill_count_48h,
                has_position=analysis.has_position,
                position_qty=analysis.position_qty,
            ))

    # Third pass: For expand events, activate sibling markets that are not already active
    if expand_events:
        print(f"\nFetching sibling markets for {len(expand_events)} expanding events...")
        sibling_recommendations = generate_sibling_activations(
            expand_events, existing_configs, all_markets_by_event
        )
        recommendations.extend(sibling_recommendations)

    return recommendations


def generate_sibling_activations(
    expand_events: set,
    existing_configs: Dict[str, MarketConfig],
    active_markets_by_event: Dict[str, List[str]]
) -> List[RecommendedAction]:
    """Generate recommendations to activate sibling markets for expanding events.

    Args:
        expand_events: Set of event tickers with expanding markets
        existing_configs: Dictionary of existing market configs in DynamoDB
        active_markets_by_event: Dictionary of currently active markets by event

    Returns:
        List of activate_sibling recommendations
    """
    recommendations = []
    existing_market_ids = set(existing_configs.keys())

    for event_ticker in expand_events:
        # Fetch all markets for this event from Kalshi API
        event_markets = fetch_event_markets(event_ticker)

        if not event_markets:
            print(f"  {event_ticker}: No markets found")
            continue

        # Find markets that are not already in our config
        active_markets = set(active_markets_by_event.get(event_ticker, []))
        new_markets = []

        for market in event_markets:
            market_id = market.get('ticker')
            if not market_id:
                continue

            # Skip if already active or already in config (even if disabled)
            if market_id in active_markets or market_id in existing_market_ids:
                continue

            # Skip closed/settled markets
            status = market.get('status', '')
            if status in ('closed', 'settled'):
                continue

            # Skip markets with low volume (< 10 in last 24hrs)
            volume_24h = market.get('volume_24h', 0) or 0
            if volume_24h < 10:
                continue

            new_markets.append(market)

        if not new_markets:
            print(f"  {event_ticker}: No new sibling markets to activate")
            continue

        # Sort by 24hr volume (descending) and take top 5
        new_markets.sort(key=lambda m: m.get('volume_24h', 0) or 0, reverse=True)
        top_markets = new_markets[:5]

        print(f"  {event_ticker}: Found {len(new_markets)} new sibling markets, activating top {len(top_markets)} by volume...")

        # Assess fair value for each top market (in parallel for efficiency)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for market in top_markets:
                market_id = market.get('ticker')
                future = executor.submit(
                    assess_fair_value,
                    market.get('title', ''),
                    market.get('subtitle', ''),
                    market.get('rules_primary', '')
                )
                futures[future] = (market_id, market)

            for future in as_completed(futures):
                market_id, market = futures[future]
                fv_result = future.result()

                # Parse probability from result
                prob_str = fv_result.get("probability", "N/A")
                fair_value = None
                if prob_str and prob_str != "N/A":
                    try:
                        fair_value = float(str(prob_str).replace("%", "").strip())
                    except (ValueError, AttributeError):
                        pass

                volume_24h = market.get('volume_24h', 0) or 0
                print(f"    {market_id}: volume_24h={volume_24h}, fair_value={fair_value}")

                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='activate_sibling',
                    reason=f"Sibling of expanding market in event {event_ticker}",
                    new_enabled=True,
                    new_min_spread=DEFAULT_MIN_SPREAD,
                    new_quote_size=DEFAULT_QUOTE_SIZE,
                    new_max_inventory_yes=DEFAULT_MAX_INVENTORY_YES,
                    new_max_inventory_no=DEFAULT_MAX_INVENTORY_NO,
                    pnl_24h=0.0,
                    fill_count_24h=0,
                    fill_count_48h=0,
                    has_position=False,
                    position_qty=0,
                    fair_value=fair_value,
                    fair_value_rationale=fv_result.get("rationale"),
                ))

    return recommendations


def write_recommendations_csv(
    recommendations: List[RecommendedAction],
    output_path: Path
) -> None:
    """Write recommendations to a CSV file for review.

    Args:
        recommendations: List of recommended actions
        output_path: Path to write CSV file
    """
    fieldnames = [
        'market_id',
        'event_ticker',
        'action',
        'reason',
        'new_enabled',
        'new_min_spread',
        'new_quote_size',
        'new_max_inventory_yes',
        'new_max_inventory_no',
        'pnl_24h',
        'fill_count_24h',
        'fill_count_48h',
        'has_position',
        'position_qty',
        'fair_value',
        'fair_value_rationale',
        'approve'  # User can set to 'yes' or 'no'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in recommendations:
            # Default approve based on action type
            # skip_event_active actions default to 'no' since no changes needed
            default_approve = 'no' if rec.action == 'skip_event_active' else 'yes'

            writer.writerow({
                'market_id': rec.market_id,
                'event_ticker': rec.event_ticker or '',
                'action': rec.action,
                'reason': rec.reason,
                'new_enabled': rec.new_enabled if rec.new_enabled is not None else '',
                'new_min_spread': rec.new_min_spread if rec.new_min_spread is not None else '',
                'new_quote_size': int(rec.new_quote_size) if rec.new_quote_size is not None else '',
                'new_max_inventory_yes': int(rec.new_max_inventory_yes) if rec.new_max_inventory_yes is not None else '',
                'new_max_inventory_no': int(rec.new_max_inventory_no) if rec.new_max_inventory_no is not None else '',
                'pnl_24h': f'{rec.pnl_24h:.2f}',
                'fill_count_24h': rec.fill_count_24h,
                'fill_count_48h': rec.fill_count_48h,
                'has_position': rec.has_position,
                'position_qty': int(rec.position_qty) if rec.position_qty else 0,
                'fair_value': f'{rec.fair_value:.1f}' if rec.fair_value is not None else '',
                'fair_value_rationale': rec.fair_value_rationale or '',
                'approve': default_approve,
            })

    print(f"Wrote {len(recommendations)} recommendations to {output_path}")


def read_approved_recommendations(csv_path: Path) -> List[Dict]:
    """Read and filter approved recommendations from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of approved recommendation dictionaries
    """
    approved = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('approve', '').lower() in ('yes', 'y', 'true', '1'):
                approved.append(row)

    return approved


def execute_updates(
    dynamo: DynamoDBClient,
    approved: List[Dict]
) -> Tuple[int, int]:
    """Execute approved market config updates.

    Args:
        dynamo: DynamoDB client
        approved: List of approved recommendation dictionaries

    Returns:
        Tuple of (success_count, failure_count)
    """
    success = 0
    failure = 0

    for rec in approved:
        market_id = rec['market_id']
        action = rec.get('action', '')

        # Get current config or create new one for activate_sibling
        config = dynamo.get_market_config(market_id)

        if config is None:
            if action == 'activate_sibling':
                # Create new config for sibling market
                config = MarketConfig(
                    market_id=market_id,
                    enabled=True,
                    max_inventory_yes=DEFAULT_MAX_INVENTORY_YES,
                    max_inventory_no=DEFAULT_MAX_INVENTORY_NO,
                    min_spread=DEFAULT_MIN_SPREAD,
                    quote_size=DEFAULT_QUOTE_SIZE,
                    inventory_skew_factor=0.5,
                    event_ticker=rec.get('event_ticker') or None,
                    created_at=datetime.now(timezone.utc),
                )
                # Parse fair_value if present and set it
                if rec.get('fair_value'):
                    try:
                        config.fair_value = float(rec['fair_value']) / 100.0  # Convert from % to decimal
                    except (ValueError, TypeError):
                        pass
            else:
                print(f"Warning: Market config not found for {market_id}, skipping")
                failure += 1
                continue
        else:
            # Apply updates to existing config
            if rec.get('new_enabled'):
                enabled_str = rec['new_enabled'].lower()
                if enabled_str in ('true', 'yes', '1'):
                    config.enabled = True
                elif enabled_str in ('false', 'no', '0'):
                    config.enabled = False

            if rec.get('new_min_spread'):
                try:
                    config.min_spread = float(rec['new_min_spread'])
                except ValueError:
                    pass

            if rec.get('new_quote_size'):
                try:
                    # Handle float strings like "20.0" from CSV
                    config.quote_size = int(float(rec['new_quote_size']))
                except (ValueError, TypeError):
                    pass

            if rec.get('new_max_inventory_yes'):
                try:
                    # Handle float strings like "40.0" from CSV
                    config.max_inventory_yes = int(float(rec['new_max_inventory_yes']))
                except (ValueError, TypeError):
                    pass

            if rec.get('new_max_inventory_no'):
                try:
                    # Handle float strings like "40.0" from CSV
                    config.max_inventory_no = int(float(rec['new_max_inventory_no']))
                except (ValueError, TypeError):
                    pass

        # Save updated/new config
        if dynamo.put_market_config(config):
            if action == 'activate_sibling':
                print(f"✓ Created {market_id}: {action} (fair_value={rec.get('fair_value', 'N/A')})")
            else:
                print(f"✓ Updated {market_id}: {action}")
            success += 1
        else:
            print(f"✗ Failed to update {market_id}")
            failure += 1

    return success, failure


def main():
    parser = argparse.ArgumentParser(
        description='Generate and apply market config updates based on performance analysis.'
    )
    parser.add_argument(
        '--env',
        choices=['demo', 'prod'],
        required=True,
        help='Environment to analyze (demo or prod)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV path (default: market_updates_YYYYMMDD_HHMMSS.csv)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate CSV but do not prompt for execution'
    )
    parser.add_argument(
        '--execute',
        type=Path,
        default=None,
        help='Skip analysis and execute from an existing CSV file'
    )
    parser.add_argument(
        '--pnl-lookback',
        type=int,
        default=24,
        help='Hours to look back for P&L calculation (default: 24)'
    )
    parser.add_argument(
        '--volume-lookback',
        type=int,
        default=48,
        help='Hours to look back for fill count/volume (default: 48)'
    )

    args = parser.parse_args()

    # Initialize DynamoDB client
    dynamo = DynamoDBClient(region=args.region, environment=args.env)

    if args.execute:
        # Execute mode: apply updates from existing CSV
        if not args.execute.exists():
            print(f"Error: CSV file not found: {args.execute}")
            sys.exit(1)

        approved = read_approved_recommendations(args.execute)
        if not approved:
            print("No approved recommendations found in CSV")
            sys.exit(0)

        print(f"\nFound {len(approved)} approved updates to apply...")
        success, failure = execute_updates(dynamo, approved)
        print(f"\nCompleted: {success} succeeded, {failure} failed")
        sys.exit(0 if failure == 0 else 1)

    # Analysis mode
    print(f"Analyzing markets in {args.env} environment...")
    print(f"  P&L lookback: {args.pnl_lookback} hours")
    print(f"  Volume lookback: {args.volume_lookback} hours")
    analyses = analyze_markets(
        dynamo, args.env,
        pnl_lookback_hours=args.pnl_lookback,
        volume_lookback_hours=args.volume_lookback
    )
    print(f"Analyzed {len(analyses)} markets")

    # Get all existing configs (for sibling activation check)
    existing_configs = dynamo.get_all_market_configs(enabled_only=False)

    # Generate recommendations
    recommendations = generate_recommendations(analyses, existing_configs)

    if not recommendations:
        print("\nNo recommended actions at this time.")
        sys.exit(0)

    # Summary
    exits = [r for r in recommendations if r.action == 'exit']
    scale_downs = [r for r in recommendations if r.action == 'scale_down']
    expands = [r for r in recommendations if r.action == 'expand']
    siblings = [r for r in recommendations if r.action == 'activate_sibling']
    print(f"\nRecommendations: {len(exits)} exits, {len(scale_downs)} scale-downs, {len(expands)} expansions, {len(siblings)} sibling activations")

    # Write CSV
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"market_updates_{args.env}_{timestamp}.csv")

    write_recommendations_csv(recommendations, output_path)

    if args.dry_run:
        print("\nDry run mode: Exiting without applying changes.")
        print(f"Review the CSV at: {output_path}")
        sys.exit(0)

    # Interactive approval flow
    print("\n" + "="*60)
    print("Please review and edit the CSV file if needed.")
    print(f"File: {output_path}")
    print("\nSet 'approve' column to 'yes' or 'no' for each row.")
    print("="*60)

    while True:
        response = input("\nPress Enter when ready to apply approved updates (or 'q' to quit): ")
        if response.lower() in ('q', 'quit', 'exit'):
            print("Aborted.")
            sys.exit(0)

        # Re-read the CSV to get user edits
        approved = read_approved_recommendations(output_path)

        if not approved:
            print("No approved recommendations found. Edit the CSV and try again.")
            continue

        print(f"\nReady to apply {len(approved)} updates:")
        for rec in approved:
            print(f"  - {rec['market_id']}: {rec['action']}")

        confirm = input("\nConfirm? (y/n): ")
        if confirm.lower() in ('y', 'yes'):
            break
        else:
            print("Edit the CSV and press Enter when ready.")

    # Execute updates
    print("\nApplying updates...")
    success, failure = execute_updates(dynamo, approved)
    print(f"\nCompleted: {success} succeeded, {failure} failed")
    sys.exit(0 if failure == 0 else 1)


if __name__ == '__main__':
    main()
