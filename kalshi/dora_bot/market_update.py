#!/usr/bin/env python3
"""
Market Update Script for Dora Bot.

This script analyzes market performance and generates recommended config updates:
1. Markets to EXIT:
   - (A) P&L < -$0.50 over last 24hrs, OR
   - (B) No fills in last 48hrs
   For exit markets: if position exists, set min_spread=0.50; otherwise set enabled=False
   NOTE: Markets are kept active if other markets in the same event are still active

2. Markets to EXPAND:
   - Positive P&L in last 24hrs AND median fill size = current quote size
   For expand markets: double quote_size, max_inventory_yes, max_inventory_no

Usage:
    python -m dora_bot.market_update --env prod
    python -m dora_bot.market_update --env demo --dry-run
"""

import argparse
import base64
import csv
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

from dora_bot.dynamo import DynamoDBClient
from dora_bot.exchange_client import KalshiExchangeClient
from dora_bot.kalshi_client import KalshiHttpClient, Environment
from dora_bot.models import MarketConfig, Position


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
    action: str  # 'exit' or 'expand' or 'skip_event_active'
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


def fetch_event_tickers(
    exchange: KalshiExchangeClient,
    market_ids: List[str]
) -> Dict[str, Optional[str]]:
    """Fetch event_ticker for each market from the exchange.

    Args:
        exchange: Exchange client
        market_ids: List of market IDs to fetch

    Returns:
        Dictionary mapping market_id to event_ticker (or None if not found)
    """
    event_tickers = {}

    for market_id in market_ids:
        try:
            market_info = exchange.get_market_info(market_id)
            # The API returns market info with 'market' key containing the data
            market_data = market_info.get('market', market_info)
            event_ticker = market_data.get('event_ticker')
            event_tickers[market_id] = event_ticker
        except Exception as e:
            print(f"Warning: Could not fetch event_ticker for {market_id}: {e}")
            event_tickers[market_id] = None

    return event_tickers


def analyze_markets(
    dynamo: DynamoDBClient,
    exchange: KalshiExchangeClient,
    environment: str
) -> Dict[str, MarketAnalysis]:
    """Analyze all markets based on fills and positions.

    Args:
        dynamo: DynamoDB client
        exchange: Exchange client for fetching market info
        environment: 'demo' or 'prod'

    Returns:
        Dictionary mapping market_id to MarketAnalysis
    """
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_48h = now - timedelta(hours=48)

    # Get all market configs (including disabled ones)
    configs = dynamo.get_all_market_configs(enabled_only=False)

    # Fetch event_tickers for all markets
    print(f"Fetching event info for {len(configs)} markets...")
    event_tickers = fetch_event_tickers(exchange, list(configs.keys()))

    # Get current positions
    positions = dynamo.get_positions()

    # Get fills for the last 48 hours
    suffix = '_demo' if environment == 'demo' else '_prod'
    table_name = f"dora_trade_log{suffix}"

    fills = get_fills_for_date_range(
        dynamo.dynamodb,
        table_name,
        cutoff_48h,
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
        fills_24h = []
        fills_48h = []
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
                    if ts >= cutoff_48h:
                        fills_48h.append(fill)
                        if ts >= cutoff_24h:
                            fills_24h.append(fill)
                except (ValueError, TypeError):
                    # If we can't parse the timestamp, include in 48h window
                    fills_48h.append(fill)

        # Calculate P&L for last 24 hours
        pnl_24h = 0.0
        fill_sizes = []
        for fill in fills_24h:
            # P&L is calculated as: (sell_price - buy_price) * size - fees
            # For individual fills, we use pnl_realized if available
            pnl = fill.get('pnl_realized', 0.0) or 0.0
            fees = fill.get('fees', 0.0) or 0.0
            pnl_24h += pnl - fees

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
            pnl_24h=pnl_24h,
            fill_count_24h=len(fills_24h),
            fill_count_48h=len(fills_48h),
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
    analyses: Dict[str, MarketAnalysis]
) -> List[RecommendedAction]:
    """Generate recommended actions based on market analysis.

    Args:
        analyses: Dictionary of market analyses

    Returns:
        List of recommended actions
    """
    recommendations = []

    # Build maps for event-based analysis
    # markets_by_event: all enabled markets grouped by event
    # exit_candidates: markets that meet exit criteria
    all_markets_by_event: Dict[str, List[str]] = defaultdict(list)
    exit_candidates: Dict[str, str] = {}  # market_id -> exit_reason

    # First pass: identify exit candidates and build event map
    for market_id, analysis in analyses.items():
        # Skip disabled markets without positions (already exited)
        if not analysis.current_enabled and not analysis.has_position:
            continue

        # Track all enabled markets by event
        if analysis.current_enabled and analysis.event_ticker:
            all_markets_by_event[analysis.event_ticker].append(market_id)

        # Check EXIT conditions
        exit_reason = None

        # (A) P&L < -$0.50 over last 24hrs
        if analysis.pnl_24h < -0.50:
            exit_reason = f"P&L < -$0.50 in 24h (${analysis.pnl_24h:.2f})"

        # (B) No fills in last 48hrs (only for enabled markets)
        elif analysis.current_enabled and analysis.fill_count_48h == 0:
            exit_reason = "No fills in 48 hours"

        if exit_reason:
            exit_candidates[market_id] = exit_reason

    # Second pass: generate recommendations, checking event siblings
    for market_id, analysis in analyses.items():
        # Skip disabled markets without positions (already exited)
        if not analysis.current_enabled and not analysis.has_position:
            continue

        exit_reason = exit_candidates.get(market_id)

        if exit_reason:
            # Check if other markets in the same event are still active AND not exit candidates
            event_ticker = analysis.event_ticker
            sibling_active_non_exit_markets = []
            if event_ticker:
                sibling_active_non_exit_markets = [
                    m for m in all_markets_by_event.get(event_ticker, [])
                    if m != market_id and m not in exit_candidates
                ]

            if sibling_active_non_exit_markets:
                # Other markets in this event are active and NOT exiting - skip exit, just note it
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='skip_event_active',
                    reason=f"{exit_reason} (but {len(sibling_active_non_exit_markets)} sibling markets active)",
                    new_enabled=None,
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
            elif analysis.has_position:
                # Has position: set min_spread to 0.50 to exit while allowing sells
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='exit',
                    reason=exit_reason,
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
                # No position: disable the market
                recommendations.append(RecommendedAction(
                    market_id=market_id,
                    event_ticker=event_ticker,
                    action='exit',
                    reason=exit_reason,
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

            recommendations.append(RecommendedAction(
                market_id=market_id,
                event_ticker=analysis.event_ticker,
                action='expand',
                reason=f"Positive P&L (${analysis.pnl_24h:.2f}) and median fill = quote size ({analysis.current_quote_size})",
                new_enabled=None,
                new_min_spread=None,
                new_quote_size=analysis.current_quote_size * 2,
                new_max_inventory_yes=analysis.current_max_inventory_yes * 2,
                new_max_inventory_no=analysis.current_max_inventory_no * 2,
                pnl_24h=analysis.pnl_24h,
                fill_count_24h=analysis.fill_count_24h,
                fill_count_48h=analysis.fill_count_48h,
                has_position=analysis.has_position,
                position_qty=analysis.position_qty,
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
                'new_quote_size': rec.new_quote_size if rec.new_quote_size is not None else '',
                'new_max_inventory_yes': rec.new_max_inventory_yes if rec.new_max_inventory_yes is not None else '',
                'new_max_inventory_no': rec.new_max_inventory_no if rec.new_max_inventory_no is not None else '',
                'pnl_24h': f'{rec.pnl_24h:.2f}',
                'fill_count_24h': rec.fill_count_24h,
                'fill_count_48h': rec.fill_count_48h,
                'has_position': rec.has_position,
                'position_qty': rec.position_qty,
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

        # Get current config
        config = dynamo.get_market_config(market_id)
        if config is None:
            print(f"Warning: Market config not found for {market_id}, skipping")
            failure += 1
            continue

        # Apply updates
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
                config.quote_size = int(rec['new_quote_size'])
            except ValueError:
                pass

        if rec.get('new_max_inventory_yes'):
            try:
                config.max_inventory_yes = int(rec['new_max_inventory_yes'])
            except ValueError:
                pass

        if rec.get('new_max_inventory_no'):
            try:
                config.max_inventory_no = int(rec['new_max_inventory_no'])
            except ValueError:
                pass

        # Save updated config
        if dynamo.put_market_config(config):
            print(f"✓ Updated {market_id}: {rec['action']}")
            success += 1
        else:
            print(f"✗ Failed to update {market_id}")
            failure += 1

    return success, failure


def create_exchange_client(environment: str) -> KalshiExchangeClient:
    """Create an exchange client for the given environment.

    Args:
        environment: 'demo' or 'prod'

    Returns:
        Configured KalshiExchangeClient
    """
    use_demo = environment == 'demo'

    # Check for container mode (base64 encoded keys)
    if os.getenv('KALSHI_KEY_ID') and os.getenv('KALSHI_PRIVATE_KEY'):
        keyid = os.getenv('KALSHI_KEY_ID')
        private_key_b64 = os.getenv('KALSHI_PRIVATE_KEY')

        private_key_pem = base64.b64decode(private_key_b64)
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None
        )
    else:
        # Local mode: load credentials from .env file
        kalshi_dir = os.path.join(os.path.dirname(__file__), '..')
        load_dotenv(os.path.join(kalshi_dir, '.env'))

        keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
        keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')

        if not keyid or not keyfile:
            raise ValueError(f"Missing API credentials in .env file for {environment}")

        # Resolve keyfile path relative to kalshi directory
        keyfile_path = os.path.join(kalshi_dir, keyfile)

        if not os.path.exists(keyfile_path):
            raise FileNotFoundError(f"Private key file not found: {keyfile_path}")

        with open(keyfile_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )

    env = Environment.DEMO if use_demo else Environment.PROD

    kalshi_client = KalshiHttpClient(
        key_id=keyid,
        private_key=private_key,
        environment=env
    )

    return KalshiExchangeClient(
        kalshi_client,
        environment=environment,
    )


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

    # Analysis mode - create exchange client for fetching market info
    print(f"Initializing exchange client for {args.env}...")
    exchange = create_exchange_client(args.env)

    print(f"Analyzing markets in {args.env} environment...")
    analyses = analyze_markets(dynamo, exchange, args.env)
    print(f"Analyzed {len(analyses)} markets")

    # Generate recommendations
    recommendations = generate_recommendations(analyses)

    if not recommendations:
        print("\nNo recommended actions at this time.")
        sys.exit(0)

    # Summary
    exits = [r for r in recommendations if r.action == 'exit']
    expands = [r for r in recommendations if r.action == 'expand']
    skipped = [r for r in recommendations if r.action == 'skip_event_active']
    print(f"\nRecommendations: {len(exits)} exits, {len(expands)} expansions, {len(skipped)} skipped (event active)")

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
