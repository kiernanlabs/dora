#!/usr/bin/env python3
"""One-time script to backfill event_ticker and created_at fields for existing market configs."""

import sys
sys.path.insert(0, '/home/joey32/dora/kalshi')

import boto3
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

# Use prod tables
MARKET_CONFIG_TABLE = "dora_market_config_prod"
KALSHI_API_BASE = "https://api.elections.kalshi.com"


def fetch_event_ticker(market_id: str) -> Optional[str]:
    """Fetch event_ticker for a market from Kalshi API."""
    try:
        url = f"{KALSHI_API_BASE}/trade-api/v2/markets/{market_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        market_data = data.get('market', data)
        return market_data.get('event_ticker')
    except Exception as e:
        print(f"  Warning: Could not fetch event_ticker for {market_id}: {e}")
        return None


def get_default_created_at() -> str:
    """Return yesterday's date as the default created_at timestamp."""
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    return yesterday.isoformat()


def main():
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    config_table = dynamodb.Table(MARKET_CONFIG_TABLE)

    # Scan all market configs
    response = config_table.scan()
    items = response.get('Items', [])

    print(f"Found {len(items)} market configs")
    print()

    updated = 0
    skipped = 0

    for item in items:
        market_id = item['market_id']
        existing_event = item.get('event_ticker')
        existing_created = item.get('created_at')

        print(f"{market_id}:")

        # Fetch event_ticker if not already set
        event_ticker = existing_event
        if not existing_event:
            event_ticker = fetch_event_ticker(market_id)
            if event_ticker:
                print(f"  event_ticker: {event_ticker}")
            else:
                print(f"  event_ticker: (not found)")
        else:
            print(f"  event_ticker: {existing_event} (already set)")

        # Set created_at to yesterday if not already set
        created_at = existing_created
        if not existing_created:
            created_at = get_default_created_at()
            print(f"  created_at: {created_at} (defaulting to yesterday)")
        else:
            print(f"  created_at: {existing_created} (already set)")

        # Update if we have new values
        update_expr_parts = []
        expr_values = {}

        if event_ticker and not existing_event:
            update_expr_parts.append('event_ticker = :evt')
            expr_values[':evt'] = event_ticker

        if created_at and not existing_created:
            update_expr_parts.append('created_at = :cat')
            expr_values[':cat'] = created_at

        if update_expr_parts:
            update_expr_parts.append('updated_at = :ts')
            expr_values[':ts'] = datetime.now(timezone.utc).isoformat()

            config_table.update_item(
                Key={'market_id': market_id},
                UpdateExpression='SET ' + ', '.join(update_expr_parts),
                ExpressionAttributeValues=expr_values
            )
            updated += 1
            print(f"  -> Updated")
        else:
            skipped += 1
            print(f"  -> Skipped (no changes)")

        print()

    print(f"Done: {updated} updated, {skipped} skipped")


if __name__ == '__main__':
    main()
