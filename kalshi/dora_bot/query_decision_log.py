#!/usr/bin/env python3
"""Query the last 10 decision log entries for a given market."""

import argparse
import json
from decimal import Decimal
from datetime import datetime, timedelta
import boto3
from boto3.dynamodb.conditions import Key, Attr


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def query_decision_log(market_id: str, environment: str = "prod", limit: int = 10, days_back: int = 7):
    """Query the last N decision log entries for a market.

    Args:
        market_id: The market ticker to query
        environment: 'demo' or 'prod'
        limit: Number of entries to return
        days_back: Number of days to look back
    """
    suffix = "_demo" if environment == "demo" else "_prod"
    table_name = f"dora_decision_log{suffix}"

    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(table_name)

    # Query across multiple days (partition keys)
    all_items = []
    today = datetime.utcnow()

    for i in range(days_back):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            response = table.query(
                KeyConditionExpression=Key('date').eq(date),
                FilterExpression=Attr('market_id').eq(market_id),
                ScanIndexForward=False  # Descending order by timestamp
            )
            all_items.extend(response.get('Items', []))
        except Exception as e:
            print(f"Error querying date {date}: {e}")

    # Sort all items by timestamp descending and take the last N
    all_items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    results = all_items[:limit]

    return results


def main():
    parser = argparse.ArgumentParser(description='Query decision log for a market')
    parser.add_argument('market_id', help='Market ticker to query')
    parser.add_argument('--env', '-e', choices=['demo', 'prod'], default='prod',
                        help='Environment (default: prod)')
    parser.add_argument('--limit', '-n', type=int, default=10,
                        help='Number of entries to return (default: 10)')
    parser.add_argument('--days', '-d', type=int, default=7,
                        help='Days to look back (default: 7)')

    args = parser.parse_args()

    print(f"Querying {args.env} decision log for market: {args.market_id}")
    print(f"Looking back {args.days} days, limit {args.limit} entries\n")

    results = query_decision_log(
        market_id=args.market_id,
        environment=args.env,
        limit=args.limit,
        days_back=args.days
    )

    if not results:
        print("No decision log entries found.")
        return

    print(f"Found {len(results)} entries:\n")
    print("-" * 80)

    for item in results:
        print(json.dumps(item, indent=2, cls=DecimalEncoder))
        print("-" * 80)


if __name__ == "__main__":
    main()
