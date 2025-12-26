#!/usr/bin/env python3
"""One-time script to cut max_inventory_yes/no values by 50%."""

import sys
sys.path.insert(0, '/home/joey32/dora/kalshi')

from decimal import Decimal
import boto3
from datetime import datetime, timezone

# Use prod table
TABLE_NAME = "dora_market_config_prod"

def main():
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(TABLE_NAME)

    # Scan all items
    response = table.scan()
    items = response.get('Items', [])

    print(f"Found {len(items)} market configs")
    print()

    updated = 0
    for item in items:
        market_id = item['market_id']
        old_yes = int(item.get('max_inventory_yes', 100))
        old_no = int(item.get('max_inventory_no', 100))

        new_yes = max(1, old_yes // 2)  # Floor division, minimum 1
        new_no = max(1, old_no // 2)

        print(f"{market_id}: max_inventory_yes {old_yes} -> {new_yes}, max_inventory_no {old_no} -> {new_no}")

        # Update the item
        table.update_item(
            Key={'market_id': market_id},
            UpdateExpression='SET max_inventory_yes = :yes, max_inventory_no = :no, updated_at = :ts',
            ExpressionAttributeValues={
                ':yes': new_yes,
                ':no': new_no,
                ':ts': datetime.now(timezone.utc).isoformat()
            }
        )
        updated += 1

    print()
    print(f"Updated {updated} market configs")

if __name__ == '__main__':
    main()
