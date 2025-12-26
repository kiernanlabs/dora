"""One-time migration script to fix trade_log dates.

The date column was incorrectly set to the log time instead of the fill time.
This script:
1. Scans all items in the trade_log table
2. Extracts the actual date from fill_timestamp or timestamp#order_id
3. If the date doesn't match, deletes the old item and inserts a corrected one
"""

import argparse
from datetime import datetime
import boto3
from botocore.exceptions import ClientError


TABLE_SUFFIXES = {
    "demo": "_demo",
    "prod": "_prod",
}


def fix_trade_log_dates(region: str, environment: str, dry_run: bool = True) -> None:
    """Fix dates in trade_log table.

    Args:
        region: AWS region
        environment: 'demo' or 'prod'
        dry_run: If True, only print what would be done without making changes
    """
    if environment not in TABLE_SUFFIXES:
        raise ValueError(f"Invalid environment: {environment}")

    suffix = TABLE_SUFFIXES[environment]
    table_name = f"dora_trade_log{suffix}"

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    print(f"Scanning {table_name}...")

    # Scan all items
    items_to_fix = []
    scan_kwargs = {}

    while True:
        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])

        for item in items:
            current_date = item.get("date")
            timestamp_order_id = item.get("timestamp#order_id", "")
            fill_timestamp = item.get("fill_timestamp")

            # Extract actual timestamp
            if fill_timestamp:
                # fill_timestamp is an ISO string like "2025-12-08T14:05:11.024452+00:00"
                try:
                    actual_dt = datetime.fromisoformat(fill_timestamp)
                    actual_date = actual_dt.strftime("%Y-%m-%d")
                except ValueError:
                    print(f"  Warning: Could not parse fill_timestamp: {fill_timestamp}")
                    continue
            elif "#" in timestamp_order_id:
                # Extract from timestamp#order_id like "2025-12-08T14:05:11.024452+00:00#abc123"
                timestamp_part = timestamp_order_id.split("#")[0]
                try:
                    actual_dt = datetime.fromisoformat(timestamp_part)
                    actual_date = actual_dt.strftime("%Y-%m-%d")
                except ValueError:
                    print(f"  Warning: Could not parse timestamp from sort key: {timestamp_order_id}")
                    continue
            else:
                print(f"  Warning: No timestamp found for item with date={current_date}")
                continue

            if current_date != actual_date:
                items_to_fix.append({
                    "old_item": item,
                    "old_date": current_date,
                    "new_date": actual_date,
                })

        # Handle pagination
        if "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        else:
            break

    print(f"\nFound {len(items_to_fix)} items with incorrect dates.")

    if not items_to_fix:
        print("Nothing to fix!")
        return

    # Show what will be fixed
    print("\nItems to fix:")
    for fix in items_to_fix:
        market = fix["old_item"].get("market_id", "unknown")
        fill_id = fix["old_item"].get("fill_id", "unknown")
        print(f"  {market} (fill_id={fill_id}): {fix['old_date']} -> {fix['new_date']}")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run with --execute to apply fixes.")
        return

    # Apply fixes
    print("\nApplying fixes...")
    fixed_count = 0
    error_count = 0

    for fix in items_to_fix:
        old_item = fix["old_item"]
        old_date = fix["old_date"]
        new_date = fix["new_date"]
        timestamp_order_id = old_item.get("timestamp#order_id")

        try:
            # Delete old item
            table.delete_item(
                Key={
                    "date": old_date,
                    "timestamp#order_id": timestamp_order_id,
                }
            )

            # Insert new item with corrected date
            new_item = dict(old_item)
            new_item["date"] = new_date
            table.put_item(Item=new_item)

            fixed_count += 1
            market = old_item.get("market_id", "unknown")
            print(f"  Fixed: {market} ({old_date} -> {new_date})")

        except ClientError as e:
            error_count += 1
            print(f"  Error fixing item: {e}")

    print(f"\nDone! Fixed {fixed_count} items, {error_count} errors.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix trade_log dates to match fill timestamps."
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--env",
        choices=sorted(TABLE_SUFFIXES.keys()),
        default="demo",
        help="Environment (default: demo)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply the fixes (default is dry-run)",
    )
    args = parser.parse_args()

    fix_trade_log_dates(
        region=args.region,
        environment=args.env,
        dry_run=not args.execute,
    )


if __name__ == "__main__":
    main()
