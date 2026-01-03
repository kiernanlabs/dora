"""Add market_id-timestamp GSI to existing dora_decision_log tables.

This migration script adds a Global Secondary Index to the decision_log table
to enable efficient queries by market_id, significantly improving performance
for decision context lookups.

Usage:
    python add_decision_log_gsi.py <region> [--env {demo|prod|both}]

Example:
    python add_decision_log_gsi.py us-east-1 --env both
"""

import argparse
import boto3
from botocore.exceptions import ClientError
import time


TABLE_SUFFIXES = {
    "demo": "_demo",
    "prod": "_prod",
}


def table_name(base_name: str, environment: str) -> str:
    """Generate table name with environment suffix."""
    if environment not in TABLE_SUFFIXES:
        raise ValueError(f"Invalid environment: {environment}")
    return f"{base_name}{TABLE_SUFFIXES[environment]}"


def add_gsi_to_decision_log(region: str, environment: str, dry_run: bool = False) -> None:
    """Add market_id-timestamp GSI to the decision_log table.

    Args:
        region: AWS region
        environment: 'demo' or 'prod'
        dry_run: If True, only check if GSI exists without creating
    """
    client = boto3.client("dynamodb", region_name=region)
    tbl_name = table_name("dora_decision_log", environment)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing table: {tbl_name}")

    try:
        # Check if table exists and get current schema
        response = client.describe_table(TableName=tbl_name)
        table_info = response['Table']

        # Check if GSI already exists
        existing_gsis = table_info.get('GlobalSecondaryIndexes', [])
        gsi_name = "market_id-timestamp-index"

        if any(gsi['IndexName'] == gsi_name for gsi in existing_gsis):
            print(f"✓ GSI '{gsi_name}' already exists on {tbl_name}")
            return

        if dry_run:
            print(f"[DRY RUN] Would create GSI '{gsi_name}' on {tbl_name}")
            return

        # Add the GSI
        print(f"Creating GSI '{gsi_name}' on {tbl_name}...")
        client.update_table(
            TableName=tbl_name,
            AttributeDefinitions=[
                {"AttributeName": "market_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexUpdates=[
                {
                    "Create": {
                        "IndexName": gsi_name,
                        "KeySchema": [
                            {"AttributeName": "market_id", "KeyType": "HASH"},
                            {"AttributeName": "timestamp", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    }
                }
            ],
        )

        print(f"GSI creation initiated. Waiting for table to become active...")

        # Wait for the table to be active (GSI creation can take several minutes)
        waiter = client.get_waiter('table_exists')
        waiter.wait(TableName=tbl_name)

        # Poll until GSI is active
        # For large tables (600MB+), GSI creation can take 30+ minutes
        max_wait_time = 1800  # 30 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            response = client.describe_table(TableName=tbl_name)
            table_status = response['Table']['TableStatus']

            gsis = response['Table'].get('GlobalSecondaryIndexes', [])
            target_gsi = next((g for g in gsis if g['IndexName'] == gsi_name), None)

            if target_gsi:
                gsi_status = target_gsi['IndexStatus']
                print(f"  Table: {table_status}, GSI: {gsi_status}")

                if gsi_status == 'ACTIVE' and table_status == 'ACTIVE':
                    print(f"✓ GSI '{gsi_name}' successfully created on {tbl_name}")
                    return

            time.sleep(10)

        print(f"⚠ Timeout waiting for GSI to become active. Check AWS console.")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            print(f"✗ Table {tbl_name} does not exist")
        else:
            print(f"✗ Error: {e}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add market_id-timestamp GSI to dora_decision_log tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("region", help="AWS region, e.g. us-east-1")
    parser.add_argument(
        "--env",
        choices=sorted(TABLE_SUFFIXES.keys()) + ["both"],
        default="both",
        help="Environment to update (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check if GSI exists without creating it",
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"DynamoDB Decision Log GSI Migration")
    print(f"Region: {args.region}")
    print(f"Environment: {args.env}")
    print(f"Dry Run: {args.dry_run}")
    print(f"{'='*60}")

    if args.env == "both":
        for env in sorted(TABLE_SUFFIXES.keys()):
            add_gsi_to_decision_log(args.region, env, args.dry_run)
    else:
        add_gsi_to_decision_log(args.region, args.env, args.dry_run)

    print(f"\n{'='*60}")
    print("Migration complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
