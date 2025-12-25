"""
Sync markets.csv to DynamoDB market config table.

This script reads a markets.csv file (output from market_screener.py) with additional
manually-added columns and creates/updates items in the dora_market_config_prod table
for each row where activate=1.

Required CSV columns:
    - ticker: Market ticker
    - activate: 1 to enable, 0 to skip

Optional columns (defaults used if missing):
    - min_spread: Minimum spread to quote (default: 0.04)
    - inventory_skew_factor: How aggressively to skew quotes (default: 0.5)
    - max_inventory_no: Maximum NO contracts to hold (default: 20)
    - max_inventory_yes: Maximum YES contracts to hold (default: 20)
    - quote_size: Size of quotes to post (default: 10)
    - fair_value: AI-estimated fair value probability (CSV: 0-100%, converted to 0-1 decimal)

Usage:
    python sync_to_market_config.py markets.csv [--dry-run] [--environment prod]
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Required columns that must be present and non-null for activation
REQUIRED_COLUMNS = [
    "ticker",
    "activate",
]

# Columns with default values if not present in CSV
DEFAULT_VALUES = {
    "min_spread": 0.04,
    "inventory_skew_factor": 0.5,
    "max_inventory_no": 20,
    "max_inventory_yes": 20,
    "quote_size": 10,
}

# Optional columns
OPTIONAL_COLUMNS = ["fair_value", "title"]


def to_dynamo_item(obj: Any) -> Any:
    """Convert Python types to DynamoDB compatible types."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_dynamo_item(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dynamo_item(item) for item in obj]
    return obj


def validate_csv(df: pd.DataFrame) -> List[str]:
    """Validate that CSV has required columns.

    Args:
        df: DataFrame to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check for required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    return errors


def parse_row(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Parse a CSV row into a market config dict.

    Args:
        row: DataFrame row

    Returns:
        Market config dict or None if row should be skipped
    """
    # Check if activated
    activate = row.get("activate")
    if pd.isna(activate) or int(activate) != 1:
        return None

    ticker = row.get("ticker")
    if pd.isna(ticker):
        return None

    # Helper to get value from row or use default
    def get_value(col: str, convert_func, default=None):
        val = row.get(col)
        if pd.isna(val) if hasattr(pd, 'isna') else val is None:
            return default if default is not None else DEFAULT_VALUES.get(col)
        return convert_func(val)

    # Parse fields with defaults
    try:
        config = {
            "market_id": str(ticker),
            "enabled": True,
            "min_spread": get_value("min_spread", float),
            "inventory_skew_factor": get_value("inventory_skew_factor", float),
            "max_inventory_no": get_value("max_inventory_no", int),
            "max_inventory_yes": get_value("max_inventory_yes", int),
            "quote_size": get_value("quote_size", int),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, KeyError) as e:
        print(f"  WARNING: Could not parse row for {ticker}: {e}")
        return None

    # Parse optional fields
    fair_value = row.get("fair_value")
    if not pd.isna(fair_value):
        try:
            # Convert from percentage (0-100) to decimal (0-1)
            config["fair_value"] = float(fair_value) / 100.0
        except ValueError:
            pass

    return config


def sync_to_dynamodb(
    configs: List[Dict[str, Any]],
    environment: str = "prod",
    dry_run: bool = False,
) -> Dict[str, int]:
    """Sync market configs to DynamoDB.

    Args:
        configs: List of market config dicts
        environment: Environment ('demo' or 'prod')
        dry_run: If True, don't actually write to DynamoDB

    Returns:
        Dict with counts: {'created': N, 'updated': N, 'failed': N}
    """
    table_name = f"dora_market_config_{environment}"
    stats = {"created": 0, "updated": 0, "failed": 0}

    if dry_run:
        print(f"\n[DRY RUN] Would sync to table: {table_name}")
        for config in configs:
            print(f"  - {config['market_id']}: min_spread={config['min_spread']}, "
                  f"quote_size={config['quote_size']}, "
                  f"max_inv_yes={config['max_inventory_yes']}, "
                  f"max_inv_no={config['max_inventory_no']}")
            stats["created"] += 1
        return stats

    # Connect to DynamoDB
    try:
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.Table(table_name)

        # Verify table exists
        table.load()
    except ClientError as e:
        print(f"ERROR: Could not connect to table {table_name}: {e}")
        stats["failed"] = len(configs)
        return stats

    print(f"\nSyncing to table: {table_name}")

    for config in configs:
        market_id = config["market_id"]

        try:
            # Check if item exists
            response = table.get_item(Key={"market_id": market_id})
            exists = "Item" in response

            # Convert to DynamoDB types
            item = to_dynamo_item(config)

            # Put item
            table.put_item(Item=item)

            if exists:
                print(f"  UPDATED: {market_id}")
                stats["updated"] += 1
            else:
                print(f"  CREATED: {market_id}")
                stats["created"] += 1

        except ClientError as e:
            print(f"  FAILED: {market_id} - {e}")
            stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Sync markets.csv to DynamoDB market config table."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to markets.csv file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually writing to DynamoDB",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="prod",
        choices=["demo", "prod"],
        help="Environment to sync to (default: prod)",
    )
    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    print(f"Reading {args.input_file}...")

    # Read CSV
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        sys.exit(1)

    print(f"  Found {len(df)} rows")

    # Validate
    errors = validate_csv(df)
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Parse rows
    configs = []
    skipped = 0
    for _, row in df.iterrows():
        config = parse_row(row)
        if config:
            configs.append(config)
        else:
            skipped += 1

    print(f"  Parsed {len(configs)} active markets (skipped {skipped})")

    if not configs:
        print("\nNo markets to sync (none have activate=1)")
        return

    # Show summary before sync
    print("\nMarkets to sync:")
    for config in configs:
        fv = config.get("fair_value", "N/A")
        # fair_value is stored as decimal (0-1), display as percentage
        fv_str = f"{fv * 100:.0f}%" if isinstance(fv, (int, float)) else fv
        print(f"  {config['market_id']}: "
              f"spread={config['min_spread']}, "
              f"size={config['quote_size']}, "
              f"skew={config['inventory_skew_factor']}, "
              f"max_yes={config['max_inventory_yes']}, "
              f"max_no={config['max_inventory_no']}, "
              f"fair_value={fv_str}")

    # Sync to DynamoDB
    stats = sync_to_dynamodb(
        configs,
        environment=args.environment,
        dry_run=args.dry_run,
    )

    # Print summary
    print(f"\nSync complete:")
    print(f"  Created: {stats['created']}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Failed: {stats['failed']}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Remove --dry-run to sync.")


if __name__ == "__main__":
    main()
