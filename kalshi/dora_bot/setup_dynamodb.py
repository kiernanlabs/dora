"""Setup DynamoDB tables for Dora Bot."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError


TABLE_SUFFIXES = {
    "demo": "_demo",
    "prod": "_prod",
}


def _table_name(base_name: str, environment: str) -> str:
    if environment not in TABLE_SUFFIXES:
        raise ValueError(f"Invalid environment: {environment}")
    return f"{base_name}{TABLE_SUFFIXES[environment]}"


def _ensure_table(
    client,
    table_name: str,
    key_schema: List[Dict[str, str]],
    attribute_definitions: List[Dict[str, str]],
    gsi: Optional[List[Dict[str, object]]] = None,
) -> None:
    try:
        client.describe_table(TableName=table_name)
        return
    except client.exceptions.ResourceNotFoundException:
        pass

    params: Dict[str, object] = {
        "TableName": table_name,
        "KeySchema": key_schema,
        "AttributeDefinitions": attribute_definitions,
        "BillingMode": "PAY_PER_REQUEST",
    }
    if gsi:
        params["GlobalSecondaryIndexes"] = gsi

    client.create_table(**params)
    client.get_waiter("table_exists").wait(TableName=table_name)


def _enable_ttl(client, table_name: str, attribute_name: str) -> None:
    try:
        client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": attribute_name,
            },
        )
    except ClientError:
        # TTL can fail if the table is still creating or already enabled.
        pass


def create_tables(region: str, environment: str) -> None:
    client = boto3.client("dynamodb", region_name=region)

    _ensure_table(
        client,
        _table_name("dora_market_config", environment),
        key_schema=[{"AttributeName": "market_id", "KeyType": "HASH"}],
        attribute_definitions=[{"AttributeName": "market_id", "AttributeType": "S"}],
    )

    _ensure_table(
        client,
        _table_name("dora_state", environment),
        key_schema=[{"AttributeName": "key", "KeyType": "HASH"}],
        attribute_definitions=[{"AttributeName": "key", "AttributeType": "S"}],
    )

    _ensure_table(
        client,
        _table_name("dora_trade_log", environment),
        key_schema=[
            {"AttributeName": "date", "KeyType": "HASH"},
            {"AttributeName": "timestamp#order_id", "KeyType": "RANGE"},
        ],
        attribute_definitions=[
            {"AttributeName": "date", "AttributeType": "S"},
            {"AttributeName": "timestamp#order_id", "AttributeType": "S"},
        ],
    )

    _ensure_table(
        client,
        _table_name("dora_decision_log", environment),
        key_schema=[
            {"AttributeName": "date", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"},
        ],
        attribute_definitions=[
            {"AttributeName": "date", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
        ],
    )

    execution_table_name = _table_name("dora_execution_log", environment)
    _ensure_table(
        client,
        execution_table_name,
        key_schema=[
            {"AttributeName": "bot_run_id", "KeyType": "HASH"},
            {"AttributeName": "decision_id#event_ts", "KeyType": "RANGE"},
        ],
        attribute_definitions=[
            {"AttributeName": "bot_run_id", "AttributeType": "S"},
            {"AttributeName": "decision_id#event_ts", "AttributeType": "S"},
            {"AttributeName": "decision_id", "AttributeType": "S"},
            {"AttributeName": "event_ts", "AttributeType": "S"},
        ],
        gsi=[
            {
                "IndexName": "decision_id_event_ts",
                "KeySchema": [
                    {"AttributeName": "decision_id", "KeyType": "HASH"},
                    {"AttributeName": "event_ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )
    _enable_ttl(client, execution_table_name, "expires_at")


def create_sample_market_config(
    market_id: str,
    region: str = "us-east-1",
    environment: str = "demo",
) -> None:
    """Insert a minimal market config entry."""
    if environment not in TABLE_SUFFIXES:
        raise ValueError(f"Invalid environment: {environment}")
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(_table_name("dora_market_config", environment))
    table.put_item(
        Item={
            "market_id": market_id,
            "enabled": False,
            "max_inventory_yes": 100,
            "max_inventory_no": 100,
            "min_spread": 0.06,
            "quote_size": 10,
            "inventory_skew_factor": 0.5,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DynamoDB tables for Dora Bot.")
    parser.add_argument("region", help="AWS region, e.g. us-east-1")
    parser.add_argument(
        "--env",
        choices=sorted(TABLE_SUFFIXES.keys()) + ["both"],
        default="both",
        help="Environment to create tables for.",
    )
    args = parser.parse_args()

    if args.env == "both":
        for env in sorted(TABLE_SUFFIXES.keys()):
            create_tables(args.region, env)
    else:
        create_tables(args.region, args.env)


if __name__ == "__main__":
    main()
