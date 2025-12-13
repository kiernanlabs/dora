"""Script to create DynamoDB tables for Dora Bot."""

import boto3
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_tables(region: str = "us-east-1"):
    """Create all required DynamoDB tables.

    Args:
        region: AWS region
    """
    dynamodb = boto3.client('dynamodb', region_name=region)

    tables = [
        {
            'TableName': 'dora_market_config',
            'KeySchema': [
                {'AttributeName': 'market_id', 'KeyType': 'HASH'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'market_id', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST'
        },
        {
            'TableName': 'dora_state',
            'KeySchema': [
                {'AttributeName': 'key', 'KeyType': 'HASH'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'key', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST'
        },
        {
            'TableName': 'dora_trade_log',
            'KeySchema': [
                {'AttributeName': 'date', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp#order_id', 'KeyType': 'RANGE'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'date', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp#order_id', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST'
        },
        {
            'TableName': 'dora_decision_log',
            'KeySchema': [
                {'AttributeName': 'date', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'date', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST'
        }
    ]

    for table_config in tables:
        table_name = table_config['TableName']
        try:
            logger.info(f"Creating table: {table_name}")
            dynamodb.create_table(**table_config)
            logger.info(f"Table {table_name} created successfully")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                logger.info(f"Table {table_name} already exists")
            else:
                logger.error(f"Error creating table {table_name}: {e}")
                raise

    # Wait for tables to be created
    logger.info("Waiting for tables to become active...")
    for table_config in tables:
        table_name = table_config['TableName']
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        logger.info(f"Table {table_name} is active")

    logger.info("All tables created successfully!")


def initialize_global_config(region: str = "us-east-1"):
    """Initialize global config with defaults.

    Args:
        region: AWS region
    """
    from decimal import Decimal

    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table('dora_state')

    try:
        table.put_item(Item={
            'key': 'global_config',
            'max_total_exposure': 500,
            'max_daily_loss': Decimal('100.0'),
            'loop_interval_ms': 5000,  # 5 seconds for testing
            'trading_enabled': False,  # Start disabled for safety
            'risk_aversion_k': Decimal('0.5'),
            'cancel_on_startup': True
        })
        logger.info("Global config initialized")
    except ClientError as e:
        logger.error(f"Error initializing global config: {e}")


def create_sample_market_config(market_id: str, region: str = "us-east-1"):
    """Create a sample market configuration.

    Args:
        market_id: Market ticker
        region: AWS region
    """
    from decimal import Decimal

    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table('dora_market_config')

    from datetime import datetime

    try:
        table.put_item(Item={
            'market_id': market_id,
            'enabled': False,  # Start disabled
            'max_inventory_yes': 50,
            'max_inventory_no': 50,
            'min_spread': Decimal('0.06'),
            'quote_size': 10,
            'inventory_skew_factor': Decimal('0.5'),
            'updated_at': datetime.utcnow().isoformat()
        })
        logger.info(f"Sample market config created for {market_id}")
    except ClientError as e:
        logger.error(f"Error creating sample config: {e}")


if __name__ == "__main__":
    import sys

    region = "us-east-1"
    if len(sys.argv) > 1:
        region = sys.argv[1]

    logger.info(f"Setting up DynamoDB tables in region: {region}")

    # Create tables
    create_tables(region)

    # Initialize global config
    initialize_global_config(region)

    logger.info("\nSetup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Create market configs using create_sample_market_config()")
    logger.info("2. Enable markets by setting 'enabled': True in dora_market_config table")
    logger.info("3. Set 'trading_enabled': True in global_config when ready to trade")
