"""Migration script to copy data from old DynamoDB tables to new environment-specific tables."""

import boto3
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Old table names (without suffix)
OLD_TABLES = [
    'dora_market_config',
    'dora_state',
    'dora_trade_log',
    'dora_decision_log'
]

# New table suffix
TARGET_SUFFIX = '_demo'


def migrate_table(dynamodb, old_table_name: str, new_table_name: str) -> int:
    """Migrate all items from old table to new table.

    Returns:
        Number of items migrated
    """
    old_table = dynamodb.Table(old_table_name)
    new_table = dynamodb.Table(new_table_name)

    items_migrated = 0

    try:
        # Scan all items from old table
        response = old_table.scan()
        items = response.get('Items', [])

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = old_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        logger.info(f"Found {len(items)} items in {old_table_name}")

        # Write items to new table
        with new_table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)
                items_migrated += 1

        logger.info(f"Migrated {items_migrated} items to {new_table_name}")
        return items_migrated

    except Exception as e:
        logger.error(f"Error migrating {old_table_name}: {e}")
        raise


def main():
    region = 'us-east-1'

    # Allow overriding target suffix via command line
    target_suffix = TARGET_SUFFIX
    if len(sys.argv) > 1:
        target_suffix = sys.argv[1]
        if not target_suffix.startswith('_'):
            target_suffix = f'_{target_suffix}'

    logger.info(f"Migrating tables to *{target_suffix} suffix")

    dynamodb = boto3.resource('dynamodb', region_name=region)

    total_migrated = 0

    for old_table in OLD_TABLES:
        new_table = f"{old_table}{target_suffix}"
        logger.info(f"\n{'='*50}")
        logger.info(f"Migrating: {old_table} -> {new_table}")
        logger.info('='*50)

        try:
            count = migrate_table(dynamodb, old_table, new_table)
            total_migrated += count
        except Exception as e:
            logger.error(f"Failed to migrate {old_table}: {e}")
            logger.info("Continuing with next table...")

    logger.info(f"\n{'='*50}")
    logger.info(f"Migration complete! Total items migrated: {total_migrated}")
    logger.info('='*50)


if __name__ == '__main__':
    main()
