"""
Script to create the DynamoDB table for AI insights.

This table stores AI-generated insights for events and markets.
"""
import boto3
import sys


def create_insights_table(environment='prod', region='us-east-1'):
    """Create the dora_ai_insights table in DynamoDB.

    Table schema:
    - Primary key: insight_type (partition key) + entity_id (sort key)
      - insight_type: 'event' or 'market'
      - entity_id: event_ticker or market_id
    - GSI: proposal_id-index
      - Used to query all insights for a given proposal batch

    Args:
        environment: 'demo' or 'prod'
        region: AWS region
    """
    dynamodb = boto3.client('dynamodb', region_name=region)
    table_name = f"dora_ai_insights_{environment}"

    try:
        # Check if table already exists
        existing_tables = dynamodb.list_tables()['TableNames']
        if table_name in existing_tables:
            print(f"Table {table_name} already exists.")
            return

        # Create the table
        print(f"Creating table {table_name}...")
        response = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'insight_type', 'KeyType': 'HASH'},  # Partition key
                {'AttributeName': 'entity_id', 'KeyType': 'RANGE'}      # Sort key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'insight_type', 'AttributeType': 'S'},
                {'AttributeName': 'entity_id', 'AttributeType': 'S'},
                {'AttributeName': 'proposal_id', 'AttributeType': 'S'},
                {'AttributeName': 'created_at', 'AttributeType': 'S'},
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'proposal_id-index',
                    'KeySchema': [
                        {'AttributeName': 'proposal_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'created_at', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'},
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )

        # Wait for table to be created
        print("Waiting for table to be created...")
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=table_name)

        # Enable TTL after table creation
        print("Enabling TTL...")
        dynamodb.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={
                'Enabled': True,
                'AttributeName': 'ttl_timestamp'
            }
        )

        print(f"✓ Table {table_name} created successfully!")
        print("\nTable details:")
        print(f"  - Primary key: insight_type (HASH) + entity_id (RANGE)")
        print(f"  - GSI: proposal_id-index")
        print(f"  - TTL: Enabled on ttl_timestamp attribute")

    except Exception as e:
        print(f"Error creating table: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create DynamoDB table for AI insights")
    parser.add_argument(
        "--env",
        type=str,
        default="prod",
        choices=["demo", "prod"],
        help="Environment (default: prod)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )

    args = parser.parse_args()

    create_insights_table(environment=args.env, region=args.region)
