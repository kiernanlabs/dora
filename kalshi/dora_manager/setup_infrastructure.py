#!/usr/bin/env python3
"""
Infrastructure setup script for Dora Manager.

Creates:
1. DynamoDB table for market proposals
2. Secrets Manager secret for URL signing
3. IAM policy updates (outputs JSON for manual application)

Usage:
    python setup_infrastructure.py --env prod --region us-east-1
    python setup_infrastructure.py --env demo --region us-east-1 --dry-run
"""
import argparse
import json
import secrets
import sys
from typing import Dict, Any

import boto3
from botocore.exceptions import ClientError


def create_proposals_table(
    dynamodb_client,
    environment: str,
    dry_run: bool = False
) -> bool:
    """Create DynamoDB table for market proposals.

    Args:
        dynamodb_client: boto3 DynamoDB client
        environment: 'demo' or 'prod'
        dry_run: If True, only print what would be created

    Returns:
        True if successful
    """
    table_name = f"dora_market_proposals_{environment}"

    table_definition = {
        'TableName': table_name,
        'KeySchema': [
            {'AttributeName': 'proposal_id', 'KeyType': 'HASH'},  # Partition key
            {'AttributeName': 'market_id', 'KeyType': 'RANGE'},   # Sort key
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'proposal_id', 'AttributeType': 'S'},
            {'AttributeName': 'market_id', 'AttributeType': 'S'},
            {'AttributeName': 'status', 'AttributeType': 'S'},
            {'AttributeName': 'created_at', 'AttributeType': 'S'},
        ],
        'GlobalSecondaryIndexes': [
            {
                'IndexName': 'status-created_at-index',
                'KeySchema': [
                    {'AttributeName': 'status', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            },
        ],
        'BillingMode': 'PAY_PER_REQUEST',
        'Tags': [
            {'Key': 'Environment', 'Value': environment},
            {'Key': 'Application', 'Value': 'dora-manager'},
            {'Key': 'ManagedBy', 'Value': 'setup-script'},
        ],
    }

    if dry_run:
        print(f"\n[DRY RUN] Would create table: {table_name}")
        print(json.dumps(table_definition, indent=2))
        return True

    try:
        # Check if table already exists
        try:
            response = dynamodb_client.describe_table(TableName=table_name)
            print(f"✓ Table {table_name} already exists")
            print(f"  Status: {response['Table']['TableStatus']}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                raise

        # Create table
        print(f"Creating table: {table_name}...")
        response = dynamodb_client.create_table(**table_definition)

        print(f"✓ Table {table_name} created successfully")
        print(f"  ARN: {response['TableDescription']['TableArn']}")

        # Wait for table to be active
        print(f"  Waiting for table to be active...")
        waiter = dynamodb_client.get_waiter('table_exists')
        waiter.wait(TableName=table_name)

        # Enable TTL
        print(f"  Enabling TTL on ttl_timestamp attribute...")
        dynamodb_client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={
                'Enabled': True,
                'AttributeName': 'ttl_timestamp'
            }
        )

        # Enable Point-in-Time Recovery
        print(f"  Enabling Point-in-Time Recovery...")
        dynamodb_client.update_continuous_backups(
            TableName=table_name,
            PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True}
        )

        print(f"✓ Table {table_name} fully configured")
        return True

    except Exception as e:
        print(f"✗ Error creating table {table_name}: {e}")
        return False


def create_url_signing_secret(
    secrets_client,
    environment: str,
    dry_run: bool = False
) -> bool:
    """Create Secrets Manager secret for URL signing.

    Args:
        secrets_client: boto3 Secrets Manager client
        environment: 'demo' or 'prod'
        dry_run: If True, only print what would be created

    Returns:
        True if successful
    """
    secret_name = f"dora-manager/url-signing-key/{environment}"

    if dry_run:
        print(f"\n[DRY RUN] Would create secret: {secret_name}")
        print(f"  Secret value: <randomly generated 64-character hex string>")
        return True

    try:
        # Check if secret already exists
        try:
            secrets_client.describe_secret(SecretId=secret_name)
            print(f"✓ Secret {secret_name} already exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                raise

        # Generate random secret key
        secret_value = secrets.token_hex(32)  # 64-character hex string

        # Create secret
        print(f"Creating secret: {secret_name}...")
        response = secrets_client.create_secret(
            Name=secret_name,
            Description=f'HMAC secret key for signing proposal approval URLs ({environment})',
            SecretString=secret_value,
            Tags=[
                {'Key': 'Environment', 'Value': environment},
                {'Key': 'Application', 'Value': 'dora-manager'},
                {'Key': 'ManagedBy', 'Value': 'setup-script'},
            ]
        )

        print(f"✓ Secret {secret_name} created successfully")
        print(f"  ARN: {response['ARN']}")
        return True

    except Exception as e:
        print(f"✗ Error creating secret {secret_name}: {e}")
        return False


def generate_iam_policy(environment: str, region: str) -> Dict[str, Any]:
    """Generate IAM policy JSON for Lambda function.

    Args:
        environment: 'demo' or 'prod'
        region: AWS region

    Returns:
        IAM policy document
    """
    account_id = boto3.client('sts').get_caller_identity()['Account']

    table_arn = f"arn:aws:dynamodb:{region}:{account_id}:table/dora_market_proposals_{environment}"
    secret_arn = f"arn:aws:secretsmanager:{region}:{account_id}:secret:dora-manager/url-signing-key/{environment}-*"

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ProposalsTableAccess",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchGetItem",
                    "dynamodb:BatchWriteItem"
                ],
                "Resource": [
                    table_arn,
                    f"{table_arn}/index/*"
                ]
            },
            {
                "Sid": "URLSigningSecretAccess",
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue"
                ],
                "Resource": secret_arn
            }
        ]
    }

    return policy


def main():
    parser = argparse.ArgumentParser(
        description='Set up infrastructure for Dora Manager'
    )
    parser.add_argument(
        '--env',
        choices=['demo', 'prod'],
        required=True,
        help='Environment to set up'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be created without actually creating'
    )
    parser.add_argument(
        '--skip-table',
        action='store_true',
        help='Skip DynamoDB table creation'
    )
    parser.add_argument(
        '--skip-secret',
        action='store_true',
        help='Skip Secrets Manager secret creation'
    )

    args = parser.parse_args()

    print(f"="*60)
    print(f"Dora Manager Infrastructure Setup")
    print(f"Environment: {args.env}")
    print(f"Region: {args.region}")
    print(f"Dry Run: {args.dry_run}")
    print(f"="*60)

    # Initialize AWS clients
    dynamodb_client = boto3.client('dynamodb', region_name=args.region)
    secrets_client = boto3.client('secretsmanager', region_name=args.region)

    success = True

    # Create DynamoDB table
    if not args.skip_table:
        print(f"\n--- Creating DynamoDB Table ---")
        if not create_proposals_table(dynamodb_client, args.env, args.dry_run):
            success = False

    # Create Secrets Manager secret
    if not args.skip_secret:
        print(f"\n--- Creating Secrets Manager Secret ---")
        if not create_url_signing_secret(secrets_client, args.env, args.dry_run):
            success = False

    # Generate IAM policy
    print(f"\n--- IAM Policy for Lambda Function ---")
    policy = generate_iam_policy(args.env, args.region)
    policy_json = json.dumps(policy, indent=2)

    print(f"\nAdd this policy to your Lambda function's IAM role:")
    print(f"\n{policy_json}")

    # Save policy to file
    policy_file = f"iam_policy_{args.env}.json"
    with open(policy_file, 'w') as f:
        f.write(policy_json)
    print(f"\n✓ IAM policy saved to: {policy_file}")

    # Summary
    print(f"\n{'='*60}")
    if success:
        print(f"✓ Infrastructure setup completed successfully")
        if args.dry_run:
            print(f"\n  This was a DRY RUN - no resources were created")
            print(f"  Run without --dry-run to create resources")
    else:
        print(f"✗ Infrastructure setup completed with errors")
        sys.exit(1)
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
