#!/usr/bin/env python3
"""
API Gateway setup script for Dora Manager.

Creates:
1. REST API with two endpoints:
   - GET /proposals/{proposal_id} - View proposal details
   - POST /proposals/{proposal_id}/execute - Execute proposals
2. Lambda integration for both endpoints
3. Outputs API Gateway URL

Usage:
    python setup_api_gateway.py --env prod --region us-east-1 --lambda-arn <arn>
    python setup_api_gateway.py --env demo --region us-east-1 --lambda-arn <arn> --dry-run
"""
import argparse
import json
import sys
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError


def create_rest_api(
    apigw_client,
    environment: str,
    dry_run: bool = False
) -> Optional[str]:
    """Create REST API.

    Args:
        apigw_client: boto3 API Gateway client
        environment: 'demo' or 'prod'
        dry_run: If True, only print what would be created

    Returns:
        API ID if successful, None otherwise
    """
    api_name = f"dora-proposals-api-{environment}"

    if dry_run:
        print(f"\n[DRY RUN] Would create REST API: {api_name}")
        return "dry-run-api-id"

    try:
        # Check if API already exists
        apis = apigw_client.get_rest_apis()
        for api in apis.get('items', []):
            if api['name'] == api_name:
                print(f"✓ REST API {api_name} already exists")
                print(f"  API ID: {api['id']}")
                return api['id']

        # Create REST API
        print(f"Creating REST API: {api_name}...")
        response = apigw_client.create_rest_api(
            name=api_name,
            description=f'Dora Manager Proposal Approval API ({environment})',
            endpointConfiguration={'types': ['REGIONAL']},
            tags={
                'Environment': environment,
                'Application': 'dora-manager',
                'ManagedBy': 'setup-script',
            }
        )

        api_id = response['id']
        print(f"✓ REST API created: {api_id}")
        return api_id

    except Exception as e:
        print(f"✗ Error creating REST API: {e}")
        return None


def get_root_resource_id(apigw_client, api_id: str) -> Optional[str]:
    """Get the root resource ID for the API.

    Args:
        apigw_client: boto3 API Gateway client
        api_id: API Gateway API ID

    Returns:
        Root resource ID if found
    """
    try:
        resources = apigw_client.get_resources(restApiId=api_id)
        for resource in resources.get('items', []):
            if resource['path'] == '/':
                return resource['id']
        return None
    except Exception as e:
        print(f"✗ Error getting root resource: {e}")
        return None


def create_api_resources(
    apigw_client,
    api_id: str,
    lambda_arn: str,
    region: str,
    dry_run: bool = False
) -> bool:
    """Create API Gateway resources and methods.

    Args:
        apigw_client: boto3 API Gateway client
        api_id: API Gateway API ID
        lambda_arn: Lambda function ARN
        region: AWS region
        dry_run: If True, only print what would be created

    Returns:
        True if successful
    """
    if dry_run:
        print(f"\n[DRY RUN] Would create resources:")
        print(f"  - /proposals")
        print(f"  - /proposals/{{proposal_id}}")
        print(f"  - GET /proposals/{{proposal_id}}")
        print(f"  - POST /proposals/{{proposal_id}}/execute")
        return True

    try:
        # Get root resource
        root_id = get_root_resource_id(apigw_client, api_id)
        if not root_id:
            print(f"✗ Could not find root resource")
            return False

        # Create /proposals resource
        print(f"Creating /proposals resource...")
        proposals_resource = apigw_client.create_resource(
            restApiId=api_id,
            parentId=root_id,
            pathPart='proposals'
        )
        proposals_id = proposals_resource['id']

        # Create /proposals/{proposal_id} resource
        print(f"Creating /proposals/{{proposal_id}} resource...")
        proposal_id_resource = apigw_client.create_resource(
            restApiId=api_id,
            parentId=proposals_id,
            pathPart='{proposal_id}'
        )
        proposal_id_resource_id = proposal_id_resource['id']

        # Create GET method for /proposals/{proposal_id}
        print(f"Creating GET method...")
        apigw_client.put_method(
            restApiId=api_id,
            resourceId=proposal_id_resource_id,
            httpMethod='GET',
            authorizationType='NONE',
            requestParameters={
                'method.request.path.proposal_id': True,
                'method.request.querystring.signature': True,
                'method.request.querystring.expiry': True,
            }
        )

        # Create Lambda integration for GET
        lambda_uri = f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
        apigw_client.put_integration(
            restApiId=api_id,
            resourceId=proposal_id_resource_id,
            httpMethod='GET',
            type='AWS_PROXY',
            integrationHttpMethod='POST',  # Lambda always uses POST
            uri=lambda_uri
        )

        # Create /proposals/{proposal_id}/execute resource
        print(f"Creating /proposals/{{proposal_id}}/execute resource...")
        execute_resource = apigw_client.create_resource(
            restApiId=api_id,
            parentId=proposal_id_resource_id,
            pathPart='execute'
        )
        execute_resource_id = execute_resource['id']

        # Create POST method for /proposals/{proposal_id}/execute
        print(f"Creating POST method...")
        apigw_client.put_method(
            restApiId=api_id,
            resourceId=execute_resource_id,
            httpMethod='POST',
            authorizationType='NONE',
            requestParameters={
                'method.request.path.proposal_id': True,
            }
        )

        # Create Lambda integration for POST
        apigw_client.put_integration(
            restApiId=api_id,
            resourceId=execute_resource_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=lambda_uri
        )

        print(f"✓ API resources and methods created")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'ConflictException':
            print(f"✓ Resources already exist")
            return True
        print(f"✗ Error creating API resources: {e}")
        return False
    except Exception as e:
        print(f"✗ Error creating API resources: {e}")
        return False


def deploy_api(
    apigw_client,
    api_id: str,
    environment: str,
    dry_run: bool = False
) -> Optional[str]:
    """Deploy API to a stage.

    Args:
        apigw_client: boto3 API Gateway client
        api_id: API Gateway API ID
        environment: 'demo' or 'prod'
        dry_run: If True, only print what would be deployed

    Returns:
        API URL if successful
    """
    stage_name = environment

    if dry_run:
        print(f"\n[DRY RUN] Would deploy API to stage: {stage_name}")
        return f"https://dry-run.execute-api.{apigw_client.meta.region_name}.amazonaws.com/{stage_name}"

    try:
        print(f"Deploying API to stage: {stage_name}...")

        # Create deployment
        deployment = apigw_client.create_deployment(
            restApiId=api_id,
            stageName=stage_name,
            description=f'Deployment for {environment} environment'
        )

        # Get API URL
        region = apigw_client.meta.region_name
        api_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/{stage_name}"

        print(f"✓ API deployed successfully")
        print(f"  Stage: {stage_name}")
        print(f"  URL: {api_url}")

        return api_url

    except Exception as e:
        print(f"✗ Error deploying API: {e}")
        return None


def add_lambda_permission(
    lambda_client,
    lambda_arn: str,
    api_id: str,
    region: str,
    environment: str,
    dry_run: bool = False
) -> bool:
    """Add permission for API Gateway to invoke Lambda.

    Args:
        lambda_client: boto3 Lambda client
        lambda_arn: Lambda function ARN
        api_id: API Gateway API ID
        region: AWS region
        environment: 'demo' or 'prod'
        dry_run: If True, only print what would be added

    Returns:
        True if successful
    """
    if dry_run:
        print(f"\n[DRY RUN] Would add Lambda permission for API Gateway")
        return True

    try:
        # Extract function name from ARN
        function_name = lambda_arn.split(':')[-1]

        # Extract account ID from ARN
        account_id = lambda_arn.split(':')[4]

        # Source ARN for API Gateway
        source_arn = f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*"

        print(f"Adding Lambda permission for API Gateway...")

        # Add permission
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'AllowAPIGatewayInvoke-{environment}',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=source_arn
        )

        print(f"✓ Lambda permission added")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceConflictException':
            print(f"✓ Lambda permission already exists")
            return True
        print(f"✗ Error adding Lambda permission: {e}")
        return False
    except Exception as e:
        print(f"✗ Error adding Lambda permission: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Set up API Gateway for Dora Manager'
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
        '--lambda-arn',
        required=True,
        help='Lambda function ARN to integrate with'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be created without actually creating'
    )

    args = parser.parse_args()

    print(f"="*60)
    print(f"Dora Manager API Gateway Setup")
    print(f"Environment: {args.env}")
    print(f"Region: {args.region}")
    print(f"Lambda ARN: {args.lambda_arn}")
    print(f"Dry Run: {args.dry_run}")
    print(f"="*60)

    # Initialize AWS clients
    apigw_client = boto3.client('apigateway', region_name=args.region)
    lambda_client = boto3.client('lambda', region_name=args.region)

    # Create REST API
    print(f"\n--- Creating REST API ---")
    api_id = create_rest_api(apigw_client, args.env, args.dry_run)
    if not api_id:
        print(f"✗ Failed to create REST API")
        sys.exit(1)

    # Create API resources and methods
    print(f"\n--- Creating API Resources ---")
    if not create_api_resources(apigw_client, api_id, args.lambda_arn, args.region, args.dry_run):
        print(f"✗ Failed to create API resources")
        sys.exit(1)

    # Deploy API
    print(f"\n--- Deploying API ---")
    api_url = deploy_api(apigw_client, api_id, args.env, args.dry_run)
    if not api_url:
        print(f"✗ Failed to deploy API")
        sys.exit(1)

    # Add Lambda permission
    print(f"\n--- Adding Lambda Permission ---")
    if not add_lambda_permission(lambda_client, args.lambda_arn, api_id, args.region, args.env, args.dry_run):
        print(f"✗ Failed to add Lambda permission")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ API Gateway setup completed successfully")
    print(f"\nAPI Endpoint URLs:")
    print(f"  - GET  {api_url}/proposals/{{proposal_id}}?signature=XXX&expiry=XXX")
    print(f"  - POST {api_url}/proposals/{{proposal_id}}/execute")
    print(f"\nSave this API URL for use in your Lambda function:")
    print(f"  API_GATEWAY_BASE_URL={api_url}")

    if args.dry_run:
        print(f"\n  This was a DRY RUN - no resources were created")
        print(f"  Run without --dry-run to create resources")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
