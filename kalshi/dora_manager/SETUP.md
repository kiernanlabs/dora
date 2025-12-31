# Dora Manager Setup Guide

This guide walks through setting up the infrastructure for the Dora Manager Lambda function migration.

## Prerequisites

1. AWS CLI v2 configured with appropriate credentials (`aws --version` should show `aws-cli/2`)
2. Python 3.12+ installed
3. boto3 installed (`pip install boto3`)
4. Existing Lambda function deployed

## Phase 1: Infrastructure Setup

### Step 1: Create DynamoDB Table and Secrets

```bash
# Dry run first to see what will be created
python setup_infrastructure.py --env prod --region us-east-1 --dry-run

# Actually create resources
python setup_infrastructure.py --env prod --region us-east-1
```

This creates:
- DynamoDB table: `dora_market_proposals_prod`
- Secrets Manager secret: `dora-manager/url-signing-key/prod`
- IAM policy JSON file: `iam_policy_prod.json`

### Step 2: Update Lambda IAM Role

Apply the IAM policy generated in Step 1:

```bash
# Get your Lambda function's role name
aws lambda get-function --function-name dora-manager-prod \
  --query 'Configuration.Role' --output text

# Create inline policy (replace ROLE_NAME)
aws iam put-role-policy \
  --role-name ROLE_NAME \
  --policy-name DoraManagerProposalsAccess \
  --policy-document file://iam_policy_prod.json
```

### Step 3: Create API Gateway

```bash
# Get your Lambda function ARN
LAMBDA_ARN=$(aws lambda get-function --function-name dora-manager-prod \
  --query 'Configuration.FunctionArn' --output text)

# Dry run first
python setup_api_gateway.py --env prod --region us-east-1 \
  --lambda-arn $LAMBDA_ARN --dry-run

# Actually create resources
python setup_api_gateway.py --env prod --region us-east-1 \
  --lambda-arn $LAMBDA_ARN
```

This creates:
- REST API: `dora-proposals-api-prod`
- Endpoints:
  - `GET /proposals/{proposal_id}`
  - `POST /proposals/{proposal_id}/execute`
- Lambda integration and permissions

**Save the API Gateway URL** output at the end - you'll need it for the Lambda environment variables.

API Endpoint URLs:
  - GET  https://4w7qvk8sqg.execute-api.us-east-1.amazonaws.com/prod/proposals/{proposal_id}?signature=XXX&expiry=XXX
  - POST https://4w7qvk8sqg.execute-api.us-east-1.amazonaws.com/prod/proposals/{proposal_id}/execute

### Step 4: Update Lambda Environment Variables

Add these environment variables to your Lambda function:

```bash
# Set the API Gateway base URL (from Step 3 output)
aws lambda update-function-configuration \
  --function-name dora-manager-prod \
  --environment Variables="{
    ENVIRONMENT=prod,
    AWS_REGION=us-east-1,
    API_GATEWAY_BASE_URL=https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod,
    RECIPIENT_EMAIL=joe@kiernanlabs.com
  }"
```

## Phase 2: Deploy Code

After infrastructure is set up, deploy the updated Lambda code:

```bash
# From the dora_manager directory
cd /home/joe/dora/kalshi/dora_manager

# Install dependencies
pip install -r requirements.txt -t package/
cp -r *.py package/
cp -r utils/ package/
cp -r templates/ package/

# Create deployment package
cd package
zip -r ../dora-manager-deployment.zip .
cd ..

# Upload to Lambda
aws lambda update-function-code \
  --function-name dora-manager-prod \
  --zip-file fileb://dora-manager-deployment.zip
```

## Phase 3: Create EventBridge Rules

### Rule 1: Report Mode (Every 3 hours)

```bash
# Create rule
aws events put-rule \
  --name dora-report-3hr-prod \
  --schedule-expression "rate(3 hours)" \
  --description "Trigger Dora Manager report mode every 3 hours" \
  --state ENABLED

# Add Lambda target
aws events put-targets \
  --rule dora-report-3hr-prod \
  --targets "Id"="1","Arn"="$LAMBDA_ARN","Input"='{"mode":"report","environment":"prod","window_hours":3,"min_pnl_threshold":-3.0}'

# Add Lambda permission
aws lambda add-permission \
  --function-name dora-manager-prod \
  --statement-id AllowEventBridgeReport \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:ACCOUNT_ID:rule/dora-report-3hr-prod
```

### Rule 2: Market Management (Every 12 hours)

```bash
# Create rule
aws events put-rule \
  --name dora-market-management-12hr-prod \
  --schedule-expression "rate(12 hours)" \
  --description "Trigger Dora Manager market management mode every 12 hours" \
  --state ENABLED

# Add Lambda target
aws events put-targets \
  --rule dora-market-management-12hr-prod \
  --targets "Id"="1","Arn"="$LAMBDA_ARN","Input"='{"mode":"market_management","environment":"prod","pnl_lookback_hours":24,"volume_lookback_hours":48,"top_n_candidates":20,"skip_info_risk":false}'

# Add Lambda permission
aws lambda add-permission \
  --function-name dora-manager-prod \
  --statement-id AllowEventBridgeMarketManagement \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:ACCOUNT_ID:rule/dora-market-management-12hr-prod
```

## Testing

### Test Report Mode

```bash
aws lambda invoke \
  --function-name dora-manager-prod \
  --payload '{"mode":"report","environment":"prod","window_hours":3}' \
  response.json

cat response.json | jq .
```

### Test Market Management Mode

```bash
aws lambda invoke \
  --function-name dora-manager-prod \
  --payload '{"mode":"market_management","environment":"prod","pnl_lookback_hours":24,"volume_lookback_hours":48,"top_n_candidates":20}' \
  response.json

cat response.json | jq .
```

### Test Approval Endpoint

1. Run market_management mode (creates proposals and sends email)
2. Copy the proposal_id from the email
3. Test GET endpoint:

```bash
curl -i "https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/proposals/PROPOSAL_ID?signature=XXX&expiry=XXX"
```

4. Test POST endpoint:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"selected_proposals":["MARKET_ID_1","MARKET_ID_2"],"approve_all":false}' \
  "https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/proposals/PROPOSAL_ID/execute?signature=XXX&expiry=XXX"
```

## Rollback

If you need to rollback:

1. Disable EventBridge rules:
```bash
aws events disable-rule --name dora-market-management-12hr-prod
```

2. Fall back to manual scripts:
```bash
python /home/joe/dora/kalshi/dora_bot/market_screener.py --env prod
python /home/joe/dora/kalshi/dora_bot/market_update.py --env prod
```

## Cleanup (Demo Environment)

To clean up demo environment resources:

```bash
# Delete EventBridge rules
aws events remove-targets --rule dora-report-3hr-demo --ids 1
aws events remove-targets --rule dora-market-management-12hr-demo --ids 1
aws events delete-rule --name dora-report-3hr-demo
aws events delete-rule --name dora-market-management-12hr-demo

# Delete API Gateway
API_ID=$(aws apigateway get-rest-apis --query "items[?name=='dora-proposals-api-demo'].id" --output text)
aws apigateway delete-rest-api --rest-api-id $API_ID

# Delete DynamoDB table
aws dynamodb delete-table --table-name dora_market_proposals_demo

# Delete Secrets Manager secret
aws secretsmanager delete-secret --secret-id dora-manager/url-signing-key/demo --force-delete-without-recovery
```

## Monitoring

After setup, monitor:

1. CloudWatch Logs for Lambda function
2. DynamoDB table metrics (read/write capacity)
3. API Gateway metrics (request count, 4xx/5xx errors)
4. Email delivery (SES sending statistics)

Set up CloudWatch alarms for:
- Lambda errors > 0
- DynamoDB throttling
- API Gateway 5xx errors
