#!/bin/bash
# Deployment script for Dora Manager Lambda

set -e

# Disable AWS CLI v2 pager for non-interactive execution
export AWS_PAGER=""

FUNCTION_NAME="dora-manager"
REGION="${AWS_REGION:-us-east-1}"
RUNTIME="python3.12"
HANDLER="dora_manager.handler.lambda_handler"
TIMEOUT=60
MEMORY=256

# Detect AWS CLI path (prefer venv version)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_CMD="aws"

echo "Using AWS CLI: $AWS_CMD"

# Create deployment package
echo "Creating deployment package..."
cd "$SCRIPT_DIR"

# Create a temporary directory for the package with correct structure
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create dora_manager folder inside temp dir and copy code
mkdir -p "$TEMP_DIR/dora_manager"
cp *.py "$TEMP_DIR/dora_manager/"

# Create zip file with correct structure
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/lambda_package.zip" dora_manager -x "*.pyc" -x "__pycache__/*"

cd "$SCRIPT_DIR"
echo "Created lambda_package.zip"

# Check if function exists and update
if $AWS_CMD lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null; then
    echo "Updating existing Lambda function..."
    $AWS_CMD lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://lambda_package.zip" \
        --region "$REGION"
    echo "Lambda function updated successfully!"
else
    echo "Lambda function does not exist. Please create it first with:"
    echo ""
    echo "$AWS_CMD lambda create-function \\"
    echo "    --function-name $FUNCTION_NAME \\"
    echo "    --runtime $RUNTIME \\"
    echo "    --handler $HANDLER \\"
    echo "    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_LAMBDA_ROLE \\"
    echo "    --timeout $TIMEOUT \\"
    echo "    --memory-size $MEMORY \\"
    echo "    --zip-file fileb://lambda_package.zip \\"
    echo "    --environment 'Variables={ENVIRONMENT=prod}' \\"
    echo "    --region $REGION"
fi

echo "Done!"
