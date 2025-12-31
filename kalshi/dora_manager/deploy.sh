#!/bin/bash
# Deployment script for Dora Manager Lambda

set -e

# Disable AWS CLI v2 pager for non-interactive execution
export AWS_PAGER=""

FUNCTION_NAME="dora-manager"
REGION="${AWS_REGION:-us-east-1}"
RUNTIME="python3.12"
HANDLER="handler.lambda_handler"
TIMEOUT=600
MEMORY=512

# Detect AWS CLI path (prefer venv version)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_CMD="${AWS_CMD:-aws}"
if ! AWS_VERSION="$($AWS_CMD --version 2>&1)"; then
    echo "ERROR: Unable to run AWS CLI. Install AWS CLI v2 and ensure it's on PATH, or set AWS_CMD to the v2 binary."
    exit 1
fi
if [[ "$AWS_VERSION" != aws-cli/2* ]]; then
    echo "ERROR: AWS CLI v2 is required. Detected: $AWS_VERSION"
    echo "Install v2 and ensure 'aws' points to it, or set AWS_CMD to the v2 binary."
    exit 1
fi

echo "Using AWS CLI: $AWS_CMD ($AWS_VERSION)"

# Create deployment package
echo "Creating deployment package..."
cd "$SCRIPT_DIR"

# Remove old package if it exists
if [ -f "lambda_package.zip" ]; then
    echo "Removing old lambda_package.zip..."
    rm lambda_package.zip
fi

# Create a temporary directory for the package
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy all Python files
cp *.py "$TEMP_DIR/"

# Copy CSV data file
cp restricted_markets.csv "$TEMP_DIR/"

# Copy utils directory
cp -r utils "$TEMP_DIR/"

# Install Python dependencies for Lambda runtime (Linux x86_64, Python 3.12)
echo "Installing Python dependencies for Lambda runtime..."
python3 -m pip install --target "$TEMP_DIR" \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade \
    requests openai pandas pydantic pydantic-core

echo "Verifying pydantic-core installation..."
# Check for the compiled binary (the exact filename may vary)
if ls "$TEMP_DIR/pydantic_core/_pydantic_core"*.so 1> /dev/null 2>&1; then
    echo "✓ pydantic_core binary found:"
    ls -lh "$TEMP_DIR/pydantic_core/_pydantic_core"*.so
else
    echo "ERROR: pydantic_core binary (.so file) not found!"
    echo "Contents of pydantic_core directory:"
    ls -la "$TEMP_DIR/pydantic_core/" || echo "pydantic_core directory not found!"
    exit 1
fi

# Copy package directory (if it exists, for local execution support)
if [ -d "package" ]; then
    cp -r package "$TEMP_DIR/"
fi

# Create zip file
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/lambda_package.zip" . -x "*.pyc" -x "__pycache__/*" -x "*.git/*"

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
