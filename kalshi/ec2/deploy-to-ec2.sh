#!/bin/bash
# Deploy Dora Bot to EC2 instance
# Usage: ./deploy-to-ec2.sh <ec2-host> [ssh-key-path]

set -e

EC2_HOST="${1}"
SSH_KEY="${2:---}"  # Use default SSH key if not specified
REMOTE_USER="ec2-user"
REMOTE_DIR="/home/${REMOTE_USER}/dora"

if [ -z "$EC2_HOST" ]; then
    echo "Usage: $0 <ec2-host> [ssh-key-path]"
    echo "Example: $0 ec2-user@1.2.3.4 ~/.ssh/my-key.pem"
    exit 1
fi

# Build SSH command
if [ "$SSH_KEY" != "--" ]; then
    SSH_CMD="ssh -i $SSH_KEY"
    SCP_CMD="scp -i $SSH_KEY"
else
    SSH_CMD="ssh"
    SCP_CMD="scp"
fi

GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

# Get the script's directory (kalshi/)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log "Deploying from $SCRIPT_DIR to $EC2_HOST..."

# Create remote directory
log "Creating remote directory structure..."
$SSH_CMD $EC2_HOST "mkdir -p $REMOTE_DIR/kalshi"

# Sync code (excluding secrets and unnecessary files)
log "Syncing code..."
rsync -avz --progress \
    --exclude '.env' \
    --exclude '*.pem' \
    --exclude '*.key' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '*.log' \
    --exclude 'venv' \
    --exclude '.venv' \
    -e "$SSH_CMD" \
    "$SCRIPT_DIR/" "$EC2_HOST:$REMOTE_DIR/kalshi/"

# Sync .env.example if .env doesn't exist
log "Checking .env file..."
$SSH_CMD $EC2_HOST "[ -f $REMOTE_DIR/kalshi/.env ] || cp $REMOTE_DIR/kalshi/.env.example $REMOTE_DIR/kalshi/.env 2>/dev/null || true"

# Install/update dependencies
log "Installing dependencies..."
$SSH_CMD $EC2_HOST "cd $REMOTE_DIR && \
    python3 -m venv venv 2>/dev/null || python3.11 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r kalshi/dora_bot/requirements.txt && \
    pip install -r kalshi/kalshi-starter-code-python-main/requirements.txt"

# Update systemd service
log "Updating systemd service..."
$SSH_CMD $EC2_HOST "sudo cp $REMOTE_DIR/kalshi/ec2/dora-bot.service /etc/systemd/system/ && \
    sudo systemctl daemon-reload"

# Restart service
log "Restarting dora-bot service..."
$SSH_CMD $EC2_HOST "sudo systemctl restart dora-bot"

# Check status
log "Checking service status..."
$SSH_CMD $EC2_HOST "sudo systemctl status dora-bot --no-pager" || true

log "Deployment complete!"
echo ""
echo "Useful commands:"
echo "  View logs:     $SSH_CMD $EC2_HOST 'journalctl -u dora-bot -f'"
echo "  Stop bot:      $SSH_CMD $EC2_HOST 'sudo systemctl stop dora-bot'"
echo "  Start bot:     $SSH_CMD $EC2_HOST 'sudo systemctl start dora-bot'"
echo "  Check status:  $SSH_CMD $EC2_HOST 'sudo systemctl status dora-bot'"
