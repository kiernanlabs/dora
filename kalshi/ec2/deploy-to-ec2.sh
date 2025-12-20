#!/bin/bash
# Deploy Dora Bot to EC2 instance
# Usage: ./deploy-to-ec2.sh <ec2-ip> <ssh-key-path> [--setup]
#
# Examples:
#   First time setup:  ./deploy-to-ec2.sh 3.84.220.98 ./DoraBot-RSA.pem --setup
#   Update code only:  ./deploy-to-ec2.sh 3.84.220.98 ./DoraBot-RSA.pem

set -e

EC2_IP="${1}"
SSH_KEY="${2}"
SETUP_MODE="${3}"

REMOTE_USER="ec2-user"
REMOTE_DIR="/home/${REMOTE_USER}/dora"

if [ -z "$EC2_IP" ] || [ -z "$SSH_KEY" ]; then
    echo "Usage: $0 <ec2-ip> <ssh-key-path> [--setup]"
    echo ""
    echo "Examples:"
    echo "  First time:  $0 3.84.220.98 ./DoraBot-RSA.pem --setup"
    echo "  Update:      $0 3.84.220.98 ./DoraBot-RSA.pem"
    exit 1
fi

# Ensure key has correct permissions
chmod 600 "$SSH_KEY" 2>/dev/null || true

SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${EC2_IP}"
SCP_CMD="scp -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Get the kalshi directory (parent of ec2/)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log "Deploying from $SCRIPT_DIR to ${REMOTE_USER}@${EC2_IP}..."

# ============================================
# FIRST TIME SETUP (if --setup flag is passed)
# ============================================
if [ "$SETUP_MODE" == "--setup" ]; then
    log "Running first-time setup..."

    $SSH_CMD << 'SETUP_SCRIPT'
set -e
echo "[REMOTE] Updating system packages..."
if command -v dnf &> /dev/null; then
    sudo dnf update -y
    sudo dnf install -y python3.11 python3.11-pip git rsync
elif command -v apt &> /dev/null; then
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y python3.11 python3.11-venv python3-pip git rsync
fi

echo "[REMOTE] Creating directory structure..."
mkdir -p ~/dora/kalshi

echo "[REMOTE] First-time setup complete"
SETUP_SCRIPT

fi

# ============================================
# SYNC CODE
# ============================================
log "Syncing code to EC2..."

# Create remote directory structure
$SSH_CMD "mkdir -p $REMOTE_DIR/kalshi"

# Sync code (excluding secrets and unnecessary files)
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
    --exclude 'market_summaries.csv' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    "$SCRIPT_DIR/" "${REMOTE_USER}@${EC2_IP}:$REMOTE_DIR/kalshi/"

# ============================================
# COPY SECRETS (if they exist locally and not on remote)
# ============================================
log "Checking secrets..."

# Copy .env if it exists locally and doesn't exist remotely
if [ -f "$SCRIPT_DIR/.env" ]; then
    REMOTE_ENV_EXISTS=$($SSH_CMD "[ -f $REMOTE_DIR/kalshi/.env ] && echo 'yes' || echo 'no'")
    if [ "$REMOTE_ENV_EXISTS" == "no" ]; then
        log "Copying .env file..."
        $SCP_CMD "$SCRIPT_DIR/.env" "${REMOTE_USER}@${EC2_IP}:$REMOTE_DIR/kalshi/.env"
    else
        warn ".env already exists on remote, skipping (delete remote file to update)"
    fi
fi

# Copy private key files if they exist locally
for keyfile in "$SCRIPT_DIR"/private_key*.pem; do
    if [ -f "$keyfile" ]; then
        keyname=$(basename "$keyfile")
        REMOTE_KEY_EXISTS=$($SSH_CMD "[ -f $REMOTE_DIR/kalshi/$keyname ] && echo 'yes' || echo 'no'")
        if [ "$REMOTE_KEY_EXISTS" == "no" ]; then
            log "Copying $keyname..."
            $SCP_CMD "$keyfile" "${REMOTE_USER}@${EC2_IP}:$REMOTE_DIR/kalshi/$keyname"
            $SSH_CMD "chmod 600 $REMOTE_DIR/kalshi/$keyname"
        else
            warn "$keyname already exists on remote, skipping"
        fi
    fi
done

# ============================================
# INSTALL DEPENDENCIES & SETUP SERVICE
# ============================================
log "Installing dependencies and configuring service..."

$SSH_CMD << REMOTE_SCRIPT
set -e

cd $REMOTE_DIR

# Create/update virtual environment
echo "[REMOTE] Setting up Python virtual environment..."
python3 -m venv venv 2>/dev/null || python3.11 -m venv venv

source venv/bin/activate

echo "[REMOTE] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r kalshi/dora_bot/requirements.txt -q
pip install -r kalshi/kalshi-starter-code-python-main/requirements.txt -q

# Install systemd service
echo "[REMOTE] Configuring systemd service..."
sudo cp $REMOTE_DIR/kalshi/ec2/dora-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dora-bot

echo "[REMOTE] Setup complete"
REMOTE_SCRIPT

# ============================================
# RESTART SERVICE
# ============================================
log "Restarting dora-bot service..."
$SSH_CMD "sudo systemctl restart dora-bot" || warn "Service restart failed - may need secrets configured"

# Brief pause then check status
sleep 2
log "Service status:"
$SSH_CMD "sudo systemctl status dora-bot --no-pager -l" || true

echo ""
log "Deployment complete!"
echo ""
echo "Useful commands:"
echo "  View logs:     $SSH_CMD 'journalctl -u dora-bot -f'"
echo "  Stop bot:      $SSH_CMD 'sudo systemctl stop dora-bot'"
echo "  Start bot:     $SSH_CMD 'sudo systemctl start dora-bot'"
echo "  SSH in:        $SSH_CMD"
