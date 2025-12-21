#!/bin/bash
# Deploy Dora Bot to EC2 instance
# Usage: ./deploy-to-ec2.sh <ec2-ip> <ssh-key-path> [--setup] [--with-legacy-secrets]
#
# Examples:
#   First time setup:  ./deploy-to-ec2.sh 3.84.220.98 ./DoraBot-RSA.pem --setup
#   Update code only:  ./deploy-to-ec2.sh 3.84.220.98 ./DoraBot-RSA.pem
#   Include .env/.pem: ./deploy-to-ec2.sh 3.84.220.98 ./DoraBot-RSA.pem --with-legacy-secrets

set -e

EC2_IP="${1}"
SSH_KEY="${2}"
SETUP_MODE=""
COPY_LEGACY_SECRETS="false"

for arg in "${@:3}"; do
    case "$arg" in
        --setup)
            SETUP_MODE="--setup"
            ;;
        --with-legacy-secrets)
            COPY_LEGACY_SECRETS="true"
            ;;
    esac
done

REMOTE_USER="ec2-user"
REMOTE_DIR="/home/${REMOTE_USER}/dora"

if [ -z "$EC2_IP" ] || [ -z "$SSH_KEY" ]; then
    echo "Usage: $0 <ec2-ip> <ssh-key-path> [--setup] [--with-legacy-secrets]"
    echo ""
    echo "Examples:"
    echo "  First time:  $0 3.84.220.98 ./DoraBot-RSA.pem --setup"
    echo "  Update:      $0 3.84.220.98 ./DoraBot-RSA.pem"
    echo "  Legacy:      $0 3.84.220.98 ./DoraBot-RSA.pem --with-legacy-secrets"
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
    --exclude '*:Zone.Identifier' \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude 'market_summaries.csv' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    "$SCRIPT_DIR/" "${REMOTE_USER}@${EC2_IP}:$REMOTE_DIR/kalshi/"

# ============================================
# COPY SECRETS (if they exist locally and not on remote)
# ============================================
if [ "$COPY_LEGACY_SECRETS" == "true" ]; then
    log "Checking legacy secrets..."

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
else
    log "Skipping legacy secrets sync (use --with-legacy-secrets to enable)"
fi

# ============================================
# INSTALL DEPENDENCIES & SETUP SERVICE
# ============================================
log "Installing dependencies and configuring service..."

$SSH_CMD << 'REMOTE_SCRIPT'
set -e

cd /home/ec2-user/dora

# Create/update virtual environment
echo "[REMOTE] Setting up Python virtual environment..."
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
else
    PYTHON_BIN="python3"
fi

$PYTHON_BIN -m venv venv

source venv/bin/activate

echo "[REMOTE] Installing Python dependencies..."
$PYTHON_BIN -m pip install --upgrade pip -q
$PYTHON_BIN -m pip install -r kalshi/dora_bot/requirements.txt -q

# Install systemd service
echo "[REMOTE] Configuring systemd service..."
sudo cp /home/ec2-user/dora/kalshi/ec2/dora-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dora-bot

echo "[REMOTE] Preparing log directory..."
sudo mkdir -p /home/ec2-user/dora/kalshi/logs
sudo touch /home/ec2-user/dora/kalshi/logs/dora-bot.log
sudo chown -R ec2-user:ec2-user /home/ec2-user/dora/kalshi/logs

echo "[REMOTE] Setting up CloudWatch agent..."
# Install CloudWatch agent if not present
if ! command -v amazon-cloudwatch-agent-ctl &> /dev/null; then
    echo "[REMOTE] Installing CloudWatch agent..."
    if command -v dnf &> /dev/null; then
        sudo dnf install -y amazon-cloudwatch-agent
    elif command -v apt &> /dev/null; then
        wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
        sudo dpkg -i amazon-cloudwatch-agent.deb
        rm -f amazon-cloudwatch-agent.deb
    fi
fi

# Copy and configure CloudWatch agent config
sudo cp /home/ec2-user/dora/kalshi/ec2/cloudwatch-agent-config.json /opt/aws/amazon-cloudwatch-agent/etc/
# Replace {env} with demo (change to 'prod' for production)
sudo sed -i 's/{env}/demo/g' /opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-agent-config.json

# CRITICAL: Set permissions so cwagent user can read log files
# The CloudWatch agent runs as 'cwagent' user and needs read access to the entire path
chmod 755 /home/ec2-user
chmod 755 /home/ec2-user/dora
chmod 755 /home/ec2-user/dora/kalshi
chmod 755 /home/ec2-user/dora/kalshi/logs
chmod 644 /home/ec2-user/dora/kalshi/logs/dora-bot.log

# Load config and start agent
echo "[REMOTE] Starting CloudWatch agent..."
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-agent-config.json \
  -s || echo "[REMOTE] Warning: CloudWatch agent start failed (check IAM role has CloudWatchAgentServerPolicy)"

echo "[REMOTE] Setup complete"
REMOTE_SCRIPT

# ============================================
# RESTART SERVICE (skip on first-time setup)
# ============================================
if [ "$SETUP_MODE" == "--setup" ]; then
    warn "First-time setup: skipping service restart. Add /etc/dora-bot.env then start with 'sudo systemctl start dora-bot'."
else
    log "Restarting dora-bot service..."
    $SSH_CMD "sudo systemctl restart dora-bot" || warn "Service restart failed - may need secrets configured"

    # Brief pause then check status
    sleep 2
    log "Service status:"
    $SSH_CMD "sudo systemctl status dora-bot --no-pager -l" || true
fi

echo ""
log "Deployment complete!"
echo ""
echo "Useful commands:"
echo "  View logs:     $SSH_CMD 'tail -f /home/ec2-user/dora/kalshi/logs/dora-bot.log'"
echo "  Stop bot:      $SSH_CMD 'sudo systemctl stop dora-bot'"
echo "  Start bot:     $SSH_CMD 'sudo systemctl start dora-bot'"
echo "  SSH in:        $SSH_CMD"
