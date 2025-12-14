#!/bin/bash
# EC2 setup script for Dora Bot
# Run this on a fresh Amazon Linux 2023 or Ubuntu EC2 instance

set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

# Update system
log "Updating system packages..."
if command -v dnf &> /dev/null; then
    sudo dnf update -y
    sudo dnf install -y python3.11 python3.11-pip git
elif command -v apt &> /dev/null; then
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y python3.11 python3.11-venv python3-pip git
fi

# Create directory structure
log "Setting up directory structure..."
mkdir -p ~/dora
cd ~/dora

# Clone or copy your code (adjust as needed)
# git clone https://github.com/yourusername/dora.git .
log "Please copy your code to ~/dora/kalshi/"

# Create virtual environment
log "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
log "Installing Python dependencies..."
pip install --upgrade pip
pip install -r kalshi/dora_bot/requirements.txt
pip install -r kalshi/kalshi-starter-code-python-main/requirements.txt

# Setup .env file
log "Setting up environment file..."
if [ ! -f ~/dora/kalshi/.env ]; then
    cp ~/dora/kalshi/.env.example ~/dora/kalshi/.env
    echo "Please edit ~/dora/kalshi/.env with your credentials"
fi

# Copy private key (you'll need to do this manually or via SSM)
log "Remember to copy your private key to ~/dora/kalshi/private_key.pem"

# Install systemd service
log "Installing systemd service..."
sudo cp ~/dora/kalshi/ec2/dora-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dora-bot

log "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy your code to ~/dora/kalshi/"
echo "  2. Edit ~/dora/kalshi/.env with your API credentials"
echo "  3. Copy your private key to ~/dora/kalshi/private_key.pem"
echo "  4. Start the bot: sudo systemctl start dora-bot"
echo "  5. Check status: sudo systemctl status dora-bot"
echo "  6. View logs: journalctl -u dora-bot -f"
