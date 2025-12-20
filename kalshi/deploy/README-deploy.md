# Dora Bot Deployment Guide

## Prerequisites

- EC2 instance with Ubuntu 22.04+
- Python 3.10+
- AWS IAM role with DynamoDB and CloudWatch permissions
- CloudWatch Agent installed

## Quick Start

### 1. Install the bot

```bash
# Clone repo and set up virtual environment
cd /home/ubuntu
git clone <your-repo-url> dora
cd dora
python3 -m venv .venv
source .venv/bin/activate
pip install -r kalshi/requirements.txt
```

### 2. Configure credentials

Create `/etc/dora-bot.env`:

```bash
# Required
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY=<base64-encoded-private-key>

# Environment
USE_DEMO=true  # Set to 'false' for production
AWS_REGION=us-east-1

# Optional
BOT_VERSION=1.0.0
```

To base64 encode your private key:
```bash
base64 -w0 your-private-key.pem > key.b64
```

### 3. Install systemd service

```bash
sudo cp kalshi/deploy/dora-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dora-bot
sudo systemctl start dora-bot
```

### 4. Check status

```bash
# View service status
sudo systemctl status dora-bot

# View logs
sudo journalctl -u dora-bot -f

# View structured JSON logs
sudo journalctl -u dora-bot -o cat -f
```

## CloudWatch Setup

### 1. Install CloudWatch Agent

```bash
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
```

### 2. Configure agent

```bash
# Copy config
sudo cp kalshi/deploy/cloudwatch-agent-config.json /opt/aws/amazon-cloudwatch-agent/etc/

# Update {env} placeholder in config to 'demo' or 'prod'
sudo sed -i 's/{env}/demo/g' /opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-agent-config.json

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-agent-config.json \
    -s
```

### 3. Verify logs in CloudWatch

Check the log group `/dora-bot/demo/app` (or `/dora-bot/prod/app`) in CloudWatch Logs.

## CloudWatch Insights Queries

### Find errors in last hour
```
fields @timestamp, @message
| filter event_type = "ERROR"
| sort @timestamp desc
| limit 100
```

### Track order placements
```
fields @timestamp, market, side, price, size, status, latency_ms
| filter event_type = "ORDER_RESULT"
| sort @timestamp desc
| limit 100
```

### Find fills by market
```
fields @timestamp, market, side, price, size, pnl_delta, daily_pnl
| filter event_type = "FILL"
| filter market = "YOUR-MARKET-TICKER"
| sort @timestamp desc
```

### Correlate events by decision_id
```
fields @timestamp, event_type, market, @message
| filter decision_id = "YOUR-DECISION-ID"
| sort @timestamp asc
```

### View heartbeats
```
fields @timestamp, loop_count, markets_active, open_orders_count, daily_pnl
| filter event_type = "HEARTBEAT"
| sort @timestamp desc
| limit 50
```

## CloudWatch Alarms

Create metric filters and alarms in the CloudWatch console:

### Error Alarm
```
Metric filter pattern: { $.event_type = "ERROR" }
Alarm: Count > 0 in 5 minutes
Action: SNS notification
```

### Order Failure Alarm
```
Metric filter pattern: { $.event_type = "ORDER_RESULT" && $.status != "ACCEPTED" && $.status != "CANCELLED" && $.status != "ALREADY_GONE" }
Alarm: Count > 3 in 5 minutes
Action: SNS notification
```

### Risk Halt Alarm (Critical)
```
Metric filter pattern: { $.event_type = "RISK_HALT" }
Alarm: Count > 0 in 1 minute
Action: SNS notification (high priority)
```

## Troubleshooting

### Bot won't start
```bash
# Check logs for errors
sudo journalctl -u dora-bot -n 100 --no-pager

# Verify credentials
echo $KALSHI_KEY_ID
echo $KALSHI_PRIVATE_KEY | base64 -d | head -1
```

### Logs not appearing in CloudWatch
```bash
# Check CloudWatch agent status
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status

# Check agent logs
sudo tail -f /opt/aws/amazon-cloudwatch-agent/logs/amazon-cloudwatch-agent.log
```

### Manual restart
```bash
sudo systemctl restart dora-bot
```

## Log Event Reference

| Event Type | Description |
|------------|-------------|
| `STARTUP` | Bot initialization and startup sequence |
| `SHUTDOWN` | Bot shutdown (graceful or error) |
| `HEARTBEAT` | Periodic status (every 10 loops) |
| `DECISION_MADE` | Strategy produced target quotes |
| `ORDER_PLACE` | Order placement attempted |
| `ORDER_CANCEL` | Order cancellation attempted |
| `ORDER_RESULT` | API response for order operation |
| `FILL` | Trade fill processed |
| `RISK_HALT` | Trading halted due to risk limit |
| `STATE_LOAD` | State loaded from DynamoDB |
| `STATE_SAVE` | State saved to DynamoDB |
| `CONFIG_REFRESH` | Market configs reloaded |
| `ERROR` | Error occurred |
| `LOG` | Generic log message |
