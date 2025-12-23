# Dora Bot - Kalshi Market Making Bot

A market making trading bot for Kalshi prediction markets with DynamoDB state persistence.

## Architecture

See [plan.md](plan.md) for detailed architecture documentation.

### Core Components

- **exchange_client.py** - Kalshi API wrapper with retry logic
- **state_manager.py** - In-memory state with DynamoDB persistence
- **risk_manager.py** - Risk checks and limits enforcement
- **strategy.py** - Market making quote logic
- **dynamo.py** - DynamoDB helper functions
- **models.py** - Data classes
- **main.py** - Main event loop

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Ensure AWS credentials are configured (for DynamoDB access):
```bash
aws configure
```

### 3. Create DynamoDB Tables

```bash
python setup_dynamodb.py us-east-1
```

This creates:
- `dora_market_config` - Per-market configuration
- `dora_state` - Bot state (positions, risk, global config)
- `dora_trade_log` - Trade audit log
- `dora_decision_log` - Decision log (optional)
- `dora_execution_log` - Execution log (Phase 3)

To create for a single environment:
```bash
python setup_dynamodb.py us-east-1 --env demo
```

### 4. Configure Kalshi API Credentials

Ensure your `.env` file (in parent kalshi directory) has:
```
DEMO_KEYID=your_demo_key_id
DEMO_KEYFILE=path/to/demo_private_key.pem
PROD_KEYID=your_prod_key_id
PROD_KEYFILE=path/to/prod_private_key.pem
```

### 5. Add Market Configurations

Use DynamoDB console or boto3 to add market configs to `dora_market_config` table:

```python
from setup_dynamodb import create_sample_market_config
create_sample_market_config("MARKET-TICKER")
```

Then enable the market by setting `enabled: true` in DynamoDB.

### 6. Configure Global Settings

Update the `global_config` item in `dora_state` table:
- Set `trading_enabled: true` when ready to trade
- Adjust `max_total_exposure`, `max_daily_loss`, etc.

## Running the Bot

### Demo Mode (Default)
```bash
python main.py
```

### Production Mode
```bash
python main.py --prod
```

## Safety Features

1. **Cancel on Startup** - Cancels all existing orders on startup (configurable)
2. **Daily Loss Limit** - Halts trading if daily loss exceeds threshold
3. **Position Limits** - Per-market inventory caps
4. **Global Exposure Limit** - Total exposure cap across all markets
5. **Trading Halt** - Manual halt via DynamoDB config

## Monitoring

Logs are written to:
- Console (stdout)
- `dora_bot.log` file

Monitor key metrics:
- Daily PnL
- Position sizes
- Order fill rates
- Loop latency

For production, configure CloudWatch:
- Stream logs to CloudWatch Logs
- Create alarms on PnL thresholds
- Monitor loop latency

## Configuration Updates

All configuration is in DynamoDB and can be updated without restarting:
- Market configs refreshed every 10 loops (configurable)
- Global config loaded on startup

To update:
1. Edit values in DynamoDB console
2. Bot will pick up changes within ~5 seconds

## EC2 Deployment

### systemd Service

Create `/etc/systemd/system/dora-bot.service`:

```ini
[Unit]
Description=Dora Kalshi Market Making Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/dora/kalshi/dora_bot
Environment=PATH=/home/ubuntu/venv/bin
ExecStart=/home/ubuntu/venv/bin/python main.py --prod
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable dora-bot
sudo systemctl start dora-bot
sudo systemctl status dora-bot
```

View logs:
```bash
sudo journalctl -u dora-bot -f
```

## Risk Management

**IMPORTANT**: Start with small limits and demo mode first!

1. Test in demo environment extensively
2. Start with low inventory limits (e.g., 10 contracts)
3. Set conservative daily loss limits
4. Monitor closely for first few days
5. Gradually increase limits as confidence builds

## Troubleshooting

### Bot won't start
- Check AWS credentials are configured
- Verify DynamoDB tables exist
- Check Kalshi API credentials in .env

### Orders not being placed
- Check `trading_enabled: true` in global config
- Verify market `enabled: true` in market config
- Check spread is wide enough (>= min_spread)
- Review risk limits (inventory caps, exposure limits)

### High loop latency
- Reduce number of active markets
- Increase loop_interval_ms
- Check network latency to Kalshi API

## Development

Run tests (TODO):
```bash
pytest tests/
```

Code structure follows the plan in [plan.md](plan.md).
