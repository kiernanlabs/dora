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

## Order Execution Flow

This section documents the complete path from order decision to exchange placement, highlighting critical state synchronization points where mismatches can occur.

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN LOOP (main.py:260)                           │
│                                                                             │
│  1. Fetch market configs from DynamoDB                                      │
│  2. For each enabled market → process_market()                              │
│  3. Process fills                                                           │
│  4. Periodic state persistence                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROCESS MARKET (main.py:390)                            │
│                                                                             │
│  1. Fetch order book from exchange                                          │
│  2. Get current position from StateManager                                  │
│  3. Compute target quotes via Strategy                                      │
│  4. Diff existing orders vs targets                                         │
│  5. Cancel stale orders                                                     │
│  6. Place new orders (with risk checks)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  STRATEGY (step 3)   │ │  ORDER DIFF (step 4) │ │ EXECUTION (steps 5-6)│
│  strategy.py         │ │  main.py:538         │ │ exchange_client.py   │
│                      │ │                      │ │                      │
│  compute_quotes()    │ │  diff_orders()       │ │  cancel_order()      │
│  - Calculate fair    │ │  - Compare existing  │ │  place_order()       │
│    value             │ │    vs targets        │ │                      │
│  - Apply inventory   │ │  - Uses matches()    │ │  ⚠️ STATE UPDATE     │
│    skew              │ │    with tolerance    │ │  HAPPENS HERE        │
│  - Return TargetOrder│ │                      │ │                      │
│    list              │ │  Returns:            │ │                      │
│                      │ │  (to_cancel,         │ │                      │
│  Returns:            │ │   to_place)          │ │                      │
│  List[TargetOrder]   │ │                      │ │                      │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

### Critical State Locations

| State | Location | Source of Truth |
|-------|----------|-----------------|
| Open Orders | `StateManager.open_orders` (in-memory dict) | Should match exchange, but can drift |
| Positions | `StateManager.positions` (in-memory) + DynamoDB | Reconciled from fills on startup |
| Risk State | `StateManager.risk_state` | DynamoDB (persisted every 10 loops) |

### Order Matching Logic (models.py:222)

The `TargetOrder.matches()` method determines if an existing order matches a target:

```python
def matches(self, order: Order, tolerance: float = 0.01) -> bool:
    # Must match: market_id, size (exact), side mapping, price (within tolerance)

    # Side mapping:
    # - TargetOrder.side="bid" matches Order.side="yes"
    # - TargetOrder.side="ask" matches Order.side="no"

    # Price comparison uses YES price for both
```

**⚠️ Potential Issue**: If `size` changes between target computations, orders won't match and will be cancelled/re-placed unnecessarily.

### State Synchronization Points

#### 1. Startup Reconciliation (main.py:139-258)

```
startup()
├── get_exchange_status()           # Verify connectivity
├── Load risk_state from DynamoDB
├── Load logged_fills from DynamoDB
├── get_open_orders()               # ⚠️ EXCHANGE STATE
├── reconcile_with_exchange()       # Clears local, loads from exchange
├── [Optional] cancel_all_orders()  # If cancel_on_startup=True
├── get_fills()                     # All historical fills
├── reconcile_positions_from_fills() # Rebuild positions from fills
└── save_to_dynamo()
```

#### 2. Per-Market Processing (main.py:390-536)

```
process_market(market_id, config, decision_id)
├── get_order_book()
│   └── get_open_orders()           # ⚠️ Fetches own orders to filter from book
├── get_inventory()                 # From StateManager (in-memory)
├── compute_quotes()                # Strategy decision
├── get_open_orders_for_market()    # ⚠️ FROM IN-MEMORY STATE (not exchange!)
├── diff_orders()
│   └── For each existing order:
│       └── target.matches(order)   # Check if order still valid
├── For to_cancel:
│   ├── cancel_order()              # API call
│   └── remove_order()              # ⚠️ Updates local state ONLY if cancel succeeds
└── For to_place:
    ├── risk.check_order()
    ├── place_order()               # API call
    └── record_order()              # ⚠️ Updates local state ONLY if place succeeds
```

### Known State Mismatch Scenarios

#### Scenario 1: Cancel Fails but State Updated
**Problem**: If `cancel_order()` API call fails but we call `remove_order()` anyway.
**Current Code**: `remove_order()` only called if `cancel_order()` returns True (main.py:500-501).
**Risk**: Low, but verify cancel logic in exchange_client.py.

#### Scenario 2: Place Succeeds but Record Fails
**Problem**: Order placed on exchange but not tracked locally.
**Current Code**: `record_order()` called only if `place_order()` returns a valid Order (main.py:516-517).
**Risk**: If exception after API success but before record, order exists on exchange but not locally.

#### Scenario 3: Stale In-Memory State
**Problem**: `get_open_orders_for_market()` reads from `StateManager.open_orders` which may be stale.
**Current Code**: Periodic reconciliation with exchange every 10 loops (configurable via `config_refresh_interval`).
**Risk**: LOW - State drift is corrected every reconciliation cycle. Drift is logged for monitoring.

#### Scenario 4: Order Book Filtering Error
**Problem**: `get_order_book()` fetches own orders to filter them out. If this fetch fails, own orders appear in book.
**Current Code**: Logs warning but continues with unfiltered book (exchange_client.py:164-169).
**Risk**: Medium - May cause strategy to compute incorrect prices.

#### Scenario 5: Partial Cancel in diff_orders
**Problem**: If some cancels succeed and some fail, we may be in inconsistent state.
**Current Code**: Each cancel is independent; failures logged but loop continues.
**Risk**: Medium - May have more orders live than intended.

### Price/Side Convention Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRICE CONVENTIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│ Strategy Layer (TargetOrder)     │ Exchange Layer (Order, Kalshi API)  │
│                                  │                                      │
│ side="bid" → buying YES          │ side="yes" → buying YES             │
│ side="ask" → selling YES         │ side="no"  → buying NO (=sell YES)  │
│ price = YES price (always)       │ price = YES price (always)          │
│                                  │ yes_price in API = YES price        │
└─────────────────────────────────────────────────────────────────────────┘

Example: Sell YES at $0.65
- TargetOrder: side="ask", price=0.65
- Order:       side="no",  price=0.65  (yes_price=65 in API)
- Kalshi API:  action="buy", side="no", yes_price=65
```

### Periodic Reconciliation (Implemented)

The bot now performs automatic reconciliation with the exchange every 10 loops:

1. **When**: Every `config_refresh_interval` loops (default: 10)
2. **What**: Fetches all open orders from exchange and compares to local state
3. **Drift Detection**: Logs warnings for:
   - `orders_only_local`: Orders in local state but not on exchange (stale/filled/cancelled)
   - `orders_only_exchange`: Orders on exchange but not tracked locally (missed placement)
4. **Correction**: Replaces local state with exchange state (exchange is source of truth)
5. **Persistence**: Open orders are now persisted to DynamoDB (`dora_state` table, key=`open_orders`)

#### DynamoDB Open Orders Structure

Open orders are saved to the `dora_state` table with key `open_orders`:

```json
{
  "key": "open_orders",
  "orders": {
    "order-id-1": {
      "market_id": "MARKET-TICKER",
      "side": "yes",
      "price": 0.45,
      "size": 10,
      "decision_id": "run-id:market:loop",
      "client_order_id": "abc123",
      "status": "resting",
      "created_at": "2025-01-15T12:00:00",
      "tif": "gtc"
    }
  },
  "last_updated": "2025-01-15T12:05:00"
}
```

This enables external dashboards to display current open orders.

### Recommendations for Debugging State Mismatches

1. **Enable Debug Logging**: Set log level to DEBUG to see order book fetches and diffs.

2. **Monitor Drift Logs**: Watch for "Order state drift detected" warnings which indicate reconciliation corrected mismatches.

3. **Log Open Orders Before/After Diff**:
   ```python
   # Before diff
   logger.debug("Existing orders", extra={"orders": [vars(o) for o in existing_orders]})
   logger.debug("Target orders", extra={"targets": [vars(t) for t in target_orders]})
   # After diff
   logger.debug("Diff result", extra={"to_cancel": len(to_cancel), "to_place": len(to_place)})
   ```

4. **Check DynamoDB State**: Query `dora_state` table for `open_orders` key to see persisted state.

5. **Track Order Lifecycle**: Log each order from creation through fill/cancel with consistent decision_id for full traceability.

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
