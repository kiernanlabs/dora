# Kalshi Market Making Bot - Architecture Plan

## 1. Big-picture architecture

Five layers:

### Exchange Adapter Layer
- Talks to Kalshi's REST API (WebSocket for order book streaming if available)
- Handles auth, rate limits, retries, and order lifecycle
- Exposes a clean internal interface:
  - `get_order_book(market_id)`
  - `get_positions()`
  - `place_order(market_id, side, price, size, tif)`
  - `cancel_order(order_id)`
  - `cancel_all_orders(market_id=None)`
  - `get_fills(since)`
  - `get_balance()`

### Data & State Layer (DynamoDB-backed)
- **In-memory hot state** for low-latency decisions:
  - Current order books (best bid/ask, depth)
  - Your open orders
  - Current inventory per market
- **DynamoDB for persistence**:
  - Positions and inventory (survives restarts)
  - Open orders (reconciled on startup)
  - Last processed fill timestamp
  - Daily PnL snapshots

### Config Layer (DynamoDB)
- Per-market configuration stored in DynamoDB:
  - `enabled: bool`
  - `max_inventory_yes / max_inventory_no`
  - `min_spread` (required spread to quote)
  - `quote_size`
  - `inventory_skew_factor`
  - `toxicity_score` (if pre-computed)
- Global config:
  - `max_total_exposure`
  - `max_daily_loss`
  - `loop_interval_ms`
- Allows runtime config changes without redeploying

### Strategy / Decision Engine
- Inputs:
  - Current order books
  - Your inventory & risk
  - Per-market config (spread requirements, sizing)
- Outputs:
  - Target quotes per market
  - Orders to cancel/modify
- Implements:
  - Required spread based on config/toxicity
  - Inventory steering (skew quotes to flatten)
  - Per-market on/off via config

### Execution & Risk Controller
- Converts target quotes into API calls safely
- Enforces:
  - Per-market inventory caps
  - Global exposure cap
  - Daily loss limit (halt trading if breached)
  - Rate limiting / API throttling
- Handles partial fills, stale orders, race conditions
- Emergency kill switch: cancel all orders

## 2. Infrastructure & Deployment

### EC2 Setup
- **Instance**: t3.micro or t3.small sufficient for single-process bot
- **Region**: us-east-1 (closest to Kalshi servers, minimize latency)
- **OS**: Amazon Linux 2023 or Ubuntu 22.04
- **Python**: 3.11+ with venv

### DynamoDB Tables

**Table: `dora_market_config`**
```
PK: market_id (String)
Attributes:
  - enabled (Boolean)
  - max_inventory_yes (Number)
  - max_inventory_no (Number)
  - min_spread (Number, e.g., 0.06)
  - quote_size (Number)
  - inventory_skew_factor (Number)
  - toxicity_score (Number, optional)
  - updated_at (String, ISO timestamp)
```

**Table: `dora_state`**
```
PK: key (String) - e.g., "positions", "global_config", "risk_state"
Attributes vary by key type:
  - For "positions": positions (Map of market_id -> {yes_qty, no_qty, avg_cost_yes, avg_cost_no})
  - For "global_config": max_total_exposure, max_daily_loss, loop_interval_ms, trading_enabled
  - For "risk_state": daily_pnl, last_fill_timestamp, trading_halted
```

**Table: `dora_trade_log`**
```
PK: date (String, YYYY-MM-DD)
SK: timestamp#order_id (String)
Attributes:
  - market_id
  - side (buy/sell)
  - price
  - size
  - fill_price (if filled)
  - fill_size
  - status (placed/filled/cancelled)
  - pnl_realized (if applicable)
```

**Table: `dora_decision_log`** (optional, for debugging)
```
PK: date (String)
SK: timestamp (String)
Attributes:
  - market_id
  - order_book_snapshot
  - inventory
  - target_quotes
  - orders_placed
  - orders_cancelled
```

### Secrets Management
- Store Kalshi API credentials in AWS Secrets Manager or SSM Parameter Store
- Bot reads credentials on startup via boto3
- Never commit credentials to code

### Process Management
- Use systemd service for auto-restart on crash
- Or: supervisord / pm2 for Python
- CloudWatch for logs and basic alerting

## 3. Core Runtime Model

**Language**: Python 3.11+

**Dependencies**:
- `boto3` (DynamoDB, Secrets Manager)
- `requests` or `httpx` (Kalshi API)
- `websockets` (if using WS for order book)

**Single-process event loop**:
```
Loop every N ms (configurable, start with 500ms):
  1. Refresh market data (order books)
  2. Fetch recent fills, update inventory
  3. Reload per-market config from DynamoDB (every ~10 loops or on-demand)
  4. Run strategy → compute target quotes
  5. Diff vs current working orders
  6. Execute cancels/places (with risk checks)
  7. Persist state to DynamoDB (async or batched)
```

**Slower risk loop** (every 5-10 seconds):
- Check daily PnL vs max loss
- Verify positions match exchange
- Kill switch if anomalies detected

## 4. Process Flow (One Tick)

### Step 1 – Fetch Data
For each enabled market (from DynamoDB config):
- Get order book snapshot (best bid/ask, depth)
- Get your open orders
- Pull fills since `last_fill_timestamp`
- Update in-memory inventory and PnL

### Step 2 – Compute Target Quotes
For each market, feed into strategy engine:

**Inputs**:
- Order book: best_bid, best_ask, spread
- Your inventory (yes_qty, no_qty)
- Market config from DynamoDB (min_spread, quote_size, skew_factor)

**Logic**:
1. Check if `market_spread >= min_spread` → if not, don't quote (or reduce size)
2. Compute fair value or use midpoint
3. Apply inventory skew:
   - Long YES → widen bid, tighten ask (encourage selling)
   - Long NO → opposite
4. Determine quote prices (typically 1 tick inside best, subject to spread)
5. Size: full `quote_size` if neutral, reduce as inventory approaches cap

**Output**: List of target orders
```python
[
  {"market_id": "ABC", "side": "buy", "price": 0.44, "size": 15},
  {"market_id": "ABC", "side": "sell", "price": 0.50, "size": 15},
]
```

### Step 3 – Diff Orders
Compare desired vs working orders:
- **Cancel if**: price differs by > 1 tick, or size needs reduction
- **Place new**: for any target not covered by existing order
- Avoid unnecessary cancel/replace churn (costs fees, latency)

### Step 4 – Risk Checks (Pre-Send)
Before each order:
- `abs(new_inventory) <= max_inventory_{yes/no}` for that market
- `total_exposure <= max_total_exposure`
- `daily_pnl >= -max_daily_loss` (halt if breached)

If any check fails:
- Skip the order
- Optionally disable that market in config
- If PnL limit hit → cancel all orders, halt trading

## 5. Module Structure

```
dora_bot/
├── main.py              # Entry point, main loop
├── config.py            # Load from DynamoDB + environment
├── exchange_client.py   # Kalshi API wrapper
├── state_manager.py     # In-memory state + DynamoDB persistence
├── strategy.py          # Quote computation logic
├── risk_manager.py      # Risk checks and limits
├── dynamo.py            # DynamoDB helpers
└── models.py            # Data classes (Order, Position, MarketConfig)
```

### exchange_client.py
```python
class KalshiClient:
    def __init__(self, api_key, api_secret):
        # Auth setup, session management

    def get_order_book(self, market_id) -> OrderBook
    def get_open_orders(self) -> list[Order]
    def get_fills(self, since: datetime) -> list[Fill]
    def get_balance() -> Balance
    def place_order(market_id, side, price, size, tif="gtc") -> Order
    def cancel_order(order_id) -> bool
    def cancel_all_orders(market_id=None) -> int

    # Internal: retry logic, rate limiting, error handling
```

### state_manager.py
```python
class StateManager:
    def __init__(self, dynamo_client):
        self.positions: dict[str, Position] = {}
        self.open_orders: dict[str, Order] = {}
        self.last_fill_timestamp: datetime
        self.daily_pnl: float

    def load_from_dynamo(self)          # Startup recovery
    def save_to_dynamo(self)            # Periodic persistence
    def update_from_fills(self, fills)  # Process new fills
    def get_inventory(self, market_id) -> Position
    def record_order(self, order)
    def reconcile_with_exchange(self, exchange_orders)  # Startup sync
```

### strategy.py
```python
class MarketMaker:
    def compute_quotes(
        self,
        order_book: OrderBook,
        position: Position,
        config: MarketConfig
    ) -> list[TargetOrder]:
        # 1. Check spread viability
        # 2. Compute skewed bid/ask based on inventory
        # 3. Determine sizes
        # 4. Return target orders
```

### risk_manager.py
```python
class RiskManager:
    def __init__(self, global_config):
        self.max_total_exposure = global_config.max_total_exposure
        self.max_daily_loss = global_config.max_daily_loss

    def check_order(self, order, state) -> bool
    def check_market_limits(self, market_id, state, config) -> bool
    def should_halt_trading(self, state) -> bool
    def get_exposure(self, state) -> float
```

### main.py (Runner)
```python
async def main():
    # Load secrets, init clients
    client = KalshiClient(...)
    dynamo = DynamoClient(...)
    state = StateManager(dynamo)
    risk = RiskManager(global_config)
    strategy = MarketMaker()

    # Startup: reconcile state with exchange
    state.load_from_dynamo()
    state.reconcile_with_exchange(client.get_open_orders())

    while True:
        try:
            # 1. Fetch data
            market_configs = dynamo.get_enabled_markets()
            for market_id, config in market_configs.items():
                order_book = client.get_order_book(market_id)

                # 2. Compute target quotes
                position = state.get_inventory(market_id)
                targets = strategy.compute_quotes(order_book, position, config)

                # 3. Diff and execute
                to_cancel, to_place = diff_orders(state.open_orders, targets)

                for order_id in to_cancel:
                    client.cancel_order(order_id)

                for order in to_place:
                    if risk.check_order(order, state):
                        client.place_order(**order)

            # 4. Process fills, update state
            fills = client.get_fills(since=state.last_fill_timestamp)
            state.update_from_fills(fills)

            # 5. Periodic persistence
            state.save_to_dynamo()

            # 6. Risk check
            if risk.should_halt_trading(state):
                client.cancel_all_orders()
                log_alert("Trading halted - risk limit breached")
                break

        except Exception as e:
            log_error(e)
            # Consider cancel-all on unexpected errors

        await asyncio.sleep(config.loop_interval_ms / 1000)
```

## 6. Startup & Recovery

### On Bot Startup
1. Load credentials from Secrets Manager
2. Connect to Kalshi API, verify auth
3. Load state from DynamoDB (`positions`, `risk_state`, `last_fill_timestamp`)
4. Fetch all open orders from exchange
5. Reconcile: DynamoDB state vs exchange reality (exchange is source of truth)
6. Fetch fills since last timestamp to catch any missed during downtime
7. Begin main loop

### On Crash/Restart
- systemd auto-restarts the process
- Startup reconciliation handles state recovery
- Open orders remain on exchange (GTC) unless you cancel-on-exit
- Consider: cancel all orders on startup, re-quote fresh (safer but loses queue position)

## 7. Monitoring & Alerts

### CloudWatch Metrics (via boto3)
- `orders_placed` / `orders_cancelled` per minute
- `fills_count` per minute
- `daily_pnl` (gauge)
- `total_exposure` (gauge)
- `loop_latency_ms`
- `api_errors` count

### Alerts
- Daily PnL below threshold → SNS notification
- Trading halted → SNS + email
- API errors sustained → alert
- No fills in X hours (if expecting activity) → warning

### Logs
- Structured JSON logs to CloudWatch Logs
- Include: timestamp, market_id, action, prices, sizes, reason
- Retain for analysis / debugging

## 8. Key Considerations

### Kalshi-Specific
- **Tick size**: Kalshi uses cents (0.01), prices are 0.01-0.99
- **Fees**: Understand fee structure, factor into spread requirements
- **Rate limits**: Check API docs, implement backoff
- **Market hours**: Some markets have trading windows
- **Settlement**: Binary contracts settle at 0 or 1

### Risk Management Priorities
1. **Position limits** are critical - don't exceed what you can afford to lose
2. **Daily loss limit** as circuit breaker
3. **Reconciliation** on startup - never trust cached state blindly
4. **Cancel-all** capability must always work

### Performance
- t3.micro is fine for <10 markets at 500ms intervals
- DynamoDB latency adds ~10-50ms per call - batch where possible
- Consider caching market config (refresh every N loops, not every loop)

### Future Enhancements (not for v1)
- WebSocket for real-time order book updates
- Multiple market support with async/concurrent fetching
- Backtesting framework using historical data
- Web dashboard for monitoring/config changes
- Multi-instance deployment with coordination

## 9. Implementation Order

1. **exchange_client.py** - Get API working, test in sandbox/paper
2. **models.py** - Define data structures
3. **dynamo.py** - Table creation, basic CRUD
4. **state_manager.py** - In-memory + persistence
5. **risk_manager.py** - Simple limits first
6. **strategy.py** - Start with basic spread quoting
7. **main.py** - Wire it together
8. **Deploy** - EC2, systemd, CloudWatch
9. **Iterate** - Tune parameters, add features