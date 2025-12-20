# Dora Bot Logging & Observability Refactor Plan

## Goal

Fix "decision log says trade X, but bot didn't execute" by building:
- Structured JSON logs to CloudWatch for immediate diagnostics
- Correlation IDs (bot_run_id, decision_id) across all log events
- Execution event logging to track what we attempted vs what happened

Later phases (deferred): DynamoDB execution ledgers, idempotency tables, reconciliation, UI.

---

## Current State Assessment

**What's already good:**
- 100% logger-based (no print statements except in query utility)
- Logging at INFO level with structured messages
- Container mode support (stdout-only when `CONTAINER_MODE=true`)
- Decision logging to DynamoDB with market state snapshots

**What needs improvement:**
- Logs are human-readable strings, not JSON (not CloudWatch Insights friendly)
- No correlation IDs (bot_run_id, decision_id) in log events
- No bot version tracking
- Order execution attempts/results not logged as structured events
- No way to correlate a decision with its execution outcomes

---

## Phase 1: Structured Logging Foundation (This Sprint)

### 1.1 Create `structured_logger.py` utility

Create a JSON logging formatter that emits one JSON object per line with required fields.

**Required base fields (every log event):**
```json
{
  "ts": "2024-01-15T14:30:00.123Z",
  "level": "INFO",
  "service": "dora-bot",
  "env": "prod|demo",
  "bot_version": "abc123",
  "bot_run_id": "20240115-143000-x7k2",
  "message": "Human readable message",
  "event_type": "STARTUP|HEARTBEAT|DECISION_MADE|ORDER_PLACE|ORDER_CANCEL|ORDER_RESULT|FILL|ERROR|SHUTDOWN"
}
```

**Contextual fields (when applicable):**
- `market`: ticker string
- `decision_id`: unique ID for this decision cycle
- `order_id`: Kalshi order ID
- `side`: "yes"|"no"
- `price`: decimal
- `size`: int
- `latency_ms`: int
- `error_type`, `error_msg`, `stack`: for exceptions

**Implementation approach:**
- Custom `logging.Formatter` that outputs JSON
- Context manager or thread-local for correlation IDs
- `StructuredLogger` wrapper class for typed event emission

**Files to create:**
- `kalshi/dora_bot/structured_logger.py`

**Files to modify:**
- `kalshi/dora_bot/main.py` - initialize structured logging, add run_id generation
- All modules - update log calls to include event_type and contextual fields

### 1.2 Add bot versioning and run IDs

**Bot version (`BOT_VERSION`):**
- Read from `BOT_VERSION` env var if set
- Otherwise, attempt `git rev-parse --short HEAD`
- Fallback to "unknown"

**Bot run ID (`bot_run_id`):**
- Generated at startup: `{YYYYMMDD}-{HHMMSS}-{random_suffix}`
- Stored as instance variable on DoraBot
- Passed to all components that need to log

**Decision ID (`decision_id`):**
- Generated per market per loop iteration
- Format: `{bot_run_id}:{market}:{loop_counter}`
- Links decision log entries to execution log entries

### 1.3 Refactor logging calls across codebase

Update existing `logger.info/warning/error` calls to include structured data.

**Pattern transformation:**
```python
# Before
logger.info(f"Placed order: {market_id} {type}@{price} - ID: {order_id}")

# After
logger.info("Order placed", extra={
    "event_type": "ORDER_RESULT",
    "market": market_id,
    "decision_id": decision_id,
    "side": side,
    "price": price,
    "size": size,
    "order_id": order_id,
    "status": "ACCEPTED",
    "latency_ms": latency_ms
})
```

**Modules to update:**
1. `main.py` - startup, shutdown, loop events, decision events
2. `exchange_client.py` - order placement, cancellation, API errors
3. `state_manager.py` - fill processing, state transitions
4. `strategy.py` - quote generation decisions
5. `risk_manager.py` - risk events, halt events
6. `dynamo.py` - DynamoDB operation errors

### 1.4 Event types to implement

| Event Type | When | Key Fields |
|------------|------|------------|
| `STARTUP` | Bot initialization complete | bot_version, env |
| `SHUTDOWN` | Bot shutting down | reason, graceful |
| `HEARTBEAT` | Every N loops (configurable) | markets_active, open_orders_count |
| `DECISION_MADE` | Strategy produces target quotes | market, decision_id, target_count, spread |
| `ORDER_PLACE` | Before placing order | market, decision_id, side, price, size |
| `ORDER_CANCEL` | Before cancelling order | market, order_id |
| `ORDER_RESULT` | After API response | status, order_id, latency_ms, error_msg |
| `FILL` | Fill processed | market, side, price, size, pnl_delta |
| `RISK_HALT` | Trading halted | reason, daily_pnl |
| `ERROR` | Unhandled exception or critical failure | error_type, error_msg, stack |

---

## Phase 2: CloudWatch Integration (After Phase 1)

### 2.1 CloudWatch Agent configuration

Ship journald logs to CloudWatch:
- Log group: `/dora-bot/{env}/app`
- Stream name: `{instance_id}/{bot_run_id}`
- Retention: 14 days

**Files to create:**
- `kalshi/deploy/cloudwatch-agent-config.json`
- `kalshi/deploy/README-cloudwatch.md`

### 2.2 Metric filters for alerting

Create CloudWatch metric filters:
```
ERROR events: { $.event_type = "ERROR" }
ORDER failures: { $.event_type = "ORDER_RESULT" && $.status != "ACCEPTED" }
RISK halts: { $.event_type = "RISK_HALT" }
```

### 2.3 Alarms

- `dora-bot-errors`: ERROR count > 0 in 5 min → SNS
- `dora-bot-order-failures`: ORDER_RESULT failures > 3 in 5 min → SNS
- `dora-bot-risk-halt`: Any RISK_HALT → SNS (critical)

---

## Phase 3: Execution Logging (Future)

Defer to after structured logging is working:
- Add `execution_events` DynamoDB table
- Log every order attempt/result to DynamoDB
- Add decision_id to decision_log entries
- Add expected_orders_hash for diffing

---

## Phase 4: Reconciliation (Future)

Defer until execution logging is reliable:
- Post-execution reconciliation comparing expected vs live orders
- RECONCILIATION_RESULT log events
- Independent reconciler job

---

## Implementation Checklist (Phase 1)

### Files to create:
- [ ] `kalshi/dora_bot/structured_logger.py` - JSON formatter + event helpers

### Files to modify:
- [ ] `kalshi/dora_bot/main.py` - structured logging init, run_id, version
- [ ] `kalshi/dora_bot/exchange_client.py` - ORDER_PLACE, ORDER_RESULT events
- [ ] `kalshi/dora_bot/state_manager.py` - FILL events
- [ ] `kalshi/dora_bot/strategy.py` - DECISION_MADE events
- [ ] `kalshi/dora_bot/risk_manager.py` - RISK_HALT events
- [ ] `kalshi/dora_bot/dynamo.py` - error event logging

### Deployment artifacts:
- [ ] `kalshi/deploy/dora-bot.service` - systemd unit file
- [ ] `kalshi/deploy/cloudwatch-agent-config.json` - CloudWatch agent config

### Acceptance criteria:
1. All logs are valid JSON, one object per line
2. Every log event includes: ts, level, service, env, bot_version, bot_run_id, event_type
3. Decision and execution events include decision_id for correlation
4. Bot version and run_id visible in first log line at startup
5. Logs can be queried in CloudWatch Insights by event_type, market, decision_id

---

## Definition of Done (Phase 1)

Given a question "what happened with market X at time T":
1. You can filter CloudWatch logs by `market=X` and time range
2. You can find the `DECISION_MADE` event with target quotes
3. You can find `ORDER_PLACE` and `ORDER_RESULT` events for that decision
4. You can correlate all events using `decision_id`
5. Any errors are clearly logged with `event_type=ERROR` and stack traces
