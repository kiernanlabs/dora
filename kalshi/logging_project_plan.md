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
- Add DynamoDB write helpers here to keep all logging outputs in one module:
  - `log_decision_record(...)` for `decision_log`
  - `log_execution_event(...)` for `execution_log`

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

Defer to after structured logging is working. Objective: write an immutable execution ledger per decision cycle that can be correlated with decision logs now, and migrated to S3 later.

### 3.1 Data model and table design (DynamoDB)

**New table:** `execution_log`
- **PK:** `bot_run_id` (string)
- **SK:** `decision_id#event_ts` (string, sortable)
- **TTL:** `expires_at` (epoch seconds) for retention control (e.g., 30-90 days)
- **GSI1 (decision lookup):** `decision_id` (PK) + `event_ts` (SK)

**Base attributes:**
- `event_ts` (ISO-8601)
- `env`, `bot_version`
- `market`, `order_id`, `side`, `price`, `size`, `status`
- `event_type` (ORDER_PLACE|ORDER_RESULT|ORDER_CANCEL|FILL)
- `latency_ms`, `error_type`, `error_msg`

**Optional attributes (future-friendly):**
- `client_order_id` (local idempotency key)
- `request_id` (Kalshi API request id)
- `expected_orders_hash` (links decision to expected orders)
- `decision_snapshot_s3_key` (reserved for S3 move)

### 3.2 Write path integration

**What to log:** every execution-relevant event emitted to structured logs should also be persisted to `execution_log`.

**Where to instrument:**
- `exchange_client.py`: ORDER_PLACE, ORDER_RESULT, ORDER_CANCEL
- `state_manager.py`: FILL

**Implementation approach:**
- Extend `structured_logger.py` with DynamoDB-backed helpers (keep all logging execution helpers in one file).
- Provide `log_execution_event(payload)` and `log_decision_record(payload)` that can be called from runtime code.
- Require `bot_run_id` + `decision_id` on every call; reject otherwise to prevent orphaned events.
- Include a local `client_order_id` for idempotency and later reconciliation.
- Implement basic retries and best-effort error logging (never block the trading loop on logging).

### 3.3 Decision log linkage

Update decision log records to include:
- `decision_id` (already planned)
- `expected_orders_hash` computed from sorted target orders
- `execution_log_partition` (bot_run_id) to simplify cross-store queries

### 3.4 Read/analysis workflow (short-term)

**Goal:** answer "decision X -> what did we attempt, what happened?"
- Query `decision_log` by market + decision_id to fetch expected orders
- Query `execution_log` by decision_id (GSI1) to fetch actual events
- Compare by `client_order_id` or order attributes; log diff to CloudWatch

### 3.5 Migration-friendly storage strategy

Design the DynamoDB schema so it can be **replaced** by S3 later:
- Use a normalized, append-only event schema (JSON) that can be stored as newline-delimited records in S3.
- Keep `decision_id` as the primary join key between `decision_log` and `execution_log`.
- Avoid nested, DynamoDB-only types; keep payloads JSON-serializable.

When ready to migrate:
- Dual-write DynamoDB + S3 for 1-2 releases.
- Validate parity with daily reconciliation job.
- Flip reads to S3-backed queries; retain DynamoDB as short-retention hot cache.

### 3.6 Operational guardrails

**Performance and reliability:**
- Use non-blocking, best-effort writes with bounded retries (do not slow the trading loop).
- Batch writes when possible (e.g., flush at loop boundary) but ensure ORDER_RESULT/FILL events are not dropped.
- Emit a structured ERROR log on DynamoDB write failures with `event_type=ERROR` and `target=execution_log`.

**Retention and cost:**
- Use TTL via `expires_at` to keep execution logs short-lived (30-90 days).
- Keep payloads small and JSON-serializable to simplify eventual S3 storage.

### 3.7 Implementation checklist (Phase 3)

**Schema and infra:**
- [x] Create `dora_execution_log` table with GSI and TTL
- [x] Confirm table names with environment suffixes (`_demo`, `_prod`)
- [ ] Verify IAM permissions for DynamoDB writes on execution/decision logs

**Code changes:**
- [x] `structured_logger.py`: add `log_decision_record(...)` and `log_execution_event(...)`
- [ ] `dynamo.py`: add `execution_log_table` wiring (if needed for direct calls)
- [x] `exchange_client.py`: emit execution events to DynamoDB helper
- [x] `state_manager.py`: emit FILL events to DynamoDB helper
- [x] `main.py` or `strategy.py`: ensure `decision_id` is generated and attached to decision logs
- [x] Add `client_order_id` to all order placement/cancel flows

**Testing and validation:**
- [ ] Local dry-run mode that stubs DynamoDB writes
- [ ] Smoke test: place/cancel order in demo, confirm records in `dora_execution_log_demo`
- [ ] Verify decision_id joins (decision_log + execution_log) for a single loop

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
