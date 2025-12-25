# Dora Bot: Batch Order Execution Migration Plan

## Overview

Migrate from per-market order execution to batch-based execution to:
1. **Fix state synchronization bugs** - The current per-market approach has race conditions where cancelled orders get re-added to local state from stale exchange data
2. **Improve efficiency** - Reduce API calls and latency by batching operations
3. **Better rate limit handling** - Centralized rate limit management with proper backoff

## Current Architecture Problems

### State Synchronization Bug (Root Cause of 404 Errors)

```
Loop N:
  1. Fetch exchange_orders (includes order X)
  2. process_market():
     - Diff decides to cancel order X
     - cancel_order() succeeds
     - remove_order() removes X from local state
     - Lines 558-560: Re-syncs from exchange_orders (stale!)
     - Order X re-added to local state (without client_order_id)

Loop N+1:
  3. Local state still has order X
  4. Tries to cancel X again → 404 error
```

### Current Flow (Per-Market)

```
for each loop:
    for each market:
        fetch order book
        compute target quotes
        diff existing vs target
        cancel orders (one by one)
        place orders (one by one)
    process fills
```

**Problems:**
- State updates interleaved with stale data reads
- Many individual API calls (slow, rate limit prone)
- Order state can drift between cancel and place within same market

---

## Proposed Architecture

### New Flow (Batch-Based)

```
for each loop:
    1. Update configs (every X loops)
    2. Reconcile order state with exchange (SINGLE SOURCE OF TRUTH)
    3. Update positions from fills
    4. Calculate ALL target orders across ALL markets (pure computation, no API calls)
    5. Diff: compute cancellations and placements needed
    6. Batch execute cancellations (batches of 20, log failures)
    7. Batch execute placements (batches of 20, log failures)
```

### Key Principles

1. **Single source of truth**: Exchange state is authoritative. Local state is rebuilt from exchange at the START of each loop only.
2. **Separation of concerns**: Computation phase (steps 4-5) is pure and has no side effects. Execution phase (steps 6-7) handles all API calls.
3. **Failure logging**: Batch operations log individual failures for debugging, but don't reconcile mid-loop. Failed operations will be retried on next loop after fresh reconciliation.

---

## Kalshi Batch API Reference

### Batch Cancel Orders
- **Endpoint**: `DELETE /trade-api/v2/portfolio/orders/batched`
- **Limit**: 20 orders per request
- **Request Body**: `{"order_ids": ["id1", "id2", ...]}`
- **Rate Limit**: Each order counts against per-second rate limit
- **Source**: [Kalshi Batch Cancel Docs](https://docs.kalshi.com/api-reference/portfolio/batch-cancel-orders)

### Batch Create Orders
- **Endpoint**: `POST /trade-api/v2/portfolio/orders/batched`
- **Limit**: 20 orders per request
- **Request Body**:
```json
{
  "orders": [
    {
      "ticker": "MARKET-TICKER",
      "client_order_id": "unique-id",
      "side": "yes" | "no",
      "action": "buy" | "sell",
      "count": 10,
      "type": "limit",
      "yes_price": 45,
      "expiration_ts": 1234567890,
      "post_only": true
    }
  ]
}
```
- **Response**: Array of results per order (success or error)
- **Source**: [Kalshi Batch Create Docs](https://docs.kalshi.com/api-reference/portfolio/batch-create-orders)

---

## Implementation Plan

### Phase 1: Add Batch API Methods to Exchange Client

**File**: `exchange_client.py`

```python
def batch_cancel_orders(
    self,
    order_ids: List[str],
    decision_id: Optional[str] = None,
) -> BatchCancelResult:
    """Cancel up to 20 orders in a single API call.

    Returns:
        BatchCancelResult with succeeded/failed order IDs
    """

def batch_place_orders(
    self,
    orders: List[OrderRequest],
    decision_id: Optional[str] = None,
) -> BatchPlaceResult:
    """Place up to 20 orders in a single API call.

    Returns:
        BatchPlaceResult with placed orders and any errors
    """
```

**New models** (`models.py`):
```python
@dataclass
class OrderRequest:
    """Request to place a new order."""
    market_id: str
    side: str  # 'yes' or 'no'
    price: int
    size: int
    client_order_id: str
    decision_id: Optional[str] = None

@dataclass
class BatchCancelResult:
    """Result from a single batch cancel API call."""
    succeeded: List[str]  # order_ids
    failed: List[Tuple[str, str]]  # (order_id, error_message)

@dataclass
class BatchPlaceResult:
    """Result from a single batch place API call."""
    placed: List[Order]
    failed: List[Tuple[OrderRequest, str]]  # (request, error_message)

@dataclass
class BatchCancelSummary:
    """Summary across all cancel batches in a loop."""
    succeeded: int
    failed: int
    failures: List[Tuple[str, str]]  # (order_id, error_message) - for recurring failure tracking

@dataclass
class BatchPlaceSummary:
    """Summary across all place batches in a loop."""
    succeeded: int
    failed: int
    failures: List[Tuple[OrderRequest, str]]  # for recurring failure tracking
    placed_orders: List[Order]
```

### Phase 2: Add Rate Limiter with Backoff

**New file**: `rate_limiter.py`

```python
class RateLimiter:
    """Centralized rate limiter with exponential backoff."""

    def __init__(
        self,
        requests_per_second: float = 10,
        burst_limit: int = 20,
        max_backoff_seconds: float = 30,
    ):
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.max_backoff_seconds = max_backoff_seconds
        self.tokens = burst_limit
        self.last_refill = time.time()
        self.backoff_until = 0
        self.consecutive_rate_limits = 0

    def acquire(self, count: int = 1) -> float:
        """Acquire tokens, returning wait time if needed."""

    def record_rate_limit_hit(self):
        """Record a 429 response, increase backoff."""

    def record_success(self):
        """Record successful request, reset backoff."""
```

**Logging for rate limit tracking**:
```python
logger.warning("Rate limit hit", extra={
    "event_type": EventType.RATE_LIMIT,
    "consecutive_hits": self.consecutive_rate_limits,
    "backoff_seconds": backoff_time,
    "endpoint": endpoint,
})
```

### Phase 3: Refactor Main Loop

**File**: `main.py`

Replace `process_market()` per-market approach with batch orchestration:

```python
async def run_loop(self):
    while True:
        loop_start = time.time()

        try:
            # 1. Check risk limits
            should_halt, halt_reason = self.risk.should_halt_trading(self.state)
            if should_halt:
                await self._handle_halt(halt_reason)
                break

            # 2. Refresh configs (every X loops)
            if self.loop_count % self.config_refresh_interval == 0:
                self.market_configs = self.dynamo.get_all_market_configs(enabled_only=True)

            # 3. Reconcile order state with exchange (ALWAYS)
            exchange_orders = self.exchange.get_open_orders()
            self.state.reconcile_with_exchange(exchange_orders)

            # 4. Process fills and update positions
            fills = self.exchange.get_fills(since=self.state.risk_state.last_fill_timestamp)
            if fills:
                self.state.update_from_fills(fills)

            # 5. Calculate targets for ALL markets (pure computation)
            all_targets = self._compute_all_targets(exchange_orders)

            # 6. Diff to get cancellations and placements
            to_cancel, to_place = self._diff_all_orders(exchange_orders, all_targets)

            # 7. Execute cancellations in batches
            if to_cancel:
                await self._batch_cancel(to_cancel)

            # 8. Execute placements in batches
            if to_place:
                await self._batch_place(to_place)

            # 9. Periodic persistence
            if self.loop_count % 10 == 0:
                self.state.save_to_dynamo()

            self.loop_count += 1
            await self._sleep_for_interval(loop_start)

        except Exception as e:
            logger.error("Error in main loop", ...)
            await asyncio.sleep(5)
```

### Phase 4: Implement Batch Execution Methods

```python
async def _batch_cancel(self, orders_to_cancel: List[Order]) -> BatchCancelSummary:
    """Cancel orders in batches of 20. Logs failures but does not reconcile mid-loop."""
    BATCH_SIZE = 20
    decision_id = generate_decision_id(self.bot_run_id, "BATCH_CANCEL", self.loop_count)

    total_succeeded = 0
    total_failed = 0
    all_failures = []

    for i in range(0, len(orders_to_cancel), BATCH_SIZE):
        batch = orders_to_cancel[i:i + BATCH_SIZE]
        order_ids = [o.order_id for o in batch]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(orders_to_cancel) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info("Executing cancel batch", extra={
            "event_type": EventType.BATCH_CANCEL,
            "decision_id": decision_id,
            "batch_number": batch_num,
            "total_batches": total_batches,
            "batch_size": len(batch),
            "total_to_cancel": len(orders_to_cancel),
            "order_ids": order_ids,
        })

        result = self.exchange.batch_cancel_orders(order_ids, decision_id)
        total_succeeded += len(result.succeeded)
        total_failed += len(result.failed)

        # Log individual failures for debugging recurring issues
        for order_id, error_msg in result.failed:
            order = next((o for o in batch if o.order_id == order_id), None)
            logger.warning("Cancel failed", extra={
                "event_type": EventType.BATCH_CANCEL_FAILED,
                "decision_id": decision_id,
                "order_id": order_id,
                "market": order.market_id if order else "unknown",
                "error_msg": error_msg,
                "batch_number": batch_num,
            })
            all_failures.append((order_id, error_msg))

    # Log batch summary
    logger.info("Batch cancel complete", extra={
        "event_type": EventType.BATCH_CANCEL_SUMMARY,
        "decision_id": decision_id,
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "total_requested": len(orders_to_cancel),
    })

    return BatchCancelSummary(
        succeeded=total_succeeded,
        failed=total_failed,
        failures=all_failures,
    )


async def _batch_place(self, orders_to_place: List[OrderRequest]) -> BatchPlaceSummary:
    """Place orders in batches of 20. Logs failures but does not reconcile mid-loop."""
    BATCH_SIZE = 20
    decision_id = generate_decision_id(self.bot_run_id, "BATCH_PLACE", self.loop_count)

    total_succeeded = 0
    total_failed = 0
    all_failures = []
    placed_orders = []

    for i in range(0, len(orders_to_place), BATCH_SIZE):
        batch = orders_to_place[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(orders_to_place) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info("Executing place batch", extra={
            "event_type": EventType.BATCH_PLACE,
            "decision_id": decision_id,
            "batch_number": batch_num,
            "total_batches": total_batches,
            "batch_size": len(batch),
            "total_to_place": len(orders_to_place),
            "orders": [
                {"market": o.market_id, "side": o.side, "price": o.price, "size": o.size}
                for o in batch
            ],
        })

        result = self.exchange.batch_place_orders(batch, decision_id)
        total_succeeded += len(result.placed)
        total_failed += len(result.failed)
        placed_orders.extend(result.placed)

        # Log successful placements
        for order in result.placed:
            logger.info("Order placed", extra={
                "event_type": EventType.ORDER_PLACED,
                "decision_id": decision_id,
                "order_id": order.order_id,
                "market": order.market_id,
                "side": order.side,
                "price": order.price,
                "size": order.size,
                "client_order_id": order.client_order_id,
                "batch_number": batch_num,
            })

        # Log individual failures for debugging recurring issues
        for request, error_msg in result.failed:
            logger.warning("Place failed", extra={
                "event_type": EventType.BATCH_PLACE_FAILED,
                "decision_id": decision_id,
                "market": request.market_id,
                "side": request.side,
                "price": request.price,
                "size": request.size,
                "client_order_id": request.client_order_id,
                "error_msg": error_msg,
                "batch_number": batch_num,
            })
            all_failures.append((request, error_msg))

    # Log batch summary
    logger.info("Batch place complete", extra={
        "event_type": EventType.BATCH_PLACE_SUMMARY,
        "decision_id": decision_id,
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "total_requested": len(orders_to_place),
    })

    return BatchPlaceSummary(
        succeeded=total_succeeded,
        failed=total_failed,
        failures=all_failures,
        placed_orders=placed_orders,
    )
```

### Phase 5: Compute All Targets

```python
def _compute_all_targets(
    self,
    exchange_orders: List[Order],
) -> Dict[str, List[TargetOrder]]:
    """Compute target orders for all enabled markets.

    This is a PURE function - no API calls, no state mutations.

    Returns:
        Dict mapping market_id -> list of target orders
    """
    all_targets = {}

    # Group exchange orders by market
    orders_by_market = defaultdict(list)
    for order in exchange_orders:
        orders_by_market[order.market_id].append(order)

    for market_id, config in self.market_configs.items():
        decision_id = generate_decision_id(self.bot_run_id, market_id, self.loop_count)

        try:
            # Fetch order book
            market_orders = orders_by_market.get(market_id, [])
            order_book = self.exchange.get_order_book(market_id, own_orders=market_orders)

            # Get position
            position = self.state.get_inventory(market_id)

            # Compute quotes (pure strategy computation)
            target_orders, price_calc = self.strategy.compute_quotes(
                order_book, position, config
            )

            all_targets[market_id] = target_orders or []

            # Log decision
            logger.info("Decision made", extra={
                "event_type": EventType.DECISION_MADE,
                "market": market_id,
                "decision_id": decision_id,
                "target_count": len(all_targets[market_id]),
            })

        except Exception as e:
            logger.error("Error computing targets for market", extra={
                "event_type": EventType.ERROR,
                "market": market_id,
                "error": str(e),
            })
            all_targets[market_id] = []

    return all_targets


def _diff_all_orders(
    self,
    exchange_orders: List[Order],
    all_targets: Dict[str, List[TargetOrder]],
) -> Tuple[List[Order], List[OrderRequest]]:
    """Diff current exchange orders against all targets.

    Returns:
        (orders_to_cancel, orders_to_place)
    """
    to_cancel = []
    to_place = []

    # Group exchange orders by market
    orders_by_market = defaultdict(list)
    for order in exchange_orders:
        orders_by_market[order.market_id].append(order)

    # Get all markets (union of exchange orders and targets)
    all_markets = set(orders_by_market.keys()) | set(all_targets.keys())

    for market_id in all_markets:
        existing = orders_by_market.get(market_id, [])
        targets = all_targets.get(market_id, [])

        market_cancel, market_place = self.diff_orders(existing, targets)

        to_cancel.extend(market_cancel)

        # Convert TargetOrders to OrderRequests
        for target in market_place:
            to_place.append(OrderRequest(
                market_id=market_id,
                side='yes' if target.side == 'bid' else 'no',
                price=target.price,
                size=target.size,
                client_order_id=secrets.token_hex(8),
                decision_id=generate_decision_id(self.bot_run_id, market_id, self.loop_count),
            ))

    return to_cancel, to_place
```

---

## New Event Types for Logging

Add to `structured_logger.py`:

```python
class EventType:
    # ... existing types ...

    # Batch operations
    BATCH_CANCEL = "BATCH_CANCEL"              # Starting a cancel batch
    BATCH_CANCEL_FAILED = "BATCH_CANCEL_FAILED"  # Individual cancel failure (for tracking recurring issues)
    BATCH_CANCEL_SUMMARY = "BATCH_CANCEL_SUMMARY"  # Summary after all cancel batches complete
    BATCH_PLACE = "BATCH_PLACE"                # Starting a place batch
    BATCH_PLACE_FAILED = "BATCH_PLACE_FAILED"    # Individual place failure (for tracking recurring issues)
    BATCH_PLACE_SUMMARY = "BATCH_PLACE_SUMMARY"  # Summary after all place batches complete

    # Rate limiting
    RATE_LIMIT = "RATE_LIMIT"                  # Hit rate limit (429 response)
    RATE_LIMIT_BACKOFF = "RATE_LIMIT_BACKOFF"  # Backing off due to rate limits
```

---

## Migration Steps

### Step 1: Add Infrastructure (Non-Breaking)
- [ ] Add `OrderRequest`, `BatchCancelResult`, `BatchPlaceResult` to models.py
- [ ] Add `batch_cancel_orders()`, `batch_place_orders()` to exchange_client.py
- [ ] Add `RateLimiter` class
- [ ] Add new event types to structured_logger.py
- [ ] Write unit tests for new methods

### Step 2: Add New Loop Implementation (Feature Flag)
- [ ] Add `use_batch_execution` config flag (default: False)
- [ ] Implement `_compute_all_targets()`, `_diff_all_orders()`
- [ ] Implement `_batch_cancel()`, `_batch_place()`
- [ ] Add new `run_loop_batch()` method alongside existing `run_loop()`

### Step 3: Test in Demo Environment
- [ ] Deploy with `use_batch_execution=True` to demo
- [ ] Monitor for state drift (should be zero)
- [ ] Verify rate limit handling under load
- [ ] Compare latency and fill rates vs old implementation

### Step 4: Production Rollout
- [ ] Enable feature flag in production
- [ ] Monitor CloudWatch for errors
- [ ] Verify no 404 cancel errors
- [ ] Remove old per-market code path after validation period

---

## Rollback Plan

If issues arise:
1. Set `use_batch_execution=False` in config
2. Bot will use existing per-market execution
3. No code changes required for rollback

---

## Success Metrics

1. **Zero 404 cancel errors** - No more "order not found" when cancelling
2. **Reduced API calls** - ~60% fewer calls for same number of orders
3. **Consistent state** - No drift between local and exchange state
4. **Rate limit visibility** - Clear logging when approaching limits
5. **Lower loop latency** - Batch operations should be faster than sequential

---

## Design Decisions

1. **Partial batch failures**: If 5 of 20 cancels fail, skip failed ones. They'll be retried on next loop after fresh reconciliation.

2. **Order of operations**: Cancel all before placing any to free up capital/exposure.

3. **Reconciliation frequency**: Once at the start of each loop only. No mid-loop reconciliation to reduce API calls and complexity. Failed operations will naturally retry on next loop.

4. **Failure tracking**: Log each individual failure with market, order details, and error message. This allows CloudWatch queries to identify recurring failures (e.g., same market failing repeatedly).

## Open Questions

1. **Market priority**: Should some markets be processed before others?
   - Recommendation: Not initially, but could add priority config later

2. **Failure thresholds**: Should we halt if too many failures occur in a single loop?
   - Recommendation: Log warning if >50% of operations fail, but continue. Let risk manager handle halts.
