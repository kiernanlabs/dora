# Bid/Ask Refactoring Summary

## Problem
The codebase was mixing YES ask prices (e.g., 0.33) with NO prices (e.g., 0.67), causing confusion in order matching and cancellation logic. This led to orders being incorrectly identified as needing updates when they were actually correct.

## Solution
Converted all internal logic to use **bid/ask terminology with YES prices** throughout.

## Two-Layer Architecture

### Layer 1: Internal Strategy Layer (bid/ask)
- **TargetOrder**: Uses `side="bid"` or `"ask"` with prices always in YES terms
- `side="bid"`: Buying YES at this price (e.g., bid at 0.29 = buying YES at $0.29)
- `side="ask"`: Selling YES at this price (e.g., ask at 0.33 = selling YES at $0.33)

### Layer 2: Exchange API Layer (yes/no)
- **Order**: Uses `side="yes"` or `"no"` to match Kalshi's API format
- `Order.price` is ALWAYS the YES price (the `yes_price` field from Kalshi)
- Examples:
  - Buying YES at $0.29: `Order(side="yes", price=0.29)`
  - Selling YES at $0.33: `Order(side="no", price=0.33)` (yes_price=0.33 in API)

## Key Changes

### 1. TargetOrder Model (models.py)
```python
@dataclass
class TargetOrder:
    """Uses bid/ask with YES prices."""
    side: Literal["bid", "ask"]  # Changed from Literal["yes", "no"]
    price: float  # Always YES price
```

### 2. Order Model (models.py)
```python
@dataclass
class Order:
    """Uses Kalshi's yes/no format with YES prices."""
    side: Literal["yes", "no"]  # Kalshi format
    price: float  # Always YES price (yes_price from API)
```

### 3. Strategy (strategy.py)
- Changed from creating `TargetOrder(side="yes")` and `TargetOrder(side="no")`
- Now creates `TargetOrder(side="bid")` and `TargetOrder(side="ask")`
- All prices are in YES terms

**Before:**
```python
# Old code - confusing!
TargetOrder(side="no", price=0.67)  # NO price
```

**After:**
```python
# New code - clear!
TargetOrder(side="ask", price=0.33)  # YES ask price
```

### 4. Exchange Client (exchange_client.py)
**place_order()**: Accepts bid/ask, converts to Kalshi's yes/no format
```python
def place_order(self, market_id: str, side: str, price: float, ...):
    """Args: side is 'bid' or 'ask', price is always YES price."""
    if side == "bid":
        kalshi_side = "yes"
        yes_price = int(price * 100)
    else:  # side == "ask"
        kalshi_side = "no"
        yes_price = int((1.0 - price) * 100)  # Convert to NO price for API
```

**get_open_orders()**: Returns Order objects with Kalshi's format
```python
Order(
    side=order_data.get('side'),  # 'yes' or 'no' from Kalshi
    price=order_data.get('yes_price') / 100.0  # Always YES price
)
```

### 5. TargetOrder.matches() (models.py)
Handles conversion between bid/ask and yes/no:
```python
def matches(self, order: Order, tolerance: float = 0.01) -> bool:
    if self.side == "bid":
        # bid @ 0.29 matches Order(side="yes", price=0.29)
        return order.side == "yes" and abs(order.price - self.price) <= tolerance
    else:  # self.side == "ask"
        # ask @ 0.33 matches Order(side="no", price=0.33)
        return order.side == "no" and abs(order.price - self.price) <= tolerance
```

### 6. Risk Manager (risk_manager.py)
Updated to handle bid/ask from TargetOrder:
```python
# TargetOrder.side is "bid" or "ask"
if order.side == "bid":
    new_position_yes += order.size
else:  # order.side == "ask"
    new_position_no += order.size
```

### 7. Main Loop Logging (main.py)
All logging now in YES price terms, no conversion needed:
```python
# Order.price is already YES price!
logger.info(f"Cancelling order: {order_type} {order.size}@{order.price:.2f}")
```

## Benefits

1. **No More Confusion**: Always working with YES prices internally
2. **Clearer Intent**: "bid" and "ask" are more intuitive than "yes" and "no"
3. **Consistent Logging**: All prices logged are YES prices
4. **Proper Matching**: Orders now correctly match targets without conversion errors
5. **Clear Separation**: Internal representation (bid/ask) separate from API format (yes/no)

## Example: Selling YES at $0.33

**Internal (Strategy):**
```python
TargetOrder(side="ask", price=0.33)
```

**API (Exchange):**
```python
# Converted by exchange_client.place_order()
payload = {
    "side": "no",  # Buying NO = selling YES
    "yes_price": 67  # 100 - 33 = 67 cents (NO price)
}
```

**Stored (Order object):**
```python
Order(side="no", price=0.33)  # Kalshi format, but price is YES price!
```

**Logged:**
```
Placed order: MARKET-123 ASK 10@0.33
```

All in YES terms, crystal clear!

## Critical Bug Fix: Order Book Parsing

### The Bug
The original order book parsing code was incorrectly converting NO ask prices:
```python
# WRONG - was converting YES price to its complement
no_price = level[0]
yes_price = (100 - no_price) / 100.0  # Converts 33 to 67!
```

This caused our ASK at 0.33 to show as 0.67 in the next order book fetch.

### The Root Cause
**Kalshi's order book API returns ALL prices as YES prices (the `yes_price` field):**
- `yes` array: Contains YES orders with YES prices
- `no` array: Contains NO orders with **YES prices** (not NO prices!)

Example from API:
```json
{
  "orderbook": {
    "yes": [[27, 10]],  // Buying YES at 27 cents
    "no": [[33, 10]]    // Buying NO at yes_price=33 (selling YES at 33 cents)
  }
}
```

### The Fix
Simply use the price directly - it's already in YES terms:
```python
# CORRECT - price is already YES price
yes_price = level[0] / 100.0  // Convert cents to decimal, no complement needed!
```

This ensures:
- Our ASK at 0.33 → Kalshi stores as NO order with yes_price=33
- Next fetch shows `no: [[33, 10]]` → Parses as ASK at 0.33 ✓
- No more phantom price changes!

## Critical Bug Fix #2: Order Placement

### The Bug
The order placement code was also incorrectly converting ASK prices:
```python
# WRONG - converting YES price to complement when placing NO orders
if side == "ask":
    kalshi_side = "no"
    yes_price = int((1.0 - price) * 100)  # Converts 0.33 to 67!
```

This caused ASK orders at 0.33 to be placed at 0.67 in the Kalshi UI.

### The Fix
The `yes_price` parameter in Kalshi's order placement API is ALWAYS the YES price:
```python
# CORRECT - yes_price is always YES price, even for NO orders
if side == "ask":
    kalshi_side = "no"
    yes_price = int(price * 100)  # Keep it at 33!
```

### Key Insight
**Kalshi's API uses `yes_price` consistently everywhere:**
- Order book: `no: [[33, 10]]` means NO order at yes_price=33
- Order placement: `{"side": "no", "yes_price": 33}` means place NO order at yes_price=33
- Both mean: selling YES at $0.33 (buying NO at YES price of 33¢)

The `yes_price` field is not the NO price - it's literally the YES price, used across the entire API!
