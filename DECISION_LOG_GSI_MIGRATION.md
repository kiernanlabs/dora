# Decision Log GSI Migration

## Problem

The "Load Decision Context" button on the market_deep_dive page was taking over 5 minutes to load. This was caused by inefficient DynamoDB queries:

1. The `dora_decision_log` table only had indexes on `date` (partition key) and `timestamp` (sort key)
2. When querying for decisions for a specific market, the code had to:
   - Query all decisions for each date (potentially thousands across all markets)
   - Filter by `market_id` in memory
   - This resulted in full table scans with pagination

## Solution

Added a **Global Secondary Index (GSI)** on `market_id` + `timestamp` to enable efficient queries by market.

### Changes Made

1. **Table Schema** ([setup_dynamodb.py](kalshi/dora_bot/setup_dynamodb.py))
   - Added `market_id-timestamp-index` GSI to `dora_decision_log` table
   - This enables O(log n) lookups by market instead of O(n) scans

2. **DB Client Optimization** ([db_client.py](kalshi/dora_streamlit_ui/db_client.py))
   - Updated `get_decision_context_for_fill()` to use the GSI
   - Updated `get_recent_decision_logs()` to use the GSI when filtering by market
   - Added fallback logic for backwards compatibility if GSI doesn't exist yet

3. **Migration Script** ([add_decision_log_gsi.py](kalshi/dora_bot/add_decision_log_gsi.py))
   - Standalone script to add the GSI to existing tables
   - Supports both demo and prod environments

### Performance Improvement

**Before:**
- Full table scan with pagination across all markets
- 5+ minutes to load decision context
- O(n) where n = total decisions across all markets

**After:**
- Direct GSI query for specific market
- Expected: <1 second to load decision context
- O(log m) where m = decisions for one market

### Migration Instructions

#### Step 1: Check Current State (Dry Run)

```bash
cd /home/joe/dora/kalshi/dora_bot
python add_decision_log_gsi.py us-east-1 --env both --dry-run
```

This will check if the GSI already exists without making any changes.

#### Step 2: Run the Migration

```bash
# For demo environment only
python add_decision_log_gsi.py us-east-1 --env demo

# For prod environment only
python add_decision_log_gsi.py us-east-1 --env prod

# For both environments
python add_decision_log_gsi.py us-east-1 --env both
```

**Important Notes:**
- GSI creation can take **5-10 minutes** for large tables
- The table remains **online and available** during GSI creation
- The script will wait for the GSI to become ACTIVE before completing
- No data migration or downtime required

#### Step 3: Verify

After the migration completes:

1. Check AWS Console:
   - Navigate to DynamoDB → Tables → `dora_decision_log_demo` or `dora_decision_log_prod`
   - Go to "Indexes" tab
   - Verify `market_id-timestamp-index` exists and status is ACTIVE

2. Test in Streamlit:
   - Open the market_deep_dive page
   - Select a fill
   - Click "Load Decision Context"
   - Should load in <1 second (previously 5+ minutes)

### Backwards Compatibility

The code includes fallback logic, so it will work both with and without the GSI:

- **GSI exists**: Uses fast GSI query
- **GSI doesn't exist**: Falls back to old date-based query with in-memory filtering

This means:
- Safe to deploy code changes before running migration
- Safe to run migration while app is running
- No breaking changes

### Cost Impact

- GSI uses additional storage (duplicate of `market_id` + `timestamp` attributes)
- For typical usage: ~$1-5/month additional cost
- Significant reduction in read capacity units consumed (fewer scans = lower cost)
- **Net cost: likely neutral or reduced** due to query efficiency

### Rollback

If needed, you can remove the GSI:

```bash
aws dynamodb update-table \
  --table-name dora_decision_log_demo \
  --global-secondary-index-updates '[{"Delete": {"IndexName": "market_id-timestamp-index"}}]' \
  --region us-east-1
```

The code will automatically fall back to the old query method.

## Testing

To test the optimization without running the full migration:

```python
# Test with GSI (after migration)
from kalshi.dora_streamlit_ui.db_client import ReadOnlyDynamoDBClient
import time

client = ReadOnlyDynamoDBClient(region='us-east-1', environment='demo')

market_id = 'YOUR_MARKET_ID'
fill_timestamp = '2026-01-02T12:00:00Z'

start = time.time()
result = client.get_decision_context_for_fill(market_id, fill_timestamp, days=7)
elapsed = time.time() - start

print(f"Query took {elapsed:.2f} seconds")
print(f"Found decision: {result is not None}")
```

Expected results:
- **Before migration**: 5-60+ seconds
- **After migration**: <1 second
