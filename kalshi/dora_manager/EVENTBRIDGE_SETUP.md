# EventBridge Rules Setup for Split Lambda Architecture

This document outlines how to configure EventBridge rules for the split market management architecture.

## Architecture Overview

Instead of running both market_update and market_screener in a single Lambda invocation (which causes timeouts), we now split them into 3 separate invocations:

1. **market_update_only** - Analyzes existing markets, saves proposals
2. **market_screener_only** - Screens new candidates, saves proposals
3. **send_proposals_email** - Queries all recent proposals and sends combined email

## EventBridge Rules Configuration

### Rule 1: Market Update (Every 12 hours at :00)

```bash
aws events put-rule \
  --name dora-market-update-12hr \
  --description "Trigger market update analysis every 12 hours" \
  --schedule-expression "cron(0 */12 * * ? *)" \
  --state ENABLED \
  --region us-east-1

aws events put-targets \
  --rule dora-market-update-12hr \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:dora-manager","Input"='{"mode":"market_update_only","environment":"prod","skip_info_risk":false}' \
  --region us-east-1
```

**Schedule**: Every 12 hours at 00:00 and 12:00 UTC
**Runtime**: ~2-4 minutes (with info risk assessment)

---

### Rule 2: Market Screener (Every 12 hours at :05)

```bash
aws events put-rule \
  --name dora-market-screener-12hr \
  --description "Trigger market screener every 12 hours (5 min offset)" \
  --schedule-expression "cron(5 */12 * * ? *)" \
  --state ENABLED \
  --region us-east-1

aws events put-targets \
  --rule dora-market-screener-12hr \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:dora-manager","Input"='{"mode":"market_screener_only","environment":"prod","top_n_candidates":20,"skip_info_risk":false}' \
  --region us-east-1
```

**Schedule**: Every 12 hours at 00:05 and 12:05 UTC (5 min after market_update)
**Runtime**: ~2-4 minutes (with info risk assessment)

---

### Rule 3: Send Combined Email (Every 12 hours at :10)

```bash
aws events put-rule \
  --name dora-send-proposals-email-12hr \
  --description "Send combined proposals email every 12 hours (10 min offset)" \
  --schedule-expression "cron(10 */12 * * ? *)" \
  --state ENABLED \
  --region us-east-1

aws events put-targets \
  --rule dora-send-proposals-email-12hr \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:dora-manager","Input"='{"mode":"send_proposals_email","environment":"prod","lookback_minutes":15}' \
  --region us-east-1
```

**Schedule**: Every 12 hours at 00:10 and 12:10 UTC (10 min after market_update, 5 min after screener)
**Runtime**: <30 seconds (just queries DynamoDB and sends email)
**Lookback**: 15 minutes (captures proposals from both update and screener)

---

### Rule 4: Report Mode (Every 3 hours) - UNCHANGED

```bash
aws events put-rule \
  --name dora-report-3hr \
  --description "Trigger P&L report every 3 hours" \
  --schedule-expression "rate(3 hours)" \
  --state ENABLED \
  --region us-east-1

aws events put-targets \
  --rule dora-report-3hr \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:dora-manager","Input"='{"mode":"report","environment":"prod","window_hours":3,"min_pnl_threshold":-3.0}' \
  --region us-east-1
```

**Schedule**: Every 3 hours
**Runtime**: <1 minute

---

## Timeline Visualization

```
00:00 UTC - market_update_only starts (2-4 min runtime)
00:05 UTC - market_screener_only starts (2-4 min runtime)
00:10 UTC - send_proposals_email starts (queries last 15 min, sends combined email)

12:00 UTC - market_update_only starts
12:05 UTC - market_screener_only starts
12:10 UTC - send_proposals_email starts
```

---

## Benefits of Split Architecture

### ✅ Pros:
1. **No timeout issues** - Each Lambda runs independently with its own 10-minute timeout
2. **Parallel execution** - Update and screener can run simultaneously (offset by 5 min)
3. **Better error handling** - If one fails, the other still completes
4. **Single combined email** - User gets one email with all proposals at :10

### ⚠️ Considerations:
1. **More EventBridge rules** - 3 rules instead of 1 for market management
2. **Coordination via DynamoDB** - Proposals are coordinated through database, not in-memory
3. **Timing dependency** - Email sender must run after both others complete (15 min lookback ensures this)

---

## Monitoring

Check CloudWatch Logs for each mode:

```bash
# Market Update logs
aws logs tail /aws/lambda/dora-manager --since 10m --filter-pattern "market_update_only"

# Market Screener logs
aws logs tail /aws/lambda/dora-manager --since 10m --filter-pattern "market_screener_only"

# Email sender logs
aws logs tail /aws/lambda/dora-manager --since 10m --filter-pattern "send_proposals_email"
```

---

## Testing Individually

Test each mode separately:

```bash
# Test market update
aws lambda invoke \
  --function-name dora-manager \
  --payload '{"mode":"market_update_only","environment":"demo","skip_info_risk":true}' \
  response.json

# Test market screener
aws lambda invoke \
  --function-name dora-manager \
  --payload '{"mode":"market_screener_only","environment":"demo","top_n_candidates":5,"skip_info_risk":true}' \
  response.json

# Test email sender (after running the above two)
aws lambda invoke \
  --function-name dora-manager \
  --payload '{"mode":"send_proposals_email","environment":"demo","lookback_minutes":60}' \
  response.json
```

---

## Migration from Old Architecture

The old `market_management` mode is still available but **DEPRECATED**. To migrate:

1. Create the 3 new EventBridge rules (above)
2. Test in demo environment first
3. Disable the old `market_management` EventBridge rule
4. Monitor the new split architecture for one full cycle
5. Once stable, delete the old EventBridge rule

---

## Rollback Plan

If issues occur:
1. Disable the 3 new EventBridge rules
2. Re-enable the old `market_management` rule
3. Investigate and fix issues
4. Re-test the split architecture
