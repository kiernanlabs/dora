# Dora Manager Migration Plan

## Executive Summary

This plan outlines the migration of `market_screener.py` and `market_update.py` functionality into the `dora_manager` Lambda function, creating a unified market management system with approval-based automation.

### Key Changes

1. **Consolidated Execution**: Two EventBridge-triggered modes
   - `report` mode: Every 3 hours (P&L monitoring, auto-deactivation)
   - `market_management` mode: Every 12 hours (combined update + screener)

2. **Approval Flow**: API Gateway + signed URLs
   - Proposals saved to new DynamoDB table
   - Email sent with approval link (12-hour expiry)
   - User reviews and approves via web interface
   - Execution endpoint applies approved changes
   - Confirmation email sent with results

3. **Top-Level Decisions**:
   - ✅ Combine market_update and market_screener into single trigger
   - ✅ Top 20 candidates from screener (by volume)
   - ✅ 12-hour approval window with auto-reject
   - ✅ Confirmation email after execution
   - ✅ Continue on partial failures, report in email
   - ✅ API Gateway approval flow (Option 1)

---

## Overview

Migrate `market_screener.py` and `market_update.py` functionality into the `dora_manager` Lambda function with two distinct execution modes triggered by EventBridge schedules.

## Current State

### Existing Scripts
1. **market_screener.py** - Discovers new candidate markets from Kalshi API
   - Filters by volume, spread, close time, restricted prefixes
   - Assesses information risk via OpenAI
   - Checks side volume from trade history
   - Generates CSV for manual review
   - Uploads approved markets to market_config table

2. **market_update.py** - Analyzes existing market performance
   - Calculates P&L and fill metrics per market
   - Generates recommendations (exit, scale_down, expand, reset_defaults)
   - Activates sibling markets for expanding events
   - Assesses info risk for active/sibling markets
   - Generates CSV for manual review
   - Applies approved updates to market_config table

3. **handler.py** - Current Lambda (runs on demand)
   - Fetches trading data from DynamoDB
   - Calculates P&L summary for window (3hrs)
   - Disables markets below P&L threshold
   - Sends email report via SES

## Target Architecture

### Execution Modes

The Lambda will support 3 execution modes, distinguished by the `mode` field in the event payload. **Each mode is a separate Lambda invocation** - the Lambda does not stay running between proposal generation and execution.

#### Mode 1: `report` (Every 3 hours)
- **Trigger**: EventBridge rule every 3 hours
- **Invocation Type**: Scheduled
- **Function**: Current handler.py functionality
- **Actions**:
  - Calculate P&L and trading stats for window
  - Flag and disable underperforming markets (P&L < threshold)
  - Send email report via SES
  - **Lambda exits immediately after sending email**
- **No approval required** - executes immediately

#### Mode 2: `market_management` (Every 12 hours)
- **Trigger**: EventBridge rule every 12 hours
- **Invocation Type**: Scheduled
- **Function**: Combined market_update.py + market_screener.py functionality
- **Actions**:
  - **Part A: Market Update**
    - Analyze existing market performance
    - Generate recommendations (exit, scale_down, expand, activate_sibling)
    - Assess information risk for active markets
  - **Part B: Market Screener**
    - Fetch all open markets from Kalshi API
    - Apply filters (volume, spread, close time, info risk, side volume)
    - Generate candidate list (top 20 by volume)
  - **Combined Output**:
    - Save all proposals to `dora_market_proposals` DynamoDB table with single proposal_id
    - Send single email with both update recommendations and new candidates
    - Email sent to joe@kiernanlabs.com
  - **Lambda exits after sending email** - does NOT wait for approval
- **Approval required** - user clicks link in email to trigger execution
- **Auto-reject after 12 hours** if no action taken (via DynamoDB TTL)

#### Mode 3: `execute_proposals` (On-demand via API Gateway)
- **Trigger**: User clicks approval link in email → API Gateway → Lambda
- **Invocation Type**: API Gateway proxy integration
- **Function**: Execute approved proposals
- **Actions**:
  - Validate signature and expiry from URL parameters
  - Query proposals from DynamoDB by proposal_id
  - Filter by status=pending
  - Execute each approved proposal (update/create market_config entries)
  - Update proposal status to executed/failed
  - Send confirmation email with results
  - **Lambda exits after sending confirmation**
- **This is a completely separate invocation** from Mode 2

---

## New DynamoDB Table: `dora_market_proposals`

### Schema

```
Partition Key: proposal_id (String)   # UUID
Sort Key: market_id (String)          # Market ticker

Attributes:
- proposal_id: UUID for the proposal batch (same for all proposals in one run)
- market_id: Market ticker
- proposal_source: "market_update" | "market_screener"
- action: "exit" | "scale_down" | "expand" | "activate_sibling" | "reset_defaults" | "new_market"
- status: "pending" | "approved" | "rejected" | "executed" | "failed"
- created_at: ISO timestamp
- approved_at: ISO timestamp (nullable)
- executed_at: ISO timestamp (nullable)
- environment: "demo" | "prod"

# Proposed changes (from CSV columns)
- new_enabled: Boolean (nullable)
- new_min_spread: Decimal (nullable)
- new_quote_size: Integer (nullable)
- new_max_inventory_yes: Integer (nullable)
- new_max_inventory_no: Integer (nullable)
- fair_value: Decimal (nullable)
- fair_value_rationale: String (nullable)

# Context for review
- reason: String (why this action is recommended)
- event_ticker: String (nullable)
- event_name: String (nullable)
- pnl_24h: Decimal (for market_update)
- fill_count_24h: Integer (for market_update)
- fill_count_48h: Integer (for market_update)
- volume_24h: Integer (for market_screener)
- buy_volume: Integer (for market_screener)
- sell_volume: Integer (for market_screener)
- current_bid: Integer (nullable)
- current_ask: Integer (nullable)
- has_position: Boolean
- position_qty: Integer
- info_risk_probability: Decimal (nullable)
- info_risk_rationale: String (nullable)

# For retrieval
- approval_url: String (URL to approve/reject this proposal)
- GSI: status-created_at-index (for querying pending proposals)
```

### GSI: `status-created_at-index`
- Partition Key: status
- Sort Key: created_at
- Allows efficient querying of all pending proposals

### TTL Configuration
- TTL attribute: `ttl_timestamp`
- Set to `created_at + 12 hours`
- Auto-rejects pending proposals after 12 hours
- DynamoDB will automatically delete rejected proposals after 7 days

---

## User Approval Mechanism

### Selected Approach: API Gateway + URL Links

**Architecture:**
```
┌─ INVOCATION 1: Proposal Generation (EventBridge scheduled) ─┐
│                                                               │
│  EventBridge (12hr) → Lambda (market_management mode)        │
│                          ├─ Analyze markets                  │
│                          ├─ Generate proposals               │
│                          ├─ Save to DynamoDB                 │
│                          ├─ Send email with signed URL       │
│                          └─ EXIT                             │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ↓
                         (Lambda exits)
                              ↓
                    Proposals stored in DynamoDB
                    Email sent with approval link
                              ↓
                    ⏰ User has 12 hours to respond
                              ↓
┌─ INVOCATION 2: Execution (API Gateway triggered) ────────────┐
│                                                               │
│  User clicks URL → API Gateway → Lambda (execute_proposals)  │
│                                      ├─ Validate signature   │
│                                      ├─ Query DynamoDB       │
│                                      ├─ Execute changes      │
│                                      ├─ Update statuses      │
│                                      ├─ Send confirmation    │
│                                      └─ EXIT                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Key Point**: These are two **completely separate Lambda invocations**. The Lambda does not stay running waiting for approval (which would be impossible given Lambda's 15-minute max runtime).

**Key Benefits:**
- Simple and reliable - no dependency on email parsing
- Rich approval UI - HTML page showing all proposal details
- Supports partial approval - approve some, reject others individually
- Secure - HMAC-signed URLs with 12-hour expiry
- Works on any device with a browser
- Easy to extend with additional features

**Implementation Details:**
1. Create API Gateway REST API with two endpoints:
   - `GET /proposals/{proposal_id}` - View proposal details (HTML page)
   - `POST /proposals/{proposal_id}/execute` - Execute approved proposals

2. Generate signed URLs with proposal_id and HMAC signature
   - Include signature and expiry in URL params
   - Lambda validates signature before execution
   - **12-hour expiry** - proposals auto-reject after this time

3. Email contains:
   - Summary section showing counts (updates, new markets, expansions, etc.)
   - Combined table of all proposals (both market_update and market_screener)
   - "Review & Approve" button → HTML page showing full details
   - "Approve All" button → Direct POST to execute endpoint
   - "Reject All" button → Updates status to rejected
   - Expiry notice: "This link expires in 12 hours"

4. HTML approval page allows:
   - Review each proposal individually with full context
   - Edit proposed values before approval (quote_size, max_inventory, min_spread)
   - Select which proposals to approve/reject via checkboxes
   - Submit → Lambda executes only approved items
   - Shows separate sections for market updates vs new candidates

5. Execution confirmation email sent after approval:
   - Summary of execution results
   - Success count and failure count
   - Details of each executed proposal (status, market_id, action)
   - Error messages for any failed executions
   - Partial failures handled gracefully - continue with remaining proposals

**Security:**
- Use HMAC-SHA256 signature with secret key stored in AWS Secrets Manager
- Include expiry timestamp (12 hours from creation)
- Lambda validates signature and expiry before execution
- Constant-time comparison to prevent timing attacks

---

## Implementation Plan

### Phase 1: New Infrastructure

1. **Create DynamoDB table**: `dora_market_proposals_{env}`
   - Define schema as above
   - Create GSI for status-created_at queries
   - Set TTL attribute `ttl_timestamp` = `created_at + 12 hours`
   - Auto-reject pending proposals after 12 hours
   - DynamoDB auto-deletes expired records after 48 hours

2. **Create API Gateway REST API**: `dora-proposals-api`
   - Endpoint 1: `GET /proposals/{proposal_id}` (HTML approval page)
   - Endpoint 2: `POST /proposals/{proposal_id}/execute` (execute proposals)
   - Use Lambda proxy integration
   - **Both endpoints invoke the same Lambda function** with `mode=execute_proposals`
   - Lambda determines whether to render HTML (GET) or execute (POST) based on HTTP method
   - API Gateway passes proposal_id, signature, and expiry as query parameters/body

3. **Update Lambda IAM role** to include:
   - DynamoDB permissions for new proposals table
   - API Gateway invoke permissions (for URL generation)

### Phase 2: Refactor Handler

1. **Update `handler.py`** to support multiple execution modes:
   ```python
   def lambda_handler(event, context):
       """
       Main Lambda handler - routes to appropriate mode based on event source.

       Event sources:
       - EventBridge scheduled: mode from event payload (report or market_management)
       - API Gateway: mode=execute_proposals (determined by presence of httpMethod)
       """
       # Detect API Gateway invocation
       if 'httpMethod' in event:
           # API Gateway proxy integration - execute proposals
           return handle_execute_proposals_mode(event, context)

       # EventBridge scheduled invocation
       mode = event.get('mode', 'report')

       if mode == 'report':
           return handle_report_mode(event, context)
       elif mode == 'market_management':
           return handle_market_management_mode(event, context)
       else:
           return {'statusCode': 400, 'body': 'Invalid mode'}
   ```

2. **Extract shared utilities** into modules:
   - `utils/kalshi_client.py` - API calls to Kalshi
   - `utils/openai_client.py` - Information risk assessment
   - `utils/proposal_manager.py` - DynamoDB operations for proposals
   - `utils/url_signer.py` - HMAC signature generation/validation

3. **Create `market_management_handler.py`** - Combined screener + update:
   - **Part A: Market Update Logic**
     - Keep all analysis and recommendation logic
     - Generate recommendations (exit, scale_down, expand, activate_sibling)
     - Assess info risk for active markets
   - **Part B: Market Screener Logic**
     - Keep all filtering logic (volume, spread, close time, info risk, side volume)
     - Generate top 20 candidates by volume
   - **Combined Output**:
     - Single proposal_id for the entire batch
     - Save all proposals to DynamoDB (both updates and new candidates)
     - Generate single email with both sections
     - Replace interactive approval with signed URL generation

### Phase 3: Email & Approval System

1. **Create email templates** for proposals:
   - `templates/market_proposals.html` - Single combined HTML email template
     - Section 1: Summary (counts of updates, new markets, expansions, etc.)
     - Section 2: Market Updates table (exit, scale_down, expand, activate_sibling)
     - Section 3: New Market Candidates table (top 20 by volume)
     - Include "Review & Approve" button with signed URL
     - Include "Approve All" and "Reject All" buttons
     - Expiry notice: "This link expires in 12 hours"
   - `templates/confirmation_email.html` - Execution confirmation template
     - Summary of results (success count, failure count)
     - Table of all executed proposals with status
     - Error details for any failures
     - Timestamp of execution

2. **Create HTML approval page** (`templates/approval_page.html`):
   - Display all proposals in two sections (updates vs new candidates)
   - Show full context for each proposal (P&L, volume, info risk, etc.)
   - Allow individual selection via checkboxes
   - Allow editing values before approval (quote_size, max_inventory, min_spread)
   - "Approve Selected" button → POST to execute endpoint
   - Show expiry countdown timer

3. **Implement execute endpoint** (`handle_execute_proposals_mode`):
   - **For GET requests** (Review page):
     - Validate signature and expiry
     - Query proposals from DynamoDB by proposal_id
     - Render HTML approval page with proposal details
     - Allow user to select which proposals to approve
     - Return HTTP 200 with HTML content
   - **For POST requests** (Execute):
     - Validate signature and expiry (reject if > 12 hours old)
     - Parse request body to get selected proposal IDs or "approve_all" flag
     - Query proposals from DynamoDB by proposal_id
     - Filter by status=pending and user selection
     - Execute each approved proposal:
       - For market_update: Apply updates to existing market_config
       - For market_screener: Create new market_config entries
     - **Partial failure handling**: Continue with remaining proposals if some fail
     - Update proposal status to executed/failed individually
     - Send confirmation email with success/failure summary
     - Return HTTP 200 with success message

### Phase 4: EventBridge Rules

1. **Create 2 EventBridge rules**:
   - Rule 1: `dora-report-3hr` - Trigger every 3 hours with `mode=report`
     - Schedule expression: `rate(3 hours)`
   - Rule 2: `dora-market-management-12hr` - Trigger every 12 hours with `mode=market_management`
     - Schedule expression: `rate(12 hours)`

2. **Example EventBridge event payloads**:
   ```json
   // Report mode (every 3 hours)
   {
     "mode": "report",
     "environment": "prod",
     "window_hours": 3,
     "min_pnl_threshold": -3.0
   }

   // Market management mode (every 12 hours) - Combined update + screener
   {
     "mode": "market_management",
     "environment": "prod",
     "pnl_lookback_hours": 24,
     "volume_lookback_hours": 48,
     "top_n_candidates": 20,
     "skip_info_risk": false
   }
   ```

### Phase 5: Testing & Deployment

1. **Local testing**:
   - Test each mode with sample events
   - Test proposal creation and storage
   - Test email generation (with dry_run=true)
   - Test URL signature generation/validation
   - Test execute endpoint with mock proposals

2. **Deploy to demo environment**:
   - Deploy Lambda with all dependencies
   - Create DynamoDB table in demo
   - Create API Gateway in demo
   - Set up EventBridge rules (disabled initially)
   - Manual testing with real data

3. **Deploy to prod**:
   - Deploy Lambda to prod
   - Create DynamoDB table in prod
   - Create API Gateway in prod
   - Enable EventBridge rules with appropriate schedules

---

## Migration Timeline

### Week 1: Infrastructure & Core Refactoring
- Day 1-2: Create DynamoDB table, API Gateway endpoints
- Day 3-4: Refactor handler.py to support multiple modes
- Day 5: Extract shared utilities

### Week 2: Port Screener & Update Logic
- Day 1-3: Port market_screener.py functionality
- Day 4-5: Port market_update.py functionality

### Week 3: Approval System & Testing
- Day 1-2: Implement email templates and approval UI
- Day 3: Implement execute endpoint
- Day 4-5: Testing (unit, integration, end-to-end)

### Week 4: Deployment & Monitoring
- Day 1-2: Deploy to demo, test with real data
- Day 3: Deploy to prod, enable EventBridge rules
- Day 4-5: Monitor first runs, fix any issues

---

## Security Considerations

1. **Signed URLs**:
   - Use HMAC-SHA256 with 256-bit secret key
   - Store secret in AWS Secrets Manager
   - Include expiry timestamp (48 hours default)
   - Include proposal_id in signature to prevent tampering

2. **API Gateway**:
   - Enable throttling (10 requests/second per IP)
   - Enable CloudWatch logging
   - Consider adding WAF rules for additional protection

3. **DynamoDB**:
   - Enable point-in-time recovery
   - Enable encryption at rest
   - Use least-privilege IAM permissions

4. **Email**:
   - Use SES DKIM signing
   - Include disclaimer about not sharing URLs
   - Consider adding user-specific tokens for audit trail

---

## Rollback Plan

If issues arise post-deployment:

1. **Disable EventBridge rules** for market_update and market_screener modes
2. **Keep report mode running** (existing functionality)
3. **Fall back to manual execution** of market_screener.py and market_update.py scripts
4. **Investigate and fix issues** in Lambda
5. **Re-enable EventBridge rules** once fixed

---

## Future Enhancements

1. **Web Dashboard**:
   - Build a simple web UI to view historical proposals
   - Show execution history and results
   - Allow manual triggering of screener/update

2. **Slack Integration**:
   - Send proposal notifications to Slack channel
   - Use Slack interactive components for approval
   - Better for team collaboration

3. **Machine Learning**:
   - Train model to predict which proposals user will approve
   - Auto-approve high-confidence proposals
   - Flag unusual proposals for manual review

4. **A/B Testing**:
   - Test different market config strategies
   - Compare performance across different approaches
   - Automatically optimize parameters

---

## Design Decisions (Confirmed)

1. **Cadence**: ✓
   - Report mode: Every 3 hours
   - Market management mode: Every 12 hours (combined update + screener)

2. **Top N markets for screener**: ✓
   - Top 20 candidates by volume

3. **Approval expiry**: ✓
   - 12 hours from creation
   - Auto-reject after expiry

4. **Error handling**: ✓
   - Continue with remaining proposals if some fail
   - Report successes and failures in confirmation email

5. **Execution confirmation**: ✓
   - Yes, send confirmation email after execution
   - Include success count, failure count, and details

## Remaining Open Questions

1. **API Gateway domain**:
   - Should we use a custom domain for the approval API?
   - Or use the default API Gateway URL?

2. **Monitoring & Alerts**:
   - Should we set up CloudWatch alarms for failed executions?
   - SNS notifications for proposal generation failures?

3. **Environment separation**:
   - Should demo and prod have separate API Gateway endpoints?
   - Or use the same endpoint with environment in the URL?

4. **Rate limiting**:
   - What throttle limits should we set on the API Gateway endpoints?
   - 10 requests/second per IP reasonable?

---

## Appendix A: URL Signature Algorithm

```python
import hmac
import hashlib
import time
from urllib.parse import urlencode

def generate_signed_url(proposal_id: str, secret_key: str, ttl_hours: int = 12) -> str:
    """Generate a signed URL for proposal approval."""
    expiry = int(time.time()) + (ttl_hours * 3600)

    # Create signature payload
    payload = f"{proposal_id}:{expiry}"
    signature = hmac.new(
        secret_key.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    # Build URL
    base_url = "https://api.example.com/proposals"
    params = urlencode({
        'proposal_id': proposal_id,
        'expiry': expiry,
        'signature': signature
    })

    return f"{base_url}/{proposal_id}?{params}"

def validate_signature(proposal_id: str, expiry: int, signature: str, secret_key: str) -> bool:
    """Validate a signed URL."""
    # Check expiry
    if time.time() > expiry:
        return False

    # Recompute signature
    payload = f"{proposal_id}:{expiry}"
    expected_signature = hmac.new(
        secret_key.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(signature, expected_signature)
```

---

## Appendix B: Email Template Example

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .header { background-color: #4CAF50; color: white; padding: 20px; }
        .summary { background-color: #f9f9f9; padding: 15px; margin: 20px 0; }
        .section-header {
            background-color: #2196F3;
            color: white;
            padding: 10px;
            margin: 20px 0 10px 0;
        }
        .proposal-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        .proposal-table th, .proposal-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .proposal-table th { background-color: #4CAF50; color: white; }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            text-decoration: none;
            display: inline-block;
            margin: 10px 5px;
            border-radius: 4px;
        }
        .button-reject { background-color: #f44336; }
        .expiry-notice {
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>DORA Market Management Proposals</h1>
        <p>Environment: PROD | Generated: 2025-12-27 10:00:00 UTC</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li><strong>Total Proposals: 25</strong></li>
            <li>Market Updates: 15
                <ul>
                    <li>Expansions: 5</li>
                    <li>Scale Downs: 3</li>
                    <li>Exits: 2</li>
                    <li>Sibling Activations: 5</li>
                </ul>
            </li>
            <li>New Market Candidates: 10 (top 20 by volume)</li>
        </ul>
    </div>

    <div class="expiry-notice">
        ⏰ <strong>Important:</strong> This approval link expires in 12 hours.
        Proposals will auto-reject at 2025-12-27 22:00:00 UTC.
    </div>

    <!-- Market Updates Section -->
    <div class="section-header">
        <h2>📊 Market Updates (Existing Markets)</h2>
    </div>
    <table class="proposal-table">
        <thead>
            <tr>
                <th>Market ID</th>
                <th>Action</th>
                <th>Reason</th>
                <th>P&L (24h)</th>
                <th>Fills (48h)</th>
                <th>New Quote Size</th>
                <th>Info Risk</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>KXYZ-25</td>
                <td>expand</td>
                <td>Positive P&L, median fill = quote size</td>
                <td style="color: green;">$12.50</td>
                <td>45</td>
                <td>10 → 20</td>
                <td>15%</td>
            </tr>
            <tr>
                <td>KABC-30</td>
                <td>scale_down</td>
                <td>P&L < $0 (-$2.30)</td>
                <td style="color: red;">-$2.30</td>
                <td>12</td>
                <td>10 → 5</td>
                <td>18%</td>
            </tr>
            <!-- More rows... -->
        </tbody>
    </table>

    <!-- New Candidates Section -->
    <div class="section-header">
        <h2>🆕 New Market Candidates (From Screener)</h2>
    </div>
    <table class="proposal-table">
        <thead>
            <tr>
                <th>Market ID</th>
                <th>Event Name</th>
                <th>Volume (24h)</th>
                <th>Bid/Ask</th>
                <th>Quote Size</th>
                <th>Info Risk</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>KNEW-01</td>
                <td>2025 NFL Super Bowl</td>
                <td>5,432</td>
                <td>45/50</td>
                <td>5</td>
                <td>12%</td>
            </tr>
            <tr>
                <td>KNEW-02</td>
                <td>Bitcoin Price Jan 2025</td>
                <td>3,210</td>
                <td>30/35</td>
                <td>5</td>
                <td>20%</td>
            </tr>
            <!-- More rows... -->
        </tbody>
    </table>

    <div style="text-align: center; margin: 30px 0;">
        <a href="{{review_url}}" class="button">Review & Approve</a>
        <a href="{{approve_all_url}}" class="button">Approve All</a>
        <a href="{{reject_all_url}}" class="button button-reject">Reject All</a>
    </div>

    <p style="color: #666; font-size: 12px;">
        Do not share this link with others. For security, links are single-use and expire after 12 hours.
    </p>
</body>
</html>
```
