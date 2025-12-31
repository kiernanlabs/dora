"""
AWS Lambda Handler for Dora Manager

This Lambda function supports multiple execution modes:
1. report: P&L monitoring and auto-deactivation (every 3 hours)
2. market_management: Combined market update + screener (every 12 hours)
3. execute_proposals: Proposal approval endpoint (API Gateway triggered)

Deployment:
- Runtime: Python 3.12
- Handler: handler.lambda_handler
- Timeout: 300 seconds (5 minutes for market_management mode)
- Memory: 512 MB
- Required IAM permissions:
  - dynamodb:* (all DynamoDB tables)
  - secretsmanager:GetSecretValue
  - ses:SendEmail

Environment variables:
- ENVIRONMENT: 'demo' or 'prod' (default: 'prod')
- AWS_REGION: AWS region (default: 'us-east-1')
- API_GATEWAY_BASE_URL: Base URL for API Gateway (for signed URLs)
- RECIPIENT_EMAIL: Email recipient (default: 'joe@kiernanlabs.com')
- MIN_PNL_THRESHOLD: P&L threshold for flagging markets (default: -3.0)
- WINDOW_HOURS: Hours for the reporting window (default: 3)
- DRY_RUN: If 'true', don't make changes (default: 'false')
"""
import json
import logging
import os
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler - routes to appropriate mode based on event source.

    Event sources:
    - EventBridge scheduled: mode from event payload
      - report: P&L monitoring (every 3 hours)
      - market_management: Combined update + screener (DEPRECATED - use split modes)
      - market_update_only: Just market updates (every 12 hours)
      - market_screener_only: Just market screener (every 12 hours, offset by 5 min)
      - send_proposals_email: Send combined email (every 12 hours, offset by 10 min)
    - API Gateway: execute_proposals (determined by presence of httpMethod)

    Args:
        event: Lambda event from EventBridge or API Gateway
        context: Lambda context

    Returns:
        Response with execution results
    """
    try:
        # Detect API Gateway invocation
        if 'httpMethod' in event:
            # API Gateway proxy integration - execute proposals
            logger.info("Detected API Gateway invocation - execute_proposals mode")
            from execute_proposals_handler import handle_execute_proposals
            return handle_execute_proposals(event, context)

        # EventBridge scheduled invocation
        mode = event.get('mode', 'report')
        logger.info(f"Detected EventBridge invocation - mode: {mode}")

        if mode == 'report':
            from report_handler import handle_report
            return handle_report(event, context)
        elif mode == 'market_management':
            # DEPRECATED: Use market_update_only + market_screener_only + send_proposals_email instead
            from market_management_handler import handle_market_management
            return handle_market_management(event, context)
        elif mode == 'market_update_only':
            from market_update_handler import handle_market_update_only
            return handle_market_update_only(event, context)
        elif mode == 'market_screener_only':
            from market_screener_handler import handle_market_screener_only
            return handle_market_screener_only(event, context)
        elif mode == 'send_proposals_email':
            from send_proposals_email_handler import handle_send_proposals_email
            return handle_send_proposals_email(event, context)
        else:
            logger.error(f"Invalid mode: {mode}")
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'success': False,
                    'error': f'Invalid mode: {mode}. Valid modes: report, market_update_only, market_screener_only, send_proposals_email'
                })
            }

    except Exception as e:
        logger.error(f"Error in lambda_handler routing: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }


# For local testing
if __name__ == "__main__":
    # Test with dry run
    test_event = {
        'environment': 'prod',
        'dry_run': 'true',
        'window_hours': 3,
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, default=str))
