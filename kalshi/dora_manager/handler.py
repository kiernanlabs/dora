"""
AWS Lambda Handler for Dora Manager

This Lambda function:
1. Connects to DynamoDB to fetch trading data
2. Calculates P&L and trade statistics for the last 3 hours
3. Flags and disables underperforming markets (P&L < -$3 in the window)
4. Sends a summary email via SES

Deployment:
- Runtime: Python 3.12
- Handler: handler.lambda_handler
- Timeout: 60 seconds
- Memory: 256 MB
- Required IAM permissions:
  - dynamodb:GetItem, dynamodb:Query, dynamodb:Scan, dynamodb:UpdateItem
  - ses:SendEmail

Environment variables:
- ENVIRONMENT: 'demo' or 'prod' (default: 'prod')
- AWS_REGION: AWS region (default: 'us-east-1')
- RECIPIENT_EMAIL: Email recipient (default: 'joey32@gmail.com')
- MIN_PNL_THRESHOLD: P&L threshold for flagging markets (default: -3.0)
- WINDOW_HOURS: Hours for the reporting window (default: 3)
- DRY_RUN: If 'true', don't disable markets or send emails (default: 'false')
"""
import json
import logging
import os
from typing import Dict, Any

from .db_client import DynamoDBClient
from .calculator import TradingCalculator, TradingSummary
from .email_sender import EmailSender

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for Dora Manager.

    Args:
        event: Lambda event (can include overrides for environment variables)
        context: Lambda context

    Returns:
        Response with summary data
    """
    # Get configuration from environment variables or event overrides
    environment = event.get('environment') or os.environ.get('ENVIRONMENT', 'prod')
    region = event.get('region') or os.environ.get('AWS_REGION', 'us-east-1')
    recipient_email = event.get('recipient_email') or os.environ.get('RECIPIENT_EMAIL', 'joe@kiernanlabs.com')
    min_pnl_threshold = float(event.get('min_pnl_threshold') or os.environ.get('MIN_PNL_THRESHOLD', '-3.0'))
    window_hours = int(event.get('window_hours') or os.environ.get('WINDOW_HOURS', '3'))
    dry_run = str(event.get('dry_run') or os.environ.get('DRY_RUN', 'false')).lower() == 'true'

    logger.info(f"Starting Dora Manager - Environment: {environment}, Region: {region}")
    logger.info(f"Window: {window_hours}hrs, Min P&L Threshold: ${min_pnl_threshold}, Dry Run: {dry_run}")

    try:
        # Initialize clients
        db_client = DynamoDBClient(region=region, environment=environment)
        email_sender = EmailSender(region=region)

        # Fetch data from DynamoDB
        logger.info("Fetching data from DynamoDB...")

        positions = db_client.get_positions()
        logger.info(f"Fetched {len(positions)} positions")

        market_configs = db_client.get_all_market_configs(enabled_only=True)
        logger.info(f"Fetched {len(market_configs)} enabled market configs")

        all_trades = db_client.get_all_trades(days=30)
        logger.info(f"Fetched {len(all_trades)} total trades (30 days)")

        window_trades = db_client.get_trades_in_window(hours=window_hours)
        logger.info(f"Fetched {len(window_trades)} trades in {window_hours}hr window")

        decisions_by_market = db_client.get_most_recent_decision_by_market()
        logger.info(f"Fetched decisions for {len(decisions_by_market)} markets")

        open_orders_by_market = db_client.get_open_orders_by_market()
        logger.info(f"Fetched open orders for {len(open_orders_by_market)} markets")

        # Calculate summary
        logger.info("Calculating trading summary...")
        calculator = TradingCalculator(
            positions=positions,
            market_configs=market_configs,
            all_trades=all_trades,
            window_trades=window_trades,
            decisions_by_market=decisions_by_market,
            open_orders_by_market=open_orders_by_market,
            window_hours=window_hours,
            min_pnl_threshold=min_pnl_threshold,
        )

        summary = calculator.calculate_summary()
        summary.environment = environment

        logger.info(f"Summary calculated:")
        logger.info(f"  - Total P&L (Window): ${summary.total_realized_pnl_window:.2f}")
        logger.info(f"  - Total P&L (All Time): ${summary.total_realized_pnl_all_time:.2f}")
        logger.info(f"  - Trades in Window: {summary.total_trade_count}")
        logger.info(f"  - Markets with Trades: {summary.markets_with_trades}")
        logger.info(f"  - Markets Flagged: {summary.markets_flagged_count}")

        # Disable flagged markets
        disabled_markets = []
        if summary.flagged_markets:
            logger.info(f"Processing {len(summary.flagged_markets)} flagged markets...")
            for market in summary.flagged_markets:
                if dry_run:
                    logger.info(f"[DRY RUN] Would disable market: {market.market_id}")
                    disabled_markets.append({
                        'market_id': market.market_id,
                        'reason': market.deactivation_reason,
                        'dry_run': True
                    })
                else:
                    success = db_client.disable_market(
                        market_id=market.market_id,
                        reason=market.deactivation_reason
                    )
                    disabled_markets.append({
                        'market_id': market.market_id,
                        'reason': market.deactivation_reason,
                        'success': success
                    })
                    if success:
                        logger.info(f"Disabled market: {market.market_id}")
                    else:
                        logger.error(f"Failed to disable market: {market.market_id}")

        # Send email report
        email_sent = False
        if dry_run:
            logger.info(f"[DRY RUN] Would send email to: {recipient_email}")
        else:
            logger.info(f"Sending email report to: {recipient_email}")
            email_sent = email_sender.send_report(
                summary=summary,
                environment=environment,
                recipient=recipient_email
            )

        # Build response
        response = {
            'statusCode': 200,
            'body': {
                'success': True,
                'environment': environment,
                'window_hours': window_hours,
                'report_timestamp': summary.report_timestamp,
                'summary': {
                    'realized_pnl_window': summary.total_realized_pnl_window,
                    'realized_pnl_all_time': summary.total_realized_pnl_all_time,
                    'unrealized_pnl_worst': summary.total_unrealized_pnl_worst,
                    'unrealized_pnl_best': summary.total_unrealized_pnl_best,
                    'total_trade_count': summary.total_trade_count,
                    'total_contracts_traded': summary.total_contracts_traded,
                    'total_fees_paid': summary.total_fees_paid,
                    'total_exposure': summary.total_exposure,
                    'active_markets_count': summary.active_markets_count,
                    'markets_with_trades': summary.markets_with_trades,
                    'markets_flagged_count': summary.markets_flagged_count,
                },
                'disabled_markets': disabled_markets,
                'email_sent': email_sent,
                'dry_run': dry_run,
            }
        }

        logger.info(f"Dora Manager completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in Dora Manager: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'error': str(e)
            }
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
