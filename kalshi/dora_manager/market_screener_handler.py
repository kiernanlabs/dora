"""
Market Screener Handler for Dora Manager

Screens new market candidates from Kalshi API.
Saves proposals to DynamoDB but does NOT send email.
Email is sent by send_proposals_email_handler after both update and screener complete.
"""
import logging
import os
from typing import Dict, Any
from datetime import datetime, timezone

# Import from local modules
from market_screener import (
    fetch_all_markets,
    filter_and_score_markets,
)
from db_client import DynamoDBClient

# Import utilities
from utils.proposal_manager import ProposalManager

logger = logging.getLogger(__name__)


def handle_market_screener_only(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle market_screener_only mode execution.

    Screens new market candidates and saves proposals to DynamoDB.
    Does NOT send email - that's handled by send_proposals_email_handler.

    Args:
        event: Lambda event with configuration:
            - environment: 'demo' or 'prod'
            - top_n_candidates: Number of top candidates from screener (default: 20)
            - skip_info_risk: Skip OpenAI info risk assessment (default: false)
        context: Lambda context

    Returns:
        Response with proposal_id and summary data
    """
    # Get configuration
    environment = event.get('environment') or os.environ.get('ENVIRONMENT', 'prod')
    region = event.get('region') or os.environ.get('AWS_REGION', 'us-east-1')

    top_n_candidates = int(event.get('top_n_candidates', 20))
    skip_info_risk = event.get('skip_info_risk', False)

    logger.info(f"Starting Market Screener Only Mode - Environment: {environment}")
    logger.info(f"Top N candidates: {top_n_candidates}, Skip info risk: {skip_info_risk}")

    try:
        # Initialize clients
        db_client = DynamoDBClient(region=region, environment=environment)
        proposal_manager = ProposalManager(region=region, environment=environment)

        proposals = []
        screener_count = 0

        # Run market screener
        logger.info("Running market screener...")
        try:
            # Fetch all markets from Kalshi API
            all_markets = fetch_all_markets(
                status="open",
                mve_filter="exclude"
            )
            logger.info(f"Fetched {len(all_markets)} open markets from Kalshi")

            # Filter and score markets
            filtered_markets = filter_and_score_markets(
                markets=all_markets,
                db_client=db_client,
                skip_info_risk=skip_info_risk,
                top_n=top_n_candidates
            )
            logger.info(f"Filtered to {len(filtered_markets)} candidate markets")

            # Convert candidates to proposals
            for market in filtered_markets[:top_n_candidates]:
                # Build comprehensive metadata aligned with market_update structure
                metadata = {
                    # Basic market info
                    'title': market.get('title', ''),
                    'subtitle': market.get('subtitle', ''),
                    'event_ticker': market.get('event_ticker', ''),
                    'close_time': market.get('close_time'),

                    # Trading metrics
                    'volume_24h': market.get('volume_24h', 0),
                    'buy_volume': market.get('buy_volume', 0),
                    'sell_volume': market.get('sell_volume', 0),
                    'yes_bid': market.get('yes_bid'),
                    'yes_ask': market.get('yes_ask'),
                    'current_spread': market.get('current_spread', 0),
                    'previous_spread': market.get('previous_spread', 0),

                    # Order book depth
                    'bid_depth_5c': market.get('bid_depth_5c', 0),
                    'ask_depth_5c': market.get('ask_depth_5c', 0),

                    # Risk assessment
                    'info_risk': market.get('info_risk_probability'),
                    'info_risk_rationale': market.get('info_risk_rationale', ''),

                    # P&L and position (0 for new markets, or historical if previously disabled)
                    'pnl_24h': market.get('historical_realized_pnl', 0),
                    'fill_count': 0,
                    'position_qty': 0,

                    # Current config (null for new markets)
                    'current_quote_size': None,
                    'current_min_spread': None,
                    'created_at': None,  # New market, not yet created

                    # Enriched Kalshi metadata for AI model (matching market_update format)
                    'event_title': market.get('event_name', ''),  # Event name/title
                    'market_title': market.get('title', ''),  # Market title
                    'volume_24h_trades': market.get('buy_volume_trades', 0) + market.get('sell_volume_trades', 0),  # Total trade count
                    'volume_24h_contracts': market.get('volume_24h', 0),  # Total 24h volume
                    'buy_volume_trades': market.get('buy_volume_trades', 0),  # Buy side trade count
                    'buy_volume_contracts': market.get('buy_volume', 0),  # Buy side volume
                    'sell_volume_trades': market.get('sell_volume_trades', 0),  # Sell side trade count
                    'sell_volume_contracts': market.get('sell_volume', 0),  # Sell side volume
                    'current_spread': market.get('current_spread', 0),
                    'spread_24h_ago': market.get('previous_spread', 0) / 100 if market.get('previous_spread') else None,  # Convert cents to decimal
                    'yes_bid': market.get('yes_bid'),
                    'yes_ask': market.get('yes_ask'),
                    'previous_yes_bid': market.get('previous_yes_bid'),
                    'previous_yes_ask': market.get('previous_yes_ask'),
                }

                proposal = {
                    'market_id': market['ticker'],
                    'proposal_source': 'market_screener',
                    'action': 'new_market',
                    'reason': f"High volume ({market.get('volume_24h', 0)} contracts), good spread",
                    'current_config': {},
                    'proposed_changes': {
                        'quote_size': 5,
                        'max_inventory_yes': 5,
                        'max_inventory_no': 5,
                        'min_spread': 0.04,
                        'enabled': True,
                    },
                    'metadata': metadata
                }
                proposals.append(proposal)
                screener_count += 1

        except Exception as e:
            logger.error(f"Error in market screener: {e}", exc_info=True)
            return {
                'statusCode': 500,
                'body': {
                    'success': False,
                    'mode': 'market_screener_only',
                    'error': str(e)
                }
            }

        # Save Proposals to DynamoDB
        if not proposals:
            logger.warning("No proposals generated")
            return {
                'statusCode': 200,
                'body': {
                    'success': True,
                    'mode': 'market_screener_only',
                    'environment': environment,
                    'message': 'No proposals generated',
                    'screener_count': 0,
                }
            }

        logger.info(f"Saving {len(proposals)} proposals to DynamoDB...")
        proposal_id = proposal_manager.create_proposal_batch(
            proposals=proposals,
            ttl_hours=12
        )
        logger.info(f"Proposals saved with ID: {proposal_id}")

        # Build response
        response = {
            'statusCode': 200,
            'body': {
                'success': True,
                'mode': 'market_screener_only',
                'environment': environment,
                'proposal_id': proposal_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'total_proposals': len(proposals),
                    'new_candidates': screener_count,
                },
            }
        }

        logger.info(f"Market screener only mode completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in market screener only mode: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'mode': 'market_screener_only',
                'error': str(e)
            }
        }
