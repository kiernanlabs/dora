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
                    'metadata': {
                        'title': market.get('title', ''),
                        'volume_24h': market.get('volume_24h', 0),
                        'yes_bid': market.get('yes_bid'),
                        'yes_ask': market.get('yes_ask'),
                        'info_risk': market.get('info_risk_probability'),
                    }
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
