"""
Market Update Handler for Dora Manager

Analyzes existing market performance and generates recommendations.
Saves proposals to DynamoDB but does NOT send email.
Email is sent by send_proposals_email_handler after both update and screener complete.
"""
import logging
import os
from typing import Dict, Any
from datetime import datetime, timezone

# Import from local modules
from market_update import (
    analyze_markets,
    generate_recommendations,
)
from db_client import DynamoDBClient

# Import utilities
from utils.proposal_manager import ProposalManager

logger = logging.getLogger(__name__)


def handle_market_update_only(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle market_update_only mode execution.

    Analyzes existing markets and saves proposals to DynamoDB.
    Does NOT send email - that's handled by send_proposals_email_handler.

    Args:
        event: Lambda event with configuration:
            - environment: 'demo' or 'prod'
            - pnl_lookback_hours: Hours for P&L analysis (default: 24)
            - volume_lookback_hours: Hours for volume analysis (default: 48)
            - skip_info_risk: Skip OpenAI info risk assessment (default: false)
        context: Lambda context

    Returns:
        Response with proposal_id and summary data
    """
    # Get configuration
    environment = event.get('environment') or os.environ.get('ENVIRONMENT', 'prod')
    region = event.get('region') or os.environ.get('AWS_REGION', 'us-east-1')

    pnl_lookback_hours = int(event.get('pnl_lookback_hours', 24))
    volume_lookback_hours = int(event.get('volume_lookback_hours', 48))
    skip_info_risk = event.get('skip_info_risk', False)

    logger.info(f"Starting Market Update Only Mode - Environment: {environment}")
    logger.info(f"P&L lookback: {pnl_lookback_hours}h, Volume lookback: {volume_lookback_hours}h")
    logger.info(f"Skip info risk: {skip_info_risk}")

    try:
        # Initialize clients
        db_client = DynamoDBClient(region=region, environment=environment)
        proposal_manager = ProposalManager(region=region, environment=environment)

        proposals = []
        update_count = 0

        # Run market update analysis
        logger.info("Running market update analysis...")
        try:
            # Analyze existing markets
            market_analyses = analyze_markets(
                db_client=db_client,
                pnl_lookback_hours=pnl_lookback_hours,
                volume_lookback_hours=volume_lookback_hours
            )
            logger.info(f"Analyzed {len(market_analyses)} markets")

            # Generate recommendations
            recommendations = generate_recommendations(
                analyses=market_analyses,
                skip_info_risk=skip_info_risk
            )
            logger.info(f"Generated {len(recommendations)} update recommendations")

            # Convert recommendations to proposals
            for rec in recommendations:
                # Convert dataclass to dict
                rec_dict = rec.to_dict()

                # Build current_config from the recommendation
                current_config = {
                    'quote_size': rec_dict.get('current_quote_size'),
                    'min_spread': rec_dict.get('current_min_spread'),
                }

                # Build proposed_changes from the recommendation
                proposed_changes = {}
                if rec_dict.get('new_enabled') is not None:
                    proposed_changes['enabled'] = rec_dict['new_enabled']
                if rec_dict.get('new_quote_size') is not None:
                    proposed_changes['quote_size'] = rec_dict['new_quote_size']
                if rec_dict.get('new_max_inventory_yes') is not None:
                    proposed_changes['max_inventory_yes'] = rec_dict['new_max_inventory_yes']
                if rec_dict.get('new_max_inventory_no') is not None:
                    proposed_changes['max_inventory_no'] = rec_dict['new_max_inventory_no']
                if rec_dict.get('new_min_spread') is not None:
                    proposed_changes['min_spread'] = rec_dict['new_min_spread']

                proposal = {
                    'market_id': rec_dict['market_id'],
                    'proposal_source': 'market_update',
                    'action': rec_dict['action'],
                    'reason': rec_dict['reason'],
                    'current_config': current_config,
                    'proposed_changes': proposed_changes,
                    'metadata': {
                        'pnl_24h': rec_dict.get('pnl_24h', 0),
                        'fill_count': rec_dict.get('fill_count_24h', 0),
                        'median_fill_size': None,  # Not in RecommendedAction
                        'info_risk': rec_dict.get('info_risk_probability'),
                    }
                }
                proposals.append(proposal)
                update_count += 1

        except Exception as e:
            logger.error(f"Error in market update analysis: {e}", exc_info=True)
            return {
                'statusCode': 500,
                'body': {
                    'success': False,
                    'mode': 'market_update_only',
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
                    'mode': 'market_update_only',
                    'environment': environment,
                    'message': 'No proposals generated',
                    'update_count': 0,
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
                'mode': 'market_update_only',
                'environment': environment,
                'proposal_id': proposal_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'total_proposals': len(proposals),
                    'update_recommendations': update_count,
                },
            }
        }

        logger.info(f"Market update only mode completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in market update only mode: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'mode': 'market_update_only',
                'error': str(e)
            }
        }
