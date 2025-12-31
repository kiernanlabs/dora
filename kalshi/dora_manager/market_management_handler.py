"""
Market Management Handler for Dora Manager

Combines market_screener and market_update functionality into a single handler.
Generates proposals and sends them via email for user approval.

This handler:
1. Runs market update analysis (existing markets)
2. Runs market screener (new candidates)
3. Combines proposals into a single batch
4. Saves to DynamoDB
5. Sends email with signed approval URL
"""
import logging
import os
from typing import Dict, Any, List
from datetime import datetime, timezone

# Import from local modules
from market_update import (
    analyze_markets,
    generate_recommendations,
)
from market_screener import (
    fetch_all_markets,
    filter_and_score_markets,
    assess_information_risk
)
from db_client import DynamoDBClient

# Import utilities
from utils.proposal_manager import ProposalManager
from utils.url_signer import URLSigner
from email_sender import EmailSender

logger = logging.getLogger(__name__)


def handle_market_management(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle market_management mode execution.

    Combines market update and market screener logic:
    1. Analyze existing markets for updates
    2. Screen new market candidates
    3. Generate combined proposals
    4. Save to DynamoDB
    5. Send email with approval link

    Args:
        event: Lambda event with configuration:
            - environment: 'demo' or 'prod'
            - pnl_lookback_hours: Hours for P&L analysis (default: 24)
            - volume_lookback_hours: Hours for volume analysis (default: 48)
            - top_n_candidates: Number of top candidates from screener (default: 20)
            - skip_info_risk: Skip OpenAI info risk assessment (default: false)
        context: Lambda context

    Returns:
        Response with summary data
    """
    # Get configuration
    environment = event.get('environment') or os.environ.get('ENVIRONMENT', 'prod')
    region = event.get('region') or os.environ.get('AWS_REGION', 'us-east-1')
    recipient_email = event.get('recipient_email') or os.environ.get('RECIPIENT_EMAIL', 'joe@kiernanlabs.com')
    api_gateway_base_url = os.environ.get('API_GATEWAY_BASE_URL')

    pnl_lookback_hours = int(event.get('pnl_lookback_hours', 24))
    volume_lookback_hours = int(event.get('volume_lookback_hours', 48))
    top_n_candidates = int(event.get('top_n_candidates', 20))
    skip_info_risk = event.get('skip_info_risk', False)

    logger.info(f"Starting Market Management Mode - Environment: {environment}")
    logger.info(f"P&L lookback: {pnl_lookback_hours}h, Volume lookback: {volume_lookback_hours}h")
    logger.info(f"Top N candidates: {top_n_candidates}, Skip info risk: {skip_info_risk}")

    try:
        # Initialize clients
        db_client = DynamoDBClient(region=region, environment=environment)
        proposal_manager = ProposalManager(region=region, environment=environment)
        url_signer = URLSigner(region=region, environment=environment)
        email_sender = EmailSender(region=region)

        all_proposals = []
        update_count = 0
        screener_count = 0

        # === Part A: Market Update Analysis ===
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
                all_proposals.append(proposal)
                update_count += 1

        except Exception as e:
            logger.error(f"Error in market update analysis: {e}", exc_info=True)
            # Continue with screener even if update fails

        # === Part B: Market Screener ===
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
                all_proposals.append(proposal)
                screener_count += 1

        except Exception as e:
            logger.error(f"Error in market screener: {e}", exc_info=True)
            # Continue even if screener fails

        # === Save Proposals to DynamoDB ===
        if not all_proposals:
            logger.warning("No proposals generated")
            return {
                'statusCode': 200,
                'body': {
                    'success': True,
                    'mode': 'market_management',
                    'environment': environment,
                    'message': 'No proposals generated',
                    'update_count': 0,
                    'screener_count': 0,
                }
            }

        logger.info(f"Saving {len(all_proposals)} proposals to DynamoDB...")
        proposal_id = proposal_manager.create_proposal_batch(
            proposals=all_proposals,
            ttl_hours=12
        )
        logger.info(f"Proposals saved with ID: {proposal_id}")

        # === Generate Signed URLs ===
        if not api_gateway_base_url:
            raise ValueError("API_GATEWAY_BASE_URL environment variable not set")

        review_url = url_signer.generate_signed_url(
            base_url=api_gateway_base_url,
            proposal_id=proposal_id,
            endpoint="",
            ttl_hours=12
        )

        approve_all_url = url_signer.generate_signed_url(
            base_url=api_gateway_base_url,
            proposal_id=proposal_id,
            endpoint="/execute",
            ttl_hours=12
        )

        # === Send Email ===
        logger.info(f"Sending proposal email to: {recipient_email}")
        email_sent = email_sender.send_market_proposals_email(
            proposals=all_proposals,
            proposal_id=proposal_id,
            review_url=review_url,
            approve_all_url=approve_all_url,
            recipient=recipient_email,
            environment=environment
        )

        # Build response
        response = {
            'statusCode': 200,
            'body': {
                'success': True,
                'mode': 'market_management',
                'environment': environment,
                'proposal_id': proposal_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'total_proposals': len(all_proposals),
                    'update_recommendations': update_count,
                    'new_candidates': screener_count,
                },
                'email_sent': email_sent,
                'review_url': review_url,
            }
        }

        logger.info(f"Market management mode completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in market management mode: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'mode': 'market_management',
                'error': str(e)
            }
        }
