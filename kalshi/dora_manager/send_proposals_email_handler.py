"""
Send Proposals Email Handler for Dora Manager

Queries recent proposals from DynamoDB (from both market_update and market_screener)
and sends a combined email with all proposals.

This handler should be triggered ~5-10 minutes after both market_update_only
and market_screener_only have completed.
"""
import logging
import os
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# Import utilities
from utils.proposal_manager import ProposalManager
from utils.url_signer import URLSigner
from utils.insights_manager import InsightsManager
from utils.ai_insights import generate_event_insights, generate_market_insights
from email_sender import EmailSender

logger = logging.getLogger(__name__)


def handle_send_proposals_email(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle send_proposals_email mode execution.

    Queries DynamoDB for recent proposals and sends combined email.

    Args:
        event: Lambda event with configuration:
            - environment: 'demo' or 'prod'
            - lookback_minutes: How far back to look for proposals (default: 15)
        context: Lambda context

    Returns:
        Response with email status
    """
    # Get configuration
    environment = event.get('environment') or os.environ.get('ENVIRONMENT', 'prod')
    region = event.get('region') or os.environ.get('AWS_REGION', 'us-east-1')
    recipient_email = event.get('recipient_email') or os.environ.get('RECIPIENT_EMAIL', 'joe@kiernanlabs.com')
    api_gateway_base_url = os.environ.get('API_GATEWAY_BASE_URL')
    lookback_minutes = int(event.get('lookback_minutes', 15))

    logger.info(f"Starting Send Proposals Email Mode - Environment: {environment}")
    logger.info(f"Looking back {lookback_minutes} minutes for proposals")

    try:
        # Initialize clients
        proposal_manager = ProposalManager(region=region, environment=environment)
        url_signer = URLSigner(region=region, environment=environment)
        insights_manager = InsightsManager(region=region, environment=environment)
        email_sender = EmailSender(region=region)

        # Query recent pending proposals
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        cutoff_iso = cutoff_time.isoformat()

        logger.info(f"Querying proposals created after {cutoff_iso}")

        # Get all pending proposals from the last lookback window
        # Convert lookback_minutes to hours, rounding up to ensure we don't miss any
        max_age_hours = (lookback_minutes + 59) // 60  # Ceiling division
        logger.info(f"Querying DynamoDB for pending proposals from last {max_age_hours} hour(s)")
        all_proposals = proposal_manager.get_pending_proposals(max_age_hours=max_age_hours)

        # Filter to only those created in the lookback window
        logger.info(f"Retrieved {len(all_proposals)} pending proposals from DynamoDB")
        recent_proposals = [
            p for p in all_proposals
            if p.get('created_at', '') >= cutoff_iso
        ]

        logger.info(f"Filtered to {len(recent_proposals)} proposals created after {cutoff_iso}")

        if not recent_proposals:
            logger.warning(f"No proposals found created after {cutoff_iso}")
            return {
                'statusCode': 200,
                'body': {
                    'success': True,
                    'mode': 'send_proposals_email',
                    'message': 'No recent proposals to send',
                    'lookback_minutes': lookback_minutes,
                }
            }

        # Group proposals by proposal_id
        proposals_by_id = defaultdict(list)
        for proposal in recent_proposals:
            proposal_id = proposal.get('proposal_id')
            if proposal_id:
                proposals_by_id[proposal_id].append(proposal)

        logger.info(f"Found {len(recent_proposals)} proposals across {len(proposals_by_id)} batches")

        # Send email for each proposal batch
        emails_sent = 0
        for proposal_id, proposals in proposals_by_id.items():
            logger.info(f"Processing proposal batch {proposal_id} with {len(proposals)} proposals")

            # Count by source
            update_count = sum(1 for p in proposals if p.get('proposal_source') == 'market_update')
            screener_count = sum(1 for p in proposals if p.get('proposal_source') == 'market_screener')

            logger.info(f"  {update_count} from market_update, {screener_count} from market_screener")

            # Generate AI insights for this proposal batch
            event_insights = {}  # event_ticker -> insight dict
            market_insights = {}  # market_id -> insight dict

            # Separate proposals by source
            update_proposals = [p for p in proposals if p.get('proposal_source') == 'market_update']
            screener_proposals = [p for p in proposals if p.get('proposal_source') == 'market_screener']

            # Generate event-level insights for market_update proposals
            if update_proposals:
                # Group market_update proposals by event
                events_dict = defaultdict(list)
                for p in update_proposals:
                    event_ticker = p.get('metadata', {}).get('event_ticker', '')
                    if event_ticker:
                        events_dict[event_ticker].append(p)

                logger.info(f"Generating insights for {len(events_dict)} events...")
                for event_ticker, event_markets in events_dict.items():
                    try:
                        insight = generate_event_insights(event_ticker, event_markets)
                        if not insight.get('error'):
                            event_insights[event_ticker] = insight

                            # Save to DynamoDB
                            metadata = {
                                'market_count': len(event_markets),
                                'total_pnl': sum(m.get('metadata', {}).get('pnl_24h', 0) or 0 for m in event_markets),
                                'total_fills': sum(m.get('metadata', {}).get('fill_count', 0) or 0 for m in event_markets),
                            }
                            insights_manager.save_event_insight(
                                event_ticker=event_ticker,
                                proposal_id=proposal_id,
                                insights=insight['insights'],
                                recommendation=insight['recommendation'],
                                rationale=insight['rationale'],
                                metadata=metadata
                            )
                            logger.info(f"Generated insight for event {event_ticker}: {insight['recommendation']}")
                        else:
                            logger.warning(f"Failed to generate insight for event {event_ticker}: {insight['error']}")
                    except Exception as e:
                        logger.error(f"Error generating insight for event {event_ticker}: {e}")

            # Generate market-level insights for market_screener proposals
            if screener_proposals:
                logger.info(f"Generating insights for {len(screener_proposals)} new markets...")
                for p in screener_proposals:
                    market_id = p.get('market_id', '')
                    if not market_id:
                        continue

                    try:
                        # Prepare market data for insight generation
                        market_data = p.get('metadata', {})

                        insight = generate_market_insights(market_id, market_data)
                        if not insight.get('error'):
                            market_insights[market_id] = insight

                            # Save to DynamoDB
                            insights_manager.save_market_insight(
                                market_id=market_id,
                                proposal_id=proposal_id,
                                recommendation=insight['recommendation'],
                                rationale=insight['rationale'],
                                metadata=market_data
                            )
                            logger.info(f"Generated insight for market {market_id}: {insight['recommendation']}")
                        else:
                            logger.warning(f"Failed to generate insight for market {market_id}: {insight['error']}")
                    except Exception as e:
                        logger.error(f"Error generating insight for market {market_id}: {e}")

            # Convert DynamoDB items to the format expected by email sender
            formatted_proposals = []
            for p in proposals:
                formatted_proposal = {
                    'market_id': p.get('market_id'),
                    'proposal_source': p.get('proposal_source'),
                    'action': p.get('action'),
                    'reason': p.get('reason', ''),
                    'current_config': p.get('current_config', {}),
                    'proposed_changes': p.get('proposed_changes', {}),
                    'metadata': p.get('metadata', {}),
                }
                formatted_proposals.append(formatted_proposal)

            # Generate signed URLs
            if not api_gateway_base_url:
                logger.error("API_GATEWAY_BASE_URL environment variable not set")
                continue

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

            # Send email with AI insights
            logger.info(f"Sending proposal email to: {recipient_email}")
            email_sent = email_sender.send_market_proposals_email(
                proposals=formatted_proposals,
                proposal_id=proposal_id,
                review_url=review_url,
                approve_all_url=approve_all_url,
                recipient=recipient_email,
                environment=environment,
                event_insights=event_insights,
                market_insights=market_insights
            )

            if email_sent:
                emails_sent += 1
                logger.info(f"Email sent successfully for proposal {proposal_id}")
            else:
                logger.error(f"Failed to send email for proposal {proposal_id}")

        # Build response
        response = {
            'statusCode': 200,
            'body': {
                'success': True,
                'mode': 'send_proposals_email',
                'environment': environment,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'total_proposals': len(recent_proposals),
                    'proposal_batches': len(proposals_by_id),
                    'emails_sent': emails_sent,
                },
            }
        }

        logger.info(f"Send proposals email mode completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in send proposals email mode: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'mode': 'send_proposals_email',
                'error': str(e)
            }
        }
