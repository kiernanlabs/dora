"""
Execute Proposals Handler for Dora Manager

Handles API Gateway requests for proposal approval:
- GET /proposals/{proposal_id}: View proposals (HTML page)
- POST /proposals/{proposal_id}/execute: Execute approved proposals
"""
import json
import logging
import os
import sys
from typing import Dict, Any, List
from datetime import datetime, timezone
from urllib.parse import parse_qs

# Add dora_bot to path for database access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dora_bot'))

from dora_bot.dynamo import DynamoDBClient
from dora_bot.models import MarketConfig

from utils.proposal_manager import ProposalManager
from utils.url_signer import URLSigner
from email_sender import EmailSender

logger = logging.getLogger(__name__)


def handle_execute_proposals(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle API Gateway requests for proposal approval.

    Args:
        event: API Gateway proxy integration event
        context: Lambda context

    Returns:
        API Gateway response
    """
    logger.info(f"API Gateway event: {json.dumps(event)}")

    # Get environment configuration
    environment = os.environ.get('ENVIRONMENT', 'prod')
    region = os.environ.get('AWS_REGION', 'us-east-1')
    recipient_email = os.environ.get('RECIPIENT_EMAIL', 'joe@kiernanlabs.com')

    try:
        # Extract request details
        http_method = event.get('httpMethod')
        path_parameters = event.get('pathParameters', {})
        query_parameters = event.get('queryStringParameters', {}) or {}

        proposal_id = path_parameters.get('proposal_id')
        if not proposal_id:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'proposal_id is required'})
            }

        # Extract and validate signature
        signature = query_parameters.get('signature')
        expiry = query_parameters.get('expiry')

        if not signature or not expiry:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'signature and expiry are required'})
            }

        # Validate signature
        url_signer = URLSigner(region=region, environment=environment)
        is_valid, error_message = url_signer.validate_signature(proposal_id, signature, expiry)

        if not is_valid:
            return {
                'statusCode': 403,
                'headers': {'Content-Type': 'text/html'},
                'body': f"""
                <html>
                <body style="font-family: Arial; padding: 40px; text-align: center;">
                    <h1>⛔ Invalid or Expired Link</h1>
                    <p>{error_message}</p>
                    <p>Approval links expire after 12 hours for security.</p>
                </body>
                </html>
                """
            }

        # Initialize clients
        proposal_manager = ProposalManager(region=region, environment=environment)
        db_client = DynamoDBClient(region=region, environment=environment)

        # Route based on HTTP method
        if http_method == 'GET':
            return handle_get_proposals(proposal_id, proposal_manager)
        elif http_method == 'POST':
            return handle_post_execute(
                event,
                proposal_id,
                proposal_manager,
                db_client,
                recipient_email,
                environment,
                region
            )
        else:
            return {
                'statusCode': 405,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Method not allowed'})
            }

    except Exception as e:
        logger.error(f"Error handling execute_proposals: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }


def handle_get_proposals(proposal_id: str, proposal_manager: ProposalManager) -> Dict[str, Any]:
    """Handle GET request - display proposals for review.

    Args:
        proposal_id: UUID of proposal batch
        proposal_manager: ProposalManager instance

    Returns:
        API Gateway response with HTML page
    """
    logger.info(f"GET request for proposal {proposal_id}")

    # Retrieve proposals from DynamoDB
    proposals = proposal_manager.get_proposals_by_id(proposal_id, status_filter='pending')

    if not proposals:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'text/html'},
            'body': """
            <html>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1>No Pending Proposals Found</h1>
                <p>These proposals may have already been executed or rejected.</p>
            </body>
            </html>
            """
        }

    # Group by source
    update_proposals = [p for p in proposals if p['proposal_source'] == 'market_update']
    screener_proposals = [p for p in proposals if p['proposal_source'] == 'market_screener']

    # Generate HTML review page
    html = generate_review_page_html(proposal_id, update_proposals, screener_proposals)

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html
    }


def generate_review_page_html(
    proposal_id: str,
    update_proposals: List[Dict],
    screener_proposals: List[Dict]
) -> str:
    """Generate HTML page for reviewing proposals.

    Args:
        proposal_id: UUID of proposal batch
        update_proposals: List of market update proposals
        screener_proposals: List of market screener proposals

    Returns:
        HTML string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DORA Proposal Review</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 20px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; }}
            .section-header {{ background-color: #2196F3; color: white; padding: 10px; margin-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .button {{ background-color: #4CAF50; color: white; padding: 15px 32px; text-decoration: none;
                      display: inline-block; margin: 10px 5px; border: none; cursor: pointer; font-size: 16px; }}
            .button-reject {{ background-color: #f44336; }}
            .actions {{ text-align: center; margin: 30px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>DORA Market Proposal Review</h1>
            <p>Proposal ID: {proposal_id}</p>
            <p>Total Proposals: {len(update_proposals) + len(screener_proposals)}</p>
        </div>

        <form method="POST" action="execute?signature={{signature}}&expiry={{expiry}}">
    """

    # Market Updates Section
    if update_proposals:
        html += f"""
        <div class="section">
            <div class="section-header">
                <h2>📊 Market Updates ({len(update_proposals)} proposals)</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Select</th>
                        <th>Market ID</th>
                        <th>Action</th>
                        <th>Reason</th>
                        <th>P&L (24h)</th>
                        <th>Proposed Changes</th>
                    </tr>
                </thead>
                <tbody>
        """

        for p in update_proposals:
            pnl = p.get('metadata', {}).get('pnl_24h', 0)
            changes = p.get('proposed_changes', {})
            changes_str = ", ".join([f"{k}: {v}" for k, v in changes.items()])

            html += f"""
                <tr>
                    <td><input type="checkbox" name="selected" value="{p['market_id']}" checked></td>
                    <td>{p['market_id']}</td>
                    <td>{p['action']}</td>
                    <td>{p.get('reason', '')[:100]}</td>
                    <td>${pnl:.2f}</td>
                    <td>{changes_str}</td>
                </tr>
            """

        html += "</tbody></table></div>"

    # New Candidates Section
    if screener_proposals:
        html += f"""
        <div class="section">
            <div class="section-header">
                <h2>🆕 New Market Candidates ({len(screener_proposals)} proposals)</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Select</th>
                        <th>Market ID</th>
                        <th>Title</th>
                        <th>Volume (24h)</th>
                        <th>Proposed Config</th>
                    </tr>
                </thead>
                <tbody>
        """

        for p in screener_proposals:
            metadata = p.get('metadata', {})
            title = metadata.get('title', '')
            volume = metadata.get('volume_24h', 0)
            changes = p.get('proposed_changes', {})
            quote_size = changes.get('quote_size', 5)

            html += f"""
                <tr>
                    <td><input type="checkbox" name="selected" value="{p['market_id']}" checked></td>
                    <td>{p['market_id']}</td>
                    <td>{title[:60]}</td>
                    <td>{volume:,}</td>
                    <td>quote_size={quote_size}, max_inv=5</td>
                </tr>
            """

        html += "</tbody></table></div>"

    # Action buttons
    html += """
        <div class="actions">
            <button type="submit" name="action" value="approve_selected" class="button">
                Approve Selected
            </button>
            <button type="submit" name="action" value="reject_all" class="button button-reject">
                Reject All
            </button>
        </div>
        </form>
    </body>
    </html>
    """

    return html


def handle_post_execute(
    event: Dict[str, Any],
    proposal_id: str,
    proposal_manager: ProposalManager,
    db_client: DynamoDBClient,
    recipient_email: str,
    environment: str,
    region: str
) -> Dict[str, Any]:
    """Handle POST request - execute approved proposals.

    Args:
        event: API Gateway event
        proposal_id: UUID of proposal batch
        proposal_manager: ProposalManager instance
        db_client: DynamoDBClient instance
        recipient_email: Email for confirmation
        environment: Environment name
        region: AWS region

    Returns:
        API Gateway response
    """
    logger.info(f"POST request for proposal {proposal_id}")

    # Parse request body
    body = event.get('body', '')
    if event.get('isBase64Encoded'):
        import base64
        body = base64.b64decode(body).decode('utf-8')

    # Parse form data or JSON
    try:
        if 'application/json' in event.get('headers', {}).get('Content-Type', ''):
            data = json.loads(body)
            selected_markets = data.get('selected_proposals', [])
            approve_all = data.get('approve_all', False)
        else:
            # Form data
            parsed = parse_qs(body)
            selected_markets = parsed.get('selected', [])
            action = parsed.get('action', [''])[0]
            approve_all = (action == 'approve_all')

            if action == 'reject_all':
                # Reject all proposals
                proposals = proposal_manager.get_proposals_by_id(proposal_id, status_filter='pending')
                market_ids = [p['market_id'] for p in proposals]
                proposal_manager.batch_update_status(proposal_id, market_ids, 'rejected')

                return {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'text/html'},
                    'body': """
                    <html>
                    <body style="font-family: Arial; padding: 40px; text-align: center;">
                        <h1>✓ All Proposals Rejected</h1>
                        <p>All proposals have been rejected successfully.</p>
                    </body>
                    </html>
                    """
                }

    except Exception as e:
        logger.error(f"Error parsing request body: {e}")
        selected_markets = []
        approve_all = False

    # Get pending proposals
    proposals = proposal_manager.get_proposals_by_id(proposal_id, status_filter='pending')

    if not proposals:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'No pending proposals found'})
        }

    # Filter selected proposals
    if approve_all:
        proposals_to_execute = proposals
    else:
        proposals_to_execute = [p for p in proposals if p['market_id'] in selected_markets]

    logger.info(f"Executing {len(proposals_to_execute)} proposals")

    # Execute proposals
    results = []
    for proposal in proposals_to_execute:
        try:
            result = execute_single_proposal(proposal, db_client)
            results.append(result)

            # Update proposal status
            if result['success']:
                proposal_manager.update_proposal_status(
                    proposal_id,
                    proposal['market_id'],
                    'executed'
                )
            else:
                proposal_manager.update_proposal_status(
                    proposal_id,
                    proposal['market_id'],
                    'failed',
                    error_message=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"Error executing proposal for {proposal['market_id']}: {e}")
            results.append({
                'market_id': proposal['market_id'],
                'success': False,
                'error': str(e)
            })
            proposal_manager.update_proposal_status(
                proposal_id,
                proposal['market_id'],
                'failed',
                error_message=str(e)
            )

    # Send confirmation email
    email_sender = EmailSender(region=region)
    send_confirmation_email(email_sender, results, recipient_email, environment)

    # Generate success response
    success_count = sum(1 for r in results if r['success'])
    failure_count = len(results) - success_count

    html_response = f"""
    <html>
    <body style="font-family: Arial; padding: 40px; text-align: center;">
        <h1>✓ Execution Complete</h1>
        <p>Successfully executed: {success_count}</p>
        <p>Failed: {failure_count}</p>
        <p>A confirmation email has been sent to {recipient_email}</p>
    </body>
    </html>
    """

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html_response
    }


def execute_single_proposal(proposal: Dict[str, Any], db_client: DynamoDBClient) -> Dict[str, Any]:
    """Execute a single proposal.

    Args:
        proposal: Proposal dict from DynamoDB
        db_client: DynamoDBClient instance

    Returns:
        Result dict with success status
    """
    market_id = proposal['market_id']
    action = proposal['action']
    proposed_changes = proposal.get('proposed_changes', {})

    logger.info(f"Executing {action} for {market_id}")

    try:
        if action == 'new_market':
            # Create new market config
            market_config = MarketConfig(
                market_id=market_id,
                quote_size=proposed_changes.get('quote_size', 5),
                max_inventory_yes=proposed_changes.get('max_inventory_yes', 5),
                max_inventory_no=proposed_changes.get('max_inventory_no', 5),
                min_spread=proposed_changes.get('min_spread', 0.04),
                enabled=proposed_changes.get('enabled', True),
            )
            db_client.put_market_config(market_config)

        else:
            # Update existing market config
            existing = db_client.get_market_config(market_id)
            if not existing:
                return {'market_id': market_id, 'success': False, 'error': 'Market config not found'}

            # Apply proposed changes
            for key, value in proposed_changes.items():
                setattr(existing, key, value)

            db_client.put_market_config(existing)

        return {'market_id': market_id, 'success': True, 'action': action}

    except Exception as e:
        logger.error(f"Error executing proposal for {market_id}: {e}")
        return {'market_id': market_id, 'success': False, 'error': str(e)}


def send_confirmation_email(
    email_sender: EmailSender,
    results: List[Dict[str, Any]],
    recipient: str,
    environment: str
):
    """Send confirmation email with execution results.

    Args:
        email_sender: EmailSender instance
        results: List of execution results
        recipient: Email recipient
        environment: Environment name
    """
    success_count = sum(1 for r in results if r['success'])
    failure_count = len(results) - success_count

    subject = f"[DORA {environment.upper()}] Proposal Execution Complete - {success_count} Succeeded, {failure_count} Failed"

    html_body = f"""
    <html>
    <body style="font-family: Arial;">
        <h2>Proposal Execution Results</h2>
        <p><strong>Total Executed:</strong> {len(results)}</p>
        <p><strong>Successful:</strong> {success_count}</p>
        <p><strong>Failed:</strong> {failure_count}</p>

        <h3>Details:</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px;">Market ID</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Status</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Error</th>
                </tr>
            </thead>
            <tbody>
    """

    for result in results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        status_color = "green" if result['success'] else "red"
        error = result.get('error', '')

        html_body += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{result['market_id']}</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: {status_color};">{status}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{error}</td>
            </tr>
        """

    html_body += """
            </tbody>
        </table>
    </body>
    </html>
    """

    text_body = f"""
Proposal Execution Results

Total: {len(results)}
Successful: {success_count}
Failed: {failure_count}

Details:
{chr(10).join([f"- {r['market_id']}: {'Success' if r['success'] else 'Failed - ' + r.get('error', '')}" for r in results])}
    """

    try:
        email_sender.ses.send_email(
            Source=email_sender.sender_email,
            Destination={'ToAddresses': [recipient]},
            Message={
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {
                    'Text': {'Data': text_body, 'Charset': 'UTF-8'},
                    'Html': {'Data': html_body, 'Charset': 'UTF-8'}
                }
            }
        )
        logger.info(f"Confirmation email sent to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send confirmation email: {e}")
