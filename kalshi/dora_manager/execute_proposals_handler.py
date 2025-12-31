"""
Execute Proposals Handler for Dora Manager

Handles API Gateway requests for proposal approval:
- GET /proposals/{proposal_id}: View proposals (HTML page)
- POST /proposals/{proposal_id}/execute: Execute approved proposals
"""
import json
import logging
import os
from typing import Dict, Any, List
from datetime import datetime, timezone
from urllib.parse import parse_qs
from decimal import Decimal

# Use local db_client instead of dora_bot for Lambda deployment
from db_client import DynamoDBClient

from utils.proposal_manager import ProposalManager
from utils.url_signer import URLSigner
from email_sender import EmailSender

logger = logging.getLogger(__name__)


def convert_floats_to_decimal(obj: Any) -> Any:
    """Recursively convert all float values to Decimal for DynamoDB compatibility.

    Args:
        obj: Object to convert (dict, list, or primitive)

    Returns:
        Converted object with floats replaced by Decimals
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    else:
        return obj


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
            return handle_get_proposals(proposal_id, proposal_manager, signature, expiry)
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


def handle_get_proposals(proposal_id: str, proposal_manager: ProposalManager, signature: str, expiry: str) -> Dict[str, Any]:
    """Handle GET request - display proposals for review.

    Args:
        proposal_id: UUID of proposal batch
        proposal_manager: ProposalManager instance
        signature: URL signature for authentication
        expiry: URL expiry timestamp

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

    # Route to appropriate HTML generator based on proposal types
    # If only screener proposals, use card-based layout
    # If only update proposals or mixed, use table-based layout
    if screener_proposals and not update_proposals:
        html = generate_screener_candidates_html(proposal_id, screener_proposals, signature, expiry)
    else:
        html = generate_review_page_html(proposal_id, update_proposals, screener_proposals, signature, expiry)

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html
    }


def generate_review_page_html(
    proposal_id: str,
    update_proposals: List[Dict],
    screener_proposals: List[Dict],
    signature: str,
    expiry: str
) -> str:
    """Generate HTML page for reviewing proposals.

    Args:
        proposal_id: UUID of proposal batch
        update_proposals: List of market update proposals
        screener_proposals: List of market screener proposals
        signature: URL signature for authentication
        expiry: URL expiry timestamp

    Returns:
        HTML string
    """
    from collections import defaultdict
    from datetime import datetime
    import html as html_lib

    # Group proposals by event
    proposals_by_event = defaultdict(list)
    all_proposals = update_proposals + screener_proposals

    for p in all_proposals:
        # Get event info from metadata or use market_id prefix
        event_ticker = p.get('metadata', {}).get('event_ticker', '')
        if not event_ticker and p.get('market_id'):
            # Try to extract event from market_id (e.g., "PRES-2024" from "PRES-2024-01")
            parts = p['market_id'].split('-')
            if len(parts) >= 2:
                event_ticker = '-'.join(parts[:2])

        proposals_by_event[event_ticker or 'Other'].append(p)

    # Count actions
    action_counts = defaultdict(int)
    for p in update_proposals:
        action_counts[p['action']] += 1

    total_proposals = len(all_proposals)
    timestamp = datetime.utcnow().strftime("%B %d, %Y %H:%M UTC")

    # Calculate summary statistics across all proposals
    total_pnl = sum(p.get('metadata', {}).get('pnl_24h', 0) or 0 for p in all_proposals)
    total_fills = sum(p.get('metadata', {}).get('fill_count', 0) or 0 for p in all_proposals)
    markets_with_fills = sum(1 for p in all_proposals if (p.get('metadata', {}).get('fill_count', 0) or 0) > 0)
    markets_with_fills_pct = (markets_with_fills / total_proposals * 100) if total_proposals > 0 else 0

    # Sum of outstanding positions
    total_position = sum(abs(p.get('metadata', {}).get('position_qty', 0) or 0) for p in all_proposals)

    # Sum of current and proposed quote sizes
    total_current_quote_size = sum(p.get('current_config', {}).get('quote_size', 0) or 0 for p in all_proposals)
    total_proposed_quote_size = sum(
        p.get('proposed_changes', {}).get('quote_size') or p.get('current_config', {}).get('quote_size', 0) or 0
        for p in all_proposals
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DORA Proposal Review - {timestamp}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
                padding: 20px;
            }}

            .container {{
                max-width: 1800px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 28px;
            }}

            h2 {{
                color: #2c3e50;
                margin: 40px 0 20px 0;
                font-size: 22px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}

            .timestamp {{
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 25px;
            }}

            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
                padding: 20px;
                background: #ecf0f1;
                border-radius: 6px;
            }}

            .summary-item {{
                text-align: center;
            }}

            .summary-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .summary-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 5px;
            }}

            .summary-value.green {{
                color: #28a745;
            }}

            .summary-value.red {{
                color: #dc3545;
            }}

            .toggle-controls {{
                background: #e8f4f8;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 15px 20px;
                margin: 20px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .toggle-label {{
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
            }}

            .toggle-buttons {{
                display: flex;
                gap: 10px;
            }}

            .toggle-btn {{
                background: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: background 0.2s;
            }}

            .toggle-btn:hover {{
                background: #2980b9;
            }}

            .toggle-btn.deselect {{
                background: #95a5a6;
            }}

            .toggle-btn.deselect:hover {{
                background: #7f8c8d;
            }}

            .event-section {{
                margin-bottom: 40px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: visible;
            }}

            .event-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 8px 8px 0 0;
            }}

            .event-header-top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }}

            .event-title-group {{
                flex: 1;
            }}

            .event-title {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 4px;
            }}

            .event-ticker {{
                font-size: 12px;
                font-family: 'Courier New', monospace;
                opacity: 0.8;
            }}

            .event-stats {{
                display: flex;
                gap: 25px;
                font-size: 14px;
            }}

            .event-stat {{
                display: flex;
                flex-direction: column;
                align-items: flex-end;
            }}

            .event-stat-label {{
                font-size: 11px;
                opacity: 0.8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .event-stat-value {{
                font-size: 16px;
                font-weight: 600;
            }}

            .key-takeaways {{
                background: rgba(255, 255, 255, 0.15);
                padding: 12px 15px;
                border-radius: 4px;
                font-size: 13px;
                line-height: 1.6;
                margin-top: 10px;
                border-left: 3px solid rgba(255, 255, 255, 0.5);
                font-style: italic;
                color: rgba(255, 255, 255, 0.95);
            }}

            .table-wrapper {{
                overflow-x: auto;
                border-radius: 0 0 8px 8px;
                position: relative;
                z-index: 10;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}

            thead {{
                background: #34495e;
                color: white;
            }}

            th {{
                padding: 12px 8px;
                text-align: left;
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            td {{
                padding: 10px 8px;
                border-bottom: 1px solid #f0f0f0;
            }}

            tbody tr:hover {{
                background: #f8f9fa;
            }}

            .action-badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
            }}

            .action-expand {{
                background: #d4edda;
                color: #155724;
            }}

            .action-scale_down {{
                background: #fff3cd;
                color: #856404;
            }}

            .action-exit {{
                background: #f8d7da;
                color: #721c24;
            }}

            .action-activate_sibling {{
                background: #d1ecf1;
                color: #0c5460;
            }}

            .action-reset_defaults {{
                background: #e9ecef;
                color: #383d41;
            }}

            .pnl-positive {{
                color: #28a745;
                font-weight: 600;
            }}

            .pnl-negative {{
                color: #dc3545;
                font-weight: 600;
            }}

            .fill-stale {{
                color: #dc3545;
                font-weight: 600;
            }}

            .actions {{
                position: sticky;
                bottom: 0;
                background: white;
                border-top: 2px solid #dee2e6;
                padding: 20px;
                margin: 30px -30px -30px -30px;
                text-align: center;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            }}

            .button {{
                background-color: #28a745;
                color: white;
                padding: 15px 32px;
                text-decoration: none;
                display: inline-block;
                margin: 10px 5px;
                border: none;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                border-radius: 6px;
                transition: background 0.2s;
            }}

            .button:hover {{
                background-color: #218838;
            }}

            .button-reject {{
                background-color: #dc3545;
            }}

            .button-reject:hover {{
                background-color: #c82333;
            }}

            .checkbox-cell {{
                text-align: center;
                width: 50px;
            }}

            input[type="checkbox"] {{
                width: 18px;
                height: 18px;
                cursor: pointer;
            }}

            .market-id {{
                font-family: 'Courier New', monospace;
                font-size: 12px;
                color: #495057;
            }}

            .changes-list {{
                font-size: 12px;
                color: #6c757d;
            }}

            .action-dropdown {{
                padding: 6px 10px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background: white;
                font-size: 12px;
                color: #495057;
                cursor: pointer;
                width: 100%;
                max-width: 200px;
            }}

            .action-dropdown:focus {{
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }}

            .screener-section {{
                margin-top: 40px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 2px solid #3498db;
            }}

            .screener-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}

            .screener-title {{
                font-size: 22px;
                font-weight: 600;
                color: #2c3e50;
            }}

            .screener-count {{
                background: #3498db;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
            }}

            /* Clickable info styles */
            .clickable-info {{
                cursor: pointer;
                border-bottom: 1px dotted #999;
                display: inline-block;
            }}

            .clickable-info:hover {{
                border-bottom: 2px solid #3498db;
            }}

            /* Modal styles */
            .modal {{
                display: none;
                position: fixed;
                z-index: 100000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.6);
                animation: fadeIn 0.2s;
            }}

            .modal.show {{
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .modal-content {{
                background-color: #2c3e50;
                color: #fff;
                padding: 25px;
                border-radius: 8px;
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                animation: slideIn 0.3s;
            }}

            .modal-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid rgba(255,255,255,0.2);
            }}

            .modal-title {{
                font-size: 18px;
                font-weight: 600;
                color: #ecf0f1;
            }}

            .modal-close {{
                font-size: 28px;
                font-weight: bold;
                color: #95a5a6;
                cursor: pointer;
                background: none;
                border: none;
                padding: 0;
                line-height: 1;
            }}

            .modal-close:hover {{
                color: #fff;
            }}

            .modal-body {{
                font-size: 14px;
                line-height: 1.8;
            }}

            .modal-row {{
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}

            .modal-row:last-child {{
                border-bottom: none;
            }}

            .modal-label {{
                font-weight: 600;
                color: #ecf0f1;
            }}

            .modal-value {{
                color: #3498db;
                font-family: 'Courier New', monospace;
            }}

            .modal-section-title {{
                font-weight: 600;
                margin: 15px 0 10px 0;
                color: #ecf0f1;
                font-size: 15px;
                border-top: 1px solid rgba(255,255,255,0.2);
                padding-top: 15px;
            }}

            .modal-section-title:first-child {{
                margin-top: 0;
                border-top: none;
                padding-top: 0;
            }}

            .modal-rationale {{
                white-space: pre-wrap;
                line-height: 1.6;
                color: #ecf0f1;
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            @keyframes slideIn {{
                from {{
                    transform: translateY(-50px);
                    opacity: 0;
                }}
                to {{
                    transform: translateY(0);
                    opacity: 1;
                }}
            }}

            /* Data section headers */
            .data-section-header {{
                background: #34495e;
                color: white;
                padding: 8px 12px;
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-top: 2px solid #2c3e50;
            }}

            .data-section-header.internal {{
                background: #16a085;
            }}

            .data-section-header.external {{
                background: #e67e22;
            }}

            /* Event external data section */
            .event-external-data {{
                background: rgba(255, 255, 255, 0.12);
                padding: 12px 15px;
                border-radius: 4px;
                font-size: 12px;
                margin-top: 10px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
            }}

            .external-metric {{
                display: flex;
                flex-direction: column;
            }}

            .external-metric-label {{
                font-size: 10px;
                opacity: 0.85;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}

            .external-metric-value {{
                font-size: 15px;
                font-weight: 700;
                font-family: 'Courier New', monospace;
            }}

            .capture-pct {{
                color: #f39c12;
                font-weight: 700;
            }}
        </style>
        <script>
            function selectAll() {{
                document.querySelectorAll('input[type="checkbox"][name="selected"]').forEach(cb => cb.checked = true);
            }}

            function deselectAll() {{
                document.querySelectorAll('input[type="checkbox"][name="selected"]').forEach(cb => cb.checked = false);
            }}

            function showModal(title, content) {{
                const modal = document.getElementById('infoModal');
                const modalTitle = document.getElementById('modalTitle');
                const modalBody = document.getElementById('modalBody');

                modalTitle.textContent = title;
                modalBody.innerHTML = content;
                modal.classList.add('show');
            }}

            function showModalFromData(element) {{
                const title = element.getAttribute('data-modal-title');
                const content = element.getAttribute('data-modal-content');
                showModal(title, content);
            }}

            function closeModal() {{
                const modal = document.getElementById('infoModal');
                modal.classList.remove('show');
            }}

            // Close modal when clicking outside of it
            window.onclick = function(event) {{
                const modal = document.getElementById('infoModal');
                if (event.target === modal) {{
                    closeModal();
                }}
            }}

            // Close modal with Escape key
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeModal();
                }}
            }});
        </script>
    </head>
    <body>
        <!-- Modal -->
        <div id="infoModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <span class="modal-title" id="modalTitle"></span>
                    <button class="modal-close" onclick="closeModal()">&times;</button>
                </div>
                <div class="modal-body" id="modalBody"></div>
            </div>
        </div>

        <div class="container">
            <h1>DORA Market Proposal Review</h1>
            <div class="timestamp">
                Generated: {timestamp}<br>
                Proposal ID: <code>{proposal_id}</code>
            </div>

            <div class="summary">
                <div class="summary-item">
                    <div class="summary-label">Total Proposals</div>
                    <div class="summary-value">{total_proposals}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Market Updates</div>
                    <div class="summary-value">{len(update_proposals)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">New Candidates</div>
                    <div class="summary-value">{len(screener_proposals)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Total P&L (24h)</div>
                    <div class="summary-value {'green' if total_pnl >= 0 else 'red'}">${total_pnl:+,.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Total Fills</div>
                    <div class="summary-value">{total_fills:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Markets with Fills</div>
                    <div class="summary-value">{markets_with_fills} ({markets_with_fills_pct:.0f}%)</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Total Position</div>
                    <div class="summary-value">{total_position:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Current Quote Size</div>
                    <div class="summary-value">{total_current_quote_size:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Proposed Quote Size</div>
                    <div class="summary-value">{total_proposed_quote_size:,}</div>
                </div>
    """

    # Add action count summaries
    for action, count in sorted(action_counts.items()):
        action_label = action.replace('_', ' ').title()
        html += f"""
                <div class="summary-item">
                    <div class="summary-label">{action_label}</div>
                    <div class="summary-value">{count}</div>
                </div>
        """

    html += f"""
            </div>

            <div class="toggle-controls">
                <div class="toggle-label">Select Proposals to Approve:</div>
                <div class="toggle-buttons">
                    <button type="button" class="toggle-btn" onclick="selectAll()">Select All</button>
                    <button type="button" class="toggle-btn deselect" onclick="deselectAll()">Deselect All</button>
                </div>
            </div>

            <form method="POST" action="{proposal_id}/execute?signature={signature}&expiry={expiry}">
    """

    # Calculate event-level statistics and sort by P&L
    event_stats = {}
    for event_ticker, event_proposals in proposals_by_event.items():
        total_pnl = sum(p.get('metadata', {}).get('pnl_24h', 0) for p in event_proposals)
        total_fills = sum(p.get('metadata', {}).get('fill_count', 0) for p in event_proposals)
        event_update_count = len([p for p in event_proposals if p in update_proposals])
        event_screener_count = len([p for p in event_proposals if p in screener_proposals])

        # Count actions
        action_breakdown = defaultdict(int)
        for p in event_proposals:
            if p in update_proposals:
                action_breakdown[p.get('action', 'unknown')] += 1

        # Aggregate external market data (total market volume/trades)
        # Get event_title from first proposal's metadata
        event_title = None
        for p in event_proposals:
            event_title = p.get('metadata', {}).get('event_title')
            if event_title:
                break

        # Sum up total external market data
        total_market_trades = sum(p.get('metadata', {}).get('volume_24h_trades', 0) for p in event_proposals)
        total_market_volume = sum(p.get('metadata', {}).get('volume_24h_contracts', 0) for p in event_proposals)
        total_market_buy_trades = sum(p.get('metadata', {}).get('buy_volume_trades', 0) for p in event_proposals)
        total_market_buy_volume = sum(p.get('metadata', {}).get('buy_volume_contracts', 0) for p in event_proposals)
        total_market_sell_trades = sum(p.get('metadata', {}).get('sell_volume_trades', 0) for p in event_proposals)
        total_market_sell_volume = sum(p.get('metadata', {}).get('sell_volume_contracts', 0) for p in event_proposals)

        # Calculate DORA's capture percentage
        # Note: total_fills is DORA's fills, total_market_trades is total market trades
        capture_pct_trades = (total_fills / total_market_trades * 100) if total_market_trades > 0 else 0

        event_stats[event_ticker] = {
            'proposals': event_proposals,
            'total_pnl': total_pnl,
            'total_fills': total_fills,
            'update_count': event_update_count,
            'screener_count': event_screener_count,
            'action_breakdown': action_breakdown,
            'event_title': event_title,
            'total_market_trades': total_market_trades,
            'total_market_volume': total_market_volume,
            'total_market_buy_trades': total_market_buy_trades,
            'total_market_buy_volume': total_market_buy_volume,
            'total_market_sell_trades': total_market_sell_trades,
            'total_market_sell_volume': total_market_sell_volume,
            'capture_pct_trades': capture_pct_trades,
        }

    # Sort events by total P&L (descending - best first)
    sorted_events = sorted(event_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

    # Group and display proposals by event
    for event_ticker, stats in sorted_events:
        event_proposals = stats['proposals']
        total_pnl = stats['total_pnl']
        total_fills = stats['total_fills']
        event_update_count = stats['update_count']
        event_screener_count = stats['screener_count']
        action_breakdown = stats['action_breakdown']
        event_title = stats.get('event_title', event_ticker)

        # External market data
        total_market_trades = stats['total_market_trades']
        total_market_volume = stats['total_market_volume']
        total_market_buy_trades = stats['total_market_buy_trades']
        total_market_buy_volume = stats['total_market_buy_volume']
        total_market_sell_trades = stats['total_market_sell_trades']
        total_market_sell_volume = stats['total_market_sell_volume']
        capture_pct_trades = stats['capture_pct_trades']

        # Format P&L with color
        pnl_color = '#28a745' if total_pnl >= 0 else '#dc3545'
        pnl_display = f"${total_pnl:+,.2f}" if total_pnl != 0 else "$0.00"

        html += f"""
            <div class="event-section">
                <div class="event-header">
                    <div class="event-header-top">
                        <div class="event-title-group">
                            <div class="event-title">{event_title or event_ticker}</div>
                            <div class="event-ticker">{event_ticker} • {len(event_proposals)} proposal(s)</div>
                        </div>
                        <div class="event-stats">
                            <div class="event-stat">
                                <span class="event-stat-label">P&L (24h)</span>
                                <span class="event-stat-value" style="color: {pnl_color};">{pnl_display}</span>
                            </div>
                            <div class="event-stat">
                                <span class="event-stat-label">DORA Fills</span>
                                <span class="event-stat-value">{total_fills:,}</span>
                            </div>
                            <div class="event-stat">
                                <span class="event-stat-label">Updates</span>
                                <span class="event-stat-value">{event_update_count}</span>
                            </div>
                            <div class="event-stat">
                                <span class="event-stat-label">New</span>
                                <span class="event-stat-value">{event_screener_count}</span>
                            </div>
                        </div>
                    </div>
                    <div class="event-external-data">
                        <div class="external-metric">
                            <span class="external-metric-label">Total Market Trades (24h)</span>
                            <span class="external-metric-value">{total_market_trades:,}</span>
                        </div>
                        <div class="external-metric">
                            <span class="external-metric-label">Total Market Volume (24h)</span>
                            <span class="external-metric-value">{total_market_volume:,}</span>
                        </div>
                        <div class="external-metric">
                            <span class="external-metric-label">Buy Trades / Volume</span>
                            <span class="external-metric-value">{total_market_buy_trades:,} / {total_market_buy_volume:,}</span>
                        </div>
                        <div class="external-metric">
                            <span class="external-metric-label">Sell Trades / Volume</span>
                            <span class="external-metric-value">{total_market_sell_trades:,} / {total_market_sell_volume:,}</span>
                        </div>
                        <div class="external-metric">
                            <span class="external-metric-label">DORA Capture % (Trades)</span>
                            <span class="external-metric-value capture-pct">{capture_pct_trades:.1f}%</span>
                        </div>
                    </div>
                </div>

                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th class="checkbox-cell" rowspan="2">✓</th>
                                <th rowspan="2">Market ID</th>
                                <th colspan="9" class="data-section-header internal">Internal Data (DORA Bot)</th>
                                <th colspan="2" class="data-section-header external">External Data (Market)</th>
                            </tr>
                            <tr>
                                <!-- Internal Data Columns -->
                                <th>Proposed Action</th>
                                <th>Override Action</th>
                                <th>Created</th>
                                <th>DORA Fills</th>
                                <th>Last Fill</th>
                                <th>Position</th>
                                <th>Current Size</th>
                                <th>Min Spread</th>
                                <th>P&L (24h)</th>
                                <!-- External Data Columns -->
                                <th>Total Market Fills</th>
                                <th>Market Spread</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        for p in event_proposals:
            market_id = p['market_id']
            source = 'Update' if p in update_proposals else 'Screener'
            action = p.get('action', 'activate')
            reason = p.get('reason', '')[:100]
            # Get full rationale for tooltip (not truncated)
            full_rationale = p.get('reason', 'No rationale provided')

            # Get P&L
            pnl = p.get('metadata', {}).get('pnl_24h', 0)
            pnl_class = 'pnl-positive' if pnl >= 0 else 'pnl-negative'
            pnl_str = f"${pnl:+.2f}" if pnl != 0 else "—"

            # Get fill count
            fill_count = p.get('metadata', {}).get('fill_count', 0)
            if fill_count == 0:
                fill_count = p.get('metadata', {}).get('fill_count_24h', 0)
            fill_str = str(fill_count) if fill_count > 0 else "—"

            # Get position
            position_qty = p.get('metadata', {}).get('position_qty', 0)
            position_str = str(position_qty) if position_qty != 0 else "—"
            position_class = 'pnl-positive' if position_qty > 0 else ('pnl-negative' if position_qty < 0 else '')

            # Get created date
            created_at = p.get('metadata', {}).get('created_at')
            if created_at:
                # Parse ISO date string and format as relative time
                try:
                    from datetime import datetime, timezone
                    created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    age = now - created_dt
                    if age.days > 0:
                        created_str = f"{age.days}d ago"
                    elif age.seconds >= 3600:
                        created_str = f"{age.seconds // 3600}h ago"
                    else:
                        created_str = f"{age.seconds // 60}m ago"
                except:
                    created_str = "—"
            else:
                created_str = "—"

            # Get last fill date
            last_fill_at = p.get('metadata', {}).get('last_fill_at')
            last_fill_str = "—"
            last_fill_class = ""
            if last_fill_at:
                try:
                    from datetime import datetime, timezone
                    last_fill_dt = datetime.fromisoformat(last_fill_at.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    age = now - last_fill_dt

                    # Format the time difference
                    if age.days > 0:
                        last_fill_str = f"{age.days}d ago"
                    elif age.seconds >= 3600:
                        last_fill_str = f"{age.seconds // 3600}h ago"
                    else:
                        last_fill_str = f"{age.seconds // 60}m ago"

                    # Mark as stale if > 48 hours (2 days)
                    if age.days >= 2:
                        last_fill_class = "fill-stale"
                except:
                    last_fill_str = "—"

            # Get current config
            current_config = p.get('current_config', {})
            current_size = current_config.get('quote_size', '—')
            current_spread = current_config.get('min_spread', '—')

            # Format current spread as percentage if it's a number
            if isinstance(current_spread, (int, float)):
                spread_str = f"{current_spread:.2%}"
            else:
                spread_str = str(current_spread)

            # Format current size
            size_str = str(current_size) if current_size != '—' else '—'

            changes = p.get('proposed_changes', {})
            changes_parts = []
            for k, v in changes.items():
                if k == 'enabled':
                    changes_parts.append(f"<strong>{k}={v}</strong>")
                else:
                    changes_parts.append(f"{k}={v}")
            changes_str = ", ".join(changes_parts) if changes_parts else "—"

            # Action dropdown options
            action_options = ['expand', 'scale_down', 'exit', 'activate_sibling', 'reset_defaults']
            action_dropdown = f'<select name="action_{market_id}" class="action-dropdown">'
            action_dropdown += f'<option value="" selected>Use Proposed ({action})</option>'
            for opt in action_options:
                opt_label = opt.replace('_', ' ').title()
                action_dropdown += f'<option value="{opt}">{opt_label}</option>'
            action_dropdown += '</select>'

            # External market data
            metadata = p.get('metadata', {})
            volume_24h_trades = metadata.get('volume_24h_trades', 0)
            buy_volume_trades = metadata.get('buy_volume_trades', 0)
            buy_volume_contracts = metadata.get('buy_volume_contracts', 0)
            sell_volume_trades = metadata.get('sell_volume_trades', 0)
            sell_volume_contracts = metadata.get('sell_volume_contracts', 0)

            # Format Total Market Fills with modal
            total_fills_str = f"{volume_24h_trades:,}" if volume_24h_trades > 0 else "—"
            if volume_24h_trades > 0:
                modal_content = f'''<div class="modal-row"><span class="modal-label">Buy Trades:</span><span class="modal-value">{buy_volume_trades:,}</span></div><div class="modal-row"><span class="modal-label">Buy Volume:</span><span class="modal-value">{buy_volume_contracts:,}</span></div><div class="modal-row"><span class="modal-label">Sell Trades:</span><span class="modal-value">{sell_volume_trades:,}</span></div><div class="modal-row"><span class="modal-label">Sell Volume:</span><span class="modal-value">{sell_volume_contracts:,}</span></div>'''
                modal_content_escaped = html_lib.escape(modal_content, quote=True)
                total_fills_html = f'''<span class="clickable-info" data-modal-title="Total Market Fills (24h)" data-modal-content="{modal_content_escaped}" onclick="showModalFromData(this)">{total_fills_str}</span>'''
            else:
                total_fills_html = total_fills_str

            # Get bid/ask data for Market Spread
            yes_bid = metadata.get('yes_bid')
            yes_ask = metadata.get('yes_ask')
            previous_yes_bid = metadata.get('previous_yes_bid')
            previous_yes_ask = metadata.get('previous_yes_ask')
            current_spread = metadata.get('current_spread')
            spread_24h_ago = metadata.get('spread_24h_ago')

            # Format Market Spread with modal
            if current_spread is not None:
                market_spread_str = f"{current_spread:.2f}¢"
                modal_content = f'''<div class="modal-section-title">Current (Now)</div><div class="modal-row"><span class="modal-label">Bid:</span><span class="modal-value">{yes_bid}¢</span></div><div class="modal-row"><span class="modal-label">Ask:</span><span class="modal-value">{yes_ask}¢</span></div><div class="modal-row"><span class="modal-label">Spread:</span><span class="modal-value">{current_spread:.2f}¢</span></div>'''
                if spread_24h_ago is not None:
                    modal_content += f'''<div class="modal-section-title">24 Hours Ago</div><div class="modal-row"><span class="modal-label">Bid:</span><span class="modal-value">{previous_yes_bid}¢</span></div><div class="modal-row"><span class="modal-label">Ask:</span><span class="modal-value">{previous_yes_ask}¢</span></div><div class="modal-row"><span class="modal-label">Spread:</span><span class="modal-value">{spread_24h_ago:.2f}¢</span></div>'''
                modal_content_escaped = html_lib.escape(modal_content, quote=True)
                market_spread_html = f'''<span class="clickable-info" data-modal-title="Market Spread" data-modal-content="{modal_content_escaped}" onclick="showModalFromData(this)">{market_spread_str}</span>'''
            else:
                market_spread_str = "—"
                market_spread_html = market_spread_str

            html += f"""
                            <tr>
                                <td class="checkbox-cell">
                                    <input type="checkbox" name="selected" value="{market_id}" checked>
                                </td>
                                <td class="market-id">{market_id}</td>
                                <!-- Internal Data Columns -->
                                <td>
                                    <span class="clickable-info action-badge action-{action.replace(' ', '_')}"
                                          data-modal-title="Rationale"
                                          data-modal-content="{html_lib.escape('<div class="modal-rationale">' + full_rationale + '</div>', quote=True)}"
                                          onclick="showModalFromData(this)">
                                        {action}
                                    </span>
                                </td>
                                <td>{action_dropdown}</td>
                                <td>{created_str}</td>
                                <td>{fill_str}</td>
                                <td class="{last_fill_class}">{last_fill_str}</td>
                                <td class="{position_class}">{position_str}</td>
                                <td>{size_str}</td>
                                <td>{spread_str}</td>
                                <td class="{pnl_class}">{pnl_str}</td>
                                <!-- External Data Columns -->
                                <td>{total_fills_html}</td>
                                <td>{market_spread_html}</td>
                            </tr>
            """

        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        """

    # Action buttons
    html += """
            <div class="actions">
                <button type="submit" name="action" value="approve_selected" class="button">
                    ✓ Approve Selected Proposals
                </button>
                <button type="submit" name="action" value="reject_all" class="button button-reject">
                    ✕ Reject All Proposals
                </button>
            </div>
            </form>
        </div>
    </body>
    </html>
    """

    return html


def generate_screener_candidates_html(
    proposal_id: str,
    screener_proposals: List[Dict],
    signature: str,
    expiry: str
) -> str:
    """Generate card-based HTML page for reviewing market screener candidates.

    Args:
        proposal_id: UUID of proposal batch
        screener_proposals: List of market screener proposals
        signature: URL signature for authentication
        expiry: URL expiry timestamp

    Returns:
        HTML string with card-based layout
    """
    from datetime import datetime
    import html as html_lib

    total_candidates = len(screener_proposals)
    timestamp = datetime.utcnow().strftime("%B %d, %Y %H:%M UTC")

    # Calculate summary stats
    total_volume = sum(p.get('metadata', {}).get('volume_24h', 0) for p in screener_proposals)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DORA Screener Candidates - {timestamp}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
                padding: 20px;
            }}

            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 28px;
            }}

            .timestamp {{
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 25px;
            }}

            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
                padding: 20px;
                background: #ecf0f1;
                border-radius: 6px;
            }}

            .summary-item {{
                text-align: center;
            }}

            .summary-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .summary-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 5px;
            }}

            .summary-value.green {{
                color: #28a745;
            }}

            .toggle-controls {{
                background: #e8f4f8;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 15px 20px;
                margin: 20px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .toggle-label {{
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
            }}

            .toggle-buttons {{
                display: flex;
                gap: 12px;
            }}

            .toggle-btn {{
                padding: 8px 18px;
                border: 2px solid #3498db;
                background: white;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                color: #3498db;
                transition: all 0.2s;
            }}

            .toggle-btn:hover {{
                background: #e8f4f8;
            }}

            .candidate-card {{
                background: white;
                padding: 20px;
                margin-bottom: 18px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                transition: all 0.2s;
                border-left: 5px solid #3498db;
            }}

            .candidate-card:hover {{
                box-shadow: 0 4px 8px rgba(0,0,0,0.12);
            }}

            .candidate-card.selected {{
                background: #f0f9ff;
                border-left-color: #28a745;
            }}

            .candidate-header {{
                display: flex;
                justify-content: space-between;
                align-items: start;
                margin-bottom: 12px;
            }}

            .candidate-title-section {{
                flex: 1;
            }}

            .candidate-event-name {{
                font-size: 11px;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}

            .candidate-title {{
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 5px;
                line-height: 1.4;
            }}

            .candidate-id {{
                font-size: 11px;
                font-family: 'Courier New', monospace;
                color: #7f8c8d;
            }}

            .card-checkbox {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .card-checkbox input[type="checkbox"] {{
                width: 22px;
                height: 22px;
                cursor: pointer;
            }}

            .card-checkbox-label {{
                font-size: 13px;
                font-weight: 600;
                color: #28a745;
                cursor: pointer;
            }}

            .candidate-reason {{
                padding: 14px;
                border-radius: 6px;
                font-size: 13px;
                line-height: 1.7;
                margin: 14px 0;
                background: #f0f9ff;
                border-left: 3px solid #3498db;
                color: #1e3a5f;
            }}

            .candidate-metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 18px;
                margin-top: 14px;
            }}

            .candidate-metric {{
                display: flex;
                flex-direction: column;
            }}

            .candidate-metric-label {{
                font-size: 10px;
                color: #7f8c8d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}

            .candidate-metric-value {{
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                font-family: 'Courier New', monospace;
            }}

            .metric-pnl {{
                font-weight: 700;
            }}

            .metric-pnl.positive {{
                color: #28a745;
            }}

            .metric-pnl.negative {{
                color: #dc3545;
            }}

            .actions {{
                position: sticky;
                bottom: 0;
                background: white;
                padding: 20px 0;
                border-top: 2px solid #ddd;
                margin-top: 30px;
                display: flex;
                gap: 15px;
                justify-content: center;
            }}

            .button {{
                padding: 12px 30px;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                background: #28a745;
                color: white;
            }}

            .button:hover {{
                background: #218838;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}

            .button-reject {{
                background: #dc3545;
            }}

            .button-reject:hover {{
                background: #c82333;
            }}

            .info-banner {{
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
                font-size: 14px;
                color: #856404;
            }}

            .info-banner strong {{
                color: #664d03;
            }}

            /* Clickable info styles */
            .clickable-info {{
                cursor: pointer;
                border-bottom: 1px dotted #999;
                display: inline-block;
            }}

            .clickable-info:hover {{
                border-bottom: 2px solid #3498db;
            }}

            /* Modal styles */
            .modal {{
                display: none;
                position: fixed;
                z-index: 100000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.6);
                animation: fadeIn 0.2s;
            }}

            .modal.show {{
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .modal-content {{
                background-color: #2c3e50;
                color: #fff;
                padding: 25px;
                border-radius: 8px;
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                animation: slideIn 0.3s;
            }}

            .modal-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid rgba(255,255,255,0.2);
            }}

            .modal-title {{
                font-size: 18px;
                font-weight: 600;
                color: #ecf0f1;
            }}

            .modal-close {{
                font-size: 28px;
                font-weight: bold;
                color: #95a5a6;
                cursor: pointer;
                background: none;
                border: none;
                padding: 0;
                line-height: 1;
            }}

            .modal-close:hover {{
                color: #fff;
            }}

            .modal-body {{
                font-size: 14px;
                line-height: 1.8;
            }}

            .modal-row {{
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}

            .modal-row:last-child {{
                border-bottom: none;
            }}

            .modal-label {{
                font-weight: 600;
                color: #ecf0f1;
            }}

            .modal-value {{
                color: #3498db;
                font-family: 'Courier New', monospace;
            }}

            .modal-section-title {{
                font-weight: 600;
                margin: 15px 0 10px 0;
                color: #ecf0f1;
                font-size: 15px;
                border-top: 1px solid rgba(255,255,255,0.2);
                padding-top: 15px;
            }}

            .modal-section-title:first-child {{
                margin-top: 0;
                border-top: none;
                padding-top: 0;
            }}

            .modal-rationale {{
                white-space: pre-wrap;
                line-height: 1.6;
                color: #ecf0f1;
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            @keyframes slideIn {{
                from {{
                    transform: translateY(-50px);
                    opacity: 0;
                }}
                to {{
                    transform: translateY(0);
                    opacity: 1;
                }}
            }}
        </style>
    </head>
    <body>
        <!-- Modal -->
        <div id="infoModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <span class="modal-title" id="modalTitle"></span>
                    <button class="modal-close" onclick="closeModal()">&times;</button>
                </div>
                <div class="modal-body" id="modalBody"></div>
            </div>
        </div>

        <div class="container">
            <h1>📋 DORA Market Screener Candidates</h1>
            <div class="timestamp">Generated: {timestamp}</div>

            <div class="summary">
                <div class="summary-item">
                    <div class="summary-label">Total Candidates</div>
                    <div class="summary-value green">{total_candidates}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Total Volume 24h</div>
                    <div class="summary-value">{total_volume:,}</div>
                </div>
            </div>

            <div class="info-banner">
                <strong>New Market Opportunities:</strong> Review these candidates identified by the market screener.
                Check the boxes next to markets you want to enable and click "Approve Selected Proposals" to activate them.
            </div>

            <form method="POST" action="{proposal_id}/execute?signature={signature}&expiry={expiry}">
                <div class="toggle-controls">
                    <div class="toggle-label">Quick Selection</div>
                    <div class="toggle-buttons">
                        <button type="button" class="toggle-btn" onclick="selectAll()">Select All</button>
                        <button type="button" class="toggle-btn" onclick="deselectAll()">Deselect All</button>
                    </div>
                </div>

                <div id="candidates-container">
    """

    # Generate cards for each screener candidate
    for proposal in screener_proposals:
        market_id = proposal.get('market_id', '')
        metadata = proposal.get('metadata', {})

        # Extract metadata
        title = metadata.get('title', '')
        subtitle = metadata.get('subtitle', '')
        event_ticker = metadata.get('event_ticker', '')
        event_title = metadata.get('event_title', event_ticker)

        # Enriched market data
        volume_24h_trades = metadata.get('volume_24h_trades', 0)
        volume_24h_contracts = metadata.get('volume_24h_contracts', 0)
        buy_volume_trades = metadata.get('buy_volume_trades', 0)
        buy_volume_contracts = metadata.get('buy_volume_contracts', 0)
        sell_volume_trades = metadata.get('sell_volume_trades', 0)
        sell_volume_contracts = metadata.get('sell_volume_contracts', 0)

        yes_bid = metadata.get('yes_bid')
        yes_ask = metadata.get('yes_ask')
        previous_yes_bid = metadata.get('previous_yes_bid')
        previous_yes_ask = metadata.get('previous_yes_ask')
        current_spread = metadata.get('current_spread')
        spread_24h_ago = metadata.get('spread_24h_ago')

        info_risk = metadata.get('info_risk')
        info_risk_rationale = metadata.get('info_risk_rationale', '')
        bid_depth = metadata.get('bid_depth_5c', 0)
        ask_depth = metadata.get('ask_depth_5c', 0)

        # Historical P&L (if previously disabled)
        pnl = metadata.get('pnl_24h', 0)

        # Proposed config
        proposed = proposal.get('proposed_changes', {})
        quote_size = proposed.get('quote_size', 5)
        min_spread = proposed.get('min_spread', 0.04)

        # Format Total Market Trades with modal
        if volume_24h_trades > 0:
            modal_content = f'''<div class="modal-row"><span class="modal-label">Total Trades:</span><span class="modal-value">{volume_24h_trades:,}</span></div><div class="modal-row"><span class="modal-label">Total Volume:</span><span class="modal-value">{volume_24h_contracts:,}</span></div><div class="modal-row"><span class="modal-label">Buy Trades:</span><span class="modal-value">{buy_volume_trades:,}</span></div><div class="modal-row"><span class="modal-label">Buy Volume:</span><span class="modal-value">{buy_volume_contracts:,}</span></div><div class="modal-row"><span class="modal-label">Sell Trades:</span><span class="modal-value">{sell_volume_trades:,}</span></div><div class="modal-row"><span class="modal-label">Sell Volume:</span><span class="modal-value">{sell_volume_contracts:,}</span></div>'''
            modal_content_escaped = html_lib.escape(modal_content, quote=True)
            volume_str = f'''<span class="clickable-info" data-modal-title="Market Volume (24h)" data-modal-content="{modal_content_escaped}" onclick="showModalFromData(this)">{volume_24h_trades:,} trades</span>'''
        else:
            volume_str = "—"

        # Format Market Spread with modal
        if current_spread is not None:
            modal_content = f'''<div class="modal-section-title">Current (Now)</div><div class="modal-row"><span class="modal-label">Bid:</span><span class="modal-value">{yes_bid}¢</span></div><div class="modal-row"><span class="modal-label">Ask:</span><span class="modal-value">{yes_ask}¢</span></div><div class="modal-row"><span class="modal-label">Spread:</span><span class="modal-value">{current_spread:.2f}¢</span></div>'''
            if spread_24h_ago is not None:
                modal_content += f'''<div class="modal-section-title">24 Hours Ago</div><div class="modal-row"><span class="modal-label">Bid:</span><span class="modal-value">{previous_yes_bid}¢</span></div><div class="modal-row"><span class="modal-label">Ask:</span><span class="modal-value">{previous_yes_ask}¢</span></div><div class="modal-row"><span class="modal-label">Spread:</span><span class="modal-value">{spread_24h_ago:.2f}¢</span></div>'''
            modal_content_escaped = html_lib.escape(modal_content, quote=True)
            spread_str = f'''<span class="clickable-info" data-modal-title="Market Spread" data-modal-content="{modal_content_escaped}" onclick="showModalFromData(this)">{current_spread:.2f}¢</span>'''
        else:
            spread_str = "—"

        bid_ask_str = f"{yes_bid}¢ / {yes_ask}¢" if yes_bid is not None and yes_ask is not None else "—"
        info_risk_str = f"{info_risk:.0f}%" if info_risk is not None else "—"
        depth_str = f"{bid_depth:,} / {ask_depth:,}" if bid_depth or ask_depth else "—"

        # Historical P&L display
        pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
        pnl_str = f"${pnl:+.2f}" if pnl != 0 else "—"

        # Reason
        reason = proposal.get('reason', 'New market opportunity identified by screener')

        # Build commentary with rationale if available
        commentary = reason
        if info_risk_rationale:
            commentary = f"{reason}<br><br><em>Risk Assessment:</em> {info_risk_rationale[:200]}"

        html += f"""
                    <div class="candidate-card" data-market-id="{market_id}">
                        <div class="candidate-header">
                            <div class="candidate-title-section">
                                <div class="candidate-event-name">{event_title or event_ticker or 'New Market'}</div>
                                <div class="candidate-title">{title or market_id}</div>
                                <div class="candidate-id">{event_ticker} • {market_id}</div>
                            </div>
                            <div class="card-checkbox">
                                <input type="checkbox" name="selected" value="{market_id}" id="check_{market_id}" checked
                                    onchange="this.closest('.candidate-card').classList.toggle('selected', this.checked)">
                                <label for="check_{market_id}" class="card-checkbox-label">✓ Approve</label>
                            </div>
                        </div>
                        <div class="candidate-reason">{commentary}</div>
                        <div class="candidate-metrics">
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Volume 24h</div>
                                <div class="candidate-metric-value">{volume_str}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Buy / Sell</div>
                                <div class="candidate-metric-value">{buy_sell_str}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Bid / Ask</div>
                                <div class="candidate-metric-value">{bid_ask_str}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Spread</div>
                                <div class="candidate-metric-value">{spread_str}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Order Depth (±5¢)</div>
                                <div class="candidate-metric-value">{depth_str}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Info Risk</div>
                                <div class="candidate-metric-value">{info_risk_str}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Quote Size</div>
                                <div class="candidate-metric-value">{quote_size}</div>
                            </div>
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Min Spread</div>
                                <div class="candidate-metric-value">{min_spread:.2%}</div>
                            </div>"""

        # Show historical P&L if available
        if pnl != 0:
            html += f"""
                            <div class="candidate-metric">
                                <div class="candidate-metric-label">Historical P&L</div>
                                <div class="candidate-metric-value metric-pnl {pnl_class}">{pnl_str}</div>
                            </div>"""

        html += """
                        </div>
                    </div>
        """

    html += """
                </div>

                <div class="actions">
                    <button type="submit" name="action" value="approve_selected" class="button">
                        ✓ Approve Selected Proposals
                    </button>
                    <button type="submit" name="action" value="reject_all" class="button button-reject">
                        ✕ Reject All Proposals
                    </button>
                </div>
            </form>
        </div>

        <script>
            function selectAll() {
                const checkboxes = document.querySelectorAll('input[name="selected"]');
                checkboxes.forEach(checkbox => {
                    checkbox.checked = true;
                    checkbox.closest('.candidate-card').classList.add('selected');
                });
            }

            function deselectAll() {
                const checkboxes = document.querySelectorAll('input[name="selected"]');
                checkboxes.forEach(checkbox => {
                    checkbox.checked = false;
                    checkbox.closest('.candidate-card').classList.remove('selected');
                });
            }

            // Initialize card states
            document.addEventListener('DOMContentLoaded', function() {
                const checkboxes = document.querySelectorAll('input[name="selected"]');
                checkboxes.forEach(checkbox => {
                    if (checkbox.checked) {
                        checkbox.closest('.candidate-card').classList.add('selected');
                    }
                });
            });

            function showModal(title, content) {
                const modal = document.getElementById('infoModal');
                const modalTitle = document.getElementById('modalTitle');
                const modalBody = document.getElementById('modalBody');

                modalTitle.textContent = title;
                modalBody.innerHTML = content;
                modal.classList.add('show');
            }

            function showModalFromData(element) {
                const title = element.getAttribute('data-modal-title');
                const content = element.getAttribute('data-modal-content');
                showModal(title, content);
            }

            function closeModal() {
                const modal = document.getElementById('infoModal');
                modal.classList.remove('show');
            }

            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('infoModal');
                if (event.target === modal) {
                    closeModal();
                }
            }

            // Close modal with Escape key
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    closeModal();
                }
            });
        </script>
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
            action_overrides = data.get('action_overrides', {})
        else:
            # Form data
            parsed = parse_qs(body)
            selected_markets = parsed.get('selected', [])
            action = parsed.get('action', [''])[0]
            approve_all = (action == 'approve_all')

            # Extract action overrides from form data
            # Form fields named "action_{market_id}" contain override actions
            action_overrides = {}
            for key, values in parsed.items():
                if key.startswith('action_') and values and values[0]:
                    market_id = key[7:]  # Remove 'action_' prefix
                    override_action = values[0]
                    if override_action:  # Only store if not empty
                        action_overrides[market_id] = override_action

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
        action_overrides = {}

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
            # Check if there's an action override for this market
            market_id = proposal['market_id']
            override_action = action_overrides.get(market_id)

            result = execute_single_proposal(proposal, db_client, override_action)
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
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Execution Complete</title>
    </head>
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
        'headers': {'Content-Type': 'text/html; charset=utf-8'},
        'body': html_response
    }


def execute_single_proposal(
    proposal: Dict[str, Any],
    db_client: DynamoDBClient,
    override_action: str = None
) -> Dict[str, Any]:
    """Execute a single proposal.

    Args:
        proposal: Proposal dict from DynamoDB
        db_client: DynamoDBClient instance
        override_action: Optional action to use instead of the proposed action

    Returns:
        Result dict with success status
    """
    market_id = proposal['market_id']
    # Use override action if provided, otherwise use the proposed action
    action = override_action if override_action else proposal['action']
    proposed_changes = proposal.get('proposed_changes', {})

    logger.info(f"Executing {action} for {market_id}" +
                (f" (overridden from {proposal['action']})" if override_action else ""))

    try:
        if action == 'new_market':
            # Create new market config as dict
            # Get event_ticker from proposal metadata
            metadata = proposal.get('metadata', {})
            event_ticker = metadata.get('event_ticker', '')

            market_config = {
                'market_id': market_id,
                'quote_size': proposed_changes.get('quote_size', 5),
                'max_inventory_yes': proposed_changes.get('max_inventory_yes', 5),
                'max_inventory_no': proposed_changes.get('max_inventory_no', 5),
                'min_spread': proposed_changes.get('min_spread', 0.04),
                'enabled': proposed_changes.get('enabled', True),
                'inventory_skew_factor': 0.5,
                'event_ticker': event_ticker,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            # Convert floats to Decimal for DynamoDB
            market_config = convert_floats_to_decimal(market_config)
            success = db_client.put_market_config(market_config)

            if not success:
                return {'market_id': market_id, 'success': False, 'error': 'Failed to save market config to DynamoDB'}

        else:
            # Update existing market config
            existing = db_client.get_market_config(market_id)
            if not existing:
                return {'market_id': market_id, 'success': False, 'error': 'Market config not found'}

            # If action was overridden, recalculate changes based on the new action
            if override_action:
                changes_to_apply = {}
                current_quote_size = existing.get('quote_size', 5)

                if action == 'exit':
                    changes_to_apply = {'enabled': False}
                elif action == 'scale_down':
                    changes_to_apply = {
                        'quote_size': max(1, int(current_quote_size * 0.5))
                    }
                elif action == 'expand':
                    changes_to_apply = {
                        'quote_size': int(current_quote_size * 1.5)
                    }
                elif action == 'reset_defaults':
                    changes_to_apply = {
                        'enabled': True,
                        'quote_size': 5,
                        'max_inventory_yes': 5,
                        'max_inventory_no': 5,
                        'min_spread': 0.04,
                    }
                elif action == 'activate_sibling':
                    # This action is complex and requires sibling market info
                    # For now, just enable the market
                    changes_to_apply = {'enabled': True}
                else:
                    # Unknown action, use proposed changes
                    changes_to_apply = proposed_changes
            else:
                # Use the original proposed changes
                changes_to_apply = proposed_changes

            # Apply changes to existing config
            for key, value in changes_to_apply.items():
                existing[key] = value

            # Update the updated_at timestamp
            existing['updated_at'] = datetime.now(timezone.utc).isoformat()

            # Convert floats to Decimal for DynamoDB
            existing = convert_floats_to_decimal(existing)
            success = db_client.put_market_config(existing)

            if not success:
                return {'market_id': market_id, 'success': False, 'error': 'Failed to update market config in DynamoDB'}

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
