"""
Proposal Manager for DynamoDB Operations

Handles saving, querying, and updating market proposals in DynamoDB.
"""
import boto3
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from decimal import Decimal

logger = logging.getLogger(__name__)


def _serialize_for_dynamodb(obj: Any) -> Any:
    """Convert Python floats to Decimal for DynamoDB compatibility.

    DynamoDB doesn't support native Python float types - they must be Decimal.
    This function recursively converts floats in nested dicts/lists.

    Args:
        obj: Any Python object (dict, list, float, etc.)

    Returns:
        Object with floats converted to Decimal
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _serialize_for_dynamodb(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_dynamodb(item) for item in obj]
    return obj


class ProposalManager:
    """Manages market proposals in DynamoDB."""

    def __init__(self, region: str = "us-east-1", environment: str = "prod"):
        """Initialize DynamoDB client and table name.

        Args:
            region: AWS region
            environment: 'demo' or 'prod'
        """
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table_name = f"dora_market_proposals_{environment}"
        self.table = self.dynamodb.Table(self.table_name)
        self.environment = environment

    def create_proposal_batch(
        self,
        proposals: List[Dict[str, Any]],
        ttl_hours: int = 12
    ) -> str:
        """Save a batch of proposals to DynamoDB with a single proposal_id.

        Args:
            proposals: List of proposal dicts with keys:
                - market_id: Market ticker
                - proposal_source: "market_update" | "market_screener"
                - action: "exit" | "scale_down" | "expand" | "activate_sibling" | "new_market"
                - reason: Explanation for the proposal
                - current_config: Dict of current market_config values (optional)
                - proposed_changes: Dict of proposed new values
                - metadata: Additional context (P&L, volume, etc.)
            ttl_hours: Hours until proposals auto-expire (default: 12)

        Returns:
            proposal_id: UUID for this batch
        """
        proposal_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        ttl_timestamp = int(time.time()) + (ttl_hours * 3600)

        logger.info(f"Creating proposal batch {proposal_id} with {len(proposals)} proposals")

        # Batch write items
        with self.table.batch_writer() as batch:
            for proposal in proposals:
                item = {
                    'proposal_id': proposal_id,
                    'market_id': proposal['market_id'],
                    'proposal_source': proposal['proposal_source'],
                    'action': proposal['action'],
                    'status': 'pending',
                    'created_at': created_at,
                    'environment': self.environment,
                    'ttl_timestamp': ttl_timestamp,
                    'reason': proposal.get('reason', ''),
                    'current_config': proposal.get('current_config', {}),
                    'proposed_changes': proposal.get('proposed_changes', {}),
                    'metadata': proposal.get('metadata', {}),
                }
                # Convert floats to Decimal for DynamoDB compatibility
                item = _serialize_for_dynamodb(item)
                batch.put_item(Item=item)

        logger.info(f"Successfully saved {len(proposals)} proposals to DynamoDB")
        return proposal_id

    def get_proposals_by_id(
        self,
        proposal_id: str,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all proposals for a given proposal_id.

        Args:
            proposal_id: UUID of the proposal batch
            status_filter: Optional status filter ('pending', 'approved', 'rejected', 'executed', 'failed')

        Returns:
            List of proposal items from DynamoDB
        """
        try:
            # Query by proposal_id
            if status_filter:
                response = self.table.query(
                    KeyConditionExpression='proposal_id = :pid',
                    FilterExpression='#status = :status',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':pid': proposal_id,
                        ':status': status_filter
                    }
                )
            else:
                response = self.table.query(
                    KeyConditionExpression='proposal_id = :pid',
                    ExpressionAttributeValues={':pid': proposal_id}
                )

            proposals = response.get('Items', [])
            logger.info(f"Retrieved {len(proposals)} proposals for {proposal_id}")
            return proposals

        except Exception as e:
            logger.error(f"Error retrieving proposals {proposal_id}: {e}")
            return []

    def update_proposal_status(
        self,
        proposal_id: str,
        market_id: str,
        new_status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Update the status of a single proposal.

        Args:
            proposal_id: UUID of the proposal batch
            market_id: Market ticker
            new_status: New status ('approved', 'rejected', 'executed', 'failed')
            error_message: Optional error message if status is 'failed'

        Returns:
            True if successful
        """
        try:
            update_expression = 'SET #status = :status'
            expression_values = {':status': new_status}
            expression_names = {'#status': 'status'}

            # Add timestamp for approved/executed
            if new_status == 'approved':
                update_expression += ', approved_at = :timestamp'
                expression_values[':timestamp'] = datetime.now(timezone.utc).isoformat()
            elif new_status in ['executed', 'failed']:
                update_expression += ', executed_at = :timestamp'
                expression_values[':timestamp'] = datetime.now(timezone.utc).isoformat()

            # Add error message if failed
            if error_message:
                update_expression += ', error_message = :error'
                expression_values[':error'] = error_message

            self.table.update_item(
                Key={
                    'proposal_id': proposal_id,
                    'market_id': market_id
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values
            )

            logger.info(f"Updated proposal {proposal_id}/{market_id} to status: {new_status}")
            return True

        except Exception as e:
            logger.error(f"Error updating proposal {proposal_id}/{market_id}: {e}")
            return False

    def get_pending_proposals(self, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """Get all pending proposals (for monitoring/admin purposes).

        Args:
            max_age_hours: Only return proposals created within this many hours

        Returns:
            List of pending proposal items
        """
        try:
            # Calculate cutoff timestamp
            cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
            cutoff_iso = datetime.fromtimestamp(cutoff_time, tz=timezone.utc).isoformat()

            # Query using GSI
            response = self.table.query(
                IndexName='status-created_at-index',
                KeyConditionExpression='#status = :status AND created_at > :cutoff',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'pending',
                    ':cutoff': cutoff_iso
                }
            )

            proposals = response.get('Items', [])
            logger.info(f"Retrieved {len(proposals)} pending proposals")
            return proposals

        except Exception as e:
            logger.error(f"Error querying pending proposals: {e}")
            return []

    def batch_update_status(
        self,
        proposal_id: str,
        market_ids: List[str],
        new_status: str
    ) -> int:
        """Update status for multiple proposals at once.

        Args:
            proposal_id: UUID of the proposal batch
            market_ids: List of market IDs to update
            new_status: New status for all proposals

        Returns:
            Count of successfully updated proposals
        """
        success_count = 0
        for market_id in market_ids:
            if self.update_proposal_status(proposal_id, market_id, new_status):
                success_count += 1
        return success_count
