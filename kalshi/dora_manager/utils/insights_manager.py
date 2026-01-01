"""
Insights Manager for DynamoDB Operations

Handles saving and querying AI-generated insights for events and markets.
"""
import boto3
import logging
import time
from typing import Dict, Any, Optional
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


class InsightsManager:
    """Manages AI insights in DynamoDB."""

    def __init__(self, region: str = "us-east-1", environment: str = "prod"):
        """Initialize DynamoDB client and table name.

        Args:
            region: AWS region
            environment: 'demo' or 'prod'
        """
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table_name = f"dora_ai_insights_{environment}"
        self.table = self.dynamodb.Table(self.table_name)
        self.environment = environment
        self.region = region

    def save_event_insight(
        self,
        event_ticker: str,
        proposal_id: str,
        insights: str,
        recommendation: str,
        rationale: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_days: int = 30
    ) -> bool:
        """Save event-level insight to DynamoDB.

        Args:
            event_ticker: Event ticker (e.g., "KXBTC-24DEC31")
            proposal_id: UUID of the proposal batch this insight is for
            insights: Overall profitability insights (2-3 sentences)
            recommendation: "Expand", "Scale back", or "Fully Exit"
            rationale: Rationale for recommendation (2-3 sentences)
            metadata: Additional context (total_pnl, market_count, etc.)
            ttl_days: Days until insight auto-expires (default: 30)

        Returns:
            True if successful, False otherwise
        """
        try:
            created_at = datetime.now(timezone.utc).isoformat()
            ttl_timestamp = int(time.time()) + (ttl_days * 86400)

            item = {
                'insight_type': 'event',
                'entity_id': event_ticker,
                'proposal_id': proposal_id,
                'created_at': created_at,
                'insights': insights,
                'recommendation': recommendation,
                'rationale': rationale,
                'metadata': metadata or {},
                'environment': self.environment,
                'ttl_timestamp': ttl_timestamp,
            }

            # Convert floats to Decimal for DynamoDB compatibility
            item = _serialize_for_dynamodb(item)

            self.table.put_item(Item=item)
            logger.info(f"Saved event insight for {event_ticker}")
            return True

        except Exception as e:
            logger.error(f"Error saving event insight for {event_ticker}: {e}")
            return False

    def save_market_insight(
        self,
        market_id: str,
        proposal_id: str,
        recommendation: str,
        rationale: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_days: int = 30
    ) -> bool:
        """Save market-level insight to DynamoDB.

        Args:
            market_id: Market ticker
            proposal_id: UUID of the proposal batch this insight is for
            recommendation: "strong recommendation to enter", "enter with caution", or "do not enter"
            rationale: Rationale for recommendation (2-3 sentences)
            metadata: Additional context (volume, spread, etc.)
            ttl_days: Days until insight auto-expires (default: 30)

        Returns:
            True if successful, False otherwise
        """
        try:
            created_at = datetime.now(timezone.utc).isoformat()
            ttl_timestamp = int(time.time()) + (ttl_days * 86400)

            item = {
                'insight_type': 'market',
                'entity_id': market_id,
                'proposal_id': proposal_id,
                'created_at': created_at,
                'recommendation': recommendation,
                'rationale': rationale,
                'metadata': metadata or {},
                'environment': self.environment,
                'ttl_timestamp': ttl_timestamp,
            }

            # Convert floats to Decimal for DynamoDB compatibility
            item = _serialize_for_dynamodb(item)

            self.table.put_item(Item=item)
            logger.info(f"Saved market insight for {market_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving market insight for {market_id}: {e}")
            return False

    def get_insight_by_proposal(
        self,
        proposal_id: str,
        insight_type: Optional[str] = None
    ) -> list:
        """Retrieve insights for a given proposal_id.

        Args:
            proposal_id: UUID of the proposal batch
            insight_type: Optional filter ('event' or 'market')

        Returns:
            List of insight items from DynamoDB
        """
        try:
            # Query by proposal_id using GSI
            if insight_type:
                response = self.table.query(
                    IndexName='proposal_id-index',
                    KeyConditionExpression='proposal_id = :pid',
                    FilterExpression='insight_type = :itype',
                    ExpressionAttributeValues={
                        ':pid': proposal_id,
                        ':itype': insight_type
                    }
                )
            else:
                response = self.table.query(
                    IndexName='proposal_id-index',
                    KeyConditionExpression='proposal_id = :pid',
                    ExpressionAttributeValues={':pid': proposal_id}
                )

            insights = response.get('Items', [])
            logger.info(f"Retrieved {len(insights)} insights for proposal {proposal_id}")
            return insights

        except Exception as e:
            logger.error(f"Error retrieving insights for proposal {proposal_id}: {e}")
            return []

    def get_latest_event_insight(self, event_ticker: str) -> Optional[Dict[str, Any]]:
        """Get the most recent insight for an event.

        Args:
            event_ticker: Event ticker

        Returns:
            Most recent insight dict or None
        """
        try:
            response = self.table.query(
                KeyConditionExpression='insight_type = :type AND entity_id = :id',
                ExpressionAttributeValues={
                    ':type': 'event',
                    ':id': event_ticker
                },
                ScanIndexForward=False,  # Sort descending by created_at
                Limit=1
            )

            items = response.get('Items', [])
            return items[0] if items else None

        except Exception as e:
            logger.error(f"Error retrieving latest event insight for {event_ticker}: {e}")
            return None

    def get_latest_market_insight(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent insight for a market.

        Args:
            market_id: Market ticker

        Returns:
            Most recent insight dict or None
        """
        try:
            response = self.table.query(
                KeyConditionExpression='insight_type = :type AND entity_id = :id',
                ExpressionAttributeValues={
                    ':type': 'market',
                    ':id': market_id
                },
                ScanIndexForward=False,  # Sort descending by created_at
                Limit=1
            )

            items = response.get('Items', [])
            return items[0] if items else None

        except Exception as e:
            logger.error(f"Error retrieving latest market insight for {market_id}: {e}")
            return None
