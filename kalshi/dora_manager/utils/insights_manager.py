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

            # Create composite key for historical tracking
            entity_type_id = f"event#{event_ticker}"

            item = {
                'entity_type_id': entity_type_id,  # Primary key (HASH)
                'proposal_id': proposal_id,         # Primary key (RANGE)
                'insight_type': 'event',            # Separate attribute for filtering
                'entity_id': event_ticker,          # Separate attribute for access
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
            logger.info(f"Saved event insight for {event_ticker} (proposal: {proposal_id})")
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

            # Create composite key for historical tracking
            entity_type_id = f"market#{market_id}"

            item = {
                'entity_type_id': entity_type_id,  # Primary key (HASH)
                'proposal_id': proposal_id,         # Primary key (RANGE)
                'insight_type': 'market',           # Separate attribute for filtering
                'entity_id': market_id,             # Separate attribute for access
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
            logger.info(f"Saved market insight for {market_id} (proposal: {proposal_id})")
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
        """Get the most recent insight for an event across all proposals.

        Args:
            event_ticker: Event ticker

        Returns:
            Most recent insight dict or None
        """
        try:
            entity_type_id = f"event#{event_ticker}"

            response = self.table.query(
                KeyConditionExpression='entity_type_id = :entity_type_id',
                ExpressionAttributeValues={
                    ':entity_type_id': entity_type_id
                }
            )

            items = response.get('Items', [])
            if not items:
                return None

            # Sort by created_at descending (most recent first)
            sorted_items = sorted(items, key=lambda x: x.get('created_at', ''), reverse=True)
            return sorted_items[0]

        except Exception as e:
            logger.error(f"Error retrieving latest event insight for {event_ticker}: {e}")
            return None

    def get_latest_market_insight(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent insight for a market across all proposals.

        Args:
            market_id: Market ticker

        Returns:
            Most recent insight dict or None
        """
        try:
            entity_type_id = f"market#{market_id}"

            response = self.table.query(
                KeyConditionExpression='entity_type_id = :entity_type_id',
                ExpressionAttributeValues={
                    ':entity_type_id': entity_type_id
                }
            )

            items = response.get('Items', [])
            if not items:
                return None

            # Sort by created_at descending (most recent first)
            sorted_items = sorted(items, key=lambda x: x.get('created_at', ''), reverse=True)
            return sorted_items[0]

        except Exception as e:
            logger.error(f"Error retrieving latest market insight for {market_id}: {e}")
            return None

    def get_all_event_insights(self, event_ticker: str, limit: Optional[int] = None) -> list:
        """Get all historical insights for an event across all proposals.

        Args:
            event_ticker: Event ticker
            limit: Optional limit on number of insights to return (most recent first)

        Returns:
            List of insight dicts sorted by created_at descending (newest first)
        """
        try:
            entity_type_id = f"event#{event_ticker}"

            response = self.table.query(
                KeyConditionExpression='entity_type_id = :entity_type_id',
                ExpressionAttributeValues={
                    ':entity_type_id': entity_type_id
                }
            )

            items = response.get('Items', [])

            # Sort by created_at descending (most recent first)
            sorted_items = sorted(items, key=lambda x: x.get('created_at', ''), reverse=True)

            # Apply limit if specified
            if limit:
                sorted_items = sorted_items[:limit]

            logger.info(f"Retrieved {len(sorted_items)} historical insights for event {event_ticker}")
            return sorted_items

        except Exception as e:
            logger.error(f"Error retrieving all event insights for {event_ticker}: {e}")
            return []

    def get_all_market_insights(self, market_id: str, limit: Optional[int] = None) -> list:
        """Get all historical insights for a market across all proposals.

        Args:
            market_id: Market ticker
            limit: Optional limit on number of insights to return (most recent first)

        Returns:
            List of insight dicts sorted by created_at descending (newest first)
        """
        try:
            entity_type_id = f"market#{market_id}"

            response = self.table.query(
                KeyConditionExpression='entity_type_id = :entity_type_id',
                ExpressionAttributeValues={
                    ':entity_type_id': entity_type_id
                }
            )

            items = response.get('Items', [])

            # Sort by created_at descending (most recent first)
            sorted_items = sorted(items, key=lambda x: x.get('created_at', ''), reverse=True)

            # Apply limit if specified
            if limit:
                sorted_items = sorted_items[:limit]

            logger.info(f"Retrieved {len(sorted_items)} historical insights for market {market_id}")
            return sorted_items

        except Exception as e:
            logger.error(f"Error retrieving all market insights for {market_id}: {e}")
            return []
