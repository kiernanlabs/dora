"""
DynamoDB Client for Dora Manager Lambda
Supports both read and write operations for market management.
"""
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DynamoDBClient:
    """
    DynamoDB client for Dora Manager Lambda.
    Provides both read and write operations.
    """

    def __init__(self, region: str = "us-east-1", environment: str = "prod"):
        """
        Initialize the DynamoDB client.

        Args:
            region: AWS region
            environment: 'demo' or 'prod'
        """
        if environment not in ['demo', 'prod']:
            raise ValueError(f"Invalid environment: {environment}. Must be 'demo' or 'prod'")

        self.environment = environment
        self.region = region
        self.suffix = f"_{environment}"

        # Initialize boto3 DynamoDB resource (uses IAM role in Lambda)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)

        # Initialize table references
        self.market_config_table = self.dynamodb.Table(f"dora_market_config{self.suffix}")
        self.state_table = self.dynamodb.Table(f"dora_state{self.suffix}")
        self.trade_log_table = self.dynamodb.Table(f"dora_trade_log{self.suffix}")
        self.decision_log_table = self.dynamodb.Table(f"dora_decision_log{self.suffix}")

    @staticmethod
    def _deserialize_decimal(obj: Any) -> Any:
        """Convert DynamoDB Decimal types to float for JSON serialization."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: DynamoDBClient._deserialize_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DynamoDBClient._deserialize_decimal(item) for item in obj]
        return obj

    # ==================== Market Config Methods ====================

    def get_all_market_configs(self, enabled_only: bool = False) -> List[Dict]:
        """Get all market configurations."""
        try:
            response = self.market_config_table.scan()
            items = response.get('Items', [])

            if enabled_only:
                items = [item for item in items if item.get('enabled', False)]

            return [self._deserialize_decimal(item) for item in items]
        except Exception as e:
            logger.error(f"Error fetching market configs: {e}")
            return []

    def get_market_config(self, market_id: str) -> Optional[Dict]:
        """Get configuration for a specific market."""
        try:
            response = self.market_config_table.get_item(Key={'market_id': market_id})
            item = response.get('Item')
            return self._deserialize_decimal(item) if item else None
        except Exception as e:
            logger.error(f"Error fetching market config for {market_id}: {e}")
            return None

    def disable_market(self, market_id: str, reason: str) -> bool:
        """
        Disable a market by setting enabled=False.

        Args:
            market_id: The market ID to disable
            reason: Reason for disabling (stored in disabled_reason field)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.market_config_table.update_item(
                Key={'market_id': market_id},
                UpdateExpression="SET enabled = :enabled, disabled_reason = :reason, disabled_at = :ts",
                ExpressionAttributeValues={
                    ':enabled': False,
                    ':reason': reason,
                    ':ts': datetime.now(timezone.utc).isoformat()
                }
            )
            logger.info(f"Disabled market {market_id}: {reason}")
            return True
        except Exception as e:
            logger.error(f"Error disabling market {market_id}: {e}")
            return False

    # ==================== State Methods ====================

    def get_positions(self) -> Dict[str, Dict]:
        """Get all current positions."""
        try:
            response = self.state_table.get_item(Key={'key': 'positions'})
            item = response.get('Item')
            if item and 'positions' in item:
                return self._deserialize_decimal(item['positions'])
            return {}
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}

    def get_open_orders(self) -> Dict[str, Any]:
        """Get all current open orders from state table."""
        try:
            response = self.state_table.get_item(Key={'key': 'open_orders'})
            item = response.get('Item')
            if item:
                return self._deserialize_decimal(item)
            return {}
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return {}

    def get_open_orders_by_market(self) -> Dict[str, List[Dict]]:
        """Get open orders grouped by market_id."""
        try:
            open_orders_data = self.get_open_orders()
            orders = open_orders_data.get('orders', {})

            by_market: Dict[str, List[Dict]] = {}
            for order_id, order in orders.items():
                market_id = order.get('market_id')
                if market_id:
                    if market_id not in by_market:
                        by_market[market_id] = []
                    order_with_id = {'order_id': order_id, **order}
                    by_market[market_id].append(order_with_id)

            return by_market
        except Exception as e:
            logger.error(f"Error grouping open orders by market: {e}")
            return {}

    # ==================== Trade Log Methods ====================

    def get_trades_in_window(self, hours: int = 3) -> List[Dict]:
        """
        Get trades from the last N hours.

        Args:
            hours: Number of hours to look back
        """
        try:
            trades = []
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=hours)

            # Query each date that could have trades in the window
            dates_to_query = set()
            current = cutoff
            while current <= now:
                dates_to_query.add(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)

            for date_str in dates_to_query:
                response = self.trade_log_table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ScanIndexForward=False
                )
                trades.extend(response.get('Items', []))

            # Filter to only trades within the time window
            filtered_trades = []
            for trade in trades:
                ts_str = trade.get('fill_timestamp') or trade.get('timestamp')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        if ts >= cutoff:
                            filtered_trades.append(trade)
                    except Exception:
                        pass

            # Sort by timestamp (most recent first)
            filtered_trades.sort(
                key=lambda x: x.get('fill_timestamp') or x.get('timestamp', ''),
                reverse=True
            )

            return [self._deserialize_decimal(trade) for trade in filtered_trades]
        except Exception as e:
            logger.error(f"Error fetching trades in window: {e}")
            return []

    def get_all_trades(self, days: int = 30) -> List[Dict]:
        """
        Get all trades from the last N days.

        Args:
            days: Number of days to look back
        """
        try:
            trades = []
            start_date = datetime.utcnow() - timedelta(days=days)

            for i in range(days + 1):
                date_str = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                response = self.trade_log_table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ScanIndexForward=False
                )
                trades.extend(response.get('Items', []))

            # Sort by timestamp (oldest first for P&L calculation)
            trades.sort(key=lambda x: x.get('fill_timestamp') or x.get('timestamp', ''))

            return [self._deserialize_decimal(trade) for trade in trades]
        except Exception as e:
            logger.error(f"Error fetching all trades: {e}")
            return []

    # ==================== Decision Log Methods ====================

    def get_recent_decisions(self, hours: int = 3) -> List[Dict]:
        """Get decision logs from the last N hours."""
        try:
            decisions = []
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=hours)

            dates_to_query = set()
            current = cutoff
            while current <= now:
                dates_to_query.add(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)

            for date_str in dates_to_query:
                response = self.decision_log_table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ScanIndexForward=False
                )
                decisions.extend(response.get('Items', []))

            # Sort by timestamp (most recent first)
            decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return [self._deserialize_decimal(d) for d in decisions]
        except Exception as e:
            logger.error(f"Error fetching recent decisions: {e}")
            return []

    def get_most_recent_decision_by_market(self) -> Dict[str, Dict]:
        """Get the most recent decision for each market."""
        decisions = self.get_recent_decisions(hours=24)
        by_market: Dict[str, Dict] = {}
        for d in decisions:
            market_id = d.get('market_id')
            if market_id and market_id not in by_market:
                by_market[market_id] = d
        return by_market
