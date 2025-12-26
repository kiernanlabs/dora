"""DynamoDB helper functions for persistence."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError

from dora_bot.models import MarketConfig, GlobalConfig, RiskState, Position, Order
from dora_bot.structured_logger import get_logger, EventType

logger = get_logger(__name__)


class DynamoDBClient:
    """Client for interacting with DynamoDB tables."""

    # Table name suffixes by environment
    TABLE_SUFFIXES = {
        'demo': '_demo',
        'prod': '_prod'
    }

    def __init__(
        self,
        region: str = "us-east-1",
        environment: str = "demo"
    ):
        """Initialize DynamoDB client.

        Args:
            region: AWS region
            environment: Environment name ('demo' or 'prod') - determines table suffix
        """
        if environment not in self.TABLE_SUFFIXES:
            raise ValueError(f"Invalid environment: {environment}. Must be one of: {list(self.TABLE_SUFFIXES.keys())}")

        self.environment = environment
        self.region = region
        suffix = self.TABLE_SUFFIXES[environment]

        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.market_config_table = self.dynamodb.Table(f"dora_market_config{suffix}")
        self.state_table = self.dynamodb.Table(f"dora_state{suffix}")
        self.trade_log_table = self.dynamodb.Table(f"dora_trade_log{suffix}")
        self.decision_log_table = self.dynamodb.Table(f"dora_decision_log{suffix}")

        logger.info("DynamoDB initialized", extra={
            "event_type": EventType.STARTUP,
            "environment": environment,
            "tables": [
                f"dora_market_config{suffix}",
                f"dora_state{suffix}",
                f"dora_trade_log{suffix}",
                f"dora_decision_log{suffix}",
            ],
        })

    @staticmethod
    def _serialize_decimal(obj: Any) -> Any:
        """Convert Decimal to float for JSON serialization."""
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, dict):
            return {k: DynamoDBClient._serialize_decimal(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DynamoDBClient._serialize_decimal(item) for item in obj]
        return obj

    @staticmethod
    def _to_dynamo_item(obj: Any) -> Any:
        """Convert Python types to DynamoDB compatible types."""
        if isinstance(obj, float):
            return Decimal(str(obj))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: DynamoDBClient._to_dynamo_item(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DynamoDBClient._to_dynamo_item(item) for item in obj]
        return obj

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> datetime:
        """Parse an ISO timestamp and normalize to UTC if missing tzinfo."""
        if value:
            parsed = datetime.fromisoformat(value)
        else:
            parsed = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    # Market Config operations

    def get_market_config(self, market_id: str) -> Optional[MarketConfig]:
        """Fetch market configuration.

        Args:
            market_id: Market ticker

        Returns:
            MarketConfig object or None if not found
        """
        try:
            response = self.market_config_table.get_item(Key={'market_id': market_id})
            if 'Item' not in response:
                return None

            item = self._serialize_decimal(response['Item'])
            return MarketConfig(
                market_id=item['market_id'],
                enabled=item.get('enabled', False),
                max_inventory_yes=item.get('max_inventory_yes', 100),
                max_inventory_no=item.get('max_inventory_no', 100),
                min_spread=item.get('min_spread', 0.06),
                quote_size=item.get('quote_size', 10),
                inventory_skew_factor=item.get('inventory_skew_factor', 0.5),
                fair_value=item.get('fair_value'),
                toxicity_score=item.get('toxicity_score'),
                updated_at=self._parse_datetime(item.get('updated_at'))
            )
        except ClientError as e:
            logger.error("Error fetching market config", extra={
                "event_type": EventType.ERROR,
                "market": market_id,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return None

    def get_all_market_configs(self, enabled_only: bool = True) -> Dict[str, MarketConfig]:
        """Fetch all market configurations.

        Args:
            enabled_only: If True, only return enabled markets

        Returns:
            Dictionary mapping market_id to MarketConfig
        """
        try:
            response = self.market_config_table.scan()
            items = response.get('Items', [])

            configs = {}
            for item in items:
                item = self._serialize_decimal(item)
                config = MarketConfig(
                    market_id=item['market_id'],
                    enabled=item.get('enabled', False),
                    max_inventory_yes=item.get('max_inventory_yes', 100),
                    max_inventory_no=item.get('max_inventory_no', 100),
                    min_spread=item.get('min_spread', 0.06),
                    quote_size=item.get('quote_size', 10),
                    inventory_skew_factor=item.get('inventory_skew_factor', 0.5),
                    fair_value=item.get('fair_value'),
                    toxicity_score=item.get('toxicity_score'),
                    updated_at=self._parse_datetime(item.get('updated_at'))
                )

                if not enabled_only or config.enabled:
                    configs[config.market_id] = config

            return configs
        except ClientError as e:
            logger.error("Error fetching market configs", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return {}

    def put_market_config(self, config: MarketConfig) -> bool:
        """Save market configuration.

        Args:
            config: MarketConfig to save

        Returns:
            True if successful
        """
        try:
            item = {
                'market_id': config.market_id,
                'enabled': config.enabled,
                'max_inventory_yes': int(config.max_inventory_yes),
                'max_inventory_no': int(config.max_inventory_no),
                'min_spread': self._to_dynamo_item(config.min_spread),
                'quote_size': int(config.quote_size),
                'inventory_skew_factor': self._to_dynamo_item(config.inventory_skew_factor),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            if config.fair_value is not None:
                item['fair_value'] = self._to_dynamo_item(config.fair_value)

            if config.toxicity_score is not None:
                item['toxicity_score'] = self._to_dynamo_item(config.toxicity_score)

            self.market_config_table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error("Error saving market config", extra={
                "event_type": EventType.ERROR,
                "market": config.market_id,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    # State operations

    def get_positions(self) -> Dict[str, Position]:
        """Fetch all positions from state table.

        Returns:
            Dictionary mapping market_id to Position
        """
        try:
            response = self.state_table.get_item(Key={'key': 'positions'})
            if 'Item' not in response:
                return {}

            positions_data = self._serialize_decimal(response['Item'].get('positions', {}))
            positions = {}

            for market_id, pos_data in positions_data.items():
                positions[market_id] = Position(
                    market_id=market_id,
                    net_yes_qty=pos_data.get('net_yes_qty', 0),
                    avg_buy_price=pos_data.get('avg_buy_price', 0.0),
                    avg_sell_price=pos_data.get('avg_sell_price', 0.0),
                    realized_pnl=pos_data.get('realized_pnl', 0.0)
                )

            return positions
        except ClientError as e:
            logger.error("Error fetching positions", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return {}

    def save_positions(self, positions: Dict[str, Position]) -> bool:
        """Save positions to state table.

        Args:
            positions: Dictionary of Position objects

        Returns:
            True if successful
        """
        try:
            positions_data = {}
            for market_id, pos in positions.items():
                positions_data[market_id] = self._to_dynamo_item({
                    'net_yes_qty': pos.net_yes_qty,
                    'avg_buy_price': pos.avg_buy_price,
                    'avg_sell_price': pos.avg_sell_price,
                    'realized_pnl': pos.realized_pnl
                })

            self.state_table.put_item(Item={
                'key': 'positions',
                'positions': positions_data
            })
            return True
        except ClientError as e:
            logger.error("Error saving positions", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    def get_open_orders(self) -> Dict[str, Order]:
        """Fetch open orders from state table.

        Returns:
            Dictionary mapping order_id to Order
        """
        try:
            response = self.state_table.get_item(Key={'key': 'open_orders'})
            if 'Item' not in response:
                return {}

            orders_data = self._serialize_decimal(response['Item'].get('orders', {}))
            orders = {}

            for order_id, order_data in orders_data.items():
                created_at = order_data.get('created_at')
                orders[order_id] = Order(
                    order_id=order_id,
                    market_id=order_data.get('market_id', ''),
                    side=order_data.get('side', 'yes'),
                    price=order_data.get('price', 0.0),
                    size=order_data.get('size', 0),
                    decision_id=order_data.get('decision_id'),
                    client_order_id=order_data.get('client_order_id'),
                    filled_size=order_data.get('filled_size', 0),
                    status=order_data.get('status', 'pending'),
                    created_at=self._parse_datetime(created_at),
                    tif=order_data.get('tif', 'gtc')
                )

            return orders
        except ClientError as e:
            logger.error("Error fetching open orders", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return {}

    def save_open_orders(self, orders: Dict[str, Order]) -> bool:
        """Save open orders to state table.

        Args:
            orders: Dictionary mapping order_id to Order

        Returns:
            True if successful
        """
        try:
            orders_data = {}
            for order_id, order in orders.items():
                orders_data[order_id] = self._to_dynamo_item({
                    'market_id': order.market_id,
                    'side': order.side,
                    'price': order.price,
                    'size': order.size,
                    'decision_id': order.decision_id,
                    'client_order_id': order.client_order_id,
                    'filled_size': order.filled_size,
                    'status': order.status,
                    'created_at': order.created_at.isoformat() if order.created_at else None,
                    'tif': order.tif
                })

            self.state_table.put_item(Item={
                'key': 'open_orders',
                'orders': orders_data,
                'last_updated': datetime.now(timezone.utc).isoformat()
            })
            return True
        except ClientError as e:
            logger.error("Error saving open orders", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    def get_global_config(self) -> GlobalConfig:
        """Fetch global configuration.

        Returns:
            GlobalConfig object with defaults if not found
        """
        try:
            response = self.state_table.get_item(Key={'key': 'global_config'})
            if 'Item' not in response:
                return GlobalConfig()

            item = self._serialize_decimal(response['Item'])
            return GlobalConfig(
                max_total_exposure=item.get('max_total_exposure', 500),
                max_daily_loss=item.get('max_daily_loss', 100.0),
                loop_interval_ms=item.get('loop_interval_ms', 5000),
                trading_enabled=item.get('trading_enabled', True),
                risk_aversion_k=item.get('risk_aversion_k', 0.5),
                cancel_on_startup=item.get('cancel_on_startup', True),
                use_batch_execution=item.get('use_batch_execution', False),
            )
        except ClientError as e:
            logger.error("Error fetching global config", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return GlobalConfig()

    def save_global_config(self, config: GlobalConfig) -> bool:
        """Save global configuration.

        Args:
            config: GlobalConfig to save

        Returns:
            True if successful
        """
        try:
            self.state_table.put_item(Item={
                'key': 'global_config',
                'max_total_exposure': config.max_total_exposure,
                'max_daily_loss': self._to_dynamo_item(config.max_daily_loss),
                'loop_interval_ms': config.loop_interval_ms,
                'trading_enabled': config.trading_enabled,
                'risk_aversion_k': self._to_dynamo_item(config.risk_aversion_k),
                'cancel_on_startup': config.cancel_on_startup
            })
            return True
        except ClientError as e:
            logger.error("Error saving global config", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    def get_risk_state(self) -> RiskState:
        """Fetch risk state.

        Returns:
            RiskState object with defaults if not found
        """
        try:
            response = self.state_table.get_item(Key={'key': 'risk_state'})
            if 'Item' not in response:
                return RiskState()

            item = self._serialize_decimal(response['Item'])
            last_fill_ts = item.get('last_fill_timestamp')

            return RiskState(
                daily_pnl=item.get('daily_pnl', 0.0),
                last_fill_timestamp=self._parse_datetime(last_fill_ts) if last_fill_ts else None,
                trading_halted=item.get('trading_halted', False),
                halt_reason=item.get('halt_reason'),
                last_updated=self._parse_datetime(item.get('last_updated'))
            )
        except ClientError as e:
            logger.error("Error fetching risk state", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return RiskState()

    def save_risk_state(self, state: RiskState) -> bool:
        """Save risk state.

        Args:
            state: RiskState to save

        Returns:
            True if successful
        """
        try:
            item = {
                'key': 'risk_state',
                'daily_pnl': self._to_dynamo_item(state.daily_pnl),
                'trading_halted': state.trading_halted,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

            if state.last_fill_timestamp:
                item['last_fill_timestamp'] = state.last_fill_timestamp.isoformat()
            if state.halt_reason:
                item['halt_reason'] = state.halt_reason

            self.state_table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error("Error saving risk state", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    # Trade logging

    def get_all_fill_ids(self) -> set:
        """Retrieve all fill_ids from the trade log table.

        Returns:
            Set of fill_ids that have been logged
        """
        try:
            fill_ids = set()

            # Scan the table to get all fill_ids
            # Use pagination to handle large result sets
            response = self.trade_log_table.scan(
                ProjectionExpression='fill_id'
            )

            for item in response.get('Items', []):
                fill_id = item.get('fill_id')
                if fill_id and fill_id != 'unknown':
                    fill_ids.add(fill_id)

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.trade_log_table.scan(
                    ProjectionExpression='fill_id',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    fill_id = item.get('fill_id')
                    if fill_id and fill_id != 'unknown':
                        fill_ids.add(fill_id)

            return fill_ids
        except ClientError as e:
            logger.error("Error fetching fill IDs", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return set()

    def fill_exists(self, fill_id: str) -> bool:
        """Check if a fill has already been logged to the trade log.

        Args:
            fill_id: The fill ID to check

        Returns:
            True if the fill exists in the trade log
        """
        try:
            # Query the trade log table for this fill_id across all dates
            # We'll scan with a filter expression since fill_id is not a key
            response = self.trade_log_table.scan(
                FilterExpression='fill_id = :fill_id',
                ExpressionAttributeValues={':fill_id': fill_id},
                Limit=1  # We only need to know if it exists
            )
            return len(response.get('Items', [])) > 0
        except ClientError as e:
            logger.error("Error checking if fill exists", extra={
                "event_type": EventType.ERROR,
                "fill_id": fill_id,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False  # Assume doesn't exist to avoid missing fills

    def log_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Log a trade to the trade log table.

        Args:
            trade_data: Dictionary containing trade information

        Returns:
            True if successful
        """
        try:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            timestamp = datetime.now(timezone.utc).isoformat()
            order_id = trade_data.get('order_id', 'unknown')
            fill_id = trade_data.get('fill_id', 'unknown')

            # Check if this fill has already been logged
            if self.fill_exists(fill_id):
                return True

            item = self._to_dynamo_item(trade_data)
            item['date'] = date
            item['timestamp#order_id'] = f"{timestamp}#{order_id}"
            item['fill_id'] = fill_id

            self.trade_log_table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error("Error logging trade", extra={
                "event_type": EventType.ERROR,
                "fill_id": fill_id,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    def log_decision(self, decision_data: Dict[str, Any]) -> bool:
        """Log a decision to the decision log table.

        Args:
            decision_data: Dictionary containing decision information

        Returns:
            True if successful
        """
        try:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            timestamp = datetime.now(timezone.utc).isoformat()

            item = self._to_dynamo_item(decision_data)
            item['date'] = date
            item['timestamp'] = timestamp

            self.decision_log_table.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error("Error logging decision", extra={
                "event_type": EventType.ERROR,
                "market": decision_data.get('market_id'),
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False

    # API Error tracking

    def get_api_errors(self) -> Dict[str, Dict[str, Any]]:
        """Fetch API error counts by market_id from state table.

        Returns:
            Dictionary mapping market_id to error stats:
            {
                "MARKET-ID": {
                    "count": 5,
                    "last_error": "2025-12-25T12:00:00",
                    "last_error_code": "not_found",
                    "last_status_code": 404
                }
            }
        """
        try:
            response = self.state_table.get_item(Key={'key': 'api_errors'})
            if 'Item' not in response:
                return {}

            errors_data = self._serialize_decimal(response['Item'].get('errors', {}))
            return errors_data
        except ClientError as e:
            logger.error("Error fetching API errors", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return {}

    def save_api_errors(self, errors: Dict[str, Dict[str, Any]]) -> bool:
        """Save API error counts to state table.

        Args:
            errors: Dictionary mapping market_id to error stats

        Returns:
            True if successful
        """
        try:
            self.state_table.put_item(Item={
                'key': 'api_errors',
                'errors': self._to_dynamo_item(errors),
                'last_updated': datetime.now(timezone.utc).isoformat()
            })
            return True
        except ClientError as e:
            logger.error("Error saving API errors", extra={
                "event_type": EventType.ERROR,
                "error_type": "ClientError",
                "error_msg": str(e),
            })
            return False
