"""
Read-only DynamoDB Client for Streamlit UI
"""
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import logging

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReadOnlyDynamoDBClient:
    """
    Read-only client for accessing Dora Bot DynamoDB tables.
    Provides query methods for Streamlit UI visualization.
    """

    def __init__(self, region: str = "us-east-1", environment: str = "demo"):
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

        # Initialize boto3 DynamoDB resource
        # Check for Streamlit secrets first (for Streamlit Cloud deployment)
        if STREAMLIT_AVAILABLE and hasattr(st, 'secrets') and 'AWS_ACCESS_KEY_ID' in st.secrets:
            logger.info("Using AWS credentials from Streamlit secrets")
            self.dynamodb = boto3.resource(
                'dynamodb',
                region_name=region,
                aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY']
            )
        else:
            # Fall back to default credential chain (AWS CLI, environment variables, IAM role)
            logger.info("Using AWS credentials from default credential chain")
            self.dynamodb = boto3.resource('dynamodb', region_name=region)

        # Initialize table references
        self.market_config_table = self.dynamodb.Table(f"dora_market_config{self.suffix}")
        self.state_table = self.dynamodb.Table(f"dora_state{self.suffix}")
        self.trade_log_table = self.dynamodb.Table(f"dora_trade_log{self.suffix}")
        self.decision_log_table = self.dynamodb.Table(f"dora_decision_log{self.suffix}")
        self.execution_log_table = self.dynamodb.Table(f"dora_execution_log{self.suffix}")

    @staticmethod
    def _deserialize_decimal(obj: Any) -> Any:
        """Convert DynamoDB Decimal types to float for JSON serialization."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: ReadOnlyDynamoDBClient._deserialize_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ReadOnlyDynamoDBClient._deserialize_decimal(item) for item in obj]
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

    def get_global_config(self) -> Optional[Dict]:
        """Get global bot configuration."""
        try:
            response = self.state_table.get_item(Key={'key': 'global_config'})
            item = response.get('Item')
            return self._deserialize_decimal(item) if item else None
        except Exception as e:
            logger.error(f"Error fetching global config: {e}")
            return None

    def get_risk_state(self) -> Optional[Dict]:
        """Get current risk state."""
        try:
            response = self.state_table.get_item(Key={'key': 'risk_state'})
            item = response.get('Item')
            return self._deserialize_decimal(item) if item else None
        except Exception as e:
            logger.error(f"Error fetching risk state: {e}")
            return None

    def get_open_orders(self) -> Dict[str, Dict]:
        """Get all current open orders from state table.

        Returns:
            Dictionary mapping order_id to order details, or empty dict if none.
            Also includes 'last_updated' timestamp in the response.
        """
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
        """Get open orders grouped by market_id.

        Returns:
            Dictionary mapping market_id to list of orders for that market.
        """
        try:
            open_orders_data = self.get_open_orders()
            orders = open_orders_data.get('orders', {})

            # Group by market_id
            by_market: Dict[str, List[Dict]] = {}
            for order_id, order in orders.items():
                market_id = order.get('market_id')
                if market_id:
                    if market_id not in by_market:
                        by_market[market_id] = []
                    # Include order_id in the order dict for reference
                    order_with_id = {'order_id': order_id, **order}
                    by_market[market_id].append(order_with_id)

            return by_market
        except Exception as e:
            logger.error(f"Error grouping open orders by market: {e}")
            return {}

    # ==================== Trade Log Methods ====================

    def get_recent_trades(self, days: int = 7, market_id: Optional[str] = None) -> List[Dict]:
        """
        Get recent trades from the trade log.

        Args:
            days: Number of days to look back
            market_id: Optional filter for specific market
        """
        try:
            trades = []
            start_date = datetime.utcnow() - timedelta(days=days)

            # Query each date
            for i in range(days + 1):
                date_str = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                response = self.trade_log_table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ScanIndexForward=False  # Most recent first
                )
                trades.extend(response.get('Items', []))

            # Filter by market if specified
            if market_id:
                trades = [t for t in trades if t.get('market_id') == market_id]

            # Sort by timestamp (most recent first)
            trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return [self._deserialize_decimal(trade) for trade in trades]
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return []

    # ==================== Decision Log Methods ====================

    def get_recent_decision_logs(self, days: int = 7, market_id: Optional[str] = None) -> List[Dict]:
        """
        Get recent decision logs.

        Args:
            days: Number of days to look back
            market_id: Optional filter for specific market
        """
        try:
            decisions = []
            start_date = datetime.utcnow() - timedelta(days=days)

            # Query each date
            for i in range(days + 1):
                date_str = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                response = self.decision_log_table.query(
                    KeyConditionExpression=Key('date').eq(date_str),
                    ScanIndexForward=False  # Most recent first
                )
                decisions.extend(response.get('Items', []))

            # Filter by market if specified
            if market_id:
                decisions = [d for d in decisions if d.get('market_id') == market_id]

            # Sort by timestamp (most recent first)
            decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return [self._deserialize_decimal(decision) for decision in decisions]
        except Exception as e:
            logger.error(f"Error fetching decision logs: {e}")
            return []

    def get_most_recent_decision_log(self, market_id: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent decision log entry."""
        decisions = self.get_recent_decision_logs(days=1, market_id=market_id)
        return decisions[0] if decisions else None

    # ==================== Execution Log Methods ====================

    def get_recent_execution_logs(
        self,
        days: int = 7,
        market_id: Optional[str] = None,
        decision_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent execution logs.

        Args:
            days: Number of days to look back
            market_id: Optional filter for specific market
            decision_id: Optional filter for specific decision
        """
        try:
            # If decision_id is provided, use the GSI
            if decision_id:
                response = self.execution_log_table.query(
                    IndexName='decision_id-event_ts-index',
                    KeyConditionExpression=Key('decision_id').eq(decision_id),
                    ScanIndexForward=False  # Most recent first
                )
                executions = response.get('Items', [])
            else:
                # Otherwise, scan with filters (less efficient but comprehensive)
                executions = []
                scan_kwargs: Dict[str, Any] = {}
                while True:
                    response = self.execution_log_table.scan(**scan_kwargs)
                    executions.extend(response.get('Items', []))
                    last_evaluated_key = response.get('LastEvaluatedKey')
                    if not last_evaluated_key:
                        break
                    scan_kwargs['ExclusiveStartKey'] = last_evaluated_key

                # Filter by timestamp
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                executions = [
                    e for e in executions
                    if e.get('event_ts') and datetime.fromisoformat(e.get('event_ts', '').replace('Z', '+00:00')) > cutoff
                ]

            # Filter by market if specified
            if market_id:
                executions = [e for e in executions if e.get('market') == market_id]

            # Sort by timestamp (most recent first)
            executions.sort(key=lambda x: x.get('event_ts', ''), reverse=True)

            return [self._deserialize_decimal(execution) for execution in executions]
        except Exception as e:
            logger.error(f"Error fetching execution logs: {e}")
            return []

    def get_most_recent_execution_log(self, market_id: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent execution log entry."""
        executions = self.get_recent_execution_logs(days=1, market_id=market_id)
        return executions[0] if executions else None

    # ==================== Analytics Methods ====================

    def calculate_total_pnl(self, positions: Dict[str, Dict]) -> float:
        """Calculate total realized P&L from positions."""
        total_pnl = 0.0
        for position in positions.values():
            total_pnl += position.get('realized_pnl', 0.0)
        return total_pnl

    def calculate_total_exposure(self, positions: Dict[str, Dict]) -> int:
        """Calculate total exposure (sum of absolute positions)."""
        total_exposure = 0
        for position in positions.values():
            total_exposure += abs(position.get('net_yes_qty', 0))
        return total_exposure

    def get_active_quotes_for_market(self, market_id: str) -> Optional[Dict]:
        """
        Get the most recent target quotes for a market.
        Returns dict with 'bid' and 'ask' arrays containing price and size.
        """
        decision = self.get_most_recent_decision_log(market_id=market_id)
        if not decision:
            return None

        target_quotes = decision.get('target_quotes', [])

        # Separate bids and asks
        bids = [q for q in target_quotes if q.get('side') == 'bid']
        asks = [q for q in target_quotes if q.get('side') == 'ask']

        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x.get('price', 0), reverse=True)
        asks.sort(key=lambda x: x.get('price', 0))

        return {
            'bids': bids,
            'asks': asks
        }

    def get_24h_filled_contracts(self, market_id: str) -> int:
        """Get total number of contracts filled in the last 24 hours for a market."""
        try:
            trades = self.get_recent_trades(days=1, market_id=market_id)

            # Filter to last 24 hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_trades = [
                t for t in trades
                if (t.get('fill_timestamp') or t.get('timestamp'))
                and datetime.fromisoformat((t.get('fill_timestamp') or t.get('timestamp', '')).replace('Z', '+00:00')) > cutoff
            ]

            # Sum up all trade sizes
            total_contracts = sum(t.get('size', 0) for t in recent_trades)
            return total_contracts
        except Exception as e:
            logger.error(f"Error calculating 24h filled contracts for {market_id}: {e}")
            return 0

    def get_most_recent_fill_timestamp(self, market_id: str) -> Optional[str]:
        """Get the timestamp of the most recent fill for a market."""
        try:
            trades = self.get_recent_trades(days=1, market_id=market_id)
            if not trades:
                return None

            # Get the first trade (most recent)
            most_recent = trades[0]
            return most_recent.get('fill_timestamp') or most_recent.get('timestamp')
        except Exception as e:
            logger.error(f"Error getting most recent fill timestamp for {market_id}: {e}")
            return None

    def get_24h_execution_count(self, market_id: str) -> int:
        """Get the count of execution events in the last 24 hours for a market."""
        try:
            executions = self.get_recent_execution_logs(days=1, market_id=market_id)

            # Filter to last 24 hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_executions = [
                e for e in executions
                if e.get('event_ts')
                and datetime.fromisoformat(e.get('event_ts', '').replace('Z', '+00:00')) > cutoff
            ]

            return len(recent_executions)
        except Exception as e:
            logger.error(f"Error calculating 24h execution count for {market_id}: {e}")
            return 0

    def get_most_recent_execution_timestamp(self, market_id: str) -> Optional[str]:
        """Get the timestamp of the most recent execution for a market."""
        try:
            executions = self.get_recent_execution_logs(days=1, market_id=market_id)
            if not executions:
                return None

            # Get the first execution (most recent)
            return executions[0].get('event_ts')
        except Exception as e:
            logger.error(f"Error getting most recent execution timestamp for {market_id}: {e}")
            return None

    def get_pnl_over_time(self, days: int = 7) -> List[Dict]:
        """
        Calculate P&L over time from trade history using proper cost basis tracking.

        Returns list of {date, cumulative_pnl, daily_pnl}
        """
        try:
            trades = self.get_recent_trades(days=days)

            # Sort trades chronologically
            trades.sort(key=lambda t: t.get('fill_timestamp') or t.get('timestamp', ''))

            # Track position state per market (same logic as Position.update_from_fill)
            positions = {}  # market_id -> {net_yes_qty, avg_buy_price, avg_sell_price, realized_pnl}
            pnl_by_date = {}

            for trade in trades:
                market_id = trade.get('market_id')
                if not market_id:
                    continue

                # Initialize position for this market if needed
                if market_id not in positions:
                    positions[market_id] = {
                        'net_yes_qty': 0,
                        'avg_buy_price': 0.0,
                        'avg_sell_price': 0.0,
                        'realized_pnl': 0.0
                    }

                pos = positions[market_id]
                side = trade.get('side', '')
                price = trade.get('price', 0.0)
                size = trade.get('size', 0)
                fees = trade.get('fees', 0.0) or 0.0
                date_str = trade.get('date', '')

                # Initialize date bucket
                if date_str not in pnl_by_date:
                    pnl_by_date[date_str] = 0.0

                # Track P&L before update
                pnl_before = pos['realized_pnl']

                # Subtract fees from realized P&L on every fill (same as Position.update_from_fill)
                pos['realized_pnl'] -= fees

                # Update position using same logic as Position.update_from_fill()
                if side in ['buy', 'yes']:
                    # Bid fill - buying YES contracts
                    if pos['net_yes_qty'] >= 0:
                        # Adding to long position
                        total_cost = pos['avg_buy_price'] * pos['net_yes_qty'] + price * size
                        pos['net_yes_qty'] += size
                        pos['avg_buy_price'] = total_cost / pos['net_yes_qty'] if pos['net_yes_qty'] > 0 else 0
                    else:
                        # Closing short position - realize P&L
                        close_qty = min(abs(pos['net_yes_qty']), size)
                        realized = (pos['avg_sell_price'] - price) * close_qty
                        pos['realized_pnl'] += realized
                        pos['net_yes_qty'] += size

                        if pos['net_yes_qty'] > 0:
                            pos['avg_buy_price'] = price
                else:
                    # Ask fill - selling YES contracts
                    if pos['net_yes_qty'] <= 0:
                        # Adding to short position
                        total_cost = pos['avg_sell_price'] * abs(pos['net_yes_qty']) + price * size
                        pos['net_yes_qty'] -= size
                        pos['avg_sell_price'] = total_cost / abs(pos['net_yes_qty']) if pos['net_yes_qty'] != 0 else 0
                    else:
                        # Closing long position - realize P&L
                        close_qty = min(pos['net_yes_qty'], size)
                        realized = (price - pos['avg_buy_price']) * close_qty
                        pos['realized_pnl'] += realized
                        pos['net_yes_qty'] -= size

                        if pos['net_yes_qty'] < 0:
                            pos['avg_sell_price'] = price

                # Add the realized P&L change to this date
                pnl_change = pos['realized_pnl'] - pnl_before
                pnl_by_date[date_str] += pnl_change

            # Sort and create time series
            sorted_dates = sorted(pnl_by_date.keys())
            cumulative_pnl = 0.0
            result = []

            for date_str in sorted_dates:
                daily_pnl = pnl_by_date[date_str]
                cumulative_pnl += daily_pnl
                result.append({
                    'date': date_str,
                    'daily_pnl': daily_pnl,
                    'cumulative_pnl': cumulative_pnl
                })

            return result
        except Exception as e:
            logger.error(f"Error calculating P&L over time: {e}")
            return []
