"""Structured JSON logging for CloudWatch integration.

This module provides a JSON logging formatter and helper utilities for
emitting structured log events that can be queried in CloudWatch Insights.
"""

import json
import logging
import os
import secrets
import subprocess
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


class EventType(str, Enum):
    """Enumeration of structured log event types."""
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    HEARTBEAT = "HEARTBEAT"
    DECISION_MADE = "DECISION_MADE"
    ORDER_PLACE = "ORDER_PLACE"
    ORDER_PLACED = "ORDER_PLACED"  # Successful order placement
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_RESULT = "ORDER_RESULT"
    FILL = "FILL"
    RISK_HALT = "RISK_HALT"
    STATE_SAVE = "STATE_SAVE"
    STATE_LOAD = "STATE_LOAD"
    CONFIG_REFRESH = "CONFIG_REFRESH"
    ERROR = "ERROR"
    # Generic for logs that don't fit a specific event type
    LOG = "LOG"
    FLAG = "FLAG"  # for one-off debugging

    # Batch operations
    BATCH_CANCEL = "BATCH_CANCEL"  # Starting a cancel batch
    BATCH_CANCEL_FAILED = "BATCH_CANCEL_FAILED"  # Individual cancel failure
    BATCH_CANCEL_SUMMARY = "BATCH_CANCEL_SUMMARY"  # Summary after all cancel batches
    BATCH_PLACE = "BATCH_PLACE"  # Starting a place batch
    BATCH_PLACE_FAILED = "BATCH_PLACE_FAILED"  # Individual place failure
    BATCH_PLACE_SUMMARY = "BATCH_PLACE_SUMMARY"  # Summary after all place batches

    # Rate limiting
    RATE_LIMIT = "RATE_LIMIT"  # Hit rate limit (429 response)
    RATE_LIMIT_BACKOFF = "RATE_LIMIT_BACKOFF"  # Backing off due to rate limits


# Context variables for correlation IDs
_bot_run_id: ContextVar[str] = ContextVar('bot_run_id', default='')
_bot_version: ContextVar[str] = ContextVar('bot_version', default='')
_decision_id: ContextVar[str] = ContextVar('decision_id', default='')
_market: ContextVar[str] = ContextVar('market', default='')
_env: ContextVar[str] = ContextVar('env', default='')
_service: ContextVar[str] = ContextVar('service', default='dora-bot')
_aws_region: ContextVar[str] = ContextVar('aws_region', default='')


TABLE_SUFFIXES = {
    'demo': '_demo',
    'prod': '_prod',
}

_dynamo_resource_cache: Dict[str, Any] = {}
_dynamo_table_cache: Dict[str, Any] = {}


def get_bot_version() -> str:
    """Get bot version from environment or git."""
    version = os.getenv('BOT_VERSION')
    if version:
        return version

    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return 'unknown'


def generate_run_id() -> str:
    """Generate a unique run ID for this bot instance."""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    suffix = secrets.token_hex(3)  # 6 character random suffix
    return f"{timestamp}-{suffix}"


def generate_decision_id(bot_run_id: str, market: str, loop_counter: int) -> str:
    """Generate a unique decision ID for a market processing cycle."""
    return f"{bot_run_id}:{market}:{loop_counter}"


class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs one JSON object per line with standardized fields for
    CloudWatch Insights querying.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base fields present in every log
        log_entry: Dict[str, Any] = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'service': _service.get(),
            'env': _env.get(),
            'bot_version': _bot_version.get(),
            'bot_run_id': _bot_run_id.get(),
            'message': record.getMessage(),
            'event_type': getattr(record, 'event_type', EventType.LOG.value),
        }

        # Add correlation IDs if set
        decision_id = getattr(record, 'decision_id', None) or _decision_id.get()
        if decision_id:
            log_entry['decision_id'] = decision_id

        market = getattr(record, 'market', None) or _market.get()
        if market:
            log_entry['market'] = market

        # Add any extra fields from the record
        extra_fields = getattr(record, 'extra_fields', {})
        if extra_fields:
            log_entry.update(extra_fields)

        # Handle exceptions
        if record.exc_info:
            log_entry['error_type'] = record.exc_info[0].__name__ if record.exc_info[0] else 'Unknown'
            log_entry['error_msg'] = str(record.exc_info[1]) if record.exc_info[1] else ''
            log_entry['stack'] = ''.join(traceback.format_exception(*record.exc_info))

        # Add logger name for debugging
        log_entry['logger'] = record.name

        return json.dumps(log_entry, default=str)


class StructuredLogAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes structured fields."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message to include extra fields."""
        extra = kwargs.get('extra', {})

        # Extract known fields from extra dict
        event_type = extra.pop('event_type', EventType.LOG.value)
        decision_id = extra.pop('decision_id', None)
        market = extra.pop('market', None)

        # Remaining extra fields go into extra_fields
        kwargs['extra'] = {
            'event_type': event_type if isinstance(event_type, str) else event_type.value,
            'decision_id': decision_id,
            'market': market,
            'extra_fields': extra,
        }

        return msg, kwargs


def setup_structured_logging(
    service: str = 'dora-bot',
    env: str = 'demo',
    bot_version: Optional[str] = None,
    bot_run_id: Optional[str] = None,
    aws_region: Optional[str] = None,
    level: int = logging.INFO
) -> str:
    """Configure structured JSON logging for the application.

    Args:
        service: Service name for logs
        env: Environment (demo/prod)
        bot_version: Bot version string (auto-detected if not provided)
        bot_run_id: Run ID (generated if not provided)
        aws_region: AWS region for DynamoDB logging
        level: Logging level

    Returns:
        The bot_run_id being used
    """
    # Set context variables
    _service.set(service)
    _env.set(env)
    if aws_region is None:
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
    _aws_region.set(aws_region)

    if bot_version is None:
        bot_version = get_bot_version()
    _bot_version.set(bot_version)

    if bot_run_id is None:
        bot_run_id = generate_run_id()
    _bot_run_id.set(bot_run_id)

    # Configure root logger with JSON formatter
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stdout handler with JSON formatter
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter())
    root_logger.addHandler(handler)

    return bot_run_id


def get_logger(name: str) -> StructuredLogAdapter:
    """Get a structured logger adapter for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogAdapter wrapping the named logger
    """
    return StructuredLogAdapter(logging.getLogger(name), {})


logger = get_logger(__name__)


def set_context(
    decision_id: Optional[str] = None,
    market: Optional[str] = None
) -> None:
    """Set context variables for correlation.

    Args:
        decision_id: Current decision ID
        market: Current market ticker
    """
    if decision_id is not None:
        _decision_id.set(decision_id)
    if market is not None:
        _market.set(market)


def clear_context() -> None:
    """Clear context variables."""
    _decision_id.set('')
    _market.set('')


def get_run_id() -> str:
    """Get the current bot run ID."""
    return _bot_run_id.get()


def get_version() -> str:
    """Get the current bot version."""
    return _bot_version.get()


def _get_aws_region() -> str:
    region = _aws_region.get()
    if region:
        return region
    return os.getenv('AWS_REGION', 'us-east-1')


def _table_name(base_name: str, environment: str) -> str:
    suffix = TABLE_SUFFIXES.get(environment)
    if suffix is None:
        raise ValueError(f"Invalid environment: {environment}")
    return f"{base_name}{suffix}"


def _get_dynamo_table(table_name: str, region: str):
    cache_key = f"{region}:{table_name}"
    if cache_key in _dynamo_table_cache:
        return _dynamo_table_cache[cache_key]

    if region not in _dynamo_resource_cache:
        _dynamo_resource_cache[region] = boto3.resource('dynamodb', region_name=region)

    table = _dynamo_resource_cache[region].Table(table_name)
    _dynamo_table_cache[cache_key] = table
    return table


def _to_dynamo_item(obj: Any) -> Any:
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_dynamo_item(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo_item(item) for item in obj]
    return obj


def _normalize_event_type(event_type: Any) -> str:
    if isinstance(event_type, Enum):
        return event_type.value
    return str(event_type)


def _get_ttl_days() -> int:
    try:
        return int(os.getenv('EXECUTION_LOG_TTL_DAYS', '30'))
    except ValueError:
        return 30


def log_decision_record(
    decision_data: Dict[str, Any],
    region: Optional[str] = None,
    environment: Optional[str] = None,
) -> bool:
    """Persist a decision record to DynamoDB (best-effort)."""
    try:
        region = region or _get_aws_region()
        environment = environment or _env.get()
        if not environment:
            environment = 'demo'

        item = dict(decision_data)
        item.setdefault('bot_run_id', get_run_id())
        item.setdefault('bot_version', get_version())
        item.setdefault('date', datetime.utcnow().strftime('%Y-%m-%d'))
        item.setdefault('timestamp', datetime.utcnow().isoformat())

        table_name = _table_name('dora_decision_log', environment)
        table = _get_dynamo_table(table_name, region)
        table.put_item(Item=_to_dynamo_item(item))
        return True
    except ClientError as e:
        logger.error("Failed to write decision log record", extra={
            "event_type": EventType.ERROR,
            "error_type": "ClientError",
            "error_msg": str(e),
            "target": "decision_log",
        })
        return False
    except Exception as e:
        logger.error("Failed to write decision log record", extra={
            "event_type": EventType.ERROR,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "target": "decision_log",
        })
        return False


def log_execution_event(
    payload: Dict[str, Any],
    region: Optional[str] = None,
    environment: Optional[str] = None,
) -> bool:
    """Persist an execution event to DynamoDB (best-effort)."""
    try:
        region = region or _get_aws_region()
        environment = environment or _env.get()
        if not environment:
            environment = 'demo'

        decision_id = payload.get('decision_id') or _decision_id.get()
        bot_run_id = payload.get('bot_run_id') or get_run_id()
        if not decision_id or not bot_run_id:
            logger.error("Missing decision_id or bot_run_id for execution log", extra={
                "event_type": EventType.ERROR,
                "decision_id": decision_id,
                "bot_run_id": bot_run_id,
                "target": "execution_log",
            })
            return False

        event_ts = payload.get('event_ts') or datetime.now(timezone.utc).isoformat()
        event_type = payload.get('event_type', EventType.LOG)

        item = dict(payload)
        item['bot_run_id'] = bot_run_id
        item['decision_id'] = decision_id
        item['event_ts'] = event_ts
        item['decision_id#event_ts'] = f"{decision_id}#{event_ts}"
        item['event_type'] = _normalize_event_type(event_type)
        item.setdefault('bot_version', get_version())
        item.setdefault('env', environment)

        if 'expires_at' not in item:
            ttl_days = _get_ttl_days()
            expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
            item['expires_at'] = int(expires_at.timestamp())

        table_name = _table_name('dora_execution_log', environment)
        table = _get_dynamo_table(table_name, region)
        table.put_item(Item=_to_dynamo_item(item))
        return True
    except ClientError as e:
        logger.error("Failed to write execution log event", extra={
            "event_type": EventType.ERROR,
            "error_type": "ClientError",
            "error_msg": str(e),
            "target": "execution_log",
        })
        return False
    except Exception as e:
        logger.error("Failed to write execution log event", extra={
            "event_type": EventType.ERROR,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "target": "execution_log",
        })
        return False
