"""Structured JSON logging for CloudWatch integration.

This module provides a JSON logging formatter and helper utilities for
emitting structured log events that can be queried in CloudWatch Insights.
"""

import json
import logging
import os
import subprocess
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import secrets


class EventType(str, Enum):
    """Enumeration of structured log event types."""
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    HEARTBEAT = "HEARTBEAT"
    DECISION_MADE = "DECISION_MADE"
    ORDER_PLACE = "ORDER_PLACE"
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


# Context variables for correlation IDs
_bot_run_id: ContextVar[str] = ContextVar('bot_run_id', default='')
_bot_version: ContextVar[str] = ContextVar('bot_version', default='')
_decision_id: ContextVar[str] = ContextVar('decision_id', default='')
_market: ContextVar[str] = ContextVar('market', default='')
_env: ContextVar[str] = ContextVar('env', default='')
_service: ContextVar[str] = ContextVar('service', default='dora-bot')


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
    level: int = logging.INFO
) -> str:
    """Configure structured JSON logging for the application.

    Args:
        service: Service name for logs
        env: Environment (demo/prod)
        bot_version: Bot version string (auto-detected if not provided)
        bot_run_id: Run ID (generated if not provided)
        level: Logging level

    Returns:
        The bot_run_id being used
    """
    # Set context variables
    _service.set(service)
    _env.set(env)

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
