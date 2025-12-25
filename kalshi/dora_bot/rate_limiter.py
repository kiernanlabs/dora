"""Rate limiter with exponential backoff for Kalshi API calls."""

import time
from typing import Optional

from dora_bot.structured_logger import get_logger, EventType

logger = get_logger(__name__)


class RateLimiter:
    """Centralized rate limiter with exponential backoff.

    Implements token bucket algorithm with backoff on rate limit hits.
    """

    def __init__(
        self,
        requests_per_second: float = 20.0,
        burst_limit: int = 20,
        max_backoff_seconds: float = 30.0,
        base_backoff_seconds: float = 1.0,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_second: Sustained rate limit
            burst_limit: Maximum burst size (token bucket capacity)
            max_backoff_seconds: Maximum backoff time after rate limit hits
            base_backoff_seconds: Initial backoff time on first rate limit hit
        """
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.max_backoff_seconds = max_backoff_seconds
        self.base_backoff_seconds = base_backoff_seconds

        # Token bucket state
        self.tokens = float(burst_limit)
        self.last_refill = time.time()

        # Backoff state
        self.backoff_until = 0.0
        self.consecutive_rate_limits = 0

    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.requests_per_second
        self.tokens = min(self.burst_limit, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, count: int = 1, endpoint: Optional[str] = None) -> float:
        """Acquire tokens for API requests, waiting if necessary.

        Args:
            count: Number of tokens to acquire (e.g., number of orders in batch)
            endpoint: Optional endpoint name for logging

        Returns:
            Time waited in seconds
        """
        total_wait = 0.0

        # Check if we're in backoff
        now = time.time()
        if now < self.backoff_until:
            backoff_wait = self.backoff_until - now
            logger.info("Rate limiter backing off", extra={
                "event_type": EventType.RATE_LIMIT_BACKOFF,
                "backoff_seconds": backoff_wait,
                "consecutive_hits": self.consecutive_rate_limits,
                "endpoint": endpoint,
            })
            time.sleep(backoff_wait)
            total_wait += backoff_wait

        # Refill tokens
        self._refill_tokens()

        # Wait for tokens if needed
        while self.tokens < count:
            # Calculate wait time for needed tokens
            needed = count - self.tokens
            wait_time = needed / self.requests_per_second
            time.sleep(wait_time)
            total_wait += wait_time
            self._refill_tokens()

        # Consume tokens
        self.tokens -= count

        if total_wait > 0.1:  # Only log significant waits
            logger.debug("Rate limiter wait", extra={
                "event_type": EventType.LOG,
                "wait_seconds": total_wait,
                "tokens_remaining": self.tokens,
                "endpoint": endpoint,
            })

        return total_wait

    def record_rate_limit_hit(self, endpoint: Optional[str] = None) -> float:
        """Record a 429 rate limit response and calculate backoff.

        Args:
            endpoint: Optional endpoint name for logging

        Returns:
            Backoff time in seconds
        """
        self.consecutive_rate_limits += 1

        # Exponential backoff: base * 2^(consecutive - 1)
        backoff_time = min(
            self.max_backoff_seconds,
            self.base_backoff_seconds * (2 ** (self.consecutive_rate_limits - 1))
        )

        self.backoff_until = time.time() + backoff_time

        logger.warning("Rate limit hit", extra={
            "event_type": EventType.RATE_LIMIT,
            "consecutive_hits": self.consecutive_rate_limits,
            "backoff_seconds": backoff_time,
            "endpoint": endpoint,
        })

        return backoff_time

    def record_success(self) -> None:
        """Record a successful request, resetting backoff state."""
        if self.consecutive_rate_limits > 0:
            logger.debug("Rate limit backoff reset", extra={
                "event_type": EventType.LOG,
                "previous_consecutive_hits": self.consecutive_rate_limits,
            })
        self.consecutive_rate_limits = 0
        self.backoff_until = 0.0

    @property
    def is_backing_off(self) -> bool:
        """Check if currently in backoff period."""
        return time.time() < self.backoff_until

    @property
    def backoff_remaining(self) -> float:
        """Get remaining backoff time in seconds."""
        remaining = self.backoff_until - time.time()
        return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        self._refill_tokens()
        return {
            "tokens_available": self.tokens,
            "burst_limit": self.burst_limit,
            "requests_per_second": self.requests_per_second,
            "consecutive_rate_limits": self.consecutive_rate_limits,
            "is_backing_off": self.is_backing_off,
            "backoff_remaining": self.backoff_remaining,
        }
