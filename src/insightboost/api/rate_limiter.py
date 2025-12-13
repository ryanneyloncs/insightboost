"""
Rate limiting utilities for API calls.

This module implements token bucket rate limiting for controlling
API request rates and preventing overload.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from insightboost.config.logging_config import get_logger
from insightboost.utils.exceptions import RateLimitError

logger = get_logger("rate_limiter")


@dataclass
class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    The token bucket algorithm allows bursts of requests while
    maintaining an average rate limit over time.

    Attributes:
        capacity: Maximum tokens in the bucket
        refill_rate: Tokens added per second
        tokens: Current token count
        last_refill: Timestamp of last refill
        lock: Thread lock for synchronization
    """

    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Initialize bucket to full capacity."""
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_for_token(
        self,
        tokens: float = 1.0,
        timeout: float | None = None,
    ) -> bool:
        """
        Wait until tokens are available and consume them.

        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if tokens were consumed, False if timeout
        """
        start_time = time.monotonic()

        while True:
            if self.consume(tokens):
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            # Calculate wait time until we have enough tokens
            with self.lock:
                self._refill()
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate

            # Don't wait longer than remaining timeout
            if timeout is not None:
                remaining = timeout - (time.monotonic() - start_time)
                wait_time = min(wait_time, remaining)

            if wait_time > 0:
                time.sleep(min(wait_time, 0.1))  # Cap at 100ms per sleep

    def get_wait_time(self, tokens: float = 1.0) -> float:
        """
        Get estimated wait time for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated seconds to wait
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self.lock:
            self._refill()
            return self.tokens


class RateLimiter:
    """
    Rate limiter for API requests.

    Provides a simple interface for rate limiting with configurable
    requests per minute and burst handling.

    Attributes:
        requests_per_minute: Target request rate
        burst_capacity: Maximum burst size
        bucket: Underlying token bucket
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_capacity: int | None = None,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Target requests per minute
            burst_capacity: Maximum burst (defaults to 1.5x rate)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_capacity = burst_capacity or int(requests_per_minute * 1.5)

        # Convert to per-second rate
        refill_rate = requests_per_minute / 60.0

        self.bucket = TokenBucket(
            capacity=float(self.burst_capacity),
            refill_rate=refill_rate,
        )

        logger.debug(
            f"RateLimiter initialized: {requests_per_minute}/min, "
            f"burst={self.burst_capacity}"
        )

    def acquire(
        self,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """
        Acquire permission to make a request.

        Args:
            blocking: Whether to wait for permission
            timeout: Maximum wait time (for blocking mode)

        Returns:
            True if permission granted, False otherwise

        Raises:
            RateLimitError: If non-blocking and rate limited
        """
        if blocking:
            success = self.bucket.wait_for_token(timeout=timeout)
            if not success:
                raise RateLimitError(
                    message="Rate limit timeout exceeded",
                    retry_after=int(self.bucket.get_wait_time()),
                )
            return True
        else:
            success = self.bucket.consume()
            if not success:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    retry_after=int(self.bucket.get_wait_time()),
                )
            return True

    def try_acquire(self) -> bool:
        """
        Try to acquire permission without blocking.

        Returns:
            True if permission granted, False if rate limited
        """
        return self.bucket.consume()

    def get_wait_time(self) -> float:
        """
        Get estimated wait time until next request allowed.

        Returns:
            Seconds to wait
        """
        return self.bucket.get_wait_time()

    @property
    def available_requests(self) -> int:
        """Get number of requests that can be made immediately."""
        return int(self.bucket.available_tokens)

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        with self.bucket.lock:
            self.bucket.tokens = self.bucket.capacity
            self.bucket.last_refill = time.monotonic()
        logger.debug("RateLimiter reset to full capacity")


class RateLimitedExecutor:
    """
    Executor that applies rate limiting to function calls.

    Useful for wrapping API calls with automatic rate limiting.
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize the executor.

        Args:
            rate_limiter: Rate limiter to use
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
        """
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ):
        """
        Execute a function with rate limiting and retries.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function

        Raises:
            RateLimitError: If rate limit exceeded after retries
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Acquire rate limit permission
                self.rate_limiter.acquire(blocking=True, timeout=30.0)

                # Execute the function
                return func(*args, **kwargs)

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = e.retry_after or self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(wait_time)

        raise last_error or RateLimitError("Rate limit exceeded after retries")


# =============================================================================
# Web Request Rate Limiting
# =============================================================================

# Global rate limiters for web endpoints (keyed by IP address)
_ip_rate_limiters: dict[str, RateLimiter] = {}
_ip_limiter_lock = threading.Lock()


def get_ip_rate_limiter(
    ip_address: str,
    requests_per_minute: int = 60,
) -> RateLimiter:
    """Get or create a rate limiter for an IP address."""
    with _ip_limiter_lock:
        if ip_address not in _ip_rate_limiters:
            _ip_rate_limiters[ip_address] = RateLimiter(
                requests_per_minute=requests_per_minute
            )
        return _ip_rate_limiters[ip_address]


def rate_limit(
    requests_per_minute: int = 60,
    error_message: str = "Rate limit exceeded. Please try again later.",
):
    """
    Decorator to apply rate limiting to Flask routes.

    Args:
        requests_per_minute: Maximum requests per minute per IP
        error_message: Message to return when rate limited

    Usage:
        @app.route("/api/endpoint")
        @rate_limit(requests_per_minute=30)
        def my_endpoint():
            ...
    """
    from functools import wraps

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import current_app, jsonify, request

            # Skip rate limiting during tests
            if current_app.config.get("TESTING", False):
                return func(*args, **kwargs)

            # Get client IP (handle proxies)
            if request.headers.get("X-Forwarded-For"):
                ip_address = (
                    request.headers.get("X-Forwarded-For").split(",")[0].strip()
                )
            else:
                ip_address = request.remote_addr or "unknown"

            limiter = get_ip_rate_limiter(ip_address, requests_per_minute)

            if not limiter.try_acquire():
                retry_after = int(limiter.get_wait_time()) + 1
                response = jsonify(
                    {
                        "error": True,
                        "error_code": "RATE_LIMITED",
                        "message": error_message,
                        "retry_after": retry_after,
                    }
                )
                response.status_code = 429
                response.headers["Retry-After"] = str(retry_after)
                return response

            return func(*args, **kwargs)

        return wrapper

    return decorator


def cleanup_rate_limiters(max_age_seconds: int = 3600) -> int:
    """
    Clean up old rate limiters to prevent memory leaks.
    Call periodically (e.g., via a background task).

    Returns:
        Number of limiters removed
    """
    # Note: This is a simple implementation. In production,
    # consider using Redis for distributed rate limiting.
    with _ip_limiter_lock:
        # For now, just clear all if there are too many
        if len(_ip_rate_limiters) > 10000:
            count = len(_ip_rate_limiters)
            _ip_rate_limiters.clear()
            logger.info(f"Cleared {count} rate limiters")
            return count
    return 0
