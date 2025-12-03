"""
Unit Tests for Anthropic Client Module
"""

from unittest.mock import patch

from insightboost.api.anthropic_client import AnthropicClient
from insightboost.api.rate_limiter import RateLimiter


class TestAnthropicClient:
    """Test suite for AnthropicClient class."""

    def test_instantiation(self):
        """Test that AnthropicClient can be instantiated."""
        with patch("anthropic.Anthropic"):
            client = AnthropicClient()
            assert client is not None

    def test_has_client_attribute(self):
        """Test client has anthropic client attribute."""
        with patch("anthropic.Anthropic") as mock:
            client = AnthropicClient()
            assert hasattr(client, "client") or hasattr(client, "_client")


class TestRateLimiter:
    """Test suite for RateLimiter class."""

    def test_instantiation(self):
        """Test RateLimiter can be instantiated."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter is not None

    def test_has_requests_per_minute(self):
        """Test RateLimiter has requests_per_minute."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.requests_per_minute == 60

    def test_acquire_returns(self):
        """Test acquire method exists and returns."""
        limiter = RateLimiter(requests_per_minute=60)
        # Should not raise
        result = limiter.acquire()
        assert result is None or result is True or isinstance(result, (int, float))

    def test_multiple_acquires_under_limit(self):
        """Test multiple acquires under limit."""
        limiter = RateLimiter(requests_per_minute=100)
        # Should allow multiple requests
        for _ in range(5):
            limiter.acquire()
