"""
Unit Tests for Insights Generator Module
"""

import pytest
from unittest.mock import MagicMock, patch

from insightboost.core.insights_generator import InsightsGenerator


class TestInsightsGenerator:
    """Test suite for InsightsGenerator class."""

    def test_instantiation(self):
        """Test that InsightsGenerator can be instantiated."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            assert generator is not None

    def test_has_api_client(self):
        """Test generator has api_client attribute."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            assert hasattr(generator, 'api_client')

    def test_has_data_analyzer(self):
        """Test generator has data_analyzer attribute."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            assert hasattr(generator, 'data_analyzer')

    def test_has_cache(self):
        """Test generator has cache attribute."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            assert hasattr(generator, 'cache')

    def test_has_analyze_method(self):
        """Test generator has analyze method."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            assert hasattr(generator, 'analyze_dataframe')

    def test_has_clear_cache_method(self):
        """Test generator has clear_cache method."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            assert hasattr(generator, 'clear_cache')

    def test_clear_cache(self):
        """Test clear_cache method works."""
        with patch("insightboost.core.insights_generator.AnthropicClient"):
            generator = InsightsGenerator()
            generator.clear_cache()  # Should not raise
