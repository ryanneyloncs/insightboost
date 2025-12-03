"""
Unit Tests for Visualization Suggester Module
"""

from unittest.mock import patch

from insightboost.core.visualization_suggester import VisualizationSuggester


class TestVisualizationSuggester:
    """Test suite for VisualizationSuggester class."""

    def test_instantiation(self):
        """Test that VisualizationSuggester can be instantiated."""
        with patch("insightboost.core.visualization_suggester.AnthropicClient"):
            suggester = VisualizationSuggester()
            assert suggester is not None

    def test_has_api_client(self):
        """Test suggester has api_client attribute."""
        with patch("insightboost.core.visualization_suggester.AnthropicClient"):
            suggester = VisualizationSuggester()
            assert hasattr(suggester, "api_client")

    def test_has_data_analyzer(self):
        """Test suggester has data_analyzer attribute."""
        with patch("insightboost.core.visualization_suggester.AnthropicClient"):
            suggester = VisualizationSuggester()
            assert hasattr(suggester, "data_analyzer")

    def test_has_suggest_method(self):
        """Test suggester has suggest_visualizations method."""
        with patch("insightboost.core.visualization_suggester.AnthropicClient"):
            suggester = VisualizationSuggester()
            assert hasattr(suggester, "suggest_visualizations") or hasattr(
                suggester, "suggest"
            )

    def test_has_generate_method(self):
        """Test suggester has generate_visualization method."""
        with patch("insightboost.core.visualization_suggester.AnthropicClient"):
            suggester = VisualizationSuggester()
            assert hasattr(suggester, "generate_visualization") or hasattr(
                suggester, "generate"
            )
