"""
InsightBoost: AI-Enhanced Data Insights & Visualization Tool

A cutting-edge developer tool that streamlines data exploration,
automates visualization selection, and enables collaborative insights
extraction using Anthropic's Claude API.
"""

__version__ = "0.1.0"
__author__ = "InsightBoost Team"

from insightboost.api.anthropic_client import AnthropicClient
from insightboost.core.data_analyzer import DataAnalyzer
from insightboost.core.insights_generator import InsightsGenerator
from insightboost.core.visualization_suggester import VisualizationSuggester

__all__ = [
    "__version__",
    "DataAnalyzer",
    "InsightsGenerator",
    "VisualizationSuggester",
    "AnthropicClient",
]
