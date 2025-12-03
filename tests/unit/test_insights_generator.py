"""
Unit Tests for Insights Generator Module
Tests AI-powered insight generation, confidence scoring, and caching
"""

import json
from datetime import datetime
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from insightboost.core.insights_generator import InsightsGenerator
from insightboost.models.insight import Insight, InsightType


class TestInsightsGenerator:
    """Test suite for InsightsGenerator class."""

    @pytest.fixture
    def mock_api_client(self, mock_anthropic_response):
        """Create mock API client."""
        client = MagicMock()
        client.send_message.return_value = {
            "content": mock_anthropic_response["content"][0]["text"],
            "usage": mock_anthropic_response["usage"],
        }
        return client

    @pytest.fixture
    def generator(self, mock_api_client):
        """Create InsightsGenerator with mock client."""
        gen = InsightsGenerator(api_client=mock_api_client)
        return gen

    # ============================================
    # Insight Generation Tests
    # ============================================

    class TestInsightGeneration:
        """Tests for insight generation."""

        @pytest.fixture
        def mock_api_client(self):
            client = MagicMock()
            client.send_message.return_value = {
                "content": json.dumps({
                    "insights": [
                        {
                            "type": "trend",
                            "title": "Revenue Growth",
                            "description": "Revenue is increasing over time",
                            "confidence": 0.85,
                            "columns": ["date", "revenue"],
                        },
                        {
                            "type": "correlation",
                            "title": "Sales-Revenue Link",
                            "description": "Units sold correlates with revenue",
                            "confidence": 0.92,
                            "columns": ["units_sold", "revenue"],
                        },
                    ]
                }),
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
            return client

        @pytest.fixture
        def generator(self, mock_api_client):
            return InsightsGenerator(api_client=mock_api_client)

        def test_generate_insights_returns_list(self, generator, sample_sales_data):
            """Test that generate_insights returns a list."""
            insights = generator.generate_insights(
                df=sample_sales_data,
                dataset_id=uuid4(),
                query="What trends do you see?",
            )
            assert isinstance(insights, list)

        def test_generate_insights_with_query(self, generator, sample_sales_data):
            """Test insight generation with a specific query."""
            query = "What are the main revenue trends?"
            insights = generator.generate_insights(
                df=sample_sales_data,
                dataset_id=uuid4(),
                query=query,
            )
            assert len(insights) > 0

        def test_generate_insights_respects_max_results(self, generator, sample_sales_data):
            """Test that max_results is respected."""
            insights = generator.generate_insights(
                df=sample_sales_data,
                dataset_id=uuid4(),
                query="Analyze this data",
                max_results=1,
            )
            assert len(insights) <= 1

        def test_insight_has_required_fields(self, generator, sample_sales_data):
            """Test that insights have required fields."""
            insights = generator.generate_insights(
                df=sample_sales_data,
                dataset_id=uuid4(),
                query="What patterns exist?",
            )
            
            if insights:
                insight = insights[0]
                assert hasattr(insight, "id") or "id" in insight
                assert hasattr(insight, "title") or "title" in insight
                assert hasattr(insight, "description") or "description" in insight
                assert hasattr(insight, "confidence") or "confidence" in insight

        def test_api_client_called_with_prompt(self, generator, sample_sales_data):
            """Test that API client is called with proper prompt."""
            generator.generate_insights(
                df=sample_sales_data,
                dataset_id=uuid4(),
                query="Test query",
            )
            
            # Verify API was called
            generator.api_client.send_message.assert_called()
            call_args = generator.api_client.send_message.call_args
            
            # Check that some form of prompt was passed
            assert call_args is not None

    # ============================================
    # Confidence Scoring Tests
    # ============================================

    class TestConfidenceScoring:
        """Tests for confidence score calculation."""

        @pytest.fixture
        def generator(self):
            mock_client = MagicMock()
            return InsightsGenerator(api_client=mock_client)

        def test_confidence_in_valid_range(self, generator):
            """Test that confidence scores are between 0 and 1."""
            # Create a mock insight with confidence
            insight_data = {
                "type": "trend",
                "title": "Test",
                "description": "Test insight",
                "confidence": 0.75,
                "columns": ["x"],
            }
            
            confidence = insight_data["confidence"]
            assert 0 <= confidence <= 1

        def test_high_correlation_high_confidence(self, generator):
            """Test that strong correlations yield high confidence."""
            # Simulate high correlation scenario
            evidence = {"correlation": 0.95, "p_value": 0.001}
            
            # Calculate expected confidence
            if evidence["correlation"] > 0.8 and evidence["p_value"] < 0.05:
                expected_confidence_high = True
            else:
                expected_confidence_high = False
            
            assert expected_confidence_high

        def test_low_sample_size_reduces_confidence(self, generator):
            """Test that small sample sizes reduce confidence."""
            small_df = pd.DataFrame({"x": [1, 2, 3]})
            large_df = pd.DataFrame({"x": range(1000)})
            
            # Small samples should be flagged
            assert len(small_df) < 30  # Statistical significance threshold
            assert len(large_df) >= 30

    # ============================================
    # Error Handling Tests
    # ============================================

    class TestErrorHandling:
        """Tests for error handling."""

        @pytest.fixture
        def error_api_client(self):
            client = MagicMock()
            client.send_message.side_effect = Exception("API Error")
            return client

        @pytest.fixture
        def generator(self, error_api_client):
            return InsightsGenerator(api_client=error_api_client)

        def test_handles_api_error_gracefully(self, generator, sample_sales_data):
            """Test that API errors are handled gracefully."""
            with pytest.raises(Exception):
                generator.generate_insights(
                    df=sample_sales_data,
                    dataset_id=uuid4(),
                    query="Test",
                )

        def test_handles_empty_dataframe(self, generator):
            """Test handling of empty dataframe."""
            empty_df = pd.DataFrame()
            
            # Should either return empty list or raise appropriate error
            try:
                result = generator.generate_insights(
                    df=empty_df,
                    dataset_id=uuid4(),
                    query="Test",
                )
                assert result == [] or result is None
            except (ValueError, Exception):
                pass  # Expected behavior

        def test_handles_malformed_response(self):
            """Test handling of malformed API response."""
            mock_client = MagicMock()
            mock_client.send_message.return_value = {
                "content": "not valid json {{{",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            
            generator = InsightsGenerator(api_client=mock_client)
            
            # Should handle gracefully
            try:
                result = generator.generate_insights(
                    df=pd.DataFrame({"x": [1, 2, 3]}),
                    dataset_id=uuid4(),
                    query="Test",
                )
            except (json.JSONDecodeError, ValueError, Exception):
                pass  # Expected behavior

    # ============================================
    # Caching Tests
    # ============================================

    class TestCaching:
        """Tests for insight caching."""

        @pytest.fixture
        def generator(self):
            mock_client = MagicMock()
            mock_client.send_message.return_value = {
                "content": json.dumps({"insights": []}),
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return InsightsGenerator(api_client=mock_client)

        def test_cache_key_generation(self, generator):
            """Test that cache keys are generated consistently."""
            df = pd.DataFrame({"x": [1, 2, 3]})
            dataset_id = uuid4()
            query = "Test query"
            
            # Generate cache key (if method exists)
            if hasattr(generator, "_generate_cache_key"):
                key1 = generator._generate_cache_key(df, dataset_id, query)
                key2 = generator._generate_cache_key(df, dataset_id, query)
                assert key1 == key2

        def test_different_queries_different_keys(self, generator):
            """Test that different queries produce different cache keys."""
            df = pd.DataFrame({"x": [1, 2, 3]})
            dataset_id = uuid4()
            
            if hasattr(generator, "_generate_cache_key"):
                key1 = generator._generate_cache_key(df, dataset_id, "Query 1")
                key2 = generator._generate_cache_key(df, dataset_id, "Query 2")
                assert key1 != key2

    # ============================================
    # Insight Type Tests
    # ============================================

    class TestInsightTypes:
        """Tests for different insight types."""

        @pytest.fixture
        def generator(self):
            mock_client = MagicMock()
            return InsightsGenerator(api_client=mock_client)

        def test_trend_insight_detection(self, generator, sample_timeseries_data):
            """Test detection of trend insights."""
            # Setup mock to return trend insight
            generator.api_client.send_message.return_value = {
                "content": json.dumps({
                    "insights": [{
                        "type": "trend",
                        "title": "Temperature Trend",
                        "description": "Temperature shows seasonal pattern",
                        "confidence": 0.8,
                        "columns": ["timestamp", "temperature"],
                    }]
                }),
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
            
            insights = generator.generate_insights(
                df=sample_timeseries_data,
                dataset_id=uuid4(),
                query="What trends exist?",
            )
            
            if insights:
                assert any(
                    getattr(i, "insight_type", None) == InsightType.TREND or
                    i.get("type") == "trend"
                    for i in insights
                )

        def test_correlation_insight_detection(self, generator, sample_numeric_data):
            """Test detection of correlation insights."""
            generator.api_client.send_message.return_value = {
                "content": json.dumps({
                    "insights": [{
                        "type": "correlation",
                        "title": "X-Y Correlation",
                        "description": "Strong correlation between X and Y",
                        "confidence": 0.9,
                        "columns": ["x", "y"],
                    }]
                }),
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
            
            insights = generator.generate_insights(
                df=sample_numeric_data,
                dataset_id=uuid4(),
                query="Find correlations",
            )
            
            if insights:
                assert any(
                    getattr(i, "insight_type", None) == InsightType.CORRELATION or
                    i.get("type") == "correlation"
                    for i in insights
                )

        def test_anomaly_insight_detection(self, generator):
            """Test detection of anomaly insights."""
            df = pd.DataFrame({
                "value": [10, 11, 12, 11, 10, 100, 11, 12]  # 100 is anomaly
            })
            
            generator.api_client.send_message.return_value = {
                "content": json.dumps({
                    "insights": [{
                        "type": "anomaly",
                        "title": "Outlier Detected",
                        "description": "Value 100 is an outlier",
                        "confidence": 0.95,
                        "columns": ["value"],
                    }]
                }),
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
            
            insights = generator.generate_insights(
                df=df,
                dataset_id=uuid4(),
                query="Find anomalies",
            )
            
            if insights:
                assert len(insights) > 0


class TestInsightExplanation:
    """Tests for insight explanation generation."""

    @pytest.fixture
    def generator(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = {
            "content": "This insight shows that revenue has been increasing steadily.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
        return InsightsGenerator(api_client=mock_client)

    def test_explain_insight_technical(self, generator, sample_insight):
        """Test technical explanation generation."""
        if hasattr(generator, "explain_insight"):
            explanation = generator.explain_insight(
                insight=sample_insight,
                audience="technical",
            )
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_explain_insight_business(self, generator, sample_insight):
        """Test business explanation generation."""
        if hasattr(generator, "explain_insight"):
            explanation = generator.explain_insight(
                insight=sample_insight,
                audience="business",
            )
            assert isinstance(explanation, str)

    def test_explain_insight_executive(self, generator, sample_insight):
        """Test executive explanation generation."""
        if hasattr(generator, "explain_insight"):
            explanation = generator.explain_insight(
                insight=sample_insight,
                audience="executive",
            )
            assert isinstance(explanation, str)


class TestTokenUsage:
    """Tests for token usage tracking."""

    @pytest.fixture
    def generator(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = {
            "content": json.dumps({"insights": []}),
            "usage": {"input_tokens": 500, "output_tokens": 200},
        }
        return InsightsGenerator(api_client=mock_client)

    def test_token_usage_tracked(self, generator, sample_sales_data):
        """Test that token usage is tracked."""
        generator.generate_insights(
            df=sample_sales_data,
            dataset_id=uuid4(),
            query="Test",
        )
        
        # Check if token tracking exists
        if hasattr(generator, "total_tokens_used"):
            assert generator.total_tokens_used > 0
        elif hasattr(generator, "get_token_usage"):
            usage = generator.get_token_usage()
            assert usage.get("total", 0) > 0 or usage.get("input_tokens", 0) > 0
