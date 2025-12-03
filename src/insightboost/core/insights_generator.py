"""
Core insights generation engine for InsightBoost.

This module provides the main InsightsGenerator class that orchestrates
AI-powered data analysis and insight generation.
"""

from typing import Any, Literal

import pandas as pd
from cachetools import TTLCache

from insightboost.api.anthropic_client import AnthropicClient
from insightboost.config.logging_config import get_logger
from insightboost.config.settings import get_settings
from insightboost.core.data_analyzer import DataAnalyzer
from insightboost.models.insight import DatasetAnalysis, Insight, Pattern, PatternType
from insightboost.utils.exceptions import InsightGenerationError
from insightboost.utils.validators import validate_dataframe, validate_query

logger = get_logger("insights_generator")


class InsightsGenerator:
    """
    Core engine for AI-driven insights generation.

    This class orchestrates data analysis pipelines, interfaces with the
    Anthropic API for LLM-powered insights, and manages insight caching.

    Attributes:
        api_client: Anthropic API client
        data_analyzer: Statistical data analyzer
        cache: TTL cache for insights
    """

    def __init__(
        self,
        api_client: AnthropicClient | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        """
        Initialize the insights generator.

        Args:
            api_client: Anthropic API client (created if not provided)
            cache: Optional cache for insights
        """
        settings = get_settings()

        self.api_client = api_client or AnthropicClient()
        self.data_analyzer = DataAnalyzer()

        # Initialize cache
        cache_ttl = settings.visualization_cache_ttl_seconds
        self.cache = cache or TTLCache(maxsize=100, ttl=cache_ttl)

        logger.info("InsightsGenerator initialized")

    def _get_cache_key(self, df: pd.DataFrame, context: str = "") -> str:
        """Generate a cache key for a DataFrame analysis."""
        # Use shape, columns, and first/last values for cache key
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        cols_str = ",".join(sorted(df.columns[:10]))
        return f"{shape_str}:{cols_str}:{hash(context)}"

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        context: str | None = None,
        depth: Literal["quick", "standard", "deep"] = "standard",
        use_cache: bool = True,
    ) -> DatasetAnalysis:
        """
        Perform comprehensive dataset analysis.

        This method combines statistical analysis with AI-powered insights
        to provide a complete understanding of the dataset.

        Args:
            df: Input DataFrame to analyze
            context: Optional business context for analysis
            depth: Analysis depth ('quick', 'standard', 'deep')
            use_cache: Whether to use cached results

        Returns:
            DatasetAnalysis with structure, quality, and pattern insights

        Raises:
            DataValidationError: If DataFrame is invalid
            InsightGenerationError: If analysis fails
        """
        logger.info(f"Analyzing DataFrame: {df.shape}, depth={depth}")

        # Validate input
        df = validate_dataframe(df)
        context = context or ""

        # Check cache
        cache_key = self._get_cache_key(df, context)
        if use_cache and cache_key in self.cache:
            logger.debug("Returning cached analysis")
            return self.cache[cache_key]

        try:
            # First, do statistical analysis
            profile = self.data_analyzer.create_profile(df)

            # Then, get AI-powered analysis
            ai_analysis = self.api_client.analyze_dataset(
                df=df,
                context=context,
                depth=depth,
            )

            # Merge statistical patterns with AI patterns
            all_patterns = list(ai_analysis.patterns)

            # Add correlation-based patterns from statistical analysis
            for corr in profile.correlations[:5]:  # Top 5 correlations
                pattern = Pattern(
                    pattern_type=PatternType.CORRELATION,
                    description=f"{corr.strength.title()} {corr.direction} correlation between {corr.column1} and {corr.column2}",
                    columns_involved=[corr.column1, corr.column2],
                    strength=abs(corr.correlation),
                )
                all_patterns.append(pattern)

            # Add quality issues as patterns
            for issue in profile.quality_issues[:3]:
                pattern = Pattern(
                    pattern_type=PatternType.ANOMALY,
                    description=issue,
                    columns_involved=[],
                    strength=0.6,
                )
                all_patterns.append(pattern)

            # Update analysis with merged patterns
            ai_analysis.patterns = all_patterns

            # Cache the result
            if use_cache:
                self.cache[cache_key] = ai_analysis

            logger.info(
                f"Analysis complete: {len(ai_analysis.insights)} insights, {len(ai_analysis.patterns)} patterns"
            )

            return ai_analysis

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise InsightGenerationError(
                message=f"Failed to analyze dataset: {str(e)}",
                details={"shape": df.shape, "depth": depth},
            ) from e

    def get_insights(
        self,
        df: pd.DataFrame,
        query: str,
        max_insights: int = 5,
    ) -> list[Insight]:
        """
        Generate insights based on a natural language query.

        Args:
            df: Dataset to analyze
            query: Natural language question or hypothesis
            max_insights: Maximum insights to return

        Returns:
            List of Insight objects with findings and confidence scores

        Raises:
            DataValidationError: If inputs are invalid
            InsightGenerationError: If insight generation fails
        """
        logger.info(f"Generating insights for query: {query[:50]}...")

        # Validate inputs
        df = validate_dataframe(df)
        query = validate_query(query)

        try:
            insights = self.api_client.generate_insights(
                df=df,
                query=query,
                max_insights=max_insights,
            )

            logger.info(f"Generated {len(insights)} insights")
            return insights

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            raise InsightGenerationError(
                message=f"Failed to generate insights: {str(e)}",
                query=query,
            ) from e

    def discover_patterns(
        self,
        df: pd.DataFrame,
        pattern_types: list[str] | None = None,
    ) -> list[Pattern]:
        """
        Automatically discover patterns in data.

        Args:
            df: DataFrame to analyze
            pattern_types: Specific types to look for (None = all)

        Returns:
            List of discovered Pattern objects
        """
        logger.info("Discovering patterns in data")

        df = validate_dataframe(df)

        patterns = []
        allowed_types = set(pattern_types) if pattern_types else None

        # Correlation patterns
        if allowed_types is None or "correlations" in allowed_types:
            correlations = self.data_analyzer.compute_correlations(df, threshold=0.5)
            for corr in correlations[:10]:
                patterns.append(
                    Pattern(
                        pattern_type=PatternType.CORRELATION,
                        description=f"{corr.strength.title()} {corr.direction} correlation ({corr.correlation:.2f}) between {corr.column1} and {corr.column2}",
                        columns_involved=[corr.column1, corr.column2],
                        strength=abs(corr.correlation),
                        details={"correlation": corr.correlation, "method": "pearson"},
                    )
                )

        # Distribution patterns
        if allowed_types is None or "distributions" in allowed_types:
            for col in df.select_dtypes(include=["number"]).columns:
                dist = self.data_analyzer.analyze_distribution(df[col])
                if dist:
                    # Check for skewed distribution
                    if abs(dist.skewness) > 1:
                        direction = "right" if dist.skewness > 0 else "left"
                        patterns.append(
                            Pattern(
                                pattern_type=PatternType.DISTRIBUTION,
                                description=f"Column '{col}' has a {direction}-skewed distribution (skewness: {dist.skewness:.2f})",
                                columns_involved=[col],
                                strength=min(abs(dist.skewness) / 2, 1.0),
                                details={
                                    "skewness": dist.skewness,
                                    "kurtosis": dist.kurtosis,
                                },
                            )
                        )

                    # Check for outliers
                    if dist.outlier_percentage > 5:
                        patterns.append(
                            Pattern(
                                pattern_type=PatternType.ANOMALY,
                                description=f"Column '{col}' contains {dist.outlier_percentage:.1f}% outliers",
                                columns_involved=[col],
                                strength=min(dist.outlier_percentage / 20, 1.0),
                                details={
                                    "outlier_count": dist.outlier_count,
                                    "outlier_pct": dist.outlier_percentage,
                                },
                            )
                        )

        # Trend patterns
        if allowed_types is None or "trends" in allowed_types:
            for col in df.select_dtypes(include=["number"]).columns:
                ts = self.data_analyzer.detect_time_series(df, col)
                if ts and ts.has_trend:
                    patterns.append(
                        Pattern(
                            pattern_type=PatternType.TREND,
                            description=f"Column '{col}' shows an {ts.trend_direction}ward trend",
                            columns_involved=[col],
                            strength=0.7,
                            details={"trend_direction": ts.trend_direction},
                        )
                    )

        # Cluster patterns (simplified - check for natural groupings in categorical columns)
        if allowed_types is None or "clusters" in allowed_types:
            for col in df.select_dtypes(include=["object", "category"]).columns:
                cardinality = df[col].nunique()
                if 2 <= cardinality <= 10:
                    value_counts = df[col].value_counts()
                    dominant = value_counts.iloc[0] / len(df) * 100
                    if dominant > 50:
                        patterns.append(
                            Pattern(
                                pattern_type=PatternType.CLUSTER,
                                description=f"Column '{col}' is dominated by '{value_counts.index[0]}' ({dominant:.1f}%)",
                                columns_involved=[col],
                                strength=dominant / 100,
                                details={
                                    "dominant_value": str(value_counts.index[0]),
                                    "percentage": dominant,
                                },
                            )
                        )

        # Sort by strength
        patterns.sort(key=lambda p: p.strength, reverse=True)

        logger.info(f"Discovered {len(patterns)} patterns")
        return patterns

    def explain_insight(
        self,
        insight: Insight,
        audience: Literal["technical", "business", "executive"] = "business",
    ) -> str:
        """
        Generate a human-readable explanation of an insight.

        Args:
            insight: Insight to explain
            audience: Target audience for the explanation

        Returns:
            Human-readable explanation string
        """
        context = {
            "insight_type": insight.insight_type.value,
            "confidence": insight.confidence,
            "columns": insight.columns_involved,
            "supporting_data": insight.supporting_data,
            "audience": audience,
        }

        try:
            explanation = self.api_client.explain_pattern(
                pattern=f"{insight.title}: {insight.description}",
                context=context,
            )
            return explanation
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            # Fallback to basic explanation
            return (
                f"{insight.title}\n\n"
                f"{insight.description}\n\n"
                f"Confidence: {insight.confidence_label} ({insight.confidence:.0%})"
            )

    def quick_analyze(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Perform a quick statistical analysis without AI.

        Useful for getting immediate feedback while full analysis runs.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quick analysis results
        """
        logger.debug("Performing quick analysis")

        df = validate_dataframe(df)

        # Get basic stats
        profile = self.data_analyzer.create_profile(df)

        return {
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "memory_mb": round(profile.memory_usage_bytes / (1024 * 1024), 2),
            "columns": [
                {
                    "name": col.name,
                    "type": col.data_type.value,
                    "null_pct": col.null_percentage,
                    "unique_count": col.unique_count,
                }
                for col in profile.columns
            ],
            "top_correlations": [
                {
                    "columns": [c.column1, c.column2],
                    "correlation": c.correlation,
                    "strength": c.strength,
                }
                for c in profile.correlations[:5]
            ],
            "quality_issues": profile.quality_issues,
        }

    def get_token_usage(self) -> dict[str, Any]:
        """Get token usage statistics."""
        return self.api_client.get_token_usage()

    def clear_cache(self) -> None:
        """Clear the insights cache."""
        self.cache.clear()
        logger.info("Insights cache cleared")
