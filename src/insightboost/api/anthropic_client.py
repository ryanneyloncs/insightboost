"""
Anthropic API client wrapper for InsightBoost.

This module provides a robust client for interacting with the Anthropic API,
including retry logic, rate limiting, and specialized methods for data analysis.
"""

import json
from typing import Any
from uuid import uuid4

import anthropic
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from insightboost.api.rate_limiter import RateLimiter
from insightboost.config.logging_config import get_logger
from insightboost.config.settings import Settings, get_settings
from insightboost.models.insight import (
    DatasetAnalysis,
    DatasetSummary,
    Insight,
    InsightType,
    Pattern,
    PatternType,
    QualityMetrics,
)
from insightboost.models.visualization import (
    ChartType,
    VisualizationConfig,
    VisualizationSuggestion,
)
from insightboost.utils.exceptions import APIError, RateLimitError
from insightboost.utils.formatters import (
    format_columns_with_types,
    format_correlation_matrix,
    format_dataframe_for_llm,
)

logger = get_logger("anthropic_client")


# Prompt templates
DATASET_ANALYSIS_PROMPT = """You are an expert data analyst. Analyze the following dataset metadata and sample:

<dataset_info>
- Columns:
{columns_with_types}
- Shape: {rows} rows × {cols} columns
- Missing values summary: {missing_summary}
- Numeric statistics: {numeric_stats}
</dataset_info>

<sample_data>
{sample_rows}
</sample_data>

<analysis_context>
{user_context}
</analysis_context>

Provide a comprehensive analysis including:
1. Data quality assessment (completeness, consistency, validity)
2. Key patterns and correlations identified
3. Potential anomalies or outliers
4. Recommended next steps for deeper analysis

Format your response as JSON matching this schema:
{{
    "quality_score": 0.0-1.0,
    "quality_assessment": {{
        "completeness": 0.0-1.0,
        "consistency": 0.0-1.0,
        "validity": 0.0-1.0,
        "issues": ["list of data quality issues"]
    }},
    "patterns": [
        {{
            "type": "correlation|trend|distribution|anomaly",
            "description": "pattern description",
            "columns": ["involved columns"],
            "strength": 0.0-1.0
        }}
    ],
    "insights": [
        {{
            "title": "short title",
            "description": "detailed finding",
            "type": "trend|correlation|anomaly|distribution|comparison",
            "confidence": 0.0-1.0,
            "columns": ["involved columns"]
        }}
    ],
    "recommendations": ["list of recommended next steps"]
}}"""


VISUALIZATION_SUGGESTION_PROMPT = """You are a data visualization expert. Based on the dataset characteristics below, recommend optimal visualizations.

<dataset_profile>
- Column types:
{column_types}
- Cardinality per column: {cardinality}
- Correlation matrix (significant correlations): {correlations}
- Row count: {row_count}
</dataset_profile>

<analysis_goals>
{user_goals}
</analysis_goals>

For each recommended visualization:
1. Specify the chart type (scatter, bar, heatmap, line, box, histogram, pie, area, violin)
2. Identify the columns to use for each axis/dimension
3. Explain WHY this visualization is appropriate for this data
4. Rate confidence (0.0-1.0) in this recommendation

Return exactly 3-5 visualization recommendations as a JSON array:
[
    {{
        "chart_type": "scatter|line|bar|histogram|box|violin|heatmap|pie|area",
        "title": "Descriptive title",
        "description": "What this visualization shows",
        "x_column": "column name or null",
        "y_column": "column name or null",
        "color_column": "optional column for color",
        "reasoning": "Why this visualization is appropriate",
        "confidence": 0.0-1.0
    }}
]"""


INSIGHT_GENERATION_PROMPT = """You are an expert data analyst. Analyze the following dataset to answer the user's question.

<dataset_info>
{dataset_info}
</dataset_info>

<sample_data>
{sample_data}
</sample_data>

<user_question>
{query}
</user_question>

Generate up to {max_insights} insights that directly address the question. Each insight should:
1. Be data-driven and specific
2. Include supporting evidence from the data
3. Have a confidence score based on data support

Return insights as a JSON array:
[
    {{
        "title": "concise insight title",
        "description": "detailed explanation with specific data points",
        "type": "trend|correlation|anomaly|distribution|comparison|summary",
        "confidence": 0.0-1.0,
        "columns_involved": ["relevant columns"],
        "supporting_data": {{
            "key_values": {{}},
            "evidence": "specific data points supporting this insight"
        }}
    }}
]"""


class TokenUsageTracker:
    """Tracks API token usage and estimates costs."""

    # Approximate pricing per 1M tokens (as of 2024)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self) -> None:
        """Initialize tracker."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = 0
        self.model_usage: dict[str, dict[str, int]] = {}

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Track token usage for a request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.requests += 1

        if model not in self.model_usage:
            self.model_usage[model] = {"input": 0, "output": 0, "requests": 0}

        self.model_usage[model]["input"] += input_tokens
        self.model_usage[model]["output"] += output_tokens
        self.model_usage[model]["requests"] += 1

    def estimate_cost(self, model: str | None = None) -> float:
        """Estimate cost in USD."""
        total_cost = 0.0

        models_to_check = [model] if model else self.model_usage.keys()

        for m in models_to_check:
            if m not in self.model_usage or m not in self.PRICING:
                continue

            usage = self.model_usage[m]
            pricing = self.PRICING[m]

            input_cost = (usage["input"] / 1_000_000) * pricing["input"]
            output_cost = (usage["output"] / 1_000_000) * pricing["output"]
            total_cost += input_cost + output_cost

        return round(total_cost, 4)

    def get_summary(self) -> dict[str, Any]:
        """Get usage summary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.requests,
            "estimated_cost_usd": self.estimate_cost(),
            "by_model": self.model_usage,
        }


class AnthropicClient:
    """
    Robust Anthropic API client with retry logic and rate limiting.

    This client provides specialized methods for data analysis tasks
    and handles API errors gracefully.

    Attributes:
        client: Underlying Anthropic client
        model: Default model to use
        max_tokens: Default max tokens for responses
        rate_limiter: Rate limiter for API calls
        token_tracker: Token usage tracker
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the Anthropic client.

        Args:
            api_key: API key (defaults to settings)
            model: Model to use (defaults to settings)
            max_tokens: Max tokens (defaults to settings)
            settings: Settings instance
        """
        self.settings = settings or get_settings()

        self.api_key = api_key or self.settings.anthropic_api_key
        self.model = model or self.settings.anthropic_model
        self.max_tokens = max_tokens or self.settings.anthropic_max_tokens

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.settings.rate_limit_requests_per_minute,
        )

        # Initialize token tracker
        self.token_tracker = TokenUsageTracker()

        logger.info(f"AnthropicClient initialized with model: {self.model}")

    @retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def _make_request(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> anthropic.types.Message:
        """
        Make a request to the Anthropic API with retry logic.

        Args:
            messages: List of message dicts
            system: System prompt
            max_tokens: Max tokens for response
            temperature: Sampling temperature

        Returns:
            API response message

        Raises:
            APIError: If API call fails
        """
        # Apply rate limiting
        self.rate_limiter.acquire(blocking=True, timeout=60.0)

        try:
            logger.debug(f"Making API request with {len(messages)} messages")

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens or self.max_tokens,
                "messages": messages,
                "temperature": temperature,
            }

            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)

            # Track token usage
            self.token_tracker.track(
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            logger.debug(
                f"API response: {response.usage.input_tokens} input, "
                f"{response.usage.output_tokens} output tokens"
            )

            return response

        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limited by Anthropic API: {e}")
            raise RateLimitError(
                message="Anthropic API rate limit exceeded",
                retry_after=60,
            ) from e
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise APIError(
                message=f"Anthropic API error: {str(e)}",
                status_code=getattr(e, "status_code", None),
            ) from e

    def _extract_json(self, text: str) -> dict | list:
        """Extract JSON from response text."""
        # Try to find JSON in the response
        text = text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Find JSON object or array
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            if start_char in text:
                start = text.find(start_char)
                # Find matching end
                depth = 0
                for i, char in enumerate(text[start:], start):
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            text = text[start : i + 1]
                            break
                break

        return json.loads(text)

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        context: str = "",
        depth: str = "standard",
    ) -> DatasetAnalysis:
        """
        Perform comprehensive dataset analysis.

        Args:
            df: DataFrame to analyze
            context: Optional business context
            depth: Analysis depth ('quick', 'standard', 'deep')

        Returns:
            DatasetAnalysis with findings
        """
        logger.info(f"Analyzing dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # Prepare dataset info
        df_info = format_dataframe_for_llm(df, max_rows=20 if depth == "deep" else 10)

        # Build prompt
        prompt = DATASET_ANALYSIS_PROMPT.format(
            columns_with_types=format_columns_with_types(df),
            rows=len(df),
            cols=len(df.columns),
            missing_summary=json.dumps(df_info["missing_summary"], indent=2),
            numeric_stats=json.dumps(df_info.get("numeric_stats", {}), indent=2)[:2000],
            sample_rows=df_info["sample_data"][:3000],
            user_context=context or "No specific context provided.",
        )

        # Make API request
        response = self._make_request(
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert data analyst. Always respond with valid JSON.",
            max_tokens=4096 if depth == "deep" else 2048,
        )

        # Parse response
        try:
            result = self._extract_json(response.content[0].text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            raise APIError(message="Failed to parse analysis response") from e

        # Build DatasetAnalysis
        dataset_id = uuid4()

        # Parse quality metrics
        qa = result.get("quality_assessment", {})
        quality_metrics = QualityMetrics(
            completeness=qa.get("completeness", 0.8),
            uniqueness=0.8,  # Would need to calculate
            consistency=qa.get("consistency", 0.8),
            validity=qa.get("validity", 0.8),
        )

        # Parse patterns
        patterns = []
        for p in result.get("patterns", []):
            try:
                patterns.append(
                    Pattern(
                        pattern_type=PatternType(p.get("type", "distribution")),
                        description=p.get("description", ""),
                        columns_involved=p.get("columns", []),
                        strength=float(p.get("strength", 0.5)),
                    )
                )
            except (ValueError, KeyError):
                continue

        # Parse insights
        insights = []
        for i in result.get("insights", []):
            try:
                insights.append(
                    Insight(
                        title=i.get("title", "Insight"),
                        description=i.get("description", ""),
                        insight_type=InsightType(i.get("type", "summary")),
                        confidence=float(i.get("confidence", 0.5)),
                        columns_involved=i.get("columns", []),
                        supporting_data=i.get("supporting_data", {}),
                    )
                )
            except (ValueError, KeyError):
                continue

        # Build summary
        summary = DatasetSummary(
            row_count=len(df),
            column_count=len(df.columns),
            numeric_columns=len(df.select_dtypes(include=["number"]).columns),
            categorical_columns=len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            datetime_columns=len(df.select_dtypes(include=["datetime"]).columns),
            text_columns=0,
            memory_usage_bytes=df.memory_usage(deep=True).sum(),
            has_missing_values=df.isnull().any().any(),
            missing_value_count=int(df.isnull().sum().sum()),
        )

        return DatasetAnalysis(
            dataset_id=dataset_id,
            summary=summary,
            quality_score=float(result.get("quality_score", 0.7)),
            quality_metrics=quality_metrics,
            patterns=patterns,
            insights=insights,
            recommendations=result.get("recommendations", []),
            analysis_depth=depth,
            token_usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )

    def generate_insights(
        self,
        df: pd.DataFrame,
        query: str,
        max_insights: int = 5,
    ) -> list[Insight]:
        """
        Generate insights based on a natural language query.

        Args:
            df: DataFrame to analyze
            query: Natural language question
            max_insights: Maximum insights to return

        Returns:
            List of Insight objects
        """
        logger.info(f"Generating insights for query: {query[:50]}...")

        # Prepare dataset info
        df_info = format_dataframe_for_llm(df, max_rows=15)

        # Build prompt
        prompt = INSIGHT_GENERATION_PROMPT.format(
            dataset_info=json.dumps(
                {
                    "columns": df_info["columns"],
                    "rows": df_info["rows"],
                    "cols": df_info["cols"],
                },
                indent=2,
            ),
            sample_data=df_info["sample_data"][:2000],
            query=query,
            max_insights=max_insights,
        )

        # Make API request
        response = self._make_request(
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert data analyst. Always respond with valid JSON array.",
            max_tokens=2048,
        )

        # Parse response
        try:
            result = self._extract_json(response.content[0].text)
            if not isinstance(result, list):
                result = [result]
        except json.JSONDecodeError:
            logger.error("Failed to parse insights response")
            return []

        # Build Insight objects
        insights = []
        for i in result[:max_insights]:
            try:
                insights.append(
                    Insight(
                        title=i.get("title", "Insight"),
                        description=i.get("description", ""),
                        insight_type=InsightType(i.get("type", "summary")),
                        confidence=float(i.get("confidence", 0.5)),
                        columns_involved=i.get("columns_involved", []),
                        supporting_data=i.get("supporting_data", {}),
                        query=query,
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse insight: {e}")
                continue

        return insights

    def suggest_visualizations(
        self,
        df: pd.DataFrame,
        objective: str = "",
        max_suggestions: int = 5,
    ) -> list[VisualizationSuggestion]:
        """
        Suggest visualizations based on data characteristics.

        Args:
            df: DataFrame to visualize
            objective: Analysis objective
            max_suggestions: Maximum suggestions to return

        Returns:
            List of VisualizationSuggestion objects
        """
        logger.info("Generating visualization suggestions")

        # Analyze data characteristics
        column_types = {}
        cardinality = {}

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                column_types[col] = "numeric"
            elif df[col].dtype == "object":
                column_types[col] = "categorical"
            elif "datetime" in str(df[col].dtype):
                column_types[col] = "datetime"
            else:
                column_types[col] = str(df[col].dtype)

            cardinality[col] = int(df[col].nunique())

        correlations = format_correlation_matrix(df, threshold=0.3)

        # Build prompt
        prompt = VISUALIZATION_SUGGESTION_PROMPT.format(
            column_types=json.dumps(column_types, indent=2),
            cardinality=json.dumps(cardinality, indent=2),
            correlations=json.dumps(correlations.get("correlations", []), indent=2),
            row_count=len(df),
            user_goals=objective or "Explore and understand the data patterns.",
        )

        # Make API request
        response = self._make_request(
            messages=[{"role": "user", "content": prompt}],
            system="You are a data visualization expert. Always respond with valid JSON array.",
            max_tokens=2048,
        )

        # Parse response
        try:
            result = self._extract_json(response.content[0].text)
            if not isinstance(result, list):
                result = [result]
        except json.JSONDecodeError:
            logger.error("Failed to parse visualization suggestions")
            return []

        # Build VisualizationSuggestion objects
        suggestions = []
        for idx, v in enumerate(result[:max_suggestions]):
            try:
                chart_type = ChartType(v.get("chart_type", "bar"))

                config = VisualizationConfig(
                    chart_type=chart_type,
                    x_column=v.get("x_column"),
                    y_column=v.get("y_column"),
                    color_column=v.get("color_column"),
                    title=v.get("title", f"Visualization {idx + 1}"),
                )

                suggestions.append(
                    VisualizationSuggestion(
                        chart_type=chart_type,
                        title=v.get("title", f"Visualization {idx + 1}"),
                        description=v.get("description", ""),
                        config=config,
                        reasoning=v.get("reasoning", ""),
                        confidence=float(v.get("confidence", 0.5)),
                        priority=max_suggestions - idx,
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse visualization suggestion: {e}")
                continue

        return suggestions

    def explain_pattern(
        self,
        pattern: str,
        context: dict[str, Any],
    ) -> str:
        """
        Generate a human-readable explanation of a pattern.

        Args:
            pattern: Pattern description
            context: Additional context

        Returns:
            Explanation string
        """
        prompt = f"""Explain the following data pattern in clear, business-friendly terms:

Pattern: {pattern}

Context: {json.dumps(context, indent=2)}

Provide a concise explanation that a non-technical stakeholder would understand.
Include potential business implications and recommended actions."""

        response = self._make_request(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        return response.content[0].text

    def get_token_usage(self) -> dict[str, Any]:
        """Get token usage summary."""
        return self.token_tracker.get_summary()
