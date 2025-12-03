"""
Insight and analysis data models.

This module defines Pydantic models for insights, patterns, and dataset analysis.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class InsightType(str, Enum):
    """Types of insights that can be generated."""

    TREND = "trend"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    PATTERN = "pattern"
    SUMMARY = "summary"


class PatternType(str, Enum):
    """Types of patterns that can be detected."""

    CORRELATION = "correlation"
    TREND = "trend"
    SEASONALITY = "seasonality"
    CLUSTER = "cluster"
    ANOMALY = "anomaly"
    DISTRIBUTION = "distribution"


class Pattern(BaseModel):
    """
    A data pattern identified during analysis.

    Attributes:
        id: Unique identifier for the pattern
        pattern_type: Type of pattern detected
        description: Human-readable description
        columns_involved: Columns related to this pattern
        strength: Pattern strength score (0-1)
        details: Additional pattern-specific details
    """

    id: UUID = Field(default_factory=uuid4)
    pattern_type: PatternType
    description: str
    columns_involved: list[str] = Field(default_factory=list)
    strength: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strength")
    @classmethod
    def round_strength(cls, v: float) -> float:
        """Round strength to 4 decimal places."""
        return round(v, 4)


class Insight(BaseModel):
    """
    An AI-generated insight from data analysis.

    Attributes:
        id: Unique identifier for the insight
        title: Short title for the insight
        description: Detailed description of the finding
        insight_type: Category of insight
        confidence: Confidence score (0-1)
        supporting_data: Data supporting this insight
        columns_involved: Columns analyzed
        created_at: When the insight was generated
        query: Original query that generated this insight (if any)
    """

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    insight_type: InsightType
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_data: dict[str, Any] = Field(default_factory=dict)
    columns_involved: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    query: str | None = None

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 4 decimal places."""
        return round(v, 4)

    @property
    def confidence_label(self) -> str:
        """Get human-readable confidence label."""
        if self.confidence >= 0.8:
            return "High"
        elif self.confidence >= 0.5:
            return "Medium"
        else:
            return "Low"

    def to_display_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for display."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "type": self.insight_type.value,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "columns": self.columns_involved,
            "created_at": self.created_at.isoformat(),
        }


class QualityMetrics(BaseModel):
    """
    Data quality metrics for a dataset.

    Attributes:
        completeness: Percentage of non-null values
        uniqueness: Ratio of unique values to total
        consistency: Measure of data format consistency
        validity: Percentage of valid values
        timeliness: Freshness of data (if applicable)
    """

    completeness: float = Field(ge=0.0, le=1.0)
    uniqueness: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    validity: float = Field(ge=0.0, le=1.0)
    timeliness: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = [
            self.completeness,
            self.uniqueness,
            self.consistency,
            self.validity,
        ]
        if self.timeliness is not None:
            scores.append(self.timeliness)
        return round(sum(scores) / len(scores), 4)


class DatasetSummary(BaseModel):
    """
    Summary statistics for a dataset.

    Attributes:
        row_count: Number of rows
        column_count: Number of columns
        numeric_columns: Count of numeric columns
        categorical_columns: Count of categorical columns
        datetime_columns: Count of datetime columns
        text_columns: Count of text columns
        memory_usage_bytes: Memory usage in bytes
        has_missing_values: Whether dataset has missing values
        missing_value_count: Total missing values
    """

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    numeric_columns: int = Field(ge=0)
    categorical_columns: int = Field(ge=0)
    datetime_columns: int = Field(ge=0)
    text_columns: int = Field(ge=0)
    memory_usage_bytes: int = Field(ge=0)
    has_missing_values: bool
    missing_value_count: int = Field(ge=0)

    @property
    def missing_value_percentage(self) -> float:
        """Calculate percentage of missing values."""
        total_cells = self.row_count * self.column_count
        if total_cells == 0:
            return 0.0
        return round(self.missing_value_count / total_cells * 100, 2)


class DatasetAnalysis(BaseModel):
    """
    Complete analysis result for a dataset.

    Attributes:
        dataset_id: ID of the analyzed dataset
        summary: Dataset summary statistics
        quality_score: Overall data quality score
        quality_metrics: Detailed quality metrics
        patterns: Detected patterns
        insights: Generated insights
        recommendations: Suggested next steps
        analyzed_at: When the analysis was performed
        analysis_depth: Depth of analysis performed
        token_usage: API tokens used for analysis
    """

    dataset_id: UUID
    summary: DatasetSummary
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_metrics: QualityMetrics
    patterns: list[Pattern] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    analysis_depth: str = "standard"
    token_usage: dict[str, int] = Field(default_factory=dict)

    @property
    def pattern_count(self) -> int:
        """Get total pattern count."""
        return len(self.patterns)

    @property
    def insight_count(self) -> int:
        """Get total insight count."""
        return len(self.insights)

    def get_high_confidence_insights(
        self,
        threshold: float = 0.8,
    ) -> list[Insight]:
        """Get insights above confidence threshold."""
        return [i for i in self.insights if i.confidence >= threshold]

    def to_report_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for reporting."""
        return {
            "dataset_id": str(self.dataset_id),
            "analyzed_at": self.analyzed_at.isoformat(),
            "summary": self.summary.model_dump(),
            "quality_score": self.quality_score,
            "pattern_count": self.pattern_count,
            "insight_count": self.insight_count,
            "recommendations": self.recommendations,
        }


class InsightFeedback(BaseModel):
    """
    User feedback on an insight.

    Attributes:
        insight_id: ID of the insight
        user_id: ID of the user providing feedback
        is_helpful: Whether the insight was helpful
        comment: Optional feedback comment
        created_at: When feedback was provided
    """

    insight_id: UUID
    user_id: UUID
    is_helpful: bool
    comment: str | None = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)
