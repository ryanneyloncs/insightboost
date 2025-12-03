"""
Visualization data models.

This module defines Pydantic models for visualization suggestions and configurations.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ChartType(str, Enum):
    """Supported chart types."""

    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"
    BUBBLE = "bubble"
    FUNNEL = "funnel"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    PARALLEL_COORDINATES = "parallel_coordinates"
    SCATTER_MATRIX = "scatter_matrix"
    DENSITY = "density"
    CONTOUR = "contour"


class VisualizationLibrary(str, Enum):
    """Supported visualization libraries."""

    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    SEABORN = "seaborn"


class VisualizationConfig(BaseModel):
    """
    Configuration for generating a visualization.

    Attributes:
        chart_type: Type of chart to generate
        x_column: Column for X axis
        y_column: Column for Y axis
        color_column: Column for color encoding
        size_column: Column for size encoding
        facet_column: Column for faceting
        hover_columns: Columns to show in hover tooltip
        title: Chart title
        x_label: X axis label
        y_label: Y axis label
        color_scale: Color scale to use
        theme: Visual theme
        width: Chart width in pixels
        height: Chart height in pixels
        additional_options: Library-specific options
    """

    chart_type: ChartType
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    size_column: str | None = None
    facet_column: str | None = None
    hover_columns: list[str] = Field(default_factory=list)
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    color_scale: str = "viridis"
    theme: str = "plotly_white"
    width: int = Field(default=800, ge=200, le=2000)
    height: int = Field(default=600, ge=200, le=2000)
    additional_options: dict[str, Any] = Field(default_factory=dict)

    def get_column_mappings(self) -> dict[str, str | None]:
        """Get all column mappings as dictionary."""
        return {
            "x": self.x_column,
            "y": self.y_column,
            "color": self.color_column,
            "size": self.size_column,
            "facet": self.facet_column,
        }

    def get_used_columns(self) -> list[str]:
        """Get list of all columns used in this configuration."""
        columns = []
        for col in [
            self.x_column,
            self.y_column,
            self.color_column,
            self.size_column,
            self.facet_column,
        ]:
            if col:
                columns.append(col)
        columns.extend(self.hover_columns)
        return list(set(columns))


class VisualizationSuggestion(BaseModel):
    """
    AI-suggested visualization for a dataset.

    Attributes:
        id: Unique identifier
        chart_type: Recommended chart type
        title: Suggested chart title
        description: Why this visualization is recommended
        config: Complete visualization configuration
        reasoning: Detailed reasoning from the AI
        confidence: Confidence score (0-1)
        code_snippet: Ready-to-use code to generate this visualization
        library: Which library the code uses
        priority: Suggestion priority (higher = more important)
        created_at: When the suggestion was generated
    """

    id: UUID = Field(default_factory=uuid4)
    chart_type: ChartType
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    config: VisualizationConfig
    reasoning: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    code_snippet: str = Field(default="")
    library: VisualizationLibrary = VisualizationLibrary.PLOTLY
    priority: int = Field(default=0, ge=0, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 4 decimal places."""
        return round(v, 4)

    @property
    def x_column(self) -> str | None:
        """Get X column from config."""
        return self.config.x_column

    @property
    def y_column(self) -> str | None:
        """Get Y column from config."""
        return self.config.y_column

    @property
    def color_column(self) -> str | None:
        """Get color column from config."""
        return self.config.color_column

    @property
    def size_column(self) -> str | None:
        """Get size column from config."""
        return self.config.size_column

    def to_display_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for display."""
        return {
            "id": str(self.id),
            "chart_type": self.chart_type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "color_column": self.color_column,
            "reasoning": self.reasoning[:200] + "..."
            if len(self.reasoning) > 200
            else self.reasoning,
        }


class GeneratedVisualization(BaseModel):
    """
    A generated visualization.

    Attributes:
        id: Unique identifier
        suggestion_id: ID of the suggestion this was generated from
        dataset_id: ID of the dataset visualized
        config: Configuration used
        figure_json: Plotly figure as JSON
        image_base64: Optional static image as base64
        created_at: When visualization was generated
        render_time_ms: Time to render in milliseconds
    """

    id: UUID = Field(default_factory=uuid4)
    suggestion_id: UUID | None = None
    dataset_id: UUID
    config: VisualizationConfig
    figure_json: dict[str, Any] = Field(default_factory=dict)
    image_base64: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    render_time_ms: int = Field(default=0, ge=0)

    @property
    def has_interactive_figure(self) -> bool:
        """Check if interactive figure data is available."""
        return bool(self.figure_json)

    @property
    def has_static_image(self) -> bool:
        """Check if static image is available."""
        return bool(self.image_base64)


class VisualizationExport(BaseModel):
    """
    Export configuration for a visualization.

    Attributes:
        format: Export format (png, svg, pdf, html)
        width: Export width in pixels
        height: Export height in pixels
        scale: Scale factor for raster exports
        include_plotlyjs: Include Plotly.js for HTML exports
    """

    format: str = Field(default="png", pattern="^(png|svg|pdf|html|json)$")
    width: int = Field(default=1200, ge=100, le=4000)
    height: int = Field(default=800, ge=100, le=4000)
    scale: float = Field(default=2.0, ge=0.5, le=5.0)
    include_plotlyjs: bool = True


# Mapping of chart types to suitable data characteristics
CHART_TYPE_REQUIREMENTS = {
    ChartType.SCATTER: {
        "min_numeric": 2,
        "suitable_for": ["correlations", "distributions", "outliers"],
    },
    ChartType.LINE: {
        "min_numeric": 1,
        "requires_ordered": True,
        "suitable_for": ["trends", "time_series"],
    },
    ChartType.BAR: {
        "min_categorical": 1,
        "suitable_for": ["comparisons", "rankings"],
    },
    ChartType.HISTOGRAM: {
        "min_numeric": 1,
        "suitable_for": ["distributions"],
    },
    ChartType.BOX: {
        "min_numeric": 1,
        "suitable_for": ["distributions", "comparisons", "outliers"],
    },
    ChartType.HEATMAP: {
        "min_numeric": 3,
        "suitable_for": ["correlations", "patterns"],
    },
    ChartType.PIE: {
        "min_categorical": 1,
        "max_categories": 10,
        "suitable_for": ["proportions", "compositions"],
    },
}
