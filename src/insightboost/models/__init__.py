"""Data models for InsightBoost."""

from insightboost.models.dataset import (
    ColumnInfo,
    Dataset,
    DatasetMetadata,
)
from insightboost.models.insight import (
    DatasetAnalysis,
    DatasetSummary,
    Insight,
    Pattern,
    QualityMetrics,
)
from insightboost.models.user import (
    CollaborationSession,
    User,
    UserRole,
)
from insightboost.models.visualization import (
    ChartType,
    VisualizationConfig,
    VisualizationSuggestion,
)

__all__ = [
    # Insight models
    "Insight",
    "DatasetAnalysis",
    "DatasetSummary",
    "Pattern",
    "QualityMetrics",
    # Visualization models
    "VisualizationSuggestion",
    "VisualizationConfig",
    "ChartType",
    # Dataset models
    "Dataset",
    "ColumnInfo",
    "DatasetMetadata",
    # User models
    "User",
    "UserRole",
    "CollaborationSession",
]
