"""Data models for InsightBoost."""

from insightboost.models.insight import (
    Insight,
    DatasetAnalysis,
    DatasetSummary,
    Pattern,
    QualityMetrics,
)
from insightboost.models.visualization import (
    VisualizationSuggestion,
    VisualizationConfig,
    ChartType,
)
from insightboost.models.dataset import (
    Dataset,
    ColumnInfo,
    DatasetMetadata,
)
from insightboost.models.user import (
    User,
    UserRole,
    CollaborationSession,
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
