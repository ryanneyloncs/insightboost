"""Route blueprints for InsightBoost API."""

from insightboost.web.routes.collaboration import collaboration_bp
from insightboost.web.routes.insights import insights_bp
from insightboost.web.routes.visualizations import visualizations_bp

__all__ = [
    "insights_bp",
    "visualizations_bp",
    "collaboration_bp",
]
