"""
Visualization API routes for InsightBoost.

This module provides API endpoints for visualization suggestions and generation.
"""

import json
import uuid
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from insightboost.api.anthropic_client import AnthropicClient
from insightboost.config.logging_config import get_logger
from insightboost.core.visualization_suggester import VisualizationSuggester
from insightboost.models.visualization import (ChartType, VisualizationConfig,
                                               VisualizationSuggestion)
from insightboost.utils.exceptions import VisualizationError
from insightboost.utils.validators import validate_chart_type

logger = get_logger("routes.visualizations")
visualizations_bp = Blueprint("visualizations", __name__)

# In-memory storage for visualizations (replace with database in production)
_visualizations: dict[str, dict[str, Any]] = {}
_suggestions_cache: dict[str, list[dict[str, Any]]] = {}

# Lazy-loaded components
_api_client: AnthropicClient | None = None
_viz_suggester: VisualizationSuggester | None = None


def get_api_client() -> AnthropicClient:
    """Get or create the API client."""
    global _api_client
    if _api_client is None:
        _api_client = AnthropicClient()
    return _api_client


def get_viz_suggester() -> VisualizationSuggester:
    """Get or create the visualization suggester."""
    global _viz_suggester
    if _viz_suggester is None:
        _viz_suggester = VisualizationSuggester(api_client=get_api_client())
    return _viz_suggester


def get_datasets_storage() -> dict:
    """Get access to the datasets storage from insights module."""
    from insightboost.web.routes.insights import _datasets, load_dataframe

    return _datasets, load_dataframe


# =============================================================================
# Visualization Suggestion Endpoints
# =============================================================================


@visualizations_bp.route(
    "/datasets/<dataset_id>/visualizations/suggest", methods=["GET"]
)
def suggest_visualizations(dataset_id: str):
    """
    Get AI-powered visualization suggestions for a dataset.

    Query params:
        objective: Analysis objective (optional)
        max_suggestions: Maximum suggestions (default: 5)
        use_cache: Whether to use cached suggestions (default: true)
    """
    _datasets, load_dataframe = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    try:
        # Get query parameters
        objective = request.args.get("objective", "")
        max_suggestions = request.args.get("max_suggestions", 5, type=int)
        use_cache = request.args.get("use_cache", "true").lower() == "true"

        max_suggestions = min(max(1, max_suggestions), 10)

        # Check cache
        cache_key = f"{dataset_id}:{objective}:{max_suggestions}"
        if use_cache and cache_key in _suggestions_cache:
            logger.debug(f"Returning cached suggestions for {dataset_id}")
            return jsonify(
                {
                    "success": True,
                    "dataset_id": dataset_id,
                    "suggestions": _suggestions_cache[cache_key],
                    "count": len(_suggestions_cache[cache_key]),
                    "cached": True,
                }
            )

        # Load dataset
        df = load_dataframe(dataset_id)

        logger.info(f"Generating visualization suggestions for {dataset_id}")

        # Get suggestions
        suggester = get_viz_suggester()
        suggestions = suggester.suggest_visualizations(
            df=df,
            objective=objective,
            max_suggestions=max_suggestions,
        )

        # Format suggestions
        results = []
        for s in suggestions:
            suggestion_dict = {
                "id": str(s.id),
                "chart_type": s.chart_type.value,
                "title": s.title,
                "description": s.description,
                "x_column": s.x_column,
                "y_column": s.y_column,
                "color_column": s.color_column,
                "size_column": s.size_column,
                "reasoning": s.reasoning,
                "confidence": s.confidence,
                "code_snippet": s.code_snippet,
                "priority": s.priority,
                "config": {
                    "chart_type": s.config.chart_type.value,
                    "x_column": s.config.x_column,
                    "y_column": s.config.y_column,
                    "color_column": s.config.color_column,
                    "size_column": s.config.size_column,
                    "title": s.config.title,
                    "width": s.config.width,
                    "height": s.config.height,
                },
            }
            results.append(suggestion_dict)

        # Cache results
        _suggestions_cache[cache_key] = results

        logger.info(f"Generated {len(results)} visualization suggestions")

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "objective": objective,
                "suggestions": results,
                "count": len(results),
                "cached": False,
            }
        )

    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}")
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "SUGGESTION_FAILED",
                    "message": str(e),
                }
            ),
            500,
        )


@visualizations_bp.route("/datasets/<dataset_id>/visualizations/auto", methods=["GET"])
def auto_visualize(dataset_id: str):
    """
    Generate a complete visualization suite for a dataset.

    Query params:
        columns: Comma-separated list of columns to focus on (optional)
    """
    _datasets, load_dataframe = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    try:
        # Get columns parameter
        columns_param = request.args.get("columns", "")
        columns = (
            [c.strip() for c in columns_param.split(",") if c.strip()]
            if columns_param
            else None
        )

        # Load dataset
        df = load_dataframe(dataset_id)

        logger.info(f"Auto-generating visualizations for {dataset_id}")

        # Generate visualizations
        suggester = get_viz_suggester()
        figures = suggester.auto_visualize(df, columns=columns)

        # Convert figures to JSON
        results = {}
        for name, fig in figures.items():
            viz_id = str(uuid.uuid4())
            results[name] = {
                "id": viz_id,
                "name": name,
                "figure_json": json.loads(fig.to_json()),
            }

            # Store visualization
            _visualizations[viz_id] = {
                "id": viz_id,
                "dataset_id": dataset_id,
                "name": name,
                "figure_json": json.loads(fig.to_json()),
                "created_at": datetime.utcnow().isoformat(),
            }

        logger.info(f"Generated {len(results)} auto-visualizations")

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "visualizations": results,
                "count": len(results),
            }
        )

    except Exception as e:
        logger.error(f"Auto-visualization failed: {e}")
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "AUTO_VIZ_FAILED",
                    "message": str(e),
                }
            ),
            500,
        )


# =============================================================================
# Visualization Generation Endpoints
# =============================================================================


@visualizations_bp.route("/datasets/<dataset_id>/visualizations", methods=["POST"])
def create_visualization(dataset_id: str):
    """
    Generate a specific visualization.

    Request body:
        {
            "suggestion_id": "uuid",  // Use a previous suggestion
            // OR
            "config": {
                "chart_type": "scatter",
                "x_column": "col1",
                "y_column": "col2",
                ...
            }
        }
    """
    _datasets, load_dataframe = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    try:
        data = request.get_json() or {}

        # Load dataset
        df = load_dataframe(dataset_id)
        suggester = get_viz_suggester()

        suggestion = None
        config = None

        # Option 1: Use a suggestion ID
        if "suggestion_id" in data:
            suggestion_id = data["suggestion_id"]

            # Find suggestion in cache
            for cache_suggestions in _suggestions_cache.values():
                for s in cache_suggestions:
                    if s["id"] == suggestion_id:
                        # Reconstruct suggestion object
                        config = VisualizationConfig(
                            chart_type=ChartType(s["config"]["chart_type"]),
                            x_column=s["config"].get("x_column"),
                            y_column=s["config"].get("y_column"),
                            color_column=s["config"].get("color_column"),
                            size_column=s["config"].get("size_column"),
                            title=s["config"].get("title"),
                            width=s["config"].get("width", 800),
                            height=s["config"].get("height", 600),
                        )
                        suggestion = VisualizationSuggestion(
                            id=suggestion_id,
                            chart_type=ChartType(s["chart_type"]),
                            title=s["title"],
                            description=s["description"],
                            config=config,
                            reasoning=s["reasoning"],
                            confidence=s["confidence"],
                        )
                        break
                if suggestion:
                    break

            if not suggestion:
                return (
                    jsonify(
                        {
                            "error": True,
                            "error_code": "SUGGESTION_NOT_FOUND",
                            "message": "Suggestion not found",
                        }
                    ),
                    404,
                )

        # Option 2: Use custom config
        elif "config" in data:
            config_data = data["config"]

            # Validate chart type
            chart_type_str = validate_chart_type(
                config_data.get("chart_type", "scatter")
            )

            config = VisualizationConfig(
                chart_type=ChartType(chart_type_str),
                x_column=config_data.get("x_column"),
                y_column=config_data.get("y_column"),
                color_column=config_data.get("color_column"),
                size_column=config_data.get("size_column"),
                facet_column=config_data.get("facet_column"),
                hover_columns=config_data.get("hover_columns", []),
                title=config_data.get("title"),
                x_label=config_data.get("x_label"),
                y_label=config_data.get("y_label"),
                color_scale=config_data.get("color_scale", "viridis"),
                theme=config_data.get("theme", "plotly_white"),
                width=config_data.get("width", 800),
                height=config_data.get("height", 600),
            )

            suggestion = VisualizationSuggestion(
                chart_type=config.chart_type,
                title=config.title or f"{config.chart_type.value} Chart",
                description="Custom visualization",
                config=config,
                reasoning="User-defined configuration",
                confidence=1.0,
            )

        else:
            return (
                jsonify(
                    {
                        "error": True,
                        "error_code": "INVALID_REQUEST",
                        "message": "Either suggestion_id or config is required",
                    }
                ),
                400,
            )

        logger.info(f"Generating {suggestion.chart_type.value} visualization")

        # Generate the visualization
        fig = suggester.generate_visualization(
            df=df,
            suggestion=suggestion,
            interactive=True,
        )

        # Store visualization
        viz_id = str(uuid.uuid4())
        visualization = {
            "id": viz_id,
            "dataset_id": dataset_id,
            "suggestion_id": str(suggestion.id) if suggestion else None,
            "chart_type": suggestion.chart_type.value,
            "title": suggestion.title,
            "config": {
                "chart_type": config.chart_type.value,
                "x_column": config.x_column,
                "y_column": config.y_column,
                "color_column": config.color_column,
                "size_column": config.size_column,
                "title": config.title,
                "width": config.width,
                "height": config.height,
            },
            "figure_json": json.loads(fig.to_json()),
            "created_at": datetime.utcnow().isoformat(),
        }

        _visualizations[viz_id] = visualization

        logger.info(f"Visualization created: {viz_id}")

        return (
            jsonify(
                {
                    "success": True,
                    "visualization": visualization,
                }
            ),
            201,
        )

    except VisualizationError as e:
        return jsonify(e.to_dict()), 400
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "GENERATION_FAILED",
                    "message": str(e),
                }
            ),
            500,
        )


@visualizations_bp.route("/datasets/<dataset_id>/visualizations", methods=["GET"])
def list_visualizations(dataset_id: str):
    """List all generated visualizations for a dataset."""
    _datasets, _ = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    # Filter visualizations by dataset
    dataset_vizs = [
        {
            "id": v["id"],
            "chart_type": v["chart_type"],
            "title": v["title"],
            "created_at": v["created_at"],
        }
        for v in _visualizations.values()
        if v["dataset_id"] == dataset_id
    ]

    return jsonify(
        {
            "success": True,
            "dataset_id": dataset_id,
            "visualizations": dataset_vizs,
            "count": len(dataset_vizs),
        }
    )


@visualizations_bp.route(
    "/datasets/<dataset_id>/visualizations/<viz_id>", methods=["GET"]
)
def get_visualization(dataset_id: str, viz_id: str):
    """Get a specific visualization with its Plotly JSON."""
    _datasets, _ = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    if viz_id not in _visualizations:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Visualization not found",
                }
            ),
            404,
        )

    visualization = _visualizations[viz_id]

    if visualization["dataset_id"] != dataset_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Visualization not found for this dataset",
                }
            ),
            404,
        )

    return jsonify(
        {
            "success": True,
            "visualization": visualization,
        }
    )


@visualizations_bp.route(
    "/datasets/<dataset_id>/visualizations/<viz_id>", methods=["DELETE"]
)
def delete_visualization(dataset_id: str, viz_id: str):
    """Delete a visualization."""
    _datasets, _ = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    if viz_id not in _visualizations:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Visualization not found",
                }
            ),
            404,
        )

    if _visualizations[viz_id]["dataset_id"] != dataset_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Visualization not found for this dataset",
                }
            ),
            404,
        )

    del _visualizations[viz_id]

    logger.info(f"Visualization deleted: {viz_id}")

    return jsonify(
        {
            "success": True,
            "message": "Visualization deleted",
        }
    )


# =============================================================================
# Export Endpoints
# =============================================================================


@visualizations_bp.route(
    "/datasets/<dataset_id>/visualizations/<viz_id>/export", methods=["GET"]
)
def export_visualization(dataset_id: str, viz_id: str):
    """
    Export a visualization to various formats.

    Query params:
        format: Export format (html, json, png, svg)
    """
    _datasets, _ = get_datasets_storage()

    if dataset_id not in _datasets:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Dataset not found",
                }
            ),
            404,
        )

    if viz_id not in _visualizations:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Visualization not found",
                }
            ),
            404,
        )

    visualization = _visualizations[viz_id]

    if visualization["dataset_id"] != dataset_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Visualization not found for this dataset",
                }
            ),
            404,
        )

    export_format = request.args.get("format", "json")

    try:
        import plotly.graph_objects as go

        # Reconstruct figure from JSON
        fig = go.Figure(visualization["figure_json"])

        if export_format == "json":
            return jsonify(
                {
                    "success": True,
                    "format": "json",
                    "data": visualization["figure_json"],
                }
            )

        elif export_format == "html":
            html_content = fig.to_html(include_plotlyjs="cdn")
            return jsonify(
                {
                    "success": True,
                    "format": "html",
                    "data": html_content,
                }
            )

        elif export_format in ("png", "svg"):
            # Note: This requires kaleido package
            try:
                image_bytes = fig.to_image(format=export_format, scale=2)
                import base64

                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                return jsonify(
                    {
                        "success": True,
                        "format": export_format,
                        "data": image_b64,
                        "encoding": "base64",
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": True,
                            "error_code": "EXPORT_FAILED",
                            "message": f"Image export requires kaleido package: {e}",
                        }
                    ),
                    500,
                )

        else:
            return (
                jsonify(
                    {
                        "error": True,
                        "error_code": "INVALID_FORMAT",
                        "message": f"Unsupported format: {export_format}. Use: json, html, png, svg",
                    }
                ),
                400,
            )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "EXPORT_FAILED",
                    "message": str(e),
                }
            ),
            500,
        )


# =============================================================================
# Chart Type Information Endpoints
# =============================================================================


@visualizations_bp.route("/visualizations/chart-types", methods=["GET"])
def list_chart_types():
    """List all supported chart types with descriptions."""
    chart_types = [
        {
            "type": "scatter",
            "name": "Scatter Plot",
            "description": "Shows relationship between two numeric variables",
            "best_for": ["correlations", "distributions", "outliers"],
            "required_columns": {"numeric": 2},
        },
        {
            "type": "line",
            "name": "Line Chart",
            "description": "Shows trends over time or ordered data",
            "best_for": ["trends", "time series"],
            "required_columns": {"numeric": 1, "ordered": 1},
        },
        {
            "type": "bar",
            "name": "Bar Chart",
            "description": "Compares values across categories",
            "best_for": ["comparisons", "rankings"],
            "required_columns": {"categorical": 1},
        },
        {
            "type": "histogram",
            "name": "Histogram",
            "description": "Shows distribution of a numeric variable",
            "best_for": ["distributions"],
            "required_columns": {"numeric": 1},
        },
        {
            "type": "box",
            "name": "Box Plot",
            "description": "Shows distribution with quartiles and outliers",
            "best_for": ["distributions", "comparisons", "outliers"],
            "required_columns": {"numeric": 1},
        },
        {
            "type": "violin",
            "name": "Violin Plot",
            "description": "Shows distribution shape with density",
            "best_for": ["distributions", "comparisons"],
            "required_columns": {"numeric": 1},
        },
        {
            "type": "heatmap",
            "name": "Heatmap",
            "description": "Shows patterns in matrix data",
            "best_for": ["correlations", "patterns"],
            "required_columns": {"numeric": 3},
        },
        {
            "type": "pie",
            "name": "Pie Chart",
            "description": "Shows proportions of a whole",
            "best_for": ["proportions", "compositions"],
            "required_columns": {"categorical": 1},
            "max_categories": 10,
        },
        {
            "type": "area",
            "name": "Area Chart",
            "description": "Shows cumulative totals over time",
            "best_for": ["trends", "compositions"],
            "required_columns": {"numeric": 1, "ordered": 1},
        },
        {
            "type": "bubble",
            "name": "Bubble Chart",
            "description": "Scatter plot with size encoding",
            "best_for": ["multi-variable relationships"],
            "required_columns": {"numeric": 3},
        },
    ]

    return jsonify(
        {
            "success": True,
            "chart_types": chart_types,
            "count": len(chart_types),
        }
    )
