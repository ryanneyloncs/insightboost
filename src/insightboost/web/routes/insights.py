"""
Insights API routes for InsightBoost.

This module provides API endpoints for dataset management and insight generation.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from insightboost.api.anthropic_client import AnthropicClient
from insightboost.config.logging_config import get_logger
from insightboost.config.settings import get_settings
from insightboost.core.data_analyzer import DataAnalyzer
from insightboost.core.insights_generator import InsightsGenerator
from insightboost.utils.exceptions import (
    DatasetError,
    DataValidationError,
    InsightGenerationError,
)
from insightboost.utils.validators import (
    validate_analysis_depth,
    validate_file_upload,
    validate_query,
)

logger = get_logger("routes.insights")
insights_bp = Blueprint("insights", __name__)

# In-memory storage for demo (replace with database in production)
_datasets: dict[str, dict[str, Any]] = {}
_insights: dict[str, list[dict[str, Any]]] = {}

# Lazy-loaded components
_api_client: AnthropicClient | None = None
_insights_generator: InsightsGenerator | None = None
_data_analyzer: DataAnalyzer | None = None


def get_api_client() -> AnthropicClient:
    """Get or create the API client."""
    global _api_client
    if _api_client is None:
        _api_client = AnthropicClient()
    return _api_client


def get_insights_generator() -> InsightsGenerator:
    """Get or create the insights generator."""
    global _insights_generator
    if _insights_generator is None:
        _insights_generator = InsightsGenerator(api_client=get_api_client())
    return _insights_generator


def get_data_analyzer() -> DataAnalyzer:
    """Get or create the data analyzer."""
    global _data_analyzer
    if _data_analyzer is None:
        _data_analyzer = DataAnalyzer()
    return _data_analyzer


def get_upload_folder() -> Path:
    """Get the upload folder path, creating it if needed."""
    upload_folder = Path(current_app.root_path) / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    return upload_folder


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    """Load a DataFrame from storage."""
    if dataset_id not in _datasets:
        raise DatasetError(
            message="Dataset not found",
            dataset_id=dataset_id,
        )

    storage_path = _datasets[dataset_id]["storage_path"]
    file_format = _datasets[dataset_id]["metadata"]["file_format"]

    if file_format == ".csv":
        return pd.read_csv(storage_path)
    elif file_format in (".xlsx", ".xls"):
        return pd.read_excel(storage_path)
    elif file_format == ".json":
        return pd.read_json(storage_path)
    elif file_format == ".parquet":
        return pd.read_parquet(storage_path)
    else:
        raise DatasetError(
            message=f"Unsupported file format: {file_format}",
            dataset_id=dataset_id,
        )


# =============================================================================
# Dataset Endpoints
# =============================================================================


@insights_bp.route("/datasets", methods=["POST"])
def upload_dataset():
    """
    Upload a new dataset.

    Accepts CSV, Excel, JSON, or Parquet files.
    Returns dataset ID and initial analysis.
    """
    logger.info("Dataset upload request received")

    # Check if file is present
    if "file" not in request.files:
        return jsonify(
            {
                "error": True,
                "error_code": "NO_FILE",
                "message": "No file provided",
            }
        ), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify(
            {
                "error": True,
                "error_code": "NO_FILENAME",
                "message": "No file selected",
            }
        ), 400

    try:
        settings = get_settings()

        # Get file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        # Validate file
        extension, mime_type = validate_file_upload(
            filename=file.filename,
            file_size=file_size,
            max_size_bytes=settings.max_upload_size_bytes,
        )

        # Generate dataset ID and save file
        dataset_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        upload_folder = get_upload_folder()
        storage_path = upload_folder / f"{dataset_id}{extension}"
        file.save(str(storage_path))

        logger.info(f"File saved: {storage_path}")

        # Load and analyze the dataset
        if extension == ".csv":
            df = pd.read_csv(storage_path)
        elif extension in (".xlsx", ".xls"):
            df = pd.read_excel(storage_path)
        elif extension == ".json":
            df = pd.read_json(storage_path)
        elif extension == ".parquet":
            df = pd.read_parquet(storage_path)
        else:
            raise DatasetError(f"Unsupported format: {extension}")

        # Analyze columns
        analyzer = get_data_analyzer()
        columns = []
        for col in df.columns:
            col_info = analyzer.analyze_column(df, col)
            columns.append(
                {
                    "name": col_info.name,
                    "data_type": col_info.data_type.value,
                    "original_dtype": col_info.original_dtype,
                    "null_count": col_info.null_count,
                    "null_percentage": col_info.null_percentage,
                    "unique_count": col_info.unique_count,
                    "unique_percentage": col_info.unique_percentage,
                    "sample_values": col_info.sample_values[:5],
                }
            )

        # Get optional metadata from request
        name = request.form.get("name", filename)
        description = request.form.get("description", "")

        # Store dataset info
        dataset_info = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "owner_id": str(uuid.uuid4()),  # Placeholder
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
            "storage_path": str(storage_path),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_analyzed": None,
            "metadata": {
                "file_name": filename,
                "file_size_bytes": file_size,
                "file_format": extension,
                "encoding": "utf-8",
            },
            "tags": [],
        }

        _datasets[dataset_id] = dataset_info
        _insights[dataset_id] = []

        logger.info(
            f"Dataset created: {dataset_id} ({len(df)} rows, {len(df.columns)} cols)"
        )

        # Return dataset info with initial summary
        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "dataset": dataset_info,
                "message": "Dataset uploaded successfully",
            }
        ), 201

    except DatasetError as e:
        logger.error(f"Dataset error: {e.message}")
        return jsonify(e.to_dict()), 400
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "UPLOAD_FAILED",
                "message": str(e),
            }
        ), 500


@insights_bp.route("/datasets", methods=["GET"])
def list_datasets():
    """List all datasets."""
    datasets = [
        {
            "id": d["id"],
            "name": d["name"],
            "description": d["description"],
            "row_count": d["row_count"],
            "column_count": d["column_count"],
            "created_at": d["created_at"],
            "last_analyzed": d["last_analyzed"],
            "file_format": d["metadata"]["file_format"],
        }
        for d in _datasets.values()
    ]

    return jsonify(
        {
            "success": True,
            "datasets": datasets,
            "count": len(datasets),
        }
    )


@insights_bp.route("/datasets/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id: str):
    """Get dataset metadata and summary."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    dataset = _datasets[dataset_id]

    return jsonify(
        {
            "success": True,
            "dataset": dataset,
        }
    )


@insights_bp.route("/datasets/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    # Remove file
    storage_path = _datasets[dataset_id]["storage_path"]
    try:
        os.remove(storage_path)
    except OSError:
        pass

    # Remove from storage
    del _datasets[dataset_id]
    if dataset_id in _insights:
        del _insights[dataset_id]

    logger.info(f"Dataset deleted: {dataset_id}")

    return jsonify(
        {
            "success": True,
            "message": "Dataset deleted",
        }
    )


@insights_bp.route("/datasets/<dataset_id>/data", methods=["GET"])
def get_dataset_data(dataset_id: str):
    """Get dataset data with pagination."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    # Pagination parameters
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    per_page = min(per_page, 1000)  # Cap at 1000 rows

    try:
        df = load_dataframe(dataset_id)

        # Calculate pagination
        total_rows = len(df)
        total_pages = (total_rows + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Get page of data
        page_df = df.iloc[start_idx:end_idx]

        # Convert to records, handling NaN values
        records = page_df.fillna("").to_dict(orient="records")

        return jsonify(
            {
                "success": True,
                "data": records,
                "columns": list(df.columns),
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_rows": total_rows,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to get data: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "DATA_ERROR",
                "message": str(e),
            }
        ), 500


# =============================================================================
# Analysis Endpoints
# =============================================================================


@insights_bp.route("/datasets/<dataset_id>/analyze", methods=["POST"])
def analyze_dataset(dataset_id: str):
    """
    Trigger comprehensive AI analysis of a dataset.

    Request body:
        {
            "depth": "quick|standard|deep",
            "context": "optional business context"
        }
    """
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    try:
        # Get request parameters
        data = request.get_json() or {}
        depth = data.get("depth", "standard")
        context = data.get("context", "")

        # Validate depth
        depth = validate_analysis_depth(depth)

        # Load dataset
        df = load_dataframe(dataset_id)

        logger.info(f"Analyzing dataset {dataset_id} (depth={depth})")

        # Run analysis
        generator = get_insights_generator()
        analysis = generator.analyze_dataframe(
            df=df,
            context=context,
            depth=depth,
        )

        # Update dataset metadata
        _datasets[dataset_id]["last_analyzed"] = datetime.utcnow().isoformat()

        # Store insights
        for insight in analysis.insights:
            insight_dict = {
                "id": str(insight.id),
                "title": insight.title,
                "description": insight.description,
                "insight_type": insight.insight_type.value,
                "confidence": insight.confidence,
                "columns_involved": insight.columns_involved,
                "supporting_data": insight.supporting_data,
                "created_at": insight.created_at.isoformat(),
                "query": insight.query,
            }
            _insights[dataset_id].append(insight_dict)

        # Build response
        response = {
            "success": True,
            "dataset_id": dataset_id,
            "analysis": {
                "summary": analysis.summary.model_dump(),
                "quality_score": analysis.quality_score,
                "quality_metrics": analysis.quality_metrics.model_dump(),
                "pattern_count": len(analysis.patterns),
                "insight_count": len(analysis.insights),
                "recommendations": analysis.recommendations,
                "analyzed_at": analysis.analyzed_at.isoformat(),
                "depth": depth,
            },
            "patterns": [
                {
                    "id": str(p.id),
                    "type": p.pattern_type.value,
                    "description": p.description,
                    "columns": p.columns_involved,
                    "strength": p.strength,
                }
                for p in analysis.patterns
            ],
            "insights": [
                {
                    "id": str(i.id),
                    "title": i.title,
                    "description": i.description,
                    "type": i.insight_type.value,
                    "confidence": i.confidence,
                    "columns": i.columns_involved,
                }
                for i in analysis.insights
            ],
            "token_usage": analysis.token_usage,
        }

        logger.info(f"Analysis complete: {len(analysis.insights)} insights")

        return jsonify(response)

    except DataValidationError as e:
        return jsonify(e.to_dict()), 400
    except InsightGenerationError as e:
        return jsonify(e.to_dict()), 500
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "ANALYSIS_FAILED",
                "message": str(e),
            }
        ), 500


@insights_bp.route("/datasets/<dataset_id>/quick-analyze", methods=["GET"])
def quick_analyze_dataset(dataset_id: str):
    """
    Get quick statistical analysis without AI (faster).
    """
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    try:
        df = load_dataframe(dataset_id)
        generator = get_insights_generator()

        result = generator.quick_analyze(df)

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "analysis": result,
            }
        )

    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "ANALYSIS_FAILED",
                "message": str(e),
            }
        ), 500


# =============================================================================
# Insights Endpoints
# =============================================================================


@insights_bp.route("/datasets/<dataset_id>/insights", methods=["POST"])
def generate_insights(dataset_id: str):
    """
    Generate insights from a natural language query.

    Request body:
        {
            "query": "What are the main trends?",
            "max_results": 5
        }
    """
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    try:
        # Get request parameters
        data = request.get_json() or {}
        query = data.get("query", "")
        max_results = data.get("max_results", 5)

        # Validate query
        query = validate_query(query)
        max_results = min(max(1, max_results), 10)

        # Load dataset
        df = load_dataframe(dataset_id)

        logger.info(f"Generating insights for query: {query[:50]}...")

        # Generate insights
        generator = get_insights_generator()
        insights = generator.get_insights(
            df=df,
            query=query,
            max_insights=max_results,
        )

        # Store and format insights
        results = []
        for insight in insights:
            insight_dict = {
                "id": str(insight.id),
                "title": insight.title,
                "description": insight.description,
                "insight_type": insight.insight_type.value,
                "confidence": insight.confidence,
                "confidence_label": insight.confidence_label,
                "columns_involved": insight.columns_involved,
                "supporting_data": insight.supporting_data,
                "created_at": insight.created_at.isoformat(),
                "query": query,
            }
            _insights[dataset_id].append(insight_dict)
            results.append(insight_dict)

        logger.info(f"Generated {len(results)} insights")

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "query": query,
                "insights": results,
                "count": len(results),
            }
        )

    except DataValidationError as e:
        return jsonify(e.to_dict()), 400
    except InsightGenerationError as e:
        return jsonify(e.to_dict()), 500
    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "GENERATION_FAILED",
                "message": str(e),
            }
        ), 500


@insights_bp.route("/datasets/<dataset_id>/insights", methods=["GET"])
def list_insights(dataset_id: str):
    """List all generated insights for a dataset."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    insights = _insights.get(dataset_id, [])

    # Optional filtering
    insight_type = request.args.get("type")
    min_confidence = request.args.get("min_confidence", type=float)

    if insight_type:
        insights = [i for i in insights if i["insight_type"] == insight_type]

    if min_confidence:
        insights = [i for i in insights if i["confidence"] >= min_confidence]

    # Sort by confidence (highest first)
    insights = sorted(insights, key=lambda x: x["confidence"], reverse=True)

    return jsonify(
        {
            "success": True,
            "dataset_id": dataset_id,
            "insights": insights,
            "count": len(insights),
        }
    )


@insights_bp.route("/datasets/<dataset_id>/insights/<insight_id>", methods=["GET"])
def get_insight(dataset_id: str, insight_id: str):
    """Get a specific insight."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    insights = _insights.get(dataset_id, [])
    insight = next((i for i in insights if i["id"] == insight_id), None)

    if not insight:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Insight not found",
            }
        ), 404

    return jsonify(
        {
            "success": True,
            "insight": insight,
        }
    )


@insights_bp.route(
    "/datasets/<dataset_id>/insights/<insight_id>/explain", methods=["GET"]
)
def explain_insight(dataset_id: str, insight_id: str):
    """Get a human-readable explanation of an insight."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    insights = _insights.get(dataset_id, [])
    insight_dict = next((i for i in insights if i["id"] == insight_id), None)

    if not insight_dict:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Insight not found",
            }
        ), 404

    audience = request.args.get("audience", "business")

    try:
        # Reconstruct insight object
        from insightboost.models.insight import Insight, InsightType

        insight = Insight(
            id=insight_dict["id"],
            title=insight_dict["title"],
            description=insight_dict["description"],
            insight_type=InsightType(insight_dict["insight_type"]),
            confidence=insight_dict["confidence"],
            columns_involved=insight_dict["columns_involved"],
            supporting_data=insight_dict["supporting_data"],
        )

        generator = get_insights_generator()
        explanation = generator.explain_insight(insight, audience=audience)

        return jsonify(
            {
                "success": True,
                "insight_id": insight_id,
                "audience": audience,
                "explanation": explanation,
            }
        )

    except Exception as e:
        logger.error(f"Failed to explain insight: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "EXPLANATION_FAILED",
                "message": str(e),
            }
        ), 500


@insights_bp.route("/datasets/<dataset_id>/patterns", methods=["GET"])
def discover_patterns(dataset_id: str):
    """Discover patterns in the dataset."""
    if dataset_id not in _datasets:
        return jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Dataset not found",
            }
        ), 404

    try:
        # Get pattern types from query params
        pattern_types = request.args.getlist("types")
        if not pattern_types:
            pattern_types = None

        df = load_dataframe(dataset_id)
        generator = get_insights_generator()

        patterns = generator.discover_patterns(df, pattern_types=pattern_types)

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "patterns": [
                    {
                        "id": str(p.id),
                        "type": p.pattern_type.value,
                        "description": p.description,
                        "columns": p.columns_involved,
                        "strength": p.strength,
                        "details": p.details,
                    }
                    for p in patterns
                ],
                "count": len(patterns),
            }
        )

    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        return jsonify(
            {
                "error": True,
                "error_code": "DISCOVERY_FAILED",
                "message": str(e),
            }
        ), 500
