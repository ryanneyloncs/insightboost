"""
Validation utilities for InsightBoost.

This module provides validation functions for data, files, and user inputs.
"""

from pathlib import Path

import pandas as pd

from insightboost.utils.exceptions import DatasetError, DataValidationError

# Supported file extensions and their MIME types
SUPPORTED_EXTENSIONS = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".json": "application/json",
    ".parquet": "application/octet-stream",
}

# Maximum values for validation
MAX_COLUMNS = 500
MAX_ROWS = 10_000_000
MAX_COLUMN_NAME_LENGTH = 256
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 2000


def validate_dataframe(
    df: pd.DataFrame,
    require_columns: list[str] | None = None,
    max_rows: int | None = None,
    max_columns: int | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """
    Validate a pandas DataFrame.

    Args:
        df: DataFrame to validate
        require_columns: List of required column names
        max_rows: Maximum allowed rows (defaults to MAX_ROWS)
        max_columns: Maximum allowed columns (defaults to MAX_COLUMNS)
        allow_empty: Whether to allow empty DataFrames

    Returns:
        The validated DataFrame

    Raises:
        DataValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            message="Input must be a pandas DataFrame",
            field="dataframe",
        )

    # Check for empty DataFrame
    if df.empty and not allow_empty:
        raise DataValidationError(
            message="DataFrame is empty",
            field="dataframe",
        )

    # Check row count
    max_rows = max_rows or MAX_ROWS
    if len(df) > max_rows:
        raise DataValidationError(
            message=f"DataFrame exceeds maximum row count of {max_rows:,}",
            field="rows",
            details={"row_count": len(df), "max_rows": max_rows},
        )

    # Check column count
    max_columns = max_columns or MAX_COLUMNS
    if len(df.columns) > max_columns:
        raise DataValidationError(
            message=f"DataFrame exceeds maximum column count of {max_columns}",
            field="columns",
            details={"column_count": len(df.columns), "max_columns": max_columns},
        )

    # Check column names
    for col in df.columns:
        col_str = str(col)
        if len(col_str) > MAX_COLUMN_NAME_LENGTH:
            raise DataValidationError(
                message=f"Column name exceeds maximum length of {MAX_COLUMN_NAME_LENGTH}",
                field="column_name",
                details={"column": col_str[:50] + "..."},
            )

    # Check for required columns
    if require_columns:
        missing_columns = set(require_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(
                message=f"Missing required columns: {missing_columns}",
                field="columns",
                details={"missing_columns": list(missing_columns)},
            )

    # Check for duplicate column names
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise DataValidationError(
            message="DataFrame contains duplicate column names",
            field="columns",
            details={"duplicate_columns": duplicate_columns},
        )

    return df


def validate_file_upload(
    filename: str,
    file_size: int,
    max_size_bytes: int,
) -> tuple[str, str]:
    """
    Validate a file upload.

    Args:
        filename: Name of the uploaded file
        file_size: Size of the file in bytes
        max_size_bytes: Maximum allowed file size in bytes

    Returns:
        Tuple of (extension, mime_type)

    Raises:
        DatasetError: If validation fails
    """
    if not filename:
        raise DatasetError(
            message="Filename is required",
            details={"field": "filename"},
        )

    # Get file extension
    path = Path(filename)
    extension = path.suffix.lower()

    # Check if extension is supported
    if extension not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            message=f"Unsupported file type: {extension}",
            details={
                "extension": extension,
                "supported_extensions": list(SUPPORTED_EXTENSIONS.keys()),
            },
        )

    # Check file size
    if file_size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        file_mb = file_size / (1024 * 1024)
        raise DatasetError(
            message=f"File size ({file_mb:.1f} MB) exceeds maximum ({max_mb:.1f} MB)",
            details={
                "file_size_bytes": file_size,
                "max_size_bytes": max_size_bytes,
            },
        )

    # Check for empty file
    if file_size == 0:
        raise DatasetError(
            message="File is empty",
            details={"filename": filename},
        )

    mime_type = SUPPORTED_EXTENSIONS[extension]
    return extension, mime_type


def validate_query(
    query: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> str:
    """
    Validate a natural language query.

    Args:
        query: Query string to validate
        min_length: Minimum query length (defaults to MIN_QUERY_LENGTH)
        max_length: Maximum query length (defaults to MAX_QUERY_LENGTH)

    Returns:
        The validated and cleaned query

    Raises:
        DataValidationError: If validation fails
    """
    if not isinstance(query, str):
        raise DataValidationError(
            message="Query must be a string",
            field="query",
        )

    # Clean the query
    cleaned_query = query.strip()

    # Check length
    min_length = min_length or MIN_QUERY_LENGTH
    max_length = max_length or MAX_QUERY_LENGTH

    if len(cleaned_query) < min_length:
        raise DataValidationError(
            message=f"Query must be at least {min_length} characters",
            field="query",
            details={"length": len(cleaned_query), "min_length": min_length},
        )

    if len(cleaned_query) > max_length:
        raise DataValidationError(
            message=f"Query exceeds maximum length of {max_length} characters",
            field="query",
            details={"length": len(cleaned_query), "max_length": max_length},
        )

    return cleaned_query


def validate_uuid(value: str, field_name: str = "id") -> str:
    """
    Validate a UUID string.

    Args:
        value: UUID string to validate
        field_name: Name of the field for error messages

    Returns:
        The validated UUID string

    Raises:
        DataValidationError: If validation fails
    """
    import uuid

    if not isinstance(value, str):
        raise DataValidationError(
            message=f"{field_name} must be a string",
            field=field_name,
        )

    try:
        # Validate UUID format
        uuid.UUID(value)
        return value
    except ValueError as e:
        raise DataValidationError(
            message=f"Invalid UUID format for {field_name}",
            field=field_name,
            details={"value": value, "error": str(e)},
        ) from e


def validate_chart_type(chart_type: str) -> str:
    """
    Validate a chart type string.

    Args:
        chart_type: Chart type to validate

    Returns:
        The validated chart type

    Raises:
        DataValidationError: If validation fails
    """
    valid_chart_types = {
        "scatter",
        "line",
        "bar",
        "histogram",
        "box",
        "violin",
        "heatmap",
        "pie",
        "area",
        "bubble",
        "funnel",
        "treemap",
        "sunburst",
        "parallel_coordinates",
        "scatter_matrix",
    }

    if not isinstance(chart_type, str):
        raise DataValidationError(
            message="Chart type must be a string",
            field="chart_type",
        )

    normalized = chart_type.lower().strip()

    if normalized not in valid_chart_types:
        raise DataValidationError(
            message=f"Invalid chart type: {chart_type}",
            field="chart_type",
            details={
                "chart_type": chart_type,
                "valid_types": sorted(valid_chart_types),
            },
        )

    return normalized


def validate_analysis_depth(depth: str) -> str:
    """
    Validate analysis depth setting.

    Args:
        depth: Analysis depth to validate

    Returns:
        The validated depth

    Raises:
        DataValidationError: If validation fails
    """
    valid_depths = {"quick", "standard", "deep"}

    if not isinstance(depth, str):
        raise DataValidationError(
            message="Analysis depth must be a string",
            field="depth",
        )

    normalized = depth.lower().strip()

    if normalized not in valid_depths:
        raise DataValidationError(
            message=f"Invalid analysis depth: {depth}",
            field="depth",
            details={
                "depth": depth,
                "valid_depths": sorted(valid_depths),
            },
        )

    return normalized
