"""
Formatting utilities for InsightBoost.

This module provides functions to format data for display, LLM prompts,
and API responses.
"""

from typing import Any

import numpy as np
import pandas as pd


def format_dataframe_for_llm(
    df: pd.DataFrame,
    max_rows: int = 10,
    max_cols: int = 20,
    include_stats: bool = True,
) -> dict[str, Any]:
    """
    Format a DataFrame for LLM consumption.

    This function prepares DataFrame metadata and sample data in a format
    suitable for sending to an LLM for analysis.

    Args:
        df: DataFrame to format
        max_rows: Maximum sample rows to include
        max_cols: Maximum columns to include in details
        include_stats: Whether to include statistical summaries

    Returns:
        Dictionary containing formatted DataFrame information
    """
    # Basic info
    info = {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": [],
        "sample_data": None,
        "missing_summary": {},
        "numeric_stats": {},
    }

    # Column information
    columns_to_process = list(df.columns)[:max_cols]
    for col in columns_to_process:
        col_info = {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        # Add sample values for non-numeric columns
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            sample_values = df[col].dropna().head(5).tolist()
            col_info["sample_values"] = [str(v)[:50] for v in sample_values]

        info["columns"].append(col_info)

    # Sample data (as string for LLM)
    sample_df = df.head(max_rows)
    info["sample_data"] = sample_df.to_string(max_cols=max_cols)

    # Missing value summary
    missing = df.isnull().sum()
    info["missing_summary"] = {
        str(col): int(count) for col, count in missing.items() if count > 0
    }

    # Numeric statistics
    if include_stats:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats = numeric_df.describe().round(4)
            info["numeric_stats"] = stats.to_dict()

    return info


def format_columns_with_types(df: pd.DataFrame) -> str:
    """
    Format column names with their data types.

    Args:
        df: DataFrame to analyze

    Returns:
        Formatted string of columns and types
    """
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 0:
            lines.append(f"  - {col}: {dtype} ({null_pct:.1f}% null)")
        else:
            lines.append(f"  - {col}: {dtype}")
    return "\n".join(lines)


def format_number(
    value: float | int,
    precision: int = 2,
    use_grouping: bool = True,
) -> str:
    """
    Format a number for display.

    Args:
        value: Number to format
        precision: Decimal precision for floats
        use_grouping: Whether to use thousand separators

    Returns:
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"

    if isinstance(value, float):
        if abs(value) >= 1_000_000:
            # Use scientific notation for very large numbers
            return f"{value:.{precision}e}"
        elif abs(value) >= 1:
            if use_grouping:
                return f"{value:,.{precision}f}"
            return f"{value:.{precision}f}"
        else:
            # More precision for small numbers
            return f"{value:.{precision + 2}f}"
    else:
        if use_grouping:
            return f"{value:,}"
        return str(value)


def truncate_string(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate a string to a maximum length.

    Args:
        text: String to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


def format_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Format correlation matrix for LLM analysis.

    Args:
        df: DataFrame to analyze
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Only include correlations above this absolute value

    Returns:
        Dictionary with correlation information
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or len(numeric_df.columns) < 2:
        return {"correlations": [], "message": "Not enough numeric columns"}

    corr_matrix = numeric_df.corr(method=method)

    # Find significant correlations
    correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= threshold:
                    correlations.append(
                        {
                            "column1": str(col1),
                            "column2": str(col2),
                            "correlation": round(corr_value, 4),
                            "strength": _correlation_strength(corr_value),
                        }
                    )

    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "method": method,
        "threshold": threshold,
        "correlations": correlations[:20],  # Limit to top 20
    }


def _correlation_strength(value: float) -> str:
    """Classify correlation strength."""
    abs_val = abs(value)
    if abs_val >= 0.9:
        return "very strong"
    elif abs_val >= 0.7:
        return "strong"
    elif abs_val >= 0.5:
        return "moderate"
    elif abs_val >= 0.3:
        return "weak"
    else:
        return "very weak"


def format_distribution_info(
    series: pd.Series,
    bins: int = 10,
) -> dict[str, Any]:
    """
    Format distribution information for a series.

    Args:
        series: Pandas Series to analyze
        bins: Number of histogram bins

    Returns:
        Dictionary with distribution information
    """
    clean_series = series.dropna()

    if clean_series.empty:
        return {"empty": True}

    if pd.api.types.is_numeric_dtype(clean_series):
        # Numeric distribution
        info = {
            "type": "numeric",
            "mean": float(clean_series.mean()),
            "median": float(clean_series.median()),
            "std": float(clean_series.std()),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "skewness": float(clean_series.skew()),
            "kurtosis": float(clean_series.kurtosis()),
        }

        # Histogram data
        hist, bin_edges = np.histogram(clean_series, bins=bins)
        info["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": [round(e, 4) for e in bin_edges.tolist()],
        }

    else:
        # Categorical distribution
        value_counts = clean_series.value_counts()
        info = {
            "type": "categorical",
            "unique_count": int(clean_series.nunique()),
            "top_values": [
                {"value": str(v), "count": int(c)}
                for v, c in value_counts.head(10).items()
            ],
            "concentration": (
                float(value_counts.iloc[0] / len(clean_series))
                if len(value_counts) > 0
                else 0
            ),
        }

    return info


def format_time_for_display(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(size: int) -> str:
    """
    Format bytes into human-readable size string.

    Args:
        size: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def format_insight_for_display(
    title: str,
    description: str,
    confidence: float,
    insight_type: str,
) -> str:
    """
    Format an insight for display.

    Args:
        title: Insight title
        description: Insight description
        confidence: Confidence score (0-1)
        insight_type: Type of insight

    Returns:
        Formatted insight string
    """
    confidence_pct = int(confidence * 100)

    # Confidence indicator
    if confidence >= 0.8:
        confidence_label = "High confidence"
    elif confidence >= 0.5:
        confidence_label = "Moderate confidence"
    else:
        confidence_label = "Low confidence"

    return (
        f"[{insight_type.upper()}] {title}\n"
        f"{description}\n"
        f"({confidence_label}: {confidence_pct}%)"
    )
