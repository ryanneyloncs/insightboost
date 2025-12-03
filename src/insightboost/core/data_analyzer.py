"""
Data analysis utilities for InsightBoost.

This module provides statistical analysis functions that prepare data
for LLM-powered insights generation.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from insightboost.config.logging_config import get_logger
from insightboost.models.dataset import ColumnDataType, ColumnInfo
from insightboost.utils.validators import validate_dataframe


logger = get_logger("data_analyzer")


@dataclass
class DistributionStats:
    """Statistics about a numeric distribution."""
    
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    q1: float
    q3: float
    skewness: float
    kurtosis: float
    is_normal: bool
    outlier_count: int
    outlier_percentage: float


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two columns."""
    
    column1: str
    column2: str
    correlation: float
    p_value: float | None
    strength: str
    direction: str


@dataclass
class TimeSeriesCharacteristics:
    """Characteristics of time series data."""
    
    has_trend: bool
    trend_direction: str | None
    has_seasonality: bool
    seasonality_period: int | None
    is_stationary: bool
    autocorrelation_lag1: float | None


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    
    row_count: int
    column_count: int
    memory_usage_bytes: int
    columns: list[ColumnInfo]
    correlations: list[CorrelationResult]
    numeric_distributions: dict[str, DistributionStats]
    time_series_columns: dict[str, TimeSeriesCharacteristics]
    quality_issues: list[str]


class DataAnalyzer:
    """
    Statistical analysis utilities for data preparation.
    
    This class provides methods to analyze DataFrames and extract
    statistical properties that inform LLM-powered insights.
    """
    
    def __init__(self) -> None:
        """Initialize the data analyzer."""
        self.logger = get_logger("DataAnalyzer")
    
    def detect_column_type(
        self,
        series: pd.Series,
    ) -> ColumnDataType:
        """
        Detect the semantic type of a column.
        
        Args:
            series: Column to analyze
            
        Returns:
            Detected column type
        """
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnDataType.DATETIME
        
        # Check for numeric types
        if pd.api.types.is_integer_dtype(series):
            return ColumnDataType.INTEGER
        if pd.api.types.is_float_dtype(series):
            return ColumnDataType.FLOAT
        if pd.api.types.is_numeric_dtype(series):
            return ColumnDataType.NUMERIC
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return ColumnDataType.BOOLEAN
        
        # Check for categorical
        if pd.api.types.is_categorical_dtype(series):
            return ColumnDataType.CATEGORICAL
        
        # Check object dtype - could be text or categorical
        if series.dtype == "object":
            # Try to parse as datetime
            try:
                pd.to_datetime(series.dropna().head(100))
                return ColumnDataType.DATETIME
            except (ValueError, TypeError):
                pass
            
            # Check cardinality for categorical vs text
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            avg_length = series.astype(str).str.len().mean()
            
            if unique_ratio < 0.5 and avg_length < 50:
                return ColumnDataType.CATEGORICAL
            elif avg_length > 100:
                return ColumnDataType.TEXT
            else:
                return ColumnDataType.CATEGORICAL
        
        return ColumnDataType.UNKNOWN
    
    def analyze_column(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> ColumnInfo:
        """
        Analyze a single column.
        
        Args:
            df: DataFrame containing the column
            column: Column name to analyze
            
        Returns:
            ColumnInfo with analysis results
        """
        series = df[column]
        
        data_type = self.detect_column_type(series)
        
        null_count = int(series.isnull().sum())
        null_pct = (null_count / len(series) * 100) if len(series) > 0 else 0
        unique_count = int(series.nunique())
        unique_pct = (unique_count / len(series) * 100) if len(series) > 0 else 0
        
        # Get sample values (non-null)
        sample_values = series.dropna().head(5).tolist()
        
        # Calculate statistics for numeric columns
        statistics: dict[str, Any] = {}
        if data_type in (ColumnDataType.NUMERIC, ColumnDataType.INTEGER, ColumnDataType.FLOAT):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                statistics = {
                    "mean": float(clean_series.mean()),
                    "median": float(clean_series.median()),
                    "std": float(clean_series.std()),
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "q1": float(clean_series.quantile(0.25)),
                    "q3": float(clean_series.quantile(0.75)),
                }
        
        # Check if column could be an ID
        is_potential_id = (
            unique_pct > 95 and
            data_type in (ColumnDataType.INTEGER, ColumnDataType.CATEGORICAL) and
            ("id" in column.lower() or "key" in column.lower() or "code" in column.lower())
        )
        
        # Check if column could be a date
        is_potential_date = data_type == ColumnDataType.DATETIME or (
            data_type == ColumnDataType.CATEGORICAL and
            any(kw in column.lower() for kw in ["date", "time", "created", "updated"])
        )
        
        return ColumnInfo(
            name=column,
            data_type=data_type,
            original_dtype=str(series.dtype),
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            sample_values=sample_values,
            statistics=statistics,
            is_potential_id=is_potential_id,
            is_potential_date=is_potential_date,
        )
    
    def analyze_distribution(
        self,
        series: pd.Series,
    ) -> DistributionStats | None:
        """
        Analyze the distribution of a numeric series.
        
        Args:
            series: Numeric series to analyze
            
        Returns:
            DistributionStats or None if not numeric
        """
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return None
        
        # Basic statistics
        mean = float(clean_series.mean())
        median = float(clean_series.median())
        std = float(clean_series.std())
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        
        # Skewness and kurtosis
        skewness = float(clean_series.skew())
        kurtosis = float(clean_series.kurtosis())
        
        # Check for normality (rough approximation)
        is_normal = abs(skewness) < 0.5 and abs(kurtosis) < 1
        
        # Detect outliers using IQR
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(clean_series) * 100) if len(clean_series) > 0 else 0
        
        return DistributionStats(
            mean=mean,
            median=median,
            std=std,
            min_val=min_val,
            max_val=max_val,
            q1=q1,
            q3=q3,
            skewness=skewness,
            kurtosis=kurtosis,
            is_normal=is_normal,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage,
        )
    
    def compute_correlations(
        self,
        df: pd.DataFrame,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        threshold: float = 0.3,
    ) -> list[CorrelationResult]:
        """
        Compute correlations between numeric columns.
        
        Args:
            df: DataFrame to analyze
            method: Correlation method
            threshold: Minimum absolute correlation to include
            
        Returns:
            List of CorrelationResult objects
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return []
        
        corr_matrix = numeric_df.corr(method=method)
        
        results = []
        columns = list(corr_matrix.columns)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Only upper triangle
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if pd.isna(corr_value):
                        continue
                    
                    if abs(corr_value) >= threshold:
                        # Determine strength
                        abs_corr = abs(corr_value)
                        if abs_corr >= 0.9:
                            strength = "very strong"
                        elif abs_corr >= 0.7:
                            strength = "strong"
                        elif abs_corr >= 0.5:
                            strength = "moderate"
                        else:
                            strength = "weak"
                        
                        direction = "positive" if corr_value > 0 else "negative"
                        
                        results.append(CorrelationResult(
                            column1=col1,
                            column2=col2,
                            correlation=round(corr_value, 4),
                            p_value=None,  # Would require scipy
                            strength=strength,
                            direction=direction,
                        ))
        
        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return results
    
    def detect_outliers(
        self,
        series: pd.Series,
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 1.5,
    ) -> pd.Series:
        """
        Detect outliers in a numeric series.
        
        Args:
            series: Series to analyze
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series([False] * len(series), index=series.index)
        
        clean_series = series.fillna(series.median())
        
        if method == "iqr":
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (clean_series < lower) | (clean_series > upper)
        
        elif method == "zscore":
            mean = clean_series.mean()
            std = clean_series.std()
            if std == 0:
                return pd.Series([False] * len(series), index=series.index)
            z_scores = abs((clean_series - mean) / std)
            return z_scores > threshold
        
        return pd.Series([False] * len(series), index=series.index)
    
    def analyze_missing_values(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Analyze missing value patterns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with missing value analysis
        """
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        
        # Per-column missing
        column_missing = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                column_missing[col] = {
                    "count": missing_count,
                    "percentage": round(missing_count / len(df) * 100, 2),
                }
        
        # Rows with any missing
        rows_with_missing = int(df.isnull().any(axis=1).sum())
        
        # Complete rows
        complete_rows = len(df) - rows_with_missing
        
        return {
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "missing_percentage": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
            "column_missing": column_missing,
            "rows_with_missing": rows_with_missing,
            "complete_rows": complete_rows,
            "complete_row_percentage": round(complete_rows / len(df) * 100, 2) if len(df) > 0 else 0,
        }
    
    def detect_time_series(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> TimeSeriesCharacteristics | None:
        """
        Detect time series characteristics.
        
        Args:
            df: DataFrame containing the column
            column: Column to analyze
            
        Returns:
            TimeSeriesCharacteristics or None
        """
        if column not in df.columns:
            return None
        
        series = df[column].dropna()
        
        if len(series) < 10:
            return None
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(series))
        try:
            slope = np.polyfit(x, series.values, 1)[0]
            has_trend = abs(slope) > series.std() * 0.01
            trend_direction = "up" if slope > 0 else "down" if slope < 0 else None
        except (np.linalg.LinAlgError, ValueError):
            has_trend = False
            trend_direction = None
        
        # Autocorrelation at lag 1
        try:
            autocorr = series.autocorr(lag=1)
        except (ValueError, TypeError):
            autocorr = None
        
        # Simple seasonality check (would need more sophisticated methods)
        has_seasonality = False
        seasonality_period = None
        
        # Stationarity check (simplified)
        is_stationary = not has_trend and (autocorr is None or abs(autocorr) < 0.5)
        
        return TimeSeriesCharacteristics(
            has_trend=has_trend,
            trend_direction=trend_direction if has_trend else None,
            has_seasonality=has_seasonality,
            seasonality_period=seasonality_period,
            is_stationary=is_stationary,
            autocorrelation_lag1=round(autocorr, 4) if autocorr is not None else None,
        )
    
    def get_cardinality(
        self,
        df: pd.DataFrame,
    ) -> dict[str, int]:
        """
        Get cardinality (unique count) for each column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to unique counts
        """
        return {col: int(df[col].nunique()) for col in df.columns}
    
    def identify_quality_issues(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Identify data quality issues.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of quality issue descriptions
        """
        issues = []
        
        # Check for high missing rates
        missing = self.analyze_missing_values(df)
        if missing["missing_percentage"] > 20:
            issues.append(f"High missing value rate: {missing['missing_percentage']:.1f}%")
        
        for col, info in missing["column_missing"].items():
            if info["percentage"] > 50:
                issues.append(f"Column '{col}' has {info['percentage']:.1f}% missing values")
        
        # Check for potential duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"Column '{col}' has only one unique value")
        
        # Check for high cardinality in object columns
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                issues.append(f"Column '{col}' has very high cardinality (possible ID column)")
        
        # Check for outliers in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            outliers = self.detect_outliers(df[col])
            outlier_pct = outliers.sum() / len(df) * 100
            if outlier_pct > 5:
                issues.append(f"Column '{col}' has {outlier_pct:.1f}% outliers")
        
        return issues
    
    def create_profile(
        self,
        df: pd.DataFrame,
    ) -> DataProfile:
        """
        Create a complete data profile.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            DataProfile with complete analysis
        """
        validate_dataframe(df)
        
        self.logger.info(f"Creating profile for DataFrame: {df.shape}")
        
        # Analyze all columns
        columns = [self.analyze_column(df, col) for col in df.columns]
        
        # Compute correlations
        correlations = self.compute_correlations(df)
        
        # Analyze distributions for numeric columns
        numeric_distributions = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            dist = self.analyze_distribution(df[col])
            if dist:
                numeric_distributions[col] = dist
        
        # Check for time series columns
        time_series_columns = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            ts = self.detect_time_series(df, col)
            if ts and ts.has_trend:
                time_series_columns[col] = ts
        
        # Identify quality issues
        quality_issues = self.identify_quality_issues(df)
        
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage_bytes=df.memory_usage(deep=True).sum(),
            columns=columns,
            correlations=correlations,
            numeric_distributions=numeric_distributions,
            time_series_columns=time_series_columns,
            quality_issues=quality_issues,
        )
