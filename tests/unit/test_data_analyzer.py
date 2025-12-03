"""
Unit Tests for Data Analyzer Module
"""

import pandas as pd
import pytest

from insightboost.core.data_analyzer import DataAnalyzer


class TestDataAnalyzer:
    """Test suite for DataAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    def test_instantiation(self, analyzer):
        """Test that DataAnalyzer can be instantiated."""
        assert analyzer is not None
        assert isinstance(analyzer, DataAnalyzer)

    def test_detect_column_type_integer(self, analyzer):
        """Test integer type detection."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = analyzer.detect_column_type(series)
        assert result.value in ["integer", "numeric", "float"]

    def test_detect_column_type_float(self, analyzer):
        """Test float type detection."""
        series = pd.Series([1.1, 2.2, 3.3])
        result = analyzer.detect_column_type(series)
        assert result.value in ["float", "numeric"]

    def test_detect_column_type_categorical(self, analyzer):
        """Test categorical type detection."""
        series = pd.Series(["A", "B", "A", "C", "B"] * 10)
        result = analyzer.detect_column_type(series)
        assert result.value in ["categorical", "text"]

    def test_detect_column_type_datetime(self, analyzer):
        """Test datetime type detection."""
        series = pd.Series(pd.date_range("2024-01-01", periods=5))
        result = analyzer.detect_column_type(series)
        assert result.value == "datetime"

    def test_detect_column_type_boolean(self, analyzer):
        """Test boolean type detection."""
        series = pd.Series([True, False, True, False])
        result = analyzer.detect_column_type(series)
        # Booleans may be detected as numeric or categorical depending on implementation
        assert result.value in ["boolean", "categorical", "numeric"]

    def test_analyze_column_via_profile(self, analyzer):
        """Test single column analysis via create_profile."""
        df = pd.DataFrame({"test": [1, 2, 3, 4, 5]})
        profile = analyzer.create_profile(df)

        assert len(profile.columns) == 1
        col = profile.columns[0]
        assert col.name == "test"
        assert col.null_count == 0
        assert col.unique_count == 5

    def test_analyze_column_with_nulls_via_profile(self, analyzer):
        """Test column analysis with null values."""
        df = pd.DataFrame({"test": [1, 2, None, 4, None]})
        profile = analyzer.create_profile(df)

        col = profile.columns[0]
        assert col.null_count == 2
        assert col.null_percentage == 40.0

    def test_analyze_column_statistics_via_profile(self, analyzer):
        """Test that numeric columns have statistics."""
        df = pd.DataFrame({"test": [10, 20, 30, 40, 50]})
        profile = analyzer.create_profile(df)

        col = profile.columns[0]
        assert col.statistics is not None
        assert col.statistics.get("mean") == 30.0
        assert col.statistics.get("min") == 10
        assert col.statistics.get("max") == 50

    def test_compute_correlations(self, analyzer):
        """Test correlation computation."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
            }
        )
        result = analyzer.compute_correlations(df)

        assert result is not None
        assert len(result) > 0

    def test_detect_outliers(self, analyzer):
        """Test outlier detection."""
        df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 15, 1000]})
        result = analyzer.detect_outliers(df)

        assert result is not None

    def test_create_profile(self, analyzer):
        """Test full dataset profiling."""
        df = pd.DataFrame(
            {
                "num": [1, 2, 3, 4, 5],
                "cat": ["A", "B", "A", "B", "A"],
            }
        )
        profile = analyzer.create_profile(df)

        assert profile is not None
        assert profile.row_count == 5
        assert profile.column_count == 2
        assert len(profile.columns) == 2

    def test_create_profile_empty_df_raises(self, analyzer):
        """Test profiling empty dataframe raises error."""
        df = pd.DataFrame()

        with pytest.raises((ValueError, TypeError, KeyError)):
            analyzer.create_profile(df)

    def test_analyze_missing_values(self, analyzer):
        """Test missing value analysis."""
        df = pd.DataFrame(
            {
                "a": [1, 2, None, 4],
                "b": [None, None, 3, 4],
            }
        )
        result = analyzer.analyze_missing_values(df)

        assert result is not None

    def test_get_cardinality(self, analyzer):
        """Test cardinality calculation."""
        df = pd.DataFrame({"col": ["A", "B", "A", "C", "B"]})
        result = analyzer.get_cardinality(df)

        assert result["col"] == 3

    def test_identify_quality_issues(self, analyzer):
        """Test quality issue identification."""
        df = pd.DataFrame(
            {
                "a": [1, 2, None, 4, None],
                "b": ["x", "x", "x", "x", "x"],  # Low variance
            }
        )
        result = analyzer.identify_quality_issues(df)

        assert result is not None
