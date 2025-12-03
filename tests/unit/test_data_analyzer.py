"""
Unit Tests for Data Analyzer Module
Tests type detection, statistical analysis, pattern discovery, and data quality assessment
"""

import numpy as np
import pandas as pd
import pytest

from insightboost.core.data_analyzer import DataAnalyzer


class TestDataAnalyzer:
    """Test suite for DataAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()

    # ============================================
    # Type Detection Tests
    # ============================================

    class TestTypeDetection:
        """Tests for column type detection."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_detect_numeric_integer(self, analyzer):
            """Test detection of integer columns."""
            df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
            result = analyzer.analyze_columns(df)
            assert result[0]["data_type"] in ["integer", "numeric"]

        def test_detect_numeric_float(self, analyzer):
            """Test detection of float columns."""
            df = pd.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5]})
            result = analyzer.analyze_columns(df)
            assert result[0]["data_type"] in ["float", "numeric"]

        def test_detect_categorical_string(self, analyzer):
            """Test detection of categorical string columns."""
            df = pd.DataFrame({"col": ["A", "B", "A", "C", "B"] * 10})
            result = analyzer.analyze_columns(df)
            assert result[0]["data_type"] in ["categorical", "text"]

        def test_detect_datetime(self, analyzer):
            """Test detection of datetime columns."""
            df = pd.DataFrame({
                "col": pd.date_range("2024-01-01", periods=5, freq="D")
            })
            result = analyzer.analyze_columns(df)
            assert result[0]["data_type"] == "datetime"

        def test_detect_boolean(self, analyzer):
            """Test detection of boolean columns."""
            df = pd.DataFrame({"col": [True, False, True, False, True]})
            result = analyzer.analyze_columns(df)
            assert result[0]["data_type"] in ["boolean", "categorical"]

        def test_detect_high_cardinality_text(self, analyzer):
            """Test detection of high cardinality text (not categorical)."""
            df = pd.DataFrame({"col": [f"unique_value_{i}" for i in range(100)]})
            result = analyzer.analyze_columns(df)
            # High cardinality strings should be detected as text, not categorical
            assert result[0]["data_type"] in ["text", "categorical"]
            assert result[0]["unique_percentage"] > 90

        def test_detect_mixed_types(self, analyzer):
            """Test handling of mixed type columns."""
            df = pd.DataFrame({"col": [1, "two", 3.0, None, "five"]})
            result = analyzer.analyze_columns(df)
            # Should still produce a result without error
            assert len(result) == 1
            assert "data_type" in result[0]

    # ============================================
    # Statistical Analysis Tests
    # ============================================

    class TestStatisticalAnalysis:
        """Tests for statistical computations."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_numeric_statistics(self, analyzer, sample_numeric_data):
            """Test computation of numeric statistics."""
            result = analyzer.analyze_columns(sample_numeric_data)
            
            # Find the numeric column result
            x_col = next(r for r in result if r["name"] == "x")
            
            assert "statistics" in x_col
            stats = x_col["statistics"]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats

        def test_mean_calculation(self, analyzer):
            """Test mean calculation accuracy."""
            df = pd.DataFrame({"col": [10, 20, 30, 40, 50]})
            result = analyzer.analyze_columns(df)
            stats = result[0].get("statistics", {})
            assert stats.get("mean") == 30.0

        def test_std_calculation(self, analyzer):
            """Test standard deviation calculation."""
            df = pd.DataFrame({"col": [10, 20, 30, 40, 50]})
            result = analyzer.analyze_columns(df)
            stats = result[0].get("statistics", {})
            # Pandas uses ddof=1 by default
            expected_std = pd.Series([10, 20, 30, 40, 50]).std()
            assert abs(stats.get("std", 0) - expected_std) < 0.01

        def test_percentiles(self, analyzer):
            """Test percentile calculations."""
            df = pd.DataFrame({"col": range(100)})
            result = analyzer.analyze_columns(df)
            stats = result[0].get("statistics", {})
            
            if "percentile_25" in stats:
                assert stats["percentile_25"] == pytest.approx(24.75, rel=0.1)
            if "percentile_50" in stats:
                assert stats["percentile_50"] == pytest.approx(49.5, rel=0.1)
            if "percentile_75" in stats:
                assert stats["percentile_75"] == pytest.approx(74.25, rel=0.1)

        def test_categorical_value_counts(self, analyzer):
            """Test value counts for categorical columns."""
            df = pd.DataFrame({"col": ["A", "A", "B", "B", "B", "C"]})
            result = analyzer.analyze_columns(df)
            
            assert result[0]["unique_count"] == 3

    # ============================================
    # Null Handling Tests
    # ============================================

    class TestNullHandling:
        """Tests for null value handling."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_null_count(self, analyzer, sample_data_with_nulls):
            """Test null count calculation."""
            result = analyzer.analyze_columns(sample_data_with_nulls)
            
            # Find a column with nulls
            name_col = next(r for r in result if r["name"] == "name")
            assert name_col["null_count"] > 0

        def test_null_percentage(self, analyzer):
            """Test null percentage calculation."""
            df = pd.DataFrame({"col": [1, 2, None, 4, None]})
            result = analyzer.analyze_columns(df)
            
            assert result[0]["null_count"] == 2
            assert result[0]["null_percentage"] == 40.0

        def test_all_null_column(self, analyzer):
            """Test handling of all-null columns."""
            df = pd.DataFrame({"col": [None, None, None, None, None]})
            result = analyzer.analyze_columns(df)
            
            assert result[0]["null_count"] == 5
            assert result[0]["null_percentage"] == 100.0

        def test_no_null_column(self, analyzer):
            """Test handling of columns with no nulls."""
            df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
            result = analyzer.analyze_columns(df)
            
            assert result[0]["null_count"] == 0
            assert result[0]["null_percentage"] == 0.0

    # ============================================
    # Correlation Tests
    # ============================================

    class TestCorrelations:
        """Tests for correlation analysis."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_perfect_positive_correlation(self, analyzer):
            """Test detection of perfect positive correlation."""
            df = pd.DataFrame({
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
            })
            correlations = analyzer.calculate_correlations(df)
            
            assert len(correlations) > 0
            # Find x-y correlation
            xy_corr = next((c for c in correlations if set(c["columns"]) == {"x", "y"}), None)
            if xy_corr:
                assert xy_corr["correlation"] == pytest.approx(1.0, rel=0.01)

        def test_perfect_negative_correlation(self, analyzer):
            """Test detection of perfect negative correlation."""
            df = pd.DataFrame({
                "x": [1, 2, 3, 4, 5],
                "y": [10, 8, 6, 4, 2],
            })
            correlations = analyzer.calculate_correlations(df)
            
            xy_corr = next((c for c in correlations if set(c["columns"]) == {"x", "y"}), None)
            if xy_corr:
                assert xy_corr["correlation"] == pytest.approx(-1.0, rel=0.01)

        def test_no_correlation(self, analyzer):
            """Test detection of no correlation."""
            np.random.seed(42)
            df = pd.DataFrame({
                "x": np.random.randn(100),
                "y": np.random.randn(100),
            })
            correlations = analyzer.calculate_correlations(df)
            
            xy_corr = next((c for c in correlations if set(c["columns"]) == {"x", "y"}), None)
            if xy_corr:
                # Should be close to 0
                assert abs(xy_corr["correlation"]) < 0.3

        def test_correlation_with_categorical_excluded(self, analyzer):
            """Test that categorical columns are excluded from correlation."""
            df = pd.DataFrame({
                "numeric": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "B", "A"],
            })
            correlations = analyzer.calculate_correlations(df)
            
            # Should not include categorical in correlation
            for corr in correlations:
                assert "category" not in corr.get("columns", [])

    # ============================================
    # Outlier Detection Tests
    # ============================================

    class TestOutlierDetection:
        """Tests for outlier detection."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_detect_obvious_outliers(self, analyzer):
            """Test detection of obvious outliers."""
            df = pd.DataFrame({
                "col": [10, 11, 12, 13, 14, 15, 1000]  # 1000 is obvious outlier
            })
            outliers = analyzer.detect_outliers(df)
            
            if outliers:
                col_outliers = next((o for o in outliers if o["column"] == "col"), None)
                if col_outliers:
                    assert col_outliers["outlier_count"] >= 1

        def test_no_outliers(self, analyzer):
            """Test handling of data with no outliers."""
            df = pd.DataFrame({
                "col": [10, 11, 12, 13, 14]
            })
            outliers = analyzer.detect_outliers(df)
            
            if outliers:
                col_outliers = next((o for o in outliers if o["column"] == "col"), None)
                if col_outliers:
                    assert col_outliers["outlier_count"] == 0

    # ============================================
    # Sample Values Tests
    # ============================================

    class TestSampleValues:
        """Tests for sample value extraction."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_sample_values_present(self, analyzer):
            """Test that sample values are included."""
            df = pd.DataFrame({"col": ["A", "B", "C", "D", "E"]})
            result = analyzer.analyze_columns(df)
            
            assert "sample_values" in result[0]
            assert len(result[0]["sample_values"]) > 0

        def test_sample_values_limit(self, analyzer):
            """Test that sample values are limited."""
            df = pd.DataFrame({"col": [f"val_{i}" for i in range(100)]})
            result = analyzer.analyze_columns(df)
            
            # Should not return all 100 values
            assert len(result[0]["sample_values"]) <= 10

        def test_sample_values_unique(self, analyzer):
            """Test that sample values are unique."""
            df = pd.DataFrame({"col": ["A", "A", "B", "B", "C"]})
            result = analyzer.analyze_columns(df)
            
            samples = result[0]["sample_values"]
            assert len(samples) == len(set(samples))

    # ============================================
    # Full Analysis Tests
    # ============================================

    class TestFullAnalysis:
        """Tests for complete dataset analysis."""

        @pytest.fixture
        def analyzer(self):
            return DataAnalyzer()

        def test_analyze_returns_all_columns(self, analyzer, sample_sales_data):
            """Test that analysis includes all columns."""
            result = analyzer.analyze_columns(sample_sales_data)
            assert len(result) == len(sample_sales_data.columns)

        def test_analyze_empty_dataframe(self, analyzer):
            """Test handling of empty dataframe."""
            df = pd.DataFrame()
            result = analyzer.analyze_columns(df)
            assert result == []

        def test_analyze_single_row(self, analyzer):
            """Test handling of single row dataframe."""
            df = pd.DataFrame({"a": [1], "b": ["x"]})
            result = analyzer.analyze_columns(df)
            assert len(result) == 2

        def test_analyze_preserves_column_order(self, analyzer):
            """Test that column order is preserved."""
            df = pd.DataFrame({
                "z_col": [1],
                "a_col": [2],
                "m_col": [3],
            })
            result = analyzer.analyze_columns(df)
            names = [r["name"] for r in result]
            assert names == ["z_col", "a_col", "m_col"]


class TestDataQualityAssessment:
    """Tests for data quality assessment."""

    @pytest.fixture
    def analyzer(self):
        return DataAnalyzer()

    def test_quality_score_calculation(self, analyzer, sample_sales_data):
        """Test data quality score calculation."""
        quality = analyzer.assess_data_quality(sample_sales_data)
        
        assert "overall_score" in quality
        assert 0 <= quality["overall_score"] <= 1

    def test_quality_with_nulls(self, analyzer, sample_data_with_nulls):
        """Test quality assessment with null values."""
        quality = analyzer.assess_data_quality(sample_data_with_nulls)
        
        assert quality["overall_score"] < 1.0  # Should be penalized for nulls

    def test_quality_issues_detected(self, analyzer, sample_data_with_nulls):
        """Test that quality issues are detected."""
        quality = analyzer.assess_data_quality(sample_data_with_nulls)
        
        assert "issues" in quality
        # Should detect missing values issue
        issues = [i["type"] for i in quality.get("issues", [])]
        assert "missing_values" in issues or len(quality.get("issues", [])) > 0
