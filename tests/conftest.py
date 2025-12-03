"""
Pytest fixtures and configuration for InsightBoost tests.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import tempfile
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from insightboost.config.settings import Settings
from insightboost.models.dataset import ColumnInfo, Dataset, DatasetMetadata
from insightboost.models.insight import Insight, InsightType
from insightboost.models.visualization import (
    ChartType,
    GeneratedVisualization,
    VisualizationConfig,
)

# ============================================
# Configuration Fixtures
# ============================================


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with mock API key."""
    return Settings(
        anthropic_api_key="test-api-key-12345",
        debug=True,
        log_level="DEBUG",
        max_file_size_mb=10,
        rate_limit_requests_per_minute=100,
        rate_limit_tokens_per_minute=100000,
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================
# Sample Data Fixtures
# ============================================


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Sample sales dataset for testing."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "product": ["Widget A", "Widget B", "Gadget X", "Gadget Y"] * 25,
            "region": ["North", "South", "East", "West"] * 25,
            "units_sold": [
                int(x) for x in (50 + 30 * np.random.randn(100)).clip(1, 200)
            ],
            "revenue": [
                float(x) for x in (1000 + 500 * np.random.randn(100)).clip(100, 5000)
            ],
            "cost": [
                float(x) for x in (500 + 200 * np.random.randn(100)).clip(50, 2000)
            ],
        }
    )


@pytest.fixture
def sample_numeric_data() -> pd.DataFrame:
    """Sample numeric-heavy dataset for testing."""
    n = 200
    return pd.DataFrame(
        {
            "x": range(n),
            "y": [x * 2 + 10 + (5 * (i % 10)) for i, x in enumerate(range(n))],
            "z": [x**0.5 * 10 for x in range(n)],
            "category": ["A", "B", "C", "D"] * 50,
            "value": [100 + i * 0.5 for i in range(n)],
        }
    )


@pytest.fixture
def sample_categorical_data() -> pd.DataFrame:
    """Sample categorical-heavy dataset for testing."""
    return pd.DataFrame(
        {
            "category": ["Electronics", "Clothing", "Food", "Books", "Home"] * 20,
            "subcategory": ["Sub1", "Sub2", "Sub3", "Sub4", "Sub5"] * 20,
            "status": ["Active", "Inactive", "Pending"] * 33 + ["Active"],
            "priority": ["High", "Medium", "Low"] * 33 + ["High"],
            "count": range(100),
        }
    )


@pytest.fixture
def sample_timeseries_data() -> pd.DataFrame:
    """Sample time series dataset for testing."""
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": [
                20 + 10 * np.sin(i / 30) + np.random.randn() for i in range(365)
            ],
            "humidity": [
                60 + 20 * np.cos(i / 30) + np.random.randn() * 5 for i in range(365)
            ],
            "pressure": [1013 + np.random.randn() * 10 for _ in range(365)],
            "sensor_id": ["S001", "S002", "S003"] * 121 + ["S001", "S002"],
        }
    )


@pytest.fixture
def sample_data_with_nulls() -> pd.DataFrame:
    """Sample dataset with null values for testing."""
    df = pd.DataFrame(
        {
            "id": range(50),
            "name": [f"Item {i}" if i % 5 != 0 else None for i in range(50)],
            "value": [float(i * 10) if i % 7 != 0 else None for i in range(50)],
            "category": ["A", "B", None, "C", "D"] * 10,
            "score": [i / 10 if i % 3 != 0 else None for i in range(50)],
        }
    )
    return df


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_sales_data: pd.DataFrame) -> Path:
    """Create a sample CSV file for testing."""
    file_path = temp_dir / "test_sales.csv"
    sample_sales_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_excel_file(temp_dir: Path, sample_sales_data: pd.DataFrame) -> Path:
    """Create a sample Excel file for testing."""
    file_path = temp_dir / "test_sales.xlsx"
    sample_sales_data.to_excel(file_path, index=False)
    return file_path


@pytest.fixture
def sample_json_file(temp_dir: Path, sample_sales_data: pd.DataFrame) -> Path:
    """Create a sample JSON file for testing."""
    file_path = temp_dir / "test_sales.json"
    sample_sales_data.to_json(file_path, orient="records")
    return file_path


# ============================================
# Model Fixtures
# ============================================


@pytest.fixture
def sample_column_info() -> list[ColumnInfo]:
    """Sample column information for testing."""
    return [
        ColumnInfo(
            name="date",
            data_type="datetime",
            sample_values=["2024-01-01", "2024-01-02", "2024-01-03"],
            null_count=0,
            null_percentage=0.0,
            unique_count=100,
            unique_percentage=100.0,
        ),
        ColumnInfo(
            name="revenue",
            data_type="numeric",
            sample_values=[1000.0, 1500.0, 2000.0],
            null_count=0,
            null_percentage=0.0,
            unique_count=95,
            unique_percentage=95.0,
            statistics={"mean": 1500.0, "std": 500.0, "min": 100.0, "max": 5000.0},
        ),
        ColumnInfo(
            name="product",
            data_type="categorical",
            sample_values=["Widget A", "Widget B", "Gadget X"],
            null_count=0,
            null_percentage=0.0,
            unique_count=4,
            unique_percentage=4.0,
        ),
    ]


@pytest.fixture
def sample_dataset(sample_column_info: list[ColumnInfo]) -> Dataset:
    """Sample dataset model for testing."""
    return Dataset(
        id=uuid4(),
        name="Test Sales Data",
        description="Sample sales dataset for testing",
        file_path="/tmp/test_sales.csv",
        file_format=".csv",
        file_size=1024,
        row_count=100,
        column_count=6,
        columns=sample_column_info,
        metadata=DatasetMetadata(
            file_format=".csv",
            encoding="utf-8",
            has_header=True,
        ),
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_insight() -> Insight:
    """Sample insight model for testing."""
    return Insight(
        id=uuid4(),
        dataset_id=uuid4(),
        insight_type=InsightType.TREND,
        title="Revenue Growth Trend",
        description="Revenue shows a positive growth trend of 5% month-over-month",
        confidence=0.85,
        columns_involved=["date", "revenue"],
        statistical_evidence={"slope": 0.05, "r_squared": 0.78},
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_visualization() -> GeneratedVisualization:
    """Sample visualization model for testing."""
    return GeneratedVisualization(
        id=uuid4(),
        dataset_id=uuid4(),
        chart_type=ChartType.LINE,
        title="Revenue Over Time",
        description="Line chart showing revenue trends",
        config=VisualizationConfig(
            chart_type=ChartType.LINE,
            x_column="date",
            y_column="revenue",
            title="Revenue Over Time",
        ),
        figure_json={
            "data": [{"type": "scatter", "mode": "lines", "x": [], "y": []}],
            "layout": {"title": "Revenue Over Time"},
        },
        created_at=datetime.utcnow(),
    )


# ============================================
# Mock API Fixtures
# ============================================


@pytest.fixture
def mock_anthropic_response() -> dict[str, Any]:
    """Mock Anthropic API response."""
    return {
        "id": "msg_01234567890",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "insights": [
                            {
                                "type": "trend",
                                "title": "Revenue Growth",
                                "description": "Revenue is increasing over time",
                                "confidence": 0.85,
                                "columns": ["date", "revenue"],
                            }
                        ],
                        "patterns": [
                            {
                                "type": "correlation",
                                "description": "Strong positive correlation between units_sold and revenue",
                                "strength": 0.92,
                                "columns": ["units_sold", "revenue"],
                            }
                        ],
                    }
                ),
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 500,
            "output_tokens": 200,
        },
    }


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response: dict[str, Any]):
    """Mock Anthropic client for testing."""
    with patch("anthropic.Anthropic") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        # Mock messages.create
        mock_message = MagicMock()
        mock_message.id = mock_anthropic_response["id"]
        mock_message.content = [
            MagicMock(text=mock_anthropic_response["content"][0]["text"])
        ]
        mock_message.usage = MagicMock(
            input_tokens=mock_anthropic_response["usage"]["input_tokens"],
            output_tokens=mock_anthropic_response["usage"]["output_tokens"],
        )
        mock_message.stop_reason = mock_anthropic_response["stop_reason"]

        mock_instance.messages.create.return_value = mock_message

        yield mock_instance


# ============================================
# Flask App Fixtures
# ============================================


@pytest.fixture
def flask_app(test_settings: Settings):
    """Create Flask test application."""
    from insightboost.web.app import create_app

    app = create_app(
        config_override={
            "TESTING": True,
            "DEBUG": True,
        }
    )
    app.config["SETTINGS"] = test_settings

    return app


@pytest.fixture
def flask_client(flask_app):
    """Create Flask test client."""
    return flask_app.test_client()


@pytest.fixture
def flask_app_context(flask_app):
    """Create Flask application context."""
    with flask_app.app_context():
        yield flask_app


# ============================================
# Utility Fixtures
# ============================================


@pytest.fixture
def mock_uuid():
    """Generate predictable UUIDs for testing."""
    counter = [0]

    def _generate():
        counter[0] += 1
        return uuid4()

    return _generate


@pytest.fixture
def capture_logs(caplog):
    """Capture log output for testing."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


# ============================================
# Pytest Configuration
# ============================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Add markers based on test location
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
