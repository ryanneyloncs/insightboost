"""
Integration Tests for Web Routes
Tests all API endpoints with Flask test client
"""

import io
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


class TestDatasetsAPI:
    """Integration tests for /api/v1/datasets endpoints."""

    @pytest.fixture
    def client(self, flask_client):
        return flask_client

    @pytest.fixture
    def sample_csv_bytes(self, sample_sales_data):
        """Create CSV bytes for upload testing."""
        buffer = io.BytesIO()
        sample_sales_data.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    # ============================================
    # Dataset Upload Tests
    # ============================================

    def test_upload_csv_dataset(self, client, sample_csv_bytes):
        """Test CSV file upload."""
        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test_data.csv"),
                "name": "Test Dataset",
                "description": "Test description",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code in [200, 201]
        data = response.get_json()
        assert data.get("success") is True
        assert "dataset_id" in data

    def test_upload_without_file(self, client):
        """Test upload without file returns error."""
        response = client.post(
            "/api/v1/datasets",
            data={"name": "Test"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data.get("error") is True

    def test_upload_unsupported_format(self, client):
        """Test upload of unsupported file format."""
        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (io.BytesIO(b"test content"), "test.txt"),
                "name": "Test",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400

    def test_upload_empty_file(self, client):
        """Test upload of empty file."""
        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (io.BytesIO(b""), "empty.csv"),
                "name": "Empty",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400

    # ============================================
    # Dataset List/Get Tests
    # ============================================

    def test_list_datasets_empty(self, client):
        """Test listing datasets when none exist."""
        response = client.get("/api/v1/datasets")

        assert response.status_code == 200
        data = response.get_json()
        assert data.get("success") is True
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_list_datasets_after_upload(self, client, sample_csv_bytes):
        """Test listing datasets after upload."""
        # Upload a dataset first
        client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test.csv"),
                "name": "Test",
            },
            content_type="multipart/form-data",
        )

        response = client.get("/api/v1/datasets")
        data = response.get_json()

        assert len(data["datasets"]) >= 1

    def test_get_dataset_by_id(self, client, sample_csv_bytes):
        """Test getting a specific dataset."""
        # Upload first
        upload_response = client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test.csv"),
                "name": "Test",
            },
            content_type="multipart/form-data",
        )
        dataset_id = upload_response.get_json()["dataset_id"]

        # Get by ID
        response = client.get(f"/api/v1/datasets/{dataset_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["dataset"]["id"] == dataset_id

    def test_get_nonexistent_dataset(self, client):
        """Test getting a dataset that doesn't exist."""
        fake_id = str(uuid4())
        response = client.get(f"/api/v1/datasets/{fake_id}")

        assert response.status_code == 404

    # ============================================
    # Dataset Data Retrieval Tests
    # ============================================

    def test_get_dataset_data(self, client, sample_csv_bytes):
        """Test retrieving dataset data with pagination."""
        # Upload first
        upload_response = client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test.csv"),
                "name": "Test",
            },
            content_type="multipart/form-data",
        )
        dataset_id = upload_response.get_json()["dataset_id"]

        # Get data
        response = client.get(f"/api/v1/datasets/{dataset_id}/data")

        assert response.status_code == 200
        data = response.get_json()
        assert "data" in data
        assert "pagination" in data

    def test_get_dataset_data_pagination(self, client, sample_csv_bytes):
        """Test pagination parameters."""
        # Upload first
        upload_response = client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test.csv"),
                "name": "Test",
            },
            content_type="multipart/form-data",
        )
        dataset_id = upload_response.get_json()["dataset_id"]

        # Get with pagination params
        response = client.get(f"/api/v1/datasets/{dataset_id}/data?page=1&per_page=10")

        data = response.get_json()
        assert len(data["data"]) <= 10
        assert data["pagination"]["per_page"] == 10

    # ============================================
    # Dataset Delete Tests
    # ============================================

    def test_delete_dataset(self, client, sample_csv_bytes):
        """Test deleting a dataset."""
        # Upload first
        upload_response = client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test.csv"),
                "name": "Test",
            },
            content_type="multipart/form-data",
        )
        dataset_id = upload_response.get_json()["dataset_id"]

        # Delete
        response = client.delete(f"/api/v1/datasets/{dataset_id}")

        assert response.status_code == 200

        # Verify deleted
        get_response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_dataset(self, client):
        """Test deleting a dataset that doesn't exist."""
        fake_id = str(uuid4())
        response = client.delete(f"/api/v1/datasets/{fake_id}")

        assert response.status_code == 404


class TestInsightsAPI:
    """Integration tests for insights endpoints."""

    @pytest.fixture
    def client(self, flask_client):
        return flask_client

    @pytest.fixture
    def uploaded_dataset(self, client, sample_sales_data):
        """Upload a dataset for testing."""
        buffer = io.BytesIO()
        sample_sales_data.to_csv(buffer, index=False)
        buffer.seek(0)

        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (buffer, "sales.csv"),
                "name": "Sales Data",
            },
            content_type="multipart/form-data",
        )
        return response.get_json()["dataset_id"]

    def test_generate_insights(self, client, uploaded_dataset):
        """Test insight generation endpoint."""
        with patch("insightboost.web.routes.insights.InsightsGenerator") as mock_gen:
            mock_instance = MagicMock()
            mock_gen.return_value = mock_instance
            mock_instance.generate_insights.return_value = [
                {
                    "id": str(uuid4()),
                    "title": "Test Insight",
                    "description": "Test description",
                    "confidence": 0.85,
                    "insight_type": "trend",
                }
            ]

            response = client.post(
                f"/api/v1/datasets/{uploaded_dataset}/insights",
                json={"query": "What trends exist?", "max_results": 5},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data.get("success") is True

    def test_list_insights(self, client, uploaded_dataset):
        """Test listing insights for a dataset."""
        response = client.get(f"/api/v1/datasets/{uploaded_dataset}/insights")

        assert response.status_code == 200
        data = response.get_json()
        assert "insights" in data

    def test_quick_analyze(self, client, uploaded_dataset):
        """Test quick analysis endpoint."""
        with patch(
            "insightboost.web.routes.insights.get_insights_generator"
        ) as mock_get_gen:
            mock_generator = MagicMock()
            mock_get_gen.return_value = mock_generator
            mock_generator.quick_analyze.return_value = {
                "row_count": 100,
                "column_count": 6,
                "memory_mb": 0.5,
                "columns": [],
                "quality_score": 0.95,
            }

            response = client.get(f"/api/v1/datasets/{uploaded_dataset}/quick-analyze")

            assert response.status_code == 200
            data = response.get_json()
            assert data.get("success") is True


class TestVisualizationsAPI:
    """Integration tests for visualization endpoints."""

    @pytest.fixture
    def client(self, flask_client):
        return flask_client

    @pytest.fixture
    def uploaded_dataset(self, client, sample_sales_data):
        """Upload a dataset for testing."""
        buffer = io.BytesIO()
        sample_sales_data.to_csv(buffer, index=False)
        buffer.seek(0)

        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (buffer, "sales.csv"),
                "name": "Sales Data",
            },
            content_type="multipart/form-data",
        )
        return response.get_json()["dataset_id"]

    def test_get_suggestions(self, client, uploaded_dataset):
        """Test visualization suggestions endpoint."""
        with patch(
            "insightboost.web.routes.visualizations.get_viz_suggester"
        ) as mock_get_sug:
            mock_suggester = MagicMock()
            mock_get_sug.return_value = mock_suggester

            # Create suggestion mock with all required attributes
            suggestion_mock = MagicMock()
            suggestion_mock.id = str(uuid4())
            suggestion_mock.chart_type = MagicMock(value="line")
            suggestion_mock.title = "Revenue Over Time"
            suggestion_mock.description = "Shows revenue trends"
            suggestion_mock.confidence = 0.9
            suggestion_mock.x_column = "date"
            suggestion_mock.y_column = "revenue"
            suggestion_mock.color_column = None
            suggestion_mock.size_column = None
            suggestion_mock.reasoning = "Time series data"
            suggestion_mock.code_snippet = "fig = px.line(df, x='date', y='revenue')"
            suggestion_mock.priority = 1

            # Create config mock
            config_mock = MagicMock()
            config_mock.chart_type = MagicMock(value="line")
            config_mock.x_column = "date"
            config_mock.y_column = "revenue"
            config_mock.color_column = None
            config_mock.size_column = None
            config_mock.title = "Revenue Over Time"
            config_mock.width = 800
            config_mock.height = 600
            suggestion_mock.config = config_mock

            mock_suggester.suggest_visualizations.return_value = [suggestion_mock]

            response = client.get(
                f"/api/v1/datasets/{uploaded_dataset}/visualizations/suggest"
            )

            assert response.status_code == 200
            data = response.get_json()
            assert "suggestions" in data

    def test_create_visualization(self, client, uploaded_dataset):
        """Test creating a visualization."""
        with patch(
            "insightboost.web.routes.visualizations.get_viz_suggester"
        ) as mock_get_sug:
            mock_suggester = MagicMock()
            mock_get_sug.return_value = mock_suggester

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(x=["A", "B"], y=[1, 2]))
            mock_suggester.generate_visualization.return_value = fig

            response = client.post(
                f"/api/v1/datasets/{uploaded_dataset}/visualizations",
                json={
                    "config": {
                        "chart_type": "bar",
                        "x_column": "product",
                        "y_column": "revenue",
                        "title": "Revenue by Product",
                    }
                },
            )

            assert response.status_code in [200, 201]

    def test_list_visualizations(self, client, uploaded_dataset):
        """Test listing visualizations."""
        response = client.get(f"/api/v1/datasets/{uploaded_dataset}/visualizations")

        assert response.status_code == 200
        data = response.get_json()
        assert "visualizations" in data

    def test_get_chart_types(self, client):
        """Test getting available chart types."""
        response = client.get("/api/v1/visualizations/chart-types")

        assert response.status_code == 200
        data = response.get_json()
        assert "chart_types" in data


class TestCollaborationAPI:
    """Integration tests for collaboration endpoints."""

    @pytest.fixture
    def client(self, flask_client):
        return flask_client

    @pytest.fixture
    def uploaded_dataset(self, client, sample_sales_data):
        """Upload a dataset for testing."""
        buffer = io.BytesIO()
        sample_sales_data.to_csv(buffer, index=False)
        buffer.seek(0)

        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (buffer, "sales.csv"),
                "name": "Sales Data",
            },
            content_type="multipart/form-data",
        )
        return response.get_json()["dataset_id"]

    def test_create_session(self, client, uploaded_dataset):
        """Test creating a collaboration session."""
        response = client.post(
            "/api/v1/sessions",
            json={
                "dataset_id": uploaded_dataset,
                "name": "Test Session",
                "user_id": "user-123",
            },
        )

        assert response.status_code in [200, 201]
        data = response.get_json()
        assert data.get("success") is True
        assert "session" in data

    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.get_json()
        assert "sessions" in data

    def test_join_session(self, client, uploaded_dataset):
        """Test joining a session."""
        # Create session first
        create_response = client.post(
            "/api/v1/sessions",
            json={
                "dataset_id": uploaded_dataset,
                "name": "Test Session",
                "user_id": "user-123",
            },
        )
        session_id = create_response.get_json()["session"]["id"]

        # Join session
        response = client.post(
            f"/api/v1/sessions/{session_id}/join",
            json={"user_id": "user-456"},
        )

        assert response.status_code == 200

    def test_add_comment(self, client, uploaded_dataset):
        """Test adding a comment to a session."""
        # Create session first
        create_response = client.post(
            "/api/v1/sessions",
            json={
                "dataset_id": uploaded_dataset,
                "name": "Test Session",
                "user_id": "user-123",
            },
        )
        session_id = create_response.get_json()["session"]["id"]

        # Add comment
        response = client.post(
            f"/api/v1/sessions/{session_id}/comments",
            json={
                "user_id": "user-123",
                "content": "This is a test comment",
            },
        )

        assert response.status_code in [200, 201]

    def test_create_snapshot(self, client, uploaded_dataset):
        """Test creating a session snapshot."""
        # Create session first
        create_response = client.post(
            "/api/v1/sessions",
            json={
                "dataset_id": uploaded_dataset,
                "name": "Test Session",
                "user_id": "user-123",
            },
        )
        session_id = create_response.get_json()["session"]["id"]

        # Create snapshot
        response = client.post(
            f"/api/v1/sessions/{session_id}/snapshots",
            json={
                "user_id": "user-123",
                "name": "Test Snapshot",
                "visualization_ids": [],
                "insight_ids": [],
            },
        )

        assert response.status_code in [200, 201]


class TestHealthAndInfo:
    """Integration tests for health and info endpoints."""

    @pytest.fixture
    def client(self, flask_client):
        return flask_client

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data.get("status") == "healthy"

    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api/v1/info")

        assert response.status_code == 200
        data = response.get_json()
        assert "version" in data or "name" in data


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.fixture
    def client(self, flask_client):
        return flask_client

    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent-endpoint")

        assert response.status_code == 404

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/sessions",
            data="not valid json",
            content_type="application/json",
        )

        assert response.status_code in [400, 415, 500]

    def test_missing_required_field(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/api/v1/sessions",
            json={},  # Missing required fields
        )

        assert response.status_code == 400
