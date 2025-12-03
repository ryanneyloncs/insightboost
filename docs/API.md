# InsightBoost API Reference

Complete reference documentation for the InsightBoost REST API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication for local deployments. For production deployments, configure authentication via environment variables.

## Response Format

All API responses follow a consistent JSON format:

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

Error responses:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  }
}
```

## HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource does not exist |
| 413 | Payload Too Large - File exceeds size limit |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

---

## Datasets

### Upload Dataset

Upload a new dataset file for analysis.

```
POST /api/v1/datasets
```

**Request:**
- Content-Type: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | Dataset file (CSV, XLSX, XLS, JSON, Parquet) |
| name | String | No | Dataset name (defaults to filename) |
| description | String | No | Dataset description |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -F "file=@sales_data.csv" \
  -F "name=Q4 Sales Data" \
  -F "description=Sales data for Q4 2024"
```

**Response:**

```json
{
  "success": true,
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "dataset": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Q4 Sales Data",
    "description": "Sales data for Q4 2024",
    "file_format": ".csv",
    "file_size": 102400,
    "row_count": 1000,
    "column_count": 8,
    "columns": [
      {
        "name": "date",
        "data_type": "datetime",
        "null_count": 0,
        "null_percentage": 0.0,
        "unique_count": 100
      }
    ],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### List Datasets

Retrieve all uploaded datasets.

```
GET /api/v1/datasets
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | Integer | 1 | Page number |
| per_page | Integer | 20 | Results per page (max 100) |
| sort_by | String | created_at | Sort field |
| sort_order | String | desc | Sort order (asc/desc) |

**Example:**

```bash
curl http://localhost:8000/api/v1/datasets?page=1&per_page=10
```

**Response:**

```json
{
  "success": true,
  "datasets": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Q4 Sales Data",
      "row_count": 1000,
      "column_count": 8,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total_pages": 1,
    "total_rows": 1
  }
}
```

### Get Dataset

Retrieve details for a specific dataset.

```
GET /api/v1/datasets/{dataset_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| dataset_id | UUID | Dataset identifier |

**Example:**

```bash
curl http://localhost:8000/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000
```

**Response:**

```json
{
  "success": true,
  "dataset": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Q4 Sales Data",
    "description": "Sales data for Q4 2024",
    "file_path": "/data/uploads/sales_data.csv",
    "file_format": ".csv",
    "file_size": 102400,
    "row_count": 1000,
    "column_count": 8,
    "columns": [
      {
        "name": "date",
        "data_type": "datetime",
        "null_count": 0,
        "null_percentage": 0.0,
        "unique_count": 100,
        "unique_percentage": 10.0,
        "sample_values": ["2024-01-01", "2024-01-02", "2024-01-03"]
      },
      {
        "name": "revenue",
        "data_type": "numeric",
        "null_count": 5,
        "null_percentage": 0.5,
        "unique_count": 850,
        "statistics": {
          "mean": 1500.50,
          "std": 450.25,
          "min": 100.00,
          "max": 5000.00,
          "percentile_25": 1100.00,
          "percentile_50": 1450.00,
          "percentile_75": 1900.00
        }
      }
    ],
    "metadata": {
      "encoding": "utf-8",
      "has_header": true
    },
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Get Dataset Data

Retrieve the actual data rows from a dataset.

```
GET /api/v1/datasets/{dataset_id}/data
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | Integer | 1 | Page number |
| per_page | Integer | 50 | Rows per page (max 1000) |
| columns | String | all | Comma-separated column names |
| sort_by | String | - | Column to sort by |
| sort_order | String | asc | Sort order |

**Example:**

```bash
curl "http://localhost:8000/api/v1/datasets/550e8400.../data?page=1&per_page=100"
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "date": "2024-01-01",
      "product": "Widget A",
      "revenue": 1250.00,
      "units_sold": 50
    }
  ],
  "columns": ["date", "product", "revenue", "units_sold"],
  "pagination": {
    "page": 1,
    "per_page": 100,
    "total_pages": 10,
    "total_rows": 1000
  }
}
```

### Delete Dataset

Delete a dataset and all associated insights and visualizations.

```
DELETE /api/v1/datasets/{dataset_id}
```

**Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000
```

**Response:**

```json
{
  "success": true,
  "message": "Dataset deleted successfully"
}
```

---

## Insights

### Generate Insights

Generate AI-powered insights for a dataset.

```
POST /api/v1/datasets/{dataset_id}/insights
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | String | No | Specific question or focus area |
| max_results | Integer | No | Maximum insights to generate (default: 10) |
| insight_types | Array | No | Filter by insight types |

**Insight Types:**
- `trend` - Time-based patterns
- `correlation` - Relationships between variables
- `anomaly` - Outliers and unusual values
- `distribution` - Data distribution patterns
- `comparison` - Group comparisons
- `summary` - Statistical summaries

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/datasets/550e8400.../insights \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main revenue trends by region?",
    "max_results": 5,
    "insight_types": ["trend", "correlation"]
  }'
```

**Response:**

```json
{
  "success": true,
  "insights": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
      "insight_type": "trend",
      "title": "Strong Revenue Growth in North Region",
      "description": "The North region shows consistent month-over-month revenue growth of 8.5%, outperforming other regions by a significant margin.",
      "confidence": 0.92,
      "importance": "high",
      "columns_involved": ["date", "revenue", "region"],
      "statistical_evidence": {
        "growth_rate": 0.085,
        "r_squared": 0.89,
        "p_value": 0.001
      },
      "created_at": "2024-01-15T10:35:00Z"
    }
  ],
  "query": "What are the main revenue trends by region?",
  "token_usage": {
    "input_tokens": 1500,
    "output_tokens": 800
  }
}
```

### List Insights

Retrieve all insights for a dataset.

```
GET /api/v1/datasets/{dataset_id}/insights
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| type | String | all | Filter by insight type |
| min_confidence | Float | 0.0 | Minimum confidence score |
| sort_by | String | confidence | Sort field |

**Example:**

```bash
curl "http://localhost:8000/api/v1/datasets/550e8400.../insights?min_confidence=0.8"
```

### Quick Analyze

Perform a quick automated analysis of a dataset.

```
GET /api/v1/datasets/{dataset_id}/quick-analyze
```

**Example:**

```bash
curl http://localhost:8000/api/v1/datasets/550e8400.../quick-analyze
```

**Response:**

```json
{
  "success": true,
  "summary": {
    "row_count": 1000,
    "column_count": 8,
    "numeric_columns": 4,
    "categorical_columns": 3,
    "datetime_columns": 1,
    "missing_values_total": 15,
    "missing_values_percentage": 0.19
  },
  "data_quality": {
    "overall_score": 0.95,
    "issues": [
      {
        "type": "missing_values",
        "column": "revenue",
        "count": 5,
        "severity": "low"
      }
    ]
  },
  "correlations": [
    {
      "columns": ["units_sold", "revenue"],
      "correlation": 0.94,
      "strength": "strong_positive"
    }
  ],
  "top_insights": [
    {
      "title": "Strong correlation between units sold and revenue",
      "confidence": 0.95
    }
  ]
}
```

---

## Visualizations

### Get Visualization Suggestions

Get AI-powered visualization recommendations.

```
GET /api/v1/datasets/{dataset_id}/visualizations/suggest
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_suggestions | Integer | 5 | Maximum suggestions |
| objective | String | all | Focus area (trend/comparison/distribution/correlation) |

**Example:**

```bash
curl "http://localhost:8000/api/v1/datasets/550e8400.../visualizations/suggest?max_suggestions=3"
```

**Response:**

```json
{
  "success": true,
  "suggestions": [
    {
      "chart_type": "line",
      "title": "Revenue Trend Over Time",
      "description": "Line chart showing how revenue changes over the date range",
      "x_column": "date",
      "y_column": "revenue",
      "confidence": 0.95,
      "rationale": "Time series data with clear temporal progression"
    },
    {
      "chart_type": "bar",
      "title": "Revenue by Region",
      "description": "Bar chart comparing total revenue across regions",
      "x_column": "region",
      "y_column": "revenue",
      "aggregation": "sum",
      "confidence": 0.88,
      "rationale": "Categorical comparison with numeric values"
    }
  ]
}
```

### Create Visualization

Create a new visualization.

```
POST /api/v1/datasets/{dataset_id}/visualizations
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| config | Object | Yes | Visualization configuration |

**Configuration Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| chart_type | String | Yes | Type of chart |
| x_column | String | Yes* | X-axis column |
| y_column | String | Yes* | Y-axis column |
| color_column | String | No | Column for color grouping |
| size_column | String | No | Column for size (bubble charts) |
| title | String | No | Chart title |
| aggregation | String | No | Aggregation function (sum/mean/count/min/max) |

*Required for most chart types

**Chart Types:**
- `line` - Line chart
- `bar` - Bar chart
- `scatter` - Scatter plot
- `histogram` - Histogram
- `box` - Box plot
- `violin` - Violin plot
- `heatmap` - Heatmap
- `pie` - Pie chart
- `area` - Area chart
- `bubble` - Bubble chart

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/datasets/550e8400.../visualizations \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "chart_type": "line",
      "x_column": "date",
      "y_column": "revenue",
      "color_column": "region",
      "title": "Revenue Trends by Region"
    }
  }'
```

**Response:**

```json
{
  "success": true,
  "visualization": {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "chart_type": "line",
    "title": "Revenue Trends by Region",
    "config": {
      "chart_type": "line",
      "x_column": "date",
      "y_column": "revenue",
      "color_column": "region"
    },
    "figure_json": {
      "data": [...],
      "layout": {...}
    },
    "created_at": "2024-01-15T10:40:00Z"
  }
}
```

### List Visualizations

Retrieve all visualizations for a dataset.

```
GET /api/v1/datasets/{dataset_id}/visualizations
```

### Get Chart Types

Get available chart types and their requirements.

```
GET /api/v1/visualizations/chart-types
```

**Response:**

```json
{
  "success": true,
  "chart_types": [
    {
      "type": "line",
      "name": "Line Chart",
      "description": "Shows trends over continuous data",
      "required_columns": {
        "x": ["datetime", "numeric"],
        "y": ["numeric"]
      },
      "optional_columns": ["color", "size"]
    }
  ]
}
```

---

## Collaboration

### Create Session

Create a new collaboration session.

```
POST /api/v1/sessions
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| dataset_id | UUID | Yes | Dataset to collaborate on |
| name | String | Yes | Session name |
| user_id | String | Yes | Creator's user ID |
| settings | Object | No | Session settings |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Q4 Sales Analysis Session",
    "user_id": "user-123"
  }'
```

**Response:**

```json
{
  "success": true,
  "session": {
    "id": "880e8400-e29b-41d4-a716-446655440003",
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Q4 Sales Analysis Session",
    "owner_id": "user-123",
    "participants": ["user-123"],
    "status": "active",
    "created_at": "2024-01-15T10:45:00Z"
  }
}
```

### List Sessions

Retrieve all collaboration sessions.

```
GET /api/v1/sessions
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| status | String | all | Filter by status (active/ended) |
| user_id | String | - | Filter by participant |

### Get Session

Retrieve details for a specific session.

```
GET /api/v1/sessions/{session_id}
```

### Join Session

Join an existing collaboration session.

```
POST /api/v1/sessions/{session_id}/join
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | String | Yes | User ID joining the session |

### Add Comment

Add a comment to a session.

```
POST /api/v1/sessions/{session_id}/comments
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | String | Yes | Comment author |
| content | String | Yes | Comment text |
| insight_id | UUID | No | Related insight |
| visualization_id | UUID | No | Related visualization |
| parent_id | UUID | No | Parent comment (for replies) |

### Create Snapshot

Save a snapshot of the current session state.

```
POST /api/v1/sessions/{session_id}/snapshots
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | String | Yes | Snapshot creator |
| name | String | Yes | Snapshot name |
| description | String | No | Snapshot description |
| insight_ids | Array | No | Insights to include |
| visualization_ids | Array | No | Visualizations to include |

---

## WebSocket Events

For real-time collaboration, connect to the WebSocket endpoint:

```
ws://localhost:8000/socket.io
```

### Events

**Client to Server:**
- `session.join` - Join a session
- `session.leave` - Leave a session
- `cursor.move` - Update cursor position
- `insight.share` - Share an insight
- `visualization.share` - Share a visualization
- `comment.add` - Add a comment

**Server to Client:**
- `session.user_joined` - User joined session
- `session.user_left` - User left session
- `cursor.moved` - Cursor position update
- `insight.created` - New insight shared
- `visualization.shared` - New visualization shared
- `comment.added` - New comment added
- `session.ended` - Session ended

---

## Rate Limiting

The API implements rate limiting to ensure fair usage:

| Limit Type | Default | Description |
|------------|---------|-------------|
| Requests per minute | 60 | Total API requests |
| Tokens per minute | 100,000 | Claude API tokens |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1705312800
```

When rate limited, the API returns a 429 status code:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please retry after 60 seconds.",
    "retry_after": 60
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| INVALID_REQUEST | Request validation failed |
| DATASET_NOT_FOUND | Dataset does not exist |
| SESSION_NOT_FOUND | Session does not exist |
| UNSUPPORTED_FORMAT | File format not supported |
| FILE_TOO_LARGE | File exceeds size limit |
| ANALYSIS_FAILED | AI analysis failed |
| RATE_LIMIT_EXCEEDED | Rate limit exceeded |
| INTERNAL_ERROR | Internal server error |

---

## SDKs and Client Libraries

Official client libraries:

- Python: `pip install insightboost-client`
- JavaScript/TypeScript: `npm install @insightboost/client`

Example using Python client:

```python
from insightboost_client import InsightBoostClient

client = InsightBoostClient(base_url="http://localhost:8000")

# Upload dataset
dataset = client.datasets.upload("sales.csv", name="Sales Data")

# Generate insights
insights = client.insights.generate(
    dataset_id=dataset.id,
    query="What are the main trends?"
)

# Create visualization
viz = client.visualizations.create(
    dataset_id=dataset.id,
    chart_type="line",
    x_column="date",
    y_column="revenue"
)
```
