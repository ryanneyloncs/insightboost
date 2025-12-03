# InsightBoost

AI-powered data insights platform that transforms raw data into actionable intelligence using Claude AI.

## Overview

InsightBoost is a comprehensive data analysis platform that leverages Anthropic's Claude AI to automatically discover patterns, trends, anomalies, and correlations in your datasets. It provides intelligent visualization suggestions and supports real-time collaboration for team-based analysis.

### Key Features

- **AI-Powered Analysis**: Automatic insight generation using Claude AI
- **Smart Visualizations**: Intelligent chart recommendations based on data characteristics
- **Multiple Data Formats**: Support for CSV, Excel, JSON, and Parquet files
- **Real-Time Collaboration**: Share sessions and collaborate with team members
- **RESTful API**: Full API access for integration with other tools
- **Export Options**: Download insights and visualizations in multiple formats

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Development](#development)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Contributing](#contributing)
11. [License](#license)

## Requirements

- Python 3.10 or higher
- Anthropic API key
- Redis (optional, for caching and sessions)

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/ryanneyloncs/insightboost.git
cd insightboost

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Using Docker

```bash
# Clone the repository
git clone https://github.com/ryanneyloncs/insightboost.git
cd insightboost

# Build and run with Docker Compose
docker-compose up -d
```

## Quick Start

### 1. Set up environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Anthropic API key
ANTHROPIC_API_KEY=your-api-key-here
```

### 2. Start the application

```bash
# Development mode
flask run --debug

# Production mode
gunicorn 'insightboost.web.app:create_app()' --bind 0.0.0.0:8000
```

### 3. Access the web interface

Open your browser and navigate to `http://localhost:8000`

### 4. Upload your first dataset

1. Click "Upload Dataset" on the dashboard
2. Select a CSV, Excel, or JSON file
3. Wait for automatic analysis to complete
4. Explore generated insights and visualizations

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `ANTHROPIC_MODEL` | Claude model to use | `claude-sonnet-4-20250514` |
| `FLASK_ENV` | Environment mode | `production` |
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_FILE_SIZE_MB` | Maximum upload file size | `50` |
| `REDIS_URL` | Redis connection URL | `None` |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | API rate limit | `60` |
| `RATE_LIMIT_TOKENS_PER_MINUTE` | Token rate limit | `100000` |

### Configuration File

You can also configure InsightBoost using a YAML configuration file:

```yaml
# config.yaml
anthropic:
  model: claude-sonnet-4-20250514
  max_tokens: 4096
  temperature: 0.7

analysis:
  max_rows_preview: 1000
  correlation_threshold: 0.7
  outlier_method: iqr

visualization:
  default_chart_height: 400
  color_palette: plotly
```

## Usage

### Web Interface

The web interface provides an intuitive way to:

- Upload and manage datasets
- View automatically generated insights
- Create and customize visualizations
- Collaborate with team members in real-time
- Export results in various formats

### Command Line

```bash
# Analyze a dataset
insightboost analyze data.csv --output results.json

# Generate visualizations
insightboost visualize data.csv --charts all --output charts/

# Start interactive session
insightboost interactive data.csv
```

### Python API

```python
from insightboost import InsightBoost

# Initialize
ib = InsightBoost(api_key="your-api-key")

# Load dataset
dataset = ib.load_dataset("sales_data.csv")

# Generate insights
insights = ib.generate_insights(
    dataset,
    query="What are the main revenue trends?",
    max_results=10
)

# Print insights
for insight in insights:
    print(f"[{insight.confidence:.0%}] {insight.title}")
    print(f"  {insight.description}\n")

# Get visualization suggestions
suggestions = ib.suggest_visualizations(dataset)

# Create a specific visualization
chart = ib.create_visualization(
    dataset,
    chart_type="line",
    x_column="date",
    y_column="revenue"
)

# Export
chart.save("revenue_trend.html")
```

## API Reference

### Datasets

```
POST   /api/v1/datasets              Upload a new dataset
GET    /api/v1/datasets              List all datasets
GET    /api/v1/datasets/{id}         Get dataset details
GET    /api/v1/datasets/{id}/data    Get dataset data (paginated)
DELETE /api/v1/datasets/{id}         Delete a dataset
```

### Insights

```
POST   /api/v1/datasets/{id}/insights       Generate insights
GET    /api/v1/datasets/{id}/insights       List insights for dataset
GET    /api/v1/datasets/{id}/quick-analyze  Quick analysis
```

### Visualizations

```
GET    /api/v1/datasets/{id}/visualizations/suggest  Get suggestions
POST   /api/v1/datasets/{id}/visualizations          Create visualization
GET    /api/v1/datasets/{id}/visualizations          List visualizations
GET    /api/v1/visualizations/chart-types            Get available chart types
```

### Collaboration

```
POST   /api/v1/sessions                    Create collaboration session
GET    /api/v1/sessions                    List sessions
GET    /api/v1/sessions/{id}               Get session details
POST   /api/v1/sessions/{id}/join          Join a session
POST   /api/v1/sessions/{id}/comments      Add a comment
POST   /api/v1/sessions/{id}/snapshots     Create a snapshot
```

For complete API documentation, see [docs/API.md](docs/API.md).

## Development

### Setting up the development environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run in development mode
flask run --debug
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linting
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
insightboost/
├── src/
│   └── insightboost/
│       ├── api/              # Anthropic API integration
│       ├── config/           # Configuration management
│       ├── core/             # Core analysis engine
│       ├── models/           # Data models
│       ├── utils/            # Utility functions
│       └── web/              # Flask web application
│           ├── routes/       # API route blueprints
│           ├── templates/    # HTML templates
│           └── static/       # CSS and JavaScript
├── tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test data
├── docs/                     # Documentation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Testing

### Running tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=insightboost --cov-report=html

# Run specific test file
pytest tests/unit/test_insights_generator.py -v
```

### Test markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run API tests
pytest -m api

# Run web tests
pytest -m web
```

## Deployment

### Docker Deployment

```bash
# Production deployment
docker-compose up -d app redis

# With custom environment
docker-compose --env-file .env.production up -d
```

### Manual Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn 'insightboost.web.app:create_app()' \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --threads 2 \
    --timeout 120
```

### Environment-specific configuration

- **Development**: Uses Flask development server with debug mode
- **Staging**: Docker deployment with reduced resources
- **Production**: Full Docker deployment with Redis caching

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Quick contribution guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/ryanneyloncs/insightboost/issues)
- Discussions: [GitHub Discussions](https://github.com/ryanneyloncs/insightboost/discussions)

