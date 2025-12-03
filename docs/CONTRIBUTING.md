# Contributing to InsightBoost

Thank you for your interest in contributing to InsightBoost. This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)
9. [Release Process](#release-process)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community and project

---

## Getting Started

### Finding Issues

- Browse open issues labeled `good first issue` for beginner-friendly tasks
- Look for issues labeled `help wanted` for tasks where we need assistance
- Check the project roadmap for planned features

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix reported issues
- **Features**: Implement new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code quality without changing behavior

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Docker (optional, for containerized development)

### Setting Up Your Environment

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/insightboost.git
   cd insightboost
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/original-org/insightboost.git
   ```

4. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

6. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

7. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

8. **Verify setup**

   ```bash
   pytest tests/unit/ -v
   ```

---

## Making Changes

### Branch Naming

Create a descriptive branch name:

- `feature/add-export-pdf` - New feature
- `fix/insight-generation-error` - Bug fix
- `docs/api-reference-update` - Documentation
- `refactor/data-analyzer-cleanup` - Code refactoring
- `test/visualization-coverage` - Test improvements

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed. Explain what and why,
not how (the code shows how).

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(insights): add support for correlation analysis

Implement Pearson and Spearman correlation detection
in the insights generator. Correlations above 0.7 are
flagged as significant.

Closes #45
```

```
fix(api): handle empty dataset upload gracefully

Return proper 400 error with message instead of 500
when user uploads an empty file.

Fixes #78
```

### Keep Changes Focused

- One feature or fix per pull request
- Break large changes into smaller, reviewable chunks
- Avoid mixing refactoring with functional changes

---

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications. The project uses these tools:

- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **Ruff**: Linting
- **mypy**: Type checking

### Running Style Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/
```

### Style Guidelines

**Imports:**

```python
# Standard library
import os
from datetime import datetime
from typing import Dict, List, Optional

# Third-party
import pandas as pd
from flask import Flask, request

# Local
from insightboost.core.analyzer import DataAnalyzer
from insightboost.models.dataset import Dataset
```

**Type Hints:**

```python
def analyze_column(
    self,
    column_name: str,
    data: pd.Series,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """Analyze a single column.
    
    Args:
        column_name: Name of the column to analyze.
        data: Pandas Series containing the column data.
        sample_size: Number of sample values to include.
        
    Returns:
        Dictionary containing column analysis results.
        
    Raises:
        ValueError: If column_name is empty.
    """
    ...
```

**Docstrings:**

Use Google-style docstrings:

```python
def generate_insights(
    self,
    df: pd.DataFrame,
    dataset_id: UUID,
    query: Optional[str] = None,
    max_results: int = 10,
) -> List[Insight]:
    """Generate AI-powered insights from a dataset.
    
    Analyzes the provided DataFrame using Claude AI to discover
    patterns, trends, anomalies, and correlations.
    
    Args:
        df: The DataFrame to analyze.
        dataset_id: Unique identifier for the dataset.
        query: Optional specific question to focus the analysis.
        max_results: Maximum number of insights to return.
        
    Returns:
        List of Insight objects ordered by confidence score.
        
    Raises:
        ValueError: If DataFrame is empty.
        APIError: If Claude API call fails.
        
    Example:
        >>> generator = InsightsGenerator(api_client)
        >>> insights = generator.generate_insights(df, dataset_id)
        >>> print(insights[0].title)
        'Strong correlation between revenue and units sold'
    """
    ...
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_data_analyzer.py
│   ├── test_insights_generator.py
│   └── ...
├── integration/       # Tests with external dependencies
│   └── test_web_routes.py
├── fixtures/          # Test data files
│   └── sample_datasets/
└── conftest.py        # Shared fixtures
```

### Writing Tests

**Unit Tests:**

```python
class TestDataAnalyzer:
    """Tests for DataAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DataAnalyzer()

    def test_detect_numeric_column(self, analyzer):
        """Test that numeric columns are correctly identified."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        
        result = analyzer.analyze_columns(df)
        
        assert len(result) == 1
        assert result[0]["data_type"] in ["integer", "numeric"]

    def test_handles_empty_dataframe(self, analyzer):
        """Test graceful handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = analyzer.analyze_columns(df)
        
        assert result == []
```

**Integration Tests:**

```python
class TestDatasetsAPI:
    """Integration tests for datasets API."""

    def test_upload_csv_dataset(self, client, sample_csv_bytes):
        """Test CSV file upload via API."""
        response = client.post(
            "/api/v1/datasets",
            data={
                "file": (sample_csv_bytes, "test.csv"),
                "name": "Test Dataset",
            },
            content_type="multipart/form-data",
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data["success"] is True
        assert "dataset_id" in data
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_data_analyzer.py

# Specific test
pytest tests/unit/test_data_analyzer.py::TestDataAnalyzer::test_detect_numeric_column

# With coverage
pytest --cov=insightboost --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Test Coverage

We aim for at least 80% test coverage. Check coverage with:

```bash
pytest --cov=insightboost --cov-report=term-missing
```

---

## Submitting Changes

### Before Submitting

1. **Sync with upstream**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**

   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/
   
   # Lint
   ruff check src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest
   ```

3. **Update documentation** if needed

4. **Add or update tests** for your changes

### Creating a Pull Request

1. Push your branch to your fork:

   ```bash
   git push origin feature/your-feature
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template:

   - Describe what the PR does
   - Link related issues
   - List any breaking changes
   - Include screenshots for UI changes

### PR Title Format

Use the same format as commit messages:

```
feat(insights): add support for seasonal trend detection
```

---

## Review Process

### What Reviewers Look For

- Code correctness and completeness
- Test coverage
- Documentation updates
- Code style compliance
- Performance implications
- Security considerations
- Breaking changes

### Responding to Review

- Address all review comments
- Push new commits (don't force-push during review)
- Re-request review when ready
- Be patient and respectful

### Merging

Once approved:

1. Squash commits if needed
2. Update branch with main
3. Maintainer will merge the PR

---

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release tag:

   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

4. GitHub Actions will build and publish automatically

---

## Questions and Help

- Open a GitHub Discussion for questions
- Join our community chat (link in README)
- Check existing issues and discussions first

Thank you for contributing to InsightBoost.
