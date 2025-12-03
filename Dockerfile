# InsightBoost Dockerfile
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================
# Stage 2: Production
# ============================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="InsightBoost Team" \
    version="1.0.0" \
    description="AI-powered data insights platform"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    APP_HOME=/app \
    # Flask settings
    FLASK_APP=insightboost.web.app:create_app \
    FLASK_ENV=production \
    # Gunicorn settings
    GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=120 \
    # App settings
    LOG_LEVEL=INFO \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r insightboost && \
    useradd -r -g insightboost -d /app -s /sbin/nologin insightboost

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/uploads && \
    chown -R insightboost:insightboost /app

# Copy application code
COPY --chown=insightboost:insightboost src/ /app/src/
COPY --chown=insightboost:insightboost pyproject.toml /app/

# Install the package in editable mode
RUN pip install -e .

# Switch to non-root user
USER insightboost

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command - run with Gunicorn
CMD ["sh", "-c", "gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    --timeout ${GUNICORN_TIMEOUT} \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    'insightboost.web.app:create_app()'"]

# ============================================
# Stage 3: Development
# ============================================
FROM production as development

# Switch back to root for dev setup
USER root

# Install dev dependencies
COPY requirements-dev.txt /app/
RUN pip install -r requirements-dev.txt

# Copy test files
COPY --chown=insightboost:insightboost tests/ /app/tests/

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1 \
    LOG_LEVEL=DEBUG

# Switch back to non-root user
USER insightboost

# Override command for development
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000", "--reload"]
