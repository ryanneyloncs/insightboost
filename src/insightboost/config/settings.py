"""
Application settings using pydantic-settings for type-safe environment variables.

This module provides centralized configuration management for InsightBoost,
loading settings from environment variables with validation and defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        anthropic_api_key: API key for Anthropic Claude API (required)
        database_url: Database connection URL
        secret_key: Secret key for Flask sessions and security
        log_level: Logging level for the application
        max_upload_size_mb: Maximum file upload size in megabytes
        rate_limit_requests_per_minute: API rate limit
        visualization_cache_ttl_seconds: Cache TTL for visualizations
        redis_url: Optional Redis URL for caching
        flask_debug: Enable Flask debug mode
        flask_env: Flask environment
        flask_host: Host to bind Flask server
        flask_port: Port for Flask server
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required settings
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude API access",
    )

    secret_key: str = Field(
        default="insightboost-dev-secret-key-change-in-production",
        description="Secret key for Flask sessions",
    )

    # Database settings
    database_url: str = Field(
        default="sqlite:///insightboost.db",
        description="Database connection URL",
    )

    # Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum upload size in MB",
    )

    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="API rate limit per minute",
    )

    visualization_cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Visualization cache TTL in seconds",
    )

    # Redis settings (optional)
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for caching and sessions",
    )

    # Flask settings
    flask_debug: bool = Field(
        default=False,
        description="Enable Flask debug mode",
    )

    flask_env: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Flask environment",
    )

    flask_host: str = Field(
        default="0.0.0.0",
        description="Flask server host",
    )

    flask_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Flask server port",
    )

    # Anthropic API settings
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Anthropic model to use",
    )

    anthropic_max_tokens: int = Field(
        default=4096,
        ge=100,
        le=200000,
        description="Maximum tokens for API responses",
    )

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty or placeholder."""
        if not v or v == "your-api-key-here":
            raise ValueError(
                "ANTHROPIC_API_KEY must be set to a valid API key. "
                "Get your key at: https://console.anthropic.com/"
            )
        return v

    @property
    def max_upload_size_bytes(self) -> int:
        """Get maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.flask_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.flask_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


def get_settings_uncached() -> Settings:
    """
    Get fresh application settings (not cached).

    Useful for testing when settings need to be reloaded.

    Returns:
        Settings: Fresh application settings instance
    """
    return Settings()
