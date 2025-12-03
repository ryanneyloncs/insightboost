"""
Custom exceptions for InsightBoost.

This module defines a hierarchy of exceptions for handling errors
throughout the application in a consistent manner.
"""

from typing import Any


class InsightBoostError(Exception):
    """
    Base exception for all InsightBoost errors.
    
    All custom exceptions in the application should inherit from this class
    to enable consistent error handling and logging.
    
    Attributes:
        message: Human-readable error message
        details: Additional error details for debugging
        error_code: Optional error code for API responses
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional error details
            error_code: Optional error code
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or "INSIGHTBOOST_ERROR"
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(InsightBoostError):
    """
    Raised when there is a configuration error.
    
    Examples:
        - Missing required environment variables
        - Invalid configuration values
        - Missing API keys
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            details=details,
            error_code="CONFIGURATION_ERROR",
        )


class APIError(InsightBoostError):
    """
    Raised when there is an error with external API calls.
    
    Attributes:
        status_code: HTTP status code from the API
        response_body: Raw response body from the API
    """
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate
        
        super().__init__(
            message=message,
            details=details,
            error_code="API_ERROR",
        )
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """
    Raised when API rate limits are exceeded.
    
    Attributes:
        retry_after: Seconds to wait before retrying
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            status_code=429,
            details=details,
        )
        self.error_code = "RATE_LIMIT_ERROR"
        self.retry_after = retry_after


class DataValidationError(InsightBoostError):
    """
    Raised when data validation fails.
    
    Examples:
        - Invalid DataFrame structure
        - Missing required columns
        - Invalid data types
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if field:
            details["field"] = field
        
        super().__init__(
            message=message,
            details=details,
            error_code="DATA_VALIDATION_ERROR",
        )
        self.field = field


class DatasetError(InsightBoostError):
    """
    Raised when there is an error with dataset operations.
    
    Examples:
        - File upload failures
        - Unsupported file formats
        - Dataset not found
    """
    
    def __init__(
        self,
        message: str,
        dataset_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if dataset_id:
            details["dataset_id"] = dataset_id
        
        super().__init__(
            message=message,
            details=details,
            error_code="DATASET_ERROR",
        )
        self.dataset_id = dataset_id


class VisualizationError(InsightBoostError):
    """
    Raised when there is an error generating visualizations.
    
    Examples:
        - Invalid chart configuration
        - Incompatible data for chart type
        - Rendering failures
    """
    
    def __init__(
        self,
        message: str,
        chart_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if chart_type:
            details["chart_type"] = chart_type
        
        super().__init__(
            message=message,
            details=details,
            error_code="VISUALIZATION_ERROR",
        )
        self.chart_type = chart_type


class InsightGenerationError(InsightBoostError):
    """
    Raised when insight generation fails.
    
    Examples:
        - LLM response parsing errors
        - Invalid insight format
        - Analysis failures
    """
    
    def __init__(
        self,
        message: str,
        query: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query[:200]  # Truncate
        
        super().__init__(
            message=message,
            details=details,
            error_code="INSIGHT_GENERATION_ERROR",
        )
        self.query = query


class AuthenticationError(InsightBoostError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            details=details,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(InsightBoostError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Not authorized to perform this action",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            details=details,
            error_code="AUTHORIZATION_ERROR",
        )
