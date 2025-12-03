"""Utility modules for InsightBoost."""

from insightboost.utils.exceptions import (
    InsightBoostError,
    APIError,
    DataValidationError,
    RateLimitError,
    ConfigurationError,
    DatasetError,
    VisualizationError,
)
from insightboost.utils.validators import (
    validate_dataframe,
    validate_file_upload,
    validate_query,
)
from insightboost.utils.formatters import (
    format_dataframe_for_llm,
    format_number,
    truncate_string,
)

__all__ = [
    # Exceptions
    "InsightBoostError",
    "APIError",
    "DataValidationError",
    "RateLimitError",
    "ConfigurationError",
    "DatasetError",
    "VisualizationError",
    # Validators
    "validate_dataframe",
    "validate_file_upload",
    "validate_query",
    # Formatters
    "format_dataframe_for_llm",
    "format_number",
    "truncate_string",
]
