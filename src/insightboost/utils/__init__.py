"""Utility modules for InsightBoost."""

from insightboost.utils.exceptions import (
    APIError,
    ConfigurationError,
    DatasetError,
    DataValidationError,
    InsightBoostError,
    RateLimitError,
    VisualizationError,
)
from insightboost.utils.formatters import (
    format_dataframe_for_llm,
    format_number,
    truncate_string,
)
from insightboost.utils.validators import (
    validate_dataframe,
    validate_file_upload,
    validate_query,
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
