"""
Logging configuration for InsightBoost.

Provides structured logging with customizable formats and handlers.
"""

import logging
import sys
from typing import Any

from insightboost.config.settings import get_settings


def setup_logging(
    level: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up application logging.
    
    Args:
        level: Logging level (defaults to settings.log_level)
        format_string: Custom log format string
        
    Returns:
        Logger: Configured root logger for InsightBoost
    """
    settings = get_settings()
    log_level = level or settings.log_level
    
    # Default format with timestamp, level, module, and message
    default_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    log_format = format_string or default_format
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Create InsightBoost logger
    logger = logging.getLogger("insightboost")
    logger.setLevel(getattr(logging, log_level))
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the InsightBoost namespace.
    
    Args:
        name: Logger name (will be prefixed with 'insightboost.')
        
    Returns:
        Logger: Configured logger instance
    """
    return logging.getLogger(f"insightboost.{name}")


class LoggerMixin:
    """Mixin class to add logging capability to classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(
    logger: logging.Logger,
    func_name: str,
    **kwargs: Any,
) -> None:
    """
    Log a function call with its arguments.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function arguments to log
    """
    # Truncate long values for logging
    truncated_kwargs = {}
    for key, value in kwargs.items():
        str_value = str(value)
        if len(str_value) > 100:
            truncated_kwargs[key] = f"{str_value[:100]}..."
        else:
            truncated_kwargs[key] = value
    
    logger.debug(f"Calling {func_name} with args: {truncated_kwargs}")
