"""Logging utilities.

This module provides logging configuration for
training pipelines.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for training.
    
    Args:
        level: Logging level.
        log_file: Optional file to write logs to.
        format_string: Custom format string.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
