"""Logging Configuration

Sets up logging for the application with colored console output
and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_str: Optional format string.

    Returns:
        Root logger instance.
    """
    # Default format
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if HAS_COLORLOG:
        color_format = (
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = colorlog.ColoredFormatter(
            color_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        console_formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if path provided)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    return logger
