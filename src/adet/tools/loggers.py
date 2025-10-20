"""
Tools for setting up logger utilities
"""

import logging
import sys
from typing import Sequence


class WordFilter(logging.Filter):
    def __init__(self, filter_out: str, name: str = '') -> None:
        self.filter_out = filter_out
        super().__init__(name)

    def filter(self, record: logging.LogRecord):
        return self.filter_out not in record.getMessage()


class ModuleFilter(logging.Filter):
    def __init__(self, filter_out: str, name: str = '') -> None:
        self.filter_out = filter_out
        super().__init__(name)

    def filter(self, record: logging.LogRecord):
        return self.filter_out not in record.pathname


def setup_logger(
    logger: logging.Logger,
    level=logging.INFO,
    root_level=logging.INFO,
    suppress_modules: Sequence[str] | None = None,
    banned_keywords: Sequence[str] | None = None,
) -> None:
    """
    Configure a logger instance with standard formatting.
    """
    logger.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Remove existing handlers from root
    if root_logger.handlers:
        for h in root_logger.handlers:
            root_logger.removeHandler(h)

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
        '>>> %(levelname)s | %(asctime)s,%(msecs)04d | %(name)s [%(funcName)s'
        + ':line %(lineno)d] ->> %(message)s',
        datefmt='%H:%M:%S',
    )
    console_handler.setFormatter(formatter)

    if suppress_modules:
        for mod in suppress_modules:
            console_handler.addFilter(ModuleFilter(mod))
    if banned_keywords:
        for word in banned_keywords:
            console_handler.addFilter(WordFilter(word))

    # Add handler to root so all loggers use it
    root_logger.addHandler(console_handler)
