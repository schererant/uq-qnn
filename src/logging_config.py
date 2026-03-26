"""
Centralized logging configuration for the UQ-QNN framework.

All modules should use:
    from .logging_config import get_logger
    logger = get_logger(__name__)
instead of the print built-in for user-visible output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

_ROOT_NAME = "uqqnn"
_root_logger = logging.getLogger(_ROOT_NAME)
_root_logger.setLevel(logging.DEBUG)  # handlers decide what shows

if not _root_logger.handlers:  # guard against notebook reloads
    _console = logging.StreamHandler(sys.stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-7s — %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    _root_logger.addHandler(_console)

_root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'uqqnn' hierarchy."""
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)


def set_verbosity(level: Union[int, str] = logging.INFO) -> None:
    """Set console verbosity for the whole package (DEBUG / INFO / WARNING …)."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    for h in _root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            h.setLevel(level)


def add_file_handler(
    path: Union[str, Path],
    level: int = logging.DEBUG,
) -> logging.FileHandler:
    """Attach a file handler (captures ALL levels).  Returns it for later removal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _root_logger.addHandler(fh)
    return fh


def remove_file_handler(fh: logging.FileHandler) -> None:
    """Remove a previously added file handler and close it."""
    _root_logger.removeHandler(fh)
    fh.close()


def log_params(
    logger: logging.Logger,
    params: dict,
    title: str = "Run parameters",
) -> None:
    """Log a parameter dict in a readable block (replaces print_run_params)."""
    lines = [f"--- {title} ---"]
    for k, v in sorted(params.items()):
        if v is not None:
            lines.append(f"  {k}: {v}")
    logger.info("\n".join(lines))
