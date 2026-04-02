"""
Utility helpers for the UQ-QNN framework.
"""

from __future__ import annotations

from typing import Optional

# from .data import compute_n_swipe
from .hardware import TimingParams
from .logging_config import get_logger, log_params

logger = get_logger(__name__)


def print_run_params(title: str = "Run parameters", **params) -> None:
    """Log all set parameters at the beginning of a run for reproducibility.

    .. deprecated:: Use ``log_params()`` from ``logging_config`` directly.
    """
    log_params(logger, params, title=title)


def resolve_n_swipe(
    timing: TimingParams,
    *,
    n_swipe_override: Optional[int] = None,
) -> int:
    """Resolve the number of swipe points from timing params or an explicit override.

    Args:
        timing: Hardware timing parameters.
        n_swipe_override: If not None, use this value directly instead
            of computing from timing.

    Returns:
        Odd integer swipe count.
    """
    if n_swipe_override is not None:
        return n_swipe_override
    auto = timing.compute_n_swipe()
    logger.info("Computed n_swipe = %d", auto)
    return auto
