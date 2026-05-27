from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple

from src.logging_config import add_file_handler, get_logger, remove_file_handler


def _reports_root() -> Path:
    return Path(__file__).resolve().parents[1] / "reports"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


@contextmanager
def example_run(script_path: str) -> Iterator[Tuple[Path, object]]:
    """Context manager that sets up a run directory + logger for an example script."""

    script_name = Path(script_path).stem
    run_dir = _reports_root() / script_name / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"examples.{script_name}")
    file_handler = add_file_handler(run_dir / "run.log")
    logger.info("=== %s ===", script_name.replace("_", " ").title())
    logger.info("Run directory: %s", run_dir)
    try:
        yield run_dir, logger
    finally:
        remove_file_handler(file_handler)


def write_summary(
    run_dir: Path,
    *,
    summary: dict | None = None,
    simulation_stats: dict | None = None,
) -> None:
    """Persist a lightweight summary JSON for convenience."""

    payload = {
        "schema": "uq-qnn.example_run.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary or {},
        "simulation": simulation_stats or {},
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2))
