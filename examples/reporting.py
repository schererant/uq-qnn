# -*- coding: utf-8 -*-
"""Helpers for saving example outputs under ``reports/<script_name>/<timestamp>/``."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def project_root() -> Path:
    """Path to repo root (parent of ``examples/``)."""
    return Path(__file__).resolve().parent.parent


def make_run_dir(example_file: str) -> Path:
    """
    Create ``reports/<example_stem>/<YYYY-mm-dd_HHMMSS>/`` and return it.

    Parameters
    ----------
    example_file
        Pass ``__file__`` from the calling example script.
    """
    path = Path(example_file).resolve()
    name = path.stem
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = project_root() / "reports" / name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_sha_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(obj: Any) -> Any:
    """Convert numpy types and nested structures to JSON-serializable values."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_run_summary(
    run_dir: Path,
    *,
    example_name: str | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_summary.json`` into the run directory."""
    summary: dict[str, Any] = {
        "schema": "uq-qnn.example_run.v1",
        "example": example_name or run_dir.parent.name,
        "run_dir": str(run_dir.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
    }
    sha = _git_sha_short()
    if sha:
        summary["git_commit_short"] = sha
    if metrics:
        summary["metrics"] = json_safe(metrics)
    if artifacts:
        resolved = []
        for a in artifacts:
            p = Path(a)
            resolved.append(str(p if p.is_absolute() else (run_dir / p).resolve()))
        summary["artifacts"] = resolved
    if extra:
        summary["extra"] = json_safe(extra)

    out = run_dir / "run_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out


def print_report_banner(run_dir: Path) -> None:
    """Print a clear line to stdout so runs are easy to spot in logs."""
    print(f"\n[Report directory] {run_dir.resolve()}\n")


@contextmanager
def tee_stdout(log_path: Path) -> Iterator[None]:
    """Duplicate stdout to ``log_path`` while still printing to the console."""

    class _Tee:
        __slots__ = ("_file",)

        def __init__(self, file) -> None:
            self._file = file

        def write(self, data: str) -> None:
            sys.__stdout__.write(data)
            self._file.write(data)

        def flush(self) -> None:
            sys.__stdout__.flush()
            self._file.flush()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = _Tee(f)
        try:
            yield
        finally:
            sys.stdout = old
