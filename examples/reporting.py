from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from example_logging import example_run, write_summary


def begin_console_capture(script_path: str):
    """Drop-in replacement that leverages logging-based example_run."""

    ctx = example_run(script_path)
    run_dir, _logger = ctx.__enter__()

    def _end_capture():
        ctx.__exit__(None, None, None)

    return run_dir, _end_capture


@contextmanager
def tee_stdout(path: Path | str):
    """Simple tee that mirrors stdout to a file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with path.open("w", encoding="utf-8") as sink:

        class Tee:
            def write(self, data: str) -> int:
                original_stdout.write(data)
                sink.write(data)
                return len(data)

            def flush(self) -> None:
                original_stdout.flush()
                sink.flush()

        sys.stdout = Tee()  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = original_stdout


def write_run_summary(
    run_dir: Path,
    *,
    metrics: dict | None = None,
    artifacts: list[str] | None = None,
    extra: dict | None = None,
    simulation: dict | None = None,
) -> None:
    """Compatibility wrapper that stores a summary JSON."""

    summary = {
        "metrics": metrics or {},
        "artifacts": artifacts or [],
        "extra": extra or {},
    }
    write_summary(run_dir, summary=summary, simulation_stats=simulation)
