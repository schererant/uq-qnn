"""Import hygiene for the simulation runner (no eager Perceval)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_python(root: Path, code: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(proc.stderr or proc.stdout or "subprocess failed")


def test_runner_import_does_not_load_perceval() -> None:
    """Load ``src.simulation.runner`` without executing ``src/__init__.py`` (which pulls Perceval)."""

    root = Path(__file__).resolve().parents[1]
    code = r"""
import importlib
import sys
import types
from pathlib import Path

root = Path(%r).resolve()
src_path = root / "src"

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(src_path)]
sys.modules["src"] = src_pkg

sim_pkg = types.ModuleType("src.simulation")
sim_pkg.__path__ = [str(src_path / "simulation")]
sys.modules["src.simulation"] = sim_pkg

importlib.import_module("src.simulation.runner")
if "perceval" in sys.modules:
    raise AssertionError("perceval loaded on runner import")
""" % str(
        root
    )
    _run_python(root, code)


def test_import_src_does_not_load_perceval() -> None:
    root = Path(__file__).resolve().parents[1]
    code = r"""
import sys

import src

if "perceval" in sys.modules:
    raise AssertionError("perceval loaded on import src")

_ = src.run_simulation_sequence
_ = src.SimConfig
_ = src.circuits
_ = src.display_circuit_annotated

if "perceval" in sys.modules:
    raise AssertionError("perceval loaded while resolving lazy top-level exports")
"""  # nosec B703
    _run_python(root, code)


def test_run_simulation_sequence_alias_matches_primary() -> None:
    from src.simulation import run_simulation_sequence, run_simulation_sequence_np

    assert run_simulation_sequence_np is run_simulation_sequence
