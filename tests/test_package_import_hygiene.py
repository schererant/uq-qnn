"""Public ``import src`` must not eagerly load Perceval (NumPy-only workflows)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_subprocess(code: str) -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(proc.stderr or proc.stdout or "subprocess failed")


def test_import_src_does_not_eagerly_load_perceval() -> None:
    root = Path(__file__).resolve().parents[1]
    code = r"""
import sys
sys.path.insert(0, %r)
import src
if "perceval" in sys.modules:
    raise AssertionError("perceval should not load on import src")
""" % str(
        root
    )
    _run_subprocess(code)


def test_from_src_simulation_run_simulation_sequence_stays_perceval_free() -> None:
    root = Path(__file__).resolve().parents[1]
    code = r"""
import sys
sys.path.insert(0, %r)
from src.simulation import run_simulation_sequence
if "perceval" in sys.modules:
    raise AssertionError("perceval loaded when importing run_simulation_sequence")
""" % str(
        root
    )
    _run_subprocess(code)


def test_numpy_forward_via_public_package_without_perceval_module() -> None:
    root = Path(__file__).resolve().parents[1]
    code = r"""
import sys
sys.path.insert(0, %r)
import src
import numpy as np
from src.config import SimConfig

if "perceval" in sys.modules:
    raise AssertionError("perceval after import src")

cfg = SimConfig.from_experiment_config({
    "n_modes": 3,
    "input_state": (0,),
    "encoding_phase_idx": 0,
    "photon_distinguishability": None,
    "target_mode": (2,),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "singles",
    "working_detectors": None,
    "noise_std": None,
    "n_samples": 10,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "sim_backend": "numpy",
    "loss_type": "mse",
    "n_classes": 1,
})
theta = np.linspace(0.1, 0.5, 6)
enc = np.array([0.2, 0.3])
out = src.run_simulation_sequence(theta, enc, cfg)
assert out.shape == (2,)
if "perceval" in sys.modules:
    raise AssertionError("perceval loaded after numpy-only forward")
""" % str(
        root
    )
    _run_subprocess(code)


def test_build_circuit_imports_perceval_when_called() -> None:
    root = Path(__file__).resolve().parents[1]
    code = r"""
import sys
sys.path.insert(0, %r)
import numpy as np
import src

if "perceval" in sys.modules:
    raise AssertionError("unexpected perceval before build_circuit")
_ = src.build_circuit(np.zeros(6, dtype=float), 0.1, n_modes=3, encoding_phase_idx=0)
if "perceval" not in sys.modules:
    raise AssertionError("perceval should load when build_circuit runs")
""" % str(
        root
    )
    _run_subprocess(code)


def test_src_reexports_remain_available_in_process() -> None:
    """Sanity: lazy Perceval in circuits does not break attribute exports."""

    import src

    assert src.run_simulation_sequence is not None
    assert src.build_circuit is not None
    assert src.get_mzi_modes_for_phase(0, 3) == (0, 1)
