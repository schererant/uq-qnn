"""Reference-case checks aligned with examples/coincidence_regression.py."""

from __future__ import annotations

import numpy as np

from src.config import SimConfig
from src.numpy_backend import run_vectorized_non_memristive
from src.simulation import run_simulation_sequence


def _coincidence_regression_sim_config() -> SimConfig:
    """Subset of CONFIG from examples/coincidence_regression.py (NumPy coincidence)."""

    return SimConfig.from_experiment_config(
        {
            "n_modes": 6,
            "input_state": (0, 3),
            "encoding_phase_idx": 7,
            "photon_distinguishability": "indistinguishable",
            "target_mode": (0, 1),
            "memristive_phase_idx": None,
            "memristive_output_modes": None,
            "output_mode": "coincidence",
            "working_detectors": tuple(range(6)),
            "loss_type": "mse",
            "n_classes": 1,
            "n_samples": 1000,
            "memory_depth": 2,
            "n_swipe": 0,
            "swipe_span": 0.0,
            "noise_std": None,
            "sim_backend": "numpy",
        }
    )


def test_coincidence_regression_numpy_predictions_match_vectorized_path() -> None:
    cfg = _coincidence_regression_sim_config()
    theta = np.linspace(0.05, 1.2, 30)
    enc = np.array([0.1, 0.4, 0.9])

    via_runner = run_simulation_sequence(theta, enc, cfg)
    via_vec = run_vectorized_non_memristive(theta, enc, cfg)

    np.testing.assert_allclose(via_runner, via_vec, rtol=0, atol=1e-12)


def test_coincidence_regression_numpy_golden_values() -> None:
    cfg = _coincidence_regression_sim_config()
    theta = np.linspace(0.05, 1.2, 30)
    enc = np.array([0.1, 0.4, 0.9])

    expected = np.array([0.03396306, 0.03566954, 0.03737525])
    got = run_simulation_sequence(theta, enc, cfg)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)


def test_sim_config_sim_backend_alias() -> None:
    cfg = _coincidence_regression_sim_config()
    assert cfg.sim_backend == cfg.backend == "numpy"
