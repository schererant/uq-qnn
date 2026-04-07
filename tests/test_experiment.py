from __future__ import annotations

from typing import cast

import numpy as np

from src.experiment import Experiment
from src.hardware import GaussianNoise, HardwareProfile
from src.simulation import run_simulation_sequence_np


def base_config() -> dict[str, object]:
    return {
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
        "n_samples": 100,
        "memory_depth": 2,
        "n_swipe": 0,
        "swipe_span": 0.0,
        "sim_backend": "numpy",
        "loss_type": "mse",
        "n_classes": 1,
        "lr": 0.05,
        "epochs": 1,
        "seed": 7,
    }


def test_predict_with_hardware_noise_preserves_scalar_regression_shape():
    config = base_config()
    n_modes = cast(int, config["n_modes"])
    theta = np.linspace(0.1, 0.6, n_modes * (n_modes - 1))
    encoded = np.array([0.2, 0.7, 1.1])

    baseline = run_simulation_sequence_np(
        theta, encoded, Experiment("baseline", config=config).sim_cfg
    )

    zero_noise_profile = HardwareProfile(
        name="zero_noise",
        backend="numpy",
        noise=GaussianNoise(std=0.0),
        timing=None,
        coincidence_window_ns=None,
        coincidence_window_factor=None,
    )
    exp = Experiment("regression_hw_noise", config=config, hardware=zero_noise_profile)

    preds = exp.predict(theta, encoded)

    np.testing.assert_allclose(preds, baseline, atol=1e-12)
    assert not np.allclose(preds, np.ones_like(preds))


def test_uncertainty_analysis_accepts_int_memristive_phase_idx():
    config = base_config()
    config["memristive_phase_idx"] = 2

    exp = Experiment("uq_int_memristive", config=config)
    n_modes = cast(int, config["n_modes"])
    n_phases = n_modes * (n_modes - 1)
    theta = np.concatenate([np.linspace(0.1, 0.6, n_phases), np.array([0.4])])
    encoded = np.array([0.2, 0.7])

    result = exp.run_uncertainty_analysis(
        theta,
        encoded,
        n_passes=1,
        noise_std=0.01,
    )

    assert result["mean"].shape == encoded.shape
    assert result["std"].shape == encoded.shape
    assert result["all_preds"].shape == (len(encoded), 1)
