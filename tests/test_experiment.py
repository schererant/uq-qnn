from __future__ import annotations

import json
import warnings
from typing import cast

import numpy as np

from src.experiment import Experiment
from src.hardware import GaussianNoise, HardwareProfile
from src.loss import TrainedPhotonicState


def base_config() -> dict[str, object]:
    return {
        "n_modes": 3,
        "input_state": (1, 0, 0),
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


def test_predict_deprecated_wrapper_matches_serialized_state():
    config = base_config()
    X_train = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
    y_train = np.array([0.2, 0.4, 0.5, 0.7], dtype=np.float64)
    encoded = np.array([0.2, 0.7, 1.1], dtype=np.float64)

    zero_noise_profile = HardwareProfile(
        name="zero_noise",
        backend="numpy",
        noise=GaussianNoise(std=0.0),
        timing=None,
        coincidence_window_ns=None,
        coincidence_window_factor=None,
    )
    exp = Experiment("regression_hw_noise", config=config, hardware=zero_noise_profile)
    trained_state, _, model = exp.train(X_train, y_train)

    expected = trained_state.predict(encoded)
    np.testing.assert_allclose(model.predict(encoded), expected, atol=1e-12)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preds = exp.predict(trained_state, encoded)

    np.testing.assert_allclose(preds, expected, atol=1e-12)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_train_saves_artifacts_and_run_summary_refs(tmp_path):
    config = base_config()
    X_train = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    y_train = np.array([0.3, 0.6, 0.9], dtype=np.float64)

    exp = Experiment("artifact_save", config=config)
    exp.project_root = tmp_path
    exp.run_dir = tmp_path / "reports" / "artifact_save" / exp.timestamp

    with exp:
        trained_state, history, _ = exp.train(X_train, y_train)

    state_path = exp.run_dir / "trained_state.json"
    history_path = exp.run_dir / "loss_history.json"
    summary_path = exp.run_dir / "run_summary.json"

    assert state_path.is_file()
    assert history_path.is_file()
    assert summary_path.is_file()

    restored = TrainedPhotonicState.load_json(state_path)
    np.testing.assert_allclose(
        restored.theta_array(),
        trained_state.theta_array(),
        atol=1e-12,
    )
    assert json.loads(history_path.read_text(encoding="utf-8")) == history

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["trained_state"] == "trained_state.json"
    assert summary["metrics"]["loss_history"] == "loss_history.json"
    assert str(state_path.resolve()) in summary["artifacts"]
    assert str(history_path.resolve()) in summary["artifacts"]


def test_trained_state_roundtrip_preserves_predictions():
    config = base_config()
    X_train = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    y_train = np.array([0.3, 0.6, 0.9], dtype=np.float64)
    encoded = np.array([0.1, 0.9], dtype=np.float64)

    exp = Experiment("trained_state_roundtrip", config=config)
    trained_state, _, _ = exp.train(X_train, y_train)

    restored = TrainedPhotonicState.from_dict(trained_state.to_dict())

    np.testing.assert_allclose(
        restored.predict(encoded),
        trained_state.predict(encoded),
        atol=1e-12,
    )


def test_uncertainty_analysis_accepts_int_memristive_phase_idx():
    config = base_config()
    config["memristive_phase_idx"] = 2

    exp = Experiment("uq_int_memristive", config=config)
    n_modes = cast(int, config["n_modes"])
    n_phases = n_modes * (n_modes - 1)
    theta = np.concatenate([np.linspace(0.1, 0.6, n_phases), np.array([0.4])])
    encoded = np.array([0.2, 0.7])

    trained_state = TrainedPhotonicState(
        theta=tuple(float(x) for x in theta),
        head_weight=((1.0, 0.0, 0.0),),
        head_bias=(0.0,),
        sim_cfg=exp.sim_cfg.to_dict(),
    )

    result = exp.run_uncertainty_analysis(
        trained_state,
        encoded,
        n_passes=1,
        noise_std=0.01,
    )

    assert result["mean"].shape == encoded.shape
    assert result["std"].shape == encoded.shape
    assert result["all_preds"].shape == (len(encoded), 1)
