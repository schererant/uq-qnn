#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the legacy coincidence configuration against the validated one."""

import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimConfig
from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger
from src.simulation import run_simulation_sequence_np
from src.training import train_pytorch_generic

logger = get_logger(__name__)

CASES = {
    "legacy_cc24": {
        "target_mode": (2, 4),
        "input_state": tuple(1 if i in (1, 4) else 0 for i in range(6)),
        "encoding_phase_idx": 7,
        "photon_distinguishability": "indistinguishable",
        "notes": "Previous example-style configuration.",
    },
    "validated_cc01": {
        "target_mode": (0, 1),
        "input_state": tuple(1 if i in (0, 3) else 0 for i in range(6)),
        "encoding_phase_idx": 7,
        "photon_distinguishability": "indistinguishable",
        "notes": "Best validated input/readout configuration from the input-pair sweep.",
    },
}

INPUT_STATE = CASES["validated_cc01"]["input_state"]
TARGET_CC_PAIR = CASES["validated_cc01"]["target_mode"]
ENCODING_PHASE_IDX = CASES["validated_cc01"]["encoding_phase_idx"]

CONFIG: Dict = {
    "n_modes": 6,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "coincidence",
    "working_detectors": tuple(range(6)),
    "loss_type": "mse",
    "n_classes": 1,
    "lr": 0.05,
    "epochs": 150,
    "n_samples": 1000,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    "sim_backend": "numpy",
    "n_data": 100,
    "sigma_noise": 0.005,
    "data_function": "quartic_data",
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
    "input_state": INPUT_STATE,
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "photon_distinguishability": "indistinguishable",
    "target_mode": TARGET_CC_PAIR,
    "cases": CASES,
}


def _sim_cfg(case: Dict[str, object]) -> SimConfig:
    base = {k: v for k, v in CONFIG.items() if k != "cases"}
    merged = dict(base)
    for k, v in case.items():
        if k != "notes":
            merged[k] = v
    return SimConfig.from_experiment_config(merged)


def _run_uncertainty(
    theta: np.ndarray,
    enc_test: np.ndarray,
    sim_cfg: SimConfig,
    *,
    n_passes: int,
    noise_std: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_preds = np.zeros((len(enc_test), n_passes), dtype=float)
    for idx in range(n_passes):
        perturbed = theta + rng.normal(0.0, noise_std, size=len(theta))
        all_preds[:, idx] = run_simulation_sequence_np(perturbed, enc_test, sim_cfg)
    return {
        "mean": np.mean(all_preds, axis=1),
        "std": np.std(all_preds, axis=1),
    }


def _compute_metrics(
    y_test: np.ndarray,
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
) -> Dict[str, float]:
    residuals = mean_preds - y_test
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    lo = mean_preds - 1.96 * std_preds
    hi = mean_preds + 1.96 * std_preds
    coverage_95 = float(np.mean((y_test >= lo) & (y_test <= hi)))
    sigma = np.clip(std_preds, 1e-6, None)
    nll = float(np.mean(0.5 * (residuals / sigma) ** 2 + np.log(sigma)))
    cal_rho, _ = stats.spearmanr(std_preds, np.abs(residuals))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coverage_95": coverage_95,
        "mean_std": float(np.mean(std_preds)),
        "nll": nll,
        "calibration_rho": float(cal_rho),
    }


def _display(case_name: str, case: Dict[str, object]) -> str:
    target_mode = tuple(int(v) for v in case["target_mode"])
    inp = tuple(int(v) for v in case["input_state"])  # type: ignore[arg-type]
    return (
        f"{case_name}: input_state={inp}, "
        f"CC{target_mode[0]}{target_mode[1]}, encoding_phase_idx={int(case['encoding_phase_idx'])}"
    )


def _plot_predictions(
    results: Dict[str, Dict[str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, result) in zip(axes, results.items()):
        ax.scatter(X_train, y_train, s=18, alpha=0.55, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
        ax.plot(
            X_test, result["mean_preds"], color="tab:red", lw=2.0, label="Prediction"
        )
        ax.fill_between(
            X_test,
            result["mean_preds"] - 1.96 * result["std_preds"],
            X_test * 0 + result["mean_preds"] + 1.96 * result["std_preds"],
            color="tab:red",
            alpha=0.2,
            label="95% CI",
        )
        metrics = result["metrics"]
        ax.set_title(f"{name}\nRMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
        ax.set(xlabel="x", ylabel="y")
        ax.grid(True)
    axes[0].legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_metric_bars(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    names = list(results.keys())
    rmse = [float(results[name]["metrics"]["rmse"]) for name in names]
    r2 = [float(results[name]["metrics"]["r2"]) for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rmse, color=["tab:gray", "tab:blue"])
    axes[0].set(title="RMSE", ylabel="Value")
    axes[0].grid(True, axis="y")

    axes[1].bar(names, r2, color=["tab:gray", "tab:blue"])
    axes[1].set(title="R2", ylabel="Value")
    axes[1].grid(True, axis="y")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment("Coincidence Configuration Comparison", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))
        X_train, y_train, X_test, y_test = get_data(
            int(CONFIG["n_data"]),
            float(CONFIG["sigma_noise"]),
            str(CONFIG["data_function"]),
        )
        enc_train = 2 * np.arccos(np.clip(X_train, 0.0, 1.0))
        enc_test = 2 * np.arccos(np.clip(X_test, 0.0, 1.0))

        results: Dict[str, Dict[str, object]] = {}
        for name, case in CASES.items():
            logger.info("Training %s", _display(name, case))
            sim_cfg = _sim_cfg(case)
            theta, history = train_pytorch_generic(
                enc_train,
                y_train,
                sim_cfg=sim_cfg,
                lr=float(CONFIG["lr"]),
                epochs=int(CONFIG["epochs"]),
                seed=int(CONFIG["seed"]),
            )
            unc = _run_uncertainty(
                theta,
                enc_test,
                sim_cfg,
                n_passes=int(CONFIG["unc_n_passes"]),
                noise_std=float(CONFIG["unc_noise_std"]),
                seed=int(CONFIG["seed"]),
            )
            metrics = _compute_metrics(y_test, unc["mean"], unc["std"])
            results[name] = {
                "case": case,
                "history": history,
                "mean_preds": unc["mean"],
                "std_preds": unc["std"],
                "metrics": metrics,
                "final_loss": float(history[-1]),
            }
            logger.info(
                "  final_loss=%.6f mse=%.6f rmse=%.4f r2=%.4f",
                float(history[-1]),
                metrics["mse"],
                metrics["rmse"],
                metrics["r2"],
            )

        winner = min(results, key=lambda name: float(results[name]["metrics"]["mse"]))
        exp.save_metrics(
            {
                "by_case": {
                    name: {
                        "target_mode": tuple(case["target_mode"]),
                        "input_state": tuple(int(v) for v in case["input_state"]),  # type: ignore[arg-type]
                        "encoding_phase_idx": int(case["encoding_phase_idx"]),
                        "final_loss": float(result["final_loss"]),
                        **result["metrics"],
                    }
                    for name, result in results.items()
                    for case in [result["case"]]
                },
                "winner": winner,
                "status": "completed",
            }
        )
        exp.savefig(
            _plot_predictions(results, X_train, y_train, X_test, y_test),
            "predictions_comparison.png",
        )
        exp.savefig(_plot_metric_bars(results), "metrics_comparison.png")

        logger.info("Best configuration: %s", winner)


if __name__ == "__main__":
    main()
