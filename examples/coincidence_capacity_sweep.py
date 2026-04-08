#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capacity sweep for the validated coincidence configuration."""

import os
import sys
from typing import Dict, Tuple

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

N_MODES_LIST: Tuple[int, ...] = (6, 8, 12)
# Photons injected in modes 0 and 3 (occupation built per n_modes in _sim_cfg).
INJECTION_MODES: Tuple[int, int] = (0, 3)
TARGET_CC_PAIR = (0, 1)
ENCODING_PHASE_IDX = 7

CONFIG: Dict = {
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "input_state": tuple(
        1 if i in INJECTION_MODES else 0 for i in range(N_MODES_LIST[0])
    ),
    "photon_distinguishability": "indistinguishable",
    "output_mode": "coincidence",
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
    "n_modes": N_MODES_LIST[0],
    "target_mode": TARGET_CC_PAIR,
    "working_detectors": tuple(range(N_MODES_LIST[0])),
    "n_modes_list": list(N_MODES_LIST),
}


def _sim_cfg(n_modes: int) -> SimConfig:
    base = {k: v for k, v in CONFIG.items() if k != "n_modes_list"}
    base["n_modes"] = n_modes
    base["working_detectors"] = tuple(range(n_modes))
    base["input_state"] = tuple(
        1 if i in INJECTION_MODES else 0 for i in range(n_modes)
    )
    return SimConfig.from_experiment_config(base)


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


def _plot_predictions(
    results: Dict[int, Dict[str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1, len(N_MODES_LIST), figsize=(5 * len(N_MODES_LIST), 4.5), sharey=True
    )
    axes_arr = np.atleast_1d(axes)
    for ax, n_modes in zip(axes_arr, N_MODES_LIST):
        result = results[n_modes]
        metrics = result["metrics"]
        ax.scatter(X_train, y_train, s=18, alpha=0.55, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
        ax.plot(
            X_test, result["mean_preds"], color="tab:blue", lw=2.0, label="Prediction"
        )
        ax.fill_between(
            X_test,
            result["mean_preds"] - 1.96 * result["std_preds"],
            result["mean_preds"] + 1.96 * result["std_preds"],
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )
        ax.set_title(
            f"n_modes={n_modes}\nRMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
        )
        ax.set(xlabel="x", ylabel="y")
        ax.grid(True)
    axes_arr[0].legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_capacity_metrics(results: Dict[int, Dict[str, object]]) -> plt.Figure:
    n_modes_values = list(N_MODES_LIST)
    rmse = [float(results[n_modes]["metrics"]["rmse"]) for n_modes in n_modes_values]
    r2 = [float(results[n_modes]["metrics"]["r2"]) for n_modes in n_modes_values]
    params = [int(results[n_modes]["n_params"]) for n_modes in n_modes_values]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(n_modes_values, rmse, marker="o")
    axes[0].set(title="RMSE vs n_modes", xlabel="n_modes", ylabel="RMSE")
    axes[0].grid(True)

    axes[1].plot(n_modes_values, r2, marker="o")
    axes[1].set(title="R2 vs n_modes", xlabel="n_modes", ylabel="R2")
    axes[1].grid(True)

    axes[2].plot(n_modes_values, params, marker="o")
    axes[2].set(title="Phase params vs n_modes", xlabel="n_modes", ylabel="count")
    axes[2].grid(True)

    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment("Coincidence Capacity Sweep", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))
        X_train, y_train, X_test, y_test = get_data(
            int(CONFIG["n_data"]),
            float(CONFIG["sigma_noise"]),
            str(CONFIG["data_function"]),
        )
        enc_train = 2 * np.arccos(np.clip(X_train, 0.0, 1.0))
        enc_test = 2 * np.arccos(np.clip(X_test, 0.0, 1.0))

        results: Dict[int, Dict[str, object]] = {}
        for n_modes in N_MODES_LIST:
            sim_cfg = _sim_cfg(n_modes)
            logger.info(
                "Training validated coincidence configuration with n_modes=%d, input_state=%s, target_mode=%s",
                n_modes,
                sim_cfg.input_state,
                TARGET_CC_PAIR,
            )
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
            results[n_modes] = {
                "history": history,
                "mean_preds": unc["mean"],
                "std_preds": unc["std"],
                "metrics": metrics,
                "final_loss": float(history[-1]),
                "n_params": n_modes * (n_modes - 1),
            }
            logger.info(
                "  final_loss=%.6f mse=%.6f rmse=%.4f r2=%.4f",
                float(history[-1]),
                metrics["mse"],
                metrics["rmse"],
                metrics["r2"],
            )

        winner = min(
            results, key=lambda n_modes: float(results[n_modes]["metrics"]["mse"])
        )
        exp.save_metrics(
            {
                "by_n_modes": {
                    f"n_modes_{n_modes}": {
                        "n_params": int(result["n_params"]),
                        "final_loss": float(result["final_loss"]),
                        **result["metrics"],
                    }
                    for n_modes, result in results.items()
                },
                "best_n_modes": winner,
                "status": "completed",
            }
        )
        exp.savefig(
            _plot_predictions(results, X_train, y_train, X_test, y_test),
            "predictions_by_capacity.png",
        )
        exp.savefig(_plot_capacity_metrics(results), "capacity_metrics.png")

        logger.info("Best n_modes=%d", winner)


if __name__ == "__main__":
    main()
