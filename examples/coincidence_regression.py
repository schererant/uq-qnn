#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coincidence regression with the validated configuration."""

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
from src.simulation import run_simulation_sequence
from src.training import train_pytorch_generic

logger = get_logger(__name__)

# Coincidence readout models a two-photon input across distinct modes.
INPUT_STATE = (0, 1)
TARGET_CC_PAIR = (0, 1)
ENCODING_PHASE_IDX = 7

CONFIG = {
    "n_modes": 6,
    "input_state": INPUT_STATE,
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "photon_distinguishability": "indistinguishable",
    "target_mode": TARGET_CC_PAIR,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "coincidence",
    "working_detectors": tuple(range(6)),
    "loss_type": "mse",
    "n_classes": 1,
    "lr": 0.05,
    "epochs": 300,
    "n_samples": 0,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    "sim_backend": "numpy",
    "n_data": 1000,
    "sigma_noise": 0.005,
    "data_function": "step_function_data",
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}


def _sim_cfg() -> SimConfig:
    return SimConfig.from_experiment_config(CONFIG)


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
        all_preds[:, idx] = run_simulation_sequence(perturbed, enc_test, sim_cfg)
    return {
        "mean": np.mean(all_preds, axis=1),
        "std": np.std(all_preds, axis=1),
        "all_preds": all_preds,
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


def _plot_training_loss(history: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history, color="tab:blue", lw=2)
    ax.set_yscale("log")
    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title=(
            "Training Loss: input_state=%s, CC%s%s, encoding_phase_idx=%d"
            % (INPUT_STATE, TARGET_CC_PAIR[0], TARGET_CC_PAIR[1], ENCODING_PHASE_IDX)
        ),
    )
    ax.grid(True)
    fig.tight_layout()
    return fig


def _plot_summary(
    history: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
    metrics: Dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history, color="tab:blue")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    axes[0, 0].grid(True)

    axes[0, 1].scatter(X_train, y_train, s=20, alpha=0.7, label="Training data")
    axes[0, 1].plot(X_test, y_test, "k--", label="Ground truth")
    axes[0, 1].plot(X_test, mean_preds, color="tab:red", lw=2, label="Mean prediction")
    axes[0, 1].fill_between(
        X_test,
        mean_preds - 1.96 * std_preds,
        mean_preds + 1.96 * std_preds,
        color="tab:red",
        alpha=0.25,
        label="95% CI",
    )
    axes[0, 1].set(
        xlabel="x",
        ylabel="y",
        title=(
            "Validated coincidence regression\n"
            f"input_state={INPUT_STATE}, target_mode={TARGET_CC_PAIR}, encoding_phase_idx={ENCODING_PHASE_IDX}"
        ),
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].scatter(X_test, std_preds, c=np.abs(mean_preds - y_test), cmap="viridis")
    axes[1, 0].set(xlabel="x", ylabel="Std dev", title="Uncertainty vs. Input")
    axes[1, 0].grid(True)

    axes[1, 1].scatter(std_preds, np.abs(mean_preds - y_test), alpha=0.7)
    max_std = float(np.max(std_preds)) if len(std_preds) else 0.0
    axes[1, 1].plot([0.0, max_std], [0.0, 2.0 * max_std], "k--", label="y=2x")
    axes[1, 1].set(
        xlabel="Uncertainty (std)",
        ylabel="Absolute error",
        title=(
            "Calibration: RMSE=%.4f, R2=%.4f, coverage95=%.3f"
            % (metrics["rmse"], metrics["r2"], metrics["coverage_95"])
        ),
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment("Coincidence Regression", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))
        X_train, y_train, X_test, y_test = get_data(
            CONFIG["n_data"],
            CONFIG["sigma_noise"],
            CONFIG["data_function"],
        )
        enc_train = 2 * np.arccos(np.clip(X_train, 0.0, 1.0))
        enc_test = 2 * np.arccos(np.clip(X_test, 0.0, 1.0))
        sim_cfg = _sim_cfg()

        logger.info(
            "Training coincidence regression with input_state=%s, target_mode=%s, encoding_phase_idx=%d",
            INPUT_STATE,
            TARGET_CC_PAIR,
            ENCODING_PHASE_IDX,
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
        final_loss = float(history[-1])

        exp.save_metrics(
            {
                "target_cc_pair": TARGET_CC_PAIR,
                "input_state": INPUT_STATE,
                "encoding_phase_idx": ENCODING_PHASE_IDX,
                "final_loss": final_loss,
                **metrics,
            }
        )
        exp.savefig(_plot_training_loss(np.asarray(history)), "training_loss.png")
        exp.savefig(
            _plot_summary(
                np.asarray(history),
                X_train,
                y_train,
                X_test,
                y_test,
                unc["mean"],
                unc["std"],
                metrics,
            ),
            "coincidence_regression_with_uncertainty.png",
        )

        logger.info(
            "Validated coincidence regression: final_loss=%.6f mse=%.6f rmse=%.4f r2=%.4f",
            final_loss,
            metrics["mse"],
            metrics["rmse"],
            metrics["r2"],
        )


if __name__ == "__main__":
    main()
