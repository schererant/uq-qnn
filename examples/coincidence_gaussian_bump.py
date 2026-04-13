#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static vs memristive coincidence regression on a Gaussian bump target.

This is a compact comparison benchmark for an easier nonlinear target:
- one smooth localized bump instead of several separated peaks
- bounded in [0, 1]
- shared data split for both variants

The script runs both a static coincidence model and a memristive coincidence
model, writes a normal ``Experiment`` report for each, and saves one aggregate
comparison figure under ``reports/coincidence_gaussian_bump_comparison.png``.
"""

import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger

logger = get_logger(__name__)

INPUT_STATE = tuple(1 if i in (1, 6) else 0 for i in range(8))
TARGET_CC_PAIR = (0, 7)
ENCODING_PHASE_IDX = 17

BASE_CONFIG = {
    "n_modes": 8,
    "input_state": INPUT_STATE,
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "photon_distinguishability": "indistinguishable",
    "target_mode": TARGET_CC_PAIR,
    "output_mode": "coincidence",
    "working_detectors": tuple(range(8)),
    "loss_type": "mse",
    "n_classes": 1,
    "n_samples": 0,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    "sim_backend": "numpy",
    "n_data": 600,
    "sigma_noise": 0.003,
    "data_function": "gaussian_bump_data",
    "unc_n_passes": 20,
    "unc_noise_std": 0.03,
}

VARIANTS = {
    "static": {
        "title": "Coincidence Gaussian Bump (Static)",
        "color": "tab:blue",
        "config": {
            "memristive_phase_idx": None,
            "memristive_output_modes": None,
            "feedback_mode": "internal_arm",
            "feedback_modes": None,
            "computation_modes": None,
            "lr": 0.02,
            "epochs": 250,
            "memory_depth": 2,
        },
    },
    "memristive": {
        "title": "Coincidence Gaussian Bump (Memristive)",
        "color": "tab:red",
        "config": {
            "memristive_phase_idx": (11,),
            "memristive_output_modes": ((5, 6),),
            "feedback_mode": "chip_output",
            "feedback_modes": ((5, 6),),
            "computation_modes": TARGET_CC_PAIR,
            "lr": 0.015,
            "epochs": 300,
            "memory_depth": 4,
        },
    },
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


def _plot_single_summary(
    *,
    label: str,
    color: str,
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

    axes[0, 0].plot(history, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title=f"{label} Training Loss")
    axes[0, 0].grid(True)

    axes[0, 1].scatter(X_train, y_train, s=16, alpha=0.45, label="Training data")
    axes[0, 1].plot(X_test, y_test, "k--", lw=1.3, label="Ground truth")
    axes[0, 1].plot(X_test, mean_preds, color=color, lw=2, label="Mean prediction")
    axes[0, 1].fill_between(
        X_test,
        mean_preds - 1.96 * std_preds,
        mean_preds + 1.96 * std_preds,
        color=color,
        alpha=0.22,
        label="95% CI",
    )
    axes[0, 1].set(
        xlabel="x",
        ylabel="y",
        title=f"{label}\nRMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}",
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
        title=f"Coverage95={metrics['coverage_95']:.3f}, NLL={metrics['nll']:.3f}",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.tight_layout()
    return fig


def _plot_comparison(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results: dict[str, dict],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, (variant_name, result) in zip(axes[:2], results.items()):
        color = result["color"]
        metrics = result["metrics"]
        ax.scatter(X_train, y_train, s=14, alpha=0.4, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.3, label="Ground truth")
        ax.plot(X_test, result["mean"], color=color, lw=2, label=variant_name)
        ax.fill_between(
            X_test,
            result["mean"] - 1.96 * result["std"],
            result["mean"] + 1.96 * result["std"],
            color=color,
            alpha=0.2,
            label="95% CI",
        )
        ax.set_title(
            f"{variant_name}\nRMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
        )
        ax.set(xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[2]
    labels = list(results.keys())
    rmse_vals = [results[name]["metrics"]["rmse"] for name in labels]
    r2_vals = [results[name]["metrics"]["r2"] for name in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, rmse_vals, width, label="RMSE", color="tab:orange")
    ax.bar(x + width / 2, r2_vals, width, label="R2", color="tab:green")
    ax.set_xticks(x, labels)
    ax.set_title("Metric Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Gaussian Bump: Static vs Memristive Coincidence Regression", fontsize=13)
    fig.tight_layout()
    return fig


def main() -> None:
    np.random.seed(int(BASE_CONFIG["seed"]))
    X_train, y_train, X_test, y_test = get_data(
        int(BASE_CONFIG["n_data"]),
        float(BASE_CONFIG["sigma_noise"]),
        str(BASE_CONFIG["data_function"]),
    )
    enc_test = 2 * np.arccos(np.clip(X_test, 0.0, 1.0))

    results: dict[str, dict] = {}

    for variant_name, variant in VARIANTS.items():
        config = dict(BASE_CONFIG)
        config.update(variant["config"])
        title = str(variant["title"])
        color = str(variant["color"])

        with Experiment(title, config=config) as exp:
            trained_state, history, _ = exp.train(X_train, y_train)
            unc = exp.run_uncertainty_analysis(
                trained_state,
                enc_test,
                n_passes=int(config["unc_n_passes"]),
                noise_std=float(config["unc_noise_std"]),
            )
            metrics = _compute_metrics(y_test, unc["mean"], unc["std"])

            state_path = exp.run_dir / "trained_state.json"
            trained_state.save_json(state_path)
            exp.artifacts.append(str(state_path))

            exp.save_metrics(
                {
                    "variant": variant_name,
                    "input_state": INPUT_STATE,
                    "target_mode": TARGET_CC_PAIR,
                    "encoding_phase_idx": ENCODING_PHASE_IDX,
                    "final_loss": float(history[-1]),
                    **metrics,
                }
            )

            fig = _plot_single_summary(
                label=variant_name.capitalize(),
                color=color,
                history=np.asarray(history),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                mean_preds=unc["mean"],
                std_preds=unc["std"],
                metrics=metrics,
            )
            exp.savefig(fig, f"gaussian_bump_{variant_name}.png")
            plt.close(fig)

            results[variant_name] = {
                "color": color,
                "history": history,
                "mean": unc["mean"],
                "std": unc["std"],
                "metrics": metrics,
            }

            logger.info(
                "[%s] final_loss=%.6f rmse=%.4f r2=%.4f",
                variant_name,
                float(history[-1]),
                metrics["rmse"],
                metrics["r2"],
            )

    fig = _plot_comparison(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        results=results,
    )
    out_path = "reports/coincidence_gaussian_bump_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved aggregate comparison to %s", out_path)


if __name__ == "__main__":
    main()
