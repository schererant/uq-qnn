#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function and Memristor Comparison Experiment.

Compares the same photonic regression task with two circuit variants:

  - standard   : plain 6-mode Clements mesh
  - memristor  : same mesh with one memristive phase and feedback memory

For the 6-mode mesh, the memristor is placed at phase index 6, i.e. the first
phase of the 4th MZI, which couples modes 1 and 2. With
memristive_output_modes left as None, feedback is taken from that MZI's own
output modes by default.

The experiment mirrors `examples/function_comparison.py`, but keeps the
measurement fixed to singles so that the only change is the presence of the
memristive element.
"""

import os
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
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

# ===========================================================================
# CONFIGURATION
# ===========================================================================

CONFIG: Dict = {
    "n_modes": 6,
    "input_state": (0, 1, 0, 0, 1, 0),
    "encoding_phase_idx": 7,
    "photon_distinguishability": None,
    "target_mode": (4,),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "singles",
    "working_detectors": (0, 1, 2, 3, 4, 5),
    "n_samples": 300,
    "noise_std": None,
    "loss_type": "mse",
    "n_classes": 1,
    "sim_backend": "numpy",
    "seed": 42,
    "lr": 0.04,
    "epochs": 180,
    "memory_depth": 1,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "n_data": 80,
    "sigma_noise": 0.02,
    "unc_n_passes": 15,
    "unc_noise_std": 0.05,
    "data_functions": [
        "quartic_data",
        "sinusoid_data",
        "multi_modal_data",
        "step_function_data",
    ],
    "func_labels": {
        "quartic_data": r"$x^4$",
        "sinusoid_data": r"$\sin(0.7\pi x)$",
        "multi_modal_data": "Multi-modal Gaussians",
        "step_function_data": "Smooth step",
    },
    "architectures": ["standard", "memristor"],
    "arch_labels": {"standard": "Standard", "memristor": "Memristor"},
    "arch_colors": {"standard": "#1f77b4", "memristor": "#d62728"},
}

_MEMRISTOR_OVERRIDES: Dict = {
    "memristive_phase_idx": (6,),
    "memristive_output_modes": None,
    "memory_depth": 2,
}

FUNCTIONS = CONFIG["data_functions"]
FUNC_LABELS = CONFIG["func_labels"]
ARCHITECTURES = CONFIG["architectures"]
ARCH_LABELS = CONFIG["arch_labels"]
ARCH_COLORS = CONFIG["arch_colors"]


# ===========================================================================
# HELPERS
# ===========================================================================


def _sim_cfg(architecture: str) -> SimConfig:
    d = dict(CONFIG)
    if architecture == "memristor":
        d.update(_MEMRISTOR_OVERRIDES)
    return SimConfig.from_experiment_config(d)


def _run_uncertainty(
    theta: np.ndarray,
    enc_test: np.ndarray,
    sim_cfg: SimConfig,
    *,
    n_passes: int,
    noise_std: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Monte-Carlo uncertainty via parameter perturbations."""
    rng = np.random.default_rng(seed)
    n = len(enc_test)
    all_preds = np.zeros((n, n_passes))

    for i in range(n_passes):
        sample_count = max(100, sim_cfg.n_samples + int(rng.integers(-50, 51)))
        perturbed = theta + rng.normal(0, noise_std, size=len(theta))
        preds = run_simulation_sequence_np(
            perturbed,
            enc_test,
            sim_cfg.replace(n_samples=sample_count),
        )
        all_preds[:, i] = preds

    return {
        "mean": np.mean(all_preds, axis=1),
        "std": np.std(all_preds, axis=1),
        "all_preds": all_preds,
    }


def compute_metrics(
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

    mean_std = float(np.mean(std_preds))
    sigma = np.clip(std_preds, 1e-6, None)
    nll = float(np.mean(0.5 * (residuals / sigma) ** 2 + np.log(sigma)))

    cal_rho, _ = stats.spearmanr(std_preds, np.abs(residuals))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coverage_95": coverage_95,
        "mean_std": mean_std,
        "nll": nll,
        "calibration_rho": float(cal_rho),
    }


def train_and_evaluate(
    func_name: str,
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    logger.info("▶ Training  func=%-22s  architecture=%s", func_name, architecture)
    sc = _sim_cfg(architecture)
    enc_train = 2 * np.arccos(np.clip(X_train, 0, 1))

    theta, history = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sc,
        lr=float(CONFIG["lr"]),
        epochs=int(CONFIG["epochs"]),
        seed=int(CONFIG["seed"]),
    )
    logger.info("  final loss: %.6f", history[-1])

    enc_test = 2 * np.arccos(np.clip(X_test, 0, 1))
    unc = _run_uncertainty(
        theta,
        enc_test,
        sc,
        n_passes=int(CONFIG["unc_n_passes"]),
        noise_std=float(CONFIG["unc_noise_std"]),
        seed=int(CONFIG["seed"]),
    )
    metrics = compute_metrics(y_test, unc["mean"], unc["std"])
    logger.info(
        "  RMSE=%.4f  R²=%.4f  Coverage=%.2f  NLL=%.3f  Cal.ρ=%.3f",
        metrics["rmse"],
        metrics["r2"],
        metrics["coverage_95"],
        metrics["nll"],
        metrics["calibration_rho"],
    )
    return {
        "theta": theta,
        "history": history,
        "mean_preds": unc["mean"],
        "std_preds": unc["std"],
        "metrics": metrics,
    }


# ===========================================================================
# PLOTS
# ===========================================================================


def plot_predictions_grid(
    results: Dict,
    data: Dict,
    functions: list,
    architectures: list,
) -> plt.Figure:
    n_f, n_a = len(functions), len(architectures)
    fig, axes = plt.subplots(n_f, n_a, figsize=(6 * n_a, 4 * n_f))
    axes = np.atleast_2d(axes)

    for i, func in enumerate(functions):
        X_train, y_train, X_test, y_test = data[func]
        for j, architecture in enumerate(architectures):
            ax = axes[i, j]
            r = results[func][architecture]
            m = r["metrics"]

            ax.scatter(
                X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train"
            )
            ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
            ax.plot(
                X_test,
                r["mean_preds"],
                color=ARCH_COLORS[architecture],
                lw=2.0,
                label="Prediction",
            )
            ax.fill_between(
                X_test,
                r["mean_preds"] - 1.96 * r["std_preds"],
                r["mean_preds"] + 1.96 * r["std_preds"],
                color=ARCH_COLORS[architecture],
                alpha=0.22,
                label="95 % CI",
            )
            ax.set_title(
                f"{FUNC_LABELS[func]}  ·  {ARCH_LABELS[architecture]}\n"
                f"RMSE={m['rmse']:.4f}   R²={m['r2']:.3f}   Cov={m['coverage_95']:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Photonic QNN Regression - Functions x Circuit Architecture",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_training_loss(
    results: Dict, functions: list, architectures: list
) -> plt.Figure:
    n_f = len(functions)
    fig, axes = plt.subplots(1, n_f, figsize=(5 * n_f, 4))
    axes = np.atleast_1d(axes)

    for ax, func in zip(axes, functions):
        for architecture in architectures:
            ax.plot(
                results[func][architecture]["history"],
                color=ARCH_COLORS[architecture],
                lw=1.6,
                alpha=0.85,
                label=ARCH_LABELS[architecture],
            )
        ax.set_yscale("log")
        ax.set_title(FUNC_LABELS[func])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Training Convergence: Standard vs Memristor", fontsize=12)
    fig.tight_layout()
    return fig


def plot_calibration(
    results: Dict,
    data: Dict,
    functions: list,
    architectures: list,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(architectures), figsize=(6 * len(architectures), 5))
    axes = np.atleast_1d(axes)

    for j, architecture in enumerate(architectures):
        ax = axes[j]
        all_stds = []
        all_errs = []
        for func in functions:
            _, _, _, y_test = data[func]
            r = results[func][architecture]
            stds = r["std_preds"]
            errs = np.abs(r["mean_preds"] - y_test)
            ax.scatter(stds, errs, alpha=0.45, s=9, label=FUNC_LABELS[func])
            all_stds.extend(stds.tolist())
            all_errs.extend(errs.tolist())

        max_val = max(float(np.max(all_stds)), float(np.max(all_errs))) * 1.05
        ax.plot([0, max_val], [0, max_val], "k-", lw=1.0, alpha=0.5, label="σ = |err|")
        ax.plot(
            [0, max_val], [0, 1.96 * max_val], "k--", lw=0.8, alpha=0.35, label="1.96σ"
        )

        rho, _ = stats.spearmanr(all_stds, all_errs)
        ax.set_title(
            f"{ARCH_LABELS[architecture]} calibration\nSpearman ρ = {rho:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Predictive uncertainty σ")
        ax.set_ylabel("Absolute error |y - ŷ|")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("Uncertainty Calibration by Circuit Architecture", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(
    results: Dict, functions: list, architectures: list
) -> plt.Figure:
    metrics_spec = [
        ("rmse", "RMSE  (down better)"),
        ("r2", "R²  (up better)"),
        ("coverage_95", "Coverage @ 95 %  (to 0.95)"),
        ("nll", "NLL  (down better)"),
    ]
    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(5 * len(metrics_spec), 5))

    x = np.arange(len(functions))
    width = 0.35

    for ax, (key, label) in zip(axes, metrics_spec):
        for k, architecture in enumerate(architectures):
            vals = [results[f][architecture]["metrics"][key] for f in functions]
            bars = ax.bar(
                x + (k - 0.5) * width,
                vals,
                width,
                label=ARCH_LABELS[architecture],
                color=ARCH_COLORS[architecture],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.02,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=40,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [FUNC_LABELS[f] for f in functions], rotation=20, ha="right", fontsize=9
        )
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Performance Metrics: Standard vs Memristor", fontsize=12)
    fig.tight_layout()
    return fig


def plot_r2_heatmap(results: Dict, functions: list, architectures: list) -> plt.Figure:
    matrix = np.array(
        [[results[f][a]["metrics"]["r2"] for a in architectures] for f in functions]
    )
    fig, ax = plt.subplots(figsize=(4, 1 + len(functions)))
    im = ax.imshow(matrix, vmin=-0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="R²", fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels([ARCH_LABELS[a] for a in architectures], fontsize=11)
    ax.set_yticks(range(len(functions)))
    ax.set_yticklabels([FUNC_LABELS[f] for f in functions], fontsize=10)
    ax.set_title("R² Score Heatmap\n(functions x circuit architecture)", fontsize=11)

    for i in range(len(functions)):
        for j in range(len(architectures)):
            val = matrix[i, j]
            txt_color = "white" if abs(val) > 0.65 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


def plot_uncertainty_distribution(
    results: Dict,
    functions: list,
    architectures: list,
) -> plt.Figure:
    n_f = len(functions)
    fig, axes = plt.subplots(1, n_f, figsize=(4 * n_f, 4))
    axes = np.atleast_1d(axes)

    for ax, func in zip(axes, functions):
        plot_data = [results[func][a]["std_preds"] for a in architectures]
        parts = ax.violinplot(
            plot_data,
            positions=range(len(architectures)),
            showmedians=True,
            showextrema=True,
        )
        for pc, architecture in zip(parts["bodies"], architectures):
            pc.set_facecolor(ARCH_COLORS[architecture])
            pc.set_alpha(0.72)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(architectures)))
        ax.set_xticklabels([ARCH_LABELS[a] for a in architectures], fontsize=10)
        ax.set_title(FUNC_LABELS[func], fontsize=10)
        ax.set_ylabel("Prediction σ")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Uncertainty Distribution by Circuit Architecture", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metrics_table(
    results: Dict, functions: list, architectures: list
) -> plt.Figure:
    columns = [
        "Function",
        "Architecture",
        "RMSE ↓",
        "MAE ↓",
        "R² ↑",
        "Cov@95 %",
        "Mean σ",
        "NLL ↓",
        "Cal. ρ ↑",
    ]
    rows = []
    for func in functions:
        for architecture in architectures:
            m = results[func][architecture]["metrics"]
            rows.append(
                [
                    FUNC_LABELS[func],
                    ARCH_LABELS[architecture],
                    f"{m['rmse']:.4f}",
                    f"{m['mae']:.4f}",
                    f"{m['r2']:.3f}",
                    f"{m['coverage_95']:.3f}",
                    f"{m['mean_std']:.4f}",
                    f"{m['nll']:.3f}",
                    f"{m['calibration_rho']:.3f}",
                ]
            )

    fig, ax = plt.subplots(figsize=(15, 0.52 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)

    for r_idx, row in enumerate(rows):
        color = "#daeeff" if row[1] == ARCH_LABELS["standard"] else "#ffd9d9"
        for c_idx in range(len(columns)):
            tbl[(r_idx + 1, c_idx)].set_facecolor(color)

    ax.set_title(
        "Comprehensive Metrics Summary - Functions x Circuit Architecture",
        fontsize=11,
        pad=18,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    with Experiment(
        "Function and Memristor Comparison",
        config=CONFIG,
    ) as exp:
        np.random.seed(int(CONFIG["seed"]))

        data: Dict[str, Tuple] = {}
        for func_name in FUNCTIONS:
            data[func_name] = get_data(
                int(CONFIG["n_data"]),
                float(CONFIG["sigma_noise"]),
                func_name,
            )
            logger.info(
                "Data loaded: %-22s  train=%d  test=%d",
                func_name,
                len(data[func_name][0]),
                len(data[func_name][2]),
            )

        results: Dict = {}
        all_metrics: Dict = {}

        for func_name in FUNCTIONS:
            results[func_name] = {}
            all_metrics[func_name] = {}
            X_train, y_train, X_test, y_test = data[func_name]
            for architecture in ARCHITECTURES:
                r = train_and_evaluate(
                    func_name,
                    architecture,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
                results[func_name][architecture] = r
                all_metrics[func_name][architecture] = r["metrics"]

        exp.save_metrics({"by_function_and_architecture": all_metrics})

        logger.info("Generating figures...")

        figs = [
            (
                "predictions_grid.png",
                plot_predictions_grid(results, data, FUNCTIONS, ARCHITECTURES),
            ),
            (
                "training_loss.png",
                plot_training_loss(results, FUNCTIONS, ARCHITECTURES),
            ),
            (
                "calibration.png",
                plot_calibration(results, data, FUNCTIONS, ARCHITECTURES),
            ),
            (
                "metrics_comparison.png",
                plot_metrics_comparison(results, FUNCTIONS, ARCHITECTURES),
            ),
            (
                "r2_heatmap.png",
                plot_r2_heatmap(results, FUNCTIONS, ARCHITECTURES),
            ),
            (
                "uncertainty_distribution.png",
                plot_uncertainty_distribution(results, FUNCTIONS, ARCHITECTURES),
            ),
            (
                "metrics_table.png",
                plot_metrics_table(results, FUNCTIONS, ARCHITECTURES),
            ),
        ]

        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("  saved %s", fname)

        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
