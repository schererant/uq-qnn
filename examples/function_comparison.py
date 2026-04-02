#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function and Measurement Mode Comparison Experiment.

Trains a photonic QNN on multiple synthetic regression functions and
evaluates each under two quantum measurement strategies:

  • singles   — single-photon output probabilities (Born rule on |ψ|²)
  • coincidence — two-photon correlations (post-selected coincidence counts)

Coincidence measurements are fundamentally different: they are inherently
two-photon events and, in physical hardware, naturally reject single-photon
noise and dark counts.  Comparing them reveals how the measurement basis
affects the circuit's expressivity and uncertainty calibration.

Produced artefacts
------------------
predictions_grid.png        Fit + 95 % CI for every (function, mode) pair
training_loss.png           Log-scale convergence curves, singles vs coincidence
calibration.png             σ vs |error| scatter — ideal calibration if σ ≈ |error|
metrics_comparison.png      Grouped bar charts: RMSE, R², coverage, NLL
r2_heatmap.png              R² heatmap (functions × modes) — quick overview
uncertainty_distribution.png Violin plots of per-point σ, one violin per mode
metrics_table.png           Full numeric summary table
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

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

_COMMON: Dict = {
    # Circuit
    "n_modes": 6,
    "encoding_mode": 0,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "encoding_phase_idx": None,
    "n_photons": None,
    # Training
    "memory_depth": 1,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "n_samples": 300,
    "noise_std": None,
    "loss_type": "mse",
    "n_classes": 1,
    "sim_backend": "numpy",
    "seed": 42,
    "lr": 0.05,
    "epochs": 150,
}

# Singles: read one output detector (last mode)
SINGLES_CONFIG: Dict = {
    **_COMMON,
    "output_mode": "singles",
    "target_mode": (5,),
    "input_modes": None,
    "working_detectors": None,
}

# Coincidence: inject photons at modes 1 and 4, read all CC channels
COINCIDENCE_CONFIG: Dict = {
    **_COMMON,
    "output_mode": "coincidence",
    "target_mode": None,
    "input_modes": (1, 4),
    "working_detectors": tuple(range(6)),
}

# Data
N_DATA = 80
SIGMA_NOISE = 0.02

# Uncertainty quantification
UNC_N_PASSES = 15
UNC_NOISE_STD = 0.05

# Functions and labels
FUNCTIONS = [
    "quartic_data",
    "sinusoid_data",
    "multi_modal_data",
    "step_function_data",
]

FUNC_LABELS = {
    "quartic_data": r"$x^4$",
    "sinusoid_data": r"$\sin(0.7\pi x)$",
    "multi_modal_data": "Multi-modal Gaussians",
    "step_function_data": "Smooth step",
}

MEASUREMENT_MODES = ["singles", "coincidence"]
MODE_COLORS = {"singles": "#1f77b4", "coincidence": "#ff7f0e"}

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _sim_cfg(mode: str) -> SimConfig:
    cfg = SINGLES_CONFIG if mode == "singles" else COINCIDENCE_CONFIG
    return SimConfig.from_experiment_config(cfg)


def _run_uncertainty(
    theta: np.ndarray,
    enc_test: np.ndarray,
    sim_cfg: SimConfig,
    *,
    n_passes: int,
    noise_std: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Monte-Carlo uncertainty: perturb phase parameters, aggregate predictions."""
    rng = np.random.default_rng(seed)
    n = len(enc_test)
    all_preds = np.zeros((n, n_passes))

    for i in range(n_passes):
        sample_count = max(100, sim_cfg.n_samples + int(rng.integers(-50, 51)))
        perturbed = theta + rng.normal(0, noise_std, size=len(theta))
        preds = run_simulation_sequence_np(
            perturbed, enc_test, sim_cfg.replace(n_samples=sample_count)
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
    """
    Compute regression and calibration metrics.

    Returns
    -------
    rmse            Root-mean-square error
    mae             Mean absolute error
    r2              Coefficient of determination R² ∈ (-∞, 1]
    coverage_95     Fraction of test points inside ŷ ± 1.96σ  (ideal ≈ 0.95)
    mean_std        Average predictive uncertainty σ  (sharpness — lower = more confident)
    nll             Gaussian NLL = ½·((y-ŷ)/σ)² + log σ  (lower = better joint accuracy+calibration)
    calibration_rho Spearman ρ between σ and |error|  (> 0 = uncertainty tracks error)
    """
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
    mode: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Train one (function, mode) pair; return predictions, history and metrics."""
    logger.info("▶ Training  func=%-22s  mode=%s", func_name, mode)
    sc = _sim_cfg(mode)
    enc_train = 2 * np.arccos(np.clip(X_train, 0, 1))

    theta, history = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sc,
        lr=_COMMON["lr"],
        epochs=_COMMON["epochs"],
        seed=_COMMON["seed"],
    )
    logger.info("  final loss: %.6f", history[-1])

    enc_test = 2 * np.arccos(np.clip(X_test, 0, 1))
    unc = _run_uncertainty(
        theta,
        enc_test,
        sc,
        n_passes=UNC_N_PASSES,
        noise_std=UNC_NOISE_STD,
        seed=_COMMON["seed"],
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


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════


def plot_predictions_grid(
    results: Dict,
    data: Dict,
    functions: list,
    modes: list,
) -> plt.Figure:
    """Fit + 95 % CI for every (function, mode) pair."""
    n_f, n_m = len(functions), len(modes)
    fig, axes = plt.subplots(n_f, n_m, figsize=(6 * n_m, 4 * n_f))
    axes = np.atleast_2d(axes)

    for i, func in enumerate(functions):
        X_train, y_train, X_test, y_test = data[func]
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            r = results[func][mode]
            m = r["metrics"]

            ax.scatter(X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train")
            ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
            ax.plot(
                X_test, r["mean_preds"],
                color=MODE_COLORS[mode], lw=2.0, label="Prediction",
            )
            ax.fill_between(
                X_test,
                r["mean_preds"] - 1.96 * r["std_preds"],
                r["mean_preds"] + 1.96 * r["std_preds"],
                color=MODE_COLORS[mode], alpha=0.22, label="95 % CI",
            )
            ax.set_title(
                f"{FUNC_LABELS[func]}  ·  {mode}\n"
                f"RMSE={m['rmse']:.4f}   R²={m['r2']:.3f}   Cov={m['coverage_95']:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Photonic QNN Regression — Functions × Measurement Modes", fontsize=13, y=1.01
    )
    fig.tight_layout()
    return fig


def plot_training_loss(results: Dict, functions: list, modes: list) -> plt.Figure:
    """Log-scale loss curves — visually reveals convergence speed and final loss."""
    n_f = len(functions)
    fig, axes = plt.subplots(1, n_f, figsize=(5 * n_f, 4))
    axes = np.atleast_1d(axes)

    for ax, func in zip(axes, functions):
        for mode in modes:
            ax.plot(
                results[func][mode]["history"],
                color=MODE_COLORS[mode],
                lw=1.6,
                alpha=0.85,
                label=mode,
            )
        ax.set_yscale("log")
        ax.set_title(FUNC_LABELS[func])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Training Convergence: Singles vs Coincidence", fontsize=12
    )
    fig.tight_layout()
    return fig


def plot_calibration(
    results: Dict,
    data: Dict,
    functions: list,
    modes: list,
) -> plt.Figure:
    """
    Calibration scatter: predictive σ vs absolute error.

    A well-calibrated model has points near the identity line (σ ≈ |error|).
    Points below the line indicate under-confidence; above indicates overconfidence.
    The 2σ band should contain ≈95 % of data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for j, mode in enumerate(modes):
        ax = axes[j]
        all_stds: list = []
        all_errs: list = []
        for func in functions:
            _, _, X_test, y_test = data[func]
            r = results[func][mode]
            stds = r["std_preds"]
            errs = np.abs(r["mean_preds"] - y_test)
            ax.scatter(stds, errs, alpha=0.45, s=9, label=FUNC_LABELS[func])
            all_stds.extend(stds.tolist())
            all_errs.extend(errs.tolist())

        max_val = max(float(np.max(all_stds)), float(np.max(all_errs))) * 1.05
        ax.plot(
            [0, max_val], [0, max_val], "k-", lw=1.0, alpha=0.5, label="σ = |err| (ideal)"
        )
        ax.plot(
            [0, max_val], [0, 1.96 * max_val], "k--", lw=0.8, alpha=0.35,
            label="1.96σ (95 % bound)",
        )

        rho, _ = stats.spearmanr(all_stds, all_errs)
        ax.set_title(
            f"{mode.capitalize()} — Calibration\nSpearman ρ = {rho:.3f}  (↑ = σ tracks |error|)",
            fontsize=10,
        )
        ax.set_xlabel("Predictive uncertainty σ")
        ax.set_ylabel("Absolute error |y − ŷ|")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("Uncertainty Calibration by Measurement Mode", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(results: Dict, functions: list, modes: list) -> plt.Figure:
    """
    Grouped bar chart for RMSE, R², coverage@95 %, NLL.

    Placing singles and coincidence side-by-side per function highlights
    whether coincidence's quantum correlations genuinely improve fit quality.
    """
    metrics_spec = [
        ("rmse", "RMSE  (↓ better)"),
        ("r2", "R²  (↑ better)"),
        ("coverage_95", "Coverage @ 95 %  (→ 0.95)"),
        ("nll", "NLL  (↓ better)"),
    ]
    n_m = len(metrics_spec)
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5))

    x = np.arange(len(functions))
    width = 0.35

    for ax, (key, label) in zip(axes, metrics_spec):
        for k, mode in enumerate(modes):
            vals = [results[f][mode]["metrics"][key] for f in functions]
            bars = ax.bar(
                x + (k - 0.5) * width,
                vals,
                width,
                label=mode,
                color=MODE_COLORS[mode],
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

    fig.suptitle("Performance Metrics: Singles vs Coincidence", fontsize=12)
    fig.tight_layout()
    return fig


def plot_r2_heatmap(results: Dict, functions: list, modes: list) -> plt.Figure:
    """
    R² heatmap (functions × modes).

    Provides an at-a-glance view of which (function, mode) combinations
    succeed or fail.  R² < 0 means the model is worse than the mean baseline.
    """
    matrix = np.array(
        [[results[f][m]["metrics"]["r2"] for m in modes] for f in functions]
    )
    fig, ax = plt.subplots(figsize=(4, 1 + len(functions)))
    im = ax.imshow(matrix, vmin=-0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="R²", fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_yticks(range(len(functions)))
    ax.set_yticklabels([FUNC_LABELS[f] for f in functions], fontsize=10)
    ax.set_title("R² Score Heatmap\n(functions × measurement mode)", fontsize=11)

    for i in range(len(functions)):
        for j in range(len(modes)):
            val = matrix[i, j]
            txt_color = "white" if abs(val) > 0.65 else "black"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                color=txt_color, fontsize=11, fontweight="bold",
            )

    fig.tight_layout()
    return fig


def plot_uncertainty_distribution(results: Dict, functions: list, modes: list) -> plt.Figure:
    """
    Violin plots of per-point predictive σ.

    Wider violins → more spread in σ across test points.
    Coincidence measurements may show qualitatively different σ distributions
    due to the normalisation of coincidence probabilities.
    """
    n_f = len(functions)
    fig, axes = plt.subplots(1, n_f, figsize=(4 * n_f, 4))
    axes = np.atleast_1d(axes)

    for ax, func in zip(axes, functions):
        plot_data = [results[func][m]["std_preds"] for m in modes]
        parts = ax.violinplot(
            plot_data,
            positions=range(len(modes)),
            showmedians=True,
            showextrema=True,
        )
        for pc, mode in zip(parts["bodies"], modes):
            pc.set_facecolor(MODE_COLORS[mode])
            pc.set_alpha(0.72)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, fontsize=10)
        ax.set_title(FUNC_LABELS[func], fontsize=10)
        ax.set_ylabel("Prediction σ")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Uncertainty Distribution by Measurement Mode", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metrics_table(results: Dict, functions: list, modes: list) -> plt.Figure:
    """Full numeric summary table for all (function, mode) pairs."""
    columns = [
        "Function", "Mode",
        "RMSE ↓", "MAE ↓", "R² ↑",
        "Cov@95 %", "Mean σ", "NLL ↓", "Cal. ρ ↑",
    ]
    rows = []
    for func in functions:
        for mode in modes:
            m = results[func][mode]["metrics"]
            rows.append([
                FUNC_LABELS[func],
                mode,
                f"{m['rmse']:.4f}",
                f"{m['mae']:.4f}",
                f"{m['r2']:.3f}",
                f"{m['coverage_95']:.3f}",
                f"{m['mean_std']:.4f}",
                f"{m['nll']:.3f}",
                f"{m['calibration_rho']:.3f}",
            ])

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
        color = "#daeeff" if row[1] == "singles" else "#ffe5cc"
        for c_idx in range(len(columns)):
            tbl[(r_idx + 1, c_idx)].set_facecolor(color)

    ax.set_title(
        "Comprehensive Metrics Summary — Functions × Measurement Modes",
        fontsize=11, pad=18,
    )
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    with Experiment(
        "Function and Measurement Comparison",
        config=SINGLES_CONFIG,
    ) as exp:
        np.random.seed(_COMMON["seed"])

        # ── Load data ─────────────────────────────────────────────────────
        data: Dict[str, Tuple] = {}
        for func_name in FUNCTIONS:
            data[func_name] = get_data(N_DATA, SIGMA_NOISE, func_name)
            logger.info(
                "Data loaded: %-22s  train=%d  test=%d",
                func_name,
                len(data[func_name][0]),
                len(data[func_name][2]),
            )

        # ── Train all (function, mode) combinations ────────────────────────
        results: Dict = {}
        all_metrics: Dict = {}

        for func_name in FUNCTIONS:
            results[func_name] = {}
            all_metrics[func_name] = {}
            X_train, y_train, X_test, y_test = data[func_name]
            for mode in MEASUREMENT_MODES:
                r = train_and_evaluate(
                    func_name, mode, X_train, y_train, X_test, y_test
                )
                results[func_name][mode] = r
                all_metrics[func_name][mode] = r["metrics"]

        exp.save_metrics({"by_function_and_mode": all_metrics})

        # ── Plots ──────────────────────────────────────────────────────────
        logger.info("Generating figures…")

        figs = [
            ("predictions_grid.png", plot_predictions_grid(results, data, FUNCTIONS, MEASUREMENT_MODES)),
            ("training_loss.png", plot_training_loss(results, FUNCTIONS, MEASUREMENT_MODES)),
            ("calibration.png", plot_calibration(results, data, FUNCTIONS, MEASUREMENT_MODES)),
            ("metrics_comparison.png", plot_metrics_comparison(results, FUNCTIONS, MEASUREMENT_MODES)),
            ("r2_heatmap.png", plot_r2_heatmap(results, FUNCTIONS, MEASUREMENT_MODES)),
            ("uncertainty_distribution.png", plot_uncertainty_distribution(results, FUNCTIONS, MEASUREMENT_MODES)),
            ("metrics_table.png", plot_metrics_table(results, FUNCTIONS, MEASUREMENT_MODES)),
        ]

        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("  saved %s", fname)

        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
