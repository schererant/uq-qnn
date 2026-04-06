#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Mode Comparison Experiment.

Trains a photonic QNN on one synthetic regression function and evaluates
the result using all six available singles output detectors (modes 0–5) of
the 6-mode Clements mesh.

Each run is identical except for which detector is read out (`target_mode`).
This isolates the effect of readout-mode selection: different detectors access
different linear combinations of the circuit's unitary, so some modes may be
more expressive for a given function than others.

Produced artefacts
------------------
predictions_grid.png        Fit + 95 % CI for every output mode
training_loss.png           Log-scale convergence curves, one line per mode
calibration.png             σ vs |error| scatter — ideal calibration if σ ≈ |error|
metrics_comparison.png      Grouped bar charts: RMSE, R², coverage, NLL
r2_bar.png                  R² by output mode — quick overview of best detectors
uncertainty_distribution.png Violin plots of per-point σ, one violin per mode
metrics_table.png           Full numeric summary table
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
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

N_MODES = 12
OUTPUT_MODES: List[int] = list(range(N_MODES))  # 0, 1, 2, 3, 4, 5

_COMMON: Dict = {
    "n_modes": N_MODES,
    "encoding_mode": 0,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "encoding_phase_idx": None,
    "n_photons": None,
    "output_mode": "singles",
    "input_modes": None,
    "working_detectors": None,
    "n_samples": 300,
    "noise_std": None,
    "loss_type": "mse",
    "n_classes": 1,
    "memory_depth": 1,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "sim_backend": "numpy",
    "seed": 42,
    "lr": 0.05,
    "epochs": 150,
}

# The single function under investigation
FUNCTION = "sinusoid_data"
FUNC_LABEL = r"$\sin(0.7\pi x)$"

N_DATA = 80
SIGMA_NOISE = 0.02
UNC_N_PASSES = 15
UNC_NOISE_STD = 0.05

# Distinct colour for each of the 6 output modes
_CMAP = cm.get_cmap("tab10", N_MODES)
MODE_COLORS: Dict[int, str] = {m: _CMAP(m) for m in OUTPUT_MODES}


def _mode_label(m: int) -> str:
    return f"mode {m}"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _sim_cfg(target_mode: int) -> SimConfig:
    cfg = {**_COMMON, "target_mode": (target_mode,)}
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
    residuals = mean_preds - y_test
    rmse = float(np.sqrt(np.mean(residuals**2)))
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
    target_mode: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    logger.info("▶ Training  target_mode=%d", target_mode)
    sc = _sim_cfg(target_mode)
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    modes: List[int],
) -> plt.Figure:
    """Fit + 95 % CI for every output mode, 2-column layout."""
    n_cols = 2
    n_rows = (len(modes) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, m in enumerate(modes):
        ax = axes_flat[idx]
        r = results[m]
        met = r["metrics"]
        color = MODE_COLORS[m]

        ax.scatter(X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
        ax.plot(X_test, r["mean_preds"], color=color, lw=2.0, label="Prediction")
        ax.fill_between(
            X_test,
            r["mean_preds"] - 1.96 * r["std_preds"],
            r["mean_preds"] + 1.96 * r["std_preds"],
            color=color,
            alpha=0.22,
            label="95 % CI",
        )
        ax.set_title(
            f"Output mode {m}\n"
            f"RMSE={met['rmse']:.4f}   R²={met['r2']:.3f}   Cov={met['coverage_95']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide any unused subplot
    for ax in axes_flat[len(modes) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Photonic QNN Regression — {FUNC_LABEL} across All Output Modes",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_training_loss(results: Dict, modes: List[int]) -> plt.Figure:
    """Log-scale convergence curves for all 6 output modes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in modes:
        ax.plot(
            results[m]["history"],
            color=MODE_COLORS[m],
            lw=1.5,
            alpha=0.85,
            label=_mode_label(m),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Convergence — {FUNC_LABEL}", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_calibration(
    results: Dict,
    y_test: np.ndarray,
    modes: List[int],
) -> plt.Figure:
    """σ vs |error| scatter for all modes on one axes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    all_stds: list = []
    all_errs: list = []

    for m in modes:
        r = results[m]
        stds = r["std_preds"]
        errs = np.abs(r["mean_preds"] - y_test)
        ax.scatter(
            stds, errs, alpha=0.45, s=12, color=MODE_COLORS[m], label=_mode_label(m)
        )
        all_stds.extend(stds.tolist())
        all_errs.extend(errs.tolist())

    max_val = max(float(np.max(all_stds)), float(np.max(all_errs))) * 1.05
    ax.plot(
        [0, max_val], [0, max_val], "k-", lw=1.0, alpha=0.5, label="σ = |err| (ideal)"
    )
    ax.plot(
        [0, max_val],
        [0, 1.96 * max_val],
        "k--",
        lw=0.8,
        alpha=0.35,
        label="1.96σ (95 % bound)",
    )

    rho, _ = stats.spearmanr(all_stds, all_errs)
    ax.set_title(
        f"Uncertainty Calibration — {FUNC_LABEL}\nSpearman ρ (all modes) = {rho:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Predictive uncertainty σ")
    ax.set_ylabel("Absolute error |y − ŷ|")
    ax.legend(fontsize=8, markerscale=2, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(results: Dict, modes: List[int]) -> plt.Figure:
    """Grouped bar chart for RMSE, R², coverage@95 %, NLL across all modes."""
    metrics_spec = [
        ("rmse", "RMSE  (↓ better)"),
        ("r2", "R²  (↑ better)"),
        ("coverage_95", "Coverage @ 95 %  (→ 0.95)"),
        ("nll", "NLL  (↓ better)"),
    ]
    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(5 * len(metrics_spec), 5))

    x = np.arange(len(modes))
    colors = [MODE_COLORS[m] for m in modes]

    for ax, (key, label) in zip(axes, metrics_spec):
        vals = [results[m]["metrics"][key] for m in modes]
        bars = ax.bar(
            x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=35,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"mode {m}" for m in modes], rotation=25, ha="right", fontsize=9
        )
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Performance Metrics by Output Mode — {FUNC_LABEL}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_r2_bar(results: Dict, modes: List[int]) -> plt.Figure:
    """Horizontal bar chart of R² sorted by score — quick best-detector view."""
    r2_vals = np.array([results[m]["metrics"]["r2"] for m in modes])
    order = np.argsort(r2_vals)[::-1]
    sorted_modes = [modes[i] for i in order]
    sorted_r2 = r2_vals[order]
    sorted_colors = [MODE_COLORS[m] for m in sorted_modes]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        range(len(sorted_modes)),
        sorted_r2,
        color=sorted_colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_yticks(range(len(sorted_modes)))
    ax.set_yticklabels([f"mode {m}" for m in sorted_modes], fontsize=10)
    ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("R²")
    ax.set_title(
        f"R² by Output Mode — {FUNC_LABEL}\n(sorted best → worst)", fontsize=10
    )
    ax.grid(True, axis="x", alpha=0.25)

    for bar, v in zip(bars, sorted_r2):
        ax.text(
            v + 0.01 if v >= 0 else v - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_uncertainty_distribution(results: Dict, modes: List[int]) -> plt.Figure:
    """Violin plots of per-point predictive σ, one violin per output mode."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_data = [results[m]["std_preds"] for m in modes]
    parts = ax.violinplot(
        plot_data,
        positions=range(len(modes)),
        showmedians=True,
        showextrema=True,
    )
    for pc, m in zip(parts["bodies"], modes):
        pc.set_facecolor(MODE_COLORS[m])
        pc.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([f"mode {m}" for m in modes], fontsize=10)
    ax.set_ylabel("Prediction σ")
    ax.set_title(f"Uncertainty Distribution by Output Mode — {FUNC_LABEL}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metrics_table(results: Dict, modes: List[int]) -> plt.Figure:
    """Full numeric summary table for all output modes."""
    columns = [
        "Output mode",
        "RMSE ↓",
        "MAE ↓",
        "R² ↑",
        "Cov@95 %",
        "Mean σ",
        "NLL ↓",
        "Cal. ρ ↑",
    ]
    rows = []
    for m in modes:
        met = results[m]["metrics"]
        rows.append(
            [
                f"mode {m}",
                f"{met['rmse']:.4f}",
                f"{met['mae']:.4f}",
                f"{met['r2']:.3f}",
                f"{met['coverage_95']:.3f}",
                f"{met['mean_std']:.4f}",
                f"{met['nll']:.3f}",
                f"{met['calibration_rho']:.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(rows) + 1.8))
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

    for r_idx, m in enumerate(modes):
        color = tuple(list(MODE_COLORS[m][:3]) + [0.18])
        for c_idx in range(len(columns)):
            tbl[(r_idx + 1, c_idx)].set_facecolor(color)

    ax.set_title(
        f"Metrics Summary — {FUNC_LABEL} across All Output Modes",
        fontsize=11,
        pad=18,
    )
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    base_cfg = {**_COMMON, "target_mode": (OUTPUT_MODES[0],)}

    with Experiment(
        "Output Mode Comparison",
        config=base_cfg,
    ) as exp:
        np.random.seed(_COMMON["seed"])

        # ── Load data ─────────────────────────────────────────────────────
        X_train, y_train, X_test, y_test = get_data(N_DATA, SIGMA_NOISE, FUNCTION)
        logger.info("Data: %s  train=%d  test=%d", FUNCTION, len(X_train), len(X_test))

        # ── Train all 6 output modes ───────────────────────────────────────
        results: Dict[int, Dict] = {}
        all_metrics: Dict[str, Dict] = {}

        for m in OUTPUT_MODES:
            r = train_and_evaluate(m, X_train, y_train, X_test, y_test)
            results[m] = r
            all_metrics[f"mode_{m}"] = r["metrics"]

        exp.save_metrics({"by_output_mode": all_metrics})

        # ── Plots ──────────────────────────────────────────────────────────
        logger.info("Generating figures…")

        figs: List[Tuple[str, plt.Figure]] = [
            (
                "predictions_grid.png",
                plot_predictions_grid(
                    results, X_train, y_train, X_test, y_test, OUTPUT_MODES
                ),
            ),
            ("training_loss.png", plot_training_loss(results, OUTPUT_MODES)),
            ("calibration.png", plot_calibration(results, y_test, OUTPUT_MODES)),
            ("metrics_comparison.png", plot_metrics_comparison(results, OUTPUT_MODES)),
            ("r2_bar.png", plot_r2_bar(results, OUTPUT_MODES)),
            (
                "uncertainty_distribution.png",
                plot_uncertainty_distribution(results, OUTPUT_MODES),
            ),
            ("metrics_table.png", plot_metrics_table(results, OUTPUT_MODES)),
        ]

        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("  saved %s", fname)

        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
