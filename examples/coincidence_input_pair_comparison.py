#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coincidence Input-Pair Comparison Experiment.

Trains a photonic QNN on one synthetic regression function using
coincidence (two-photon) measurements, sweeping all 15 possible
input_modes pairs for a 6-mode Clements mesh.

Each run injects two photons at a different pair of input ports and reads
a fixed postselected coincidence channel (default: CC01) with all 6 detectors
working. Changing input_modes changes which columns of the unitary
matrix contribute to the two-photon amplitudes — a fundamentally
different physical experiment each time.

Note on the coincidence regression output:  after postselection across
all working CC channels, the scalar output is always the probability of
one explicit readout coincidence channel. The
normalization denominator includes all 15 channels, so the output
reflects the relative weight of that readout channel against the full coincidence
distribution — and this distribution depends on input_modes.

Produced artefacts
------------------
predictions_grid.png         Fit + 95 % CI for every input pair (3-column grid)
training_loss.png            Log-scale convergence, one curve per input pair
calibration.png              σ vs |error| scatter, all input pairs overlaid
metrics_comparison.png       Bar charts: RMSE, R², coverage@95 %, NLL
r2_bar.png                   R² sorted best → worst
uncertainty_distribution.png Violin plots of per-point σ per input pair
metrics_table.png            Full numeric summary
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coincidence import (
    get_cc_mode_pairs,
    mode_pair_to_cc_index,
    working_detectors_to_cc_indices,
)
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

N_MODES = 6

# All 15 possible two-photon injection pairs
INPUT_PAIRS: List[Tuple[int, int]] = get_cc_mode_pairs(N_MODES)

# All detectors working → 15 CC channels for normalization
WORKING_DETECTORS = tuple(range(N_MODES))
READOUT_CC_PAIR = (0, 1)

_COMMON: Dict = {
    "n_modes": N_MODES,
    "encoding_mode": 0,
    "target_mode": READOUT_CC_PAIR,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "encoding_phase_idx": None,
    "output_mode": "coincidence",
    "working_detectors": WORKING_DETECTORS,
    "n_photons": tuple([2] * (N_MODES * (N_MODES - 1))),
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

FUNCTION = "sinusoid_data"
FUNC_LABEL = r"$\sin(0.7\pi x)$"

N_DATA = 80
SIGMA_NOISE = 0.02
UNC_N_PASSES = 15
UNC_NOISE_STD = 0.05

_CMAP = matplotlib.colormaps["tab20"].resampled(len(INPUT_PAIRS))
PAIR_COLORS: Dict[Tuple[int, int], object] = {
    p: _CMAP(i) for i, p in enumerate(INPUT_PAIRS)
}


def _pair_label(j: int, k: int) -> str:
    return f"in({j},{k})"


def _readout_label() -> str:
    return f"CC{READOUT_CC_PAIR[0]}{READOUT_CC_PAIR[1]}"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _sim_cfg(input_pair: Tuple[int, int]) -> SimConfig:
    cfg = {**_COMMON, "input_modes": input_pair}
    sim_cfg = SimConfig.from_experiment_config(cfg)

    working_cc_indices = working_detectors_to_cc_indices(
        sim_cfg.working_detectors or tuple(range(sim_cfg.n_modes)),
        sim_cfg.n_modes,
    )
    if len(working_cc_indices) <= 1:
        raise ValueError(
            "Coincidence input-pair comparison requires multiple working CC "
            "channels for non-degenerate postselection. "
            f"Got working_detectors={sim_cfg.working_detectors}."
        )

    readout_cc_idx = mode_pair_to_cc_index(
        READOUT_CC_PAIR[0], READOUT_CC_PAIR[1], sim_cfg.n_modes
    )
    if readout_cc_idx not in working_cc_indices:
        raise ValueError(
            f"Readout pair {_readout_label()} is not available under "
            f"working_detectors={sim_cfg.working_detectors}."
        )

    return sim_cfg


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
    input_pair: Tuple[int, int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    j, k = input_pair
    logger.info("▶ Training  input_modes=(%d, %d)", j, k)
    sc = _sim_cfg(input_pair)
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
    pairs: List[Tuple[int, int]],
) -> plt.Figure:
    n_cols = 3
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, pair in enumerate(pairs):
        j, k = pair
        ax = axes_flat[idx]
        r = results[pair]
        met = r["metrics"]
        color = PAIR_COLORS[pair]

        ax.scatter(
            X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train"
        )
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
            f"{_pair_label(j, k)}\n"
            f"RMSE={met['rmse']:.4f}  R²={met['r2']:.3f}  Cov={met['coverage_95']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8)

    for ax in axes_flat[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Photonic QNN Regression — {FUNC_LABEL}\n"
        f"All input-mode pairs  (reading {_readout_label()} after postselection)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_training_loss(
    results: Dict, pairs: List[Tuple[int, int]]
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    for pair in pairs:
        j, k = pair
        ax.plot(
            results[pair]["history"],
            color=PAIR_COLORS[pair],
            lw=1.4,
            alpha=0.8,
            label=_pair_label(j, k),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(
        f"Training Convergence — {FUNC_LABEL}\nAll input-mode pairs",
        fontsize=11,
    )
    ax.legend(fontsize=7, ncol=5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_calibration(
    results: Dict, y_test: np.ndarray, pairs: List[Tuple[int, int]]
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    all_stds: list = []
    all_errs: list = []
    for pair in pairs:
        j, k = pair
        r = results[pair]
        stds = r["std_preds"]
        errs = np.abs(r["mean_preds"] - y_test)
        ax.scatter(
            stds,
            errs,
            alpha=0.45,
            s=12,
            color=PAIR_COLORS[pair],
            label=_pair_label(j, k),
        )
        all_stds.extend(stds.tolist())
        all_errs.extend(errs.tolist())

    max_val = max(float(np.max(all_stds)), float(np.max(all_errs))) * 1.05
    ax.plot(
        [0, max_val],
        [0, max_val],
        "k-",
        lw=1.0,
        alpha=0.5,
        label="σ = |err| (ideal)",
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
        f"Uncertainty Calibration — {FUNC_LABEL}\n"
        f"Spearman ρ (all pairs) = {rho:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Predictive uncertainty σ")
    ax.set_ylabel("Absolute error |y − ŷ|")
    ax.legend(fontsize=7, markerscale=2, ncol=3)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(
    results: Dict, pairs: List[Tuple[int, int]]
) -> plt.Figure:
    metrics_spec = [
        ("rmse", "RMSE  (↓ better)"),
        ("r2", "R²  (↑ better)"),
        ("coverage_95", "Coverage @ 95 %  (→ 0.95)"),
        ("nll", "NLL  (↓ better)"),
    ]
    labels = [_pair_label(j, k) for j, k in pairs]
    colors = [PAIR_COLORS[p] for p in pairs]
    x = np.arange(len(pairs))

    fig, axes = plt.subplots(
        1, len(metrics_spec), figsize=(5 * len(metrics_spec), 5)
    )
    for ax, (key, title) in zip(axes, metrics_spec):
        vals = [results[p]["metrics"][key] for p in pairs]
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
                fontsize=7,
                rotation=40,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Performance Metrics by Input Pair — {FUNC_LABEL}\n"
        f"Readout channel: {_readout_label()}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_r2_bar(results: Dict, pairs: List[Tuple[int, int]]) -> plt.Figure:
    r2_vals = np.array([results[p]["metrics"]["r2"] for p in pairs])
    order = np.argsort(r2_vals)[::-1]
    sorted_pairs = [pairs[i] for i in order]
    sorted_r2 = r2_vals[order]
    sorted_colors = [PAIR_COLORS[p] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.barh(
        range(len(sorted_pairs)),
        sorted_r2,
        color=sorted_colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_yticks(range(len(sorted_pairs)))
    ax.set_yticklabels(
        [_pair_label(j, k) for j, k in sorted_pairs], fontsize=10
    )
    ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("R²")
    ax.set_title(
        f"R² by Input Pair — {FUNC_LABEL}\n(sorted best → worst)",
        fontsize=10,
    )
    ax.grid(True, axis="x", alpha=0.25)
    for bar, v in zip(bars, sorted_r2):
        ax.text(
            v + 0.005 if v >= 0 else v - 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def plot_uncertainty_distribution(
    results: Dict, pairs: List[Tuple[int, int]]
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 5))
    plot_data = [results[p]["std_preds"] for p in pairs]
    parts = ax.violinplot(
        plot_data,
        positions=range(len(pairs)),
        showmedians=True,
        showextrema=True,
    )
    for pc, pair in zip(parts["bodies"], pairs):
        pc.set_facecolor(PAIR_COLORS[pair])
        pc.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(
        [_pair_label(j, k) for j, k in pairs],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Prediction σ")
    ax.set_title(
        f"Uncertainty Distribution by Input Pair — {FUNC_LABEL}", fontsize=10
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metrics_table(
    results: Dict, pairs: List[Tuple[int, int]]
) -> plt.Figure:
    columns = [
        "Input pair",
        "RMSE ↓",
        "MAE ↓",
        "R² ↑",
        "Cov@95 %",
        "Mean σ",
        "NLL ↓",
        "Cal. ρ ↑",
    ]
    rows = []
    for pair in pairs:
        j, k = pair
        met = results[pair]["metrics"]
        rows.append([
            _pair_label(j, k),
            f"{met['rmse']:.4f}",
            f"{met['mae']:.4f}",
            f"{met['r2']:.3f}",
            f"{met['coverage_95']:.3f}",
            f"{met['mean_std']:.4f}",
            f"{met['nll']:.3f}",
            f"{met['calibration_rho']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=columns, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)
    for r_idx, pair in enumerate(pairs):
        color = tuple(list(PAIR_COLORS[pair][:3]) + [0.18])
        for c_idx in range(len(columns)):
            tbl[(r_idx + 1, c_idx)].set_facecolor(color)
    ax.set_title(
        f"Metrics Summary — {FUNC_LABEL}  "
        f"(sweeping input_modes, readout {_readout_label()})",
        fontsize=11,
        pad=18,
    )
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    base_cfg = {**_COMMON, "input_modes": INPUT_PAIRS[0]}

    with Experiment("Coincidence Input Pair Comparison", config=base_cfg) as exp:
        np.random.seed(_COMMON["seed"])

        X_train, y_train, X_test, y_test = get_data(N_DATA, SIGMA_NOISE, FUNCTION)
        logger.info(
            "Data: %s  train=%d  test=%d", FUNCTION, len(X_train), len(X_test)
        )
        logger.info(
            "Sweeping %d input_modes pairs, all %d detectors working, readout=%s",
            len(INPUT_PAIRS),
            N_MODES,
            _readout_label(),
        )

        results: Dict[Tuple[int, int], Dict] = {}
        all_metrics: Dict[str, Dict] = {}

        for pair in INPUT_PAIRS:
            j, k = pair
            r = train_and_evaluate(pair, X_train, y_train, X_test, y_test)
            results[pair] = r
            all_metrics[f"in_{j}_{k}"] = r["metrics"]

        exp.save_metrics(
            {"readout_cc_pair": READOUT_CC_PAIR, "by_input_pair": all_metrics}
        )

        logger.info("Generating figures…")

        figs: List[Tuple[str, plt.Figure]] = [
            (
                "predictions_grid.png",
                plot_predictions_grid(
                    results, X_train, y_train, X_test, y_test, INPUT_PAIRS
                ),
            ),
            ("training_loss.png", plot_training_loss(results, INPUT_PAIRS)),
            (
                "calibration.png",
                plot_calibration(results, y_test, INPUT_PAIRS),
            ),
            (
                "metrics_comparison.png",
                plot_metrics_comparison(results, INPUT_PAIRS),
            ),
            ("r2_bar.png", plot_r2_bar(results, INPUT_PAIRS)),
            (
                "uncertainty_distribution.png",
                plot_uncertainty_distribution(results, INPUT_PAIRS),
            ),
            ("metrics_table.png", plot_metrics_table(results, INPUT_PAIRS)),
        ]

        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("  saved %s", fname)

        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
