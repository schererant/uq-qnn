#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare smooth-step fitting quality for different single-memristor placements."""

import os
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.circuits import get_mzi_modes_for_phase
from src.config import SimConfig
from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger
from src.loss import PhotonicModel
from src.training import train_pytorch_generic

logger = get_logger(__name__)

CONFIG: Dict = {
    "n_modes": 6,
    "input_state": tuple(1 if i == 0 else 0 for i in range(6)),
    "encoding_phase_idx": 0,
    "photon_distinguishability": None,
    "target_mode": (5,),
    "output_mode": "singles",
    "working_detectors": None,
    "n_samples": 300,
    "noise_std": None,
    "loss_type": "mse",
    "n_classes": 1,
    "sim_backend": "numpy",
    "seed": 42,
    "lr": 0.04,
    "epochs": 180,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "n_data": 80,
    "sigma_noise": 0.02,
    "unc_n_passes": 15,
    "unc_noise_std": 0.05,
    "data_function": "step_function_data",
    "func_label": "Smooth step",
}

FUNCTION_LABEL = CONFIG["func_label"]

PLACEMENTS = {
    "standard": None,
    "phase_0": 0,
    "phase_2": 2,
    "phase_4": 4,
    "phase_6": 6,
    "phase_8": 8,
}

PLACEMENT_COLORS = {
    "standard": "#1f77b4",
    "phase_0": "#ff7f0e",
    "phase_2": "#2ca02c",
    "phase_4": "#d62728",
    "phase_6": "#9467bd",
    "phase_8": "#8c564b",
}


def _placement_label(name: str) -> str:
    phase_idx = PLACEMENTS[name]
    if phase_idx is None:
        return "Standard"
    modes = get_mzi_modes_for_phase(phase_idx, CONFIG["n_modes"])
    return f"Phase {phase_idx} · MZI{modes}"


PLACEMENT_LABELS = {name: _placement_label(name) for name in PLACEMENTS}


def _sim_cfg(placement: str) -> SimConfig:
    phase_idx = PLACEMENTS[placement]
    cfg = {
        **CONFIG,
        "memristive_phase_idx": None if phase_idx is None else (phase_idx,),
        "memristive_output_modes": None,
        "memory_depth": 1 if phase_idx is None else int(CONFIG["memory_depth"]),
    }
    return SimConfig.from_experiment_config(cfg)


def _run_uncertainty(
    model: PhotonicModel,
    enc_test: np.ndarray,
    *,
    n_passes: int,
    noise_std: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_preds = np.zeros((len(enc_test), n_passes))
    theta_base = model.theta.detach().cpu().numpy()

    for i in range(n_passes):
        sample_count = max(100, model.sim_cfg.n_samples + int(rng.integers(-50, 51)))
        perturbed = theta_base + rng.normal(0, noise_std, size=len(theta_base))
        all_preds[:, i] = model.predict(
            enc_test, theta_np=perturbed, n_samples=sample_count
        )

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

    sigma = np.clip(std_preds, 1e-6, None)
    nll = float(np.mean(0.5 * (residuals / sigma) ** 2 + np.log(sigma)))
    cal_rho, _ = stats.spearmanr(std_preds, np.abs(residuals))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coverage_95": coverage_95,
        "mean_std": float(np.mean(std_preds)),
        "nll": nll,
        "calibration_rho": float(cal_rho),
    }


def train_and_evaluate(
    placement: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    sc = _sim_cfg(placement)
    logger.info(
        "▶ Training placement=%-8s  label=%s", placement, PLACEMENT_LABELS[placement]
    )

    enc_train = 2 * np.arccos(np.clip(X_train, 0, 1))
    trained_state, history, model = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sc,
        lr=float(CONFIG["lr"]),
        epochs=int(CONFIG["epochs"]),
        seed=int(CONFIG["seed"]),
    )

    enc_test = 2 * np.arccos(np.clip(X_test, 0, 1))
    unc = _run_uncertainty(
        model,
        enc_test,
        n_passes=int(CONFIG["unc_n_passes"]),
        noise_std=float(CONFIG["unc_noise_std"]),
        seed=int(CONFIG["seed"]),
    )
    metrics = compute_metrics(y_test, unc["mean"], unc["std"])
    logger.info(
        "  RMSE=%.4f  R2=%.4f  Coverage=%.2f  NLL=%.3f",
        metrics["rmse"],
        metrics["r2"],
        metrics["coverage_95"],
        metrics["nll"],
    )
    return {
        "trained_state": trained_state,
        "history": history,
        "mean_preds": unc["mean"],
        "std_preds": unc["std"],
        "metrics": metrics,
    }


def plot_predictions_grid(
    results: Dict[str, Dict[str, object]], data: Tuple[np.ndarray, ...]
) -> plt.Figure:
    X_train, y_train, X_test, y_test = data
    names = list(PLACEMENTS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

    for ax, name in zip(axes.flat, names):
        r = results[name]
        m = r["metrics"]
        ax.scatter(X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
        ax.plot(
            X_test,
            r["mean_preds"],
            color=PLACEMENT_COLORS[name],
            lw=2.0,
            label="Prediction",
        )
        ax.fill_between(
            X_test,
            r["mean_preds"] - 1.96 * r["std_preds"],
            r["mean_preds"] + 1.96 * r["std_preds"],
            color=PLACEMENT_COLORS[name],
            alpha=0.22,
            label="95 % CI",
        )
        ax.set_title(
            f"{PLACEMENT_LABELS[name]}\nRMSE={m['rmse']:.4f}   R2={m['r2']:.3f}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")

    fig.suptitle(f"{FUNCTION_LABEL} fitting across memristor placements", fontsize=14)
    fig.tight_layout()
    return fig


def plot_training_loss(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in PLACEMENTS:
        ax.plot(
            results[name]["history"],
            label=PLACEMENT_LABELS[name],
            color=PLACEMENT_COLORS[name],
            lw=1.7,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training loss by memristor placement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_metrics(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    metrics_spec = [
        ("rmse", "RMSE"),
        ("r2", "R2"),
        ("coverage_95", "Coverage @ 95 %"),
        ("nll", "NLL"),
    ]
    names = list(PLACEMENTS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(18, 5))

    for ax, (key, title) in zip(axes, metrics_spec):
        vals = [results[name]["metrics"][key] for name in names]
        ax.bar(x, vals, color=[PLACEMENT_COLORS[name] for name in names], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [PLACEMENT_LABELS[name] for name in names], rotation=30, ha="right"
        )
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{FUNCTION_LABEL} metrics by memristor placement", fontsize=13)
    fig.tight_layout()
    return fig


def plot_metrics_table(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    columns = [
        "Placement",
        "RMSE",
        "MAE",
        "R2",
        "Cov@95 %",
        "Mean sigma",
        "NLL",
        "Cal. rho",
    ]
    rows = []
    for name in PLACEMENTS:
        m = results[name]["metrics"]
        rows.append(
            [
                PLACEMENT_LABELS[name],
                f"{m['rmse']:.4f}",
                f"{m['mae']:.4f}",
                f"{m['r2']:.3f}",
                f"{m['coverage_95']:.3f}",
                f"{m['mean_std']:.4f}",
                f"{m['nll']:.3f}",
                f"{m['calibration_rho']:.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)

    for r_idx, name in enumerate(PLACEMENTS):
        color = "#daeeff" if name == "standard" else "#ffe5cc"
        for c_idx in range(len(columns)):
            tbl[(r_idx + 1, c_idx)].set_facecolor(color)

    ax.set_title(
        "Smooth-step fitting summary by memristor placement", fontsize=11, pad=16
    )
    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment("Smooth Step Memristor Placement Comparison", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))
        data = get_data(
            int(CONFIG["n_data"]),
            float(CONFIG["sigma_noise"]),
            str(CONFIG["data_function"]),
        )
        X_train, y_train, X_test, y_test = data

        results: Dict[str, Dict[str, object]] = {}
        metrics_out: Dict[str, Dict[str, float]] = {}
        for placement in PLACEMENTS:
            result = train_and_evaluate(placement, X_train, y_train, X_test, y_test)
            results[placement] = result
            metrics_out[placement] = result["metrics"]

        exp.save_metrics(
            {
                "function": CONFIG["data_function"],
                "placement_labels": PLACEMENT_LABELS,
                "by_placement": metrics_out,
            }
        )

        figs = [
            ("predictions_grid.png", plot_predictions_grid(results, data)),
            ("training_loss.png", plot_training_loss(results)),
            ("metrics_comparison.png", plot_metrics(results)),
            ("metrics_table.png", plot_metrics_table(results)),
        ]
        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("saved %s", fname)

        best_name = min(PLACEMENTS, key=lambda name: results[name]["metrics"]["rmse"])
        logger.info(
            "Best placement by RMSE: %s (%s)", best_name, PLACEMENT_LABELS[best_name]
        )
        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
