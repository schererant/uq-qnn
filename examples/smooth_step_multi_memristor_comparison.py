#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare smooth-step fitting with single and multiple memristors in a 6-mode mesh."""

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
from src.simulation import run_simulation_sequence_np
from src.training import train_pytorch_generic

logger = get_logger(__name__)

_BASE_CONFIG: Dict = {
    "n_modes": 6,
    "encoding_mode": 0,
    "target_mode": (5,),
    "encoding_phase_idx": None,
    "output_mode": "singles",
    "input_modes": None,
    "working_detectors": None,
    "n_photons": None,
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
}

N_DATA = 80
SIGMA_NOISE = 0.02
UNC_N_PASSES = 15
UNC_NOISE_STD = 0.05
FUNCTION_NAME = "step_function_data"
FUNCTION_LABEL = "Smooth step"

CONFIGS = {
    "standard": None,
    "single_6": (6,),
    "single_8": (8,),
    "double_6_8": (6, 8),
}

COLORS = {
    "standard": "#1f77b4",
    "single_6": "#ff7f0e",
    "single_8": "#2ca02c",
    "double_6_8": "#d62728",
}


def _phase_label(phase_idx: int) -> str:
    return f"phase {phase_idx} @ MZI{get_mzi_modes_for_phase(phase_idx, _BASE_CONFIG['n_modes'])}"


def _config_label(name: str) -> str:
    memristive = CONFIGS[name]
    if memristive is None:
        return "Standard"
    if len(memristive) == 1:
        return f"1 memristor: {_phase_label(memristive[0])}"
    joined = " + ".join(_phase_label(idx) for idx in memristive)
    return f"2 memristors: {joined}"


LABELS = {name: _config_label(name) for name in CONFIGS}


def _sim_cfg(name: str) -> SimConfig:
    memristive = CONFIGS[name]
    cfg = {
        **_BASE_CONFIG,
        "memristive_phase_idx": memristive,
        "memristive_output_modes": None,
        "memory_depth": 1 if memristive is None else _BASE_CONFIG["memory_depth"],
    }
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
    all_preds = np.zeros((len(enc_test), n_passes))

    for i in range(n_passes):
        sample_count = max(100, sim_cfg.n_samples + int(rng.integers(-50, 51)))
        perturbed = theta + rng.normal(0, noise_std, size=len(theta))
        all_preds[:, i] = run_simulation_sequence_np(
            perturbed,
            enc_test,
            sim_cfg.replace(n_samples=sample_count),
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
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    sim_cfg = _sim_cfg(name)
    enc_train = 2 * np.arccos(np.clip(X_train, 0, 1))
    logger.info("▶ Training config=%-12s  label=%s", name, LABELS[name])

    theta, history = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=_BASE_CONFIG["lr"],
        epochs=_BASE_CONFIG["epochs"],
        seed=_BASE_CONFIG["seed"],
    )

    enc_test = 2 * np.arccos(np.clip(X_test, 0, 1))
    unc = _run_uncertainty(
        theta,
        enc_test,
        sim_cfg,
        n_passes=UNC_N_PASSES,
        noise_std=UNC_NOISE_STD,
        seed=_BASE_CONFIG["seed"],
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
        "theta": theta,
        "history": history,
        "mean_preds": unc["mean"],
        "std_preds": unc["std"],
        "metrics": metrics,
    }


def plot_predictions(results: Dict[str, Dict[str, object]], data: Tuple[np.ndarray, ...]) -> plt.Figure:
    X_train, y_train, X_test, y_test = data
    names = list(CONFIGS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

    for ax, name in zip(axes.flat, names):
        r = results[name]
        m = r["metrics"]
        ax.scatter(X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
        ax.plot(X_test, r["mean_preds"], color=COLORS[name], lw=2.0, label="Prediction")
        ax.fill_between(
            X_test,
            r["mean_preds"] - 1.96 * r["std_preds"],
            r["mean_preds"] + 1.96 * r["std_preds"],
            color=COLORS[name],
            alpha=0.22,
            label="95 % CI",
        )
        ax.set_title(f"{LABELS[name]}\nRMSE={m['rmse']:.4f}   R2={m['r2']:.3f}", fontsize=10)
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.suptitle(f"{FUNCTION_LABEL} fitting: single vs multiple memristors", fontsize=14)
    fig.tight_layout()
    return fig


def plot_training_loss(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in CONFIGS:
        ax.plot(results[name]["history"], color=COLORS[name], lw=1.7, label=LABELS[name])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training loss: single vs multiple memristors")
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
    names = list(CONFIGS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(18, 5))

    for ax, (key, title) in zip(axes, metrics_spec):
        vals = [results[name]["metrics"][key] for name in names]
        ax.bar(x, vals, color=[COLORS[name] for name in names], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[name] for name in names], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{FUNCTION_LABEL} metrics: single vs multiple memristors", fontsize=13)
    fig.tight_layout()
    return fig


def plot_metrics_table(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    columns = ["Config", "RMSE", "MAE", "R2", "Cov@95 %", "Mean sigma", "NLL", "Cal. rho"]
    rows = []
    for name in CONFIGS:
        m = results[name]["metrics"]
        rows.append([
            LABELS[name],
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['r2']:.3f}",
            f"{m['coverage_95']:.3f}",
            f"{m['mean_std']:.4f}",
            f"{m['nll']:.3f}",
            f"{m['calibration_rho']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 0.58 * len(rows) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)

    for r_idx, name in enumerate(CONFIGS):
        color = "#daeeff" if name == "standard" else "#ffe5cc"
        for c_idx in range(len(columns)):
            tbl[(r_idx + 1, c_idx)].set_facecolor(color)

    ax.set_title("Smooth-step summary: multiple memristor configurations", fontsize=11, pad=16)
    fig.tight_layout()
    return fig


def main() -> None:
    exp_cfg = {**_BASE_CONFIG, "memristive_phase_idx": None, "memristive_output_modes": None}
    with Experiment("Smooth Step Multi Memristor Comparison", config=exp_cfg) as exp:
        np.random.seed(_BASE_CONFIG["seed"])
        data = get_data(N_DATA, SIGMA_NOISE, FUNCTION_NAME)
        X_train, y_train, X_test, y_test = data

        results: Dict[str, Dict[str, object]] = {}
        metrics_out: Dict[str, Dict[str, float]] = {}
        for name in CONFIGS:
            result = train_and_evaluate(name, X_train, y_train, X_test, y_test)
            results[name] = result
            metrics_out[name] = result["metrics"]

        exp.save_metrics({
            "function": FUNCTION_NAME,
            "labels": LABELS,
            "by_config": metrics_out,
        })

        figs = [
            ("predictions_grid.png", plot_predictions(results, data)),
            ("training_loss.png", plot_training_loss(results)),
            ("metrics_comparison.png", plot_metrics(results)),
            ("metrics_table.png", plot_metrics_table(results)),
        ]
        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("saved %s", fname)

        best_name = min(CONFIGS, key=lambda key: results[key]["metrics"]["rmse"])
        logger.info("Best config by RMSE: %s (%s)", best_name, LABELS[best_name])
        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
