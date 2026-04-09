#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matched singles-vs-coincidence benchmark on one fixed regression task.

This experiment keeps the circuit size, dataset, split, seed, optimizer, and
training schedule fixed across both measurement families:

  - singles: scalar readout from one output mode
  - coincidence: scalar readout from one explicit CC pair after postselection

The purpose is to compare readout quality under matched conditions instead of
mixing different circuit sizes or older runs with incompatible settings.
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

from src.config import SimConfig
from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger
from src.loss import PhotonicModel
from src.training import train_pytorch_generic

logger = get_logger(__name__)

N_MODES = 6
SINGLES_MODES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
COINCIDENCE_PAIRS: Tuple[Tuple[int, int], ...] = ((2, 4), (3, 4), (3, 5))

CONFIG: Dict[str, object] = {
    "n_modes": N_MODES,
    "encoding_phase_idx": 5,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
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
    "n_data": 80,
    "sigma_noise": 0.02,
    "unc_n_passes": 15,
    "unc_noise_std": 0.05,
    "data_function": "sinusoid_data",
    "func_label": r"$\sin(0.7\pi x)$",
    "output_mode": "singles",
    "input_state": tuple(1 if i == 0 else 0 for i in range(N_MODES)),
    "photon_distinguishability": None,
    "target_mode": (SINGLES_MODES[0],),
    "working_detectors": None,
}

FUNC_LABEL = CONFIG["func_label"]


def _case_specs() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for mode in SINGLES_MODES:
        cases.append(
            {
                "label": f"singles_m{mode}",
                "family": "singles",
                "input_state": tuple(1 if i == mode else 0 for i in range(N_MODES)),
                "photon_distinguishability": None,
                "target_mode": (mode,),
                "working_detectors": None,
            }
        )
    for pair in COINCIDENCE_PAIRS:
        cases.append(
            {
                "label": f"coinc_cc{pair[0]}{pair[1]}",
                "family": "coincidence",
                "input_state": tuple(1 if i in (1, 4) else 0 for i in range(N_MODES)),
                "photon_distinguishability": "indistinguishable",
                "target_mode": pair,
                "working_detectors": tuple(range(N_MODES)),
            }
        )
    return cases


CASES = _case_specs()
CASE_COLORS = {
    case["label"]: color
    for case, color in zip(
        CASES,
        matplotlib.colormaps["tab20"].resampled(len(CASES)).colors,
    )
}


def _display_label(case: Dict[str, object]) -> str:
    family = str(case["family"])
    target_mode = tuple(int(v) for v in tuple(case["target_mode"]))  # type: ignore[arg-type]
    if family == "singles":
        return f"singles m{target_mode[0]}"
    return f"coinc CC{target_mode[0]}{target_mode[1]}"


def _sim_cfg(case: Dict[str, object]) -> SimConfig:
    cfg = {
        **CONFIG,
        "output_mode": case["family"],
        "input_state": case["input_state"],
        "photon_distinguishability": case["photon_distinguishability"],
        "target_mode": case["target_mode"],
        "working_detectors": case["working_detectors"],
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
    n = len(enc_test)
    all_preds = np.zeros((n, n_passes), dtype=float)
    theta_base = model.theta.detach().cpu().numpy()
    for i in range(n_passes):
        sample_count = max(100, model.sim_cfg.n_samples + int(rng.integers(-50, 51)))
        perturbed = theta_base + rng.normal(0.0, noise_std, size=len(theta_base))
        all_preds[:, i] = model.predict(
            enc_test, theta_np=perturbed, n_samples=sample_count
        )
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
    case: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    sim_cfg = _sim_cfg(case)
    logger.info("Training %-14s  readout=%s", case["family"], _display_label(case))

    enc_train = 2 * np.arccos(np.clip(X_train, 0.0, 1.0))
    theta, history, model = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=float(CONFIG["lr"]),
        epochs=int(CONFIG["epochs"]),
        seed=int(CONFIG["seed"]),
    )

    enc_test = 2 * np.arccos(np.clip(X_test, 0.0, 1.0))
    unc = _run_uncertainty(
        model,
        enc_test,
        n_passes=int(CONFIG["unc_n_passes"]),
        noise_std=float(CONFIG["unc_noise_std"]),
        seed=int(CONFIG["seed"]),
    )
    metrics = compute_metrics(y_test, unc["mean"], unc["std"])
    logger.info(
        "  final loss=%.6f  RMSE=%.4f  R2=%.4f",
        float(history[-1]),
        metrics["rmse"],
        metrics["r2"],
    )
    return {
        "theta": theta,
        "history": history,
        "mean_preds": unc["mean"],
        "std_preds": unc["std"],
        "metrics": metrics,
        "family": str(case["family"]),
    }


def _best_case_label(results: Dict[str, Dict[str, object]], family: str) -> str:
    candidates = [
        label for label, result in results.items() if str(result["family"]) == family
    ]
    return max(candidates, key=lambda label: float(results[label]["metrics"]["r2"]))  # type: ignore[index]


def plot_predictions_grid(
    results: Dict[str, Dict[str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    n_cols = 3
    n_rows = (len(CASES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, case in enumerate(CASES):
        label = str(case["label"])
        result = results[label]
        metrics = result["metrics"]
        ax = axes_flat[idx]

        ax.scatter(X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
        ax.plot(
            X_test,
            result["mean_preds"],
            color=CASE_COLORS[label],
            lw=2.0,
            label="Prediction",
        )
        ax.fill_between(
            X_test,
            result["mean_preds"] - 1.96 * result["std_preds"],
            result["mean_preds"] + 1.96 * result["std_preds"],
            color=CASE_COLORS[label],
            alpha=0.22,
            label="95 % CI",
        )
        ax.set_title(
            f"{_display_label(case)}\nRMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.3f}",
            fontsize=9,
        )
        ax.set(xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8)

    for ax in axes_flat[len(CASES) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Matched Readout Benchmark — {FUNC_LABEL}\n"
        f"n_modes={N_MODES}, same split/seed/optimizer across all cases",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_training_loss(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    for case in CASES:
        label = str(case["label"])
        ax.plot(
            results[label]["history"],
            color=CASE_COLORS[label],
            lw=1.4,
            alpha=0.8,
            label=_display_label(case),
        )
    ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="MSE Loss", title="Training Convergence")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    metrics_spec = [
        ("rmse", "RMSE  (lower better)"),
        ("r2", "R2  (higher better)"),
        ("coverage_95", "Coverage @ 95 %"),
        ("nll", "NLL  (lower better)"),
    ]
    labels = [_display_label(case) for case in CASES]
    case_labels = [str(case["label"]) for case in CASES]
    x = np.arange(len(CASES))

    fig, axes = plt.subplots(1, len(metrics_spec), figsize=(5 * len(metrics_spec), 5))
    for ax, (key, title) in zip(axes, metrics_spec):
        vals = [results[label]["metrics"][key] for label in case_labels]
        bars = ax.bar(
            x,
            vals,
            color=[CASE_COLORS[label] for label in case_labels],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=40,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Matched Readout Metrics", fontsize=12)
    fig.tight_layout()
    return fig


def plot_best_family_summary(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    best_singles = _best_case_label(results, "singles")
    best_coinc = _best_case_label(results, "coincidence")
    labels = [best_singles, best_coinc]
    metric_names = ["rmse", "r2", "coverage_95", "nll"]
    metric_titles = ["RMSE", "R2", "Coverage@95", "NLL"]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, label in enumerate(labels):
        vals = [results[label]["metrics"][name] for name in metric_names]
        ax.bar(
            x + (idx - 0.5) * width,
            vals,
            width,
            label=label,
            color=CASE_COLORS[label],
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_titles)
    ax.set_title("Best Readout per Measurement Family")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metrics_table(results: Dict[str, Dict[str, object]]) -> plt.Figure:
    columns = [
        "Case",
        "Family",
        "RMSE",
        "MAE",
        "R2",
        "Cov@95 %",
        "Mean sigma",
        "NLL",
        "Cal. rho",
    ]
    rows = []
    for case in CASES:
        label = str(case["label"])
        metrics = results[label]["metrics"]
        rows.append(
            [
                _display_label(case),
                str(case["family"]),
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['r2']:.3f}",
                f"{metrics['coverage_95']:.3f}",
                f"{metrics['mean_std']:.4f}",
                f"{metrics['nll']:.3f}",
                f"{metrics['calibration_rho']:.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(15, 0.55 * len(rows) + 1.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)
    for row_idx, case in enumerate(CASES):
        face = "#daeeff" if case["family"] == "singles" else "#ffe5cc"
        for col_idx in range(len(columns)):
            tbl[(row_idx + 1, col_idx)].set_facecolor(face)
    ax.set_title("Matched Benchmark Metrics Summary", fontsize=11, pad=18)
    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment("Matched Readout Benchmark", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))

        X_train, y_train, X_test, y_test = get_data(
            int(CONFIG["n_data"]),
            float(CONFIG["sigma_noise"]),
            str(CONFIG["data_function"]),
        )
        logger.info(
            "Data: %s  train=%d  test=%d  n_modes=%d",
            CONFIG["data_function"],
            len(X_train),
            len(X_test),
            N_MODES,
        )

        results: Dict[str, Dict[str, object]] = {}
        for case in CASES:
            results[str(case["label"])] = train_and_evaluate(
                case, X_train, y_train, X_test, y_test
            )

        best_singles = _best_case_label(results, "singles")
        best_coinc = _best_case_label(results, "coincidence")
        exp.save_metrics(
            {
                "function": CONFIG["data_function"],
                "n_modes": N_MODES,
                "cases": {
                    str(case["label"]): results[str(case["label"])]["metrics"]
                    for case in CASES
                },
                "best_by_family": {
                    "singles": {
                        "label": best_singles,
                        "metrics": results[best_singles]["metrics"],
                    },
                    "coincidence": {
                        "label": best_coinc,
                        "metrics": results[best_coinc]["metrics"],
                    },
                },
            }
        )

        figs: List[Tuple[str, plt.Figure]] = [
            (
                "predictions_grid.png",
                plot_predictions_grid(results, X_train, y_train, X_test, y_test),
            ),
            ("training_loss.png", plot_training_loss(results)),
            ("metrics_comparison.png", plot_metrics_comparison(results)),
            ("best_family_summary.png", plot_best_family_summary(results)),
            ("metrics_table.png", plot_metrics_table(results)),
        ]
        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("  saved %s", fname)

        logger.info(
            "Best singles=%s  Best coincidence=%s",
            best_singles,
            best_coinc,
        )


if __name__ == "__main__":
    main()
