#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function comparison with chip-output memristive feedback (2-photon, 6-mode).

This mirrors the "all functions" sweep style of examples/function_comparison.py,
but uses the 2-photon coincidence configuration:

- 6 modes
- input photons in modes 1 and 4
- encoding phase index 7
- coincidence readout pair (4, 5)
- memristive feedback taken from chip-output pair (4, 5)
- memristor phase index 9 (the 9/10 MZI pair anchor)
- NumPy backend (analytical 2-photon SLOS fast path)

It compares two circuit variants across all functions:
  - standard  : no memristor
  - memristor : one memristive phase + chip-output feedback memory
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
from src.loss import PhotonicModel
from src.training import train_pytorch_generic

logger = get_logger(__name__)

INPUT_STATE = tuple(1 if i in (1, 4) else 0 for i in range(6))
COINCIDENCE_PAIR = (4, 5)

CONFIG: Dict = {
    "n_modes": 6,
    "input_state": INPUT_STATE,
    "encoding_phase_idx": 7,
    "photon_distinguishability": "indistinguishable",
    "target_mode": COINCIDENCE_PAIR,
    "computation_modes": COINCIDENCE_PAIR,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "coincidence",
    "working_detectors": tuple(range(6)),
    "feedback_mode": "chip_output",
    "feedback_modes": ((4, 5),),
    "n_samples": 0,
    "noise_std": None,
    "loss_type": "mse",
    "n_classes": 1,
    "sim_backend": "numpy",
    "seed": 42,
    "lr": 0.05,
    "epochs": 20,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "n_data": 500,
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
    "memristive_phase_idx": (9,),
    "memristive_output_modes": ((4, 5),),
    "memory_depth": 2,
}

FUNCTIONS = CONFIG["data_functions"]
FUNC_LABELS = CONFIG["func_labels"]
ARCHITECTURES = CONFIG["architectures"]
ARCH_LABELS = CONFIG["arch_labels"]
ARCH_COLORS = CONFIG["arch_colors"]


def _sim_cfg(architecture: str) -> SimConfig:
    d = dict(CONFIG)
    if architecture == "memristor":
        d.update(_MEMRISTOR_OVERRIDES)
    return SimConfig.from_experiment_config(d)


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
    all_preds = np.zeros((n, n_passes))
    theta_base = model.theta.detach().cpu().numpy()

    for i in range(n_passes):
        sample_count = max(0, model.sim_cfg.n_samples + int(rng.integers(-50, 51)))
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
        "  RMSE=%.4f  R²=%.4f  Coverage=%.2f  NLL=%.3f",
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
        "Functions × Architecture (2-photon chip-output coincidence)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment(
        "Function and Memristor Chip-Output Comparison",
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
        fig = plot_predictions_grid(results, data, FUNCTIONS, ARCHITECTURES)
        exp.savefig(fig, "predictions_grid.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("Experiment complete. Report: %s", exp.run_dir)


if __name__ == "__main__":
    main()
