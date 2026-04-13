#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted coincidence-regression sweep for harder 1D functions.

This script packages a few higher-capacity configurations that are worth trying
 when smooth global functions fit well but localized or sharper targets do not.

The preset list starts with multi-modal runs, since that was the clearest
failure mode in ``examples/function_comparison.py``. Each preset runs as its
own ``Experiment`` and stores:

- ``summary.png``           prediction fit + uncertainty + loss curve
- ``trained_state.json``    serialized photonic parameters + linear head
- ``run_summary.json``      config and metrics

Edit ``PRESET_NAMES`` below to run a subset.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger

logger = get_logger(__name__)


def _occupation(n_modes: int, occupied_modes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 if i in occupied_modes else 0 for i in range(n_modes))


def _base_config(
    *,
    n_modes: int,
    occupied_modes: tuple[int, ...],
    encoding_phase_idx: int,
    target_mode: tuple[int, int],
    data_function: str,
    lr: float,
    epochs: int,
    n_data: int,
    sigma_noise: float,
    unc_n_passes: int,
    unc_noise_std: float,
    memristive_phase_idx: tuple[int, ...] | None = None,
    feedback_modes: tuple[tuple[int, int], ...] | None = None,
    memory_depth: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    is_memristive = memristive_phase_idx is not None and len(memristive_phase_idx) > 0
    return {
        "n_modes": n_modes,
        "input_state": _occupation(n_modes, occupied_modes),
        "encoding_phase_idx": encoding_phase_idx,
        "photon_distinguishability": "indistinguishable",
        "target_mode": target_mode,
        "computation_modes": target_mode if is_memristive else None,
        "memristive_phase_idx": memristive_phase_idx,
        "memristive_output_modes": feedback_modes,
        "output_mode": "coincidence",
        "working_detectors": tuple(range(n_modes)),
        "feedback_mode": "chip_output" if is_memristive else "internal_arm",
        "feedback_modes": feedback_modes,
        "loss_type": "mse",
        "n_classes": 1,
        "lr": lr,
        "epochs": epochs,
        "n_samples": 0,
        "memory_depth": memory_depth,
        "n_swipe": 0,
        "swipe_span": 0.0,
        "noise_std": None,
        "seed": seed,
        "sim_backend": "numpy",
        "n_data": n_data,
        "sigma_noise": sigma_noise,
        "data_function": data_function,
        "unc_n_passes": unc_n_passes,
        "unc_noise_std": unc_noise_std,
    }


PRESETS: dict[str, dict[str, Any]] = {
    "multimodal_static_8m_pair07": {
        "title": "Multimodal Static 8-Mode (0,7)",
        "notes": (
            "Higher-capacity static coincidence baseline for the multi-modal target."
        ),
        "config": _base_config(
            n_modes=8,
            occupied_modes=(1, 6),
            encoding_phase_idx=17,
            target_mode=(0, 7),
            data_function="multi_modal_data",
            lr=0.02,
            epochs=350,
            n_data=900,
            sigma_noise=0.005,
            unc_n_passes=20,
            unc_noise_std=0.03,
        ),
    },
    "multimodal_static_8m_pair16": {
        "title": "Multimodal Static 8-Mode (1,6)",
        "notes": (
            "Same capacity, different encoding/readout geometry to test basis mismatch."
        ),
        "config": _base_config(
            n_modes=8,
            occupied_modes=(1, 6),
            encoding_phase_idx=13,
            target_mode=(1, 6),
            data_function="multi_modal_data",
            lr=0.02,
            epochs=350,
            n_data=900,
            sigma_noise=0.005,
            unc_n_passes=20,
            unc_noise_std=0.03,
        ),
    },
    "multimodal_memristor_8m": {
        "title": "Multimodal Memristor 8-Mode",
        "notes": (
            "Control run: checks whether memory helps or smears the separated peaks."
        ),
        "config": _base_config(
            n_modes=8,
            occupied_modes=(1, 6),
            encoding_phase_idx=17,
            target_mode=(0, 7),
            data_function="multi_modal_data",
            lr=0.015,
            epochs=400,
            n_data=900,
            sigma_noise=0.005,
            unc_n_passes=20,
            unc_noise_std=0.03,
            memristive_phase_idx=(11,),
            feedback_modes=((5, 6),),
            memory_depth=4,
        ),
    },
    "step_memristor_8m": {
        "title": "Smooth Step Memristor 8-Mode",
        "notes": "Larger memristive configuration for the transition-like target.",
        "config": _base_config(
            n_modes=8,
            occupied_modes=(1, 6),
            encoding_phase_idx=17,
            target_mode=(0, 7),
            data_function="step_function_data",
            lr=0.015,
            epochs=400,
            n_data=900,
            sigma_noise=0.005,
            unc_n_passes=20,
            unc_noise_std=0.03,
            memristive_phase_idx=(9, 23),
            feedback_modes=((2, 3), (5, 6)),
            memory_depth=4,
        ),
    },
    "damped_cosine_static_10m": {
        "title": "Damped Cosine Static 10-Mode",
        "notes": "A larger static model for genuinely oscillatory targets.",
        "config": _base_config(
            n_modes=10,
            occupied_modes=(1, 8),
            encoding_phase_idx=27,
            target_mode=(0, 9),
            data_function="damped_cosine_data",
            lr=0.01,
            epochs=500,
            n_data=1200,
            sigma_noise=0.0,
            unc_n_passes=20,
            unc_noise_std=0.02,
        ),
    },
}

# Multi-modal presets first, then step / oscillatory follow-ups.
PRESET_NAMES = [
    "multimodal_static_8m_pair07",
    "multimodal_static_8m_pair16",
    "multimodal_memristor_8m",
    "step_memristor_8m",
    "damped_cosine_static_10m",
]


def _compute_metrics(
    y_test: np.ndarray,
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
) -> dict[str, float]:
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
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coverage_95": coverage_95,
        "mean_std": float(np.mean(std_preds)),
        "nll": nll,
    }


def _plot_summary(
    *,
    title: str,
    notes: str,
    history: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
    metrics: dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(history, color="tab:blue", lw=2)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(X_train, y_train, s=14, alpha=0.45, color="dimgray", label="Train")
    axes[1].plot(X_test, y_test, "k--", lw=1.4, label="Ground truth")
    axes[1].plot(X_test, mean_preds, color="tab:red", lw=2, label="Prediction")
    axes[1].fill_between(
        X_test,
        mean_preds - 1.96 * std_preds,
        mean_preds + 1.96 * std_preds,
        color="tab:red",
        alpha=0.22,
        label="95% CI",
    )
    axes[1].set(
        xlabel="x",
        ylabel="y",
        title=(
            f"{title}\n"
            f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.3f}  "
            f"Coverage95={metrics['coverage_95']:.3f}"
        ),
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(notes, fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def _run_preset(name: str, spec: dict[str, Any]) -> dict[str, float]:
    config = dict(spec["config"])
    title = str(spec["title"])
    notes = str(spec["notes"])
    data_function = str(config["data_function"])

    with Experiment(title, config=config) as exp:
        np.random.seed(int(config["seed"]))
        X_train, y_train, X_test, y_test = get_data(
            int(config["n_data"]),
            float(config["sigma_noise"]),
            data_function,
        )

        trained_state, history, _ = exp.train(X_train, y_train)
        enc_test = 2 * np.arccos(np.clip(X_test, 0.0, 1.0))
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
                "preset_name": name,
                "notes": notes,
                "data_function": data_function,
                "final_loss": float(history[-1]),
                **metrics,
            }
        )

        fig = _plot_summary(
            title=title,
            notes=notes,
            history=np.asarray(history),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            mean_preds=unc["mean"],
            std_preds=unc["std"],
            metrics=metrics,
        )
        exp.savefig(fig, "summary.png")
        plt.close(fig)

        logger.info(
            "[%s] final_loss=%.6f rmse=%.4f r2=%.4f coverage95=%.3f",
            name,
            float(history[-1]),
            metrics["rmse"],
            metrics["r2"],
            metrics["coverage_95"],
        )
        return metrics


def main() -> None:
    logger.info("Running %d hard-function presets", len(PRESET_NAMES))
    results: dict[str, dict[str, float]] = {}

    for preset_name in PRESET_NAMES:
        spec = PRESETS[preset_name]
        logger.info("▶ %s", preset_name)
        results[preset_name] = _run_preset(preset_name, spec)

    logger.info("Sweep summary:")
    for preset_name in PRESET_NAMES:
        metrics = results[preset_name]
        logger.info(
            "  %-28s rmse=%.4f r2=%.4f coverage95=%.3f",
            preset_name,
            metrics["rmse"],
            metrics["r2"],
            metrics["coverage_95"],
        )


if __name__ == "__main__":
    main()
