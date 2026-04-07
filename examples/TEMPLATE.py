#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<EXPERIMENT TITLE>

<One-paragraph description of what this experiment does and why.
Explain the scientific question or comparison being made.>

Produced artefacts
------------------
<list each output file and what it shows, e.g.:>
predictions.png         Fit + 95% CI
training_loss.png       Log-scale convergence curve
"""

# ════════════════════════════════════════════════════════════════════════════
# TEMPLATE GUIDE (delete this block before running)
# ════════════════════════════════════════════════════════════════════════════
#
# This template follows the UQ-QNN experiment convention. To create a new
# experiment, copy this file to examples/<your_experiment>.py and fill in
# the sections marked with TODO.
#
# Key conventions:
#   1. CONFIG must contain ALL required keys (no hidden defaults).
#   2. Use `with Experiment(...) as exp:` context manager for lifecycle.
#   3. Use exp.train(), exp.predict(), exp.run_uncertainty_analysis() —
#      these handle encoding, hardware noise, and logging automatically.
#   4. Use exp.savefig() to save plots (auto-tracks artifacts).
#   5. Use exp.save_metrics() to record numeric results.
#   6. Use matplotlib "Agg" backend for headless environments.
#   7. Reports go to reports/<experiment_slug>/<timestamp>/.
#
# Available data functions (src/data.py):
#   Regression (1D → 1D, values in [0,1]):
#     "quartic_data", "neg_quadratic_data", "neg_qubic_data",
#     "sinusoid_data", "multi_modal_data", "step_function_data",
#     "oscillating_poly_data", "damped_cosine_data"
#   Classification (2D):
#     get_classification_data(n_samples, dataset="two_moons"|"circles"|...)
#     get_two_moons_data(n_samples)
#     encode_2d_to_phase(X_2d, method="weighted_sum") → 1D phases
#
# Measurement modes:
#   "singles"     — single-photon Born-rule probabilities (fast, simple)
#   "coincidence" — two-photon correlations (input_state length 2, working_detectors, distinguishability)
#
# Hardware profiles (src/hardware.py):
#   IDEAL            — no noise, numpy backend
#   LAB_6MODE        — realistic lab noise (Gaussian + shot + dark counts)
#   NOISY_PROTOTYPE  — higher noise for stress-testing
#   Or pass hardware="ideal" / hardware="lab_6mode" as a string.
#
# ════════════════════════════════════════════════════════════════════════════

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger

logger = get_logger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# All keys below are REQUIRED by the Experiment class — do not omit any.
# Adjust values to match your experiment's needs.

CONFIG = {
    # ── Circuit geometry ──────────────────────────────────────────
    "n_modes": 6,              # Number of waveguide modes (circuit width)
    "input_state": (0,),       # Occupied input modes: one int for singles, two for coincidence
    "encoding_phase_idx": 5,   # Mesh phase index that receives the encoded data phase
    "photon_distinguishability": None,  # None for singles; "indistinguishable" for two-photon
    "target_mode": (5,),       # Output mode(s) to read — tuple of ints
    "memristive_phase_idx": None,       # Index/indices of memristive phases, or None
    "memristive_output_modes": None,    # Feedback mode pairs for memristor, or None

    # ── Measurement mode ──────────────────────────────────────────
    "output_mode": "singles",  # "singles" or "coincidence"
    "working_detectors": None, # For coincidence: non-empty tuple of detector indices; None for singles
    "noise_std": None,         # Simulation-level noise std (None = no noise)

    # ── Simulation ────────────────────────────────────────────────
    "n_samples": 300,          # Number of simulation samples
    "memory_depth": 1,         # Memristive memory depth
    "n_swipe": 0,              # Number of discrete phase swipes (0 = off)
    "swipe_span": 0.0,         # Continuous swipe range
    "sim_backend": "numpy",    # "numpy" (fast) or "perceval" (memristive/SLOS)

    # ── Task ──────────────────────────────────────────────────────
    "loss_type": "mse",        # "mse" (regression) or "cross_entropy" (classification)
    "n_classes": 1,            # 1 for regression, >1 for classification

    # ── Training ──────────────────────────────────────────────────
    "lr": 0.05,                # Adam learning rate
    "epochs": 150,             # Training epochs
    "seed": 42,                # Random seed for reproducibility

    # ── Data (experiment-level, not passed to SimConfig) ──────────
    "n_data": 80,              # Number of training data points
    "sigma_noise": 0.02,       # Noise added to training targets

    # ── Uncertainty quantification ────────────────────────────────
    "unc_n_passes": 15,        # Number of noisy forward passes for UQ
    "unc_noise_std": 0.05,     # Std-dev of parameter perturbation per UQ pass
}

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    with Experiment(
        "TODO: Experiment Name",  # Used for report directory name
        config=CONFIG,
        # hardware=None,          # Optional: "ideal", "lab_6mode", or HardwareProfile
    ) as exp:
        np.random.seed(CONFIG["seed"])

        # ── 1. Load data ──────────────────────────────────────────
        # TODO: Choose your data function and adjust n_data / sigma_noise above.
        X_train, y_train, X_test, y_test = get_data(
            CONFIG["n_data"], CONFIG["sigma_noise"], "quartic_data"
        )

        # ── 2. Train ──────────────────────────────────────────────
        theta, history = exp.train(X_train, y_train)
        exp.save_metrics({"final_loss": history[-1]})

        # ── 3. Predict with uncertainty ───────────────────────────
        enc_test = 2 * np.arccos(X_test)
        unc = exp.run_uncertainty_analysis(
            theta,
            enc_test,
            n_passes=CONFIG["unc_n_passes"],
            noise_std=CONFIG["unc_noise_std"],
        )
        mean_preds = unc["mean"]
        std_preds = unc["std"]

        # ── 4. Compute & save metrics ─────────────────────────────
        mse = float(np.mean((mean_preds - y_test) ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum((mean_preds - y_test) ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        coverage = float(np.mean(
            (y_test >= mean_preds - 1.96 * std_preds)
            & (y_test <= mean_preds + 1.96 * std_preds)
        ))

        exp.save_metrics({
            "test_mse": mse,
            "test_rmse": rmse,
            "r2": r2,
            "coverage_95": coverage,
        })
        logger.info("RMSE=%.4f  R2=%.4f  Coverage@95%%=%.2f", rmse, r2, coverage)

        # ── 5. Plot ───────────────────────────────────────────────
        # TODO: Add/modify plots for your experiment.

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Training loss
        axes[0].plot(history)
        axes[0].set_yscale("log")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
        axes[0].grid(True, alpha=0.3)

        # Predictions + uncertainty
        axes[1].scatter(X_train, y_train, s=15, alpha=0.5, color="gray", label="Train")
        axes[1].plot(X_test, y_test, "k--", lw=1, label="Ground truth")
        axes[1].plot(X_test, mean_preds, "tab:blue", lw=1.5, label="Prediction")
        axes[1].fill_between(
            X_test,
            mean_preds - 1.96 * std_preds,
            mean_preds + 1.96 * std_preds,
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )
        axes[1].set(
            xlabel="x", ylabel="y",
            title=f"Predictions  (RMSE={rmse:.4f}, R²={r2:.3f})",
        )
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        exp.savefig(fig, "results.png", bbox_inches="tight")
        plt.close(fig)

        logger.info("Report saved to %s", exp.run_dir)


if __name__ == "__main__":
    main()
