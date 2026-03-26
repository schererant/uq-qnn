#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple regression example using the UQ-QNN framework."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger

logger = get_logger(__name__)

# ===================== EXPERIMENT CONFIG =====================
CONFIG = {
    # Circuit
    "n_modes": 6,  # waveguide modes; gives 6*(6-1)=30 phase parameters
    "encoding_mode": 4,  # input mode that receives the data-encoded phase
    "target_mode": (4,),  # output mode(s) whose Born-rule probability is the prediction
    "memristive_phase_idx": None,  # phase index(es) with history-dependent feedback; None = pure Clements
    "memristive_output_modes": None,  # output mode pairs feeding back into memristive phases
    "encoding_phase_idx": None,  # override which phase slot holds the encoding; None = auto
    # Task
    "output_mode": "singles",  # "singles" = 1-photon probabilities; "coincidence" = 2-photon
    "input_modes": None,  # coincidence only: modes where the two photons enter
    "working_detectors": None,  # coincidence only: functioning detector indices for postselection
    "loss_type": "mse",  # "mse" for regression, "cross_entropy" for classification
    "n_classes": 1,  # 1 for regression; must equal len(target_mode) for classification
    # Training
    "lr": 0.05,  # Adam learning rate
    "epochs": 100,  # full passes over the training set
    "n_samples": 20,  # photon samples per data point (higher = less shot noise, slower)
    "memory_depth": 2,  # memristor buffer length (past time steps); irrelevant here
    "n_photons": None,  # photon count per phase for PSR shift computation; None = auto-infer
    "n_swipe": 0,  # phase points swept per sample in continuous-swipe mode; 0 = discrete
    "swipe_span": 0.0,  # total phase range (rad) swept around each encoded phase
    "noise_std": None,  # Gaussian noise on coincidence counts; None = noiseless
    "seed": 42,  # RNG seed for parameter initialisation and UQ pass noise
    # Backend
    "sim_backend": "numpy",  # "numpy" = fast vectorised path; "perceval" = full SLOS (required for memristor)
    # Data
    "n_data": 20,  # number of synthetic training + test samples to generate
    "sigma_noise": 0.005,  # label noise std on the synthetic quartic targets
    # Uncertainty
    "unc_n_passes": 100,  # number of noisy forward passes for uncertainty estimation
    "unc_noise_std": 0.05,  # std of Gaussian noise added to phases on each UQ pass
}
# =============================================================


def main():
    with Experiment("Simple Regression", config=CONFIG) as exp:
        np.random.seed(CONFIG["seed"])

        X_train, y_train, X_test, y_test = get_data(
            CONFIG["n_data"], CONFIG["sigma_noise"], "quartic_data"
        )

        theta, history = exp.train(X_train, y_train)
        exp.save_metrics({"final_loss": history[-1]})

        enc_test = 2 * np.arccos(X_test)
        unc = exp.run_uncertainty_analysis(
            theta,
            enc_test,
            n_passes=CONFIG["unc_n_passes"],
            noise_std=CONFIG["unc_noise_std"],
        )
        mean_preds = unc["mean"]
        std_preds = unc["std"]

        mse = np.mean((mean_preds - y_test) ** 2)
        logger.info("Test MSE: %.6f", mse)
        exp.save_metrics({"test_mse": mse})

        # ── plots ──
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(history)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
        axes[0, 0].grid(True)

        axes[0, 1].scatter(X_train, y_train, s=20, label="Training data", alpha=0.7)
        axes[0, 1].plot(X_test, y_test, "k--", label="Ground truth")
        axes[0, 1].plot(X_test, mean_preds, "r-", label="Mean prediction")
        axes[0, 1].fill_between(
            X_test,
            mean_preds - 2 * std_preds,
            mean_preds + 2 * std_preds,
            color="r",
            alpha=0.3,
            label="95% CI",
        )
        axes[0, 1].set(xlabel="x", ylabel="y", title="Regression with Uncertainty")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        sc = axes[1, 0].scatter(
            X_test, std_preds, c=np.abs(mean_preds - y_test), cmap="viridis"
        )
        fig.colorbar(sc, ax=axes[1, 0], label="Absolute error")
        axes[1, 0].set(xlabel="x", ylabel="Std dev", title="Uncertainty vs. Input")
        axes[1, 0].grid(True)

        axes[1, 1].scatter(std_preds, np.abs(mean_preds - y_test), alpha=0.7)
        axes[1, 1].plot(
            [0, np.max(std_preds)], [0, 2 * np.max(std_preds)], "k--", label="y=2x"
        )
        axes[1, 1].set(
            xlabel="Uncertainty (std)",
            ylabel="Absolute error",
            title="Calibration: Error vs. Uncertainty",
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        fig.tight_layout()
        exp.savefig(fig, "regression_with_uncertainty.png")
        # plt.show()


if __name__ == "__main__":
    main()
