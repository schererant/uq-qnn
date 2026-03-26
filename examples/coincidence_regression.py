#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression example using coincidence (2-photon) measurements."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.experiment import Experiment

# ===================== EXPERIMENT CONFIG =====================
CONFIG = {
    # Circuit
    "n_modes": 6,
    "encoding_mode": 0,
    "target_mode": None,
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "encoding_phase_idx": None,
    # Task — coincidence mode
    "output_mode": "coincidence",
    "input_modes": (1, 4),
    "working_detectors": tuple(range(6)),
    "loss_type": "mse",
    "n_classes": 1,
    # Training
    "lr": 0.05,
    "epochs": 100,
    "n_samples": 1000,
    "memory_depth": 2,
    "n_photons": None,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    # Backend
    "sim_backend": "numpy",
    # Data
    "n_data": 100,
    "sigma_noise": 0.005,
    # Uncertainty
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}
# =============================================================


def main():
    with Experiment("Coincidence Regression", config=CONFIG) as exp:
        np.random.seed(CONFIG["seed"])

        X_train, y_train, X_test, y_test = get_data(
            CONFIG["n_data"], CONFIG["sigma_noise"], "quartic_data"
        )

        theta, history = exp.train(X_train, y_train)
        exp.save_metrics({"final_loss": history[-1]})

        enc_test = 2 * np.arccos(X_test)
        unc = exp.run_uncertainty_analysis(
            theta, enc_test,
            n_passes=CONFIG["unc_n_passes"],
            noise_std=CONFIG["unc_noise_std"],
        )
        mean_preds = unc["mean"]
        std_preds = unc["std"]

        mse = np.mean((mean_preds - y_test) ** 2)
        print(f"Test MSE: {mse:.6f}")
        exp.save_metrics({"test_mse": mse})

        # ── plots ──
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(history)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss (coincidence)")
        axes[0, 0].grid(True)

        axes[0, 1].scatter(X_train, y_train, s=20, label="Training data", alpha=0.7)
        axes[0, 1].plot(X_test, y_test, "k--", label="Ground truth")
        axes[0, 1].plot(X_test, mean_preds, "r-", label="Mean prediction")
        axes[0, 1].fill_between(
            X_test,
            mean_preds - 2 * std_preds,
            mean_preds + 2 * std_preds,
            color="r", alpha=0.3, label="95% CI",
        )
        axes[0, 1].set(xlabel="x", ylabel="y", title="Regression with Uncertainty (coincidence)")
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
            [0, np.max(std_preds)], [0, 2 * np.max(std_preds)],
            "k--", label="y=2x (well calibrated)",
        )
        axes[1, 1].set(
            xlabel="Uncertainty (std)", ylabel="Absolute error",
            title="Calibration: Error vs. Uncertainty",
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        fig.tight_layout()
        exp.savefig(fig, "coincidence_regression_with_uncertainty.png")
        # plt.show()


if __name__ == "__main__":
    main()
