#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple regression example with Mauser-style data re-uploading (L layers).

Same quartic target as ``simple_regression.py``, but the scalar input x is
phase-encoded into one slot per layer of a stacked Clements mesh.
"""

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
# phi = 2*arccos(x) is injected into slot 0 of each of L stacked Clements blocks
# (n_enc_features=1, n_layers=L). Compare to two_moons_classification.py which
# uses the same scheme with two features per layer.
N_LAYERS = 2
N_MODES = 4
PHASES_PER_LAYER = N_MODES * (N_MODES - 1)  # = 12 for n_modes=4
# One encoding slot per layer block (slot 0 of layer 0, slot 0 of layer 1, ...).
ENCODING_PHASE_IDX = tuple(layer * PHASES_PER_LAYER for layer in range(N_LAYERS))

CONFIG = {
    # Circuit — L=2 re-uploading; same scalar x injected into one phase per layer
    "n_modes": N_MODES,
    "n_layers": N_LAYERS,
    "input_state": (1, 0, 0, 0),
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "n_enc_features": 1,
    "photon_distinguishability": None,
    "target_mode": (1,),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    # Task
    "output_mode": "singles",
    "working_detectors": None,
    "loss_type": "mse",
    "n_classes": 1,
    # Training — smaller mesh + re-uploading: modest lr, more epochs to push through plateaus
    "lr": 0.04,
    "epochs": 1500,
    "n_samples": 300,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    # Backend
    "sim_backend": "numpy",
    # Data
    "n_data": 200,
    "sigma_noise": 0.005,
    # Uncertainty
    "unc_n_passes": 100,
    "unc_noise_std": 0.05,
}
# =============================================================


def encode_1d_to_phases_multi(X: np.ndarray, n_layers: int) -> np.ndarray:
    """Encode 1D inputs into ``n_layers`` repeated phase slots; shape ``(n, L)``."""
    phi = 2 * np.arccos(np.clip(np.asarray(X, dtype=np.float64), 0.0, 1.0))
    return np.tile(phi.reshape(-1, 1), (1, n_layers))


def main():
    with Experiment("Simple Regression (Re-uploading)", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))  # type: ignore[arg-type]

        X_train, y_train, X_test, y_test = get_data(
            CONFIG["n_data"], CONFIG["sigma_noise"], "quartic_data"
        )

        enc_train = encode_1d_to_phases_multi(X_train, n_layers=N_LAYERS)
        enc_test = encode_1d_to_phases_multi(X_test, n_layers=N_LAYERS)

        trained_state, history, _ = exp.train(enc_train, y_train, encoded=True)
        exp.save_metrics({"final_loss": history[-1]})

        unc = exp.run_uncertainty_analysis(
            trained_state,
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
        axes[0, 1].set(
            xlabel="x",
            ylabel="y",
            title=f"Regression with Uncertainty (L={N_LAYERS} re-uploading)",
        )
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
        exp.savefig(fig, "regression_with_uncertainty_reuploading.png")
        # plt.show()


if __name__ == "__main__":
    main()
