#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware profile comparison example.

Trains the same circuit under different hardware profiles (ideal vs noisy)
and compares predictions and uncertainty.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.experiment import Experiment
from src.hardware import IDEAL, NOISY_PROTOTYPE, LAB_6MODE
from src.logging_config import get_logger

logger = get_logger(__name__)

# ===================== EXPERIMENT CONFIG =====================
# Shared across all hardware profiles — only the noise/backend differ.
BASE_CONFIG = {
    # Circuit
    "n_modes": 6,
    "input_state": tuple(1 if i == 0 else 0 for i in range(6)),
    "encoding_phase_idx": 0,
    "photon_distinguishability": None,
    "target_mode": (4,),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    # Measurement
    "output_mode": "singles",
    "working_detectors": None,
    "noise_std": None,
    # Simulation
    "n_samples": 20,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "sim_backend": "numpy",
    # Task
    "loss_type": "mse",
    "n_classes": 1,
    # Training
    "lr": 0.05,
    "epochs": 100,
    "seed": 42,
    # Data
    "n_data": 30,
    "sigma_noise": 0.005,
    # Uncertainty
    "unc_n_passes": 15,
    "unc_noise_std": 0.05,
}
# =============================================================

PROFILES = [
    ("ideal", IDEAL),
    ("lab_6mode", LAB_6MODE),
    ("noisy_prototype", NOISY_PROTOTYPE),
]


def main():
    np.random.seed(BASE_CONFIG["seed"])

    X_train, y_train, X_test, y_test = get_data(
        BASE_CONFIG["n_data"], BASE_CONFIG["sigma_noise"], "quartic_data"
    )

    all_results = {}

    for profile_name, profile in PROFILES:
        print(f"\n{'=' * 60}")
        print(f"  Hardware profile: {profile_name}")
        print(f"{'=' * 60}")

        config = BASE_CONFIG.copy()

        with Experiment(
            f"hw_compare_{profile_name}",
            config=config,
            hardware=profile,
        ) as exp:
            theta, history, model = exp.train(X_train, y_train)
            exp.save_metrics({"final_loss": history[-1]})

            enc_test = 2 * np.arccos(X_test)
            unc = exp.run_uncertainty_analysis(
                theta,
                enc_test,
                n_passes=BASE_CONFIG["unc_n_passes"],
                noise_std=BASE_CONFIG["unc_noise_std"],
                model=model,
            )
            mean_preds = unc["mean"]
            std_preds = unc["std"]

            mse = float(np.mean((mean_preds - y_test) ** 2))
            exp.save_metrics({"test_mse": mse})
            logger.info("[%s] Test MSE: %.6f", profile_name, mse)

            all_results[profile_name] = {
                "mean": mean_preds,
                "std": std_preds,
                "history": history,
                "mse": mse,
            }

    # ── comparison plot ──
    fig, axes = plt.subplots(
        1, len(PROFILES), figsize=(6 * len(PROFILES), 5), sharey=True
    )
    if len(PROFILES) == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:red"]

    for ax, (pname, _), color in zip(axes, PROFILES, colors):
        r = all_results[pname]
        ax.scatter(X_train, y_train, s=15, alpha=0.5, color="gray", label="Train")
        ax.plot(X_test, y_test, "k--", lw=1, label="Ground truth")
        ax.plot(X_test, r["mean"], color=color, lw=1.5, label="Mean prediction")
        ax.fill_between(
            X_test,
            r["mean"] - 2 * r["std"],
            r["mean"] + 2 * r["std"],
            color=color,
            alpha=0.2,
            label="95% CI",
        )
        ax.set_title(f"{pname}\nMSE = {r['mse']:.6f}")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("y")
    fig.suptitle("Hardware Profile Comparison: Quartic Regression", fontsize=13)
    fig.tight_layout()

    out_path = "reports/hardware_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"\nSaved comparison plot to {out_path}")
    plt.show()

    # ── summary ──
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for pname, _ in PROFILES:
        print(f"  {pname:20s}: MSE = {all_results[pname]['mse']:.6f}")


if __name__ == "__main__":
    main()
