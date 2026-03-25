#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression example using coincidence (2-photon) measurements.

Two photons are injected into modes 1 and 4 of a 6-mode Clements circuit.
The output observable is the probability of the first working coincidence
channel (CC01 by default), used as a scalar regression target.

Working detectors: all modes active by default
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import make_run_dir, print_report_banner, write_run_summary

from src.data import get_data
from src.training import train_pytorch
from src.simulation import (
    run_simulation_sequence_np,
    sim_logger,
    uncertainty_forward_pass,
)
from src.utils import config

# Default simulation backend (fast NumPy unitary); use "perceval" for full Perceval pipeline
SIM_BACKEND = "numpy"


def main():
    print("=== UQ-QNN: Coincidence Regression Example ===")

    report_dir = make_run_dir(__file__)
    print_report_banner(report_dir)

    np.random.seed(42)

    # --- Parameters ---
    n_modes = 6
    n_phases = n_modes * (n_modes - 1)
    encoding_mode = 0
    input_modes = (1, 4)  # 2-photon input ports
    working_detectors = tuple(range(n_modes))  # all detectors active

    config["n_data"] = 100
    config["sigma_noise"] = 0.005
    config["lr"] = 0.05
    config["epochs"] = 60
    config["memory_depth"] = 2
    config["phase_idx"] = tuple(range(n_phases))
    n_input_photons = len(input_modes)
    n_photons = tuple([n_input_photons] * n_phases)
    n_samples = 100

    # --- Print settings ---
    from src.coincidence import get_cc_labels, working_detectors_to_cc_indices

    cc_labels = get_cc_labels(n_modes)
    active_cc_indices = working_detectors_to_cc_indices(working_detectors, n_modes)
    print("\n--- Settings ---")
    print(f"  n_modes:           {n_modes}")
    print(f"  n_phases:          {n_phases}")
    print(f"  encoding_mode:     {encoding_mode}")
    print(f"  input_modes:       {input_modes}")
    print(f"  working_detectors: {working_detectors}")
    print(
        f"  active CC channels ({len(active_cc_indices)}): {[cc_labels[i] for i in active_cc_indices]}"
    )
    print(f"  n_data:            {config['n_data']}")
    print(f"  sigma_noise:       {config['sigma_noise']}")
    print(f"  memory_depth:      {config['memory_depth']}")
    print(f"  n_samples:         {n_samples}")
    print(f"  lr:                {config['lr']}")
    print(f"  epochs:            {config['epochs']}")
    print()

    # --- Data ---
    print("Generating synthetic data...")
    X_train, y_train, X_test, y_test = get_data(
        config["n_data"],
        config["sigma_noise"],
        "quartic_data",
    )

    # --- Training ---
    print("Training model with coincidence measurements...")
    theta_opt, history = train_pytorch(
        X_train,
        y_train,
        memory_depth=config["memory_depth"],
        lr=config["lr"],
        epochs=config["epochs"],
        n_samples=n_samples,
        n_swipe=0,
        swipe_span=0.0,
        n_modes=n_modes,
        encoding_mode=encoding_mode,
        n_photons=n_photons,
        target_mode=None,
        memristive_phase_idx=None,
        memristive_output_modes=None,
        encoding_phase_idx=None,
        output_mode="coincidence",
        input_modes=input_modes,
        working_detectors=working_detectors,
        backend=SIM_BACKEND,
    )

    # --- Predictions ---
    print("Generating predictions...")
    enc_test = 2 * np.arccos(X_test)
    preds = run_simulation_sequence_np(
        theta_opt,
        config["memory_depth"],
        n_samples,
        encoded_phases=enc_test,
        n_swipe=0,
        swipe_span=0.0,
        n_modes=n_modes,
        encoding_mode=encoding_mode,
        target_mode=None,
        memristive_phase_idx=None,
        memristive_output_modes=None,
        encoding_phase_idx=None,
        output_mode="coincidence",
        input_modes=input_modes,
        working_detectors=working_detectors,
        backend=SIM_BACKEND,
    )

    mse = np.mean((preds - y_test) ** 2)
    print(f"Test MSE: {mse:.6f}")

    # --- Uncertainty via multiple forward passes ---
    print("Estimating uncertainty through multiple forward passes...")
    n_forward_passes = 10
    all_preds = np.zeros((len(X_test), n_forward_passes))

    unc_cfg = {
        "memory_depth": config["memory_depth"],
        "n_swipe": 0,
        "swipe_span": 0.0,
        "n_modes": n_modes,
        "encoding_mode": encoding_mode,
        "target_mode": None,
        "memristive_phase_idx": None,
        "memristive_output_modes": None,
        "encoding_phase_idx": None,
        "output_mode": "coincidence",
        "input_modes": input_modes,
        "working_detectors": working_detectors,
        "noise_std": None,
        "backend": SIM_BACKEND,
    }
    rng = np.random.default_rng(123)
    jobs = []
    for _ in range(n_forward_passes):
        sample_count = max(100, n_samples + int(rng.integers(-10, 11)))
        perturbed_theta = theta_opt.copy()
        perturbed_theta += rng.normal(0, 0.05, size=len(perturbed_theta))
        jobs.append((perturbed_theta, sample_count, enc_test, unc_cfg))

    max_workers = min(n_forward_passes, (os.cpu_count() or 2))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            tqdm(
                pool.map(uncertainty_forward_pass, jobs),
                total=n_forward_passes,
                desc="Forward passes",
            )
        )
    for i, p in enumerate(results):
        all_preds[:, i] = p

    mean_preds = np.mean(all_preds, axis=1)
    std_preds = np.std(all_preds, axis=1)

    # --- Plot ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (coincidence)")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(X_train, y_train, s=20, label="Training data", alpha=0.7)
    plt.plot(X_test, y_test, "k--", label="Ground truth")
    plt.plot(X_test, mean_preds, "r-", label="Mean prediction")
    plt.fill_between(
        X_test,
        mean_preds - 2 * std_preds,
        mean_preds + 2 * std_preds,
        color="r",
        alpha=0.3,
        label="95% CI",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regression with Uncertainty (coincidence)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    scatter = plt.scatter(
        X_test, std_preds, c=np.abs(mean_preds - y_test), cmap="viridis"
    )
    plt.colorbar(scatter, label="Absolute error")
    plt.xlabel("x")
    plt.ylabel("Standard deviation")
    plt.title("Uncertainty vs. Input")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.scatter(std_preds, np.abs(mean_preds - y_test), alpha=0.7)
    plt.plot(
        [0, np.max(std_preds)],
        [0, 2 * np.max(std_preds)],
        "k--",
        label="y=2x (well calibrated)",
    )
    plt.xlabel("Uncertainty (std)")
    plt.ylabel("Absolute error")
    plt.title("Calibration: Error vs. Uncertainty")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(report_dir / "coincidence_regression_with_uncertainty.png", dpi=300)
    plt.show()

    write_run_summary(
        report_dir,
        metrics={"test_mse": float(mse)},
        artifacts=["coincidence_regression_with_uncertainty.png"],
    )

    sim_logger.report()


if __name__ == "__main__":
    main()
