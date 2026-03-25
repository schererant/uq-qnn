#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple regression example using the UQ-QNN framework.

This example demonstrates how to:
1. Generate synthetic data
2. Train a photonic neural network
3. Evaluate predictions and uncertainty
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.training import train_pytorch
from src.simulation import (
    run_simulation_sequence_np,
    sim_logger,
    uncertainty_forward_pass,
)
from src.utils import config

SIM_BACKEND = "numpy"


def main():
    """Run a simple regression example."""
    print("=== UQ-QNN: Simple Regression Example ===")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configure parameters
    config["n_data"] = 20
    config["sigma_noise"] = 0.005
    config["lr"] = 0.05
    config["epochs"] = 3
    config["memory_depth"] = 2
    n_modes = 6
    n_phases = n_modes * (n_modes - 1)  # Clements: 3x3 = 6 phases
    config["phase_idx"] = tuple(range(n_phases))
    n_photons = tuple([1] * n_phases)
    n_samples = 20
    target_mode = (n_modes - 2,)
    memristive_output_modes = None
    memristive_phase_idx = None
    encoding_phase_idx = None

    # Generate synthetic data
    print("Generating synthetic data...")
    X_train, y_train, X_test, y_test = get_data(
        config["n_data"], config["sigma_noise"], "quartic_data"
    )

    # Train the model with discrete phases
    print("Training model with discrete phases...")
    theta_discrete, history_discrete = train_pytorch(
        X_train,
        y_train,
        memory_depth=config["memory_depth"],
        lr=config["lr"],
        epochs=config["epochs"],
        n_samples=n_samples,
        n_swipe=0,
        swipe_span=0.0,
        n_modes=n_modes,
        encoding_mode=0,
        n_photons=n_photons,
        target_mode=target_mode,
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes,
        encoding_phase_idx=encoding_phase_idx,
        backend=SIM_BACKEND,
    )

    # Generate predictions
    print("Generating predictions...")
    enc_test = 2 * np.arccos(X_test)
    preds_discrete = run_simulation_sequence_np(
        theta_discrete,
        config["memory_depth"],
        n_samples,
        encoded_phases=enc_test,
        n_swipe=0,
        swipe_span=0.0,
        n_modes=n_modes,
        encoding_mode=0,
        target_mode=target_mode,
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes,
        encoding_phase_idx=encoding_phase_idx,
        backend=SIM_BACKEND,
    )

    # Compute MSE
    mse_discrete = np.mean((preds_discrete - y_test) ** 2)
    print(f"Discrete mode MSE: {mse_discrete:.6f}")

    # Add uncertainty estimation through multiple forward passes
    print("Estimating uncertainty through multiple forward passes...")
    n_forward_passes = 10
    all_preds = np.zeros((len(X_test), n_forward_passes))

    unc_cfg = {
        "memory_depth": config["memory_depth"],
        "n_swipe": 0,
        "swipe_span": 0.0,
        "n_modes": n_modes,
        "encoding_mode": 0,
        "target_mode": target_mode,
        "memristive_phase_idx": memristive_phase_idx,
        "memristive_output_modes": memristive_output_modes,
        "encoding_phase_idx": encoding_phase_idx,
        "output_mode": "singles",
        "input_modes": None,
        "working_detectors": None,
        "noise_std": None,
        "backend": SIM_BACKEND,
    }
    rng = np.random.default_rng(123)
    jobs = []
    for _ in range(n_forward_passes):
        sample_count = n_samples + int(rng.integers(-100, 101))
        sample_count = max(100, sample_count)
        perturbed_theta = theta_discrete.copy()
        if memristive_phase_idx is not None:
            perturbed_theta[:-1] += rng.normal(0, 0.05, size=len(perturbed_theta) - 1)
        else:
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
    for i, preds in enumerate(results):
        all_preds[:, i] = preds

    # Compute mean and standard deviation of predictions
    mean_preds = np.mean(all_preds, axis=1)
    std_preds = np.std(all_preds, axis=1)

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot 1: Training loss
    plt.subplot(2, 2, 1)
    plt.plot(history_discrete)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Plot 2: Data and predictions
    plt.subplot(2, 2, 2)
    plt.scatter(X_train, y_train, s=20, label="Training data", alpha=0.7)
    plt.plot(X_test, y_test, "k--", label="Ground truth")
    plt.plot(X_test, mean_preds, "r-", label="Mean prediction")

    # Uncertainty bounds (mean ± 2*std)
    plt.fill_between(
        X_test,
        mean_preds - 2 * std_preds,
        mean_preds + 2 * std_preds,
        color="r",
        alpha=0.3,
        label="95% confidence interval",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regression with Uncertainty")
    plt.legend()
    plt.grid(True)

    # Plot 3: Uncertainty visualization
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(
        X_test, std_preds, c=np.abs(mean_preds - y_test), cmap="viridis"
    )
    plt.colorbar(scatter, label="Absolute error")
    plt.xlabel("x")
    plt.ylabel("Standard deviation")
    plt.title("Uncertainty vs. Input")
    plt.grid(True)

    # Plot 4: Calibration plot (prediction error vs. uncertainty)
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
    plt.savefig("regression_with_uncertainty.png", dpi=300)
    plt.show()

    # Print simulation statistics
    sim_logger.report()


if __name__ == "__main__":
    main()
