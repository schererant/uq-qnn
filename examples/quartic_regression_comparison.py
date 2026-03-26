#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quartic regression comparison: 6x6 architecture with different memristor configurations.

Compares four scenarios on the quartic function (y = x^4):
1. No memristor - standard Clements 6x6
2. Memristors on 4th & 5th MZI, memory_depth=1 (minimal memory)
3. Memristors on 4th & 5th MZI, memory_depth=2
4. Memristors on 4th & 5th MZI, memory_depth=3

For 6 modes, the 4th MZI (modes 1,2) uses phase indices 6,7.
The 5th MZI (modes 3,4) uses phase indices 8,9.
We use phase 6 and 8 as memristive.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_logging import example_run, write_summary

from src.config import SimConfig
from src.data import get_data
from src.simulation import run_simulation_sequence_np, sim_logger
from src.training import train_pytorch

SIM_BACKEND = "numpy"
from src.circuit_visualization import save_circuit_annotated

# # 6x6 Clements: 4th MZI = phases 6,7 (modes 1,2) | 5th MZI = phases 8,9 (modes 3,4)
# N_MODES = 6
# MEMRISTIVE_PHASE_IDX = [6, 8]  # 4th and 5th MZI (first phase of each)
# # Custom output modes for feedback: (mode_p1, mode_p2) per memristive phase.
# # Default (None) uses each MZI's own output modes. Example: [(1, 2), (3, 4)]
# MEMRISTIVE_OUTPUT_MODES = [(1, 2), (3, 4)]  # 4th MZI outputs, 5th MZI outputs
# VERBOSE = True  # Set True for per-epoch loss and parameter printing
# #TODO: print gradient methods (consistent?)
# #TODO: desfault psr?

# 6x6 Clements: 4th MZI = phases 6,7 (modes 1,2) | 5th MZI = phases 8,9 (modes 3,4)
N_MODES = 3
MEMRISTIVE_PHASE_IDX = [2]  # 4th and 5th MZI (first phase of each)
# Custom output modes for feedback: (mode_p1, mode_p2) per memristive phase.
# Default (None) uses each MZI's own output modes. Example: [(1, 2), (3, 4)]
MEMRISTIVE_OUTPUT_MODES = [(1, 2)]  # 4th MZI outputs, 5th MZI outputs
VERBOSE = True  # Set True for per-epoch loss and parameter printing
# TODO: print gradient methods (consistent?)
# TODO: desfault psr?


def run_experiment(
    label: str,
    memristive_phase_idx,
    memory_depth: int,
    X_train,
    y_train,
    X_test,
    y_test,
    n_samples: int,
    lr: float,
    epochs: int,
    memristive_output_modes=None,
    verbose: bool = VERBOSE,
):
    """Train and evaluate one configuration."""
    print(f"\n--- {label} ---")
    # Use inline encoding on the first MZI internal phase (phase index 0)
    ENCODING_PHASE_IDX = 0
    mom = memristive_output_modes
    if mom is not None:
        mom = tuple(tuple(p) for p in mom)
    sim_cfg = SimConfig(
        n_modes=N_MODES,
        encoding_mode=0,
        target_mode=(N_MODES - 1,),
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=mom,
        encoding_phase_idx=ENCODING_PHASE_IDX,
        output_mode="singles",
        input_modes=None,
        working_detectors=None,
        noise_std=None,
        n_samples=n_samples,
        memory_depth=memory_depth,
        n_swipe=0,
        swipe_span=0.0,
        n_photons=None,
        backend=SIM_BACKEND,
        loss_type="mse",
        n_classes=1,
    )
    theta, history = train_pytorch(
        X_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=lr,
        epochs=epochs,
        seed=42,
        verbose=verbose,
    )
    enc_test = 2 * np.arccos(X_test)
    preds = run_simulation_sequence_np(
        theta,
        enc_test,
        sim_cfg,
    )
    mse = np.mean((preds - y_test) ** 2)
    print(f"MSE: {mse:.6f}")
    return theta, history, preds, mse


def main():
    print("=== Quartic Regression: 6x6 Architecture Comparison ===")

    with example_run(__file__) as (report_dir, logger):
        np.random.seed(42)

        n_data = 100
        sigma_noise = 0.02
        lr = 0.04
        epochs = 300
        n_samples = 20

        # Save annotated circuit diagram (encoding, target, memristive phases)
        print("Saving annotated circuit diagram...")
        # Visualize the same inline-encoding configuration used in training
        ENCODING_PHASE_IDX = 0
        save_circuit_annotated(
            str(report_dir / "quartic_circuit_annotated.png"),
            n_modes=N_MODES,
            encoding_mode=0,
            target_mode=(N_MODES - 1,),
            memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
            memristive_output_modes=MEMRISTIVE_OUTPUT_MODES,
            encoding_phase_idx=ENCODING_PHASE_IDX,
        )

        logger.info("Generating quartic data (y = x^4)…")
        X_train, y_train, X_test, y_test = get_data(
            n_data,
            sigma_noise,
            "quartic_data",
        )

        results = {}

        # 1. No memristor
        results["no_memristor"] = run_experiment(
            "6x6, no memristor",
            memristive_phase_idx=None,
            memory_depth=2,  # unused when no memristor
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_samples=n_samples,
            lr=lr,
            epochs=epochs,
        )

        # 2. Memristors, depth 1 (minimal memory), custom output modes
        results["memristor_depth1"] = run_experiment(
            "6x6, memristors (4th & 5th MZI), depth=1",
            memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
            memory_depth=1,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_samples=n_samples,
            lr=lr,
            epochs=epochs,
            memristive_output_modes=MEMRISTIVE_OUTPUT_MODES,
        )

        # 3. Memristors, depth 2, custom output modes
        results["memristor_depth2"] = run_experiment(
            "6x6, memristors (4th & 5th MZI), depth=2",
            memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
            memory_depth=2,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_samples=n_samples,
            lr=lr,
            epochs=epochs,
            memristive_output_modes=MEMRISTIVE_OUTPUT_MODES,
        )

        # 4. Memristors, depth 3, custom output modes
        results["memristor_depth3"] = run_experiment(
            "6x6, memristors (4th & 5th MZI), depth=3",
            memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
            memory_depth=3,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_samples=n_samples,
            lr=lr,
            epochs=epochs,
            memristive_output_modes=MEMRISTIVE_OUTPUT_MODES,
        )

        # Summary table
        print("\n" + "=" * 60)
        print("Summary (MSE on test set)")
        print("=" * 60)
        for key, (_, _, _, mse) in results.items():
            print(f"  {key:25s}: {mse:.6f}")
        print("=" * 60)

        # Plot: predictions comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        labels = [
            ("No memristor", "no_memristor"),
            ("Memristors, depth=1", "memristor_depth1"),
            ("Memristors, depth=2", "memristor_depth2"),
            ("Memristors, depth=3", "memristor_depth3"),
        ]

        for ax, (title, key) in zip(axes.flat, labels):
            _, _, preds, mse = results[key]
            ax.scatter(X_train, y_train, s=15, alpha=0.6, label="Train", c="gray")
            ax.plot(X_test, y_test, "k--", lw=2, label="Ground truth")
            ax.plot(X_test, preds, "r-", lw=1.5, label="Prediction")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{title}\nMSE = {mse:.6f}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Quartic Regression: 6x6 Clements, Memristors on 4th & 5th MZI", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(report_dir / "quartic_regression_comparison.png", dpi=300)
        plt.show()

        # Plot: training loss comparison
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for key, (_, history, _, _) in results.items():
            ax2.plot(history, label=key, alpha=0.8)
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / "quartic_regression_loss_comparison.png", dpi=300)
        plt.show()

        sim_logger.report()

        logger.info(
            "Experiment metrics: %s",
            {key: float(mse) for key, (_, _, _, mse) in results.items()},
        )

        write_summary(
            report_dir,
            summary={key: float(mse) for key, (_, _, _, mse) in results.items()},
            simulation_stats=sim_logger.stats_dict(),
        )


if __name__ == "__main__":
    main()
