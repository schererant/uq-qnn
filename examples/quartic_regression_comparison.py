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

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.training import train_pytorch
from src.simulation import run_simulation_sequence_np, sim_logger
from src.utils import config


# 6x6 Clements: 4th MZI = phases 6,7 (modes 1,2) | 5th MZI = phases 8,9 (modes 3,4)
N_MODES = 6
MEMRISTIVE_PHASE_IDX = [6, 8]  # 4th and 5th MZI (first phase of each)
# Custom output modes for feedback: (mode_p1, mode_p2) per memristive phase.
# Default (None) uses each MZI's own output modes. Example: [(1, 2), (3, 4)]
MEMRISTIVE_OUTPUT_MODES = [(1, 2), (3, 4)]  # 4th MZI outputs, 5th MZI outputs
VERBOSE = True  # Set True for per-epoch loss and parameter printing
#TODO: make output modes choosable 
#TODO: circuit printer in example
#TODO: print gradient methods (consistent?)
#TODO: desfault psr? 


def run_experiment(
    label: str,
    memristive_phase_idx,
    memory_depth: int,
    X_train, y_train, X_test, y_test,
    n_samples: int,
    lr: float,
    epochs: int,
    memristive_output_modes=None,
    verbose: bool = VERBOSE,
):
    """Train and evaluate one configuration."""
    print(f"\n--- {label} ---")
    theta, history = train_pytorch(
        X_train, y_train,
        memory_depth=memory_depth,
        lr=lr,
        epochs=epochs,
        n_samples=n_samples,
        n_swipe=0,
        swipe_span=0.0,
        n_modes=N_MODES,
        encoding_mode=0,
        target_mode=(N_MODES - 1,),
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes,
        verbose=verbose,
    )
    enc_test = 2 * np.arccos(X_test)
    preds = run_simulation_sequence_np(
        theta,
        memory_depth,
        n_samples,
        encoded_phases=enc_test,
        n_swipe=0,
        swipe_span=0.0,
        n_modes=N_MODES,
        encoding_mode=0,
        target_mode=(N_MODES - 1,),
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes,
    )
    mse = np.mean((preds - y_test) ** 2)
    print(f"MSE: {mse:.6f}")
    return theta, history, preds, mse


def main():
    print("=== Quartic Regression: 6x6 Architecture Comparison ===")

    np.random.seed(42)

    config['n_data'] = 80
    config['sigma_noise'] = 0.05
    config['lr'] = 0.03
    config['epochs'] = 2
    n_samples = 500

    print("Generating quartic data (y = x^4)...")
    X_train, y_train, X_test, y_test = get_data(
        config['n_data'],
        config['sigma_noise'],
        'quartic_data',
    )

    results = {}

    # 1. No memristor
    results['no_memristor'] = run_experiment(
        "6x6, no memristor",
        memristive_phase_idx=None,
        memory_depth=2,  # unused when no memristor
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        n_samples=n_samples, lr=config['lr'], epochs=config['epochs'],
    )

    # 2. Memristors, depth 1 (minimal memory), custom output modes
    results['memristor_depth1'] = run_experiment(
        "6x6, memristors (4th & 5th MZI), depth=1",
        memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
        memory_depth=1,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        n_samples=n_samples, lr=config['lr'], epochs=config['epochs'],
        memristive_output_modes=MEMRISTIVE_OUTPUT_MODES,
    )

    # 3. Memristors, depth 2, custom output modes
    results['memristor_depth2'] = run_experiment(
        "6x6, memristors (4th & 5th MZI), depth=2",
        memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
        memory_depth=2,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        n_samples=n_samples, lr=config['lr'], epochs=config['epochs'],
        memristive_output_modes=MEMRISTIVE_OUTPUT_MODES,
    )

    # 4. Memristors, depth 3, custom output modes
    results['memristor_depth3'] = run_experiment(
        "6x6, memristors (4th & 5th MZI), depth=3",
        memristive_phase_idx=MEMRISTIVE_PHASE_IDX,
        memory_depth=3,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        n_samples=n_samples, lr=config['lr'], epochs=config['epochs'],
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

    plt.suptitle("Quartic Regression: 6x6 Clements, Memristors on 4th & 5th MZI", fontsize=12)
    plt.tight_layout()
    plt.savefig("quartic_regression_comparison.png", dpi=300)
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
    plt.savefig("quartic_regression_loss_comparison.png", dpi=300)
    plt.show()

    sim_logger.report()


if __name__ == "__main__":
    main()
