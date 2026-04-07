#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memristor Circuit Visualization Example

This script creates and visualizes a memristor circuit with various inputs.
It demonstrates:
1. How to create a memristor circuit
2. How to visualize the circuit structure
3. How to simulate the circuit with different input states
4. How to analyze the output probabilities
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import begin_console_capture, write_run_summary

import perceval as pcvl
from src.circuits import memristor_circuit, build_circuit


def create_memristor_circuit():
    """Creates a memristor circuit with random phases."""
    # Initialize random phases
    phases = np.random.uniform(0, 2 * np.pi, 3)

    # Create the memristor circuit
    circuit = memristor_circuit(phases)

    print("Created memristor circuit with 3 modes")
    print(f"Phases: {phases}")

    return circuit, phases


def create_full_circuit(encoding_phase_idx=0):
    """Creates a full Clements circuit (3x3) with internal encoding."""
    n_modes = 3
    n_phases = n_modes * (n_modes - 1)
    phases = np.random.uniform(0, 2 * np.pi, n_phases)
    enc_phi = np.random.uniform(0, 2 * np.pi)

    circuit = build_circuit(
        phases=phases,
        enc_phi=enc_phi,
        n_modes=n_modes,
        encoding_phase_idx=encoding_phase_idx,
    )

    return circuit, phases, enc_phi


def simulate_with_input(circuit, input_state, n_samples=10000):
    """Simulates the circuit with a given input state."""
    processor = pcvl.Processor("SLOS", circuit)
    processor.with_input(input_state)

    # Run the simulation
    sampler = pcvl.algorithm.Sampler(processor)
    results = sampler.probs(n_samples)["results"]

    return results


def plot_output_distribution(results, input_state, title):
    """Plots the output distribution as a bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert results to array format for plotting
    states = []
    probs = []

    for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        states.append(str(state))
        probs.append(prob)

    # Plot
    ax.bar(states, probs, color="skyblue")
    ax.set_xlabel("Output State")
    ax.set_ylabel("Probability")
    ax.set_title(f"{title} - Input: {input_state}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add probabilities as text
    for i, v in enumerate(probs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center")

    return fig


def analyze_mode_probabilities(results, n_modes=3):
    """Analyzes the probability of finding a photon in each mode."""
    mode_probs = [0] * n_modes

    for state, prob in results.items():
        state_str = str(state)
        # Parse the state string which is like |0,1,0> to get photon counts
        photon_counts = state_str.strip("|>").split(",")
        for mode, count in enumerate(photon_counts):
            if int(count) > 0:
                mode_probs[mode] += prob

    return mode_probs


def plot_mode_probabilities(mode_probs, title):
    """Plots the probability of finding a photon in each mode."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot
    modes = list(range(len(mode_probs)))
    ax.bar(modes, mode_probs, color="coral")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Probability")
    ax.set_title(f"{title} - Mode Occupation Probabilities")
    ax.set_xticks(modes)

    # Add probabilities as text
    for i, v in enumerate(mode_probs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    return fig


def main():
    """Main function to create and visualize the memristor circuit."""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Parameters
    n_samples = 10000

    print("=== Memristor Circuit Visualization ===")

    report_dir, _end_capture = begin_console_capture(__file__)
    try:
        # 1. Create a standalone memristor circuit
        mem_circuit, phases = create_memristor_circuit()
    
        print("\nPhase parameters:")
        phase_names = ["phi1", "mem_phi", "phi3"]
        for i, phase in enumerate(phases):
            print(f"  {phase_names[i]}: {phase:.4f} radians ({phase * 180 / np.pi:.1f}°)")
    
        # 2. Create a full circuit with encoding
        enc_idx = 0
        full_circuit, phases, enc_phi = create_full_circuit(enc_idx)
        print(f"\nFull circuit created with internal encoding at phase index {enc_idx}")
        print(f"Encoding phase: {enc_phi:.4f} radians ({enc_phi * 180 / np.pi:.1f}°)")
    
        # 3. Define different input states
        input_states = [
            # Standard input state: |0,1,0>
            pcvl.BasicState([0, 1, 0]),
            # Alternative input: |1,0,0>
            pcvl.BasicState([1, 0, 0]),
            # Alternative input: |0,0,1>
            pcvl.BasicState([0, 0, 1]),
            # Two-photon input: |0,2,0>
            pcvl.BasicState([0, 2, 0]),
        ]
    
        # 4. Simulate and visualize results for each input state
        for i, input_state in enumerate(input_states):
            print(f"\nSimulating with input state {i + 1}: {input_state}")
    
            # Run simulation
            results = simulate_with_input(full_circuit, input_state, n_samples)
    
            # Print all results
            print("Output states and probabilities:")
            for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
                print(f"  {state}: {prob:.6f}")
    
            # Analyze mode probabilities
            mode_probs = analyze_mode_probabilities(results)
            print("Mode occupation probabilities:")
            for mode, prob in enumerate(mode_probs):
                print(f"  Mode {mode}: {prob:.6f}")
    
            # Plot output distribution
            fig1 = plot_output_distribution(results, input_state, f"Input State {i + 1}")
            fig1.savefig(
                report_dir / f"memristor_output_dist_{i + 1}.png",
                dpi=300,
                bbox_inches="tight",
            )
    
            # Plot mode probabilities
            fig2 = plot_mode_probabilities(mode_probs, f"Input State {i + 1}")
            fig2.savefig(
                report_dir / f"memristor_mode_probs_{i + 1}.png",
                dpi=300,
                bbox_inches="tight",
            )
    
        # 5. Now simulate quartic function training with the memristor circuit
        print("\n=== Simulating Quartic Function Training ===")
    
        # Import required modules for training
        from src.data import get_data
        from src.config import SimConfig
        from src.training import train_pytorch
        from src.simulation import run_simulation_sequence_np
        # Configure parameters
        n_data = 40  # Small dataset for demonstration
        sigma_noise = 0.05
        lr = 0.03
        epochs = 5  # Few epochs for quick demonstration
        memory_depth = 2
        sim_backend = "numpy"

        # Generate synthetic data
        X_train, y_train, X_test, y_test = get_data(
            n_data, sigma_noise, "quartic_data"
        )
    
        # Train the model
        print("Training model on quartic function...")
        n_modes = 3  # 3x3 Clements with memristive phase 2
        sim_cfg = SimConfig(
            n_modes=n_modes,
            input_state=(0,),
            encoding_phase_idx=1,
            photon_distinguishability=None,
            target_mode=(n_modes - 1,),
            memristive_phase_idx=[2],
            memristive_output_modes=None,
            output_mode="singles",
            working_detectors=None,
            noise_std=None,
            n_samples=n_samples // 20,
            memory_depth=memory_depth,
            n_swipe=0,
            swipe_span=0.0,
            backend=sim_backend,
            loss_type="mse",
            n_classes=1,
        )
        theta_opt, history = train_pytorch(
            X_train,
            y_train,
            sim_cfg=sim_cfg,
            lr=lr,
            epochs=epochs,
            seed=42,
        )
    
        # Print optimized parameters
        print("\nOptimized parameters:")
        param_names = ["phi1", "phi3", "weight"]
        for i, param in enumerate(theta_opt):
            print(f"  {param_names[i]}: {param:.6f}")
    
        # Generate predictions
        print("\nGenerating predictions...")
        enc_test = 2 * np.arccos(X_test)
        predictions = run_simulation_sequence_np(
            theta_opt,
            enc_test,
            sim_cfg,
        )
    
        # Compute MSE
        mse = np.mean((predictions - y_test) ** 2)
        print(f"Test MSE: {mse:.6f}")
    
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, label="Training data", alpha=0.7)
        plt.plot(X_test, y_test, "k--", label="True function (x^4)")
        plt.plot(X_test, predictions, "r-", label="Memristor predictions")
        plt.title("Memristor Circuit - Quartic Function Learning")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(report_dir / "memristor_quartic_function.png", dpi=300, bbox_inches="tight")
    
        # Plot training loss
        plt.figure(figsize=(8, 5))
        plt.plot(history)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.savefig(report_dir / "memristor_training_loss.png", dpi=300, bbox_inches="tight")
    
        # Show all plots
        plt.show()
    
        n_in = len(input_states)
        artifacts = (
            [f"memristor_output_dist_{j}.png" for j in range(1, n_in + 1)]
            + [f"memristor_mode_probs_{j}.png" for j in range(1, n_in + 1)]
            + [
                "memristor_quartic_function.png",
                "memristor_training_loss.png",
            ]
        )
        from src.simulation import sim_logger as _sim_logger

        write_run_summary(
            report_dir,
            metrics={"quartic_test_mse": float(mse)},
            artifacts=artifacts,
            simulation=_sim_logger.stats_dict(),
        )
    
        print(f"\nVisualization complete. Artifacts saved under {report_dir.resolve()}.")
    
    
    finally:
        _end_capture()

if __name__ == "__main__":
    main()
