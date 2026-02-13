#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clements Circuit Plotter Example

This script creates a Clements circuit with configurable number of modes
and visualizes it using Perceval's plotting capabilities.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.colors as mcolors

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import perceval as pcvl
from src.circuits import clements_circuit, build_circuit

def create_clements_circuit(n_modes=6):
    """Creates a Clements circuit with the specified number of modes."""
    # Calculate required number of phases for the Clements circuit
    n_phases = n_modes * (n_modes - 1)

    # Initialize random phases with fixed seed for reproducibility
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n_phases)

    # Create the Clements circuit
    circuit = clements_circuit(phases, n_modes)

    print(f"Created Clements circuit with {n_modes} modes")
    print(f"Number of phases: {n_phases}")

    return circuit, phases

def create_full_circuit(n_modes=6, encoding_mode=0):
    """Creates a full circuit with encoding and Clements components."""
    # Calculate required number of phases for the Clements circuit
    n_phases = n_modes * (n_modes - 1)

    # Initialize random phases with fixed seed for reproducibility
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n_phases)
    enc_phi = np.pi/4  # Example encoding phase

    # Create the full circuit (Clements, no memristive phases)
    circuit = build_circuit(
        phases=phases,
        enc_phi=enc_phi,
        n_modes=n_modes,
        encoding_mode=encoding_mode
    )

    return circuit, phases, enc_phi


def plot_circuit(circuit, title):
    """
    Plot the circuit using custom visualization and Perceval capabilities.
    """
    print(f"Plotting {title}...")

    # First, create our custom visualization
    pcvl.pdisplay(circuit)

    # # Also try to use Perceval's native visualization if available
    # try:
    #     # Create a processor for the circuit
    #     processor = pcvl.Processor("SLOS", circuit)

    #     # Create figure for Perceval's visualization
    #     plt.figure(figsize=(14, 7))

    #     # Try to use Perceval's drawing capabilities
    #     if hasattr(circuit, 'draw'):
    #         circuit.draw()
    #         perceval_filename = f"{title.lower().replace(' ', '_')}_perceval.png"
    #         plt.savefig(perceval_filename, dpi=300, bbox_inches='tight')
    #         plt.close()
    #         print(f"Perceval circuit plot saved as {perceval_filename}")
    #     elif hasattr(pcvl, 'pdisplay'):
    #         # This works better in Jupyter notebooks but we'll try
    #         pcvl.pdisplay(circuit)
    #         perceval_filename = f"{title.lower().replace(' ', '_')}_perceval.png"
    #         plt.savefig(perceval_filename, dpi=300, bbox_inches='tight')
    #         plt.close()
    #         print(f"Perceval circuit plot saved as {perceval_filename}")

    # except Exception as e:
    #     print(f"Note: Could not use Perceval's native visualization: {e}")

# def run_and_plot_circuit(circuit, input_state, title):
#     """Run the circuit with the given input and plot the output distribution."""
#     # Create a processor for the circuit
#     processor = pcvl.Processor("SLOS", circuit)
#     processor.with_input(input_state)

#     # Run the simulation
#     sampler = pcvl.algorithm.Sampler(processor)
#     results = sampler.probs(10000)["results"]

#     # Plot the output distribution
#     plt.figure(figsize=(12, 6))

#     # Convert results to arrays for plotting
#     states = []
#     probs = []

#     # Sort by probability and take top 15 states
#     for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True)[:15]:
#         states.append(str(state))
#         probs.append(prob)

#     # Create the bar chart
#     plt.bar(range(len(states)), probs, color='skyblue')
#     plt.xticks(range(len(states)), states, rotation=45)
#     plt.xlabel('Output State')
#     plt.ylabel('Probability')
#     plt.title(f'{title} - Output Distribution')

#     # Add probability values on top of bars
#     for i, v in enumerate(probs):
#         plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

#     plt.tight_layout()

#     # Save the figure
#     filename = f"{title.lower().replace(' ', '_')}_distribution.png"
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Distribution plot saved as {filename}")

#     return results

# def analyze_circuit(circuit, input_state, title):
#     """Analyze and print information about the circuit."""
#     print(f"\n=== {title} Analysis ===")
#     print(f"Number of modes: {circuit.m}")
#     print(f"Input state: {input_state}")

    # # Create processor and run simulation
    # processor = pcvl.Processor("SLOS", circuit)
    # processor.with_input(input_state)
    # sampler = pcvl.algorithm.Sampler(processor)
    # results = sampler.probs(10000)["results"]

    # # Print top output states
    # print("\nTop 5 output states:")
    # for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]:
    #     print(f"  {state}: {prob:.6f}")

    # # Analyze mode probabilities
    # print("\nMode occupation probabilities:")
    # mode_probs = [0] * circuit.m
    # for state, prob in results.items():
    #     state_str = str(state)
    #     photon_counts = state_str.strip('|>').split(',')
    #     for mode, count in enumerate(photon_counts):
    #         if int(count) > 0:
    #             mode_probs[mode] += prob

    # for mode, prob in enumerate(mode_probs):
    #     print(f"  Mode {mode}: {prob:.6f}")

    # # Plot mode probabilities
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(circuit.m), mode_probs, color='coral')
    # plt.xlabel('Mode')
    # plt.ylabel('Probability')
    # plt.title(f'{title} - Mode Occupation Probabilities')
    # plt.xticks(range(circuit.m))

    # # Add probability values on top of bars
    # for i, v in enumerate(mode_probs):
    #     plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    # plt.tight_layout()

    # # Save the figure
    # filename = f"{title.lower().replace(' ', '_')}_mode_probs.png"
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"Mode probabilities plot saved as {filename}")

def main():
    """Main function to create and plot Clements circuits."""
    print("=== Clements Circuit Plotter ===")

    # Create and plot Clements circuits with different numbers of modes
    for n_modes in [3, 6]:
        print(f"\nCreating {n_modes}-mode Clements circuit...")

        # Create the standalone Clements circuit
        circuit, phases = create_clements_circuit(n_modes)

        # Print phase information
        print(f"\nPhase parameters for {n_modes}x{n_modes} Clements circuit:")
        for i, phase in enumerate(phases):
            print(f"  Phase {i}: {phase:.4f} radians ({phase * 180/np.pi:.1f}°)")

        # Create input state with a single photon in the first mode
        input_modes = [0] * n_modes
        input_modes[0] = 1
        input_state = pcvl.BasicState(input_modes)

        # Plot the circuit
        # plot_circuit(circuit, f"Clements Circuit {n_modes}x{n_modes}")
        pcvl.pdisplay(circuit)

        # # Run the circuit and plot the output distribution
        # run_and_plot_circuit(circuit, input_state, f"Clements {n_modes}x{n_modes}")

        # # Analyze the circuit
        # analyze_circuit(circuit, input_state, f"Clements {n_modes}x{n_modes}")

        # Now create and plot the full circuit with encoding
        full_circuit, _, enc_phi = create_full_circuit(n_modes)
        print(f"\nEncoding phase: {enc_phi:.4f} radians ({enc_phi * 180/np.pi:.1f}°)")

        # plot_circuit(full_circuit, f"Full Clements Circuit {n_modes}x{n_modes}")
        # run_and_plot_circuit(full_circuit, input_state, f"Full Clements {n_modes}x{n_modes}")
        # analyze_circuit(full_circuit, input_state, f"Full Clements {n_modes}x{n_modes}")

    print("\nCircuit plotting complete. Output images saved to current directory.")
    print("The images show:")
    print("1. Custom visualizations of the circuit structure")
    print("2. Output state distributions")
    print("3. Mode occupation probabilities")

if __name__ == "__main__":
    main()
