#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Circuit Printer Example

This script creates memristor and Clements circuits and prints them out in detail,
showing their structure, components, and operation.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import perceval as pcvl
from src.circuits import clements_circuit, build_circuit, encoding_circuit

def print_circuit_matrix(circuit):
    """Print the unitary matrix of the circuit."""
    try:
        # Get the unitary matrix from the circuit (Processor doesn't have get_unitary)
        matrix = circuit.compute_unitary()
        if hasattr(matrix, 'real') and hasattr(matrix, 'imag'):
            matrix = np.array(matrix)
        
        print("\nCircuit Unitary Matrix:")
        
        # Format and print the matrix
        for i in range(matrix.shape[0]):
            row = ""
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                real = np.real(val)
                imag = np.imag(val)
                
                if abs(imag) < 1e-10:  # Effectively zero imaginary part
                    row += f" {real:6.3f}        "
                else:
                    sign = "+" if imag >= 0 else ""
                    row += f" {real:6.3f}{sign}{imag:6.3f}i "
            print(f"  [{row} ]")
    except Exception as e:
        print(f"\nCouldn't print matrix: {str(e)}")

def print_circuit_details(circuit, input_state, measurement_mode, title):
    """Print detailed information about a circuit."""
    print(f"\n{'='*80}")
    print(f"=== {title} ===")
    print(f"{'='*80}")
    
    # Basic information
    print(f"Circuit name: {circuit.name}")
    print(f"Number of modes: {circuit.m}")
    print(f"Input state: {input_state}")
    print(f"Measurement mode(s): {measurement_mode}")
    
    # Print circuit structure using Perceval's built-in visualization
    print("\nCircuit Structure:")
    pcvl.pdisplay(circuit, output_format=pcvl.Format.TEXT, recursive=True)
    
    # Print unitary matrix
    if circuit.m <= 6:  # Only print matrix for smaller circuits
        print_circuit_matrix(circuit)
    else:
        print("\nUnitary matrix too large to display")
    
    # Simulation with the given input
    print("\nSimulation Results:")
    processor = pcvl.Processor("SLOS", circuit)
    processor.with_input(input_state)
    
    # Run simulation
    sampler = pcvl.algorithm.Sampler(processor)
    results = sampler.probs(10000)["results"]
    
    # Print output states and probabilities
    print("Output state probabilities:")
    for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {state}: {prob:.6f}")
    
    # Analyze measurement modes
    print(f"\nProbability of detecting photon in measurement mode(s) {measurement_mode}:")
    for mode in measurement_mode:
        total_prob = 0
        for state, prob in results.items():
            state_str = str(state)
            if '|' in state_str and '>' in state_str:
                photon_counts = state_str.strip('|>').split(',')
                if len(photon_counts) > mode and int(photon_counts[mode]) > 0:
                    total_prob += prob
        print(f"  Mode {mode}: {total_prob:.6f}")
    
    if "clements" in circuit.name.lower():
        print("\nClements Operation:")
        # FockState uses photon2mode(0) for single-photon input mode, not .index()
        input_mode = input_state.photon2mode(0) if input_state.n > 0 else 0
        print(f"  1. Input: Photon enters mode {input_mode} of {circuit.m} modes")
        print(f"  2. Mesh of {circuit.m * (circuit.m - 1) // 2} MZIs in rectangular grid pattern")
        print(f"  3. Each MZI has two phase shifters (internal and external)")
        print(f"  4. Output: Measure probability of photon in mode {measurement_mode[0]}")

def create_clements_with_memristive(n_modes=3, memristive_phase_idx=None):
    """Create Clements circuit, optionally with memristive phases."""
    n_phases = n_modes * (n_modes - 1)
    phases = np.random.uniform(0, 2*np.pi, n_phases)
    enc_phi = np.pi/4

    basic = clements_circuit(phases, n_modes)
    full = build_circuit(phases, enc_phi, n_modes=n_modes, encoding_mode=0)
    
    # Input state: single photon in mode 0
    input_modes = [0] * n_modes
    input_modes[0] = 1
    input_state = pcvl.BasicState(input_modes)
    
    # Measurement mode: last mode
    measurement_mode = (n_modes - 1,)
    
    print_circuit_details(basic, input_state, measurement_mode, f"Clements {n_modes}x{n_modes} (standalone)")
    print_circuit_details(full, input_state, measurement_mode, f"Clements {n_modes}x{n_modes} with Encoding")
    return full

def create_clements_circuit(n_modes=6):
    """Create and print a Clements circuit."""
    # Calculate required phases
    n_phases = n_modes * (n_modes - 1)
    
    # Initialize random phases with fixed seed for reproducibility
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n_phases)
    enc_phi = np.pi/4
    
    # Create basic Clements circuit
    basic_clements = clements_circuit(phases, n_modes)
    
    full_clements = build_circuit(phases, enc_phi, n_modes=n_modes, encoding_mode=0)
    
    # Input state: |100000>
    input_modes = [0] * n_modes
    input_modes[0] = 1
    input_state = pcvl.BasicState(input_modes)
    
    # Measurement mode: last mode
    measurement_mode = (n_modes - 1,)
    
    # Print circuit details
    print_circuit_details(basic_clements, input_state, measurement_mode, f"Clements Circuit {n_modes}x{n_modes} (standalone)")
    print_circuit_details(full_clements, input_state, measurement_mode, f"Clements Circuit {n_modes}x{n_modes} with Encoding")
    
    return full_clements

def main():
    """Main function to create and print circuits."""
    print("=== Circuit Printer Example ===")
    
    # 1. Create and print 3-mode Clements circuit
    create_clements_with_memristive(n_modes=3)
    
    # 2. Create and print 6-mode Clements circuit
    clements_circuit_6 = create_clements_circuit(n_modes=6)
    
    # 3. Create and print 3-mode Clements circuit for comparison
    clements_circuit_3 = create_clements_circuit(n_modes=3)
    
    print("\nCircuit printing complete.")

if __name__ == "__main__":
    main()