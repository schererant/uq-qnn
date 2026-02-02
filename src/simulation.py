from __future__ import annotations

import time
from collections import Counter
from typing import Optional, Union, Tuple
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

from .circuits import build_circuit, CircuitType


class SimulationLogger:
    def __init__(self):
        self.call_count = 0
        self.total_time = 0.0
        self.samples_counter = Counter()
        self.circuit_call_count = 0
        self.circuit_total_time = 0.0

    def log(self, elapsed: float, n_samples: int):
        self.call_count += 1
        self.total_time += elapsed
        self.samples_counter[n_samples] += 1

    def log_circuit(self, elapsed: float):
        self.circuit_call_count += 1
        self.circuit_total_time += elapsed

    def report(self):
        print(f"[SimulationLogger] Circuit sequence runs: {self.call_count}")
        print(f"[SimulationLogger] Total sequence time: {self.total_time:.3f} seconds")
        if self.call_count > 0:
            print(f"[SimulationLogger] Avg time per sequence: {self.total_time / self.call_count:.6f} seconds")
        print(f"[SimulationLogger] Sampler sample counts used:")
        for n_samples, freq in self.samples_counter.items():
            print(f"  {n_samples} samples: {freq} times")
        print(f"[SimulationLogger] Individual circuit simulations: {self.circuit_call_count}")
        print(f"[SimulationLogger] Total circuit sim time: {self.circuit_total_time:.3f} seconds")
        if self.circuit_call_count > 0:
            print(f"[SimulationLogger] Avg time per circuit sim: {self.circuit_total_time / self.circuit_call_count:.6f} seconds")


# Global simulation logger instance
sim_logger = SimulationLogger()


def run_simulation_sequence_np(
    params: np.ndarray,
    memory_depth: int,
    n_samples: int,
    encoded_phases: Optional[np.ndarray] = None,
    n_swipe: int = 0,
    swipe_span: float = 0.0,
    circuit_type: CircuitType = CircuitType.MEMRISTOR,
    n_modes: int = 3,
    encoding_mode: int = 0,
    target_mode: Optional[Tuple[int, ...]] = None,
    return_class_probs: bool = False,
) -> np.ndarray:
    """
    Runs a sequence of photonic-circuit simulations in either:
      1) Discrete-phase mode: returns p(001) for each given phase in encoded_phases (each value is a phase in radians).
      2) Continuous-swipe mode: for each X[i] in encoded_phases (each value in [0,1]), computes 2*arccos(X[i]) as the base phase, sweeps n_swipe phases around it, and returns the average p(001).

    Args:
        params (np.ndarray): [phi1, phi3, w].
        memory_depth (int): Depth of the memory buffer.
        n_samples (int): Number of samples for the Sampler.
        encoded_phases (np.ndarray):
            - Discrete mode: array of phase values (radians).
            - Continuous mode: array of X values in [0,1] (will be mapped to phase via 2*arccos(X)).
        n_swipe (int, optional): Number of phase points per X[i] (0 for discrete mode, >0 for continuous mode).
        swipe_span (float, optional): Total phase span for swiping (only used if n_swipe > 0).
        circuit_type (CircuitType): Type of circuit architecture.
        n_modes (int): Number of modes for Clements architecture.
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s). For classification, should have n_classes elements.
        return_class_probs (bool): If True and target_mode has multiple modes, returns (n_data, n_classes) array.

    Returns:
        np.ndarray: 
            - If return_class_probs=False: Predicted probability per input point (shape: (n_data,))
            - If return_class_probs=True: Class probabilities per input point (shape: (n_data, n_classes))
    """
    # validate mode selection
    if encoded_phases is None:
        raise ValueError("encoded_phases must be provided for both modes.")
    if n_swipe < 0:
        raise ValueError("n_swipe must be >= 0.")
        
    # Force discrete mode for Clements architecture
    if circuit_type == CircuitType.CLEMENTS and n_swipe > 0:
        print(f"Warning: Continuous mode (n_swipe={n_swipe}) not supported for Clements architecture. Switching to discrete mode.")
        n_swipe = 0
    
    if n_swipe == 0:
        mode = "discrete"
    elif n_swipe > 0:
        if swipe_span <= 0:
            raise ValueError("swipe_span must be > 0 for continuous mode.")
        if circuit_type != CircuitType.MEMRISTOR:
            raise ValueError("Continuous mode only supported for Memristor architecture")
        mode = "continuous"
    else:
        raise ValueError("Invalid mode selection: n_swipe must be >= 0.")
    # Optionally: print(f"Running in {mode} mode")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")

    start_time = time.perf_counter()
    # Set up circuit parameters based on circuit type
    if circuit_type == CircuitType.MEMRISTOR:
        # Memristor architecture: params = [phi1, phi3, w]
        # First two are phases, third is weight
        if len(params) != 3:
            raise ValueError(f"Memristor architecture requires 3 parameters, got {len(params)}")
        phi1, phi3, w = params[0], params[1], params[2]
        
        # Fixed input and output states for memristor
        input_state = pcvl.BasicState([0, 1, 0])
        state_001 = pcvl.BasicState([0, 0, 1])
        state_010 = pcvl.BasicState([0, 1, 0])
        
        # Default target mode for memristor if not specified
        if target_mode is None:
            target_mode = (2,)  # Mode 2 (index 2, state 001)
    else:
        # Clements architecture: params are all phases for MZIs
        # Initialize weight as the last parameter
        w = params[-1]
        
        # Create input state with a single photon in the encoding_mode
        input_modes = [0] * n_modes
        input_modes[encoding_mode] = 1
        input_state = pcvl.BasicState(input_modes)
        
        # Set up output states to track
        if target_mode is None:
            # Default to last mode if not specified
            target_mode = (n_modes - 1,)

    # prepare memory and output
    mem_p1 = np.zeros(memory_depth, dtype=float)
    mem_p2 = np.zeros(memory_depth, dtype=float)
    num_pts = len(encoded_phases)
    
    # Determine if we need multi-class output
    n_classes = len(target_mode) if target_mode is not None else 1
    if return_class_probs and n_classes > 1:
        preds = np.zeros((num_pts, n_classes), dtype=float)
    else:
        preds = np.zeros(num_pts, dtype=float)

    # Precompute base phases and offsets for continuous mode
    if mode == "continuous":
        enc_base = encoded_phases
        #TODO: Use Iris data for that
        offsets = np.linspace(
            -swipe_span / 2, swipe_span / 2, n_swipe, dtype=encoded_phases.dtype
        )
    else:
        # Initialize offsets as empty array for discrete mode to avoid reference errors
        offsets = np.array([])
        enc_base = encoded_phases

    # Create empty lists for target states if using Clements architecture
    if circuit_type == CircuitType.CLEMENTS:
        target_modes_list = []
        for mode in target_mode:
            target_mode_array = [0] * n_modes
            target_mode_array[mode] = 1
            target_modes_list.append(pcvl.BasicState(target_mode_array))
    
    # main loop
    for i in range(num_pts):
        t = i % memory_depth
        # compute memory-driven φₘ
        if i == 0:
            mem_phi = np.pi / 4
        else:
            m1 = mem_p1.mean()
            m2 = mem_p2.mean()
            arg = np.clip(m1 + w * m2, 1e-9, 1 - 1e-9)
            mem_phi = np.arccos(np.sqrt(arg))

        if mode == "discrete":
            # single-φ mode
            enc_phi = encoded_phases[i]
            # Build phases array for circuit based on architecture
            if circuit_type == CircuitType.MEMRISTOR:
                phases = np.array([phi1, mem_phi, phi3])
            else:
                # For Clements, use all original phases but replace the memory phase
                # at specific positions based on the encoding scheme
                phases = params[:-1].copy()  # All but the weight parameter
                
                # For simplicity, replace the first internal phase with memory phase
                # This can be made more sophisticated with a specific memory scheme
                phases[0] = mem_phi
                
                # Make sure phases array has the correct length for Clements circuit
                expected_phases = n_modes * (n_modes - 1)
                if len(phases) != expected_phases:
                    raise ValueError(f"Expected {expected_phases} phases for Clements with {n_modes} modes, got {len(phases)}")
                
            circ = build_circuit(phases, enc_phi, circuit_type=circuit_type, 
                                n_modes=n_modes, encoding_mode=encoding_mode)
            proc = pcvl.Processor("SLOS", circ)
            proc.with_input(input_state)
            t0 = time.perf_counter()
            probs = Sampler(proc).probs(n_samples)["results"]
            sim_logger.log_circuit(time.perf_counter() - t0)

            if circuit_type == CircuitType.MEMRISTOR:
                p001 = probs.get(state_001, 0.0)
                p010 = probs.get(state_010, 0.0)
                if return_class_probs and n_classes > 1:
                    # For classification, use multiple output modes
                    # For memristor with 2 classes, use p001 and p010
                    if n_classes == 2:
                        preds[i, 0] = p010  # Class 0 from mode 1
                        preds[i, 1] = p001  # Class 1 from mode 2
                    else:
                        # For more classes, need more modes - use available probabilities
                        for c in range(min(n_classes, 2)):
                            if c == 0:
                                preds[i, c] = p010
                            elif c == 1:
                                preds[i, c] = p001
                            else:
                                preds[i, c] = 0.0  # Not enough modes
                else:
                    preds[i] = p001
                mem_p1[t], mem_p2[t] = p010, p001
            else:
                # For Clements, handle multi-class output
                if return_class_probs and n_classes > 1:
                    # Return probabilities for each target mode (class)
                    for c, target_state in enumerate(target_modes_list):
                        preds[i, c] = probs.get(target_state, 0.0)
                    # Use average for memory (can be improved)
                    avg_prob = preds[i].mean()
                    mem_p1[t], mem_p2[t] = avg_prob, avg_prob
                else:
                    # For Clements, average probabilities of all target modes
                    target_prob = 0.0
                    for target_state in target_modes_list:
                        target_prob += probs.get(target_state, 0.0)
                    
                    # If multiple targets, normalize by number of targets
                    if len(target_modes_list) > 1:
                        target_prob /= len(target_modes_list)
                    
                    # For Clements, use the same probability for both memory values
                    # This is a simple approach - more complex memory schemes can be implemented
                    preds[i] = target_prob
                    mem_p1[t], mem_p2[t] = target_prob, target_prob

        else:
            # swipe mode: average over offsets
            # We've already validated that we're in memristor mode above
            p1_swipe = np.empty(n_swipe, dtype=float)
            p2_swipe = np.empty(n_swipe, dtype=float)
            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                # Build phases array for circuit based on architecture
                if circuit_type == CircuitType.MEMRISTOR:
                    phases = np.array([phi1, mem_phi, phi3])
                else:
                    # For Clements, use all original phases but replace the memory phase
                    phases = params[:-1].copy()  # All but the weight parameter
                    phases[0] = mem_phi  # Replace first phase with memory phase
                    
                    # Make sure phases array has the correct length for Clements circuit
                    expected_phases = n_modes * (n_modes - 1)
                    if len(phases) != expected_phases:
                        raise ValueError(f"Expected {expected_phases} phases for Clements with {n_modes} modes, got {len(phases)}")
                    
                circ = build_circuit(phases, enc_phi, circuit_type=circuit_type, 
                                    n_modes=n_modes, encoding_mode=encoding_mode)
                proc = pcvl.Processor("SLOS", circ)
                proc.with_input(input_state)
                t0 = time.perf_counter()
                probs = Sampler(proc).probs(n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)

                if circuit_type == CircuitType.MEMRISTOR:
                    p1_swipe[k] = probs.get(state_010, 0.0)
                    p2_swipe[k] = probs.get(state_001, 0.0)
                else:
                    # For Clements, use target modes
                    target_prob = 0.0
                    for target_state in target_modes_list:
                        target_prob += probs.get(target_state, 0.0)
                    
                    if len(target_modes_list) > 1:
                        target_prob /= len(target_modes_list)
                    
                    p1_swipe[k] = target_prob
                    p2_swipe[k] = target_prob

            if return_class_probs and n_classes > 1:
                # For classification in swipe mode, average probabilities across swipes
                if circuit_type == CircuitType.MEMRISTOR and n_classes == 2:
                    preds[i, 0] = p1_swipe.mean()  # Class 0
                    preds[i, 1] = p2_swipe.mean()  # Class 1
                else:
                    # For Clements or more classes, need to recompute
                    # This is a simplified version - full implementation would track per class
                    preds[i] = p2_swipe.mean() if n_classes == 1 else p2_swipe.mean()
            else:
                preds[i] = p2_swipe.mean()
            mem_p1[t], mem_p2[t] = p1_swipe.mean(), p2_swipe.mean()

    # finalize
    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, n_samples)
    return preds