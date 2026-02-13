from __future__ import annotations

import time
from collections import Counter
from typing import Optional, Union, Tuple, Sequence
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

from .circuits import build_circuit, CircuitType, get_mzi_modes_for_phase


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


def _default_memristive_phase_idx(n_modes: int) -> Tuple[int]:
    """Default memristive phase index: middle MZI internal phase. For 3-mode, returns (2,) (matches original)."""
    n_phases = n_modes * (n_modes - 1)
    if n_modes == 3:
        return (2,)  # (1,2) MZI internal phase - matches original memristor
    return (n_phases // 2 - 1,)  # Middle of circuit


def _normalize_memristive_phase_idx(
    memristive_phase_idx: Optional[Union[int, Sequence[int]]],
    n_modes: int,
    n_phases: int,
) -> Tuple[int, ...]:
    """Normalize memristive_phase_idx to a tuple of phase indices. Returns empty tuple for non-memristor."""
    if memristive_phase_idx is None:
        return _default_memristive_phase_idx(n_modes)
    if isinstance(memristive_phase_idx, int):
        idx = memristive_phase_idx
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"memristive_phase_idx must be in [0, {n_phases-1}] for {n_modes} modes, got {idx}"
            )
        return (idx,)
    # Sequence (tuple, list, etc.)
    indices = tuple(int(x) for x in memristive_phase_idx)
    for idx in indices:
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"Each memristive_phase_idx must be in [0, {n_phases-1}] for {n_modes} modes, got {idx}"
            )
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"memristive_phase_idx must not contain duplicates, got {memristive_phase_idx}"
        )
    return indices


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
    memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None,
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
        n_modes (int): Number of modes for Clements/Memristor architecture.
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s). For classification, should have n_classes elements.
        return_class_probs (bool): If True and target_mode has multiple modes, returns (n_data, n_classes) array.
        memristive_phase_idx (Optional[int]): For MEMRISTOR, which phase index (0 to n_phases-1) is replaced by
            memory-driven phase. If None, uses default (middle MZI internal phase; for 3-mode = 2).

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
        # Memristor: same structure as Clements, params = [phase_0, ..., phase_{n-1}, w]
        n_phases = n_modes * (n_modes - 1)
        if len(params) != n_phases + 1:
            raise ValueError(
                f"Memristor architecture requires {n_phases + 1} parameters "
                f"({n_phases} phases + weight) for {n_modes} modes, got {len(params)}"
            )
        w = params[-1]
        if memristive_phase_idx is None:
            memristive_phase_idx = _default_memristive_phase_idx(n_modes)
        if memristive_phase_idx < 0 or memristive_phase_idx >= n_phases:
            raise ValueError(
                f"memristive_phase_idx must be in [0, {n_phases-1}] for {n_modes} modes, "
                f"got {memristive_phase_idx}"
            )
        
        # Input state: single photon in encoding_mode
        input_modes = [0] * n_modes
        input_modes[encoding_mode] = 1
        input_state = pcvl.BasicState(input_modes)
        
        # Memory states: probabilities of photon in the two modes of the memristive MZI
        m1, m2 = get_mzi_modes_for_phase(memristive_phase_idx, n_modes)
        state_m1 = [0] * n_modes
        state_m1[m1] = 1
        state_m2 = [0] * n_modes
        state_m2[m2] = 1
        state_m1_bs = pcvl.BasicState(state_m1)
        state_m2_bs = pcvl.BasicState(state_m2)
        
        # Default target mode if not specified
        if target_mode is None:
            target_mode = (n_modes - 1,)  # Last mode
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

    # Build target states list for multi-class / probability extraction
    target_modes_list = []
    for m in target_mode:
        tm = [0] * n_modes
        tm[m] = 1
        target_modes_list.append(pcvl.BasicState(tm))

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
                phases = params[:-1].copy()
                phases[memristive_phase_idx] = mem_phi
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
                p_m1 = probs.get(state_m1_bs, 0.0)
                p_m2 = probs.get(state_m2_bs, 0.0)
                if return_class_probs and n_classes > 1:
                    for c, target_state in enumerate(target_modes_list):
                        preds[i, c] = probs.get(target_state, 0.0)
                else:
                    target_prob = sum(probs.get(ts, 0.0) for ts in target_modes_list)
                    if len(target_modes_list) > 1:
                        target_prob /= len(target_modes_list)
                    preds[i] = target_prob
                mem_p1[t], mem_p2[t] = p_m1, p_m2
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
            target_swipe = np.empty((n_swipe, n_classes) if return_class_probs and n_classes > 1 else (n_swipe,), dtype=float)
            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                # Build phases array for circuit based on architecture
                if circuit_type == CircuitType.MEMRISTOR:
                    phases = params[:-1].copy()
                    phases[memristive_phase_idx] = mem_phi
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
                    p1_swipe[k] = probs.get(state_m1_bs, 0.0)
                    p2_swipe[k] = probs.get(state_m2_bs, 0.0)
                    if return_class_probs and n_classes > 1:
                        for c, ts in enumerate(target_modes_list):
                            target_swipe[k, c] = probs.get(ts, 0.0)
                    else:
                        target_swipe[k] = sum(probs.get(ts, 0.0) for ts in target_modes_list)
                        if len(target_modes_list) > 1:
                            target_swipe[k] /= len(target_modes_list)
                else:
                    # For Clements, use target modes
                    target_prob = 0.0
                    for target_state in target_modes_list:
                        target_prob += probs.get(target_state, 0.0)
                    
                    if len(target_modes_list) > 1:
                        target_prob /= len(target_modes_list)
                    
                    p1_swipe[k] = target_prob
                    p2_swipe[k] = target_prob
                    target_swipe[k] = target_prob

            if return_class_probs and n_classes > 1:
                preds[i] = target_swipe.mean(axis=0)
            else:
                preds[i] = target_swipe.mean()
            mem_p1[t], mem_p2[t] = p1_swipe.mean(), p2_swipe.mean()

    # finalize
    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, n_samples)
    return preds