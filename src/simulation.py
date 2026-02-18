from __future__ import annotations

import time
from collections import Counter
from typing import Optional, Union, Tuple, Sequence
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

from .circuits import build_circuit, get_mzi_modes_for_phase


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


def _normalize_memristive_output_modes(
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]],
    memristive_indices: Tuple[int, ...],
    n_modes: int,
) -> Tuple[Tuple[int, int], ...]:
    """
    Normalize memristive_output_modes to a tuple of (mode_p1, mode_p2) per memristive index.
    When None, uses get_mzi_modes_for_phase for each index (default: MZI output modes).
    """
    if memristive_output_modes is None:
        return tuple(get_mzi_modes_for_phase(idx, n_modes) for idx in memristive_indices)
    modes = tuple(
        (int(m1), int(m2)) for m1, m2 in memristive_output_modes
    )
    if len(modes) != len(memristive_indices):
        raise ValueError(
            f"memristive_output_modes must have {len(memristive_indices)} entries "
            f"(one per memristive phase), got {len(modes)}"
        )
    for j, (m1, m2) in enumerate(modes):
        if m1 < 0 or m1 >= n_modes or m2 < 0 or m2 >= n_modes:
            raise ValueError(
                f"memristive_output_modes[{j}] = ({m1}, {m2}): modes must be in [0, {n_modes-1}]"
            )
        if m1 == m2:
            raise ValueError(
                f"memristive_output_modes[{j}] = ({m1}, {m2}): the two modes must differ"
            )
    return modes


def _normalize_memristive_phase_idx(
    memristive_phase_idx: Optional[Union[int, Sequence[int]]],
    n_modes: int,
    n_phases: int,
) -> Tuple[int, ...]:
    """
    Normalize memristive_phase_idx to a tuple of phase indices.
    Returns empty tuple when None or empty - no memristive behavior.
    """
    if memristive_phase_idx is None:
        return ()
    if isinstance(memristive_phase_idx, int):
        idx = memristive_phase_idx
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"memristive_phase_idx must be in [0, {n_phases-1}] for {n_modes} modes, got {idx}"
            )
        return (idx,)
    # Sequence (tuple, list, etc.)
    indices = tuple(int(x) for x in memristive_phase_idx)
    if len(indices) == 0:
        return ()
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
    encoded_phases: np.ndarray,
    n_swipe: int = 0,
    swipe_span: float = 0.0,
    n_modes: int = 3,
    encoding_mode: int = 0,
    target_mode: Optional[Tuple[int, ...]] = None,
    return_class_probs: bool = False,
    memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None,
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Runs a sequence of photonic-circuit simulations. Architecture is always Clements (3x3, 6x6, etc.).

    Args:
        params (np.ndarray): Phase parameters. If memristive_phase_idx is None/empty: [phase_0, ..., phase_{n-1}].
            If memristive: [phase_0, ..., phase_{n-1}, w_0, ..., w_{k-1}] for k memristive phases.
        memory_depth (int): Depth of the memory buffer (only used when memristive).
        n_samples (int): Number of samples for the Sampler.
        encoded_phases (np.ndarray): Phase values (radians) for each data point.
        n_swipe (int): Phase points per data point (0=discrete, >0=continuous; only when memristive).
        swipe_span (float): Phase span for swiping (only when n_swipe > 0).
        n_modes (int): Number of modes (3 for 3x3, 6 for 6x6, etc.).
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s).
        return_class_probs (bool): If True and multiple targets, returns (n_data, n_classes).
        memristive_phase_idx (Optional[Union[int, Sequence[int]]]): Phase indices to make memristive.
            None or empty = no memristive behavior. e.g. [2] or (2, 5) for one or two MZIs.
        memristive_output_modes (Optional[Sequence[Tuple[int, int]]]): For each memristive phase index,
            the (mode_p1, mode_p2) output modes to use for feedback. When None, uses the MZI's
            own output modes. e.g. [(1, 2), (3, 4)] for two memristive phases.

    Returns:
        np.ndarray: Predicted probability per input point, or class probabilities if return_class_probs.
    """
    start_time = time.perf_counter()
    if n_swipe < 0:
        raise ValueError("n_swipe must be >= 0.")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")

    n_phases = n_modes * (n_modes - 1)
    memristive_indices = _normalize_memristive_phase_idx(memristive_phase_idx, n_modes, n_phases)
    n_memristive = len(memristive_indices)
    output_modes = _normalize_memristive_output_modes(
        memristive_output_modes, memristive_indices, n_modes
    ) if n_memristive > 0 else ()

    # Continuous mode only when memristive is active
    if n_swipe > 0 and n_memristive == 0:
        print("Warning: Continuous mode requires memristive phases. Switching to discrete.")
        n_swipe = 0
    if n_swipe > 0 and swipe_span <= 0:
        raise ValueError("swipe_span must be > 0 for continuous mode.")

    mode = "continuous" if n_swipe > 0 else "discrete"
    expected_params = n_phases + n_memristive
    if len(params) != expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters ({n_phases} phases"
            + (f" + {n_memristive} weights" if n_memristive else "")
            + f") for {n_modes} modes, got {len(params)}"
        )

    weights = params[-n_memristive:] if n_memristive else None

    input_modes = [0] * n_modes
    input_modes[encoding_mode] = 1
    input_state = pcvl.BasicState(input_modes)

    if target_mode is None:
        target_mode = (n_modes - 1,)

    state_m1_list = []
    state_m2_list = []
    for j in range(n_memristive):
        m1, m2 = output_modes[j]
        s1, s2 = [0] * n_modes, [0] * n_modes
        s1[m1], s2[m2] = 1, 1
        state_m1_list.append(pcvl.BasicState(s1))
        state_m2_list.append(pcvl.BasicState(s2))

    # Build target states list for multi-class / probability extraction
    target_modes_list = []
    for m in target_mode:
        tm = [0] * n_modes
        tm[m] = 1
        target_modes_list.append(pcvl.BasicState(tm))

    if n_memristive > 0:
        mem_p1 = np.zeros((memory_depth, n_memristive), dtype=float)
        mem_p2 = np.zeros((memory_depth, n_memristive), dtype=float)
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
        if n_memristive > 0:
            mem_phis = np.empty(n_memristive, dtype=float)
            for j in range(n_memristive):
                if i == 0:
                    mem_phis[j] = np.pi / 4
                else:
                    m1 = mem_p1[:, j].mean()
                    m2 = mem_p2[:, j].mean()
                    arg = np.clip(m1 + weights[j] * m2, 1e-9, 1 - 1e-9)
                    mem_phis[j] = np.arccos(np.sqrt(arg))

        if mode == "discrete":
            enc_phi = encoded_phases[i]
            if n_memristive > 0:
                phases = params[:-n_memristive].copy()
                for j, idx in enumerate(memristive_indices):
                    phases[idx] = mem_phis[j]
            else:
                phases = params.copy()

            circ = build_circuit(phases, enc_phi, n_modes=n_modes, encoding_mode=encoding_mode)
            proc = pcvl.Processor("SLOS", circ)
            proc.with_input(input_state)
            t0 = time.perf_counter()
            probs = Sampler(proc).probs(n_samples)["results"]
            sim_logger.log_circuit(time.perf_counter() - t0)

            if n_memristive > 0:
                for j in range(n_memristive):
                    mem_p1[t, j] = probs.get(state_m1_list[j], 0.0)
                    mem_p2[t, j] = probs.get(state_m2_list[j], 0.0)
            if return_class_probs and n_classes > 1:
                for c, target_state in enumerate(target_modes_list):
                    preds[i, c] = probs.get(target_state, 0.0)
            else:
                target_prob = sum(probs.get(ts, 0.0) for ts in target_modes_list)
                if len(target_modes_list) > 1:
                    target_prob /= len(target_modes_list)
                preds[i] = target_prob

        else:
            # swipe mode (only when memristive)
            p1_swipe = np.empty((n_swipe, n_memristive), dtype=float)
            p2_swipe = np.empty((n_swipe, n_memristive), dtype=float)
            target_swipe = np.empty((n_swipe, n_classes) if return_class_probs and n_classes > 1 else (n_swipe,), dtype=float)
            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                phases = params[:-n_memristive].copy()
                for j, idx in enumerate(memristive_indices):
                    phases[idx] = mem_phis[j]
                circ = build_circuit(phases, enc_phi, n_modes=n_modes, encoding_mode=encoding_mode)
                proc = pcvl.Processor("SLOS", circ)
                proc.with_input(input_state)
                t0 = time.perf_counter()
                probs = Sampler(proc).probs(n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)
                for j in range(n_memristive):
                    p1_swipe[k, j] = probs.get(state_m1_list[j], 0.0)
                    p2_swipe[k, j] = probs.get(state_m2_list[j], 0.0)
                if return_class_probs and n_classes > 1:
                    for c, ts in enumerate(target_modes_list):
                        target_swipe[k, c] = probs.get(ts, 0.0)
                else:
                    target_swipe[k] = sum(probs.get(ts, 0.0) for ts in target_modes_list)
                    if len(target_modes_list) > 1:
                        target_swipe[k] /= len(target_modes_list)
            if return_class_probs and n_classes > 1:
                preds[i] = target_swipe.mean(axis=0)
            else:
                preds[i] = target_swipe.mean()
            for j in range(n_memristive):
                mem_p1[t, j], mem_p2[t, j] = p1_swipe[:, j].mean(), p2_swipe[:, j].mean()

    # finalize
    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, n_samples)
    return preds