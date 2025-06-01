from __future__ import annotations

import time
from collections import Counter
from typing import Optional
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

from .circuits import build_circuit


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

    Returns:
        np.ndarray: Predicted p(001) per input point or per phase.
    """
    # validate mode selection
    if encoded_phases is None:
        raise ValueError("encoded_phases must be provided for both modes.")
    if n_swipe < 0:
        raise ValueError("n_swipe must be >= 0.")
    if n_swipe == 0:
        mode = "discrete"
    elif n_swipe > 0:
        if swipe_span <= 0:
            raise ValueError("swipe_span must be > 0 for continuous mode.")
        mode = "continuous"
    else:
        raise ValueError("Invalid mode selection: n_swipe must be >= 0.")
    # Optionally: print(f"Running in {mode} mode")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")

    start_time = time.perf_counter()
    phi1, phi3, w = params
    input_state = pcvl.BasicState([0, 1, 0])
    state_001 = pcvl.BasicState([0, 0, 1])
    state_010 = pcvl.BasicState([0, 1, 0])

    # prepare memory and output
    mem_p1 = np.zeros(memory_depth, dtype=float)
    mem_p2 = np.zeros(memory_depth, dtype=float)
    num_pts = len(encoded_phases)
    preds = np.zeros(num_pts, dtype=float)

    if mode == "continuous":
        # precompute base phases and offsets
        enc_base = encoded_phases
        #TODO: Use Iris data for that
        offsets = np.linspace(
            -swipe_span / 2, swipe_span / 2, n_swipe, dtype=enc_base.dtype
        )

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
            circ = build_circuit(phi1, mem_phi, phi3, enc_phi)
            proc = pcvl.Processor("SLOS", circ)
            proc.with_input(input_state)
            t0 = time.perf_counter()
            probs = Sampler(proc).probs(n_samples)["results"]
            sim_logger.log_circuit(time.perf_counter() - t0)

            p001 = probs.get(state_001, 0.0)
            p010 = probs.get(state_010, 0.0)
            preds[i] = p001
            mem_p1[t], mem_p2[t] = p010, p001

        else:
            # swipe mode: average over offsets
            p1_swipe = np.empty(n_swipe, dtype=float)
            p2_swipe = np.empty(n_swipe, dtype=float)
            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                circ = build_circuit(phi1, mem_phi, phi3, enc_phi)
                proc = pcvl.Processor("SLOS", circ)
                proc.with_input(input_state)
                t0 = time.perf_counter()
                probs = Sampler(proc).probs(n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)

                p1_swipe[k] = probs.get(state_010, 0.0)
                p2_swipe[k] = probs.get(state_001, 0.0)

            preds[i] = p2_swipe.mean()
            mem_p1[t], mem_p2[t] = p1_swipe.mean(), p2_swipe.mean()

    # finalize
    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, n_samples)
    return preds