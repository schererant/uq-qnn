from __future__ import annotations

import time
from collections import Counter
from typing import Any, Optional, Union, Tuple, Sequence
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

from .config import SimConfig
from .circuits import build_circuit, build_parametric_circuit, get_mzi_modes_for_phase
from .numpy_backend import (
    run_vectorized_non_memristive,
    singles_prob_from_unitary,
    singles_class_probs_from_unitary,
    unitary_for_point,
)
from .coincidence import (
    get_cc_labels,
    working_detectors_to_cc_indices,
    probs_to_coincidences,
    postselect_measurement,
    apply_noise_to_outcomes,
)
from .logging_config import get_logger

logger = get_logger(__name__)


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

    def log_circuits(self, elapsed: float, count: int = 1):
        """Record timing for multiple logical circuit evaluations (e.g. vectorized batch)."""
        self.circuit_call_count += count
        self.circuit_total_time += elapsed

    def report(self):
        lines = [
            f"Circuit sequence runs: {self.call_count}",
            f"Total sequence time: {self.total_time:.3f}s",
        ]
        if self.call_count > 0:
            lines.append(
                f"Avg time per sequence: {self.total_time / self.call_count:.6f}s"
            )
        sample_parts = [f"{n}×{freq}" for n, freq in self.samples_counter.items()]
        lines.append(f"Sampler sample counts: {', '.join(sample_parts) or 'none'}")
        lines.append(f"Individual circuit sims: {self.circuit_call_count}")
        lines.append(f"Total circuit sim time: {self.circuit_total_time:.3f}s")
        if self.circuit_call_count > 0:
            lines.append(
                f"Avg time per circuit sim: "
                f"{self.circuit_total_time / self.circuit_call_count:.6f}s"
            )
        logger.info("Simulation statistics:\n  " + "\n  ".join(lines))

    def stats_dict(self) -> dict[str, Any]:
        """Return all statistics as a JSON-serializable dictionary."""
        return {
            "call_count": self.call_count,
            "total_time": self.total_time,
            "samples_counter": dict(self.samples_counter),
            "circuit_call_count": self.circuit_call_count,
            "circuit_total_time": self.circuit_total_time,
            "avg_time_per_sequence": self.total_time / self.call_count
            if self.call_count > 0
            else 0,
            "avg_time_per_circuit": self.circuit_total_time / self.circuit_call_count
            if self.circuit_call_count > 0
            else 0,
        }


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
        return tuple(
            get_mzi_modes_for_phase(idx, n_modes) for idx in memristive_indices
        )
    modes = tuple((int(m1), int(m2)) for m1, m2 in memristive_output_modes)
    if len(modes) != len(memristive_indices):
        raise ValueError(
            f"memristive_output_modes must have {len(memristive_indices)} entries "
            f"(one per memristive phase), got {len(modes)}"
        )
    for j, (m1, m2) in enumerate(modes):
        if m1 < 0 or m1 >= n_modes or m2 < 0 or m2 >= n_modes:
            raise ValueError(
                f"memristive_output_modes[{j}] = ({m1}, {m2}): modes must be in [0, {n_modes - 1}]"
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
                f"memristive_phase_idx must be in [0, {n_phases - 1}] for {n_modes} modes, got {idx}"
            )
        return (idx,)
    # Sequence (tuple, list, etc.)
    indices = tuple(int(x) for x in memristive_phase_idx)
    if len(indices) == 0:
        return ()
    for idx in indices:
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"Each memristive_phase_idx must be in [0, {n_phases - 1}] for {n_modes} modes, got {idx}"
            )
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"memristive_phase_idx must not contain duplicates, got {memristive_phase_idx}"
        )
    return indices


def run_simulation_sequence_np(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    cfg: SimConfig,
    *,
    return_class_probs: bool = False,
) -> np.ndarray:
    """
    Runs a sequence of photonic-circuit simulations. Architecture is always Clements (3x3, 6x6, etc.).

    Structural parameters are read from ``cfg``; ``params`` and ``encoded_phases`` vary per solve.

    Args:
        params: Phase parameters. If no memristive phases: [phase_0, ..., phase_{n-1}].
            If memristive: [phase_0, ..., phase_{n-1}, w_0, ..., w_{k-1}] for k memristive phases.
        encoded_phases: Phase values (radians) for each data point.
        cfg: :class:`SimConfig` with modes, backend, sampling counts, etc.
        return_class_probs: If True and multiple targets, returns (n_data, n_classes).

    Returns:
        Predicted probability per input point, or class probabilities if return_class_probs.
    """
    start_time = time.perf_counter()
    n_swipe = cfg.n_swipe
    if n_swipe < 0:
        raise ValueError("n_swipe must be >= 0.")
    if not isinstance(cfg.n_samples, int) or cfg.n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {cfg.n_samples!r}")
    if cfg.backend not in ("numpy", "perceval"):
        raise ValueError(f"backend must be 'numpy' or 'perceval', got {cfg.backend!r}")

    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    memristive_indices = _normalize_memristive_phase_idx(
        cfg.memristive_phase_idx, cfg.n_modes, n_phases
    )
    n_memristive = len(memristive_indices)
    output_modes = (
        _normalize_memristive_output_modes(
            cfg.memristive_output_modes, memristive_indices, cfg.n_modes
        )
        if n_memristive > 0
        else ()
    )

    # Continuous mode only when memristive is active
    if n_swipe > 0 and n_memristive == 0:
        logger.warning(
            "Continuous mode requires memristive phases — switching to discrete."
        )
        n_swipe = 0
    if n_swipe > 0 and cfg.swipe_span <= 0:
        raise ValueError("swipe_span must be > 0 for continuous mode.")

    mode = "continuous" if n_swipe > 0 else "discrete"
    expected_params = n_phases + n_memristive
    if len(params) != expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters ({n_phases} phases"
            + (f" + {n_memristive} weights" if n_memristive else "")
            + f") for {cfg.n_modes} modes, got {len(params)}"
        )

    weights = params[-n_memristive:] if n_memristive else None

    target_mode = (
        cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)
    )

    # Input state: singles (1 photon) or coincidence (2 photons)
    if cfg.output_mode == "coincidence":
        if n_memristive > 0:
            raise ValueError("Coincidence mode does not support memristive phases yet")
        in_modes = (
            (1, 4)
            if cfg.n_modes >= 6
            else (0, 1)
            if cfg.input_modes is None
            else tuple(int(m) for m in cfg.input_modes)
        )
        if len(in_modes) != 2:
            raise ValueError(
                f"Coincidence mode requires exactly 2 input modes, got {in_modes}"
            )
        inp = [0] * cfg.n_modes
        for m in in_modes:
            if m < 0 or m >= cfg.n_modes:
                raise ValueError(
                    f"input_modes {cfg.input_modes} out of range [0, {cfg.n_modes - 1}]"
                )
            inp[m] += 1
        input_state = pcvl.BasicState(inp)
        wd_tuple = (
            tuple(cfg.working_detectors)
            if cfg.working_detectors is not None
            else (0, 1, 5)
        )
        working_cc_indices = working_detectors_to_cc_indices(wd_tuple, cfg.n_modes)
        cc_labels = get_cc_labels(cfg.n_modes)
        add_noise = cfg.noise_std is not None and (
            (isinstance(cfg.noise_std, (int, float)) and float(cfg.noise_std) > 0)
            or (
                hasattr(cfg.noise_std, "__len__")
                and len(cfg.noise_std) > 0
                and any(float(s) > 0 for s in cfg.noise_std)
            )
        )
    else:
        inp = [0] * cfg.n_modes
        inp[cfg.encoding_mode] = 1
        input_state = pcvl.BasicState(inp)
        working_cc_indices: Tuple[int, ...] = ()
        cc_labels = []
        add_noise = False

    state_m1_list = []
    state_m2_list = []
    for j in range(n_memristive):
        m1, m2 = output_modes[j]
        s1, s2 = [0] * cfg.n_modes, [0] * cfg.n_modes
        s1[m1], s2[m2] = 1, 1
        state_m1_list.append(pcvl.BasicState(s1))
        state_m2_list.append(pcvl.BasicState(s2))

    # Build target states list for multi-class / probability extraction
    target_modes_list = []
    for m in target_mode:
        tm = [0] * cfg.n_modes
        tm[m] = 1
        target_modes_list.append(pcvl.BasicState(tm))

    if n_memristive > 0:
        mem_p1 = np.zeros((cfg.memory_depth, n_memristive), dtype=float)
        mem_p2 = np.zeros((cfg.memory_depth, n_memristive), dtype=float)
    num_pts = len(encoded_phases)

    # Determine if we need multi-class output
    if cfg.output_mode == "coincidence" and working_cc_indices:
        n_classes = len(working_cc_indices)
    else:
        n_classes = len(target_mode) if target_mode is not None else 1
    if return_class_probs and n_classes > 1:
        preds = np.zeros((num_pts, n_classes), dtype=float)
    else:
        preds = np.zeros(num_pts, dtype=float)

    # Precompute base phases and offsets for continuous mode
    if mode == "continuous":
        enc_base = encoded_phases
        # TODO: Use Iris data for that
        offsets = np.linspace(
            -cfg.swipe_span / 2, cfg.swipe_span / 2, n_swipe, dtype=encoded_phases.dtype
        )
    else:
        # Initialize offsets as empty array for discrete mode to avoid reference errors
        offsets = np.array([])
        enc_base = encoded_phases

    # ----- NumPy backend: vectorized non-memristive discrete -----
    if cfg.backend == "numpy" and mode == "discrete" and n_memristive == 0:
        t_np = time.perf_counter()
        cfg_vec = (
            cfg
            if cfg.target_mode is not None
            else cfg.replace(target_mode=target_mode)
        )
        preds = run_vectorized_non_memristive(
            params=params,
            encoded_phases=encoded_phases,
            cfg=cfg_vec,
            return_class_probs=return_class_probs,
        )
        elapsed_np = time.perf_counter() - t_np
        sim_logger.log_circuits(elapsed_np, num_pts)
        sim_logger.log(time.perf_counter() - start_time, cfg.n_samples)
        return preds

    # ----- NumPy backend: memristive and/or swipe (singles only) -----
    if cfg.backend == "numpy":
        if cfg.output_mode == "coincidence":
            raise ValueError(
                "numpy backend does not support coincidence with memristive/swipe; "
                "use backend='perceval'"
            )
        for i in range(num_pts):
            t = i % cfg.memory_depth
            if n_memristive > 0:
                mem_phis = np.empty(n_memristive, dtype=float)
                for j in range(n_memristive):
                    if i == 0:
                        mem_phis[j] = np.pi / 4
                    else:
                        m1m = mem_p1[:, j].mean()
                        m2m = mem_p2[:, j].mean()
                        arg = np.clip(m1m + weights[j] * m2m, 1e-9, 1 - 1e-9)
                        mem_phis[j] = np.arccos(np.sqrt(arg))

            if mode == "discrete":
                enc_phi = float(encoded_phases[i])
                if n_memristive > 0:
                    phases_loc = params[:-n_memristive].copy()
                    for j, idx in enumerate(memristive_indices):
                        phases_loc[idx] = mem_phis[j]
                else:
                    phases_loc = params.copy()

                t0 = time.perf_counter()
                u = unitary_for_point(
                    phases_loc,
                    enc_phi,
                    cfg.n_modes,
                    cfg.encoding_mode,
                    cfg.encoding_phase_idx,
                )
                if n_memristive > 0:
                    for j in range(n_memristive):
                        m1, m2 = output_modes[j]
                        mem_p1[t, j] = float(np.abs(u[m1, cfg.encoding_mode]) ** 2)
                        mem_p2[t, j] = float(np.abs(u[m2, cfg.encoding_mode]) ** 2)
                if return_class_probs and n_classes > 1:
                    preds[i, :] = singles_class_probs_from_unitary(
                        u, cfg.encoding_mode, target_mode
                    )
                else:
                    preds[i] = singles_prob_from_unitary(
                        u, cfg.encoding_mode, target_mode
                    )
                sim_logger.log_circuit(time.perf_counter() - t0)

            else:
                # swipe mode (memristive)
                p1_swipe = np.empty((n_swipe, n_memristive), dtype=float)
                p2_swipe = np.empty((n_swipe, n_memristive), dtype=float)
                target_swipe = np.empty(
                    (n_swipe, n_classes)
                    if return_class_probs and n_classes > 1
                    else (n_swipe,),
                    dtype=float,
                )
                for k, off in enumerate(offsets):
                    enc_phi = float(enc_base[i] + off)
                    phases_loc = params[:-n_memristive].copy()
                    for j, idx in enumerate(memristive_indices):
                        phases_loc[idx] = mem_phis[j]
                    t0 = time.perf_counter()
                    u = unitary_for_point(
                        phases_loc,
                        enc_phi,
                        cfg.n_modes,
                        cfg.encoding_mode,
                        cfg.encoding_phase_idx,
                    )
                    for j in range(n_memristive):
                        m1, m2 = output_modes[j]
                        p1_swipe[k, j] = float(np.abs(u[m1, cfg.encoding_mode]) ** 2)
                        p2_swipe[k, j] = float(np.abs(u[m2, cfg.encoding_mode]) ** 2)
                    if return_class_probs and n_classes > 1:
                        target_swipe[k, :] = singles_class_probs_from_unitary(
                            u, cfg.encoding_mode, target_mode
                        )
                    else:
                        target_swipe[k] = singles_prob_from_unitary(
                            u, cfg.encoding_mode, target_mode
                        )
                    sim_logger.log_circuit(time.perf_counter() - t0)
                if return_class_probs and n_classes > 1:
                    preds[i] = target_swipe.mean(axis=0)
                else:
                    preds[i] = target_swipe.mean()
                for j in range(n_memristive):
                    mem_p1[t, j], mem_p2[t, j] = (
                        p1_swipe[:, j].mean(),
                        p2_swipe[:, j].mean(),
                    )

        elapsed = time.perf_counter() - start_time
        sim_logger.log(elapsed, cfg.n_samples)
        return preds

    # ----- Perceval backend -----
    reuse_enc_param = (
        mode == "discrete" and n_memristive == 0 and cfg.encoding_phase_idx is None
    )
    enc_param = None
    sampler = None
    if reuse_enc_param:
        enc_param = pcvl.P("uqqnn_enc")
        enc_param.set_value(float(encoded_phases[0]) % (2 * np.pi))
        phases_fixed = params.copy()
        circ0 = build_parametric_circuit(
            phases_fixed,
            enc_param,
            n_modes=cfg.n_modes,
            encoding_mode=cfg.encoding_mode,
            encoding_phase_idx=None,
        )
        proc0 = pcvl.Processor("SLOS", circ0)
        proc0.with_input(input_state)
        sampler = Sampler(proc0)

    for i in range(num_pts):
        t = i % cfg.memory_depth
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

            if reuse_enc_param:
                enc_param.set_value(float(enc_phi) % (2 * np.pi))
                t0 = time.perf_counter()
                probs = sampler.probs(cfg.n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)
            else:
                circ = build_circuit(
                    phases,
                    enc_phi,
                    n_modes=cfg.n_modes,
                    encoding_mode=cfg.encoding_mode,
                    encoding_phase_idx=cfg.encoding_phase_idx,
                )
                proc = pcvl.Processor("SLOS", circ)
                proc.with_input(input_state)
                t0 = time.perf_counter()
                probs = Sampler(proc).probs(cfg.n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)

            if n_memristive > 0:
                for j in range(n_memristive):
                    mem_p1[t, j] = probs.get(state_m1_list[j], 0.0)
                    mem_p2[t, j] = probs.get(state_m2_list[j], 0.0)
            if cfg.output_mode == "coincidence":
                coinc = probs_to_coincidences(probs, cfg.n_modes)
                out = postselect_measurement(
                    coinc, working_cc_indices, cc_labels, fallback_uniform=True
                )
                if add_noise:
                    out = apply_noise_to_outcomes(
                        out, cfg.noise_std, working_cc_indices, cc_labels, seed=i
                    )
                if return_class_probs and len(working_cc_indices) > 1:
                    preds[i, :] = out[working_cc_indices]
                else:
                    preds[i] = out[working_cc_indices[0]] if working_cc_indices else 0.0
            elif return_class_probs and n_classes > 1:
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
            target_swipe = np.empty(
                (n_swipe, n_classes)
                if return_class_probs and n_classes > 1
                else (n_swipe,),
                dtype=float,
            )
            phases_sw = params[:-n_memristive].copy()
            for j, idx in enumerate(memristive_indices):
                phases_sw[idx] = mem_phis[j]
            reuse_enc_swipe = cfg.encoding_phase_idx is None
            enc_param_sw = None
            sampler_sw = None
            if reuse_enc_swipe:
                enc_param_sw = pcvl.P(f"uqqnn_sw_{i}")
                enc_param_sw.set_value(float(enc_base[i] + offsets[0]) % (2 * np.pi))
                circ_sw = build_parametric_circuit(
                    phases_sw,
                    enc_param_sw,
                    n_modes=cfg.n_modes,
                    encoding_mode=cfg.encoding_mode,
                    encoding_phase_idx=None,
                )
                proc_sw = pcvl.Processor("SLOS", circ_sw)
                proc_sw.with_input(input_state)
                sampler_sw = Sampler(proc_sw)

            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                if reuse_enc_swipe:
                    enc_param_sw.set_value(float(enc_phi) % (2 * np.pi))
                    t0 = time.perf_counter()
                    probs = sampler_sw.probs(cfg.n_samples)["results"]
                    sim_logger.log_circuit(time.perf_counter() - t0)
                else:
                    circ = build_circuit(
                        phases_sw,
                        enc_phi,
                        n_modes=cfg.n_modes,
                        encoding_mode=cfg.encoding_mode,
                        encoding_phase_idx=cfg.encoding_phase_idx,
                    )
                    proc = pcvl.Processor("SLOS", circ)
                    proc.with_input(input_state)
                    t0 = time.perf_counter()
                    probs = Sampler(proc).probs(cfg.n_samples)["results"]
                    sim_logger.log_circuit(time.perf_counter() - t0)
                for j in range(n_memristive):
                    p1_swipe[k, j] = probs.get(state_m1_list[j], 0.0)
                    p2_swipe[k, j] = probs.get(state_m2_list[j], 0.0)
                if cfg.output_mode == "coincidence":
                    coinc = probs_to_coincidences(probs, cfg.n_modes)
                    out = postselect_measurement(
                        coinc, working_cc_indices, cc_labels, fallback_uniform=True
                    )
                    if add_noise:
                        out = apply_noise_to_outcomes(
                            out,
                            cfg.noise_std,
                            working_cc_indices,
                            cc_labels,
                            seed=i * 1000 + k,
                        )
                    if return_class_probs and len(working_cc_indices) > 1:
                        target_swipe[k, :] = out[working_cc_indices]
                    else:
                        target_swipe[k] = (
                            out[working_cc_indices[0]] if working_cc_indices else 0.0
                        )
                elif return_class_probs and n_classes > 1:
                    for c, ts in enumerate(target_modes_list):
                        target_swipe[k, c] = probs.get(ts, 0.0)
                else:
                    target_swipe[k] = sum(
                        probs.get(ts, 0.0) for ts in target_modes_list
                    )
                    if len(target_modes_list) > 1:
                        target_swipe[k] /= len(target_modes_list)
            if cfg.output_mode == "coincidence" and working_cc_indices:
                if return_class_probs and len(working_cc_indices) > 1:
                    preds[i] = target_swipe.mean(axis=0)
                else:
                    preds[i] = target_swipe.mean()
            elif return_class_probs and n_classes > 1:
                preds[i] = target_swipe.mean(axis=0)
            else:
                preds[i] = target_swipe.mean()
            for j in range(n_memristive):
                mem_p1[t, j], mem_p2[t, j] = (
                    p1_swipe[:, j].mean(),
                    p2_swipe[:, j].mean(),
                )

    # finalize
    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, cfg.n_samples)
    return preds


def uncertainty_forward_pass(job: tuple) -> np.ndarray:
    """
    Picklable entry point for ProcessPoolExecutor-based uncertainty loops.

    ``job`` is ``(params, n_samples, encoded_phases, sim_cfg_dict, return_class_probs)``.
    """
    params, n_samples, encoded_phases, cfg_dict, return_class_probs = job
    sim_cfg = SimConfig.from_dict(cfg_dict).replace(n_samples=n_samples)
    return run_simulation_sequence_np(
        params,
        encoded_phases,
        sim_cfg,
        return_class_probs=return_class_probs,
    )
