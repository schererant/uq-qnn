from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

from ..circuits import (
    build_circuit,
    build_parametric_circuit,
    normalize_memristive_output_modes,
    normalize_memristive_phase_idx,
)
from ..coincidence import (
    apply_noise_to_outcomes,
    get_cc_labels,
    postselect_measurement,
    probs_to_coincidences,
    working_detectors_to_cc_indices,
)
from ..config import SimConfig
from ..logging_config import get_logger
from ..numpy_backend import (
    run_vectorized_non_memristive,
    singles_class_probs_from_unitary,
    singles_prob_from_unitary,
    unitary_for_point,
)
from .logger import sim_logger
from .memristive import MemristiveState

logger = get_logger(__name__)


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
    memristive_indices = normalize_memristive_phase_idx(
        cfg.memristive_phase_idx, cfg.n_modes, n_phases
    )
    n_memristive = len(memristive_indices)
    if n_memristive > 0:
        output_modes = normalize_memristive_output_modes(
            cfg.memristive_output_modes, memristive_indices, cfg.n_modes
        )
        mem_state = MemristiveState(
            n_indices=n_memristive,
            memory_depth=cfg.memory_depth,
            output_modes=output_modes,
            encoding_mode=cfg.encoding_mode,
        )
    else:
        output_modes = ()
        mem_state = None

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

    target_mode = cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)

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

    state_m1_list: list[pcvl.BasicState] = []
    state_m2_list: list[pcvl.BasicState] = []
    if mem_state:
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
            cfg if cfg.target_mode is not None else cfg.replace(target_mode=target_mode)
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
            mem_phis = mem_state.current_phases(weights, i) if mem_state else None

            if mode == "discrete":
                enc_phi = float(encoded_phases[i])
                if mem_state:
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
                if mem_state:
                    mem_state.update_from_unitary(i, u)
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
                if mem_state:
                    mem_state.update_from_prob_arrays(
                        i, p1_swipe.mean(axis=0), p2_swipe.mean(axis=0)
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
        mem_phis = mem_state.current_phases(weights, i) if mem_state else None

        if mode == "discrete":
            enc_phi = encoded_phases[i]
            if mem_state:
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

            if mem_state:
                p1_vals = [probs.get(state, 0.0) for state in state_m1_list]
                p2_vals = [probs.get(state, 0.0) for state in state_m2_list]
                mem_state.update_from_prob_arrays(i, p1_vals, p2_vals)
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
            if mem_state:
                mem_state.update_from_prob_arrays(
                    i,
                    p1_swipe.mean(axis=0),
                    p2_swipe.mean(axis=0),
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
