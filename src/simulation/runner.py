from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..clements_geometry import (
    normalize_memristive_output_modes,
    normalize_memristive_phase_idx,
)
from ..coincidence import (
    apply_noise_to_outcomes,
    coincidence_prob,
    get_nfold_channel_labels,
    marginal_click_prob,
    nfold_tuple_to_channel_index,
    nfold_working_detector_tuples,
    postselect_measurement,
    probs_to_nfold_coincidences,
)
from ..config import SimConfig, validate_sim_config
from ..logging_config import get_logger
from ..numpy_backend import (
    _has_positive_noise,
    run_fast_memristive_singles,
    run_vectorized_non_memristive,
    slos_2photon_numpy,
    singles_class_probs_from_unitary,
    singles_prob_from_unitary,
    unitary_for_point,
)
from .logger import sim_logger
from .memristive import MemristiveState

logger = get_logger(__name__)


def run_simulation_sequence(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    cfg: SimConfig,
    *,
    return_class_probs: bool = False,
) -> np.ndarray:
    """
    Run a batch of photonic forward passes (Clements mesh).

    Uses NumPy arrays for parameters and outputs. The simulation engine is
    selected by ``cfg.backend`` (``\"numpy\"`` or ``\"perceval\"``).

    Structural parameters are read from ``cfg``; ``params`` and ``encoded_phases``
    vary per solve.

    Args:
        params: Phase parameters. If no memristive phases: ``[phase_0, ..., phase_{n-1}]``.
            If memristive: phases plus ``[w_0, ..., w_{k-1}]`` for ``k`` memristive weights.
        encoded_phases: Phase values (radians) for each data point.
        cfg: :class:`SimConfig` with modes, backend, sampling counts, etc.
        return_class_probs: If True and multiple targets, returns ``(n_data, n_classes)``.

    Returns:
        Predicted probability per input point, or class probabilities if
        ``return_class_probs``.
    """
    start_time = time.perf_counter()
    n_swipe = cfg.n_swipe
    if n_swipe < 0:
        raise ValueError("n_swipe must be >= 0.")
    if cfg.backend not in ("numpy", "perceval"):
        raise ValueError(f"backend must be 'numpy' or 'perceval', got {cfg.backend!r}")
    if not isinstance(cfg.n_samples, int):
        raise ValueError(f"n_samples must be an int, got {cfg.n_samples!r}")
    if cfg.backend == "perceval" and cfg.n_samples <= 0:
        raise ValueError(
            f"n_samples must be a positive int for perceval backend, got {cfg.n_samples!r}"
        )
    if cfg.backend == "numpy" and cfg.n_samples < 0:
        raise ValueError(
            f"n_samples must be a non-negative int for numpy backend, got {cfg.n_samples!r}"
        )

    validate_sim_config(cfg)

    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    memristive_indices = normalize_memristive_phase_idx(
        cfg.memristive_phase_idx, cfg.n_modes, n_phases
    )
    n_memristive = len(memristive_indices)
    if n_memristive > 0:
        output_modes = normalize_memristive_output_modes(
            cfg.memristive_output_modes, memristive_indices, cfg.n_modes
        )
        # chip_output uses marginal probabilities from the SLOS dict; singles_input_mode
        # is only needed for the legacy internal_arm unitary-column extraction.
        sim_input_mode = (
            None
            if cfg.feedback_mode == "chip_output"
            else cfg.singles_input_mode
        )
        mem_state = MemristiveState(
            n_indices=n_memristive,
            memory_depth=cfg.memory_depth,
            output_modes=output_modes,
            singles_input_mode=sim_input_mode,
        )
    else:
        output_modes = ()
        mem_state = None

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

    if cfg.output_mode == "coincidence" and n_memristive > 0:
        if cfg.feedback_mode != "chip_output":
            raise ValueError(
                "Coincidence mode with memristive phases requires feedback_mode='chip_output' "
                "and backend='perceval'.  Set feedback_mode='chip_output' together with "
                "feedback_modes and (for regression) computation_modes in SimConfig."
            )
        if cfg.backend != "perceval" and not (
            cfg.backend == "numpy" and int(sum(cfg.input_state)) == 2
        ):
            raise ValueError(
                "feedback_mode='chip_output' with memristive coincidence requires "
                "backend='perceval', except 2-photon NumPy analytical SLOS fast path."
            )

    num_pts = len(encoded_phases)

    if mode == "continuous":
        enc_base = encoded_phases
        offsets = np.linspace(
            -cfg.swipe_span / 2, cfg.swipe_span / 2, n_swipe, dtype=encoded_phases.dtype
        )
    else:
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

    # ----- NumPy backend: faster memristive singles path (discrete + swipe) -----
    if cfg.backend == "numpy" and n_memristive > 0 and cfg.output_mode == "singles":
        t_np = time.perf_counter()
        cfg_vec = (
            cfg if cfg.target_mode is not None else cfg.replace(target_mode=target_mode)
        )
        preds = run_fast_memristive_singles(
            params=params,
            encoded_phases=enc_base,
            cfg=cfg_vec,
            memristive_indices=memristive_indices,
            output_modes=output_modes,
            offsets=offsets if mode == "continuous" else None,
            return_class_probs=return_class_probs,
        )
        elapsed_np = time.perf_counter() - t_np
        sim_logger.log_circuits(
            elapsed_np, num_pts * (n_swipe if mode == "continuous" else 1)
        )
        sim_logger.log(time.perf_counter() - start_time, cfg.n_samples)
        return preds

    # ----- NumPy backend: memristive singles (slow path; rare fallback) -----
    if cfg.backend == "numpy":
        if cfg.output_mode == "coincidence" and not (
            cfg.feedback_mode == "chip_output" and int(sum(cfg.input_state)) == 2
        ):
            raise ValueError(
                "numpy backend does not support coincidence with memristive phases; "
                "use backend='perceval' with feedback_mode='chip_output', except for "
                "2-photon chip_output analytical fast path"
            )
        n_classes = len(target_mode) if target_mode is not None else 1
        if return_class_probs and n_classes > 1:
            preds = np.zeros((num_pts, n_classes), dtype=float)
        else:
            preds = np.zeros(num_pts, dtype=float)
        for i in range(num_pts):
            mem_phis = mem_state.current_phases(weights, i) if mem_state else None

            if mode == "discrete":
                enc_phi = float(encoded_phases[i])
                if mem_state:
                    assert mem_phis is not None
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
                    cfg.encoding_phase_idx,
                )
                if cfg.output_mode == "coincidence":
                    input_modes = tuple(
                        int(k) for k, occ in enumerate(cfg.input_state) if int(occ) == 1
                    )
                    slos_dist = slos_2photon_numpy(u, input_modes=input_modes)
                    if mem_state:
                        assert cfg.feedback_modes is not None
                        p1_vals = [
                            marginal_click_prob(slos_dist, fm[0]) for fm in cfg.feedback_modes
                        ]
                        p2_vals = [
                            marginal_click_prob(slos_dist, fm[1]) for fm in cfg.feedback_modes
                        ]
                        mem_state.update_from_prob_arrays(i, p1_vals, p2_vals)
                    if (
                        cfg.feedback_mode == "chip_output"
                        and cfg.loss_type == "mse"
                        and cfg.computation_modes is not None
                    ):
                        preds[i] = coincidence_prob(
                            slos_dist, cfg.computation_modes[0], cfg.computation_modes[1]
                        )
                    else:
                        raise ValueError(
                            "NumPy 2-photon analytical SLOS path currently supports "
                            "chip_output coincidence regression only."
                        )
                elif mem_state:
                    mem_state.update_from_unitary(i, u)
                    if return_class_probs and n_classes > 1:
                        preds[i, :] = singles_class_probs_from_unitary(
                            u, cfg.singles_input_mode, target_mode
                        )
                    else:
                        preds[i] = singles_prob_from_unitary(
                            u, cfg.singles_input_mode, target_mode
                        )
                else:
                    if return_class_probs and n_classes > 1:
                        preds[i, :] = singles_class_probs_from_unitary(
                            u, cfg.singles_input_mode, target_mode
                        )
                    else:
                        preds[i] = singles_prob_from_unitary(
                            u, cfg.singles_input_mode, target_mode
                        )
                sim_logger.log_circuit(time.perf_counter() - t0)

            else:
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
                    assert mem_phis is not None
                    phases_loc = params[:-n_memristive].copy()
                    for j, idx in enumerate(memristive_indices):
                        phases_loc[idx] = mem_phis[j]
                    t0 = time.perf_counter()
                    u = unitary_for_point(
                        phases_loc,
                        enc_phi,
                        cfg.n_modes,
                        cfg.encoding_phase_idx,
                    )
                    for j in range(n_memristive):
                        m1, m2 = output_modes[j]
                        p1_swipe[k, j] = float(
                            np.abs(u[m1, cfg.singles_input_mode]) ** 2
                        )
                        p2_swipe[k, j] = float(
                            np.abs(u[m2, cfg.singles_input_mode]) ** 2
                        )
                    if return_class_probs and n_classes > 1:
                        target_swipe[k, :] = singles_class_probs_from_unitary(
                            u, cfg.singles_input_mode, target_mode
                        )
                    else:
                        target_swipe[k] = singles_prob_from_unitary(
                            u, cfg.singles_input_mode, target_mode
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

    # ----- Perceval backend (lazy import) -----
    from perceval.algorithm import Sampler  # type: ignore[attr-defined]

    import perceval as pcvl

    from ..circuits import build_circuit

    if cfg.output_mode == "coincidence":
        inp = [int(x) for x in cfg.input_state]
        if len(inp) != cfg.n_modes:
            raise ValueError(
                f"input_state must have length n_modes={cfg.n_modes} for Perceval, got {len(inp)}"
            )
        input_state = pcvl.BasicState(inp)
        assert cfg.working_detectors is not None
        wd_tuple = tuple(int(x) for x in cfg.working_detectors)
        n_photons_pv = int(sum(inp))
        nfold_tuples_pv = nfold_working_detector_tuples(wd_tuple, n_photons_pv)
        n_ch_pv = len(nfold_tuples_pv)
        working_cc_indices = tuple(range(n_ch_pv))
        cc_labels = get_nfold_channel_labels(wd_tuple, n_photons_pv)
        add_noise = _has_positive_noise(cfg.noise_std)
    else:
        inp = [0] * cfg.n_modes
        inp[cfg.singles_input_mode] = 1
        input_state = pcvl.BasicState(inp)
        working_cc_indices = ()
        cc_labels = []
        add_noise = False

    readout_cc_idx: int | None = None
    if cfg.output_mode == "coincidence" and cfg.loss_type == "mse":
        assert cfg.target_mode is not None
        readout_cc_idx = nfold_tuple_to_channel_index(
            cfg.target_mode,
            cfg.working_detectors,
            int(sum(cfg.input_state)),
        )

    # Legacy internal_arm feedback: build 1-photon states for probs.get() lookup.
    # chip_output feedback: marginal probabilities extracted post-hoc; no state list needed.
    state_m1_list: list[Any] = []
    state_m2_list: list[Any] = []
    if mem_state and cfg.feedback_mode == "internal_arm":
        for j in range(n_memristive):
            m1, m2 = output_modes[j]
            s1, s2 = [0] * cfg.n_modes, [0] * cfg.n_modes
            s1[m1], s2[m2] = 1, 1
            state_m1_list.append(pcvl.BasicState(s1))
            state_m2_list.append(pcvl.BasicState(s2))

    target_modes_list = []
    for m in target_mode:
        tm = [0] * cfg.n_modes
        tm[m] = 1
        target_modes_list.append(pcvl.BasicState(tm))

    if cfg.output_mode == "coincidence" and working_cc_indices:
        n_classes = len(working_cc_indices)
    else:
        n_classes = len(target_mode) if target_mode is not None else 1
    if return_class_probs and n_classes > 1:
        preds = np.zeros((num_pts, n_classes), dtype=float)
    else:
        preds = np.zeros(num_pts, dtype=float)

    for i in range(num_pts):
        mem_phis = mem_state.current_phases(weights, i) if mem_state else None

        if mode == "discrete":
            enc_phi = encoded_phases[i]
            if mem_state:
                assert mem_phis is not None
                phases = params[:-n_memristive].copy()
                for j, idx in enumerate(memristive_indices):
                    phases[idx] = mem_phis[j]
            else:
                phases = params.copy()

            circ = build_circuit(
                phases,
                enc_phi,
                n_modes=cfg.n_modes,
                encoding_phase_idx=cfg.encoding_phase_idx,
            )
            proc = pcvl.Processor("SLOS", circ)
            proc.with_input(input_state)
            t0 = time.perf_counter()
            probs = Sampler(proc).probs(cfg.n_samples)["results"]
            sim_logger.log_circuit(time.perf_counter() - t0)

            if mem_state:
                if cfg.feedback_mode == "chip_output":
                    assert cfg.feedback_modes is not None
                    p1_vals = [marginal_click_prob(probs, fm[0]) for fm in cfg.feedback_modes]
                    p2_vals = [marginal_click_prob(probs, fm[1]) for fm in cfg.feedback_modes]
                else:
                    p1_vals = [probs.get(state, 0.0) for state in state_m1_list]
                    p2_vals = [probs.get(state, 0.0) for state in state_m2_list]
                mem_state.update_from_prob_arrays(i, p1_vals, p2_vals)
            if cfg.output_mode == "coincidence" and cfg.feedback_mode == "chip_output" and cfg.loss_type == "mse":
                # Hardware-realistic regression: P(click_c AND click_d) from
                # cfg.computation_modes, read directly from the SLOS distribution.
                assert cfg.computation_modes is not None
                preds[i] = coincidence_prob(probs, cfg.computation_modes[0], cfg.computation_modes[1])
            elif cfg.output_mode == "coincidence":
                coinc = probs_to_nfold_coincidences(
                    probs,
                    cfg.n_modes,
                    wd_tuple,
                    n_photons_pv,
                )
                out = postselect_measurement(
                    coinc, working_cc_indices, cc_labels, fallback_uniform=True
                )
                if add_noise:
                    assert cfg.noise_std is not None
                    out = apply_noise_to_outcomes(
                        out, cfg.noise_std, working_cc_indices, cc_labels, seed=i
                    )
                if return_class_probs and len(working_cc_indices) > 1:
                    preds[i, :] = out[working_cc_indices]
                else:
                    if readout_cc_idx is None:
                        raise ValueError(
                            "scalar coincidence readout requires target_mode as output "
                            "target_mode as an N-tuple of distinct detector indices when using loss_type='mse'"
                        )
                    preds[i] = out[readout_cc_idx]
            elif return_class_probs and n_classes > 1:
                for c, target_state in enumerate(target_modes_list):
                    preds[i, c] = probs.get(target_state, 0.0)
            else:
                target_prob = sum(probs.get(ts, 0.0) for ts in target_modes_list)
                if len(target_modes_list) > 1:
                    target_prob /= len(target_modes_list)
                preds[i] = target_prob

        else:
            p1_swipe = np.empty((n_swipe, n_memristive), dtype=float)
            p2_swipe = np.empty((n_swipe, n_memristive), dtype=float)
            target_swipe = np.empty(
                (n_swipe, n_classes)
                if return_class_probs and n_classes > 1
                else (n_swipe,),
                dtype=float,
            )
            phases_sw = params[:-n_memristive].copy()
            assert mem_phis is not None
            for j, idx in enumerate(memristive_indices):
                phases_sw[idx] = mem_phis[j]

            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                circ = build_circuit(
                    phases_sw,
                    enc_phi,
                    n_modes=cfg.n_modes,
                    encoding_phase_idx=cfg.encoding_phase_idx,
                )
                proc = pcvl.Processor("SLOS", circ)
                proc.with_input(input_state)
                t0 = time.perf_counter()
                probs = Sampler(proc).probs(cfg.n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)
                if cfg.feedback_mode == "chip_output":
                    assert cfg.feedback_modes is not None
                    for j in range(n_memristive):
                        p1_swipe[k, j] = marginal_click_prob(probs, cfg.feedback_modes[j][0])
                        p2_swipe[k, j] = marginal_click_prob(probs, cfg.feedback_modes[j][1])
                else:
                    for j in range(n_memristive):
                        p1_swipe[k, j] = probs.get(state_m1_list[j], 0.0)
                        p2_swipe[k, j] = probs.get(state_m2_list[j], 0.0)
                if cfg.output_mode == "coincidence" and cfg.feedback_mode == "chip_output" and cfg.loss_type == "mse":
                    assert cfg.computation_modes is not None
                    target_swipe[k] = coincidence_prob(probs, cfg.computation_modes[0], cfg.computation_modes[1])
                elif cfg.output_mode == "coincidence":
                    coinc = probs_to_nfold_coincidences(
                        probs,
                        cfg.n_modes,
                        wd_tuple,
                        n_photons_pv,
                    )
                    out = postselect_measurement(
                        coinc, working_cc_indices, cc_labels, fallback_uniform=True
                    )
                    if add_noise:
                        assert cfg.noise_std is not None
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
                        if readout_cc_idx is None:
                            raise ValueError(
                                "scalar coincidence readout requires target_mode as output "
                                "target_mode as an N-tuple of distinct detector indices when using loss_type='mse'"
                            )
                        target_swipe[k] = out[readout_cc_idx]
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

    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, cfg.n_samples)
    return preds


run_simulation_sequence_np = run_simulation_sequence


def uncertainty_forward_pass(job: tuple) -> np.ndarray:
    """
    Picklable entry point for ProcessPoolExecutor-based uncertainty loops.

    ``job`` is ``(params, n_samples, encoded_phases, sim_cfg_dict, return_class_probs)``.
    """
    params, n_samples, encoded_phases, cfg_dict, return_class_probs = job
    sim_cfg = SimConfig.from_dict(cfg_dict).replace(n_samples=n_samples)
    return run_simulation_sequence(
        params,
        encoded_phases,
        sim_cfg,
        return_class_probs=return_class_probs,
    )
