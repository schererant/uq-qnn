from __future__ import annotations

import numpy as np

from src.circuits import (
    normalize_memristive_output_modes,
    normalize_memristive_phase_idx,
)
from src.config import SimConfig
from src.numpy_backend import (
    singles_class_probs_from_unitary,
    singles_prob_from_unitary,
    unitary_for_point,
)
from src.simulation import MemristiveState, run_simulation_sequence_np


def _reference_memristive_singles(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    cfg: SimConfig,
    *,
    return_class_probs: bool = False,
) -> np.ndarray:
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    memristive_indices = normalize_memristive_phase_idx(
        cfg.memristive_phase_idx, cfg.n_modes, n_phases
    )
    output_modes = normalize_memristive_output_modes(
        cfg.memristive_output_modes, memristive_indices, cfg.n_modes
    )
    weights = params[-len(memristive_indices) :]
    target_mode = cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)

    mem_state = MemristiveState(
        n_indices=len(memristive_indices),
        memory_depth=cfg.memory_depth,
        output_modes=output_modes,
        singles_input_mode=cfg.singles_input_mode,
    )

    if cfg.n_swipe > 0:
        offsets = np.linspace(
            -cfg.swipe_span / 2,
            cfg.swipe_span / 2,
            cfg.n_swipe,
            dtype=np.float64,
        )
    else:
        offsets = np.array([0.0], dtype=np.float64)

    if return_class_probs and len(target_mode) > 1:
        preds = np.zeros((len(encoded_phases), len(target_mode)), dtype=float)
    else:
        preds = np.zeros(len(encoded_phases), dtype=float)

    for i, enc_phi in enumerate(encoded_phases):
        mem_phis = mem_state.current_phases(weights, i)
        if cfg.n_swipe > 0:
            p1_swipe = np.empty((cfg.n_swipe, len(memristive_indices)), dtype=float)
            p2_swipe = np.empty((cfg.n_swipe, len(memristive_indices)), dtype=float)
            if return_class_probs and len(target_mode) > 1:
                target_swipe = np.empty((cfg.n_swipe, len(target_mode)), dtype=float)
            else:
                target_swipe = np.empty(cfg.n_swipe, dtype=float)

            for k, off in enumerate(offsets):
                phases_loc = params[: -len(memristive_indices)].copy()
                for j, idx in enumerate(memristive_indices):
                    phases_loc[idx] = mem_phis[j]
                u = unitary_for_point(
                    phases_loc,
                    float(enc_phi + off),
                    cfg.n_modes,
                    cfg.encoding_phase_idx,
                )
                for j, (m1, m2) in enumerate(output_modes):
                    p1_swipe[k, j] = float(
                        np.abs(u[m1, cfg.singles_input_mode]) ** 2
                    )
                    p2_swipe[k, j] = float(
                        np.abs(u[m2, cfg.singles_input_mode]) ** 2
                    )
                if return_class_probs and len(target_mode) > 1:
                    target_swipe[k, :] = singles_class_probs_from_unitary(
                        u, cfg.singles_input_mode, target_mode
                    )
                else:
                    target_swipe[k] = singles_prob_from_unitary(
                        u, cfg.singles_input_mode, target_mode
                    )

            mem_state.update_from_prob_arrays(
                i,
                p1_swipe.mean(axis=0),
                p2_swipe.mean(axis=0),
            )
            if return_class_probs and len(target_mode) > 1:
                preds[i, :] = target_swipe.mean(axis=0)
            else:
                preds[i] = float(target_swipe.mean())
        else:
            phases_loc = params[: -len(memristive_indices)].copy()
            for j, idx in enumerate(memristive_indices):
                phases_loc[idx] = mem_phis[j]
            u = unitary_for_point(
                phases_loc,
                float(enc_phi),
                cfg.n_modes,
                cfg.encoding_phase_idx,
            )
            mem_state.update_from_unitary(i, u)
            if return_class_probs and len(target_mode) > 1:
                preds[i, :] = singles_class_probs_from_unitary(
                    u, cfg.singles_input_mode, target_mode
                )
            else:
                preds[i] = singles_prob_from_unitary(
                    u, cfg.singles_input_mode, target_mode
                )

    return preds


def _cfg(**overrides: object) -> SimConfig:
    base = dict(
        n_modes=6,
        n_layers=1,
        input_state=tuple(1 if i == 0 else 0 for i in range(6)),
        encoding_phase_idx=0,
        n_enc_features=None,
        photon_distinguishability=None,
        target_mode=(5,),
        memristive_phase_idx=(6,),
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=200,
        memory_depth=2,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="mse",
        n_classes=1,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )
    base.update(overrides)
    return SimConfig(**base)


def test_fast_memristive_singles_matches_legacy_loop_separate_encoding():
    rng = np.random.default_rng(123)
    cfg = _cfg(target_mode=(5,))
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    params = np.concatenate(
        [rng.uniform(0.0, 2 * np.pi, size=n_phases), np.array([0.35], dtype=float)]
    )
    enc = np.linspace(0.15, 1.25, 11)

    got = run_simulation_sequence_np(params, enc, cfg)
    expected = _reference_memristive_singles(params, enc, cfg)

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_fast_memristive_singles_matches_legacy_loop_inline_multitarget():
    rng = np.random.default_rng(321)
    cfg = _cfg(
        target_mode=(4, 5),
        encoding_phase_idx=7,
        memristive_phase_idx=(6,),
        memristive_output_modes=((1, 2),),
    )
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    params = np.concatenate(
        [rng.uniform(0.0, 2 * np.pi, size=n_phases), np.array([0.62], dtype=float)]
    )
    enc = np.linspace(0.2, 1.1, 9)

    got = run_simulation_sequence_np(params, enc, cfg, return_class_probs=True)
    expected = _reference_memristive_singles(
        params,
        enc,
        cfg,
        return_class_probs=True,
    )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_fast_memristive_singles_matches_legacy_loop_swipe_mode():
    rng = np.random.default_rng(777)
    cfg = _cfg(
        target_mode=(5,),
        n_swipe=5,
        swipe_span=np.pi / 18,
        memory_depth=3,
    )
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    params = np.concatenate(
        [rng.uniform(0.0, 2 * np.pi, size=n_phases), np.array([0.41], dtype=float)]
    )
    enc = np.linspace(0.18, 1.05, 7)

    got = run_simulation_sequence_np(params, enc, cfg)
    expected = _reference_memristive_singles(params, enc, cfg)

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
