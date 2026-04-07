from __future__ import annotations

import numpy as np

from src.config import SimConfig, psr_photon_counts_for_phases, validate_sim_config
from src.data import get_data
from src.training import _init_theta, train_pytorch_generic


def _cfg(output_mode: str) -> SimConfig:
    if output_mode == "singles":
        return SimConfig(
            n_modes=6,
            input_state=(0,),
            encoding_phase_idx=0,
            photon_distinguishability=None,
            target_mode=(5,),
            memristive_phase_idx=None,
            memristive_output_modes=None,
            output_mode=output_mode,
            working_detectors=None,
            noise_std=None,
            n_samples=100,
            memory_depth=1,
            n_swipe=0,
            swipe_span=0.0,
            backend="numpy",
            loss_type="mse",
            n_classes=1,
        )
    return SimConfig(
        n_modes=6,
        input_state=(1, 4),
        encoding_phase_idx=5,
        photon_distinguishability="indistinguishable",
        target_mode=(2, 4),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode=output_mode,
        working_detectors=tuple(range(6)),
        noise_std=None,
        n_samples=100,
        memory_depth=1,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="mse",
        n_classes=1,
    )


def test_psr_photon_counts_two_for_coincidence():
    cfg = _cfg("coincidence")
    validate_sim_config(cfg)
    assert psr_photon_counts_for_phases(cfg, 30) == tuple([2] * 30)


def test_psr_photon_counts_one_for_singles():
    cfg = _cfg("singles")
    validate_sim_config(cfg)
    assert psr_photon_counts_for_phases(cfg, 30) == tuple([1] * 30)


def test_trainable_phase_indices_exclude_encoding_slot():
    from src.circuits import normalize_memristive_phase_idx

    sim_cfg = _cfg("singles")
    expected_phases = sim_cfg.n_modes * (sim_cfg.n_modes - 1)
    mem = normalize_memristive_phase_idx(
        sim_cfg.memristive_phase_idx, sim_cfg.n_modes, expected_phases
    )
    enc_pi = int(sim_cfg.encoding_phase_idx)
    phase_idx = tuple(
        i for i in range(expected_phases) if i not in mem and i != enc_pi
    )
    assert enc_pi not in phase_idx


def test_frozen_encoding_slot_is_not_clamped_like_a_weight():
    sim_cfg = _cfg("singles").replace(encoding_phase_idx=5)
    X_train, y_train, _, _ = get_data(20, 0.0, "quartic_data")
    enc_train = 2 * np.arccos(np.clip(X_train, 0.0, 1.0))

    init_theta = _init_theta(
        np.random.default_rng(42),
        sim_cfg.n_modes,
        sim_cfg.memristive_phase_idx,
    )
    enc_idx = int(sim_cfg.encoding_phase_idx)
    assert init_theta[enc_idx] > 1.0

    theta_opt, _ = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=0.0,
        epochs=1,
        seed=42,
    )

    assert np.isclose(theta_opt[enc_idx], init_theta[enc_idx])
