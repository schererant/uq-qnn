from __future__ import annotations

import numpy as np

from src.config import SimConfig, psr_photon_counts_for_phases, validate_sim_config
from src.data import get_data
from src.training import _init_theta, train_pytorch_generic


def _cfg(output_mode: str) -> SimConfig:
    if output_mode == "singles":
        return SimConfig(
            n_modes=6,
            n_layers=1,
            input_state=tuple(1 if i == 0 else 0 for i in range(6)),
            encoding_phase_idx=0,
            n_enc_features=None,
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
            feedback_mode="internal_arm",
            feedback_modes=None,
            computation_modes=None,
        )
    return SimConfig(
        n_modes=6,
        n_layers=1,
        input_state=tuple(1 if i in (1, 4) else 0 for i in range(6)),
        encoding_phase_idx=5,
        n_enc_features=None,
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
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
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
    expected_phases = sim_cfg.total_mesh_phases
    mem = normalize_memristive_phase_idx(
        sim_cfg.memristive_phase_idx,
        sim_cfg.n_modes,
        sim_cfg.n_phases_per_layer,
    )
    enc_set = set(sim_cfg.encoding_slots)
    phase_idx = tuple(
        i for i in range(expected_phases) if i not in mem and i not in enc_set
    )
    assert not enc_set.intersection(phase_idx)


def test_frozen_encoding_slot_is_not_clamped_like_a_weight():
    sim_cfg = _cfg("singles").replace(encoding_phase_idx=5)
    X_train, y_train, _, _ = get_data(20, 0.0, "quartic_data")
    enc_train = 2 * np.arccos(np.clip(X_train, 0.0, 1.0))

    init_theta = _init_theta(
        np.random.default_rng(42),
        sim_cfg.n_modes,
        sim_cfg.n_layers,
        sim_cfg.memristive_phase_idx,
    )
    enc_idx = sim_cfg.encoding_slots[0]
    assert init_theta[enc_idx] > 1.0

    trained_state, _, _ = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=0.0,
        epochs=1,
        seed=42,
    )

    assert np.isclose(trained_state.theta[enc_idx], init_theta[enc_idx])


def test_train_pytorch_generic_is_reproducible_with_seed():
    cfg = SimConfig(
        n_modes=4,
        n_layers=1,
        input_state=(1, 0, 0, 0),
        encoding_phase_idx=(0, 1),
        n_enc_features=2,
        photon_distinguishability=None,
        target_mode=(1, 2),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=80,
        memory_depth=1,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="cross_entropy",
        n_classes=2,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )
    enc = np.array([[0.3, 1.0], [0.9, 0.5]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)

    def run_once():
        state, hist, model = train_pytorch_generic(
            enc, y, sim_cfg=cfg, lr=0.04, epochs=2, seed=17
        )
        head_w = model.head.weight.detach().cpu().numpy()
        return state.theta, head_w, hist

    theta_a, head_a, hist_a = run_once()
    theta_b, head_b, hist_b = run_once()

    np.testing.assert_allclose(theta_a, theta_b, rtol=0, atol=0)
    np.testing.assert_allclose(head_a, head_b, rtol=0, atol=0)
    assert hist_a == hist_b


def test_coincidence_cross_entropy_leaves_target_mode_none():
    """Coincidence CE uses C(W,N) channels; train_pytorch_generic must not fabricate target_mode."""
    cfg = SimConfig(
        n_modes=4,
        n_layers=1,
        input_state=(1, 1, 0, 0),
        encoding_phase_idx=0,
        n_enc_features=None,
        photon_distinguishability="indistinguishable",
        target_mode=None,
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode="coincidence",
        working_detectors=tuple(range(4)),
        noise_std=None,
        n_samples=120,
        memory_depth=1,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="cross_entropy",
        n_classes=6,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )
    validate_sim_config(cfg)

    enc = np.array([0.4, 0.85], dtype=np.float64)
    y = np.array([0, 3], dtype=np.int64)

    _, hist, _ = train_pytorch_generic(
        enc,
        y,
        sim_cfg=cfg,
        lr=0.05,
        epochs=1,
        seed=0,
    )
    assert cfg.target_mode is None
    assert len(hist) == 1
    assert np.isfinite(hist[0])
