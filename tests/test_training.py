from __future__ import annotations

from src.config import SimConfig, psr_photon_counts_for_phases, validate_sim_config


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
