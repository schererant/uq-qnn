from __future__ import annotations

import pytest

from src.config import SimConfig, validate_sim_config


def _minimal_singles(**kwargs: object) -> SimConfig:
    base = dict(
        n_modes=4,
        n_layers=1,
        input_state=(1, 0, 0, 0),
        encoding_phase_idx=0,
        n_enc_features=None,
        photon_distinguishability=None,
        target_mode=(3,),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=10,
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
    base.update(kwargs)
    return SimConfig(**base)


def _minimal_coincidence(**kwargs: object) -> SimConfig:
    base = dict(
        n_modes=4,
        n_layers=1,
        input_state=(1, 1, 0, 0),
        encoding_phase_idx=0,
        n_enc_features=None,
        photon_distinguishability="indistinguishable",
        target_mode=(0, 2),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode="coincidence",
        working_detectors=(0, 1, 2, 3),
        noise_std=None,
        n_samples=10,
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
    base.update(kwargs)
    return SimConfig(**base)


def test_singles_rejects_non_none_distinguishability():
    cfg = _minimal_singles(photon_distinguishability="indistinguishable")
    with pytest.raises(ValueError, match="photon_distinguishability must be None"):
        validate_sim_config(cfg)


def test_coincidence_requires_distinguishability():
    cfg = _minimal_coincidence(photon_distinguishability=None)
    with pytest.raises(ValueError, match="photon_distinguishability is required"):
        validate_sim_config(cfg)


def test_distinguishable_raises_not_implemented():
    cfg = _minimal_coincidence(photon_distinguishability="distinguishable")
    with pytest.raises(NotImplementedError, match="distinguishable"):
        validate_sim_config(cfg)


def test_coincidence_requires_working_detectors():
    cfg = _minimal_coincidence(working_detectors=None)
    with pytest.raises(ValueError, match="working_detectors"):
        validate_sim_config(cfg)


def test_coincidence_mse_requires_target_length_n():
    cfg = _minimal_coincidence(target_mode=(0,))
    with pytest.raises(ValueError, match="len\\(target_mode\\)==sum"):
        validate_sim_config(cfg)


def test_legacy_pair_input_state_rejected_with_migration_hint():
    cfg = _minimal_coincidence(input_state=(0, 3))
    with pytest.raises(ValueError, match="occupation vector"):
        validate_sim_config(cfg)


def test_coincidence_allows_two_photons_same_mode():
    cfg = _minimal_coincidence(input_state=(2, 0, 0, 0), target_mode=(0, 1))
    validate_sim_config(cfg)


def test_sim_config_round_trip_dict():
    cfg = _minimal_singles()
    d = cfg.to_dict()
    back = SimConfig.from_dict(d)
    assert back == cfg


def test_encoding_phase_index_out_of_range():
    cfg = _minimal_singles(encoding_phase_idx=99)
    with pytest.raises(ValueError, match="encoding_phase_idx"):
        validate_sim_config(cfg)
