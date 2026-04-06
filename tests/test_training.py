from __future__ import annotations

from src.config import SimConfig
from src.training import _resolve_n_photons


def _cfg(output_mode: str, n_photons=None) -> SimConfig:
    return SimConfig(
        n_modes=6,
        encoding_mode=0,
        target_mode=(5,) if output_mode == "singles" else (2, 4),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        encoding_phase_idx=None,
        output_mode=output_mode,
        input_modes=None if output_mode == "singles" else (1, 4),
        working_detectors=None if output_mode == "singles" else tuple(range(6)),
        noise_std=None,
        n_samples=100,
        memory_depth=1,
        n_swipe=0,
        swipe_span=0.0,
        n_photons=n_photons,
        backend="numpy",
        loss_type="mse",
        n_classes=1,
    )


def test_resolve_n_photons_defaults_to_two_for_coincidence():
    cfg = _cfg("coincidence", n_photons=None)
    assert _resolve_n_photons(cfg, 30) == tuple([2] * 30)


def test_resolve_n_photons_defaults_to_one_for_singles():
    cfg = _cfg("singles", n_photons=None)
    assert _resolve_n_photons(cfg, 30) == tuple([1] * 30)
