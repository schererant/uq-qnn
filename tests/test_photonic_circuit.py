from __future__ import annotations

import numpy as np
import pytest

from src import PhotonicCircuit, SimConfig
from src.simulation import run_simulation_sequence_np


def random_phases(n_modes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=n_modes * (n_modes - 1))


def test_construction_requires_exact_phase_count():
    phases = random_phases(6)
    PhotonicCircuit(n_modes=6, phases=phases)  # no error
    with pytest.raises(ValueError):
        PhotonicCircuit(n_modes=6, phases=phases[:-1])


def test_singles_shape_and_normalization():
    circuit = PhotonicCircuit(n_modes=6, phases=random_phases(6, seed=1))
    probs = circuit.singles(encoding_phase=0.3)
    assert probs.shape == (6,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)
    assert np.all(probs >= 0)


def test_coincidences_shape():
    circuit = PhotonicCircuit(n_modes=4, phases=random_phases(4, seed=2))
    probs = circuit.coincidences(encoding_phase=1.0, input_modes=(0, 1))
    assert probs.shape == (4 * 3 // 2,)
    assert np.all(probs >= 0)


def test_singles_batch_matches_single_calls():
    circuit = PhotonicCircuit(n_modes=5, phases=random_phases(5, seed=3))
    enc = np.linspace(0.0, np.pi, 7)
    batch = circuit.singles_batch(enc)
    for i, phi in enumerate(enc):
        np.testing.assert_allclose(batch[i], circuit.singles(phi), atol=1e-12)


def test_coincidences_batch_matches_single_calls():
    circuit = PhotonicCircuit(n_modes=5, phases=random_phases(5, seed=4))
    enc = np.linspace(0.1, 0.9, 5)
    batch = circuit.coincidences_batch(enc, input_modes=(0, 1))
    for i, phi in enumerate(enc):
        np.testing.assert_allclose(
            batch[i],
            circuit.coincidences(phi, input_modes=(0, 1)),
            atol=1e-12,
        )


def test_consistency_with_run_simulation_sequence_np():
    n_modes = 4
    phases = random_phases(n_modes, seed=5)
    encoded = np.linspace(0.2, 1.3, 8)
    cfg = SimConfig(
        n_modes=n_modes,
        encoding_mode=0,
        target_mode=(n_modes - 1,),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        encoding_phase_idx=None,
        output_mode="singles",
        input_modes=None,
        working_detectors=None,
        noise_std=None,
        n_samples=500,
        memory_depth=1,
        n_swipe=0,
        swipe_span=0.0,
        n_photons=None,
        backend="numpy",
        loss_type="mse",
        n_classes=1,
    )
    preds_runner = run_simulation_sequence_np(
        phases,
        encoded,
        cfg,
        return_class_probs=False,
    )

    circuit = PhotonicCircuit(
        n_modes=n_modes,
        phases=phases,
        encoding_mode=0,
        encoding_phase_idx=None,
    )
    singles = circuit.singles_batch(encoded)[:, cfg.target_mode[0]]
    np.testing.assert_allclose(preds_runner, singles, atol=1e-12)
