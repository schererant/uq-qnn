from __future__ import annotations

import numpy as np
import pytest

from src.circuit import PhotonicCircuit
from src.config import CircuitConfig, SimConfig
from src.simulation import run_simulation_sequence_np


def random_phases(n_modes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=n_modes * (n_modes - 1))


def random_unitary(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(x)
    d = np.diag(r)
    q = q * (d / np.abs(d))
    return q


def singles_cfg(n_modes: int, *, enc_idx: int = 0) -> CircuitConfig:
    return CircuitConfig(
        n_modes=n_modes,
        input_state=tuple(1 if i == 0 else 0 for i in range(n_modes)),
        encoding_phase_idx=enc_idx,
        photon_distinguishability=None,
        output_mode="singles",
        working_detectors=None,
    )


def coincidence_cfg(n_modes: int) -> CircuitConfig:
    return CircuitConfig(
        n_modes=n_modes,
        input_state=tuple(1 if i in (0, 1) else 0 for i in range(n_modes)),
        encoding_phase_idx=0,
        photon_distinguishability="indistinguishable",
        output_mode="coincidence",
        working_detectors=tuple(range(n_modes)),
    )


def test_construction_requires_exact_phase_count():
    phases = random_phases(6)
    PhotonicCircuit(n_modes=6, phases=phases, circuit_config=singles_cfg(6))
    with pytest.raises(ValueError):
        PhotonicCircuit(n_modes=6, phases=phases[:-1], circuit_config=singles_cfg(6))


def test_singles_shape_and_normalization():
    circuit = PhotonicCircuit(
        n_modes=6, phases=random_phases(6, seed=1), circuit_config=singles_cfg(6)
    )
    probs = circuit.singles(encoding_phase=0.3)
    assert probs.shape == (6,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)
    assert np.all(probs >= 0)


def test_coincidences_shape():
    circuit = PhotonicCircuit(
        n_modes=4, phases=random_phases(4, seed=2), circuit_config=coincidence_cfg(4)
    )
    probs = circuit.coincidences(encoding_phase=1.0)
    assert probs.shape == (4 * 3 // 2,)
    assert np.all(probs >= 0)


def test_singles_batch_matches_single_calls():
    circuit = PhotonicCircuit(
        n_modes=5, phases=random_phases(5, seed=3), circuit_config=singles_cfg(5)
    )
    enc = np.linspace(0.0, np.pi, 7)
    batch = circuit.singles_batch(enc)
    for i, phi in enumerate(enc):
        np.testing.assert_allclose(batch[i], circuit.singles(phi), atol=1e-12)


def test_coincidences_batch_matches_single_calls():
    circuit = PhotonicCircuit(
        n_modes=5, phases=random_phases(5, seed=4), circuit_config=coincidence_cfg(5)
    )
    enc = np.linspace(0.1, 0.9, 5)
    batch = circuit.coincidences_batch(enc)
    for i, phi in enumerate(enc):
        np.testing.assert_allclose(
            batch[i],
            circuit.coincidences(phi),
            atol=1e-12,
        )


def test_consistency_with_run_simulation_sequence_np():
    n_modes = 4
    phases = random_phases(n_modes, seed=5)
    encoded = np.linspace(0.2, 1.3, 8)
    cfg = SimConfig(
        n_modes=n_modes,
        input_state=tuple(1 if i == 0 else 0 for i in range(n_modes)),
        encoding_phase_idx=0,
        photon_distinguishability=None,
        target_mode=(n_modes - 1,),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=500,
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
    preds_runner = run_simulation_sequence_np(
        phases,
        encoded,
        cfg,
        return_class_probs=False,
    )

    circuit = PhotonicCircuit(
        n_modes=n_modes,
        phases=phases,
        circuit_config=singles_cfg(n_modes, enc_idx=0),
    )
    assert cfg.target_mode is not None
    singles = circuit.singles_batch(encoded)[:, cfg.target_mode[0]]
    np.testing.assert_allclose(preds_runner, singles, atol=1e-12)


def test_identity_preserves_bunched_two_photon_fock_state():
    """Bosonic transition prob must include input factorials: |2,0,0> -> |2,0,0> on I is unity."""
    n_modes = 3
    circuit = PhotonicCircuit(
        n_modes=n_modes,
        phases=random_phases(n_modes, seed=20),
        circuit_config=singles_cfg(n_modes),
    )
    u = np.eye(n_modes, dtype=np.complex128)
    probs = circuit.coincidences((2, 0, 0), detector_mode="pnr", unitary=u)
    assert np.isclose(probs.get((2, 0, 0), 0.0), 1.0, atol=1e-10)


def test_numpy_backend_bunched_boson_probability_normalized():
    from src.numpy_backend import _transition_probability_boson

    u = np.eye(3, dtype=np.complex128)
    p = _transition_probability_boson(u, (2, 0, 0), (2, 0, 0))
    assert np.isclose(p, 1.0, atol=1e-10)


def test_hom_two_photon_beamsplitter_pnr_and_click():
    circuit = PhotonicCircuit(
        n_modes=6, phases=random_phases(6, seed=9), circuit_config=coincidence_cfg(6)
    )
    u = np.eye(6, dtype=np.complex128)
    bs = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    u[np.ix_([0, 1], [0, 1])] = bs

    input_state = (1, 1, 0, 0, 0, 0)
    probs_pnr = circuit.coincidences(input_state, detector_mode="pnr", unitary=u)
    probs_click = circuit.coincidences(input_state, detector_mode="click", unitary=u)

    assert np.isclose(probs_pnr[(2, 0, 0, 0, 0, 0)], 0.5, atol=1e-12)
    assert np.isclose(probs_pnr[(0, 2, 0, 0, 0, 0)], 0.5, atol=1e-12)
    assert np.isclose(probs_click[(1, 1, 0, 0, 0, 0)], 0.0, atol=1e-12)


def test_three_photon_pnr_normalization_random_unitary():
    circuit = PhotonicCircuit(
        n_modes=6, phases=random_phases(6, seed=10), circuit_config=coincidence_cfg(6)
    )
    u = random_unitary(6, seed=11)
    input_state = (1, 1, 1, 0, 0, 0)

    probs_pnr = circuit.coincidences(input_state, detector_mode="pnr", unitary=u)
    assert np.isclose(sum(probs_pnr.values()), 1.0, atol=1e-10)
    assert all(p >= 0 for p in probs_pnr.values())


def test_click_subset_of_pnr_for_three_photon_case():
    circuit = PhotonicCircuit(
        n_modes=6, phases=random_phases(6, seed=12), circuit_config=coincidence_cfg(6)
    )
    u = random_unitary(6, seed=13)
    input_state = (1, 1, 1, 0, 0, 0)

    probs_pnr = circuit.coincidences(input_state, detector_mode="pnr", unitary=u)
    probs_click = circuit.coincidences(input_state, detector_mode="click", unitary=u)

    for state, p in probs_click.items():
        assert max(state) <= 1
        assert np.isclose(p, probs_pnr[state], atol=1e-12)
    assert any(max(state) > 1 for state in probs_pnr.keys())
    assert not any(max(state) > 1 for state in probs_click.keys())
