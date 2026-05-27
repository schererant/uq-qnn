"""Reference-case checks aligned with examples/coincidence_regression.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import SimConfig
from src.numpy_backend import clements_unitary, run_vectorized_non_memristive, slos_2photon_numpy
from src.simulation import run_simulation_sequence


def _coincidence_regression_sim_config() -> SimConfig:
    """Subset of CONFIG from examples/coincidence_regression.py (NumPy coincidence)."""

    return SimConfig.from_experiment_config(
        {
            "n_modes": 6,
            "input_state": tuple(1 if i in (0, 3) else 0 for i in range(6)),
            "encoding_phase_idx": 7,
            "photon_distinguishability": "indistinguishable",
            "target_mode": (0, 1),
            "memristive_phase_idx": None,
            "memristive_output_modes": None,
            "output_mode": "coincidence",
            "working_detectors": tuple(range(6)),
            "loss_type": "mse",
            "n_classes": 1,
            "n_samples": 1000,
            "memory_depth": 2,
            "n_swipe": 0,
            "swipe_span": 0.0,
            "noise_std": None,
            "sim_backend": "numpy",
        }
    )


def test_coincidence_regression_numpy_predictions_match_vectorized_path() -> None:
    cfg = _coincidence_regression_sim_config()
    theta = np.linspace(0.05, 1.2, 30)
    enc = np.array([0.1, 0.4, 0.9])

    via_runner = run_simulation_sequence(theta, enc, cfg)
    via_vec = run_vectorized_non_memristive(theta, enc, cfg)

    np.testing.assert_allclose(via_runner, via_vec, rtol=0, atol=1e-12)


def test_coincidence_regression_numpy_golden_values() -> None:
    cfg = _coincidence_regression_sim_config()
    theta = np.linspace(0.05, 1.2, 30)
    enc = np.array([0.1, 0.4, 0.9])

    expected = np.array([0.03396306, 0.03566954, 0.03737525])
    got = run_simulation_sequence(theta, enc, cfg)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)


def test_sim_config_sim_backend_alias() -> None:
    cfg = _coincidence_regression_sim_config()
    assert cfg.sim_backend == cfg.backend == "numpy"


def test_slos_2photon_numpy_distribution_sums_to_one() -> None:
    rng = np.random.default_rng(123)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=30)
    u = clements_unitary(phases, n_modes=6)

    dist = slos_2photon_numpy(u, input_modes=(2, 3))

    assert len(dist) == 21
    np.testing.assert_allclose(sum(dist.values()), 1.0, rtol=0, atol=1e-10)


def test_slos_2photon_numpy_matches_perceval_for_random_unitary() -> None:
    pcvl = pytest.importorskip("perceval")
    from perceval.algorithm import Sampler

    from src.circuits import build_circuit

    rng = np.random.default_rng(456)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=30)
    u = clements_unitary(phases, n_modes=6)
    dist_np = slos_2photon_numpy(u, input_modes=(2, 3))

    circ = build_circuit(
        phases=phases,
        enc_phi=0.0,
        n_modes=6,
        encoding_phase_idx=7,
    )
    proc = pcvl.Processor("SLOS", circ)
    proc.with_input(pcvl.BasicState([0, 0, 1, 1, 0, 0]))
    dist_pv_raw = Sampler(proc).probs(1_000)["results"]
    dist_pv = {tuple(int(x) for x in st): float(p) for st, p in dist_pv_raw.items()}

    assert len(dist_pv) == 21
    for state, p_np in dist_np.items():
        np.testing.assert_allclose(p_np, dist_pv.get(state, 0.0), rtol=0, atol=1e-6)


def test_slos_2photon_numpy_identity_unitary_behavior() -> None:
    u = np.eye(6, dtype=np.complex128)
    dist = slos_2photon_numpy(u, input_modes=(2, 3))

    expected = (0, 0, 1, 1, 0, 0)
    for state, prob in dist.items():
        if state == expected:
            np.testing.assert_allclose(prob, 1.0, rtol=0, atol=1e-12)
        else:
            np.testing.assert_allclose(prob, 0.0, rtol=0, atol=1e-12)
