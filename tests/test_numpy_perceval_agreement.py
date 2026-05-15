#!/usr/bin/env python3
"""Check NumPy and Perceval simulation backends agree within numerical tolerance."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimConfig
from src.simulation import run_simulation_sequence_np


def _base_cfg(**kwargs) -> SimConfig:
    common: dict = dict(
        n_layers=1,
        n_enc_features=None,
        memristive_phase_idx=None,
        memristive_output_modes=None,
        memory_depth=2,
        n_swipe=0,
        swipe_span=0.0,
        noise_std=None,
        loss_type="mse",
        n_classes=1,
        output_mode="singles",
        working_detectors=None,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )
    common.update(kwargs)
    return SimConfig(**common)


class TestNumpyPercevalAgreement(unittest.TestCase):
    def test_singles_6_modes(self):
        n_modes = 6
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(0)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.1, 1.0, 12)
        cfg_np = _base_cfg(
            n_samples=30,
            n_modes=n_modes,
            input_state=tuple(1 if i == 0 else 0 for i in range(n_modes)),
            encoding_phase_idx=0,
            photon_distinguishability=None,
            target_mode=(5,),
            backend="numpy",
        )
        cfg_pc = cfg_np.replace(backend="perceval")
        a = run_simulation_sequence_np(params, enc, cfg_np)
        b = run_simulation_sequence_np(params, enc, cfg_pc)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_coincidence_6_modes(self):
        n_modes = 6
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(1)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.2, 0.9, 8)
        cfg_np = _base_cfg(
            n_samples=40,
            n_modes=n_modes,
            input_state=tuple(1 if i in (1, 4) else 0 for i in range(n_modes)),
            encoding_phase_idx=10,
            photon_distinguishability="indistinguishable",
            target_mode=(0, 1),
            output_mode="coincidence",
            working_detectors=(0, 1, 5),
            backend="numpy",
        )
        cfg_pc = cfg_np.replace(backend="perceval")
        a = run_simulation_sequence_np(params, enc, cfg_np)
        b = run_simulation_sequence_np(params, enc, cfg_pc)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_coincidence_three_photons_four_modes(self):
        """N=3 coincidence (Ryser path) agrees between NumPy and Perceval."""
        n_modes = 4
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(3)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.15, 0.95, 5)
        inp = tuple(1 if i < 3 else 0 for i in range(n_modes))
        cfg_np = _base_cfg(
            n_samples=60,
            n_modes=n_modes,
            input_state=inp,
            encoding_phase_idx=5,
            photon_distinguishability="indistinguishable",
            target_mode=(0, 1, 2),
            output_mode="coincidence",
            working_detectors=tuple(range(n_modes)),
            backend="numpy",
        )
        cfg_pc = cfg_np.replace(backend="perceval")
        a = run_simulation_sequence_np(params, enc, cfg_np)
        b = run_simulation_sequence_np(params, enc, cfg_pc)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)

    def test_inline_encoding(self):
        n_modes = 4
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(2)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.1, 1.0, 6)
        cfg_np = _base_cfg(
            n_samples=25,
            n_modes=n_modes,
            input_state=tuple(1 if i == 0 else 0 for i in range(n_modes)),
            encoding_phase_idx=5,
            photon_distinguishability=None,
            target_mode=(2,),
            backend="numpy",
        )
        cfg_pc = cfg_np.replace(backend="perceval")
        a = run_simulation_sequence_np(params, enc, cfg_np)
        b = run_simulation_sequence_np(params, enc, cfg_pc)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_clements_unitary_batch_matches_serial(self):
        from src.numpy_backend import (
            _unitary_batch_internal_encoding,
            clements_unitary,
            clements_unitary_batch,
            unitary_for_point,
        )

        n_modes = 6
        rng = np.random.default_rng(42)
        n_data = 32
        n_ph = n_modes * (n_modes - 1)
        pb = rng.random((n_data, n_ph)) * 2 * np.pi
        ub = clements_unitary_batch(pb, n_modes)
        for i in range(n_data):
            np.testing.assert_allclose(
                ub[i], clements_unitary(pb[i], n_modes), atol=1e-12, rtol=1e-12
            )

        params = rng.random(n_ph) * 2 * np.pi
        enc = rng.random(n_data) * 2 * np.pi
        enc_idx = 7
        batched = _unitary_batch_internal_encoding(
            params, enc, n_modes, enc_idx
        )
        for i in range(n_data):
            np.testing.assert_allclose(
                batched[i],
                unitary_for_point(params, float(enc[i]), n_modes, enc_idx),
                atol=1e-12,
                rtol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
