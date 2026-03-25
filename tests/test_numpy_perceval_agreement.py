#!/usr/bin/env python3
"""Check NumPy and Perceval simulation backends agree within numerical tolerance."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import run_simulation_sequence_np


class TestNumpyPercevalAgreement(unittest.TestCase):
    def test_singles_6_modes(self):
        n_modes = 6
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(0)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.1, 1.0, 12)
        kw = dict(
            memory_depth=2,
            n_samples=30,
            encoded_phases=enc,
            n_swipe=0,
            swipe_span=0.0,
            n_modes=n_modes,
            encoding_mode=0,
            target_mode=(5,),
            memristive_phase_idx=None,
            memristive_output_modes=None,
            encoding_phase_idx=None,
        )
        a = run_simulation_sequence_np(params, backend="numpy", **kw)
        b = run_simulation_sequence_np(params, backend="perceval", **kw)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_coincidence_6_modes(self):
        n_modes = 6
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(1)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.2, 0.9, 8)
        kw = dict(
            memory_depth=2,
            n_samples=40,
            encoded_phases=enc,
            n_swipe=0,
            swipe_span=0.0,
            n_modes=n_modes,
            encoding_mode=0,
            target_mode=None,
            memristive_phase_idx=None,
            memristive_output_modes=None,
            encoding_phase_idx=None,
            output_mode="coincidence",
            input_modes=(1, 4),
            working_detectors=(0, 1, 5),
        )
        a = run_simulation_sequence_np(params, backend="numpy", **kw)
        b = run_simulation_sequence_np(params, backend="perceval", **kw)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_inline_encoding(self):
        n_modes = 4
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(2)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.1, 1.0, 6)
        kw = dict(
            memory_depth=2,
            n_samples=25,
            encoded_phases=enc,
            n_swipe=0,
            swipe_span=0.0,
            n_modes=n_modes,
            encoding_mode=0,
            target_mode=(2,),
            memristive_phase_idx=None,
            memristive_output_modes=None,
            encoding_phase_idx=5,
        )
        a = run_simulation_sequence_np(params, backend="numpy", **kw)
        b = run_simulation_sequence_np(params, backend="perceval", **kw)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
