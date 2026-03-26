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
        encoding_phase_idx=None,
        memristive_phase_idx=None,
        memristive_output_modes=None,
        memory_depth=2,
        n_swipe=0,
        swipe_span=0.0,
        noise_std=None,
        n_photons=None,
        loss_type="mse",
        n_classes=1,
        output_mode="singles",
        input_modes=None,
        working_detectors=None,
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
            encoding_mode=0,
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
            encoding_mode=0,
            target_mode=None,
            output_mode="coincidence",
            input_modes=(1, 4),
            working_detectors=(0, 1, 5),
            backend="numpy",
        )
        cfg_pc = cfg_np.replace(backend="perceval")
        a = run_simulation_sequence_np(params, enc, cfg_np)
        b = run_simulation_sequence_np(params, enc, cfg_pc)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_inline_encoding(self):
        n_modes = 4
        n_ph = n_modes * (n_modes - 1)
        rng = np.random.default_rng(2)
        params = rng.random(n_ph) * 2 * np.pi
        enc = np.linspace(0.1, 1.0, 6)
        cfg_np = _base_cfg(
            n_samples=25,
            n_modes=n_modes,
            encoding_mode=0,
            target_mode=(2,),
            encoding_phase_idx=5,
            backend="numpy",
        )
        cfg_pc = cfg_np.replace(backend="perceval")
        a = run_simulation_sequence_np(params, enc, cfg_np)
        b = run_simulation_sequence_np(params, enc, cfg_pc)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
