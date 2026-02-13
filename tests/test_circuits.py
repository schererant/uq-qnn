#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the circuit architectures implementation.
This tests both Memristor and Clements architectures for correctness.
"""

import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.circuits import (
    encoding_circuit,
    mzi_unit,
    memristor_circuit,
    clements_circuit,
    build_circuit,
    get_mzi_modes_for_phase
)


class TestCircuitArchitectures(unittest.TestCase):
    """Test cases for circuit architectures."""
    
    def test_encoding_circuit(self):
        """Test the encoding circuit."""
        # Basic test - should not raise exceptions
        circ = encoding_circuit(np.pi/4)
        self.assertEqual(circ.m, 2)  # Should be 2-mode
        
        # Edge cases
        circ_zero = encoding_circuit(0)
        circ_large = encoding_circuit(10 * np.pi)  # Should handle large values
        self.assertEqual(circ_zero.m, 2)
        self.assertEqual(circ_large.m, 2)
    
    def test_mzi_unit(self):
        """Test the MZI unit circuit."""
        # Basic test
        circ = mzi_unit((0, 1), np.pi/4, np.pi/2)
        self.assertEqual(circ.m, 2)
        
        # Test with consecutive modes - note: MZI can only operate on consecutive modes
        circ_high = mzi_unit((3, 4), np.pi/3, np.pi/6)
        self.assertEqual(circ_high.m, 5)
        
        # Test with negative phases - should be normalized
        circ_neg = mzi_unit((0, 1), -np.pi/4, -np.pi/2)
        self.assertEqual(circ_neg.m, 2)
    
    def test_memristor_circuit(self):
        """Test the memristor circuit."""
        # Basic test
        phases = np.array([np.pi/4, np.pi/3, np.pi/2])
        circ = memristor_circuit(phases)
        self.assertEqual(circ.m, 3)
        
        # Test with different phases
        phases2 = np.array([0, np.pi, 2*np.pi])
        circ2 = memristor_circuit(phases2)
        self.assertEqual(circ2.m, 3)
    
    def test_clements_circuit(self):
        """Test the Clements circuit with different numbers of modes."""
        # 3-mode test
        n_modes = 3
        n_phases = n_modes * (n_modes - 1)
        phases = np.ones(n_phases) * np.pi/4
        circ = clements_circuit(phases, n_modes)
        self.assertEqual(circ.m, n_modes)
        
        # 4-mode test
        n_modes = 4
        n_phases = n_modes * (n_modes - 1)
        phases = np.ones(n_phases) * np.pi/3
        circ = clements_circuit(phases, n_modes)
        self.assertEqual(circ.m, n_modes)
        
        # Test error cases
        with self.assertRaises(ValueError):
            # Not enough phases
            clements_circuit(np.array([0.1, 0.2]), 3)
        
        with self.assertRaises(ValueError):
            # Too few modes
            clements_circuit(np.array([0.1, 0.2]), 1)
    
    def test_build_circuit(self):
        """Test building Clements circuits (3x3, 6x6, etc.)."""
        n_modes = 3
        n_phases = n_modes * (n_modes - 1)
        phases = np.ones(n_phases) * np.pi/4
        enc_phi = np.pi/6

        circ = build_circuit(phases, enc_phi, n_modes=n_modes)
        self.assertEqual(circ.m, n_modes)

        circ2 = build_circuit(phases, enc_phi, n_modes=n_modes, encoding_mode=1)
        self.assertEqual(circ2.m, n_modes)

        n_modes = 5
        n_phases = n_modes * (n_modes - 1)
        phases = np.ones(n_phases) * np.pi/4
        circ3 = build_circuit(phases, enc_phi, n_modes=n_modes)
        self.assertEqual(circ3.m, n_modes)

        with self.assertRaises(ValueError):
            build_circuit(phases, enc_phi, n_modes=1)

    def test_get_mzi_modes_for_phase(self):
        """Test phase index to MZI mode mapping."""
        # 3-mode Clements: MZI (0,1) phases 0,1; (1,2) phases 2,3; (0,1) phases 4,5
        self.assertEqual(get_mzi_modes_for_phase(0, 3), (0, 1))
        self.assertEqual(get_mzi_modes_for_phase(1, 3), (0, 1))
        self.assertEqual(get_mzi_modes_for_phase(2, 3), (1, 2))
        self.assertEqual(get_mzi_modes_for_phase(3, 3), (1, 2))
        self.assertEqual(get_mzi_modes_for_phase(4, 3), (0, 1))
        self.assertEqual(get_mzi_modes_for_phase(5, 3), (0, 1))
        with self.assertRaises(ValueError):
            get_mzi_modes_for_phase(6, 3)
        with self.assertRaises(ValueError):
            get_mzi_modes_for_phase(-1, 3)

    def test_multiple_memristive_mzis(self):
        """Test simulation with multiple memristive MZIs."""
        from src.simulation import run_simulation_sequence_np

        n_modes = 3
        n_phases = n_modes * (n_modes - 1)
        memristive_phase_idx = (2, 5)
        params = np.concatenate([
            np.ones(n_phases) * np.pi / 4,
            np.array([0.5, 0.5])
        ])
        enc = np.linspace(0, np.pi, 5)
        preds = run_simulation_sequence_np(
            params, memory_depth=2, n_samples=100,
            encoded_phases=enc,
            n_modes=n_modes,
            memristive_phase_idx=memristive_phase_idx
        )
        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all(preds >= 0) and np.all(preds <= 1))

    def test_no_memristive(self):
        """Test simulation with no memristive phases (standard Clements)."""
        from src.simulation import run_simulation_sequence_np

        n_modes = 3
        n_phases = n_modes * (n_modes - 1)
        params = np.ones(n_phases) * np.pi / 4
        enc = np.linspace(0, np.pi, 5)
        preds = run_simulation_sequence_np(
            params, memory_depth=2, n_samples=100,
            encoded_phases=enc,
            n_modes=n_modes,
            memristive_phase_idx=None
        )
        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all(preds >= 0) and np.all(preds <= 1))


if __name__ == "__main__":
    unittest.main()