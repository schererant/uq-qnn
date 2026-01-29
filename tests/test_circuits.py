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
    CircuitType, 
    encoding_circuit, 
    mzi_unit, 
    memristor_circuit, 
    clements_circuit, 
    build_circuit
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
    
    def test_build_circuit_memristor(self):
        """Test building a complete memristor circuit."""
        phases = np.array([np.pi/4, np.pi/3, np.pi/2])
        enc_phi = np.pi/6
        
        # Basic test
        circ = build_circuit(phases, enc_phi, circuit_type=CircuitType.MEMRISTOR)
        self.assertEqual(circ.m, 3)
        
        # Test with encoding mode
        circ2 = build_circuit(phases, enc_phi, circuit_type=CircuitType.MEMRISTOR, encoding_mode=1)
        self.assertEqual(circ2.m, 3)
        
        # Test with invalid encoding mode - should not crash, will use mode 1 (valid)
        circ3 = build_circuit(phases, enc_phi, circuit_type=CircuitType.MEMRISTOR, encoding_mode=10)
        self.assertEqual(circ3.m, 3)
    
    def test_build_circuit_clements(self):
        """Test building a complete Clements circuit."""
        n_modes = 3
        n_phases = n_modes * (n_modes - 1)
        phases = np.ones(n_phases) * np.pi/4
        enc_phi = np.pi/6
        
        # Basic test
        circ = build_circuit(
            phases, enc_phi, 
            circuit_type=CircuitType.CLEMENTS, 
            n_modes=n_modes
        )
        self.assertEqual(circ.m, n_modes)
        
        # Test with different encoding mode
        circ2 = build_circuit(
            phases, enc_phi, 
            circuit_type=CircuitType.CLEMENTS, 
            n_modes=n_modes, 
            encoding_mode=1
        )
        self.assertEqual(circ2.m, n_modes)
        
        # Test with larger circuit
        n_modes = 5
        n_phases = n_modes * (n_modes - 1)
        phases = np.ones(n_phases) * np.pi/4
        circ3 = build_circuit(
            phases, enc_phi, 
            circuit_type=CircuitType.CLEMENTS, 
            n_modes=n_modes
        )
        self.assertEqual(circ3.m, n_modes)
        
        # Test error cases
        with self.assertRaises(ValueError):
            # Too few modes
            build_circuit(
                phases, enc_phi, 
                circuit_type=CircuitType.CLEMENTS, 
                n_modes=1
            )


if __name__ == "__main__":
    unittest.main()