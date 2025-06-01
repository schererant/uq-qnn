#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify that all modules can be properly imported.
This ensures that the modular structure is working correctly.
"""

import sys
import os
import unittest

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_package_import(self):
        """Test that the package can be imported."""
        import src
        self.assertIsNotNone(src)
        self.assertIsNotNone(src.__version__)
    
    def test_module_imports(self):
        """Test that all modules can be imported."""
        from src import autograd
        from src import circuits
        from src import data
        from src import loss
        from src import simulation
        from src import training
        from src import utils
        
        self.assertIsNotNone(autograd)
        self.assertIsNotNone(circuits)
        self.assertIsNotNone(data)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(simulation)
        self.assertIsNotNone(training)
        self.assertIsNotNone(utils)
    
    def test_function_imports(self):
        """Test that key functions can be imported."""
        from src.data import get_data, quartic_data
        from src.circuits import encoding_circuit, memristor_circuit, build_circuit
        from src.simulation import run_simulation_sequence_np, SimulationLogger
        from src.autograd import photonic_psr_coeffs_torch, MemristorLossPSR
        from src.loss import PhotonicModel
        from src.training import train_pytorch, train_pytorch_generic
        from src.utils import config
        
        self.assertIsNotNone(get_data)
        self.assertIsNotNone(quartic_data)
        self.assertIsNotNone(encoding_circuit)
        self.assertIsNotNone(memristor_circuit)
        self.assertIsNotNone(build_circuit)
        self.assertIsNotNone(run_simulation_sequence_np)
        self.assertIsNotNone(SimulationLogger)
        self.assertIsNotNone(photonic_psr_coeffs_torch)
        self.assertIsNotNone(MemristorLossPSR)
        self.assertIsNotNone(PhotonicModel)
        self.assertIsNotNone(train_pytorch)
        self.assertIsNotNone(train_pytorch_generic)
        self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()