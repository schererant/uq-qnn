import unittest
import strawberryfields as sf
import numpy as np

from src.quantum import MemristorCircuit  # Relative import of the class to test

class TestMemristorCircuit(unittest.TestCase):
    def setUp(self):
        self.phase1 = 0.5
        self.memristor_weight = 0.8
        self.phase3 = 0.3
        self.encoded_phases = 0.2
        self.circuit = MemristorCircuit(self.phase1, self.memristor_weight, self.phase3, self.encoded_phases)

    def test_initialization(self):
        """Test that the initial parameters are set correctly."""
        self.assertEqual(self.circuit.phase1, self.phase1)
        self.assertEqual(self.circuit.memristor_weight, self.memristor_weight)
        self.assertEqual(self.circuit.phase3, self.phase3)
        self.assertEqual(self.circuit.encoded_phases, self.encoded_phases)

    def test_set_phase1(self):
        """Test the set_phase1 method."""
        new_phase1 = 0.9
        self.circuit.set_phase1(new_phase1)
        self.assertEqual(self.circuit.phase1, new_phase1)

    def test_set_memristor_weight(self):
        """Test the set_memristor_weight method."""
        new_weight = 0.7
        self.circuit.set_memristor_weight(new_weight)
        self.assertEqual(self.circuit.memristor_weight, new_weight)

    def test_set_phase3(self):
        """Test the set_phase3 method."""
        new_phase3 = 0.6
        self.circuit.set_phase3(new_phase3)
        self.assertEqual(self.circuit.phase3, new_phase3)

    def test_set_encoded_phases(self):
        """Test the set_encoded_phases method."""
        new_encoded_phases = 0.4
        self.circuit.set_encoded_phases(new_encoded_phases)
        self.assertEqual(self.circuit.encoded_phases, new_encoded_phases)

    def test_build_circuit(self):
        """Test that build_circuit returns a valid Program object."""
        program = self.circuit.build_circuit()
        self.assertIsInstance(program, sf.Program)
        # Check that the program has the correct number of subsystems
        self.assertEqual(program.num_subsystems, 3)
        # # Optionally, check the number of operations in the circuit
        # expected_operations = 3 + 3 * 6  # Initial states + 3 MZIs with 6 operations each
        # self.assertEqual(len(program.circuit), expected_operations)

if __name__ == '__main__':
    unittest.main()