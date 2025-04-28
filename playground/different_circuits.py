import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from abc import ABC, abstractmethod

class MemristorCircuit:
    def __init__(self, phase1, memristor_weight, phase3, encoded_phases):
        self.phase1 = phase1
        self.memristor_weight = memristor_weight
        self.phase3 = phase3
        self.encoded_phases = encoded_phases

    def set_phase1(self, phase1):
        self.phase1 = phase1

    def set_memristor_weight(self, memristor_weight):
        self.memristor_weight = memristor_weight

    def set_phase3(self, phase3):
        self.phase3 = phase3

    def set_encoded_phases(self, encoded_phases):
        self.encoded_phases = encoded_phases

    def build_circuit(self):
        """
        Constructs the quantum circuit with the given parameters.
        """
        circuit = sf.Program(3)
        with circuit.context as q:
            Vac     | q[0]
            Fock(1) | q[1]
            Vac     | q[2]

            # Input encoding MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.encoded_phases)           | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

            # First MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase1)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

            # Memristor (Second MZI)
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.memristor_weight)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])

            # Third MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase3)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        return circuit
    

class MemristorCircuitReviewed:
    """
    Implements a photonic quantum circuit composed of Mach-Zehnder Interferometers (MZIs),
    including a memristive element.

    Parameters
    ----------
    encoded_phase : float
        Phase shift encoding the input signal (radians).
    phase1 : float
        Phase shift applied in the first MZI (radians).
    memristor_phase : float
        Phase shift applied in the memristor emulation MZI (radians).
    phase3 : float
        Phase shift applied in the third MZI (radians).
    """
    def __init__(self, encoded_phase, phase1, memristor_phase, phase3):
        self.encoded_phase = encoded_phase
        self.phase1 = phase1
        self.memristor_phase = memristor_phase
        self.phase3 = phase3

    def _mzi(self, q, m1, m2, phase):
        """Applies an MZI with phase rotation to modes m1 and m2."""
        BSgate(np.pi / 4, np.pi / 2) | (q[m1], q[m2])
        Rgate(phase)                 | q[m2]
        BSgate(np.pi / 4, np.pi / 2) | (q[m1], q[m2])

    def build_circuit(self):

        circuit = sf.Program(3)
        with circuit.context as q:
            Vac     | q[0]
            Fock(1) | q[1]
            Vac     | q[2]

            self._mzi(q, 0, 1, self.encoded_phase)
            self._mzi(q, 0, 1, self.phase1)
            self._mzi(q, 1, 2, self.memristor_phase)
            self._mzi(q, 0, 1, self.phase3)

        return circuit
