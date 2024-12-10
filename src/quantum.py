import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

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