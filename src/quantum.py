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
    

class MemristorMegaCircuit:
    def __init__(self, phase1, memristor_weight, phase3, phase4, memristor2_weight, phase6, encoded_phases):
        self.phase1 = phase1
        self.memristor_weight = memristor_weight
        self.phase3 = phase3
        self.phase4 = phase4
        self.memristor2_weight = memristor2_weight
        self.phase6 = phase6
        self.encoded_phases = encoded_phases

    def set_phase1(self, phase1):
        self.phase1 = phase1

    def set_memristor_weight(self, memristor_weight):
        self.memristor_weight = memristor_weight

    def set_phase3(self, phase3):
        self.phase3 = phase3
    
    def set_phase4(self, phase4):
        self.phase4 = phase4

    def set_memristor2_weight(self, memristor2_weight):
        self.memristor2_weight = memristor2_weight

    def set_phase6(self, phase6):
        self.phase6 = phase6

    def set_encoded_phases(self, encoded_phases):
        self.encoded_phases = encoded_phases

    def build_circuit(self):
        """
        Constructs the longer quantum circuit with the given parameters.
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
        
            # First Memristor (Second MZI)
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.memristor_weight)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
            # Third MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase3)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

            # Fourth MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase4)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
            # Second Memristor (Fifth MZI)
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.memristor2_weight)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
            # Sixth MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase6)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

        return circuit