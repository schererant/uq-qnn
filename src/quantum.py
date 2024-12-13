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
    

class MemristorMegaBigCircuit:
    def __init__(self, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, encoded_phases):
        self.phase1 = phase1
        self.phase2 = phase2
        self.phase3 = phase3
        self.phase4 = phase4
        self.phase5 = phase5
        self.phase6 = phase6
        self.phase7 = phase7
        self.phase8 = phase8
        self.phase9 = phase9
        self.phase10 = phase10
        self.phase11 = phase11
        self.phase12 = phase12
        self.encoded_phases = encoded_phases

    def set_phase1(self, phase1):
        self.phase1 = phase1

    def set_phase2(self, phase2):
        self.phase2 = phase2

    def set_phase3(self, phase3):
        self.phase3 = phase3
    
    def set_phase4(self, phase4):
        self.phase4 = phase4

    def set_phase5(self, phase5):
        self.phase5 = phase5
    
    def set_phase6(self, phase6):
        self.phase6 = phase6

    def set_phase7(self, phase7):
        self.phase7 = phase7
    
    def set_phase8(self, phase8):
        self.phase8 = phase8

    def set_phase9(self, phase9):
        self.phase9 = phase9

    def set_phase10(self, phase10):
        self.phase10 = phase10
    
    def set_phase11(self, phase11):
        self.phase11 = phase11

    def set_phase12(self, phase12):
        self.phase12 = phase12

    def set_encoded_phases(self, encoded_phases):
        self.encoded_phases = encoded_phases

    def build_circuit(self):
        """
        Constructs the longer quantum circuit with the given parameters.
        """
        circuit = sf.Program(6)
        with circuit.context as q:
            Vac     | q[0]
            Vac     | q[1]
            Fock(1) | q[2]
            Vac     | q[3]
            Vac     | q[4]
            Vac     | q[5]
        
            # Input encoding MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            Rgate(self.encoded_phases)           | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
        
            # First MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])
            Rgate(self.phase1)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])

            # Second MZI
            BSgate(np.pi/4, np.pi/2) | (q[4], q[2])
            Rgate(self.phase2)             | q[4]
            BSgate(np.pi/4, np.pi/2) | (q[4], q[2])
        
            # Third MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])
            Rgate(self.phase3)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])

            # Fourth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            Rgate(self.phase4)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
        
            # Fith MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])
            Rgate(self.phase5)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])

            # Sixth MZI
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.phase6)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])

            # Seventh MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])
            Rgate(self.phase7)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])

            # Eith MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])
            Rgate(self.phase8)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])

            # Ninth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            Rgate(self.phase9)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])

            # Tenth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])
            Rgate(self.phase10)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])

            # Eleventh MZI
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.phase11)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])

            # Twelth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])
            Rgate(self.phase12)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])

        return circuit