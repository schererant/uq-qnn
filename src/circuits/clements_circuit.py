
import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

class ClementsCircuit:
    def __init__(self, size, input_fock, phases, encoding_phase_3x3=None):
        """
        Parameters:
            size (int): Number of modes (e.g., 3 for a 3x3 circuit).
            input_fock (List[int]): List of length `size`, each entry is 1 or 0 (Fock state 1 or vacuum).
            phases (List[float]): List of phases per layer per mode as needed by the circuit.
        """
        self.size = size
        self.input_fock = input_fock
        self.phases = phases

    def build_circuit(self):
        prog = sf.Program(self.size)

        with prog.context as q:
            # Initialize the input Fock states
            for i, val in enumerate(self.input_fock):
                if val == 1:
                    Fock(1) | q[i]
                else:
                    Vac     | q[i]
                    
            # Apply encoding phase if provided (only for 3x3 circuits)
            if self.encoding_phase_3x3 is not None:
                BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
                Rgate(self.encoding_phase_3x3) | q[0]
                BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            

            # Construct a full Clements decomposition layer
            phase_idx = 0
            for i in range(self.size):
                for j in range(i % 2, self.size - 1, 2):
                    BSgate(np.pi/4, np.pi/2) | (q[j], q[j+1])
                    Rgate(self.phases[phase_idx]) | q[j+1]
                    BSgate(np.pi/4, np.pi/2) | (q[j], q[j+1])
                    phase_idx += 1

        return prog
    
    
