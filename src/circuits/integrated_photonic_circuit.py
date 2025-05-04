from abc import ABC, abstractmethod

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np


class IntegratedPhotonicCircuit(ABC):
    """Abstract base class for integrated photonic circuits."""
    
    def __init__(self, size, input_fock, phases):
        """
        Parameters:
            size (int): Number of modes (e.g., 3 for a 3x3 circuit).
            input_fock (List[int]): List of length `size`, each entry is 1 or 0 (Fock state 1 or vacuum).
            phases (List[float]): List of phases per layer per mode as needed by the circuit.
        """
        self.size = size
        self.input_fock = input_fock
        self.phases = phases
    
    @abstractmethod
    def build_circuit(self):
        """Build the quantum circuit. Must be implemented by child classes."""
        pass
    
    def _add_mzi(self, q, input_mode, target_mode, phase):
        """Helper method to add a Mach-Zehnder interferometer to a circuit.
        
        Parameters:
            q: Quantum register
            input_mode (int): First mode index
            target_mode (int): Second mode index
            phase (float): Phase to apply in the MZI
        """
        BSgate(np.pi/4, np.pi/2) | (q[input_mode], q[target_mode])
        Rgate(phase) | q[target_mode]
        BSgate(np.pi/4, np.pi/2) | (q[input_mode], q[target_mode])


class ClementsIPC(IntegratedPhotonicCircuit):
    def __init__(self, size, input_fock, phases):
        """
        Parameters:
            size (int): Number of modes (e.g., 3 for a 3x3 circuit).
            input_fock (List[int]): List of length `size`, each entry is 1 or 0 (Fock state 1 or vacuum).
            phases (List[float]): List of phases per layer per mode as needed by the circuit.
        """
        # check if size is greater than 1
        if size <= 1:
            raise ValueError("Size must be greater than 1")
        
        # check if input_fock is a list of length size
        if len(input_fock) != size:
            raise ValueError("Input fock must be a list of length size")
        
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
            
            # Construct a full Clements decomposition layer
            phase_idx = 0
            for i in range(self.size):
                for j in range(i % 2, self.size - 1, 2):
                    BSgate(np.pi/4, np.pi/2) | (q[j], q[j+1])
                    Rgate(self.phases[phase_idx]) | q[j+1]
                    BSgate(np.pi/4, np.pi/2) | (q[j], q[j+1])
                    phase_idx += 1

        return prog
    

circuit = ClementsIPC(3, [1, 0, 1], [0, 0, 0])
print(circuit.build_circuit())  


    