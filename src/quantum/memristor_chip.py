import strawberryfields as sf
from strawberryfields.ops import *

def build_circuit(phase1, memristor_weight, phase3, encoded_phases):
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
        Rgate(encoded_phases)           | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
        # First MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(phase1)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
        # Memristor (Second MZI)
        BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        Rgate(memristor_weight)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
        # Third MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(phase3)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
    return circuit