import pennylane as qml
from pennylane import numpy as np
import strawberryfields.ops as sfops

# def build_circuit(phase1, memristor_weight, phase3, encoded_phases):
#     """
#     Constructs the quantum circuit with the given parameters using PennyLane Strawberry Fields plugin.
#     """
dev = qml.device('strawberryfields.fock', wires=3, cutoff_dim=4)
    
@qml.qnode(dev)
def circuit(phase1, memristor_weight, phase3, encoded_phases):
    # Initial state preparation
    qml.FockState(0, wires=0)  # |0⟩ state for mode 0 (Vacuum)
    qml.FockState(1, wires=1)  # |1⟩ state for mode 1 (Single photon)
    qml.FockState(0, wires=2)  # |0⟩ state for mode 2 (Vacuum)
    
    # Input encoding MZI
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[0, 1])
    qml.Rotation(encoded_phases, wires=1)
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[0, 1])
    
    # First MZI
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[0, 1])
    qml.Rotation(phase1, wires=1)
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[0, 1])
    
    # Memristor (Second MZI)
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[1, 2])
    qml.Rotation(memristor_weight, wires=1)
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[1, 2])
    
    # Third MZI
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[0, 1])
    qml.Rotation(phase3, wires=1)
    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[0, 1])
    
    return qml.expval(qml.NumberOperator(0))

    # return circuit(phase1, memristor_weight, phase3, encoded_phases)

def run_circuit():
    # Example parameters
    params = {
        'phase1': 0.1,
        'memristor_weight': 0.5,
        'phase3': 0.3,
        'encoded_phases': 0.2
    }
    
    try:
        # Run the circuit
        result = circuit(
            params['phase1'],
            params['memristor_weight'],
            params['phase3'],
            params['encoded_phases']
        )
        
        print("Circuit output probabilities:", result)
        
        # Print circuit diagram
        print("\nCircuit diagram:")
        print(qml.draw(circuit)(
            params['phase1'],
            params['memristor_weight'],
            params['phase3'],
            params['encoded_phases']
        ))
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise e

if __name__ == "__main__":
    run_circuit()