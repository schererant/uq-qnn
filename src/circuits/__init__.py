from typing import Optional
from circuits.base import CircuitBase
from circuits.encoding import EncodingCircuit
from circuits.memristor import MemristorCircuit, FullCircuit

def get_circuit(name: str, circuit_name: Optional[str] = None) -> CircuitBase:
    """
    Factory function to get a circuit by name.
    
    Args:
        name (str): Name of the circuit type ('encoding', 'memristor', 'full')
        circuit_name (str, optional): Custom name for the circuit instance
        
    Returns:
        CircuitBase: The requested circuit instance
    """
    if name == 'encoding':
        return EncodingCircuit(name=circuit_name)
    elif name == 'memristor':
        return MemristorCircuit(name=circuit_name)
    elif name == 'full':
        return FullCircuit(name=circuit_name)
    else:
        raise ValueError(f"Unknown circuit type: {name}")

__all__ = ['CircuitBase', 'EncodingCircuit', 'MemristorCircuit', 'FullCircuit', 'get_circuit'] 