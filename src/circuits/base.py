from abc import ABC, abstractmethod
import perceval as pcvl
from typing import Optional, Tuple, Union, Sequence

class CircuitBase(ABC):
    """Base class for quantum circuits."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize circuit.
        
        Args:
            name (str, optional): Name of the circuit
        """
        self.name = name or self.__class__.__name__
        
    @abstractmethod
    def build(self, *args, **kwargs) -> pcvl.Circuit:
        """
        Build and return a Perceval circuit.
        
        Returns:
            pcvl.Circuit: The constructed quantum circuit
        """
        pass
    
    def get_input_state(self, photon_numbers: Sequence[int]) -> pcvl.BasicState:
        """
        Create an input state with given photon numbers.
        
        Args:
            photon_numbers (Sequence[int]): Number of photons in each mode
            
        Returns:
            pcvl.BasicState: The input quantum state
        """
        return pcvl.BasicState(list(photon_numbers))
    
    def get_processor(self, circuit: pcvl.Circuit, input_state: pcvl.BasicState) -> pcvl.Processor:
        """
        Create a processor for the circuit with given input state.
        
        Args:
            circuit (pcvl.Circuit): The quantum circuit
            input_state (pcvl.BasicState): The input state
            
        Returns:
            pcvl.Processor: Processor ready to run the circuit
        """
        proc = pcvl.Processor("SLOS", circuit)
        proc.with_input(input_state)
        return proc 