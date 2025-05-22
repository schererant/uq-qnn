import perceval as pcvl
from typing import Optional
from circuits.base import CircuitBase

class EncodingCircuit(CircuitBase):
    """
    Two-mode encoding circuit with a phase shifter.
    The circuit consists of:
    1. Beam splitter
    2. Phase shifter on mode 1
    3. Beam splitter
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize encoding circuit.
        
        Args:
            name (str, optional): Name of the circuit
        """
        super().__init__(name or "Encoding")
        
    def build(self, encoded_phase: float) -> pcvl.Circuit:
        """
        Build the encoding circuit.
        
        Args:
            encoded_phase (float): Phase to encode (in radians)
            
        Returns:
            pcvl.Circuit: The constructed encoding circuit
        """
        c = pcvl.Circuit(2, name=self.name)
        # First beam splitter
        c.add((0, 1), pcvl.BS())
        # Phase shifter on mode 1
        c.add((1,), pcvl.PS(phi=encoded_phase))
        # Second beam splitter
        c.add((0, 1), pcvl.BS())
        return c 