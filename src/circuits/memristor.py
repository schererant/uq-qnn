import perceval as pcvl
from typing import Optional
from circuits.base import CircuitBase
from circuits.encoding import EncodingCircuit

class MemristorCircuit(CircuitBase):
    """
    Three-mode memristor circuit with phase shifters and beam splitters.
    The circuit consists of three sections, each with:
    1. Beam splitter
    2. Phase shifter
    3. Beam splitter
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize memristor circuit.
        
        Args:
            name (str, optional): Name of the circuit
        """
        super().__init__(name or "Memristor")
        
    def build(self, phi1: float, mem_phi: float, phi3: float) -> pcvl.Circuit:
        """
        Build the memristor circuit.
        
        Args:
            phi1 (float): Phase for the first phase shifter
            mem_phi (float): Phase for the memristor phase shifter
            phi3 (float): Phase for the third phase shifter
            
        Returns:
            pcvl.Circuit: The constructed memristor circuit
        """
        c = pcvl.Circuit(3, name=self.name)
        
        # First section (modes 0,1)
        c.add((0, 1), pcvl.BS())
        c.add((1,), pcvl.PS(phi=phi1))
        c.add((0, 1), pcvl.BS())
        
        # Second section (modes 1,2)
        c.add((1, 2), pcvl.BS())
        c.add((2,), pcvl.PS(phi=mem_phi))
        c.add((1, 2), pcvl.BS())
        
        # Third section (modes 0,1)
        c.add((0, 1), pcvl.BS())
        c.add((1,), pcvl.PS(phi=phi3))
        c.add((0, 1), pcvl.BS())
        
        return c

class FullCircuit(CircuitBase):
    """
    Complete circuit combining encoding and memristor circuits.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize full circuit.
        
        Args:
            name (str, optional): Name of the circuit
        """
        super().__init__(name or "Full")
        self.encoding = EncodingCircuit()
        self.memristor = MemristorCircuit()
        
    def build(self, phi1: float, mem_phi: float, phi3: float, enc_phi: float) -> pcvl.Circuit:
        """
        Build the full circuit by combining encoding and memristor circuits.
        
        Args:
            phi1 (float): Phase for the first memristor phase shifter
            mem_phi (float): Phase for the memristor phase shifter
            phi3 (float): Phase for the third memristor phase shifter
            enc_phi (float): Phase for the encoding circuit
            
        Returns:
            pcvl.Circuit: The complete circuit
        """
        c = pcvl.Circuit(3, name=self.name)
        c.add(0, self.encoding.build(enc_phi))
        c.add(0, self.memristor.build(phi1, mem_phi, phi3))
        return c 