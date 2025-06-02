from __future__ import annotations

import numpy as np
import perceval as pcvl
from enum import Enum
from typing import Optional, Tuple, List


class CircuitType(Enum):
    """Enumeration of supported circuit architectures."""
    MEMRISTOR = "memristor"
    CLEMENTS = "clements"


def encoding_circuit(encoded_phase: float) -> pcvl.Circuit:
    """
    Builds a 2-mode encoding circuit with a phase shifter.
    Args:
        encoded_phase (float): Phase to encode.
    Returns:
        pcvl.Circuit: The constructed encoding circuit.
    """
    c = pcvl.Circuit(2, name="Encoding")
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=encoded_phase))
    c.add((0, 1), pcvl.BS())
    return c


def mzi_unit(modes: Tuple[int, int], phi_int: float, phi_ext: float) -> pcvl.Circuit:
    """
    Creates a basic Mach-Zehnder Interferometer (MZI) unit with two phase shifters.
    
    Args:
        modes (Tuple[int, int]): The two modes the MZI acts on
        phi_int (float): Internal phase shift (between beamsplitters)
        phi_ext (float): External phase shift (after beamsplitters)
        
    Returns:
        pcvl.Circuit: MZI circuit component
    """
    # Ensure phases are within valid range
    phi_int = float(phi_int) % (2 * np.pi)
    phi_ext = float(phi_ext) % (2 * np.pi)
    
    # Ensure modes are consecutive to avoid Perceval error
    if abs(modes[1] - modes[0]) != 1:
        mode1, mode2 = min(modes[0], modes[1]), min(modes[0], modes[1]) + 1
        modes = (mode1, mode2)
    
    c = pcvl.Circuit(max(modes) + 1)
    c.add(modes, pcvl.BS())
    c.add((modes[1],), pcvl.PS(phi=phi_int))
    c.add(modes, pcvl.BS())
    c.add((modes[1],), pcvl.PS(phi=phi_ext))
    return c


def memristor_circuit(phases: np.ndarray) -> pcvl.Circuit:
    """
    Builds a 3-mode memristor circuit with phase shifters and beamsplitters.
    Args:
        phases (np.ndarray): Array of phases [phi1, mem_phi, phi3] for the three PS elements.
    Returns:
        pcvl.Circuit: The constructed memristor circuit.
    """
    phi1, mem_phi, phi3 = phases[0], phases[1], phases[2]
    c = pcvl.Circuit(3, name="Memristor")
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi1)).add((0, 1), pcvl.BS())
    c.add((1, 2), pcvl.BS()).add((2,), pcvl.PS(phi=mem_phi)).add((1, 2), pcvl.BS())
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi3)).add((0, 1), pcvl.BS())
    return c


def clements_circuit(phases: np.ndarray, n_modes: int) -> pcvl.Circuit:
    """
    Builds a rectangular Clements architecture circuit with the given number of modes.
    The circuit consists of a mesh of MZIs arranged in a rectangular grid pattern.
    
    Args:
        phases (np.ndarray): Array of phases for all MZIs in the circuit.
                           Each MZI requires 2 phases, so the array length should be
                           n_phases = n_modes * (n_modes - 1)
        n_modes (int): Number of modes in the circuit
        
    Returns:
        pcvl.Circuit: The constructed Clements circuit
    """
    # Validate inputs
    if n_modes < 2:
        raise ValueError(f"Clements architecture requires at least 2 modes, got {n_modes}")
    
    # Calculate expected number of phases (2 per MZI)
    expected_phases = n_modes * (n_modes - 1)
    
    if len(phases) != expected_phases:
        raise ValueError(
            f"Expected {expected_phases} phases for {n_modes} modes Clements circuit, "
            f"but got {len(phases)}. Each MZI requires 2 phases."
        )
    
    c = pcvl.Circuit(n_modes, name=f"Clements-{n_modes}")
    
    # Phase index counter
    idx = 0
    
    # Layer pattern depends on number of modes
    for layer in range(n_modes - 1):
        # Determine if this is an even or odd layer (affects starting position)
        is_even_layer = (layer % 2 == 0)
        
        # For even layers, start from the top (mode 0)
        # For odd layers, start from the second mode (mode 1)
        start_mode = 0 if is_even_layer else 1
        
        # Iterate through pairs of modes for this layer
        for m in range(start_mode, n_modes - 1, 2):
            if m + 1 < n_modes:  # Ensure the second mode exists
                # Extract the two phases for this MZI
                if idx + 1 < len(phases):  # Check array bounds
                    phi_int = phases[idx]
                    phi_ext = phases[idx + 1]
                    idx += 2
                    
                    # Add the MZI to the circuit
                    c.add(0, mzi_unit((m, m+1), phi_int, phi_ext))
                else:
                    print(f"Warning: Not enough phases provided. Expected at least {idx+2}, got {len(phases)}")
                    break
    
    return c


def build_circuit(
    phases: np.ndarray, 
    enc_phi: float,
    circuit_type: CircuitType = CircuitType.MEMRISTOR,
    n_modes: int = 3,
    encoding_mode: int = 0
) -> pcvl.Circuit:
    """
    Builds a full circuit by combining encoding and main circuit architectures.
    
    Args:
        phases (np.ndarray): Array of phases for the main circuit.
        enc_phi (float): Encoding phase.
        circuit_type (CircuitType): Type of circuit architecture to use.
        n_modes (int): Number of modes for Clements architecture (ignored for memristor).
        encoding_mode (int): Mode to apply encoding to (default: 0).
        
    Returns:
        pcvl.Circuit: The complete circuit.
    """
    # Ensure encoding phase is within valid range
    enc_phi = float(enc_phi) % (2 * np.pi)
    
    # Validate input parameters
    if encoding_mode < 0:
        raise ValueError(f"Encoding mode must be non-negative, got {encoding_mode}")
    
    if circuit_type == CircuitType.MEMRISTOR:
        # Fixed 3-mode circuit for memristor
        if len(phases) != 3:
            raise ValueError(f"Memristor circuit requires exactly 3 phases, got {len(phases)}")
        
        c = pcvl.Circuit(3, name="Full-Memristor")
        # Ensure encoding_mode is valid for the circuit
        valid_encoding_mode = min(encoding_mode, 1)  # Mode 0 or 1 for 2-mode encoding circuit
        c.add(valid_encoding_mode, encoding_circuit(enc_phi))
        c.add(0, memristor_circuit(phases))
        return c
    else:  # CLEMENTS
        if n_modes < 2:
            raise ValueError(f"Clements architecture requires at least 2 modes, got {n_modes}")
            
        if encoding_mode >= n_modes:
            print(f"Warning: Encoding mode {encoding_mode} exceeds available modes ({n_modes}). Using mode 0 instead.")
            encoding_mode = 0
            
        c = pcvl.Circuit(n_modes, name=f"Full-Clements-{n_modes}")
        # For Clements, add encoding to the specified mode
        # Ensure encoding_mode is valid for the circuit
        valid_encoding_mode = min(encoding_mode, n_modes - 1)
        c.add(valid_encoding_mode, encoding_circuit(enc_phi))
        c.add(0, clements_circuit(phases, n_modes))
        return c