from __future__ import annotations

import numpy as np
import perceval as pcvl
from typing import Optional, Tuple


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


def get_mzi_modes_for_phase(phase_idx: int, n_modes: int) -> Tuple[int, int]:
    """
    Maps a phase index to the mode pair (m1, m2) of the MZI that contains it.
    Uses the same layer ordering as clements_circuit.
    
    Args:
        phase_idx (int): Index into the phases array (0 to n_modes*(n_modes-1)-1).
        n_modes (int): Number of modes in the circuit.
        
    Returns:
        Tuple[int, int]: (mode_low, mode_high) for the MZI containing this phase.
    """
    if n_modes < 2:
        raise ValueError(f"Requires at least 2 modes, got {n_modes}")
    expected_phases = n_modes * (n_modes - 1)
    if phase_idx < 0 or phase_idx >= expected_phases:
        raise ValueError(
            f"phase_idx must be in [0, {expected_phases-1}] for {n_modes} modes, got {phase_idx}"
        )
    idx = 0
    for layer in range(n_modes - 1):
        is_even_layer = (layer % 2 == 0)
        start_mode = 0 if is_even_layer else 1
        for m in range(start_mode, n_modes - 1, 2):
            if m + 1 < n_modes:
                if phase_idx in (idx, idx + 1):
                    return (m, m + 1)
                idx += 2
    return (0, 1)  # Fallback (should not reach)


def memristor_circuit(phases: np.ndarray) -> pcvl.Circuit:
    """
    Builds a 3-mode memristor circuit with phase shifters and beamsplitters.
    DEPRECATED: Use build_circuit(phases, enc_phi, n_modes) with memristive_phase_idx
    in simulation/training for Clements-based memristive behavior. Kept for compatibility.
    
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
                    
                    # Add the MZI to the circuit (merge=True flattens to show BS/PS)
                    c.add(0, mzi_unit((m, m+1), phi_int, phi_ext), merge=True)
                else:
                    print(f"Warning: Not enough phases provided. Expected at least {idx+2}, got {len(phases)}")
                    break
    
    return c


def build_circuit(
    phases: np.ndarray,
    enc_phi: float,
    n_modes: int = 3,
    encoding_mode: int = 0,
) -> pcvl.Circuit:
    """
    Builds a full Clements circuit by combining encoding and main circuit.

    Architecture is always Clements (3x3, 6x6, etc.). Phases array length must be
    n_modes * (n_modes - 1).

    Args:
        phases (np.ndarray): Array of phases for the Clements mesh.
        enc_phi (float): Encoding phase.
        n_modes (int): Number of modes (3 for 3x3, 6 for 6x6, etc.).
        encoding_mode (int): Mode to apply encoding to (default: 0).

    Returns:
        pcvl.Circuit: The complete circuit.
    """
    enc_phi = float(enc_phi) % (2 * np.pi)
    if encoding_mode < 0:
        raise ValueError(f"Encoding mode must be non-negative, got {encoding_mode}")
    if n_modes < 2:
        raise ValueError(f"Requires at least 2 modes, got {n_modes}")

    expected_phases = n_modes * (n_modes - 1)
    if len(phases) != expected_phases:
        raise ValueError(
            f"Clements circuit requires {expected_phases} phases for {n_modes} modes, "
            f"got {len(phases)}"
        )

    c = pcvl.Circuit(n_modes, name=f"Clements-{n_modes}x{n_modes}")
    valid_encoding_mode = min(max(0, encoding_mode), n_modes - 2)
    c.add(valid_encoding_mode, encoding_circuit(enc_phi), merge=True)
    c.add(0, clements_circuit(phases, n_modes), merge=True)
    return c