from __future__ import annotations

import perceval as pcvl


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


def memristor_circuit(phi1: float, mem_phi: float, phi3: float) -> pcvl.Circuit:
    """
    Builds a 3-mode memristor circuit with phase shifters and beamsplitters.
    Args:
        phi1 (float): Phase for the first PS.
        mem_phi (float): Phase for the memristor PS.
        phi3 (float): Phase for the third PS.
    Returns:
        pcvl.Circuit: The constructed memristor circuit.
    """
    c = pcvl.Circuit(3, name="Memristor")
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi1)).add((0, 1), pcvl.BS())
    c.add((1, 2), pcvl.BS()).add((2,), pcvl.PS(phi=mem_phi)).add((1, 2), pcvl.BS())
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi3)).add((0, 1), pcvl.BS())
    return c


def build_circuit(
    phi1: float, mem_phi: float, phi3: float, enc_phi: float
) -> pcvl.Circuit:
    """
    Builds the full 3-mode circuit by combining encoding and memristor circuits.
    Args:
        phi1 (float): Phase for the first PS in memristor.
        mem_phi (float): Phase for the memristor PS.
        phi3 (float): Phase for the third PS in memristor.
        enc_phi (float): Encoding phase.
    Returns:
        pcvl.Circuit: The complete circuit.
    """
    c = pcvl.Circuit(3, name="Full")
    c.add(0, encoding_circuit(enc_phi))
    c.add(0, memristor_circuit(phi1, mem_phi, phi3))
    return c