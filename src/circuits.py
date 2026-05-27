from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np

from . import clements_geometry as _clements_geometry

def _import_perceval():
    """Load Perceval on demand so ``import src.circuits`` stays NumPy-only safe."""

    import perceval as pcvl

    return pcvl

clements_mzi_pairs = _clements_geometry.clements_mzi_pairs
get_mzi_modes_for_phase = _clements_geometry.get_mzi_modes_for_phase
normalize_memristive_phase_idx = _clements_geometry.normalize_memristive_phase_idx
normalize_memristive_output_modes = _clements_geometry.normalize_memristive_output_modes
normalize_encoding_phase_idx = _clements_geometry.normalize_encoding_phase_idx


def encoding_circuit(encoded_phase: float) -> Any:
    """
    Builds a 2-mode encoding circuit with a phase shifter.
    Args:
        encoded_phase (float): Phase to encode.
    Returns:
        pcvl.Circuit: The constructed encoding circuit.
    """
    pcvl = _import_perceval()
    c = pcvl.Circuit(2, name="Encoding")
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=encoded_phase))
    c.add((0, 1), pcvl.BS())
    return c


def mzi_unit(modes: Tuple[int, int], phi_int: float, phi_ext: float) -> Any:
    """
    Creates a basic Mach-Zehnder Interferometer (MZI) unit with two phase shifters.

    Args:
        modes (Tuple[int, int]): The two modes the MZI acts on
        phi_int (float): Internal phase shift (between beamsplitters)
        phi_ext (float): External phase shift (after beamsplitters)

    Returns:
        pcvl.Circuit: MZI circuit component
    """
    pcvl = _import_perceval()
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


def memristor_circuit(phases: np.ndarray) -> Any:
    """
    Builds a 3-mode memristor circuit with phase shifters and beamsplitters.
    DEPRECATED: Use build_circuit(phases, enc_phi, n_modes) with memristive_phase_idx
    in simulation/training for Clements-based memristive behavior. Kept for compatibility.

    Args:
        phases (np.ndarray): Array of phases [phi1, mem_phi, phi3] for the three PS elements.
    Returns:
        pcvl.Circuit: The constructed memristor circuit.
    """
    pcvl = _import_perceval()
    phi1, mem_phi, phi3 = phases[0], phases[1], phases[2]
    c = pcvl.Circuit(3, name="Memristor")
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi1)).add((0, 1), pcvl.BS())
    c.add((1, 2), pcvl.BS()).add((2,), pcvl.PS(phi=mem_phi)).add((1, 2), pcvl.BS())
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi3)).add((0, 1), pcvl.BS())
    return c


def clements_circuit(phases: np.ndarray, n_modes: int) -> Any:
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
    pcvl = _import_perceval()
    # Validate inputs
    if n_modes < 2:
        raise ValueError(
            f"Clements architecture requires at least 2 modes, got {n_modes}"
        )

    # Determine MZI ordering and required number of phases
    pairs = clements_mzi_pairs(n_modes)
    expected_phases = 2 * len(pairs)

    # Sanity check: this should match n_modes * (n_modes - 1) for a universal mesh
    if expected_phases != n_modes * (n_modes - 1):
        raise RuntimeError(
            f"Internal error: Clements pattern inconsistent for n_modes={n_modes} "
            f"(expected {n_modes * (n_modes - 1)} phases, got {expected_phases})"
        )

    if len(phases) != expected_phases:
        raise ValueError(
            f"Expected {expected_phases} phases for {n_modes} modes Clements circuit, "
            f"but got {len(phases)}. Each MZI requires 2 phases."
        )

    c = pcvl.Circuit(n_modes, name=f"Clements-{n_modes}")

    # Add one MZI per pair, using two consecutive phases
    for mzi_idx, (m1, m2) in enumerate(pairs):
        phi_int = phases[2 * mzi_idx]
        phi_ext = phases[2 * mzi_idx + 1]
        c.add(0, mzi_unit((m1, m2), phi_int, phi_ext), merge=True)

    return c


def _coerce_enc_phi(enc_phi: Union[float, np.ndarray]) -> np.ndarray:
    enc = np.asarray(enc_phi, dtype=float).reshape(-1)
    return enc


def build_circuit(
    phases: np.ndarray,
    enc_phi: Union[float, np.ndarray],
    n_modes: int,
    encoding_phase_idx: Union[int, Tuple[int, ...]],
    n_layers: int = 1,
) -> Any:
    """
    Builds a full Clements circuit with **internal** data encoding.

  Data phases are added (mod 2π) at ``encoding_phase_idx`` slot(s) inside the mesh.
  For ``n_layers > 1``, ``L`` Clements blocks are chained on the same ``n_modes``.

    Args:
        phases: Flat phase vector, length ``n_layers * n_modes * (n_modes - 1)``.
        enc_phi: Scalar or vector of data-encoded phase contributions (radians).
        n_modes: Number of modes.
        encoding_phase_idx: Flat index or tuple of indices receiving ``enc_phi``.
        n_layers: Number of stacked re-uploading layers (default 1).

    Returns:
        pcvl.Circuit: The complete circuit.
    """
    pcvl = _import_perceval()
    if n_modes < 2:
        raise ValueError(f"Requires at least 2 modes, got {n_modes}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")

    n_ph = n_modes * (n_modes - 1)
    expected_phases = n_layers * n_ph
    phases = np.asarray(phases, dtype=float).reshape(-1)
    if len(phases) != expected_phases:
        raise ValueError(
            f"Clements circuit requires {expected_phases} phases for "
            f"n_layers={n_layers}, n_modes={n_modes}, got {len(phases)}"
        )

    enc = _coerce_enc_phi(enc_phi)
    slots = normalize_encoding_phase_idx(encoding_phase_idx, n_layers, n_modes)
    if len(enc) != len(slots):
        raise ValueError(
            f"enc_phi length ({len(enc)}) must match encoding slots ({len(slots)})"
        )

    mesh_phases = phases.copy()
    for j, slot in enumerate(slots):
        mesh_phases[int(slot)] = (mesh_phases[int(slot)] + enc[j]) % (2 * np.pi)

    c = pcvl.Circuit(n_modes, name=f"Clements-stack-{n_layers}x{n_modes}")
    for layer in range(n_layers):
        start = layer * n_ph
        end = start + n_ph
        block = clements_circuit(mesh_phases[start:end], n_modes)
        c.add(0, block, merge=True)
    return c


# Backward compatibility for legacy imports -------------------------------------
_clements_mzi_pairs = clements_mzi_pairs
_normalize_memristive_phase_idx = normalize_memristive_phase_idx
_normalize_memristive_output_modes = normalize_memristive_output_modes
