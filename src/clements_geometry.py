"""Clements mesh layout and memristor index helpers (NumPy only; no Perceval)."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union


@lru_cache(maxsize=None)
def clements_mzi_pairs(n_modes: int) -> tuple[Tuple[int, int], ...]:
    """
    Return the ordered list of MZI mode pairs for a rectangular Clements mesh.

    For n_modes = 6 this gives the familiar 3×2×3×2×3×2 layout:
    (0,1),(2,3),(4,5),(1,2),(3,4), repeated three times.
    For n_modes = 3 it yields (0,1),(1,2),(0,1), matching the tests.
    """
    if n_modes < 2:
        raise ValueError(
            f"Clements architecture requires at least 2 modes, got {n_modes}"
        )

    pairs: list[Tuple[int, int]] = []

    full_blocks = n_modes // 2
    for _ in range(full_blocks):
        for j in range(0, n_modes - 1, 2):
            pairs.append((j, j + 1))
        for j in range(1, n_modes - 1, 2):
            pairs.append((j, j + 1))

    if n_modes % 2 == 1:
        for j in range(0, n_modes - 1, 2):
            pairs.append((j, j + 1))

    return tuple(pairs)


def get_mzi_modes_for_phase(phase_idx: int, n_modes: int) -> Tuple[int, int]:
    """
    Maps a phase index to the mode pair (m1, m2) of the MZI that contains it.
    Uses the same ordering as clements_circuit, with two consecutive phases per MZI.

    Args:
        phase_idx: Index into the phases array (0 to n_modes*(n_modes-1)-1).
        n_modes: Number of modes in the circuit.

    Returns:
        (mode_low, mode_high) for the MZI containing this phase.
    """
    if n_modes < 2:
        raise ValueError(f"Requires at least 2 modes, got {n_modes}")

    pairs = clements_mzi_pairs(n_modes)
    expected_phases = 2 * len(pairs)
    if phase_idx < 0 or phase_idx >= expected_phases:
        raise ValueError(
            f"phase_idx must be in [0, {expected_phases - 1}] for {n_modes} modes, got {phase_idx}"
        )

    mzi_idx = phase_idx // 2
    return pairs[mzi_idx]


def normalize_encoding_phase_idx(
    encoding_phase_idx: Union[int, Sequence[int]],
    n_layers: int,
    n_modes: int,
) -> Tuple[int, ...]:
    """Normalize data-encoding phase slot indices (flat over L stacked meshes)."""

    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    n_phases_per_layer = n_modes * (n_modes - 1)
    total_phases = n_layers * n_phases_per_layer
    if isinstance(encoding_phase_idx, int):
        idx = int(encoding_phase_idx)
        if n_layers != 1:
            raise ValueError(
                f"single int encoding_phase_idx requires n_layers=1, got n_layers={n_layers}"
            )
        if idx < 0 or idx >= n_phases_per_layer:
            raise ValueError(
                f"encoding_phase_idx must be in [0, {n_phases_per_layer - 1}] "
                f"for {n_modes} modes, got {idx}"
            )
        return (idx,)
    indices = tuple(int(x) for x in encoding_phase_idx)
    if not indices:
        raise ValueError("encoding_phase_idx must not be empty")
    for idx in indices:
        if idx < 0 or idx >= total_phases:
            raise ValueError(
                f"each encoding_phase_idx must be in [0, {total_phases - 1}] "
                f"for n_layers={n_layers}, n_modes={n_modes}, got {idx}"
            )
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"encoding_phase_idx must not contain duplicates, got {encoding_phase_idx}"
        )
    return indices


def normalize_memristive_phase_idx(
    memristive_phase_idx: Optional[Union[int, Sequence[int]]],
    n_modes: int,
    n_phases: int,
) -> Tuple[int, ...]:
    """Normalize memristive phase indices from user input."""

    if memristive_phase_idx is None:
        return ()
    if isinstance(memristive_phase_idx, int):
        idx = int(memristive_phase_idx)
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"memristive_phase_idx must be in [0, {n_phases - 1}] for {n_modes} modes, got {idx}"
            )
        return (idx,)
    indices = tuple(int(x) for x in memristive_phase_idx)
    if not indices:
        return ()
    for idx in indices:
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"Each memristive_phase_idx must be in [0, {n_phases - 1}] for {n_modes} modes, got {idx}"
            )
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"memristive_phase_idx must not contain duplicates, got {memristive_phase_idx}"
        )
    return indices


def normalize_memristive_output_modes(
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]],
    memristive_indices: Tuple[int, ...],
    n_modes: int,
) -> Tuple[Tuple[int, int], ...]:
    """Normalize memristor output monitors to valid mode pairs."""

    if memristive_output_modes is None:
        return tuple(
            get_mzi_modes_for_phase(idx, n_modes) for idx in memristive_indices
        )
    modes = tuple((int(m1), int(m2)) for m1, m2 in memristive_output_modes)
    if len(modes) != len(memristive_indices):
        raise ValueError(
            f"memristive_output_modes must have {len(memristive_indices)} entries "
            f"(one per memristive phase), got {len(modes)}"
        )
    for j, (m1, m2) in enumerate(modes):
        if m1 < 0 or m1 >= n_modes or m2 < 0 or m2 >= n_modes:
            raise ValueError(
                f"memristive_output_modes[{j}] = ({m1}, {m2}): modes must be in [0, {n_modes - 1}]"
            )
        if m1 == m2:
            raise ValueError(
                f"memristive_output_modes[{j}] = ({m1}, {m2}): the two modes must differ"
            )
    return modes
