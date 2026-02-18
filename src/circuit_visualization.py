"""
Circuit visualization with annotations for UQ-QNN.

Displays or saves the photonic circuit with a legend indicating:
- Input/encoding mode
- Target output modes
- Memristive phase indices and their MZI mode pairs
- Memristive feedback output modes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import perceval as pcvl
from PIL import Image

from .circuits import build_circuit, get_mzi_modes_for_phase
from .simulation import _normalize_memristive_phase_idx


def display_circuit_annotated(
    n_modes: int,
    encoding_mode: int = 0,
    target_mode: Optional[Tuple[int, ...]] = None,
    memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None,
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]] = None,
    phases: Optional[np.ndarray] = None,
    enc_phi: float = np.pi / 4,
    encoding_phase_idx: Optional[int] = None,
    path: Optional[str] = None,
    show: bool = True,
    skin=None,
    dpi: int = 150,
) -> None:
    """
    Display or save the circuit with annotations for input, output, and memristive phases.

    Args:
        n_modes: Number of modes (3 for 3x3, 6 for 6x6, etc.).
        encoding_mode: Mode where the input photon is injected (encoding).
        target_mode: Output mode(s) used for measurement. Default: (n_modes - 1,).
        memristive_phase_idx: Phase indices that are memristive (feedback-controlled).
        memristive_output_modes: For each memristive phase, (mode_p1, mode_p2) for feedback.
        phases: Phase values for the circuit. If None, uses π/4 for all.
        enc_phi: Encoding phase value (radians).
        encoding_phase_idx: If provided, enc_phi is embedded into this phase index
            inside the Clements mesh instead of using a separate encoding_circuit.
        path: If set, save the annotated figure to this file path.
        show: If True, display the figure (e.g. plt.show()).
        skin: Perceval skin (e.g. SymbSkin()). If None, uses default.
        dpi: Resolution for saved image.
    """
    import matplotlib.pyplot as plt

    n_phases = n_modes * (n_modes - 1)
    if phases is None:
        phases = np.ones(n_phases) * (np.pi / 4)
    if target_mode is None:
        target_mode = (n_modes - 1,)

    memristive_indices = _normalize_memristive_phase_idx(
        memristive_phase_idx, n_modes, n_phases
    )

    # Build circuit and save to temp file
    circ = build_circuit(
        phases,
        enc_phi,
        n_modes=n_modes,
        encoding_mode=encoding_mode,
        encoding_phase_idx=encoding_phase_idx,
    )
    proc = pcvl.Processor("SLOS", circ)
    input_modes = [0] * n_modes
    input_modes[encoding_mode] = 1
    proc.with_input(pcvl.BasicState(input_modes))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        if skin is None:
            from perceval.rendering.circuit import SymbSkin
            skin = SymbSkin(compact_display=False)
        pcvl.pdisplay_to_file(proc, tmp_path, skin=skin, recursive=True)
    except Exception:
        pcvl.pdisplay_to_file(circ, tmp_path, skin=skin, recursive=True)

    # Build annotation text with parameters
    lines = [
        "Circuit configuration",
        "─" * 40,
        f"Input (encoding): mode {encoding_mode}",
        f"Target output: modes {target_mode}",
        f"Encoding phase enc_φ: {enc_phi:.4f} rad ({np.degrees(enc_phi):.1f}°)",
        "",
        f"Phases ({n_phases} total):",
    ]
    # Show phases in compact form (group by 6 per line)
    for i in range(0, n_phases, 6):
        chunk = phases[i : i + 6]
        vals = ", ".join(f"{p:.3f}" for p in chunk)
        lines.append(f"  [{i}:{i + len(chunk)}] {vals}")
    lines.append("")
    if memristive_indices:
        lines.append("Memristive phases:")
        for j, idx in enumerate(memristive_indices):
            m1, m2 = get_mzi_modes_for_phase(idx, n_modes)
            modes_str = f"  phase[{idx}] = {phases[idx]:.4f} rad → MZI modes ({m1}, {m2})"
            if memristive_output_modes and j < len(memristive_output_modes):
                m1_out, m2_out = memristive_output_modes[j]
                modes_str += f", feedback from ({m1_out}, {m2_out})"
            lines.append(modes_str)
    else:
        lines.append("Memristive: none")

    # Create composite figure: circuit on top, config below
    img = Image.open(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    fig, (ax_circ, ax_cfg) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [1.2, 0.8], "hspace": 0.3}
    )

    ax_circ.imshow(img)
    ax_circ.axis("off")
    ax_circ.set_title("Circuit", fontsize=14, fontweight="bold")

    ax_cfg.axis("off")
    text = "\n".join(lines)
    ax_cfg.text(
        0.02, 0.98, text,
        transform=ax_cfg.transAxes,
        fontsize=11,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="black", linewidth=1.5),
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.3)

    if path:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved circuit to {path}")
    if show:
        plt.show()
    else:
        plt.close()


def save_circuit_annotated(
    path: str,
    n_modes: int,
    encoding_mode: int = 0,
    target_mode: Optional[Tuple[int, ...]] = None,
    memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None,
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]] = None,
    **kwargs
) -> None:
    """
    Save the annotated circuit to a file. Convenience wrapper around display_circuit_annotated.
    """
    display_circuit_annotated(
        n_modes=n_modes,
        encoding_mode=encoding_mode,
        target_mode=target_mode,
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes,
        path=path,
        show=False,
        **kwargs
    )
