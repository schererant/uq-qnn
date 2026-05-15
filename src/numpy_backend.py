"""
Fast NumPy backend for Clements linear-optical simulations.

Computes the unitary via the same MZI ordering as Perceval and extracts
probabilities using the standard bosonic (permanent) rule for collision-free
outputs. Vectorizes across data points when the mesh phases are shared
(non-memristive, or memristive with fixed mesh within a batch step).
"""

from __future__ import annotations

from math import factorial
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .clements_geometry import clements_mzi_pairs, normalize_encoding_phase_idx
from .coincidence import (
    apply_noise_to_outcomes,
    detector_tuple_to_binary_occupation,
    expanded_mode_indices_from_occupation,
    get_cc_mode_pairs,
    get_nfold_channel_labels,
    mode_pair_to_cc_index,
    nfold_tuple_to_channel_index,
    nfold_working_detector_tuples,
    postselect_measurement,
)
from .config import SimConfig


def slos_2photon_numpy(
    U: np.ndarray,
    input_modes: tuple[int, int] = (2, 3),
) -> dict[tuple[int, ...], float]:
    """
    Full 2-photon output distribution via analytical 2x2 permanents.

    Returns:
        Dict mapping n-mode Fock occupation tuples to probabilities.
        For n_modes=6 this has 21 states (6 bunched + 15 split).
    """
    u = np.asarray(U, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"U must be square, got shape {u.shape}")
    n_modes = int(u.shape[0])
    i, j = int(input_modes[0]), int(input_modes[1])
    if i == j:
        raise ValueError("input_modes must be distinct for one photon per mode")
    if not (0 <= i < n_modes and 0 <= j < n_modes):
        raise ValueError(
            f"input modes must be in [0, {n_modes - 1}], got {(i, j)}"
        )

    dist: dict[tuple[int, ...], float] = {}

    # Bunched outcomes: occupation has 2 photons in one output mode.
    for m in range(n_modes):
        occ = [0] * n_modes
        occ[m] = 2
        p = 2.0 * (np.abs(u[m, i] * u[m, j]) ** 2)
        dist[tuple(occ)] = float(np.real_if_close(p))

    # Split outcomes: one photon in m and one in n, with m < n.
    for m in range(n_modes):
        for n in range(m + 1, n_modes):
            occ = [0] * n_modes
            occ[m] = 1
            occ[n] = 1
            amp = u[m, i] * u[n, j] + u[m, j] * u[n, i]
            p = np.abs(amp) ** 2
            dist[tuple(occ)] = float(np.real_if_close(p))

    return dist


def _has_positive_noise(
    noise_std: Optional[Union[float, Sequence[float]]],
) -> bool:
    if noise_std is None:
        return False
    if isinstance(noise_std, (int, float)):
        return float(noise_std) > 0
    return any(float(s) > 0 for s in noise_std)


_BS_2X2: np.ndarray = np.array([[1, 1j], [1j, 1]], dtype=np.complex128) / np.sqrt(2)


def _mzi_unitary(phi_int: float, phi_ext: float) -> np.ndarray:
    phi_int = float(phi_int) % (2 * np.pi)
    phi_ext = float(phi_ext) % (2 * np.pi)
    bs = _BS_2X2
    p_int = np.diag(np.array([1.0, np.exp(1j * phi_int)], dtype=np.complex128))
    p_ext = np.diag(np.array([1.0, np.exp(1j * phi_ext)], dtype=np.complex128))
    return p_ext @ bs @ p_int @ bs


def _mzi_unitary_batch(phi_int: np.ndarray, phi_ext: np.ndarray) -> np.ndarray:
    """Same as ``_mzi_unitary`` but vectorized. Shapes (n_data,) → (n_data, 2, 2)."""

    phi_int = np.asarray(phi_int, dtype=np.float64) % (2 * np.pi)
    phi_ext = np.asarray(phi_ext, dtype=np.float64) % (2 * np.pi)
    bs = _BS_2X2
    n_data = int(phi_int.shape[0])
    p_int = np.zeros((n_data, 2, 2), dtype=np.complex128)
    p_int[:, 0, 0] = 1.0
    p_int[:, 1, 1] = np.exp(1j * phi_int)
    p_ext = np.zeros((n_data, 2, 2), dtype=np.complex128)
    p_ext[:, 0, 0] = 1.0
    p_ext[:, 1, 1] = np.exp(1j * phi_ext)
    a = p_ext @ bs
    a = np.matmul(a, p_int)
    return np.matmul(a, bs)


def _apply_mzi_to_state(
    state: np.ndarray,
    m1: int,
    m2: int,
    phi_int: float,
    phi_ext: float,
) -> None:
    """Apply one MZI directly to a state vector in place."""

    phi_int = float(phi_int) % (2 * np.pi)
    phi_ext = float(phi_ext) % (2 * np.pi)
    e_int = np.exp(1j * phi_int)
    e_ext = np.exp(1j * phi_ext)

    a = state[m1]
    b = state[m2]
    state[m1] = 0.5 * ((1.0 - e_int) * a + 1j * (1.0 + e_int) * b)
    state[m2] = 0.5 * e_ext * (1j * (1.0 + e_int) * a + (-1.0 + e_int) * b)


def clements_unitary(phases: np.ndarray, n_modes: int) -> np.ndarray:
    """Full n_modes x n_modes unitary for the Clements mesh (matches Perceval ordering)."""
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    pairs = clements_mzi_pairs(n_modes)
    expected = n_modes * (n_modes - 1)
    if len(phases) != expected:
        raise ValueError(f"Expected {expected} phases, got {len(phases)}")
    u = np.eye(n_modes, dtype=np.complex128)
    for mzi_idx, (m1, m2) in enumerate(pairs):
        m2x2 = _mzi_unitary(phases[2 * mzi_idx], phases[2 * mzi_idx + 1])
        u_mzi = np.eye(n_modes, dtype=np.complex128)
        u_mzi[m1 : m2 + 1, m1 : m2 + 1] = m2x2
        u = u_mzi @ u
    return u


def clements_unitary_batch(phases_batch: np.ndarray, n_modes: int) -> np.ndarray:
    """Batched Clements meshes: ``phases_batch`` (n_data, n_phases) → (n_data, n, n)."""

    phases_batch = np.asarray(phases_batch, dtype=np.float64)
    if phases_batch.ndim != 2:
        raise ValueError(f"phases_batch must be 2D, got shape {phases_batch.shape}")
    n_data, n_ph = phases_batch.shape
    expected = n_modes * (n_modes - 1)
    if n_ph != expected:
        raise ValueError(f"Expected {expected} phases per row, got {n_ph}")
    pairs = clements_mzi_pairs(n_modes)
    u = np.broadcast_to(
        np.eye(n_modes, dtype=np.complex128), (n_data, n_modes, n_modes)
    ).copy()
    for mzi_idx, (m1, m2) in enumerate(pairs):
        phi_i = 2 * mzi_idx
        m2x2 = _mzi_unitary_batch(phases_batch[:, phi_i], phases_batch[:, phi_i + 1])
        u_mzi = np.broadcast_to(
            np.eye(n_modes, dtype=np.complex128), (n_data, n_modes, n_modes)
        ).copy()
        u_mzi[:, m1, m1] = m2x2[:, 0, 0]
        u_mzi[:, m1, m2] = m2x2[:, 0, 1]
        u_mzi[:, m2, m1] = m2x2[:, 1, 0]
        u_mzi[:, m2, m2] = m2x2[:, 1, 1]
        u = np.matmul(u_mzi, u)
    return u


def _basis_input_state(input_mode: int, n_modes: int) -> np.ndarray:
    """Single-photon computational basis state for inline encoding."""

    state = np.zeros(n_modes, dtype=np.complex128)
    state[int(input_mode)] = 1.0
    return state


def _propagate_state_vector(
    phases: np.ndarray, state: np.ndarray, n_modes: int
) -> np.ndarray:
    """Propagate a single-photon state through the Clements mesh."""

    out = np.asarray(state, dtype=np.complex128).copy()
    pairs = clements_mzi_pairs(n_modes)
    for mzi_idx, (m1, m2) in enumerate(pairs):
        phi_idx = 2 * mzi_idx
        _apply_mzi_to_state(out, m1, m2, phases[phi_idx], phases[phi_idx + 1])
    return out


def _coerce_encoded_phases(enc_phases: np.ndarray) -> np.ndarray:
    """Return ``(n_data, n_enc_slots)``; 1D input is treated as a single slot."""

    enc = np.asarray(enc_phases, dtype=np.float64)
    if enc.ndim == 0:
        return enc.reshape(1, 1)
    if enc.ndim == 1:
        return enc.reshape(-1, 1)
    if enc.ndim != 2:
        raise ValueError(f"encoded_phases must be 1D or 2D, got shape {enc.shape}")
    return enc


def clements_stack_unitary_batch(
    phases: np.ndarray,
    enc: np.ndarray,
    encoding_slots: Tuple[int, ...],
    n_modes: int,
    n_layers: int,
) -> np.ndarray:
    """Stacked Clements meshes with multi-slot internal encoding.

    Returns unitary ``U`` of shape ``(n_data, n_modes, n_modes)`` with
    ``U = U_{L-1} @ ... @ U_0`` (layer 0 applied first to the input state).
    """
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    enc = _coerce_encoded_phases(enc)
    n_data, n_enc = enc.shape
    n_ph = n_modes * (n_modes - 1)
    expected_phases = n_layers * n_ph
    if len(phases) != expected_phases:
        raise ValueError(
            f"Expected {expected_phases} phases for n_layers={n_layers}, "
            f"n_modes={n_modes}, got {len(phases)}"
        )
    if len(encoding_slots) != n_enc:
        raise ValueError(
            f"encoding_slots length ({len(encoding_slots)}) must match "
            f"encoded columns ({n_enc})"
        )

    phases_tile = np.broadcast_to(phases, (n_data, expected_phases)).copy()
    for j, slot in enumerate(encoding_slots):
        phases_tile[:, int(slot)] = (
            phases_tile[:, int(slot)] + enc[:, j]
        ) % (2 * np.pi)

    if n_layers == 1:
        return clements_unitary_batch(phases_tile, n_modes)

    u_out = np.broadcast_to(
        np.eye(n_modes, dtype=np.complex128), (n_data, n_modes, n_modes)
    ).copy()
    for layer in range(n_layers):
        start = layer * n_ph
        end = start + n_ph
        u_layer = clements_unitary_batch(phases_tile[:, start:end], n_modes)
        u_out = np.matmul(u_layer, u_out)
    return u_out


def _unitary_batch_internal_encoding(
    phases: np.ndarray,
    enc_phases: np.ndarray,
    n_modes: int,
    encoding_phase_idx: Union[int, Tuple[int, ...]],
    n_layers: int = 1,
) -> np.ndarray:
    """Unitary per data point with data phases added at encoding slots.

    Shape ``(n_data, n_modes, n_modes)``.  Dispatches to
    :func:`clements_stack_unitary_batch` for multi-layer / multi-slot encoding.
    """
    enc = _coerce_encoded_phases(enc_phases)
    slots = normalize_encoding_phase_idx(encoding_phase_idx, n_layers, n_modes)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    return clements_stack_unitary_batch(phases, enc, slots, n_modes, n_layers)


def _singles_probabilities_batch(
    u_batch: np.ndarray,
    singles_input_mode: int,
    target_mode: Tuple[int, ...],
    return_class_probs: bool,
) -> np.ndarray:
    """u_batch (n_data, n, n); Born column = singles_input_mode (input photon mode)."""
    enc_col = int(singles_input_mode)
    if return_class_probs and len(target_mode) > 1:
        t = np.array(target_mode, dtype=int)
        raw = np.abs(u_batch[:, t, enc_col]) ** 2
        return raw
    tprob = np.abs(u_batch[:, list(target_mode), enc_col]) ** 2
    if len(target_mode) > 1:
        tprob = tprob.mean(axis=1)
    else:
        tprob = tprob[:, 0]
    return tprob


def _coincidence_raw_batch(
    u_batch: np.ndarray, input_modes: Tuple[int, int], n_modes: int
) -> np.ndarray:
    """Collision-free coincidence probabilities, shape (n_data, n_cc)."""
    a, b = int(input_modes[0]), int(input_modes[1])
    pairs = get_cc_mode_pairs(n_modes)
    mj = np.array([p[0] for p in pairs], dtype=int)
    mk = np.array([p[1] for p in pairs], dtype=int)
    perm = u_batch[:, mj, a] * u_batch[:, mk, b] + u_batch[:, mj, b] * u_batch[:, mk, a]
    return np.abs(perm) ** 2


def _ryser_permanent(matrix: np.ndarray) -> complex:
    """Bosonic permanent via Ryser expansion (square complex matrix)."""

    m = np.asarray(matrix, dtype=np.complex128)
    n = m.shape[0]
    if m.shape != (n, n):
        raise ValueError(f"permanent requires a square matrix, got {m.shape}")
    if n == 0:
        return complex(1.0, 0.0)
    permanent = 0.0 + 0.0j
    for mask in range(1, 1 << n):
        cols = [j for j in range(n) if (mask >> j) & 1]
        row_sums = np.sum(m[:, cols], axis=1)
        term = np.prod(row_sums)
        parity = n - len(cols)
        sign = -1.0 if (parity % 2) else 1.0
        permanent += sign * term
    return permanent


def _transition_probability_boson(
    u: np.ndarray,
    input_occ: Tuple[int, ...],
    output_occ: Tuple[int, ...],
) -> float:
    """Born probability for indistinguishable bosons (matches PhotonicCircuit._transition_probability)."""

    in_rows = expanded_mode_indices_from_occupation(input_occ)
    out_cols = expanded_mode_indices_from_occupation(output_occ)
    # Rows = output modes, cols = input modes (matches Perceval SLOS and _coincidence_raw_batch).
    sub_u = u[np.ix_(out_cols, in_rows)]
    perm = _ryser_permanent(sub_u)
    in_norm = float(np.prod([factorial(int(nu)) for nu in input_occ]))
    out_norm = float(np.prod([factorial(int(mu)) for mu in output_occ]))
    prob = (abs(perm) ** 2) / (in_norm * out_norm)
    return float(np.real_if_close(prob))


def _occupation_two_distinct_modes(
    input_occ: Tuple[int, ...],
) -> Optional[Tuple[int, int]]:
    """If occupation is exactly one photon in each of two distinct modes, return (a, b)."""

    if sum(int(x) for x in input_occ) != 2:
        return None
    if max(int(x) for x in input_occ) > 1:
        return None
    modes = [i for i, c in enumerate(input_occ) if int(c) == 1]
    if len(modes) != 2:
        return None
    return int(modes[0]), int(modes[1])


def _coincidence_nfold_raw_batch(
    u_batch: np.ndarray,
    input_occ: Tuple[int, ...],
    working_detectors: Tuple[int, ...],
    n_modes: int,
) -> np.ndarray:
    """
    N-fold postselected coincidence probabilities, shape (n_data, C(W, N)).

    Uses the vectorized two-photon closed form when possible; otherwise Ryser per channel.
    """

    n_photons = int(sum(input_occ))
    tuples = nfold_working_detector_tuples(working_detectors, n_photons)
    n_ch = len(tuples)
    n_data = int(u_batch.shape[0])
    out = np.zeros((n_data, n_ch), dtype=np.float64)

    pair = _occupation_two_distinct_modes(input_occ)
    if pair is not None and n_photons == 2:
        coinc_full = _coincidence_raw_batch(u_batch, pair, n_modes)
        idxs = [mode_pair_to_cc_index(t[0], t[1], n_modes) for t in tuples]
        out[:, :] = coinc_full[:, np.asarray(idxs, dtype=int)]
        return out

    for j, det_tup in enumerate(tuples):
        out_occ = detector_tuple_to_binary_occupation(det_tup, n_modes)
        for i in range(n_data):
            out[i, j] = _transition_probability_boson(u_batch[i], input_occ, out_occ)
    return out


def _process_coincidence_rows(
    coinc: np.ndarray,
    working_cc_indices: Sequence[int],
    cc_labels: list[str],
    add_noise: bool,
    noise_std: Optional[Union[float, Sequence[float]]],
    return_class_probs: bool,
    preds: np.ndarray,
    readout_cc_idx: Optional[int] = None,
) -> None:
    if add_noise:
        # Per-row sequential path preserves the seed=i reproducibility contract.
        n_data = coinc.shape[0]
        for i in range(n_data):
            out = postselect_measurement(
                coinc[i], working_cc_indices, cc_labels, fallback_uniform=True
            )
            assert noise_std is not None
            out = apply_noise_to_outcomes(
                out, noise_std, working_cc_indices, cc_labels, seed=i
            )
            if return_class_probs and len(working_cc_indices) > 1:
                preds[i, :] = out[list(working_cc_indices)]
            else:
                if readout_cc_idx is None:
                    raise ValueError(
                        "scalar coincidence output requires readout_cc_idx (set target_mode to an "
                        "N-tuple of distinct detector indices matching sum(input_state))"
                    )
                preds[i] = out[readout_cc_idx]
        return

    # Vectorized path (no noise): postselect and normalise the whole batch at once.
    wci = np.asarray(list(working_cc_indices), dtype=int)
    n_w = len(wci)
    working_vals = coinc[:, wci]                              # (n_data, n_working)
    totals = working_vals.sum(axis=1, keepdims=True)          # (n_data, 1)
    safe_totals = np.where(totals > 0, totals, 1.0)
    normalized = np.where(                                    # (n_data, n_working)
        totals > 0,
        working_vals / safe_totals,
        1.0 / n_w,
    )

    if return_class_probs and n_w > 1:
        preds[:, :] = normalized
    else:
        if readout_cc_idx is None:
            raise ValueError(
                "scalar coincidence output requires readout_cc_idx (set target_mode to an "
                "N-tuple of distinct detector indices matching sum(input_state))"
            )
        hits = np.flatnonzero(wci == readout_cc_idx)
        if hits.size == 0:
            raise ValueError(
                f"readout_cc_idx {readout_cc_idx} is not in working_cc_indices"
            )
        preds[:] = normalized[:, hits[0]]


def run_vectorized_non_memristive(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    cfg: SimConfig,
    *,
    return_class_probs: bool = False,
) -> np.ndarray:
    """
    Vectorized simulation for non-memristive discrete mode.
    """
    target_mode: Tuple[int, ...] = (
        cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)
    )
    n_phases = cfg.total_mesh_phases
    phases = np.asarray(params[:n_phases], dtype=np.float64)
    enc = _coerce_encoded_phases(encoded_phases)
    num_pts = enc.shape[0]

    if cfg.output_mode == "coincidence":
        input_occ = tuple(int(x) for x in cfg.input_state)
        assert cfg.working_detectors is not None
        wd = tuple(int(x) for x in cfg.working_detectors)
        n_photons = int(sum(input_occ))
        nfold_tuples = nfold_working_detector_tuples(wd, n_photons)
        n_ch = len(nfold_tuples)
        working_cc_indices = tuple(range(n_ch))
        cc_labels = get_nfold_channel_labels(wd, n_photons)
        add_noise = _has_positive_noise(cfg.noise_std)
        n_classes = n_ch
    else:
        n_classes = len(target_mode)
        add_noise = False
        working_cc_indices = ()
        cc_labels: list[str] = []

    if return_class_probs and n_classes > 1:
        preds = np.zeros((num_pts, n_classes), dtype=float)
    else:
        preds = np.zeros(num_pts, dtype=float)

    u_batch = _unitary_batch_internal_encoding(
        phases,
        enc,
        cfg.n_modes,
        cfg.encoding_phase_idx,
        n_layers=cfg.n_layers,
    )

    if cfg.output_mode == "coincidence":
        coinc = _coincidence_nfold_raw_batch(u_batch, input_occ, wd, cfg.n_modes)
        readout_cc_idx: Optional[int] = None
        if cfg.loss_type == "mse":
            assert cfg.target_mode is not None
            readout_cc_idx = nfold_tuple_to_channel_index(
                cfg.target_mode, wd, n_photons
            )
        _process_coincidence_rows(
            coinc,
            working_cc_indices,
            cc_labels,
            add_noise,
            cfg.noise_std,
            return_class_probs,
            preds,
            readout_cc_idx=readout_cc_idx,
        )
    elif return_class_probs and n_classes > 1:
        preds[:, :] = _singles_probabilities_batch(
            u_batch, cfg.singles_input_mode, target_mode, True
        )
    else:
        preds[:] = _singles_probabilities_batch(
            u_batch, cfg.singles_input_mode, target_mode, False
        )

    return preds


def run_psr_forward_batch(
    theta_np: np.ndarray,
    p_idx: int,
    shifts_np: np.ndarray,
    enc_np: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Evaluate all PSR shifts for one phase in a single ``clements_unitary_batch`` call.

    Instead of running ``n_shifts`` independent forward passes sequentially, this
    function builds a ``(n_shifts × n_data, n_phases)`` phase matrix and dispatches
    it to ``clements_unitary_batch`` once, reducing the number of unitary builds from
    ``n_shifts`` to 1 per trainable phase.

    Parameters
    ----------
    theta_np  : (n_params,)  Base parameter vector.
    p_idx     : int          Index of the phase being PSR-shifted.  Must differ from
                             ``cfg.encoding_phase_idx`` (the encoding slot is never
                             in ``phase_idx``).
    shifts_np : (n_shifts,)  PSR shift values for this phase.
    enc_np    : (n_data,)    Encoded input phases.
    cfg       : SimConfig    Feature config from ``MemristorLossPSR._feature_cfg``.
                             Must satisfy ``backend="numpy"``, ``n_swipe=0``, no
                             memristive phases.

    Returns
    -------
    np.ndarray, shape ``(n_shifts, n_data, n_features)``.
    """
    n_shifts = len(shifts_np)
    enc_2d = _coerce_encoded_phases(enc_np)
    n_data = enc_2d.shape[0]
    n_phases = cfg.total_mesh_phases
    encoding_slots = cfg.encoding_slots

    phases_base = np.asarray(theta_np[:n_phases], dtype=np.float64)

    # (n_shifts, n_phases): apply the PSR shift only at column p_idx
    theta_batch = np.tile(phases_base, (n_shifts, 1))
    theta_batch[:, p_idx] = theta_batch[:, p_idx] + shifts_np

    # (n_shifts * n_data, n_phases): tile each shifted-theta for every data point
    phases_flat = np.repeat(theta_batch, n_data, axis=0)

    # Add data encoding at every slot (enc columns tiled per shift block)
    enc_flat = np.tile(enc_2d, (n_shifts, 1))
    for j, slot in enumerate(encoding_slots):
        phases_flat[:, int(slot)] = (
            phases_flat[:, int(slot)] + enc_flat[:, j]
        ) % (2.0 * np.pi)

    # One unitary build for all (shift, data-point) combinations
    if cfg.n_layers == 1:
        u_flat = clements_unitary_batch(phases_flat, cfg.n_modes)
    else:
        u_flat = np.zeros(
            (n_shifts * n_data, cfg.n_modes, cfg.n_modes), dtype=np.complex128
        )
        n_ph = cfg.n_phases_per_layer
        for layer in range(cfg.n_layers):
            start = layer * n_ph
            end = start + n_ph
            u_layer = clements_unitary_batch(phases_flat[:, start:end], cfg.n_modes)
            u_flat = np.matmul(u_layer, u_flat)

    if cfg.output_mode == "singles":
        input_col = cfg.singles_input_mode
        probs_flat = np.abs(u_flat[:, :, input_col]) ** 2     # (n_shifts*n_data, n_modes)
        return probs_flat.reshape(n_shifts, n_data, cfg.n_modes)

    # Coincidence: compute raw pair probabilities then postselect-normalise
    assert cfg.working_detectors is not None
    input_occ = tuple(int(x) for x in cfg.input_state)
    wd = tuple(int(x) for x in cfg.working_detectors)

    coinc_flat = _coincidence_nfold_raw_batch(
        u_flat, input_occ, wd, cfg.n_modes
    )                                                          # (n_shifts*n_data, n_ch)

    n_ch = coinc_flat.shape[1]
    totals = coinc_flat.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    normalized = np.where(totals > 0, coinc_flat / safe_totals, 1.0 / n_ch)

    return normalized.reshape(n_shifts, n_data, n_ch)


def run_fast_memristive_singles(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    cfg: SimConfig,
    *,
    memristive_indices: Tuple[int, ...],
    output_modes: Tuple[Tuple[int, int], ...],
    offsets: Optional[np.ndarray] = None,
    return_class_probs: bool = False,
) -> np.ndarray:
    """
    Faster sequential NumPy path for discrete singles runs with memristive phases.

    The memristive recurrence still forces a scan over time, but each step only
    propagates a single-photon state vector instead of building the full unitary.
    """

    if cfg.output_mode != "singles":
        raise ValueError("run_fast_memristive_singles only supports singles mode")
    n_phases = cfg.total_mesh_phases
    n_memristive = len(memristive_indices)
    if n_memristive == 0:
        raise ValueError("run_fast_memristive_singles requires memristive phases")

    phases_base = np.asarray(params[:n_phases], dtype=np.float64)
    weights = np.asarray(params[n_phases:], dtype=np.float64)
    enc = _coerce_encoded_phases(encoded_phases)
    if enc.shape[1] != 1:
        raise ValueError(
            "run_fast_memristive_singles requires single-slot encoding (1D enc)"
        )
    enc = enc[:, 0]
    target_mode: Tuple[int, ...] = (
        cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)
    )
    num_pts = len(enc)
    n_classes = len(target_mode)
    target_idx = np.array(target_mode, dtype=int)
    if offsets is None:
        offsets_arr = np.array([0.0], dtype=np.float64)
    else:
        offsets_arr = np.asarray(offsets, dtype=np.float64).reshape(-1)
        if offsets_arr.size == 0:
            offsets_arr = np.array([0.0], dtype=np.float64)
    n_offsets = len(offsets_arr)

    if return_class_probs and n_classes > 1:
        preds = np.zeros((num_pts, n_classes), dtype=float)
    else:
        preds = np.zeros(num_pts, dtype=float)

    p1_hist = np.zeros((cfg.memory_depth, n_memristive), dtype=float)
    p2_hist = np.zeros((cfg.memory_depth, n_memristive), dtype=float)
    phases_work = phases_base.copy()

    for i, enc_phi_base in enumerate(enc):
        if i == 0:
            mem_phis = np.full(n_memristive, np.pi / 4, dtype=float)
        else:
            m1m = p1_hist.mean(axis=0)
            m2m = p2_hist.mean(axis=0)
            arg = np.clip(m1m + weights * m2m, 1e-9, 1.0 - 1e-9)
            mem_phis = np.arccos(np.sqrt(arg))

        p1_acc = np.zeros(n_memristive, dtype=float)
        p2_acc = np.zeros(n_memristive, dtype=float)
        if return_class_probs and n_classes > 1:
            target_acc = np.zeros(n_classes, dtype=float)
        else:
            target_acc = 0.0

        for off in offsets_arr:
            enc_phi = float(enc_phi_base + off)
            phases_work[:] = phases_base
            phases_work[list(memristive_indices)] = mem_phis

            state_in = _basis_input_state(cfg.singles_input_mode, cfg.n_modes)
            enc_slots = cfg.encoding_slots
            if len(enc_slots) != 1:
                raise ValueError(
                    "memristive fast path supports only single-slot encoding"
                )
            idx = int(enc_slots[0])
            phases_work[idx] = (phases_work[idx] + enc_phi) % (2 * np.pi)

            state_out = _propagate_state_vector(phases_work, state_in, cfg.n_modes)
            probs = np.abs(state_out) ** 2

            for j, (m1, m2) in enumerate(output_modes):
                p1_acc[j] += float(probs[m1])
                p2_acc[j] += float(probs[m2])

            if return_class_probs and n_classes > 1:
                target_acc += probs[target_idx]
            else:
                target_acc += float(probs[target_idx].mean())

        slot = i % cfg.memory_depth
        p1_hist[slot, :] = p1_acc / n_offsets
        p2_hist[slot, :] = p2_acc / n_offsets
        if return_class_probs and n_classes > 1:
            preds[i, :] = target_acc / n_offsets
        else:
            preds[i] = float(target_acc / n_offsets)

    return preds


def singles_prob_from_unitary(
    u: np.ndarray, singles_input_mode: int, target_mode: Tuple[int, ...]
) -> float:
    """Scalar target probability (possibly averaged) for singles output_mode."""
    enc_col = int(singles_input_mode)
    s = sum(np.abs(u[int(m), enc_col]) ** 2 for m in target_mode)
    if len(target_mode) > 1:
        s /= len(target_mode)
    return float(np.real(s))


def singles_class_probs_from_unitary(
    u: np.ndarray, singles_input_mode: int, target_mode: Tuple[int, ...]
) -> np.ndarray:
    enc_col = int(singles_input_mode)
    return np.array([np.abs(u[int(m), enc_col]) ** 2 for m in target_mode], dtype=float)


def all_singles_from_unitary(u: np.ndarray, input_mode: int) -> np.ndarray:
    """Return singles probabilities for every output mode."""

    col = int(input_mode)
    return np.abs(u[:, col]) ** 2


def coincidence_raw_vector(
    u: np.ndarray, input_modes: Tuple[int, int], n_modes: int
) -> np.ndarray:
    """Collision-free coincidence probabilities for one unitary, shape (n_cc,)."""
    return _coincidence_raw_batch(u[np.newaxis, ...], input_modes, n_modes)[0]


def all_coincidences_from_unitary(
    u: np.ndarray, input_modes: Tuple[int, int]
) -> np.ndarray:
    """Return coincidence probabilities for all detector pairs."""

    return coincidence_raw_vector(
        u, (int(input_modes[0]), int(input_modes[1])), u.shape[0]
    )


def unitary_for_point(
    phases: np.ndarray,
    enc_phi: Union[float, np.ndarray],
    n_modes: int,
    encoding_phase_idx: Union[int, Tuple[int, ...]],
    n_layers: int = 1,
) -> np.ndarray:
    """Full unitary for one data point (internal mesh encoding only)."""
    enc = _coerce_encoded_phases(np.asarray(enc_phi, dtype=np.float64))
    slots = normalize_encoding_phase_idx(encoding_phase_idx, n_layers, n_modes)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    return clements_stack_unitary_batch(phases, enc, slots, n_modes, n_layers)[0]
