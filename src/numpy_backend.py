"""
Fast NumPy backend for Clements linear-optical simulations.

Computes the unitary via the same MZI ordering as Perceval and extracts
probabilities using the standard bosonic (permanent) rule for collision-free
outputs. Vectorizes across data points when the mesh phases are shared
(non-memristive, or memristive with fixed mesh within a batch step).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .config import SimConfig
from .circuits import _clements_mzi_pairs
from .coincidence import (
    get_cc_mode_pairs,
    get_cc_labels,
    postselect_measurement,
    apply_noise_to_outcomes,
    working_detectors_to_cc_indices,
)


def _bs_2x2() -> np.ndarray:
    return np.array([[1, 1j], [1j, 1]], dtype=np.complex128) / np.sqrt(2)


def _mzi_unitary(phi_int: float, phi_ext: float) -> np.ndarray:
    phi_int = float(phi_int) % (2 * np.pi)
    phi_ext = float(phi_ext) % (2 * np.pi)
    bs = _bs_2x2()
    p_int = np.diag(np.array([1.0, np.exp(1j * phi_int)], dtype=np.complex128))
    p_ext = np.diag(np.array([1.0, np.exp(1j * phi_ext)], dtype=np.complex128))
    return p_ext @ bs @ p_int @ bs


def clements_unitary(phases: np.ndarray, n_modes: int) -> np.ndarray:
    """Full n_modes x n_modes unitary for the Clements mesh (matches Perceval ordering)."""
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    pairs = _clements_mzi_pairs(n_modes)
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


def _encoding_unitary_2x2_batch(enc_phases: np.ndarray) -> np.ndarray:
    """Batch of 2x2 encoding blocks: BS — PS(enc) — BS. Shape (n_data, 2, 2)."""
    phi = np.asarray(enc_phases, dtype=np.float64) % (2 * np.pi)
    bs = _bs_2x2()
    p = np.zeros(phi.shape + (2, 2), dtype=np.complex128)
    p[..., 0, 0] = 1.0
    p[..., 1, 1] = np.exp(1j * phi)
    # einsum: bs @ P @ bs for each slice
    return np.einsum("ij,...jk,kl->...il", bs, p, bs)


def _valid_encoding_mode(encoding_mode: int, n_modes: int) -> int:
    return min(max(0, int(encoding_mode)), n_modes - 2)


def _full_unitary_separate_encoding_batch(
    u_clem: np.ndarray, enc_phases: np.ndarray, encoding_mode: int, n_modes: int
) -> np.ndarray:
    """
    U_full = U_clem @ U_embed(enc) with a 2-mode encoding block on (m, m+1).
    Returns shape (n_data, n_modes, n_modes).
    """
    m = _valid_encoding_mode(encoding_mode, n_modes)
    e = _encoding_unitary_2x2_batch(enc_phases)
    n_data = e.shape[0]
    u_out = np.empty((n_data, n_modes, n_modes), dtype=np.complex128)
    u_c = u_clem  # (n_modes, n_modes)
    for j in range(n_modes):
        if j not in (m, m + 1):
            u_out[:, :, j] = u_c[:, j]
    # Columns m and m+1: (U_clem @ U_enc)[:, k] = U_clem[:, m:m+2] @ E[:, k] for k in {0,1} of 2x2 block
    u_out[:, :, m] = (
        e[:, 0, 0, None] * u_c[None, :, m] + e[:, 1, 0, None] * u_c[None, :, m + 1]
    )
    u_out[:, :, m + 1] = (
        e[:, 0, 1, None] * u_c[None, :, m] + e[:, 1, 1, None] * u_c[None, :, m + 1]
    )
    return u_out


def _singles_probabilities_batch(
    u_batch: np.ndarray,
    encoding_mode: int,
    target_mode: Tuple[int, ...],
    return_class_probs: bool,
) -> np.ndarray:
    """u_batch (n_data, n, n); photon input column = encoding_mode."""
    enc_col = int(encoding_mode)
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
    perm = (
        u_batch[:, mj, a] * u_batch[:, mk, b]
        + u_batch[:, mj, b] * u_batch[:, mk, a]
    )
    return np.abs(perm) ** 2


def _process_coincidence_rows(
    coinc: np.ndarray,
    working_cc_indices: Sequence[int],
    cc_labels: list,
    add_noise: bool,
    noise_std: Optional[Union[float, Sequence[float]]],
    return_class_probs: bool,
    preds: np.ndarray,
) -> None:
    n_data = coinc.shape[0]
    for i in range(n_data):
        out = postselect_measurement(
            coinc[i], working_cc_indices, cc_labels, fallback_uniform=True
        )
        if add_noise:
            out = apply_noise_to_outcomes(
                out, noise_std, working_cc_indices, cc_labels, seed=i
            )
        if return_class_probs and len(working_cc_indices) > 1:
            preds[i, :] = out[list(working_cc_indices)]
        else:
            preds[i] = out[working_cc_indices[0]] if working_cc_indices else 0.0


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
        cfg.target_mode
        if cfg.target_mode is not None
        else (cfg.n_modes - 1,)
    )
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    phases = np.asarray(params[:n_phases], dtype=np.float64)
    enc = np.asarray(encoded_phases, dtype=np.float64).reshape(-1)
    num_pts = len(enc)

    if cfg.output_mode == "coincidence":
        in_modes = (
            (1, 4)
            if cfg.n_modes >= 6
            else (0, 1)
            if cfg.input_modes is None
            else tuple(int(m) for m in cfg.input_modes)
        )
        wd = (
            tuple(cfg.working_detectors)
            if cfg.working_detectors is not None
            else (0, 1, 5)
        )
        working_cc_indices = working_detectors_to_cc_indices(wd, cfg.n_modes)
        cc_labels = get_cc_labels(cfg.n_modes)
        add_noise = cfg.noise_std is not None and (
            (isinstance(cfg.noise_std, (int, float)) and float(cfg.noise_std) > 0)
            or (
                hasattr(cfg.noise_std, "__len__")
                and len(cfg.noise_std) > 0
                and any(float(s) > 0 for s in cfg.noise_std)
            )
        )
        n_classes = len(working_cc_indices)
    else:
        n_classes = len(target_mode)
        add_noise = False
        working_cc_indices = ()
        cc_labels = []

    if return_class_probs and n_classes > 1:
        preds = np.zeros((num_pts, n_classes), dtype=float)
    else:
        preds = np.zeros(num_pts, dtype=float)

    if cfg.encoding_phase_idx is None:
        u_clem = clements_unitary(phases, cfg.n_modes)
        u_batch = _full_unitary_separate_encoding_batch(
            u_clem, enc, cfg.encoding_mode, cfg.n_modes
        )
    else:
        idx = int(cfg.encoding_phase_idx)
        u_batch = np.empty((num_pts, cfg.n_modes, cfg.n_modes), dtype=np.complex128)
        for i in range(num_pts):
            mp = phases.copy()
            mp[idx] = (mp[idx] + enc[i]) % (2 * np.pi)
            u_batch[i] = clements_unitary(mp, cfg.n_modes)

    if cfg.output_mode == "coincidence":
        coinc = _coincidence_raw_batch(u_batch, in_modes, cfg.n_modes)
        _process_coincidence_rows(
            coinc,
            working_cc_indices,
            cc_labels,
            add_noise,
            cfg.noise_std,
            return_class_probs,
            preds,
        )
    elif return_class_probs and n_classes > 1:
        preds[:, :] = _singles_probabilities_batch(
            u_batch, cfg.encoding_mode, target_mode, True
        )
    else:
        preds[:] = _singles_probabilities_batch(
            u_batch, cfg.encoding_mode, target_mode, False
        )

    return preds


def singles_prob_from_unitary(
    u: np.ndarray, encoding_mode: int, target_mode: Tuple[int, ...]
) -> float:
    """Scalar target probability (possibly averaged) for singles output_mode."""
    enc_col = int(encoding_mode)
    s = sum(np.abs(u[int(m), enc_col]) ** 2 for m in target_mode)
    if len(target_mode) > 1:
        s /= len(target_mode)
    return float(np.real(s))


def singles_class_probs_from_unitary(
    u: np.ndarray, encoding_mode: int, target_mode: Tuple[int, ...]
) -> np.ndarray:
    enc_col = int(encoding_mode)
    return np.array([np.abs(u[int(m), enc_col]) ** 2 for m in target_mode], dtype=float)


def coincidence_raw_vector(
    u: np.ndarray, input_modes: Tuple[int, int], n_modes: int
) -> np.ndarray:
    """Collision-free coincidence probabilities for one unitary, shape (n_cc,)."""
    return _coincidence_raw_batch(u[np.newaxis, ...], input_modes, n_modes)[0]


def unitary_for_point(
    phases: np.ndarray,
    enc_phi: float,
    n_modes: int,
    encoding_mode: int,
    encoding_phase_idx: Optional[int],
) -> np.ndarray:
    """Full unitary for one data point (separate or inline encoding)."""
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    enc_phi = float(enc_phi) % (2 * np.pi)
    if encoding_phase_idx is None:
        u_clem = clements_unitary(phases, n_modes)
        return _full_unitary_separate_encoding_batch(
            u_clem, np.array([enc_phi]), encoding_mode, n_modes
        )[0]
    mp = phases.copy()
    idx = int(encoding_phase_idx)
    mp[idx] = (mp[idx] + enc_phi) % (2 * np.pi)
    return clements_unitary(mp, n_modes)
