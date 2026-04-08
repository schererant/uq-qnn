"""
Coincidence measurement utilities for photonic circuits.

Coincidences are detection of two photons at different output modes within a time window.
For n modes and 2 photons, there are n*(n-1)/2 collision-free coincidence channels (CCjk with j < k).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np

if TYPE_CHECKING:
    import perceval as pcvl


def get_cc_mode_pairs(n_modes: int) -> List[tuple[int, int]]:
    """
    Return the list of (j, k) mode pairs with j < k for n_modes.
    Canonical order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2, n-1).
    """
    pairs = []
    for j in range(n_modes):
        for k in range(j + 1, n_modes):
            pairs.append((j, k))
    return pairs


def get_cc_labels(n_modes: int) -> List[str]:
    """
    Return CC labels in canonical order for n_modes.
    E.g. for 6 modes: CC01, CC02, ..., CC45.
    """
    pairs = get_cc_mode_pairs(n_modes)
    return [f"CC{j}{k}" for j, k in pairs]


def mode_pair_to_cc_index(j: int, k: int, n_modes: int) -> int:
    """Return the CC index for the mode pair (j, k) with j < k.

    E.g. for 6 modes: (0,1) → 0, (0,2) → 1, ..., (4,5) → 14.
    """
    a, b = (j, k) if j < k else (k, j)
    pairs = get_cc_mode_pairs(n_modes)
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    if (a, b) not in pair_to_idx:
        raise ValueError(
            f"mode pair ({a}, {b}) is not a valid CC pair for {n_modes} modes"
        )
    return pair_to_idx[(a, b)]


def working_detectors_to_cc_indices(
    working_detectors: Sequence[int],
    n_modes: int,
) -> List[int]:
    """
    Given mode indices of functioning detectors (e.g. [0, 1, 5]),
    return CC indices for all pairs from that set.

    E.g. working_detectors=[0, 1, 5] with n_modes=6:
    - Pairs: (0,1), (0,5), (1,5) -> CC01, CC05, CC15
    - CC labels order: CC01(0), CC02(1), CC03(2), CC04(3), CC05(4), CC12(5), ...
    - Indices: 0, 4, 8
    """
    pairs = get_cc_mode_pairs(n_modes)
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    working_set = set(int(m) for m in working_detectors)
    indices = []
    for j in working_set:
        for k in working_set:
            if j < k:
                pair = (j, k)
                if pair in pair_to_idx:
                    indices.append(pair_to_idx[pair])
    return sorted(indices)


def probs_to_singles(
    probs: Dict["pcvl.BasicState", float],
    n_modes: int,
) -> np.ndarray:
    """
    Extract single-photon counts (C0..C(n-1)) from SLOS probs dict.
    Cj = sum of probabilities over all states where mode j has >= 1 photon.
    """
    singles = np.zeros(n_modes, dtype=float)
    for state, prob in probs.items():
        for j in range(n_modes):
            if state[j] >= 1:
                singles[j] += prob
    return singles


def probs_to_coincidences(
    probs: Dict["pcvl.BasicState", float],
    n_modes: int,
) -> np.ndarray:
    """
    Extract collision-free coincidence probabilities from SLOS probs dict.
    For each pair (j, k) with j < k, CCjk = prob of state with exactly 1 photon in j and 1 in k.
    """
    import perceval as pcvl

    pairs = get_cc_mode_pairs(n_modes)
    coinc = np.zeros(len(pairs), dtype=float)
    for j, (mj, mk) in enumerate(pairs):
        state_list = [0] * n_modes
        state_list[mj] = 1
        state_list[mk] = 1
        state = pcvl.BasicState(state_list)
        coinc[j] = probs.get(state, 0.0)
    return coinc


@overload
def postselect_measurement(
    outcomes: Dict[str, float],
    working_cc_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    fallback_uniform: bool = False,
) -> Dict[str, float]: ...


@overload
def postselect_measurement(
    outcomes: np.ndarray,
    working_cc_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    fallback_uniform: bool = False,
) -> np.ndarray: ...


def postselect_measurement(
    outcomes: Union[Dict[str, float], np.ndarray],
    working_cc_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    fallback_uniform: bool = False,
) -> Union[Dict[str, float], np.ndarray]:
    """
    Postselect: keep only working CC channels, renormalize, others -> 0.

    Args:
        outcomes: Dict mapping label -> prob, or array of shape (n_cc,)
        working_cc_indices: Indices of working coincidence channels
        labels: CC labels (required if outcomes is dict)
        fallback_uniform: If total==0, assign 1/n to each working channel

    Returns:
        Same type as outcomes, with non-working channels zeroed and working normalized.
    """
    working_set = set(working_cc_indices)
    if isinstance(outcomes, dict):
        label_seq = (
            cast(Sequence[str], list(outcomes.keys())) if labels is None else labels
        )
        vals = np.array(
            [
                float(outcomes[str(label)]) if str(label) in outcomes else 0.0
                for label in label_seq
            ]
        )
    else:
        vals = np.asarray(outcomes).astype(float).copy()
        label_seq = [f"CC{i}" for i in range(len(vals))] if labels is None else labels

    total = sum(vals[i] for i in working_cc_indices if i < len(vals))
    if total <= 0:
        if fallback_uniform and len(working_cc_indices) > 0:
            n_w = len(working_cc_indices)
            for i in working_cc_indices:
                if i < len(vals):
                    vals[i] = 1.0 / n_w
        else:
            for i in working_cc_indices:
                if i < len(vals):
                    vals[i] = 0.0
    else:
        for i in range(len(vals)):
            if i in working_set:
                vals[i] = vals[i] / total
            else:
                vals[i] = 0.0

    if isinstance(outcomes, dict):
        return {label_seq[i]: float(vals[i]) for i in range(len(label_seq))}
    return vals


@overload
def apply_noise_to_outcomes(
    outcomes: Dict[str, float],
    noise_std: Union[float, Sequence[float]],
    working_cc_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]: ...


@overload
def apply_noise_to_outcomes(
    outcomes: np.ndarray,
    noise_std: Union[float, Sequence[float]],
    working_cc_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> np.ndarray: ...


def apply_noise_to_outcomes(
    outcomes: Union[Dict[str, float], np.ndarray],
    noise_std: Union[float, Sequence[float]],
    working_cc_indices: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Union[Dict[str, float], np.ndarray]:
    """
    Add Gaussian noise to working channels only, clip to [0,1], renormalize.

    Args:
        outcomes: Dict or array of probabilities
        noise_std: Single value for all, or per-channel list (length = len(working_cc_indices))
        working_cc_indices: Indices of channels to add noise to
        labels: Required if outcomes is dict
        seed: Random seed for reproducibility

    Returns:
        Noisy outcomes, same type as input.
    """
    rng = np.random.default_rng(seed)
    if isinstance(outcomes, dict):
        label_seq = (
            cast(Sequence[str], list(outcomes.keys())) if labels is None else labels
        )
        vals = np.array(
            [
                float(outcomes[str(label)]) if str(label) in outcomes else 0.0
                for label in label_seq
            ]
        )
    else:
        vals = np.asarray(outcomes).astype(float).copy()
        label_seq = [f"CC{i}" for i in range(len(vals))] if labels is None else labels

    stds = np.atleast_1d(np.asarray(noise_std, dtype=float))
    if len(stds) == 1:
        stds = np.full(len(working_cc_indices), stds[0])
    elif len(stds) != len(working_cc_indices):
        raise ValueError(
            f"noise_std must be scalar or length {len(working_cc_indices)}, got {len(stds)}"
        )

    for k, i in enumerate(working_cc_indices):
        if i < len(vals):
            vals[i] += rng.normal(0, stds[k])

    vals = np.clip(vals, 0.0, 1.0)
    total = vals.sum()
    if total > 0:
        vals = vals / total

    if isinstance(outcomes, dict):
        return {label_seq[i]: float(vals[i]) for i in range(len(label_seq))}
    return vals


# ── N-fold postselected coincidence (occupation-vector training stack) ─────


def expanded_mode_indices_from_occupation(occ: Sequence[int]) -> np.ndarray:
    """Repeat each mode index according to occupation (bosonic row expansion)."""

    rows: List[int] = []
    for mode, count in enumerate(occ):
        c = int(count)
        if c < 0:
            raise ValueError(f"occupation must be non-negative, got {occ!r}")
        rows.extend([mode] * c)
    return np.asarray(rows, dtype=int)


def detector_tuple_to_binary_occupation(
    det_tuple: Sequence[int], n_modes: int
) -> Tuple[int, ...]:
    """Map a collision-free detector tuple to a 0/1 occupation vector."""

    occ = [0] * int(n_modes)
    seen = set()
    for m in det_tuple:
        j = int(m)
        if j < 0 or j >= n_modes:
            raise ValueError(f"detector index {j} out of range for n_modes={n_modes}")
        if j in seen:
            raise ValueError(f"duplicate detector index in tuple {det_tuple!r}")
        seen.add(j)
        occ[j] = 1
    return tuple(occ)


def nfold_channel_count(n_working: int, n_photons: int) -> int:
    """Number of N-fold coincidence channels: C(W, N)."""

    if n_working < 0 or n_photons < 0:
        raise ValueError("n_working and n_photons must be non-negative")
    if n_photons > n_working:
        return 0
    return int(comb(n_working, n_photons))


def nfold_working_detector_tuples(
    working_detectors: Sequence[int], n_photons: int
) -> List[Tuple[int, ...]]:
    """
    Lexicographic N-tuples from sorted working detectors (same order as itertools.combinations).
    """

    wd_sorted = sorted(int(x) for x in working_detectors)
    if len(wd_sorted) != len(set(wd_sorted)):
        raise ValueError("working_detectors must contain unique mode indices")
    if n_photons < 1:
        raise ValueError(f"n_photons must be >= 1 for N-fold channels, got {n_photons}")
    if n_photons > len(wd_sorted):
        raise ValueError(
            f"n_photons={n_photons} exceeds len(working_detectors)={len(wd_sorted)}"
        )
    return list(combinations(wd_sorted, n_photons))


def canonical_sorted_detector_tuple(t: Sequence[int]) -> Tuple[int, ...]:
    """Sort and validate distinct detector indices (canonical N-fold channel key)."""

    tup = tuple(sorted(int(x) for x in t))
    if len(tup) != len(set(tup)):
        raise ValueError(f"detector indices must be distinct, got {t!r}")
    return tup


def nfold_tuple_to_channel_index(
    det_tuple: Sequence[int],
    working_detectors: Sequence[int],
    n_photons: int,
) -> int:
    """Index of an N-fold channel in the lex-ordered working-detector basis."""

    key = canonical_sorted_detector_tuple(det_tuple)
    if len(key) != n_photons:
        raise ValueError(
            f"expected target_mode length {n_photons}, got {len(key)} for tuple {key!r}"
        )
    wset = {int(x) for x in working_detectors}
    if not set(key).issubset(wset):
        raise ValueError(
            f"target_mode {key!r} must be contained in working_detectors={tuple(working_detectors)!r}"
        )
    tups = nfold_working_detector_tuples(working_detectors, n_photons)
    try:
        return tups.index(key)
    except ValueError as exc:
        raise ValueError(
            f"target_mode {key!r} is not in the N-fold coincidence basis for "
            f"working_detectors={tuple(working_detectors)!r}"
        ) from exc


def get_nfold_channel_labels(
    working_detectors: Sequence[int], n_photons: int
) -> List[str]:
    """Human-readable labels for postselection / noise helpers."""

    tups = nfold_working_detector_tuples(working_detectors, n_photons)
    return ["NF" + "".join(str(x) for x in t) for t in tups]


def migration_hint_legacy_input_state_pair(
    st: Sequence[int], n_modes: int
) -> Optional[str]:
    """If ``st`` looks like a legacy (j, k) injection pair, return a migration message."""

    if len(st) != 2:
        return None
    try:
        a, b = int(st[0]), int(st[1])
    except (TypeError, ValueError):
        return None
    if 0 <= a < n_modes and 0 <= b < n_modes:
        return (
            f"input_state {st!r} looks like a legacy pair of occupied mode indices. "
            f"Use a length-{n_modes} occupation vector instead, e.g. "
            f"tuple(1 if i in ({a}, {b}) else 0 for i in range({n_modes})) "
            f"for two photons in distinct modes."
        )
    return None


def probs_to_nfold_coincidences(
    probs: Dict["pcvl.BasicState", float],
    n_modes: int,
    working_detectors: Sequence[int],
    n_photons: int,
) -> np.ndarray:
    """Extract N-fold collision-free probabilities from an SLOS probs dict (Perceval)."""

    import perceval as pcvl

    tups = nfold_working_detector_tuples(working_detectors, n_photons)
    out = np.zeros(len(tups), dtype=float)
    for i, det_tup in enumerate(tups):
        occ = list(detector_tuple_to_binary_occupation(det_tup, n_modes))
        state = pcvl.BasicState(occ)
        out[i] = float(probs.get(state, 0.0))
    return out


# Hardware accidental correction constants
COINCIDENCE_WINDOW_FACTOR = 27e-12  # s, hardware-dependent
DEFAULT_C_WIN_NS = 170  # ns, typical coincidence window


def accidental_correction(
    cc_raw: float,
    c1: float,
    c2: float,
    int_time_sec: float,
    c_win_ns: float = DEFAULT_C_WIN_NS,
    window_factor: float = COINCIDENCE_WINDOW_FACTOR,
) -> float:
    """
    Correct raw coincidence count for accidental coincidences.

    acc = (c1 / int_time) * (c2 / int_time) * (c_win_ns * window_factor) * int_time
    CC_corrected = max(0, CC_raw - acc)
    """
    if int_time_sec <= 0:
        raise ValueError("int_time_sec must be positive")
    rate1 = c1 / int_time_sec
    rate2 = c2 / int_time_sec
    acc = rate1 * rate2 * (c_win_ns * window_factor) * int_time_sec
    return max(0.0, cc_raw - acc)
