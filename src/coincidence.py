"""
Coincidence measurement utilities for photonic circuits.

Coincidences are detection of two photons at different output modes within a time window.
For n modes and 2 photons, there are n*(n-1)/2 collision-free coincidence channels (CCjk with j < k).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union, cast, overload

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
    import perceval as pcvl

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
