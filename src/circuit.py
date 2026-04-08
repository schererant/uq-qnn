from __future__ import annotations

import logging
from itertools import combinations, combinations_with_replacement
from math import factorial
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .config import CircuitConfig
from .numpy_backend import (
    _coincidence_raw_batch,
    _occupation_two_distinct_modes,
    _unitary_batch_internal_encoding,
    clements_unitary,
    coincidence_raw_vector,
    unitary_for_point,
)

logger = logging.getLogger(__name__)


@dataclass
class _VectorizedCache:
    """Internal helper to cache vectorized computations."""

    unitary_mesh: Optional[np.ndarray] = None


class PhotonicCircuit:
    """Clements mesh helper: singles / coincidences via the NumPy backend (internal encoding).

    Training-agnostic wrapper around :class:`~src.config.CircuitConfig` and
    unitary construction; does not depend on :class:`~src.config.SimConfig`.
    """

    def __init__(
        self,
        *,
        n_modes: int,
        phases: np.ndarray,
        circuit_config: CircuitConfig,
    ):
        self._cfg = circuit_config
        if self._cfg.n_modes != n_modes:
            raise ValueError(
                f"circuit_config.n_modes={self._cfg.n_modes} != n_modes={n_modes}"
            )
        self._cfg.validate()
        phases = np.asarray(phases, dtype=np.float64).ravel()
        expected = self._cfg.n_phases
        if len(phases) != expected:
            raise ValueError(
                f"{n_modes}-mode circuit requires {expected} phases, got {len(phases)}"
            )
        self._phases = phases.copy()
        self._cache = _VectorizedCache()

    # ------------------------------------------------------------------ props
    @property
    def config(self) -> CircuitConfig:
        return self._cfg

    @property
    def n_modes(self) -> int:
        return self._cfg.n_modes

    @property
    def n_phases(self) -> int:
        return self._cfg.n_phases

    @property
    def phases(self) -> np.ndarray:
        return self._phases.copy()

    # ----------------------------------------------------------------- helpers
    def _mesh_unitary(self) -> np.ndarray:
        if self._cache.unitary_mesh is None:
            self._cache.unitary_mesh = clements_unitary(self._phases, self.n_modes)
        return self._cache.unitary_mesh

    # ------------------------------------------------------------------ public
    def unitary(self, encoding_phase: float) -> np.ndarray:
        """Full ``n_modes × n_modes`` unitary for a single encoding phase."""

        return unitary_for_point(
            self._phases,
            float(encoding_phase),
            self.n_modes,
            self.config.encoding_phase_idx,
        )

    def singles(
        self,
        encoding_phase: float,
        *,
        input_mode: Optional[int] = None,
    ) -> np.ndarray:
        """All singles probabilities for one encoding phase."""

        if sum(int(x) for x in self.config.input_state) != 1:
            raise ValueError("singles() requires single-photon input_state in config")
        mode = self.config.singles_input_mode if input_mode is None else int(input_mode)
        u = self.unitary(encoding_phase)
        return np.abs(u[:, mode]) ** 2

    def singles_batch(
        self,
        encoding_phases: Sequence[float] | np.ndarray,
        *,
        input_mode: Optional[int] = None,
    ) -> np.ndarray:
        enc = np.asarray(encoding_phases, dtype=np.float64).ravel()
        if sum(int(x) for x in self.config.input_state) != 1:
            raise ValueError(
                "singles_batch() requires single-photon input_state in config"
            )
        mode = self.config.singles_input_mode if input_mode is None else int(input_mode)
        u_batch = _unitary_batch_internal_encoding(
            self._phases,
            enc,
            self.n_modes,
            self.config.encoding_phase_idx,
        )
        return np.abs(u_batch[:, :, mode]) ** 2

    def coincidences(
        self,
        input_state: Sequence[int] | np.ndarray | float | None = None,
        detector_mode: str = "click",
        *,
        unitary: Optional[np.ndarray] = None,
        encoding_phase: Optional[float] = None,
    ) -> Union[np.ndarray, Dict[Tuple[int, ...], float]]:
        """Coincidence probabilities.

        New API:
            ``coincidences(input_state, detector_mode='click'|'pnr', unitary=None)``
            returns a dict mapping output Fock-state tuples to probabilities for
            arbitrary photon number N.

        Backward-compatible API:
            ``coincidences(encoding_phase)`` returns the legacy 2-photon
            collision-free coincidence vector.
        """
        if encoding_phase is not None:
            legacy_phase = float(encoding_phase)
            occ = tuple(int(v) for v in self.config.input_state)
            if len(occ) != self.n_modes:
                raise ValueError(
                    f"legacy coincidences(encoding_phase=...) requires input_state length "
                    f"n_modes={self.n_modes}, got {len(occ)}"
                )
            pair = _occupation_two_distinct_modes(occ)
            if pair is None:
                raise ValueError(
                    "legacy coincidences(encoding_phase=...) requires two photons in "
                    "distinct modes (occupation with exactly two 1s)"
                )
            u_legacy = self.unitary(legacy_phase)
            a, b = pair
            return coincidence_raw_vector(u_legacy, (a, b), self.n_modes)

        if input_state is None:
            raise ValueError(
                "input_state is required for N-fold coincidence distributions; "
                "or pass encoding_phase for legacy 2-photon behavior"
            )

        if np.isscalar(input_state):
            legacy_phase = float(input_state)
            occ = tuple(int(v) for v in self.config.input_state)
            if len(occ) != self.n_modes:
                raise ValueError(
                    f"legacy coincidences(scalar phase) requires input_state length "
                    f"n_modes={self.n_modes}, got {len(occ)}"
                )
            pair = _occupation_two_distinct_modes(occ)
            if pair is None:
                raise ValueError(
                    "legacy coincidences(scalar phase) requires two photons in distinct modes"
                )
            u_legacy = self.unitary(legacy_phase)
            a, b = pair
            return coincidence_raw_vector(u_legacy, (a, b), self.n_modes)

        state = tuple(int(v) for v in np.asarray(input_state, dtype=int).ravel())
        if len(state) != self.n_modes:
            raise ValueError(
                f"input_state must have length n_modes={self.n_modes}, got {len(state)}"
            )
        if any(v < 0 for v in state):
            raise ValueError(f"input_state must be non-negative, got {state!r}")

        n_photons = int(sum(state))
        if n_photons <= 0:
            raise ValueError("input_state must contain at least one photon")
        if detector_mode not in ("click", "pnr"):
            raise ValueError(
                f"detector_mode must be 'click' or 'pnr', got {detector_mode!r}"
            )

        u = (
            self._mesh_unitary()
            if unitary is None
            else np.asarray(unitary, dtype=np.complex128)
        )
        if u.shape != (self.n_modes, self.n_modes):
            raise ValueError(
                f"unitary must have shape ({self.n_modes}, {self.n_modes}), got {u.shape}"
            )

        input_rows = self._expanded_mode_indices(state)
        if detector_mode == "click":
            outputs = self._enumerate_click_output_states(self.n_modes, n_photons)
        else:
            outputs = self._enumerate_pnr_output_states(self.n_modes, n_photons)

        probs: Dict[Tuple[int, ...], float] = {}
        for out_state in outputs:
            probs[out_state] = self._transition_probability(u, input_rows, out_state)
        return probs

    def coincidences_batch(
        self,
        encoding_phases: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        occ = tuple(int(v) for v in self.config.input_state)
        if len(occ) != self.n_modes:
            raise ValueError(
                f"coincidences_batch() requires input_state length n_modes={self.n_modes}"
            )
        pair = _occupation_two_distinct_modes(occ)
        if pair is None:
            raise ValueError(
                "coincidences_batch() requires two photons in distinct input modes "
                "(occupation with two 1s)"
            )
        enc = np.asarray(encoding_phases, dtype=np.float64).ravel()
        u_batch = _unitary_batch_internal_encoding(
            self._phases,
            enc,
            self.n_modes,
            self.config.encoding_phase_idx,
        )
        a, b = pair
        return _coincidence_raw_batch(u_batch, (a, b), self.n_modes)

    @staticmethod
    def accidentals_rate(singles_rates: Sequence[float], tau: float, N: int) -> float:
        """Expected accidental N-fold coincidence rate from singles rates."""
        if N < 2:
            raise ValueError(f"N must be >= 2 for coincidences, got {N}")
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")
        rates = np.asarray(singles_rates, dtype=float).ravel()
        if len(rates) < N:
            raise ValueError(f"need at least N={N} singles rates, got {len(rates)}")
        if np.any(rates < 0):
            raise ValueError("singles_rates must be non-negative")

        prefactor = (tau ** (N - 1)) * factorial(N - 1)
        total = 0.0
        for idx in combinations(range(len(rates)), N):
            total += float(np.prod(rates[list(idx)]))
        return prefactor * total

    def target_probability(
        self,
        encoding_phase: float,
        *,
        target_modes: Union[int, Tuple[int, ...]],
        average: bool = True,
    ) -> Union[float, np.ndarray]:
        singles = self.singles(encoding_phase)
        if isinstance(target_modes, int):
            return float(singles[int(target_modes)])
        modes = np.array(target_modes, dtype=int)
        subset = singles[modes]
        if average and len(modes) > 1:
            return float(subset.mean())
        return subset

    # ---------------------------------------------------------------- utilities
    def with_phases(self, phases: np.ndarray) -> "PhotonicCircuit":
        return PhotonicCircuit(
            n_modes=self.n_modes,
            phases=phases,
            circuit_config=self._cfg,
        )

    @classmethod
    def random(
        cls,
        n_modes: int,
        *,
        circuit_config: CircuitConfig,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "PhotonicCircuit":
        if kwargs:
            raise TypeError(f"PhotonicCircuit.random got unexpected kwargs: {kwargs!r}")
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_modes * (n_modes - 1))
        return cls(n_modes=n_modes, phases=phases, circuit_config=circuit_config)

    def __repr__(self) -> str:
        return (
            f"PhotonicCircuit(n_modes={self.n_modes}, "
            f"input_state={self.config.input_state!r}, "
            f"encoding_phase_idx={self.config.encoding_phase_idx})"
        )

    @staticmethod
    def _expanded_mode_indices(occupation: Sequence[int]) -> np.ndarray:
        indices = [
            mode for mode, count in enumerate(occupation) for _ in range(int(count))
        ]
        return np.asarray(indices, dtype=int)

    @staticmethod
    def _enumerate_click_output_states(
        n_modes: int, n_photons: int
    ) -> Sequence[Tuple[int, ...]]:
        states = []
        for fired in combinations(range(n_modes), n_photons):
            occ = [0] * n_modes
            for m in fired:
                occ[m] = 1
            states.append(tuple(occ))
        return states

    @staticmethod
    def _enumerate_pnr_output_states(
        n_modes: int, n_photons: int
    ) -> Sequence[Tuple[int, ...]]:
        states = []
        for mode_multiset in combinations_with_replacement(range(n_modes), n_photons):
            occ = [0] * n_modes
            for m in mode_multiset:
                occ[m] += 1
            states.append(tuple(occ))
        return states

    @staticmethod
    def _ryser_permanent(matrix: np.ndarray) -> complex:
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

    @classmethod
    def _transition_probability(
        cls,
        unitary: np.ndarray,
        input_rows: np.ndarray,
        output_state: Sequence[int],
    ) -> float:
        output_cols = cls._expanded_mode_indices(output_state)
        # Rows = output modes, cols = input modes (Perceval / singles scattering convention).
        sub_u = unitary[np.ix_(output_cols, input_rows)]
        perm = cls._ryser_permanent(sub_u)
        n_modes_u = int(unitary.shape[0])
        in_counts = np.bincount(np.asarray(input_rows, dtype=int), minlength=n_modes_u)
        in_norm = float(np.prod([factorial(int(c)) for c in in_counts]))
        out_norm = float(np.prod([factorial(int(mu)) for mu in output_state]))
        prob = (abs(perm) ** 2) / (in_norm * out_norm)
        return float(np.real_if_close(prob))
