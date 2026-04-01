from __future__ import annotations

"""
Core photonic circuit abstraction for the UQ-QNN project.

``PhotonicCircuit`` wraps the NumPy backend unitary construction and exposes a
compact API for obtaining singles / coincidence probabilities from a set of
mesh phases.  It purposely ignores training-specific concerns so callers can
simulate circuits without touching ``SimConfig``.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .config import CircuitConfig
from .numpy_backend import (
    _coincidence_raw_batch,
    _full_unitary_separate_encoding_batch,
    clements_unitary,
    coincidence_raw_vector,
    unitary_for_point,
)


@dataclass
class _VectorizedCache:
    """Internal helper to cache vectorized computations."""

    unitary_mesh: Optional[np.ndarray] = None


class PhotonicCircuit:
    """Simple API to evaluate a rectangular Clements mesh."""

    def __init__(
        self,
        *,
        n_modes: int,
        phases: np.ndarray,
        encoding_mode: int = 0,
        encoding_phase_idx: Optional[int] = None,
    ):
        self._cfg = CircuitConfig(
            n_modes=n_modes,
            encoding_mode=encoding_mode,
            encoding_phase_idx=encoding_phase_idx,
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
            self.config.encoding_mode,
            self.config.encoding_phase_idx,
        )

    def singles(
        self,
        encoding_phase: float,
        *,
        input_mode: Optional[int] = None,
    ) -> np.ndarray:
        """All singles probabilities for one encoding phase."""

        mode = self.config.encoding_mode if input_mode is None else int(input_mode)
        u = self.unitary(encoding_phase)
        return np.abs(u[:, mode]) ** 2

    def singles_batch(
        self,
        encoding_phases: Sequence[float] | np.ndarray,
        *,
        input_mode: Optional[int] = None,
    ) -> np.ndarray:
        enc = np.asarray(encoding_phases, dtype=np.float64).ravel()
        mode = self.config.encoding_mode if input_mode is None else int(input_mode)
        if self.config.encoding_phase_idx is None:
            u_clem = self._mesh_unitary()
            u_batch = _full_unitary_separate_encoding_batch(
                u_clem,
                enc,
                self.config.encoding_mode,
                self.n_modes,
            )
            return np.abs(u_batch[:, :, mode]) ** 2
        probs = np.empty((len(enc), self.n_modes), dtype=float)
        for i, phi in enumerate(enc):
            probs[i] = self.singles(phi, input_mode=input_mode)
        return probs

    def coincidences(
        self,
        encoding_phase: float,
        *,
        input_modes: Tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        u = self.unitary(encoding_phase)
        return coincidence_raw_vector(
            u,
            (int(input_modes[0]), int(input_modes[1])),
            self.n_modes,
        )

    def coincidences_batch(
        self,
        encoding_phases: Sequence[float] | np.ndarray,
        *,
        input_modes: Tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        enc = np.asarray(encoding_phases, dtype=np.float64).ravel()
        if self.config.encoding_phase_idx is None:
            u_clem = self._mesh_unitary()
            u_batch = _full_unitary_separate_encoding_batch(
                u_clem,
                enc,
                self.config.encoding_mode,
                self.n_modes,
            )
        else:
            u_batch = np.empty(
                (len(enc), self.n_modes, self.n_modes), dtype=np.complex128
            )
            for i, phi in enumerate(enc):
                u_batch[i] = self.unitary(phi)
        return _coincidence_raw_batch(
            u_batch,
            (int(input_modes[0]), int(input_modes[1])),
            self.n_modes,
        )

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
            encoding_mode=self.config.encoding_mode,
            encoding_phase_idx=self.config.encoding_phase_idx,
        )

    @classmethod
    def random(
        cls,
        n_modes: int,
        *,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "PhotonicCircuit":
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_modes * (n_modes - 1))
        return cls(n_modes=n_modes, phases=phases, **kwargs)

    def __repr__(self) -> str:
        enc = (
            f"encoding_phase_idx={self.config.encoding_phase_idx}"
            if self.config.encoding_phase_idx is not None
            else f"encoding_mode={self.config.encoding_mode}"
        )
        return f"PhotonicCircuit(n_modes={self.n_modes}, {enc})"
