from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .config import CircuitConfig
from .numpy_backend import (
    _coincidence_raw_batch,
    _unitary_batch_internal_encoding,
    clements_unitary,
    coincidence_raw_vector,
    unitary_for_point,
)


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

        if len(self.config.input_state) != 1:
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
        if len(self.config.input_state) != 1:
            raise ValueError("singles_batch() requires single-photon input_state in config")
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
        encoding_phase: float,
    ) -> np.ndarray:
        if len(self.config.input_state) != 2:
            raise ValueError(
                "coincidences() requires two-photon input_state in circuit_config"
            )
        u = self.unitary(encoding_phase)
        a, b = int(self.config.input_state[0]), int(self.config.input_state[1])
        return coincidence_raw_vector(u, (a, b), self.n_modes)

    def coincidences_batch(
        self,
        encoding_phases: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        if len(self.config.input_state) != 2:
            raise ValueError(
                "coincidences_batch() requires two-photon input_state in circuit_config"
            )
        enc = np.asarray(encoding_phases, dtype=np.float64).ravel()
        u_batch = _unitary_batch_internal_encoding(
            self._phases,
            enc,
            self.n_modes,
            self.config.encoding_phase_idx,
        )
        a, b = int(self.config.input_state[0]), int(self.config.input_state[1])
        return _coincidence_raw_batch(u_batch, (a, b), self.n_modes)

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
