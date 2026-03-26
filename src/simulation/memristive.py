from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class MemristiveState:
    """Encapsulates memristive feedback buffers and phase computation."""

    n_indices: int
    memory_depth: int
    output_modes: Tuple[Tuple[int, int], ...]
    encoding_mode: int

    def __post_init__(self) -> None:
        if self.n_indices != len(self.output_modes):
            raise ValueError(
                f"memristive output modes length {len(self.output_modes)} "
                f"must match number of indices {self.n_indices}"
            )
        self._p1 = np.zeros((self.memory_depth, self.n_indices), dtype=float)
        self._p2 = np.zeros((self.memory_depth, self.n_indices), dtype=float)

    @property
    def active(self) -> bool:
        return self.n_indices > 0

    def reset(self) -> None:
        self._p1.fill(0.0)
        self._p2.fill(0.0)

    def _slot(self, step: int) -> int:
        return step % self.memory_depth

    def current_phases(self, weights: Optional[np.ndarray], step: int) -> np.ndarray:
        if not self.active:
            return np.empty(0, dtype=float)
        if weights is None:
            raise ValueError("weights must be provided for memristive circuits")
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != self.n_indices:
            raise ValueError(
                f"Expected {self.n_indices} memristive weights, got {weights.shape[0]}"
            )
        if step == 0:
            return np.full(self.n_indices, np.pi / 4, dtype=float)
        m1m = self._p1.mean(axis=0)
        m2m = self._p2.mean(axis=0)
        arg = np.clip(m1m + weights * m2m, 1e-9, 1 - 1e-9)
        return np.arccos(np.sqrt(arg))

    def update_from_unitary(self, step: int, unitary: np.ndarray) -> None:
        if not self.active:
            return
        slot = self._slot(step)
        enc_col = self.encoding_mode
        for j, (m1, m2) in enumerate(self.output_modes):
            self._p1[slot, j] = float(np.abs(unitary[m1, enc_col]) ** 2)
            self._p2[slot, j] = float(np.abs(unitary[m2, enc_col]) ** 2)

    def update_from_prob_arrays(
        self, step: int, p1_values: Sequence[float], p2_values: Sequence[float]
    ) -> None:
        if not self.active:
            return
        slot = self._slot(step)
        self._p1[slot, :] = np.asarray(p1_values, dtype=float)
        self._p2[slot, :] = np.asarray(p2_values, dtype=float)
