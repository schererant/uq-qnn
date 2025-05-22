import numpy as np
import torch
from typing import Sequence, Tuple
from autograd.base import GradientMethod
from functools import lru_cache

class PSRGradient(GradientMethod):
    """
    Photonic parameter-shift rule (PSR) gradient computation for phase parameters.
    """
    def __init__(self, simulation_runner, memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int], n_samples: int, n_swipe: int = 0, swipe_span: float = 0.0):
        self.simulation_runner = simulation_runner
        self.memory_depth = memory_depth
        self.phase_idx = list(phase_idx)
        self.n_photons = list(n_photons)
        self.n_samples = n_samples
        self.n_swipe = n_swipe
        self.swipe_span = swipe_span

    @lru_cache(maxsize=None)
    def _photonic_psr_coeffs(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        num_psr_terms = 2 * n
        psr_shifts = 2 * np.pi * np.arange(1, num_psr_terms + 1) / (2 * n + 1)
        grad_vector = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
        psr_coeffs_full = np.fft.ifft(np.concatenate(([0], grad_vector)))
        return psr_shifts, np.real_if_close(psr_coeffs_full[1:])

    def compute_gradients(self, params: np.ndarray, encoded_phases: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = y.size
        preds = self.simulation_runner.run_sequence(params, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
        dL_df = (preds - y) / N
        grads = np.zeros_like(params)

        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(self.phase_idx):
            shifts, coeffs = self._photonic_psr_coeffs(self.n_photons[gate_i])
            df_dtheta = np.zeros_like(preds)
            for s, c in zip(shifts, coeffs):
                theta_shift = params.copy()
                theta_shift[p_idx] += s
                out = self.simulation_runner.run_sequence(theta_shift, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
                df_dtheta += c * out
            grads[p_idx] = np.real(np.dot(dL_df, df_dtheta))
        return grads 