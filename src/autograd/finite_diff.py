import numpy as np
from typing import Sequence
from autograd.base import GradientMethod

class FiniteDiffGradient(GradientMethod):
    """
    Finite-difference gradient computation for all parameters.
    """
    def __init__(self, simulation_runner, n_samples: int, n_swipe: int = 0, swipe_span: float = 0.0, eps: float = 1e-3):
        self.simulation_runner = simulation_runner
        self.n_samples = n_samples
        self.n_swipe = n_swipe
        self.swipe_span = swipe_span
        self.eps = eps

    def compute_gradients(self, params: np.ndarray, encoded_phases: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = y.size
        grads = np.zeros_like(params)
        for idx in range(len(params)):
            theta_p = params.copy(); theta_m = params.copy()
            theta_p[idx] += self.eps; theta_m[idx] -= self.eps
            theta_p[idx] = np.clip(theta_p[idx], 0.01, 1)
            theta_m[idx] = np.clip(theta_m[idx], 0.01, 1)
            pred_p = self.simulation_runner.run_sequence(theta_p, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
            pred_m = self.simulation_runner.run_sequence(theta_m, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
            loss_p = 0.5 * np.mean((pred_p - y) ** 2)
            loss_m = 0.5 * np.mean((pred_m - y) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * self.eps)
        return grads 