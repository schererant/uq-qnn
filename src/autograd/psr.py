import numpy as np
import torch
from typing import Sequence, Tuple, Union
from autograd.base import GradientMethod
from functools import lru_cache

class PSRGradient(GradientMethod):
    """
    Photonic parameter-shift rule (PSR) gradient computation for phase parameters, supporting numpy and torch backends.
    If backend='torch', always uses a custom torch.autograd.Function for gradients.
    """
    def __init__(self, simulation_runner, memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int], n_samples: int, n_swipe: int = 0, swipe_span: float = 0.0, backend: str = 'numpy'):
        self.simulation_runner = simulation_runner
        self.memory_depth = memory_depth
        self.phase_idx = list(phase_idx)
        self.n_photons = list(n_photons)
        self.n_samples = n_samples
        self.n_swipe = n_swipe
        self.swipe_span = swipe_span
        self.backend = backend

    @staticmethod
    @lru_cache(maxsize=None)
    def _photonic_psr_coeffs_numpy(n: int) -> Tuple[np.ndarray, np.ndarray]:
        num_psr_terms = 2 * n
        psr_shifts = 2 * np.pi * np.arange(1, num_psr_terms + 1) / (2 * n + 1)
        grad_vector = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
        psr_coeffs_full = np.fft.ifft(np.concatenate(([0], grad_vector)))
        return psr_shifts, np.real_if_close(psr_coeffs_full[1:])

    @staticmethod
    @lru_cache(maxsize=None)
    def _photonic_psr_coeffs_torch(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_psr_terms = 2 * n
        psr_shifts = 2 * np.pi * np.arange(1, num_psr_terms + 1) / (2 * n + 1)
        grad_vector = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
        psr_coeffs_full = np.fft.ifft(np.concatenate(([0], grad_vector)))
        psr_shifts_t = torch.from_numpy(psr_shifts).double()
        psr_coeffs_t = torch.from_numpy(np.real_if_close(psr_coeffs_full[1:])).double()
        return psr_shifts_t, psr_coeffs_t

    def compute_gradients(self, params: Union[np.ndarray, torch.Tensor], encoded_phases: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.backend == 'torch':
            # Always use the custom autograd Function for torch backend
            return MemristorLossPSR.apply(
                params, encoded_phases, y,
                self.simulation_runner, self.memory_depth, self.phase_idx, self.n_photons, self.n_samples, self.n_swipe, self.swipe_span
            )
        else:
            params = params.copy()
            encoded_phases = encoded_phases.copy()
            y = y.copy()
            N = y.size
            preds = self.simulation_runner.run_sequence(params, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
            dL_df = (preds - y) / N
            grads = np.zeros_like(params)
            # PSR gradients for photonic phases
            for gate_i, p_idx in enumerate(self.phase_idx):
                shifts, coeffs = self._photonic_psr_coeffs_numpy(self.n_photons[gate_i])
                df_dtheta = np.zeros_like(preds)
                for s, c in zip(shifts, coeffs):
                    theta_shift = params.copy()
                    theta_shift[p_idx] += s
                    out = self.simulation_runner.run_sequence(theta_shift, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
                    df_dtheta += c * out
                grads[p_idx] = np.real(np.dot(dL_df, df_dtheta))
            # Finite-difference for other weights
            eps = 1e-3
            weight_idxs = set(range(len(params))) - set(self.phase_idx)
            for idx in weight_idxs:
                theta_p = params.copy(); theta_m = params.copy()
                theta_p[idx] += eps; theta_m[idx] -= eps
                theta_p[idx] = np.clip(theta_p[idx], 0.01, 1)
                theta_m[idx] = np.clip(theta_m[idx], 0.01, 1)
                pred_p = self.simulation_runner.run_sequence(theta_p, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
                pred_m = self.simulation_runner.run_sequence(theta_m, encoded_phases, self.n_samples, self.n_swipe, self.swipe_span)
                loss_p = 0.5 * np.mean((pred_p - y) ** 2)
                loss_m = 0.5 * np.mean((pred_m - y) ** 2)
                grads[idx] = (loss_p - loss_m) / (2 * eps)
            return grads

class MemristorLossPSR(torch.autograd.Function):
    """
    Autograd Function using PSR for photonic phase parameters and
    finite-difference for memristor weights, supporting both discrete and continuous modes.
    """
    @staticmethod
    def forward(ctx, theta, encoded_phases, y, simulation_runner, memory_depth, phase_idx, n_photons, n_samples, n_swipe=0, swipe_span=0.0):
        # Convert tensors to numpy
        theta_np = theta.detach().cpu().double().numpy()
        enc_np   = encoded_phases.detach().cpu().double().numpy()
        y_np     = y.detach().cpu().double().numpy()
        discrete = (n_swipe == 0)
        # Run simulation
        preds = simulation_runner.run_sequence(theta_np, enc_np, n_samples, n_swipe, swipe_span)
        loss_val = 0.5 * np.mean((preds - y_np) ** 2)
        ctx.save_for_backward(theta.detach(), encoded_phases.detach(), y.detach())
        ctx.simulation_runner = simulation_runner
        ctx.memory_depth = memory_depth
        ctx.phase_idx = list(phase_idx)
        ctx.n_photons = list(n_photons)
        ctx.n_samples = n_samples
        ctx.n_swipe = n_swipe
        ctx.swipe_span = swipe_span
        ctx.preds_np = preds
        ctx.discrete = discrete
        return torch.tensor(loss_val, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out):
        theta, encoded_phases, y = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np   = encoded_phases.cpu().double().numpy()
        y_np     = y.cpu().double().numpy()
        preds    = ctx.preds_np
        N        = y.numel()
        dL_df    = (preds - y_np) / N
        grads = np.zeros_like(theta_np)
        eps   = 1e-3
        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, coeffs = PSRGradient._photonic_psr_coeffs_numpy(ctx.n_photons[gate_i])
            df_dtheta = np.zeros_like(preds)
            for s, c in zip(shifts, coeffs):
                theta_shift = theta_np.copy()
                theta_shift[p_idx] += s
                out = ctx.simulation_runner.run_sequence(theta_shift, enc_np, ctx.n_samples, ctx.n_swipe, ctx.swipe_span)
                df_dtheta += c * out
            grads[p_idx] = np.real(np.dot(dL_df, df_dtheta))
        # Finite-difference for memristor weights
        weight_idxs = set(range(len(theta_np))) - set(ctx.phase_idx)
        for idx in weight_idxs:
            theta_p = theta_np.copy(); theta_m = theta_np.copy()
            theta_p[idx] += eps; theta_m[idx] -= eps
            theta_p[idx] = np.clip(theta_p[idx], 0.01, 1)
            theta_m[idx] = np.clip(theta_m[idx], 0.01, 1)
            pred_p = ctx.simulation_runner.run_sequence(theta_p, enc_np, ctx.n_samples, ctx.n_swipe, ctx.swipe_span)
            pred_m = ctx.simulation_runner.run_sequence(theta_m, enc_np, ctx.n_samples, ctx.n_swipe, ctx.swipe_span)
            loss_p = 0.5 * np.mean((pred_p - y_np) ** 2)
            loss_m = 0.5 * np.mean((pred_m - y_np) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * eps)
        return (
            g_out * torch.from_numpy(grads).to(theta),  # theta
            None,  # encoded_phases
            None,  # y
            None,  # simulation_runner
            None,  # memory_depth
            None,  # phase_idx
            None,  # n_photons
            None,  # n_samples
            None,  # n_swipe
            None   # swipe_span
        ) 