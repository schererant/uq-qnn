import numpy as np
import torch
from typing import Sequence, Union
from autograd.base import GradientMethod

class FiniteDiffGradient(GradientMethod):
    """
    Finite-difference gradient computation for all parameters, supporting numpy and torch backends.
    If backend='torch', always uses a custom torch.autograd.Function for gradients.
    """
    def __init__(self, simulation_runner, n_samples: int, n_swipe: int = 0, swipe_span: float = 0.0, eps: float = 1e-3, backend: str = 'numpy'):
        self.simulation_runner = simulation_runner
        self.n_samples = n_samples
        self.n_swipe = n_swipe
        self.swipe_span = swipe_span
        self.eps = eps
        self.backend = backend

    def compute_gradients(self, params: Union[np.ndarray, torch.Tensor], encoded_phases: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.backend == 'torch':
            # Always use the custom autograd Function for torch backend
            return FiniteDiffAutograd.apply(
                params, encoded_phases, y,
                self.simulation_runner, self.n_samples, self.n_swipe, self.swipe_span, self.eps
            )
        else:
            params = params.copy()
            encoded_phases = encoded_phases.copy()
            y = y.copy()
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

class FiniteDiffAutograd(torch.autograd.Function):
    """
    Autograd Function for finite-difference gradient computation for all parameters.
    """
    @staticmethod
    def forward(ctx, theta, encoded_phases, y, simulation_runner, n_samples, n_swipe=0, swipe_span=0.0, eps=1e-3):
        theta_np = theta.detach().cpu().double().numpy()
        enc_np   = encoded_phases.detach().cpu().double().numpy()
        y_np     = y.detach().cpu().double().numpy()
        preds = simulation_runner.run_sequence(theta_np, enc_np, n_samples, n_swipe, swipe_span)
        loss_val = 0.5 * np.mean((preds - y_np) ** 2)
        ctx.save_for_backward(theta.detach(), encoded_phases.detach(), y.detach())
        ctx.simulation_runner = simulation_runner
        ctx.n_samples = n_samples
        ctx.n_swipe = n_swipe
        ctx.swipe_span = swipe_span
        ctx.eps = eps
        ctx.preds_np = preds
        return torch.tensor(loss_val, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out):
        theta, encoded_phases, y = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np   = encoded_phases.cpu().double().numpy()
        y_np     = y.cpu().double().numpy()
        grads = np.zeros_like(theta_np)
        eps = ctx.eps
        n_samples = ctx.n_samples
        n_swipe = ctx.n_swipe
        swipe_span = ctx.swipe_span
        simulation_runner = ctx.simulation_runner
        for idx in range(len(theta_np)):
            theta_p = theta_np.copy(); theta_m = theta_np.copy()
            theta_p[idx] += eps; theta_m[idx] -= eps
            theta_p[idx] = np.clip(theta_p[idx], 0.01, 1)
            theta_m[idx] = np.clip(theta_m[idx], 0.01, 1)
            pred_p = simulation_runner.run_sequence(theta_p, enc_np, n_samples, n_swipe, swipe_span)
            pred_m = simulation_runner.run_sequence(theta_m, enc_np, n_samples, n_swipe, swipe_span)
            loss_p = 0.5 * np.mean((pred_p - y_np) ** 2)
            loss_m = 0.5 * np.mean((pred_m - y_np) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * eps)
        return (
            g_out * torch.from_numpy(grads).to(theta),  # theta
            None,  # encoded_phases
            None,  # y
            None,  # simulation_runner
            None,  # n_samples
            None,  # n_swipe
            None,  # swipe_span
            None   # eps
        ) 