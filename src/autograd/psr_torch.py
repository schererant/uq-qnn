from functools import lru_cache
from typing import Sequence, Tuple, Any
import numpy as np
import torch
from torch import Tensor

# Helper for PSR coefficients
@lru_cache(maxsize=None)
def photonic_psr_coeffs_torch(n: int) -> Tuple[Tensor, Tensor]:
    num_psr_terms = 2 * n
    psr_shifts = 2 * np.pi * np.arange(1, num_psr_terms + 1) / (2 * n + 1)
    grad_vector = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
    psr_coeffs_full = np.fft.ifft(np.concatenate(([0], grad_vector)))
    psr_shifts_t = torch.from_numpy(psr_shifts).double()
    psr_coeffs_t = torch.from_numpy(np.real_if_close(psr_coeffs_full[1:])).double()
    return psr_shifts_t, psr_coeffs_t

class MemristorLossPSR(torch.autograd.Function):
    """
    Autograd Function using PSR for photonic phase parameters and
    finite-difference for memristor weights, supporting both discrete and continuous modes.
    """
    @staticmethod
    def forward(
        ctx,
        theta: Tensor,
        enc_phases: Tensor,
        y: Tensor,
        circuit: Any,
        memory_depth: int,
        phase_idx: Sequence[int],
        n_photons: Sequence[int],
        n_samples: int,
        n_swipe: int = 0,
        swipe_span: float = 0.0,
    ) -> Tensor:
        from simulation.runner import SimulationRunner
        # Convert tensors to numpy
        theta_np = theta.detach().cpu().double().numpy()
        enc_np   = enc_phases.detach().cpu().double().numpy()
        y_np     = y.detach().cpu().double().numpy()
        discrete = (n_swipe == 0)
        runner = SimulationRunner(circuit, memory_depth=memory_depth)
        if discrete:
            preds = runner.run_sequence(theta_np, enc_np, n_samples)
        else:
            preds = runner.run_sequence(theta_np, enc_np, n_samples, n_swipe=n_swipe, swipe_span=swipe_span)
        loss_val = 0.5 * np.mean((preds - y_np) ** 2)
        ctx.save_for_backward(theta.detach(), enc_phases.detach(), y.detach())
        ctx.circuit      = circuit
        ctx.discrete     = discrete
        ctx.phase_idx    = list(phase_idx)
        ctx.n_photons    = list(n_photons)
        ctx.n_swipe      = n_swipe
        ctx.swipe_span   = swipe_span
        ctx.memory_depth = memory_depth
        ctx.n_samples    = n_samples
        ctx.preds_np     = preds
        return torch.tensor(loss_val, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out: Tensor):
        from simulation.runner import SimulationRunner
        theta, enc_tensor, y = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np   = enc_tensor.cpu().double().numpy()
        y_np     = y.cpu().double().numpy()
        preds    = ctx.preds_np
        N        = y.numel()
        dL_df    = (preds - y_np) / N
        grads = np.zeros_like(theta_np)
        eps   = 1e-3
        runner = SimulationRunner(ctx.circuit, memory_depth=ctx.memory_depth)
        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, coeffs = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])
            df_dtheta = np.zeros_like(preds)
            for s, c in zip(shifts.numpy(), coeffs.numpy()):
                theta_shift = theta_np.copy()
                theta_shift[p_idx] += s
                if ctx.discrete:
                    out = runner.run_sequence(theta_shift, enc_np, ctx.n_samples)
                else:
                    out = runner.run_sequence(theta_shift, enc_np, ctx.n_samples, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span)
                df_dtheta += c * out
            grads[p_idx] = np.real(np.dot(dL_df, df_dtheta))
        # Finite-difference for memristor weights
        weight_idxs = set(range(len(theta_np))) - set(ctx.phase_idx)
        for idx in weight_idxs:
            theta_p = theta_np.copy(); theta_m = theta_np.copy()
            theta_p[idx] += eps; theta_m[idx] -= eps
            theta_p[idx] = np.clip(theta_p[idx], 0.01, 1)
            theta_m[idx] = np.clip(theta_m[idx], 0.01, 1)
            if ctx.discrete:
                pred_p = runner.run_sequence(theta_p, enc_np, ctx.n_samples)
                pred_m = runner.run_sequence(theta_m, enc_np, ctx.n_samples)
            else:
                pred_p = runner.run_sequence(theta_p, enc_np, ctx.n_samples, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span)
                pred_m = runner.run_sequence(theta_m, enc_np, ctx.n_samples, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span)
            loss_p = 0.5 * np.mean((pred_p - y_np) ** 2)
            loss_m = 0.5 * np.mean((pred_m - y_np) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * eps)
        return (
            g_out * torch.from_numpy(grads).to(theta),  # theta
            None,  # enc_phases
            None,  # y
            None,  # circuit
            None,  # memory_depth
            None,  # phase_idx
            None,  # n_photons
            None,  # n_samples
            None,  # n_swipe
            None   # swipe_span
        ) 