from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from .simulation import run_simulation_sequence_np
from .circuits import CircuitType



@lru_cache(maxsize=None)
def photonic_psr_coeffs_torch(n: int) -> Tuple[Tensor, Tensor]:
    """
    Compute the shift angles and coefficients for the photonic parameter-shift rule (PSR).

    The photonic PSR allows the exact calculation of gradients with respect to phase parameters
    in linear optical (photonic) quantum circuits. For a phase shifter acting on n photons, the
    gradient of an expectation value can be written as a weighted sum of expectation values at
    shifted parameter values:

        d/dθ f(θ) = sum_p c_p * f(θ + shift_p)

    where the number of terms is 2n, and the shifts and coefficients are determined by the photon number.

    Args:
        n (int): Number of photons passing through the phase shifter.

    Returns:
        Tuple[Tensor, Tensor]:
            - psr_shifts (Tensor): 1D tensor of length 2n containing the shift angles (in radians) to apply to the phase parameter.
            - psr_coeffs (Tensor): 1D tensor of length 2n containing the real coefficients for each shift, to be used in the weighted sum.

    References:
        - A Photonic Parameter-shift Rule: Enabling Gradient Computation for Photonic Quantum Computers (arXiv:2410.02726)
    """
    num_psr_terms = 2 * n
    psr_shifts = 2 * np.pi * np.arange(1, num_psr_terms + 1) / (2 * n + 1)
    grad_vector = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
    psr_coeffs_full = np.fft.ifft(np.concatenate(([0], grad_vector)))
    psr_shifts_t = torch.from_numpy(psr_shifts).double()
    psr_coeffs_t = torch.from_numpy(np.real_if_close(psr_coeffs_full[1:])).double()
    return psr_shifts_t, psr_coeffs_t


class MemristorLossPSR(torch.autograd.Function):
    """
    Autograd Function using PSR for photonic‐phase parameters and
    finite‐difference only for memristor weights, in both discrete‐phase
    and continuous‐swipe modes.
    """
    @staticmethod
    def forward(
        ctx,
        theta: Tensor,
        enc_phases: Tensor,
        y: Tensor,
        memory_depth: int,
        phase_idx: Sequence[int],
        n_photons: Sequence[int],
        n_samples: int,
        n_swipe: int,
        swipe_span: float,
        n_modes: int,
        encoding_mode: int,
        target_mode: Optional[Tuple[int, ...]] = None,
        circuit_type: CircuitType = CircuitType.MEMRISTOR,
    ) -> Tensor:
        discrete = (n_swipe == 0)
        theta_np = theta.detach().cpu().double().numpy()
        enc_np   = enc_phases.detach().cpu().double().numpy()
        y_np     = y.detach().cpu().double().numpy()

        if discrete:
            preds = run_simulation_sequence_np(
                theta_np, memory_depth, n_samples,
                encoded_phases=enc_np,
                circuit_type=circuit_type,
                n_modes=n_modes,
                encoding_mode=encoding_mode,
                target_mode=target_mode,
                n_swipe=n_swipe
            )
        else:
            preds = run_simulation_sequence_np(
                theta_np, memory_depth, n_samples,
                encoded_phases=enc_np, n_swipe=n_swipe, swipe_span=swipe_span,
                circuit_type=circuit_type,
                n_modes=n_modes,
                encoding_mode=encoding_mode,
                target_mode=target_mode
            )

        loss_val = 0.5 * np.mean((preds - y_np) ** 2)

        ctx.save_for_backward(theta.detach(),
                              enc_phases.detach(),
                              y.detach())
        ctx.discrete     = discrete
        ctx.phase_idx    = list(phase_idx)
        ctx.n_photons    = list(n_photons)
        ctx.n_swipe      = n_swipe
        ctx.swipe_span   = swipe_span
        ctx.memory_depth = memory_depth
        ctx.n_samples    = n_samples
        ctx.preds_np     = preds
        ctx.circuit_type = circuit_type
        ctx.n_modes      = n_modes
        ctx.encoding_mode = encoding_mode
        ctx.target_mode  = target_mode

        return torch.tensor(loss_val, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out: Tensor):
        theta, enc_tensor, y = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np   = enc_tensor.cpu().double().numpy()
        y_np     = y.cpu().double().numpy()
        preds    = ctx.preds_np
        N        = y.numel()
        dL_df    = (preds - y_np) / N # For quadratic loss

        grads = np.zeros_like(theta_np)
        eps   = 1e-3


        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, coeffs = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])
            df_dθ = np.zeros_like(preds)
            for s, c in zip(shifts.numpy(), coeffs.numpy()):
                θ_shift = theta_np.copy()
                θ_shift[p_idx] += s
                if ctx.discrete:
                    out = run_simulation_sequence_np(
                        θ_shift, ctx.memory_depth, ctx.n_samples,
                        encoded_phases=enc_np,
                        circuit_type=ctx.circuit_type,
                        n_modes=ctx.n_modes,
                        encoding_mode=ctx.encoding_mode,
                        target_mode=ctx.target_mode,
                        n_swipe=ctx.n_swipe
                    )
                else:
                    out = run_simulation_sequence_np(
                        θ_shift, ctx.memory_depth, ctx.n_samples,
                        encoded_phases=enc_np, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span,
                        circuit_type=ctx.circuit_type,
                        n_modes=ctx.n_modes,
                        encoding_mode=ctx.encoding_mode,
                        target_mode=ctx.target_mode
                    )
                df_dθ += c * out
            grads[p_idx] = np.real(np.dot(dL_df, df_dθ))

        # Finite-difference for memristor weight parameters
        weight_idxs = set(range(len(theta_np))) - set(ctx.phase_idx)
        for idx in weight_idxs:
            θ_p = theta_np.copy(); θ_m = theta_np.copy()
            θ_p[idx] += eps; θ_m[idx] -= eps
            # Only clip the weight parameter (index 2), not the phases
            if idx == 2:  # weight parameter
                θ_p[idx] = np.clip(θ_p[idx], 0.01, 1)
                θ_m[idx] = np.clip(θ_m[idx], 0.01, 1)
            else:  # phase parameters can wrap around
                θ_p[idx] = θ_p[idx] % (2 * np.pi)
                θ_m[idx] = θ_m[idx] % (2 * np.pi)

            if ctx.discrete:
                pred_p = run_simulation_sequence_np(
                    θ_p, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np, n_swipe=ctx.n_swipe,
                    circuit_type=ctx.circuit_type,
                    n_modes=ctx.n_modes,
                    encoding_mode=ctx.encoding_mode,
                    target_mode=ctx.target_mode
                )
                pred_m = run_simulation_sequence_np(
                    θ_m, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np, n_swipe=ctx.n_swipe,
                    circuit_type=ctx.circuit_type,
                    n_modes=ctx.n_modes,
                    encoding_mode=ctx.encoding_mode,
                    target_mode=ctx.target_mode
                )
            else:
                pred_p = run_simulation_sequence_np(
                    θ_p, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span,
                    circuit_type=ctx.circuit_type,
                    n_modes=ctx.n_modes,
                    encoding_mode=ctx.encoding_mode,
                    target_mode=ctx.target_mode
                )
                pred_m = run_simulation_sequence_np(
                    θ_m, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span,
                    circuit_type=ctx.circuit_type,
                    n_modes=ctx.n_modes,
                    encoding_mode=ctx.encoding_mode,
                    target_mode=ctx.target_mode
                )

            loss_p = 0.5 * np.mean((pred_p - y_np) ** 2)
            loss_m = 0.5 * np.mean((pred_m - y_np) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * eps)

        return g_out * torch.from_numpy(grads).to(theta), None, None, None, None, None, None, None, None, None, None, None, None
