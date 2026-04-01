from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .config import SimConfig
from .simulation import run_simulation_sequence_np


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
    and continuous‐swipe modes. Supports both regression (MSE) and
    classification (cross-entropy) loss functions.
    """

    @staticmethod
    def forward(
        ctx,
        theta: Tensor,
        enc_phases: Tensor,
        y: Tensor,
        phase_idx: Sequence[int],
        n_photons: Sequence[int],
        sim_cfg: SimConfig,
    ) -> Tensor:
        theta_np = theta.detach().cpu().double().numpy()
        enc_np = enc_phases.detach().cpu().double().numpy()
        y_np = y.detach().cpu().double().numpy()

        # Determine if we need multi-class output
        return_class_probs = (
            sim_cfg.loss_type == "cross_entropy" and sim_cfg.n_classes > 1
        )

        preds = run_simulation_sequence_np(
            theta_np,
            enc_np,
            sim_cfg,
            return_class_probs=return_class_probs,
        )

        # Compute loss based on loss_type
        if sim_cfg.loss_type == "cross_entropy":
            # Classification: cross-entropy loss
            # y_np should be shape (K, n_classes) for multi-class or (K,) for binary
            # preds should be shape (K, n_classes)
            if preds.ndim == 1:
                # Binary classification: convert to 2D
                preds_2d = np.stack([1 - preds, preds], axis=1)
            else:
                preds_2d = preds

            # Add small epsilon to avoid log(0)
            eps = 1e-15
            preds_2d = np.clip(preds_2d, eps, 1 - eps)

            if y_np.ndim == 1:
                # Binary: convert to one-hot if needed
                if sim_cfg.n_classes == 2:
                    y_2d = np.stack([1 - y_np, y_np], axis=1)
                else:
                    # Multi-class with integer labels
                    y_2d = np.zeros((len(y_np), sim_cfg.n_classes))
                    y_2d[np.arange(len(y_np)), y_np.astype(int)] = 1.0
            else:
                y_2d = y_np

            # Cross-entropy: -Σ_c y_c * log(F^c_Θ(x))
            loss_val = -np.mean(np.sum(y_2d * np.log(preds_2d), axis=1))
            preds = preds_2d  # Store 2D predictions for backward
        else:
            # Regression: MSE loss
            loss_val = 0.5 * np.mean((preds - y_np) ** 2)

        ctx.save_for_backward(theta.detach(), enc_phases.detach(), y.detach())
        ctx.phase_idx = list(phase_idx)
        ctx.n_photons = list(n_photons)
        ctx.sim_cfg = sim_cfg
        ctx.preds_np = preds

        return torch.tensor(loss_val, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out: Tensor):  # ty: ignore[invalid-method-override]
        cfg = ctx.sim_cfg
        theta, enc_tensor, y = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np = enc_tensor.cpu().double().numpy()
        y_np = y.cpu().double().numpy()
        preds = ctx.preds_np
        N = y.numel()

        # Prepare predictions and targets based on loss type
        return_class_probs = cfg.loss_type == "cross_entropy" and cfg.n_classes > 1

        if cfg.loss_type == "cross_entropy":
            # Classification: prepare y and preds for Equation (15)
            if preds.ndim == 1:
                preds_2d = np.stack([1 - preds, preds], axis=1)
            else:
                preds_2d = preds

            if y_np.ndim == 1:
                if cfg.n_classes == 2:
                    y_2d = np.stack([1 - y_np, y_np], axis=1)
                else:
                    y_2d = np.zeros((len(y_np), cfg.n_classes))
                    y_2d[np.arange(len(y_np)), y_np.astype(int)] = 1.0
            else:
                y_2d = y_np

            # For classification, dL_df = -y_c / F^c_Θ(x) / K (from chain rule)
            eps = 1e-15
            preds_2d_clipped = np.clip(preds_2d, eps, 1 - eps)
            dL_df = -y_2d / preds_2d_clipped / N
        else:
            # Regression: dL_df = (preds - y_np) / N
            dL_df = (preds - y_np) / N

        grads = np.zeros_like(theta_np)
        eps = 1e-3

        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, coeffs = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])

            if cfg.loss_type == "cross_entropy" and cfg.n_classes > 1:
                # Classification PSR: Equation (15)
                # ∂L/∂θ = -(1/K) Σ_q c_q Σ_c (y_c / F^c_Θ(x)) · F^c_{Θ+Θ^q}(x)
                df_dθ = np.zeros((len(enc_np), cfg.n_classes))
                for s, c in zip(shifts.numpy(), coeffs.numpy()):
                    θ_shift = theta_np.copy()
                    θ_shift[p_idx] += s
                    out = run_simulation_sequence_np(
                        θ_shift,
                        enc_np,
                        cfg,
                        return_class_probs=True,
                    )
                    if out.ndim == 1:
                        out = np.stack([1 - out, out], axis=1)
                    df_dθ += c * out
                # Compute gradient: Σ_k Σ_c (y_c / F^c_Θ(x_k)) · F^c_{Θ+Θ^q}(x_k) for each q
                # Then sum over q with coefficients
                grads[p_idx] = np.real(np.sum(dL_df * df_dθ))
            else:
                # Regression PSR: Equation (8)
                df_dθ = (
                    np.zeros_like(preds) if preds.ndim == 1 else np.zeros(len(preds))
                )
                for s, c in zip(shifts.numpy(), coeffs.numpy()):
                    θ_shift = theta_np.copy()
                    θ_shift[p_idx] += s
                    out = run_simulation_sequence_np(
                        θ_shift,
                        enc_np,
                        cfg,
                        return_class_probs=return_class_probs,
                    )
                    if out.ndim > 1:
                        out = out[:, -1]  # Use last class for regression
                    df_dθ += c * out
                grads[p_idx] = np.real(np.dot(dL_df.flatten(), df_dθ))

        # Finite-difference for memristor weight parameters
        weight_idxs = sorted(set(range(len(theta_np))) - set(ctx.phase_idx))
        for idx in weight_idxs:
            θ_p = theta_np.copy()
            θ_m = theta_np.copy()
            θ_p[idx] += eps
            θ_m[idx] -= eps
            θ_p[idx] = np.clip(θ_p[idx], 0.01, 1.0)
            θ_m[idx] = np.clip(θ_m[idx], 0.01, 1.0)

            fdc_return_class = cfg.loss_type == "cross_entropy" and cfg.n_classes > 1

            pred_p = run_simulation_sequence_np(
                θ_p,
                enc_np,
                cfg,
                return_class_probs=fdc_return_class,
            )
            pred_m = run_simulation_sequence_np(
                θ_m,
                enc_np,
                cfg,
                return_class_probs=fdc_return_class,
            )

            # Compute loss based on loss_type
            if cfg.loss_type == "cross_entropy":
                # Classification loss
                if pred_p.ndim == 1:
                    pred_p_2d = np.stack([1 - pred_p, pred_p], axis=1)
                else:
                    pred_p_2d = pred_p
                if pred_m.ndim == 1:
                    pred_m_2d = np.stack([1 - pred_m, pred_m], axis=1)
                else:
                    pred_m_2d = pred_m

                eps_loss = 1e-15
                pred_p_2d = np.clip(pred_p_2d, eps_loss, 1 - eps_loss)
                pred_m_2d = np.clip(pred_m_2d, eps_loss, 1 - eps_loss)

                if y_np.ndim == 1:
                    if cfg.n_classes == 2:
                        y_2d = np.stack([1 - y_np, y_np], axis=1)
                    else:
                        y_2d = np.zeros((len(y_np), cfg.n_classes))
                        y_2d[np.arange(len(y_np)), y_np.astype(int)] = 1.0
                else:
                    y_2d = y_np

                loss_p = -np.mean(np.sum(y_2d * np.log(pred_p_2d), axis=1))
                loss_m = -np.mean(np.sum(y_2d * np.log(pred_m_2d), axis=1))
            else:
                # Regression loss
                loss_p = 0.5 * np.mean((pred_p - y_np) ** 2)
                loss_m = 0.5 * np.mean((pred_m - y_np) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * eps)

        return (
            g_out * torch.from_numpy(grads).to(theta),
            None,
            None,
            None,
            None,
            None,
        )
