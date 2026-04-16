from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .circuits import normalize_memristive_phase_idx
from .coincidence import nfold_channel_count, nfold_working_detector_tuples
from .config import SimConfig, psr_photon_counts_for_phases
from .numpy_backend import run_psr_forward_batch
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
    and continuous‐swipe modes.
    """

    @staticmethod
    def _feature_cfg(sim_cfg: SimConfig) -> SimConfig:
        """Build a SimConfig that requests full readout features."""
        if sim_cfg.output_mode == "singles":
            return sim_cfg.replace(
                target_mode=tuple(range(sim_cfg.n_modes)),
                n_classes=sim_cfg.n_modes,
            )
        if sim_cfg.working_detectors is None:
            raise ValueError(
                "coincidence features require working_detectors in SimConfig"
            )
        n_photons = int(sum(sim_cfg.input_state))
        n_channels = nfold_channel_count(len(sim_cfg.working_detectors), n_photons)
        target_cc_modes = nfold_working_detector_tuples(sim_cfg.working_detectors, n_photons)
        return sim_cfg.replace(
            target_mode=target_cc_modes[0],
            n_classes=n_channels,
            loss_type="cross_entropy",
        )

    @staticmethod
    def forward(
        ctx,
        theta: Tensor,
        enc_phases: Tensor,
        phase_idx: Sequence[int],
        sim_cfg: SimConfig,
    ) -> Tensor:
        theta_np = theta.detach().cpu().double().numpy()
        enc_np = enc_phases.detach().cpu().double().numpy()
        feature_cfg = MemristorLossPSR._feature_cfg(sim_cfg)
        n_photons = psr_photon_counts_for_phases(sim_cfg, len(phase_idx))

        preds = run_simulation_sequence_np(
            theta_np,
            enc_np,
            feature_cfg,
            return_class_probs=True,
        )
        if preds.ndim == 1:
            preds = preds[:, None]

        ctx.save_for_backward(theta.detach(), enc_phases.detach())
        ctx.phase_idx = list(phase_idx)
        ctx.n_photons = list(n_photons)
        ctx.feature_cfg = feature_cfg
        return torch.from_numpy(preds).to(device=theta.device, dtype=theta.dtype)

    @staticmethod
    def backward(ctx, g_out: Tensor):  # ty: ignore[invalid-method-override]
        cfg = ctx.feature_cfg
        theta, enc_tensor = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np = enc_tensor.cpu().double().numpy()
        g_out_np = g_out.detach().cpu().double().numpy()
        if g_out_np.ndim == 1:
            g_out_np = g_out_np[:, None]

        grads = np.zeros_like(theta_np)
        eps = 1e-3

        # Determine whether the fast batched PSR path is applicable.
        # Conditions: numpy backend, discrete mode (n_swipe==0), no memristive phases.
        # Memristive circuits fall back to the sequential path because their feedback
        # dynamics depend on the full sequential scan over data points.
        _n_ph = cfg.n_modes * (cfg.n_modes - 1)
        _no_mem = (
            len(normalize_memristive_phase_idx(cfg.memristive_phase_idx, cfg.n_modes, _n_ph)) == 0
        )
        _use_psr_batch = cfg.backend == "numpy" and cfg.n_swipe == 0 and _no_mem

        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, coeffs = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])
            shifts_np = shifts.numpy()
            coeffs_np = coeffs.numpy()

            if _use_psr_batch:
                # One clements_unitary_batch call for all 2n shifts of this phase.
                outs = run_psr_forward_batch(
                    theta_np, p_idx, shifts_np, enc_np, cfg
                )                                          # (n_shifts, n_data, n_features)
                df_dtheta = np.einsum("s,sdf->df", coeffs_np, outs)
            else:
                df_dtheta = np.zeros_like(g_out_np)
                for shift, coeff in zip(shifts_np, coeffs_np):
                    theta_shift = theta_np.copy()
                    theta_shift[p_idx] += shift
                    out = run_simulation_sequence_np(
                        theta_shift,
                        enc_np,
                        cfg,
                        return_class_probs=True,
                    )
                    if out.ndim == 1:
                        out = out[:, None]
                    df_dtheta += coeff * out

            grads[p_idx] = np.real(np.sum(g_out_np * df_dtheta))

        # Finite-difference for memristor weight parameters
        n_phases = cfg.n_modes * (cfg.n_modes - 1)
        weight_idxs = range(n_phases, len(theta_np))
        for idx in weight_idxs:
            theta_p = theta_np.copy()
            theta_m = theta_np.copy()
            theta_p[idx] += eps
            theta_m[idx] -= eps
            theta_p[idx] = np.clip(theta_p[idx], 0.01, 1.0)
            theta_m[idx] = np.clip(theta_m[idx], 0.01, 1.0)
            pred_p = run_simulation_sequence_np(
                theta_p,
                enc_np,
                cfg,
                return_class_probs=True,
            )
            pred_m = run_simulation_sequence_np(
                theta_m,
                enc_np,
                cfg,
                return_class_probs=True,
            )
            if pred_p.ndim == 1:
                pred_p = pred_p[:, None]
            if pred_m.ndim == 1:
                pred_m = pred_m[:, None]
            grads[idx] = np.real(np.sum(g_out_np * ((pred_p - pred_m) / (2 * eps))))

        return (
            torch.from_numpy(grads).to(theta),
            None,
            None,
            None,
        )
