from __future__ import annotations

import numpy as np
import torch

from src.autograd import MemristorLossPSR
from src.circuits import normalize_memristive_phase_idx
from src.config import SimConfig
from src.simulation import run_simulation_sequence_np


def test_multi_memristive_weight_gradients_match_finite_difference():
    n_modes = 3
    n_phases = n_modes * (n_modes - 1)
    memristive_phase_idx = (2, 5)
    enc_slot = 0
    mem = normalize_memristive_phase_idx(memristive_phase_idx, n_modes, n_phases)
    phase_idx = tuple(i for i in range(n_phases) if i not in mem and i != enc_slot)
    cfg = SimConfig(
        n_modes=n_modes,
        n_layers=1,
        input_state=(1, 0, 0),
        encoding_phase_idx=enc_slot,
        n_enc_features=None,
        photon_distinguishability=None,
        target_mode=(2,),
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=60,
        memory_depth=2,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="mse",
        n_classes=1,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )

    enc = np.array([0.2, 0.8, 1.1], dtype=np.float64)
    y = np.array([0.1, 0.7, 0.4], dtype=np.float64)
    theta0 = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 0.35, 0.75], dtype=np.float64)

    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    feats = MemristorLossPSR.apply(
        theta,
        torch.from_numpy(enc),
        phase_idx,
        cfg,
    )
    loss = 0.5 * torch.mean((feats[:, -1] - torch.from_numpy(y)) ** 2)
    loss.backward()
    assert theta.grad is not None
    grads = theta.grad.detach().cpu().numpy()

    def loss_fn(params: np.ndarray) -> float:
        preds = run_simulation_sequence_np(params, enc, cfg)
        return float(0.5 * np.mean((preds - y) ** 2))

    eps = 1e-3
    for idx in (n_phases, n_phases + 1):
        plus = theta0.copy()
        minus = theta0.copy()
        plus[idx] = np.clip(plus[idx] + eps, 0.01, 1.0)
        minus[idx] = np.clip(minus[idx] - eps, 0.01, 1.0)
        fd_grad = (loss_fn(plus) - loss_fn(minus)) / (2 * eps)
        np.testing.assert_allclose(grads[idx], fd_grad, rtol=1e-4, atol=1e-5)
