from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .autograd import MemristorLossPSR
from .coincidence import nfold_channel_count
from .config import SimConfig
from .simulation import run_simulation_sequence_np


class PhotonicModel(torch.nn.Module):
    """
    PyTorch model class for photonic circuit training.
    Args:
        init_theta (Sequence[float]): Initial parameter values.
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        phase_idx (Sequence[int]): Indices of phase parameters (trainable mesh phases).
        sim_cfg (SimConfig): Circuit, simulation, and task parameters.
    """

    def __init__(
        self,
        init_theta: Sequence[float],
        enc_np: np.ndarray,
        phase_idx: Sequence[int],
        sim_cfg: SimConfig,
        head: bool = True,
    ) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.phase_idx = phase_idx
        self.sim_cfg = sim_cfg

        if sim_cfg.output_mode == "singles":
            n_features = sim_cfg.n_modes
        else:
            if sim_cfg.working_detectors is None:
                raise ValueError(
                    "coincidence linear head requires working_detectors in SimConfig"
                )
            n_features = nfold_channel_count(
                len(sim_cfg.working_detectors), int(sum(sim_cfg.input_state))
            )
        out_dim = 1 if sim_cfg.loss_type == "mse" else sim_cfg.n_classes
        self.head = (
            torch.nn.Linear(n_features, out_dim, bias=True, dtype=torch.float64)
            if head
            else torch.nn.Identity()
        )

    def predict(
        self,
        enc_np: np.ndarray,
        *,
        theta_np: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
        n_swipe: Optional[int] = None,
        swipe_span: Optional[float] = None,
        uq_pass_cfg: bool = False,
    ) -> np.ndarray:
        """Photonic features plus linear head (no autograd). Matches training readout.

        If ``uq_pass_cfg`` is True, use discrete UQ settings (``n_swipe=0``, ``swipe_span=0``,
        ``noise_std=None``) like :meth:`Experiment.run_uncertainty_analysis`, while still
        honoring ``n_samples`` when provided.
        """
        if uq_pass_cfg:
            cfg = self.sim_cfg.replace(
                n_samples=n_samples
                if n_samples is not None
                else self.sim_cfg.n_samples,
                n_swipe=0,
                swipe_span=0.0,
                noise_std=None,
            )
        else:
            cfg = self.sim_cfg.replace(
                n_samples=n_samples
                if n_samples is not None
                else self.sim_cfg.n_samples,
                n_swipe=n_swipe if n_swipe is not None else self.sim_cfg.n_swipe,
                swipe_span=swipe_span
                if swipe_span is not None
                else self.sim_cfg.swipe_span,
            )
        feature_cfg = MemristorLossPSR._feature_cfg(cfg)
        theta_use = (
            theta_np if theta_np is not None else self.theta.detach().cpu().numpy()
        )
        preds = run_simulation_sequence_np(
            theta_use,
            enc_np,
            feature_cfg,
            return_class_probs=True,
        )
        if preds.ndim == 1:
            preds = preds[:, None]
        device = self.theta.device
        features = torch.from_numpy(preds).double().to(device)
        with torch.no_grad():
            out = self.head(features)
            if self.sim_cfg.loss_type == "mse":
                return out.squeeze(-1).cpu().numpy()
            return F.softmax(out, dim=-1).cpu().numpy()

    def forward(
        self, y: Tensor, n_samples: int, n_swipe: int, swipe_span: float
    ) -> Tensor:
        """
        Computes the loss from photonic features with a trainable linear head.
        Args:
            y (Tensor): Regression targets or class labels.
            n_samples (int): Number of samples for the Sampler.
            n_swipe (int): Number of phase points per data point (0 for discrete).
            swipe_span (float): Total phase span for swiping.
        Returns:
            Tensor: Scalar loss value.
        """
        cfg = self.sim_cfg.replace(
            n_samples=n_samples, n_swipe=n_swipe, swipe_span=swipe_span
        )
        features = MemristorLossPSR.apply(
            self.theta,
            self.enc,
            self.phase_idx,
            cfg,
        )
        preds = self.head(features)
        if self.sim_cfg.loss_type == "mse":
            return F.mse_loss(preds.squeeze(-1), y.double())
        return F.cross_entropy(preds, y.long())
