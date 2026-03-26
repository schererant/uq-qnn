from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from torch import Tensor

from .autograd import MemristorLossPSR
from .config import SimConfig


class PhotonicModel(torch.nn.Module):
    """
    PyTorch model class for photonic circuit training.
    Args:
        init_theta (Sequence[float]): Initial parameter values.
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        phase_idx (Sequence[int]): Indices of phase parameters.
        n_photons (Sequence[int]): Number of photons for each phase.
        sim_cfg (SimConfig): Circuit, simulation, and task parameters.
    """

    def __init__(
        self,
        init_theta: Sequence[float],
        enc_np: np.ndarray,
        y_np: np.ndarray,
        phase_idx: Sequence[int],
        n_photons: Sequence[int],
        sim_cfg: SimConfig,
    ) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.register_buffer("y", torch.from_numpy(y_np).double())
        self.phase_idx = phase_idx
        self.n_photons = n_photons
        self.sim_cfg = sim_cfg

        # Validate inputs for classification
        if sim_cfg.loss_type == "cross_entropy":
            if sim_cfg.target_mode is None or len(sim_cfg.target_mode) != sim_cfg.n_classes:
                raise ValueError(
                    f"For classification with n_classes={sim_cfg.n_classes}, "
                    f"target_mode must have {sim_cfg.n_classes} elements, "
                    f"got {sim_cfg.target_mode}"
                )
            # Validate y shape
            if y_np.ndim == 1:
                # Integer labels or binary (0/1)
                if sim_cfg.n_classes == 2:
                    # Binary: OK
                    pass
                else:
                    # Multi-class: should be integer labels
                    if not np.all((y_np >= 0) & (y_np < sim_cfg.n_classes)):
                        raise ValueError(
                            f"For multi-class classification, y must contain integer labels "
                            f"in [0, {sim_cfg.n_classes - 1}], got min={y_np.min()}, max={y_np.max()}"
                        )
            elif y_np.ndim == 2:
                # One-hot encoding
                if y_np.shape[1] != sim_cfg.n_classes:
                    raise ValueError(
                        f"For classification, y shape[1] must equal n_classes={sim_cfg.n_classes}, "
                        f"got {y_np.shape[1]}"
                    )

    def forward(self, n_samples: int, n_swipe: int, swipe_span: float) -> Tensor:
        """
        Computes the loss using the custom autograd function.
        Args:
            n_samples (int): Number of samples for the Sampler.
            n_swipe (int): Number of phase points per data point (0 for discrete).
            swipe_span (float): Total phase span for swiping.
        Returns:
            Tensor: Scalar loss value.
        """
        cfg = self.sim_cfg.replace(
            n_samples=n_samples, n_swipe=n_swipe, swipe_span=swipe_span
        )
        return MemristorLossPSR.apply(
            self.theta,
            self.enc,
            self.y,
            self.phase_idx,
            self.n_photons,
            cfg,
        )
