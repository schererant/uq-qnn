from __future__ import annotations

from typing import Sequence, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor

from .autograd import MemristorLossPSR


class PhotonicModel(torch.nn.Module):
    """
    PyTorch model class for photonic circuit training.
    Args:
        init_theta (Sequence[float]): Initial parameter values.
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        memory_depth (int): Memory buffer depth.
        phase_idx (Sequence[int]): Indices of phase parameters.
        n_photons (Sequence[int]): Number of photons for each phase.
        n_modes (int): Number of modes (3 for 3x3, 6 for 6x6, etc.).
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s).
        loss_type (str): Loss function type ('mse' for regression, 'cross_entropy' for classification).
        n_classes (int): Number of classes for classification (default: 1 for regression).
    """
    def __init__(self, init_theta: Sequence[float], enc_np: np.ndarray, y_np: np.ndarray,
                 memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int],
                 n_modes: int = 3, 
                 encoding_mode: int = 0, target_mode: Optional[Tuple[int, ...]] = None,
                 loss_type: str = 'mse', n_classes: int = 1,
                 memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.register_buffer("y", torch.from_numpy(y_np).double())
        self.memory_depth = memory_depth
        self.phase_idx = phase_idx
        self.n_photons = n_photons
        
        self.n_modes = n_modes
        self.encoding_mode = encoding_mode
        self.target_mode = target_mode
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.memristive_phase_idx = memristive_phase_idx
        
        # Validate inputs for classification
        if loss_type == 'cross_entropy':
            if target_mode is None or len(target_mode) != n_classes:
                raise ValueError(
                    f"For classification with n_classes={n_classes}, "
                    f"target_mode must have {n_classes} elements, got {target_mode}"
                )
            # Validate y shape
            if y_np.ndim == 1:
                # Integer labels or binary (0/1)
                if n_classes == 2:
                    # Binary: OK
                    pass
                else:
                    # Multi-class: should be integer labels
                    if not np.all((y_np >= 0) & (y_np < n_classes)):
                        raise ValueError(
                            f"For multi-class classification, y must contain integer labels "
                            f"in [0, {n_classes-1}], got min={y_np.min()}, max={y_np.max()}"
                        )
            elif y_np.ndim == 2:
                # One-hot encoding
                if y_np.shape[1] != n_classes:
                    raise ValueError(
                        f"For classification, y shape[1] must equal n_classes={n_classes}, "
                        f"got {y_np.shape[1]}"
                    )

    def forward(self, n_samples: int, n_swipe: int = 0, swipe_span: float = 0.0) -> Tensor:
        """
        Computes the loss using the custom autograd function.
        Args:
            n_samples (int): Number of samples for the Sampler.
            n_swipe (int): Number of phase points per data point (0 for discrete).
            swipe_span (float): Total phase span for swiping.
        Returns:
            Tensor: Scalar loss value.
        """
        return MemristorLossPSR.apply(
            self.theta, self.enc, self.y,
            self.memory_depth, self.phase_idx, self.n_photons, n_samples,
            n_swipe, swipe_span,
            self.n_modes, self.encoding_mode, self.target_mode,
            self.loss_type, self.n_classes,
            self.memristive_phase_idx
        )