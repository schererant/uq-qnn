from __future__ import annotations

from typing import Sequence, Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from .autograd import MemristorLossPSR
from .circuits import CircuitType


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
        circuit_type (str): Type of circuit architecture ('memristor' or 'clements').
        n_modes (int): Number of modes for Clements architecture.
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s).
    """
    def __init__(self, init_theta: Sequence[float], enc_np: np.ndarray, y_np: np.ndarray,
                 memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int],
                 circuit_type: CircuitType = CircuitType.MEMRISTOR, n_modes: int = 3, 
                 encoding_mode: int = 0, target_mode: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.register_buffer("y", torch.from_numpy(y_np).double())
        self.memory_depth = memory_depth
        self.phase_idx = phase_idx
        self.n_photons = n_photons
        
        # Circuit architecture parameters
        self.circuit_type = CircuitType.MEMRISTOR if circuit_type == CircuitType.MEMRISTOR else CircuitType.CLEMENTS
        self.n_modes = n_modes
        self.encoding_mode = encoding_mode
        self.target_mode = target_mode

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
            self.circuit_type, self.n_modes, self.encoding_mode, self.target_mode
        )