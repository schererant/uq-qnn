from typing import Sequence, Tuple, List, Any
import numpy as np
import torch
from torch import Tensor
from autograd.psr_torch import MemristorLossPSR

class PhotonicModel(torch.nn.Module):
    """
    PyTorch model class for photonic memristor circuit training.
    Args:
        circuit: The quantum circuit instance (should be FullCircuit)
        init_theta (Sequence[float]): Initial parameter values.
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        memory_depth (int): Memory buffer depth.
        phase_idx (Sequence[int]): Indices of phase parameters.
        n_photons (Sequence[int]): Number of photons for each phase.
    """
    def __init__(self, circuit: Any, init_theta: Sequence[float], enc_np: np.ndarray, y_np: np.ndarray,
                 memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int]) -> None:
        super().__init__()
        self.circuit = circuit
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.register_buffer("y", torch.from_numpy(y_np).double())
        self.memory_depth, self.phase_idx, self.n_photons = memory_depth, phase_idx, n_photons

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
        return MemristorLossPSR.apply(self.theta, self.enc, self.y,
                                      self.circuit,
                                      self.memory_depth, self.phase_idx, self.n_photons, n_samples, n_swipe, swipe_span)

def _init_theta(rng: np.random.Generator) -> np.ndarray:
    return np.array([
        rng.uniform(0.01, 1) * 2 * np.pi,   # φ1
        rng.uniform(0.01, 1) * 2 * np.pi,   # φ3
        rng.uniform(0.01, 1)                # w
    ])

def train_pytorch_generic(
    circuit: Any,
    enc_np: np.ndarray,
    y_np: np.ndarray,
    *,
    memory_depth: int = 2,
    lr: float = 0.03,
    epochs: int = 150,
    phase_idx: Sequence[int] = (0, 1),
    n_photons: Sequence[int] = (1, 1),
    seed: int = 42,
    n_samples: int = 1000,
    n_swipe: int = 0,
    swipe_span: float = 0.0
) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    init_theta = _init_theta(rng)
    model = PhotonicModel(circuit, init_theta, enc_np, y_np, memory_depth, phase_idx, n_photons)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        optim.zero_grad()
        loss = model(n_samples=n_samples, n_swipe=n_swipe, swipe_span=swipe_span)
        loss.backward()
        optim.step()
        with torch.no_grad():
            model.theta.data[2].clamp_(0.01, 1.0)        # w ∈ [0.01,1]
            model.theta.data[:2].remainder_(2 * np.pi)   # Phasen ∈ [0,2π)
        hist.append(loss.item())
    return model.theta.detach().cpu().numpy(), hist

def train_pytorch(
    circuit: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_swipe: int = 0,
    swipe_span: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, List[float]]:
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(circuit, enc, y, n_swipe=n_swipe, swipe_span=swipe_span, **kwargs) 