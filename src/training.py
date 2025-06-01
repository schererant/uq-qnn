from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
import torch
from tqdm import tqdm

from .loss import PhotonicModel


def _init_theta(rng: np.random.Generator) -> np.ndarray:
    """
    Initializes model parameters randomly within specified ranges.
    Args:
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Array of initial parameter values [phi1, phi3, w].
    """
    return np.array([
        rng.uniform(0.01, 1) * 2 * np.pi,   # φ1
        rng.uniform(0.01, 1) * 2 * np.pi,   # φ3
        rng.uniform(0.01, 1)                # w
    ])


def train_pytorch_generic(
    enc_np: np.ndarray,
    y_np: np.ndarray,
    *,
    memory_depth: int = 2,
    lr: float = 0.03,
    epochs: int = 150,
    phase_idx: Sequence[int] = (0, 1),
    n_photons: Sequence[int] = (1, 1),
    seed: int = 42,
    n_samples: int,
    n_swipe: int = 0,
    swipe_span: float = 0.0
) -> Tuple[np.ndarray, List[float]]:
    """
    Trains the photonic model using PyTorch and returns optimized parameters and loss history.
    Args:
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        memory_depth (int): Memory buffer depth.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        phase_idx (Sequence[int]): Indices of phase parameters.
        n_photons (Sequence[int]): Number of photons for each phase.
        seed (int): Random seed for reproducibility.
        n_samples (int): Number of samples for the Sampler.
        n_swipe (int): Number of phase points per data point (0 for discrete).
        swipe_span (float): Total phase span for swiping.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    rng = np.random.default_rng(seed)
    init_theta = _init_theta(rng)
    model = PhotonicModel(init_theta, enc_np, y_np, memory_depth, phase_idx, n_photons)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in tqdm(range(epochs), desc="Training", ncols=100):
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
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_swipe: int = 0,
    swipe_span: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, List[float]]:
    """
    Unified training path for both discrete and continuous modes.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        n_swipe (int): Number of phase points per data point (0 for discrete).
        swipe_span (float): Total phase span for swiping.
        **kwargs: Additional arguments for training.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(enc, y, n_swipe=n_swipe, swipe_span=swipe_span, **kwargs)


def gradient_check() -> None:
    """
    Performs a gradient check comparing finite difference and analytic gradients.
    Prints the results and their absolute/max errors.
    Returns:
        None
    """
    from .data import get_data
    from .autograd import MemristorLossPSR
    from .simulation import run_simulation_sequence_np
    
    X, y, *_ = get_data(60, 0.0)
    enc = 2 * np.arccos(X)
    theta0 = np.array([1.2, 2.3, 0.5])
    mem_depth = 2
    n_samples = 5
    def L(params):
        return 0.5 * ((run_simulation_sequence_np(params, mem_depth, n_samples, encoded_phases=enc) - y) ** 2).mean()
    # Finite Difference
    eps = 1e-5
    num_grad = np.zeros_like(theta0)
    for k in range(len(theta0)):
        p_plus, p_minus = theta0.copy(), theta0.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        num_grad[k] = (L(p_plus) - L(p_minus)) / (2 * eps)
    th_t = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    loss = MemristorLossPSR.apply(th_t, torch.from_numpy(enc).double(), torch.from_numpy(y).double(),
                                  mem_depth, (0, 1), (1, 1), n_samples)
    loss.backward()
    psr_grad = th_t.grad.detach().cpu().numpy()
    print("Finite‑diff  :", num_grad)
    print("PSR / Torch :", psr_grad)
    print("Abs‑error   :", np.abs(num_grad - psr_grad))
    print("Max‑error   :", np.abs(num_grad - psr_grad).max())