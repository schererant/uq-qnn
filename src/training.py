from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm

from .loss import PhotonicModel
from .simulation import _normalize_memristive_phase_idx


def _init_theta(
    rng: np.random.Generator,
    n_modes: int,
    memristive_phase_idx: Optional[Union[int, Sequence[int]]],
) -> np.ndarray:
    """
    Initializes model parameters. Architecture is always Clements.
    params = [phase_0, ..., phase_{n-1}] or [phases..., w_0, ..., w_{k-1}] if memristive.
    """
    expected_phases = n_modes * (n_modes - 1)
    phases = rng.uniform(0.0, 2 * np.pi, size=expected_phases)
    memristive_indices = _normalize_memristive_phase_idx(memristive_phase_idx, n_modes, expected_phases)
    if len(memristive_indices) == 0:
        return phases
    weights = rng.uniform(0.01, 1, size=len(memristive_indices))
    return np.concatenate([phases, weights])


def train_pytorch_generic(
    enc_np: np.ndarray,
    y_np: np.ndarray,
    *,
    memory_depth: int,
    lr: float,
    epochs: int,
    n_samples: int,
    n_swipe: int,
    swipe_span: float,
    n_modes: int,
    encoding_mode: int,
    target_mode: Optional[Tuple[int, ...]] = None,
    loss_type: str = 'mse',
    n_classes: int = 1,
    phase_idx: Optional[Sequence[int]] = None,
    n_photons: Optional[Sequence[int]] = None,
    seed: int = 42,
    memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None,
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]] = None,
    verbose: bool = False,
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
        n_modes (int): Number of modes for Clements architecture.
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s).
        loss_type (str): Loss function type ('mse' for regression, 'cross_entropy' for classification).
        n_classes (int): Number of classes for classification (default: 1 for regression).
        memristive_phase_idx (Optional[Union[int, Sequence[int]]]): Phase indices to make memristive.
            None or empty = no memristive. e.g. [2] or (2, 5) for one or two MZIs.
        memristive_output_modes (Optional[Sequence[Tuple[int, int]]]): For each memristive phase,
            the (mode_p1, mode_p2) output modes for feedback. None = use MZI's own modes.
        verbose (bool): If True, print per-epoch loss and final parameters.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    # Validate classification setup
    if loss_type == 'cross_entropy':
        if target_mode is None:
            if n_classes > n_modes:
                raise ValueError(
                    f"For {n_classes} classes, need at least {n_classes} modes, got {n_modes}"
                )
            target_mode = tuple(range(n_classes))
        elif len(target_mode) != n_classes:
            raise ValueError(
                f"For classification with n_classes={n_classes}, "
                f"target_mode must have {n_classes} elements, got {len(target_mode)}"
            )
    
    rng = np.random.default_rng(seed)
    init_theta = _init_theta(rng, n_modes, memristive_phase_idx)

    expected_phases = n_modes * (n_modes - 1)
    memristive_indices = _normalize_memristive_phase_idx(memristive_phase_idx, n_modes, expected_phases)
    phase_idx = tuple(i for i in range(expected_phases) if i not in memristive_indices)
    n_photons = tuple([1] * len(phase_idx))

    model = PhotonicModel(
        init_theta, enc_np, y_np, memory_depth, phase_idx, n_photons,
        n_modes=n_modes,
        encoding_mode=encoding_mode, target_mode=target_mode,
        loss_type=loss_type, n_classes=n_classes,
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    iterator = range(epochs)
    if not verbose:
        iterator = tqdm(iterator, desc="Training", ncols=100)
    for e in iterator:
        optim.zero_grad()
        loss = model(n_samples=n_samples, n_swipe=n_swipe, swipe_span=swipe_span)
        loss.backward()
        optim.step()
        with torch.no_grad():
            # Clamp weight params to [0.01, 1]; phases to [0, 2π)
            weight_idxs = set(range(len(model.theta))) - set(phase_idx)
            for idx in weight_idxs:
                model.theta.data[idx].clamp_(0.01, 1.0)
            for idx in phase_idx:
                model.theta.data[idx].remainder_(2 * np.pi)
        
        hist.append(loss.item())
        if verbose:
            print(f"  Epoch {e + 1}/{epochs}: loss = {loss.item():.6f}")
    theta_opt = model.theta.detach().cpu().numpy()
    if verbose:
        print(f"  Final parameters: {theta_opt}")
    return theta_opt, hist


def train_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    memory_depth: int,
    lr: float,
    epochs: int,
    n_samples: int,
    n_swipe: int,
    swipe_span: float,
    n_modes: int,
    encoding_mode: int,
    target_mode: Optional[Tuple[int, ...]] = None,
    loss_type: str = 'mse',
    n_classes: int = 1,
    memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None,
    memristive_output_modes: Optional[Sequence[Tuple[int, int]]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[float]]:
    """
    Unified training path for both discrete and continuous modes.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        n_swipe (int): Number of phase points per data point (0 for discrete).
        swipe_span (float): Total phase span for swiping.
        n_modes (int): Number of modes (3 for 3x3, 6 for 6x6, etc.).
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s).
        **kwargs: Additional arguments for training.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(
        enc, y,
        memory_depth=memory_depth,
        lr=lr,
        epochs=epochs,
        n_samples=n_samples,
        n_swipe=n_swipe,
        swipe_span=swipe_span,
        n_modes=n_modes,
        encoding_mode=encoding_mode,
        target_mode=target_mode,
        loss_type=loss_type,
        n_classes=n_classes,
        memristive_phase_idx=memristive_phase_idx,
        memristive_output_modes=memristive_output_modes,
        verbose=verbose,
    )


def gradient_check(n_modes: int = 3, memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None) -> None:
    """
    Performs a gradient check comparing finite difference and analytic gradients.
    """
    from .data import get_data
    from .autograd import MemristorLossPSR
    from .simulation import run_simulation_sequence_np

    X, y, *_ = get_data(60, 0.0)
    enc = 2 * np.arccos(X)
    n_phases = n_modes * (n_modes - 1)
    memristive_indices = _normalize_memristive_phase_idx(memristive_phase_idx, n_modes, n_phases)
    n_memristive = len(memristive_indices)
    phase_idx = tuple(i for i in range(n_phases) if i not in memristive_indices)
    n_photons = tuple([1] * len(phase_idx))

    theta0 = np.random.rand(n_phases + n_memristive)
    theta0[:n_phases] *= 2 * np.pi
    theta0[n_phases:] *= 0.5
    mem_depth = 2
    n_samples = 5

    def L(params):
        return 0.5 * ((run_simulation_sequence_np(
            params, mem_depth, n_samples,
            encoded_phases=enc,
            n_swipe=0,
            swipe_span=0.0,
            n_modes=n_modes,
            encoding_mode=0,
            target_mode=(n_modes - 1,) if n_modes else None,
            memristive_phase_idx=memristive_phase_idx
        ) - y) ** 2).mean()

    eps = 1e-5
    num_grad = np.zeros_like(theta0)
    for k in range(len(theta0)):
        p_plus, p_minus = theta0.copy(), theta0.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        num_grad[k] = (L(p_plus) - L(p_minus)) / (2 * eps)

    th_t = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    loss = MemristorLossPSR.apply(
        th_t, torch.from_numpy(enc).double(), torch.from_numpy(y).double(),
        mem_depth, phase_idx, n_photons, n_samples,
        0, 0.0,  # n_swipe, swipe_span
        n_modes, 0, (n_modes - 1,) if n_modes else None,  # n_modes, encoding_mode, target_mode
        'mse', 1,  # loss_type, n_classes
        memristive_phase_idx, None  # memristive_phase_idx, memristive_output_modes
    )
    loss.backward()
    psr_grad = th_t.grad.detach().cpu().numpy()
    print("Finite‑diff  :", num_grad)
    print("PSR / Torch :", psr_grad)
    print("Abs‑error   :", np.abs(num_grad - psr_grad))
    print("Max‑error   :", np.abs(num_grad - psr_grad).max())