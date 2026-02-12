from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
import torch
from tqdm import tqdm

from .loss import PhotonicModel


def _init_theta(rng: np.random.Generator, 
                n_phases: int, 
                circuit_type: str, 
                n_modes: int) -> np.ndarray:
    """
    Initializes model parameters randomly within specified ranges.
    Args:
        rng (np.random.Generator): Random number generator.
        n_phases (int): Number of phase parameters to initialize (excluding memory phase).
        circuit_type (str): Type of circuit architecture ('memristor' or 'clements').
        n_modes (int): Number of modes for Clements architecture.
    Returns:
        np.ndarray: Array of initial parameter values.
    """
    if circuit_type.lower() == 'memristor':
        # Fixed structure for memristor: [phi1, phi2, w]
        phases = rng.uniform(0.01, 1, size=2) * 2 * np.pi
        w = rng.uniform(0.01, 1)
        return np.concatenate([phases, [w]])
    else:  # Clements
        # For Clements: [phi1_int, phi1_ext, phi2_int, phi2_ext, ..., w]
        # Calculate expected number of phases based on n_modes
        expected_phases = n_modes * (n_modes - 1)
        
        # If n_phases doesn't match the expected value, use the correct value
        if n_phases != expected_phases:
            print(f"Warning: Adjusting n_phases from {n_phases} to {expected_phases} for {n_modes}-mode Clements circuit")
            n_phases = expected_phases
            
        if n_phases == 0:
            raise ValueError(f"Clements architecture requires at least 2 modes, got {n_modes}")
            
        # Initialize phases with Haar-random values for unitarity
        phases = rng.uniform(0.0, 2 * np.pi, size=n_phases)
        w = rng.uniform(0.01, 1)
        return np.concatenate([phases, [w]])


def train_pytorch_generic(
    enc_np: np.ndarray,
    y_np: np.ndarray,
    *,
    memory_depth: int,
    lr: float,
    epochs: int,
    phase_idx: Sequence[int],
    n_photons: Sequence[int],
    n_phases: int,
    n_samples: int,
    n_swipe: int,
    swipe_span: float,
    circuit_type: str,
    n_modes: int,
    encoding_mode: int,
    target_mode: Optional[Tuple[int, ...]],
    seed: int = 42,
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
    init_theta = _init_theta(rng=rng, 
                        n_phases=n_phases, 
                        circuit_type=circuit_type, 
                        n_modes=n_modes
                        )
    model = PhotonicModel(init_theta=init_theta, 
                        enc_np=enc_np, 
                        y_np=y_np,
                        memory_depth=memory_depth, 
                        phase_idx=phase_idx, 
                        n_photons=n_photons,
                        circuit_type=circuit_type,
                        n_modes=n_modes,
                        encoding_mode=encoding_mode, 
                        target_mode=target_mode
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in tqdm(range(epochs), desc="Training", ncols=100):
        optim.zero_grad()
        loss = model(n_samples=n_samples, n_swipe=n_swipe, swipe_span=swipe_span)
        loss.backward()
        optim.step()
        with torch.no_grad():
            model.theta.data[-1].clamp_(0.01, 1.0)       # w ∈ [0.01,1]
            model.theta.data[:-1].remainder_(2 * np.pi)  # Phases ∈ [0,2π)
        
            # Apply additional constraints for specific circuit types if needed
            if circuit_type.lower() == 'clements':
                # For Clements architecture, ensure phases stay within valid ranges
                # Ensure phases are between 0 and 2π
                model.theta.data[:-1].remainder_(2 * np.pi)
            
                # Normalize weight parameter 
                model.theta.data[-1].clamp_(0.01, 1.0)
        hist.append(loss.item())
    return model.theta.detach().cpu().numpy(), hist


def train_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_swipe: int,
    swipe_span: float,
    n_phases: int,
    circuit_type: str,
    n_modes: int,
    encoding_mode: int,
    target_mode: Optional[Tuple[int, ...]],
    **kwargs
) -> Tuple[np.ndarray, List[float]]:
    """
    Unified training path for both discrete and continuous modes.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        n_swipe (int): Number of phase points per data point (0 for discrete).
        swipe_span (float): Total phase span for swiping.
        n_phases (int): Number of phase parameters (excluding memory phase).
        circuit_type (str): Type of circuit architecture ('memristor' or 'clements').
        n_modes (int): Number of modes for Clements architecture.
        encoding_mode (int): Mode to apply encoding to.
        target_mode (Optional[Tuple[int, ...]]): Target output mode(s).
        **kwargs: Additional arguments for training.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(
        enc, y, 
        n_swipe=n_swipe, 
        swipe_span=swipe_span, 
        n_phases=n_phases,
        circuit_type=circuit_type,
        n_modes=n_modes,
        encoding_mode=encoding_mode,
        target_mode=target_mode,
        **kwargs
    )


def gradient_check(circuit_type: str = 'memristor', n_modes: int = 3) -> None:
    """
    Performs a gradient check comparing finite difference and analytic gradients.
    Prints the results and their absolute/max errors.
    Args:
        circuit_type (str): Type of circuit architecture ('memristor' or 'clements').
        n_modes (int): Number of modes for Clements architecture.
    Returns:
        None
    """
    from .data import get_data
    from .autograd import MemristorLossPSR
    from .simulation import run_simulation_sequence_np, CircuitType
    from .circuits import CircuitType as CircuitTypeEnum
    
    X, y, *_ = get_data(60, 0.0)
    enc = 2 * np.arccos(X)
    
    # Initialize parameters based on circuit type
    if circuit_type.lower() == 'memristor':
        theta0 = np.array([1.2, 2.3, 0.5])
        phase_idx = (0, 1)
        n_photons = (1, 1)
        circuit_enum = CircuitTypeEnum.MEMRISTOR
    else:  # Clements
        # For a simple test with Clements, we'll use 3 modes (3 MZIs, 6 phases)
        n_phases = n_modes * (n_modes - 1)
        theta0 = np.random.rand(n_phases + 1)  # +1 for weight
        theta0[:-1] *= 2 * np.pi  # phases in [0, 2π)
        theta0[-1] *= 0.5  # weight in [0, 0.5]
        phase_idx = tuple(range(n_phases))
        n_photons = tuple([1] * n_phases)
        circuit_enum = CircuitTypeEnum.CLEMENTS
        
    mem_depth = 2
    n_samples = 5
    
    def L(params):
        return 0.5 * ((run_simulation_sequence_np(
            params, mem_depth, n_samples, 
            encoded_phases=enc,
            circuit_type=circuit_enum,
            n_modes=n_modes
        ) - y) ** 2).mean()
    
    # Finite Difference
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
        circuit_type=circuit_type, n_modes=n_modes
    )
    loss.backward()
    psr_grad = th_t.grad.detach().cpu().numpy()
    print("Finite‑diff  :", num_grad)
    print("PSR / Torch :", psr_grad)
    print("Abs‑error   :", np.abs(num_grad - psr_grad))
    print("Max‑error   :", np.abs(num_grad - psr_grad).max())