from __future__ import annotations

import argparse
import pickle
from functools import lru_cache
from typing import Callable, Sequence, List, Tuple
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import perceval as pcvl
import torch
from perceval.algorithm import Sampler
from torch import Tensor
from tqdm import tqdm

# ===================== CONFIGURATION =====================
config = {
    # Data generation
    'n_data': 100,
    'sigma_noise': 0.1,
    'datafunction': 'quartic_data',
    'memory_depth': 2,

    # Timing (defaults: 10 ms heater, 50 kHz laser, 10 µs detector)
    't_phase_ms': 10.0,
    'f_laser_khz': 50.0,
    'det_window_us': 10.0,
    'max_swipe': 21,

    # Continuous swipe
    'use_continuous': True,
    'n_swipe': None,
    'swipe_span': np.pi / 20,

    # Training
    'lr': 0.03,
    'epochs': 1,
    'phase_idx': (0, 1),
    'n_photons': (1, 1),
    
    # Model initialization
    'init_theta': None,  # Will be set in trainer

    # Plotting
    'do_plot': False,

    # Sampler
    'n_samples': 100
}
# =========================================================

###############################################################################
# 0.  Helpers for measured data & phase swipes                                 #
###############################################################################

def load_measurement_pickle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads measured data from a pickle file and returns it as two numpy arrays.
    Args:
        path (str): Path to the pickle file containing (X, y) data.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of X and y arrays.
    """
    with open(path, "rb") as fh:
        X, y = pickle.load(fh)
    return np.asarray(X), np.asarray(y)


def compute_n_swipe(
    t_phase_ms: float,
    f_laser_khz: float,
    det_window_us: float,
    max_swipe: int = 201,
) -> int:
    """
    Translates hardware-timing limits into a safe, odd swipe count. 
    It divides the heater's settle time by the slower of the laser 
    period and detector window to see how many optical "slots" fit, 
    then forces the result to be odd (so the original point stays centered) 
    and caps it at `max_swipe` to keep memory footprint reasonable.
    Args:
        t_phase_ms (float): Heater settle time in milliseconds.
        f_laser_khz (float): Laser repetition rate in kHz.
        det_window_us (float): Detector integration window in microseconds.
        max_swipe (int): Maximum allowed swipe count.
    Returns:
        int: Odd integer swipe count, capped at max_swipe.
    """
    if t_phase_ms <= 0 or f_laser_khz <= 0 or det_window_us <= 0:
        raise ValueError("Timing inputs must be positive.")

    period_laser_us = 1_000 / f_laser_khz  # µs
    slot_us = max(period_laser_us, det_window_us)
    slots_total = int((t_phase_ms * 1_000) // slot_us)  # integer slots
    n_swipe = max(1, 2 * (slots_total // 2) + 1)        # force odd
    return min(n_swipe, max_swipe)


def get_cont_swipe_data(
    X: np.ndarray,
    y: np.ndarray,
    n_swipe: int,
    swipe_span: float = np.pi / 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands each (X[i], y[i]) into n_swipe neighboring phase points for data augmentation.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        n_swipe (int): Number of phase points to generate per data point (must be odd).
        swipe_span (float): Total phase span for swiping.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Expanded phase-encoded X and repeated y arrays.
    """
    if n_swipe < 1 or n_swipe % 2 == 0:
        raise ValueError(f"n_swipe must be a positive odd integer (got {n_swipe})")
    enc_base = 2 * np.arccos(X)
    offsets = np.linspace(-swipe_span / 2, swipe_span / 2, n_swipe, dtype=enc_base.dtype)
    enc_swipe = np.concatenate([enc + offsets for enc in enc_base])
    y_swipe = np.repeat(y, n_swipe)
    return enc_swipe, y_swipe

###############################################################################
# 1.  Data generator                                                          #
###############################################################################

def quartic_data(x: np.ndarray) -> np.ndarray:
    """
    Computes the quartic (x^4) of the input array.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with each element raised to the 4th power.
    """
    return np.power(x, 4)


def get_data(
    n_data: int = 100,
    sigma_noise: float = 0.0,
    datafunction: str = 'quartic_data',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic training and test data, with a gap in the training set.
    Args:
        n_data (int): Number of training data points.
        sigma_noise (float): Standard deviation of Gaussian noise added to y.
        datafunction (str): Name of the function to generate y from X.
    Returns:
        Tuple: (X_train, y_train, X_test, y_test) arrays.
    """
    x_min, x_max = 0.0, 1.0
    X = np.linspace(x_min, x_max, n_data)
    # Explicit mapping of string to function
    datafunction_map = {
        'quartic_data': quartic_data,
        # Add more mappings here as needed
    }
    if datafunction not in datafunction_map:
        raise ValueError(f"Unknown datafunction: {datafunction}")
    datafunc = datafunction_map[datafunction]
    y = datafunc(X) + np.random.normal(0, sigma_noise, size=n_data)
    gap = (x_min + 0.35 * (x_max - x_min), x_min + 0.60 * (x_max - x_min))
    mask = ~((X > gap[0]) & (X < gap[1]))
    X_train, y_train = X[mask], y[mask]
    X_test = np.linspace(x_min, x_max, 500)
    y_test = datafunc(X_test)
    return X_train, y_train, X_test, y_test

###############################################################################
# 2.  Circuit‑Aufbau (unchanged)                                               #
###############################################################################

def encoding_circuit(encoded_phase: float) -> pcvl.Circuit:
    """
    Builds a 2-mode encoding circuit with a phase shifter.
    Args:
        encoded_phase (float): Phase to encode.
    Returns:
        pcvl.Circuit: The constructed encoding circuit.
    """
    c = pcvl.Circuit(2, name="Encoding")
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=encoded_phase))
    c.add((0, 1), pcvl.BS())
    return c


def memristor_circuit(phi1: float, mem_phi: float, phi3: float) -> pcvl.Circuit:
    """
    Builds a 3-mode memristor circuit with phase shifters and beamsplitters.
    Args:
        phi1 (float): Phase for the first PS.
        mem_phi (float): Phase for the memristor PS.
        phi3 (float): Phase for the third PS.
    Returns:
        pcvl.Circuit: The constructed memristor circuit.
    """
    c = pcvl.Circuit(3, name="Memristor")
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi1)).add((0, 1), pcvl.BS())
    c.add((1, 2), pcvl.BS()).add((2,), pcvl.PS(phi=mem_phi)).add((1, 2), pcvl.BS())
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi3)).add((0, 1), pcvl.BS())
    return c


def build_circuit(
    phi1: float, mem_phi: float, phi3: float, enc_phi: float
) -> pcvl.Circuit:
    """
    Builds the full 3-mode circuit by combining encoding and memristor circuits.
    Args:
        phi1 (float): Phase for the first PS in memristor.
        mem_phi (float): Phase for the memristor PS.
        phi3 (float): Phase for the third PS in memristor.
        enc_phi (float): Encoding phase.
    Returns:
        pcvl.Circuit: The complete circuit.
    """
    c = pcvl.Circuit(3, name="Full")
    c.add(0, encoding_circuit(enc_phi))
    c.add(0, memristor_circuit(phi1, mem_phi, phi3))
    return c

###############################################################################
# 3.  NumPy‑Simulation                                                         #
###############################################################################

# Logger for simulation tracking
class SimulationLogger:
    def __init__(self):
        self.call_count = 0
        self.total_time = 0.0
        self.samples_counter = Counter()
        self.circuit_call_count = 0
        self.circuit_total_time = 0.0

    def log(self, elapsed: float, n_samples: int):
        self.call_count += 1
        self.total_time += elapsed
        self.samples_counter[n_samples] += 1

    def log_circuit(self, elapsed: float):
        self.circuit_call_count += 1
        self.circuit_total_time += elapsed

    def report(self):
        print(f"[SimulationLogger] Circuit sequence runs: {self.call_count}")
        print(f"[SimulationLogger] Total sequence time: {self.total_time:.3f} seconds")
        if self.call_count > 0:
            print(f"[SimulationLogger] Avg time per sequence: {self.total_time / self.call_count:.6f} seconds")
        print(f"[SimulationLogger] Sampler sample counts used:")
        for n_samples, freq in self.samples_counter.items():
            print(f"  {n_samples} samples: {freq} times")
        print(f"[SimulationLogger] Individual circuit simulations: {self.circuit_call_count}")
        print(f"[SimulationLogger] Total circuit sim time: {self.circuit_total_time:.3f} seconds")
        if self.circuit_call_count > 0:
            print(f"[SimulationLogger] Avg time per circuit sim: {self.circuit_total_time / self.circuit_call_count:.6f} seconds")

sim_logger = SimulationLogger()


def run_simulation_sequence_np(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    memory_depth: int,
    n_samples: int,
) -> np.ndarray:
    """
    Runs a sequence of photonic circuit simulations and returns p(001) for each encoded phase.
    Args:
        params (np.ndarray): Array of [phi1, phi3, w] parameters.
        encoded_phases (np.ndarray): Array of encoded phases.
        memory_depth (int): Depth of the memory buffer.
        n_samples (int): Number of samples for the Sampler.
    Returns:
        np.ndarray: Array of predicted p(001) values for each phase.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")
    start_time = time.perf_counter()
    phi1, phi3, w = params
    input_state = pcvl.BasicState([0, 1, 0])
    state_001, state_010 = pcvl.BasicState([0, 0, 1]), pcvl.BasicState([0, 1, 0])
    mem_p1 = np.zeros(memory_depth)
    mem_p2 = np.zeros(memory_depth)
    preds = np.zeros_like(encoded_phases)
    for i, enc_phi in enumerate(encoded_phases):
        t = i % memory_depth
        mem_phi = (
            np.pi / 4 if i == 0 else np.arccos(
                np.sqrt(np.clip(mem_p1.mean() + w * mem_p2.mean(), 1e-9, 1 - 1e-9))
            )
        )
        circ = build_circuit(phi1, mem_phi, phi3, enc_phi)
        proc = pcvl.Processor("SLOS", circ)
        proc.with_input(input_state)
        circuit_start = time.perf_counter()
        probs = Sampler(proc).probs(n_samples)["results"]
        circuit_elapsed = time.perf_counter() - circuit_start
        sim_logger.log_circuit(circuit_elapsed)
        p001 = probs.get(state_001, 0.0)
        p010 = probs.get(state_010, 0.0)
        preds[i] = p001
        mem_p1[t], mem_p2[t] = p010, p001
    elapsed = time.perf_counter() - start_time
    sim_logger.log(elapsed, n_samples)
    return preds

###############################################################################
# 4.  Corrected PSR‑Coefficients (Torch)                                       #
###############################################################################

@lru_cache(maxsize=None)
def photonic_psr_coeffs_torch(n: int) -> Tuple[Tensor, Tensor]:
    """
    Computes phase-shift rule (PSR) coefficients for photonic gradients.
    Args:
        n (int): Number of photons.
    Returns:
        Tuple[Tensor, Tensor]: Tuple of (shifts, c_p) tensors for PSR.
    """
    P = 2 * n
    shifts = 2 * np.pi * np.arange(1, P + 1) / (2 * n + 1)
    grad_vec = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
    cp_full = np.fft.ifft(np.concatenate(([0], grad_vec)))
    shifts_t = torch.from_numpy(shifts).double()
    cp_t = torch.from_numpy(np.real_if_close(cp_full[1:])).double()
    return shifts_t, cp_t

###############################################################################
# 5.  Autograd Function   (PSR  +  Chain rule)                                 #
###############################################################################

class MemristorLossPSR(torch.autograd.Function):
    """
    Custom autograd function for MSE loss with analytic PSR and finite difference gradients.
    Forward computes the MSE loss. Backward computes gradients using PSR for phases and FD for weights.
    Inputs:
        theta (Tensor): Model parameters.
        enc_phases (Tensor): Encoded phase values.
        y (Tensor): Target values.
        memory_depth (int): Memory buffer depth.
        phase_idx (Sequence[int]): Indices of phase parameters.
        n_photons (Sequence[int]): Number of photons for each phase.
        n_samples (int): Number of samples for the Sampler.
    Returns:
        Tensor: Scalar loss value.
    """
    @staticmethod
    def forward(ctx, theta: Tensor, enc_phases: Tensor, y: Tensor,
                memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int], n_samples: int) -> Tensor:
        theta_np, enc_np, y_np = map(lambda t: t.detach().cpu().double().numpy(),
                                     (theta, enc_phases, y))
        preds = run_simulation_sequence_np(theta_np, enc_np, memory_depth, n_samples=n_samples)
        loss = 0.5 * ((preds - y_np) ** 2).mean()
        ctx.save_for_backward(theta.detach(), enc_phases.detach(), y.detach())
        ctx.memory_depth, ctx.phase_idx, ctx.n_photons, ctx.preds_np, ctx.n_samples = memory_depth, list(phase_idx), list(n_photons), preds, n_samples
        return torch.tensor(loss, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out: Tensor):
        theta, enc_phases, y = ctx.saved_tensors
        N = y.numel()
        theta_np = theta.cpu().double().numpy()
        enc_np = enc_phases.cpu().double().numpy()
        y_np = y.cpu().double().numpy()
        preds = ctx.preds_np
        dL_df = (preds - y_np) / N
        grads = np.zeros_like(theta_np)
        # --- PSR for photonic phases ---
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, cp = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])
            df_dtheta = np.zeros_like(preds)
            for s, c in zip(shifts.numpy(), cp.numpy()):
                th_shift = theta_np.copy(); th_shift[p_idx] += s
                df_dtheta += c * run_simulation_sequence_np(th_shift, enc_np, ctx.memory_depth, n_samples=ctx.n_samples)
            grads[p_idx] = np.real(np.dot(dL_df, df_dtheta))
        # --- Finite Difference for the memristor weight ---
        eps = 1e-3
        for idx in set(range(len(theta_np))) - set(ctx.phase_idx):
            th_plus, th_minus = theta_np.copy(), theta_np.copy()
            th_plus[idx] += eps; th_minus[idx] -= eps
            th_plus[idx] = np.clip(th_plus[idx], 0.01, 1)
            th_minus[idx] = np.clip(th_minus[idx], 0.01, 1)
            loss_p = 0.5 * ((run_simulation_sequence_np(th_plus, enc_np, ctx.memory_depth, n_samples=ctx.n_samples) - y_np) ** 2).mean()
            loss_m = 0.5 * ((run_simulation_sequence_np(th_minus, enc_np, ctx.memory_depth, n_samples=ctx.n_samples) - y_np) ** 2).mean()
            grads[idx] = (loss_p - loss_m) / (2 * eps)
        return g_out * torch.from_numpy(grads).to(theta), None, None, None, None, None, None

###############################################################################
# 6.  Model‑Klasse                                                             #
###############################################################################

class PhotonicModel(torch.nn.Module):
    """
    PyTorch model class for photonic memristor circuit training.
    Args:
        init_theta (Sequence[float]): Initial parameter values.
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        memory_depth (int): Memory buffer depth.
        phase_idx (Sequence[int]): Indices of phase parameters.
        n_photons (Sequence[int]): Number of photons for each phase.
    """
    def __init__(self, init_theta: Sequence[float], enc_np: np.ndarray, y_np: np.ndarray,
                 memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int]) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.register_buffer("y", torch.from_numpy(y_np).double())
        self.memory_depth, self.phase_idx, self.n_photons = memory_depth, phase_idx, n_photons

    def forward(self, n_samples: int) -> Tensor:
        """
        Computes the loss using the custom autograd function.
        Args:
            n_samples (int): Number of samples for the Sampler.
        Returns:
            Tensor: Scalar loss value.
        """
        return MemristorLossPSR.apply(self.theta, self.enc, self.y,
                                      self.memory_depth, self.phase_idx, self.n_photons, n_samples)

###############################################################################
# 7.  Training (discrete & continuous)                                         #
###############################################################################

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
    n_samples: int
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
        loss = model(n_samples=n_samples)
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
    **kwargs
) -> Tuple[np.ndarray, List[float]]:
    """
    Wrapper for discrete training path using phase-encoded X.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        **kwargs: Additional arguments for training.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(enc, y, **kwargs)


def train_pytorch_cont(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_swipe: int,
    swipe_span: float,
    n_samples: int,
    **kwargs
) -> Tuple[np.ndarray, List[float]]:
    """
    Continuous-swipe training path. Expands each (X, y) into n_swipe neighboring phase points.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        n_swipe (int): Number of phase points per data point.
        swipe_span (float): Total phase span for swiping.
        n_samples (int): Number of samples for the Sampler.
        **kwargs: Additional arguments for training.
    Returns:
        Tuple[np.ndarray, List[float]]: Optimized parameters and loss history.
    """
    enc_swipe, y_swipe = get_cont_swipe_data(X, y, n_swipe=n_swipe, swipe_span=swipe_span)
    return train_pytorch_generic(enc_swipe, y_swipe, n_samples=n_samples, **kwargs)

###############################################################################
# 8.  Gradient‑Check (unchanged)                                               #
###############################################################################

def gradient_check() -> None:
    """
    Performs a gradient check comparing finite difference and analytic gradients.
    Prints the results and their absolute/max errors.
    Returns:
        None
    """
    X, y, *_ = get_data(60, 0.0)
    enc = 2 * np.arccos(X)
    theta0 = np.array([1.2, 2.3, 0.5])
    mem_depth = 2
    n_samples = 5
    def L(params):
        return 0.5 * ((run_simulation_sequence_np(params, enc, mem_depth, n_samples=n_samples) - y) ** 2).mean()
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

###############################################################################
# 9.  Main                                                                     #
###############################################################################


def _resolve_n_swipe() -> int:
    """
    Resolves the number of swipes to use, either from config or by computing it.
    Returns:
        int: Number of swipes.
    """
    if config['n_swipe'] is not None:
        return config['n_swipe']
    auto = compute_n_swipe(
        config['t_phase_ms'],
        config['f_laser_khz'],
        config['det_window_us'],
        config['max_swipe'],
    )
    print(f"[timing] computed n_swipe = {auto}")
    return auto


def _run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cont: bool,
    n_samples: int
) -> None:
    """
    Runs the training process and plots results for both discrete and continuous modes.
    Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training output data.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test output data.
        cont (bool): Whether to use continuous-swipe training.
        n_samples (int): Number of samples for the Sampler.
    Returns:
        None
    """
    n_swipe = _resolve_n_swipe()
    config['n_swipe'] = n_swipe  # freeze for the rest of the run

    # ── choose path ──
    if cont:
        theta_opt, history = train_pytorch_cont(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            phase_idx=config['phase_idx'],
            n_photons=config['n_photons'],
            n_swipe=config['n_swipe'],
            swipe_span=config['swipe_span'],
            n_samples=n_samples
        )
    else:
        theta_opt, history = train_pytorch(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            phase_idx=config['phase_idx'],
            n_photons=config['n_photons'],
            n_samples=n_samples
        )

    print("Optimized θ:", theta_opt)

    # ── predictions on dense grid ──
    enc_test = 2 * np.arccos(X_test)
    preds = run_simulation_sequence_np(theta_opt, enc_test, memory_depth=config['memory_depth'], n_samples=n_samples)

    if config['do_plot']:
        # ————————————————————————————————————————————————————————————————
        # Plot 1: loss curve
        plt.figure(figsize=(9, 4))
        plt.plot(history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()

        # Plot 2: data cloud + model fit
        plt.figure(figsize=(9, 5))
        plt.scatter(X_train, y_train, s=20, label='Original data', zorder=3)

        if cont:
            enc_swipe, _ = get_cont_swipe_data(X_train, y_train, n_swipe=config['n_swipe'], swipe_span=config['swipe_span'])
            X_swipe = np.cos(enc_swipe / 2)
            plt.scatter(X_swipe, np.repeat(y_train, config['n_swipe']), s=8, alpha=0.35, label=f'Swipe (n={config['n_swipe']})', zorder=2)

        plt.plot(X_test, y_test, label='Quartic', ls='--', zorder=1)
        plt.plot(X_test, preds, label='Model', c='red', zorder=4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        # Plot 3: index vs encoded phase
        plt.figure(figsize=(9, 4))
        idx = np.arange(len(X_train))
        enc_orig = 2 * np.arccos(X_train)
        plt.plot(idx, enc_orig, '-o', label='Original enc φ', lw=1.5)

        if cont:
            # enc_swipe already computed above
            idx_swipe = np.repeat(idx, config['n_swipe'])
            plt.scatter(idx_swipe, enc_swipe, s=6, alpha=0.35, label='Swipe enc φ')

        plt.xlabel('Data index')
        plt.ylabel('Encoding phase [rad]')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


def main(n_samples: int, measured_data: str | None = None, use_continuous: bool = False) -> None:
    """
    Main entry point for running the training and evaluation pipeline.
    Args:
        n_samples (int): Number of samples for the Sampler.
        measured_data (str | None): Path to measured data pickle file, or None for synthetic data.
        use_continuous (bool): Whether to use continuous-swipe training.
    Returns:
        None
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")
    if measured_data is not None:
        X_train, y_train = load_measurement_pickle(measured_data)
        # Synthetic test set: densify full phase range
        X_test = np.linspace(0.0, 1.0, 500)
        y_test = quartic_data(X_test)  # Placeholder when ground truth unknown
    else:
        X_train, y_train, X_test, y_test = get_data(
            config['n_data'],
            config['sigma_noise'],
            config['datafunction']
        )
    _run_training(X_train, y_train, X_test, y_test, cont=use_continuous, n_samples=n_samples)
    sim_logger.report()


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--check", action="store_true", help="Run gradient check and exit")
    argp.add_argument("--cont", action="store_true", help="Use continuous‑swipe training")
    argp.add_argument("--data", type=str, help="Pickle file with measured (X, y) traces")
    argp.add_argument("--n_samples", type=int, default=config['n_samples'], help="Number of samples for the Sampler")
    args, _ = argp.parse_known_args()
    if args.check:
        gradient_check()
        sim_logger.report()
    else:
        main(measured_data=args.data, use_continuous=args.cont or config['use_continuous'], n_samples=args.n_samples)
