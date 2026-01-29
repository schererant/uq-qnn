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
    'use_continuous': False,
    'n_swipe': None,
    'swipe_span': np.pi / 20,

    # Training
    'lr': 0.03,
    'epochs': 50,
    'phase_idx': (0, 1),
    'n_photons': (1, 1),
    
    # Model initialization
    'init_theta': None,  # Will be set in trainer

    # Plotting
    'do_plot': True,

    # Sampler
    'n_samples': 1000
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

#TODO: Check if this is correct
def get_measured_swipe_data(
    encoded_phases: np.ndarray,
    measured_phases: np.ndarray,
    n_swipe: int
) -> np.ndarray:
    """
    For each encoded phase, find the closest measured phase and select a window of n_swipe measured phases
    centered around it. Handles edge cases at the boundaries.
    Args:
        encoded_phases (np.ndarray): 1D array of encoded phases (e.g., 2*arccos(X)).
        measured_phases (np.ndarray): 1D array of measured phase values.
        n_swipe (int): Number of phase points per data point (should be odd).
    Returns:
        np.ndarray: 2D array of shape (len(encoded_phases), n_swipe) with measured phase values for each swipe.
    """
    if n_swipe < 1 or n_swipe % 2 == 0:
        raise ValueError(f"n_swipe must be a positive odd integer (got {n_swipe})")
    n_data = len(encoded_phases)
    n_meas = len(measured_phases)
    enc_samples = np.empty((n_data, n_swipe), dtype=measured_phases.dtype)
    n2 = n_swipe // 2
    for i in range(n_data):
        # Find the index of the measured phase closest to the encoded phase
        diffs = np.abs(measured_phases - encoded_phases[i])
        k = np.argmin(diffs)
        # Compute window bounds, handle boundaries
        start = max(0, k - n2)
        end = min(n_meas, k + n2 + 1)
        # If at the left edge, pad right
        if end - start < n_swipe:
            if start == 0:
                end = min(n_meas, start + n_swipe)
            else:
                start = max(0, end - n_swipe)
        # Fill the row
        window = measured_phases[start:end]
        # If window is too short (at edges), pad with edge values
        if len(window) < n_swipe:
            if start == 0:
                window = np.pad(window, (0, n_swipe - len(window)), mode='edge')
            else:
                window = np.pad(window, (n_swipe - len(window), 0), mode='edge')
        enc_samples[i, :] = window
    return enc_samples

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

# quartic tan(h), sin, cos, exp etc.


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
    memory_depth: int,
    n_samples: int,
    encoded_phases: np.ndarray = None,
    n_swipe: int = 0,
    swipe_span: float = 0.0,
) -> np.ndarray:
    """
    Runs a sequence of photonic-circuit simulations in either:
      1) Discrete-phase mode: returns p(001) for each given phase in encoded_phases (each value is a phase in radians).
      2) Continuous-swipe mode: for each X[i] in encoded_phases (each value in [0,1]), computes 2*arccos(X[i]) as the base phase, sweeps n_swipe phases around it, and returns the average p(001).

    Args:
        params (np.ndarray): [phi1, phi3, w].
        memory_depth (int): Depth of the memory buffer.
        n_samples (int): Number of samples for the Sampler.
        encoded_phases (np.ndarray):
            - Discrete mode: array of phase values (radians).
            - Continuous mode: array of X values in [0,1] (will be mapped to phase via 2*arccos(X)).
        n_swipe (int, optional): Number of phase points per X[i] (0 for discrete mode, >0 for continuous mode).
        swipe_span (float, optional): Total phase span for swiping (only used if n_swipe > 0).

    Returns:
        np.ndarray: Predicted p(001) per input point or per phase.
    """
    # validate mode selection
    if encoded_phases is None:
        raise ValueError("encoded_phases must be provided for both modes.")
    if n_swipe < 0:
        raise ValueError("n_swipe must be >= 0.")
    if n_swipe == 0:
        mode = "discrete"
    elif n_swipe > 0:
        if swipe_span <= 0:
            raise ValueError("swipe_span must be > 0 for continuous mode.")
        mode = "continuous"
    else:
        raise ValueError("Invalid mode selection: n_swipe must be >= 0.")
    # Optionally: print(f"Running in {mode} mode")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")

    start_time = time.perf_counter()
    phi1, phi3, w = params
    input_state = pcvl.BasicState([0, 1, 0])
    state_001 = pcvl.BasicState([0, 0, 1])
    state_010 = pcvl.BasicState([0, 1, 0])

    # prepare memory and output
    mem_p1 = np.zeros(memory_depth, dtype=float)
    mem_p2 = np.zeros(memory_depth, dtype=float)
    num_pts = len(encoded_phases)
    preds = np.zeros(num_pts, dtype=float)

    if mode == "continuous":
        # precompute base phases and offsets
        enc_base = encoded_phases
        #TODO: Use Iris data for that
        offsets = np.linspace(
            -swipe_span / 2, swipe_span / 2, n_swipe, dtype=enc_base.dtype
        )

    # main loop
    for i in range(num_pts):
        t = i % memory_depth
        # compute memory-driven φₘ
        if i == 0:
            mem_phi = np.pi / 4
        else:
            m1 = mem_p1.mean()
            m2 = mem_p2.mean()
            arg = np.clip(m1 + w * m2, 1e-9, 1 - 1e-9)
            mem_phi = np.arccos(np.sqrt(arg))

        if mode == "discrete":
            # single-φ mode
            enc_phi = encoded_phases[i]
            circ = build_circuit(phi1, mem_phi, phi3, enc_phi)
            proc = pcvl.Processor("SLOS", circ)
            proc.with_input(input_state)
            t0 = time.perf_counter()
            probs = Sampler(proc).probs(n_samples)["results"]
            sim_logger.log_circuit(time.perf_counter() - t0)

            p001 = probs.get(state_001, 0.0)
            p010 = probs.get(state_010, 0.0)
            preds[i] = p001
            mem_p1[t], mem_p2[t] = p010, p001

        else:
            # swipe mode: average over offsets
            p1_swipe = np.empty(n_swipe, dtype=float)
            p2_swipe = np.empty(n_swipe, dtype=float)
            for k, off in enumerate(offsets):
                enc_phi = enc_base[i] + off
                circ = build_circuit(phi1, mem_phi, phi3, enc_phi)
                proc = pcvl.Processor("SLOS", circ)
                proc.with_input(input_state)
                t0 = time.perf_counter()
                probs = Sampler(proc).probs(n_samples)["results"]
                sim_logger.log_circuit(time.perf_counter() - t0)

                p1_swipe[k] = probs.get(state_010, 0.0)
                p2_swipe[k] = probs.get(state_001, 0.0)

            preds[i] = p2_swipe.mean()
            mem_p1[t], mem_p2[t] = p1_swipe.mean(), p2_swipe.mean()

    # finalize
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
    Autograd Function using PSR for photonic‐phase parameters and
    finite‐difference only for memristor weights, in both discrete‐phase
    and continuous‐swipe modes.
    """
    @staticmethod
    def forward(
        ctx,
        theta: Tensor,
        enc_phases: Tensor,
        y: Tensor,
        memory_depth: int,
        phase_idx: Sequence[int],
        n_photons: Sequence[int],
        n_samples: int,
        n_swipe: int = 0,
        swipe_span: float = 0.0,
    ) -> Tensor:
        discrete = (n_swipe == 0)
        theta_np = theta.detach().cpu().double().numpy()
        enc_np   = enc_phases.detach().cpu().double().numpy()
        y_np     = y.detach().cpu().double().numpy()

        if discrete:
            preds = run_simulation_sequence_np(
                theta_np, memory_depth, n_samples,
                encoded_phases=enc_np
            )
        else:
            preds = run_simulation_sequence_np(
                theta_np, memory_depth, n_samples,
                encoded_phases=enc_np, n_swipe=n_swipe, swipe_span=swipe_span
            )

        loss_val = 0.5 * np.mean((preds - y_np) ** 2)

        ctx.save_for_backward(theta.detach(),
                              enc_phases.detach(),
                              y.detach())
        ctx.discrete     = discrete
        ctx.phase_idx    = list(phase_idx)
        ctx.n_photons    = list(n_photons)
        ctx.n_swipe      = n_swipe
        ctx.swipe_span   = swipe_span
        ctx.memory_depth = memory_depth
        ctx.n_samples    = n_samples
        ctx.preds_np     = preds

        return torch.tensor(loss_val, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out: Tensor):
        theta, enc_tensor, y = ctx.saved_tensors
        theta_np = theta.cpu().double().numpy()
        enc_np   = enc_tensor.cpu().double().numpy()
        y_np     = y.cpu().double().numpy()
        preds    = ctx.preds_np
        N        = y.numel()
        dL_df    = (preds - y_np) / N

        grads = np.zeros_like(theta_np)
        eps   = 1e-3

        # PSR gradients for photonic phases
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, coeffs = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])
            df_dθ = np.zeros_like(preds)
            for s, c in zip(shifts.numpy(), coeffs.numpy()):
                θ_shift = theta_np.copy()
                θ_shift[p_idx] += s
                if ctx.discrete:
                    out = run_simulation_sequence_np(
                        θ_shift, ctx.memory_depth, ctx.n_samples,
                        encoded_phases=enc_np
                    )
                else:
                    out = run_simulation_sequence_np(
                        θ_shift, ctx.memory_depth, ctx.n_samples,
                        encoded_phases=enc_np, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span
                    )
                df_dθ += c * out
            grads[p_idx] = np.real(np.dot(dL_df, df_dθ))

        # Finite-difference for memristor weight parameters
        weight_idxs = set(range(len(theta_np))) - set(ctx.phase_idx)
        for idx in weight_idxs:
            θ_p = theta_np.copy(); θ_m = theta_np.copy()
            θ_p[idx] += eps; θ_m[idx] -= eps
            θ_p[idx] = np.clip(θ_p[idx], 0.01, 1)
            θ_m[idx] = np.clip(θ_m[idx], 0.01, 1)

            if ctx.discrete:
                pred_p = run_simulation_sequence_np(
                    θ_p, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np
                )
                pred_m = run_simulation_sequence_np(
                    θ_m, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np
                )
            else:
                pred_p = run_simulation_sequence_np(
                    θ_p, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span
                )
                pred_m = run_simulation_sequence_np(
                    θ_m, ctx.memory_depth, ctx.n_samples,
                    encoded_phases=enc_np, n_swipe=ctx.n_swipe, swipe_span=ctx.swipe_span
                )

            loss_p = 0.5 * np.mean((pred_p - y_np) ** 2)
            loss_m = 0.5 * np.mean((pred_m - y_np) ** 2)
            grads[idx] = (loss_p - loss_m) / (2 * eps)

        return g_out * torch.from_numpy(grads).to(theta), None, None, None, None, None, None, None, None


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
                                      self.memory_depth, self.phase_idx, self.n_photons, n_samples, n_swipe, swipe_span)

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
    if cont:
        n_swipe = _resolve_n_swipe()
        config['n_swipe'] = n_swipe  # freeze for the rest of the run
        theta_opt, history = train_pytorch(
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
            n_swipe=0,
            swipe_span=0.0,
            n_samples=n_samples
        )

    print("Optimized θ:", theta_opt)

    # ── predictions on dense grid ──
    if cont:
        preds = run_simulation_sequence_np(
            theta_opt, config['memory_depth'], n_samples,
            encoded_phases=2 * np.arccos(X_test), n_swipe=config['n_swipe'], swipe_span=config['swipe_span']
        )
    else:
        enc_test = 2 * np.arccos(X_test)
        preds = run_simulation_sequence_np(theta_opt, config['memory_depth'], n_samples, encoded_phases=enc_test)

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
