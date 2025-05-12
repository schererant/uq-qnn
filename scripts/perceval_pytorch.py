
from __future__ import annotations

import argparse
import pickle
from functools import lru_cache
from typing import Callable, Sequence, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import perceval as pcvl
import torch
from perceval.algorithm import Sampler
from torch import Tensor
from tqdm import tqdm

###############################################################################
# 0.  Helpers for measured data & phase swipes                                 #
###############################################################################

def load_measurement_pickle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays saved in a pickle file as a 2‑tuple."""
    with open(path, "rb") as fh:
        X, y = pickle.load(fh)
    return np.asarray(X), np.asarray(y)


def get_cont_swipe_data(
    X: np.ndarray,
    y: np.ndarray,
    n_swipe: int = 11,
    swipe_span: float = np.pi / 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fan out every (X, y) into *n_swipe* neighbouring phase points.

    Each original *encoded* phase ϕ is replaced by a small linear segment
    [ϕ − Δ/2, …, ϕ + Δ/2] with *n_swipe* points.  Targets *y* are simply
    repeated.  Works well when the underlying function varies slowly in
    the local neighbourhood (as is typical for phase responses).
    """
    enc_base = 2 * np.arccos(X)
    offsets = np.linspace(-swipe_span / 2, swipe_span / 2, n_swipe, dtype=enc_base.dtype)
    enc_swipe = np.concatenate([enc + offsets for enc in enc_base])
    y_swipe = np.repeat(y, n_swipe)
    return enc_swipe, y_swipe

###############################################################################
# 1.  Data generator                                                          #
###############################################################################

def quartic_data(x: np.ndarray) -> np.ndarray:
    return np.power(x, 4)


def get_data(
    n_data: int = 100,
    sigma_noise: float = 0.0,
    datafunction: Callable[[np.ndarray], np.ndarray] = quartic_data,
):
    x_min, x_max = 0.0, 1.0
    X = np.linspace(x_min, x_max, n_data)

    y = datafunction(X) + np.random.normal(0, sigma_noise, size=n_data)

    gap = (x_min + 0.35 * (x_max - x_min), x_min + 0.60 * (x_max - x_min))
    mask = ~((X > gap[0]) & (X < gap[1]))
    X_train, y_train = X[mask], y[mask]

    X_test = np.linspace(x_min, x_max, 500)
    y_test = datafunction(X_test)
    return X_train, y_train, X_test, y_test

###############################################################################
# 2.  Circuit‑Aufbau (unchanged)                                               #
###############################################################################

def encoding_circuit(encoded_phase: float) -> pcvl.Circuit:
    c = pcvl.Circuit(2, name="Encoding")
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=encoded_phase))
    c.add((0, 1), pcvl.BS())
    return c


def memristor_circuit(phi1: float, mem_phi: float, phi3: float) -> pcvl.Circuit:
    c = pcvl.Circuit(3, name="Memristor")
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi1)).add((0, 1), pcvl.BS())
    c.add((1, 2), pcvl.BS()).add((2,), pcvl.PS(phi=mem_phi)).add((1, 2), pcvl.BS())
    c.add((0, 1), pcvl.BS()).add((1,), pcvl.PS(phi=phi3)).add((0, 1), pcvl.BS())
    return c


def build_circuit(phi1: float, mem_phi: float, phi3: float, enc_phi: float) -> pcvl.Circuit:
    c = pcvl.Circuit(3, name="Full")
    c.add(0, encoding_circuit(enc_phi))
    c.add(0, memristor_circuit(phi1, mem_phi, phi3))
    return c

###############################################################################
# 3.  NumPy‑Simulation                                                         #
###############################################################################

def run_simulation_sequence_np(
    params: np.ndarray,                # [phi1, phi3, w]
    encoded_phases: np.ndarray,        # (B,)
    memory_depth: int,
) -> np.ndarray:
    """Return p(001) for every encoded phase (NumPy backend)."""

    phi1, phi3, w = params
    input_state = pcvl.BasicState([0, 1, 0])
    state_001, state_010 = pcvl.BasicState([0, 0, 1]), pcvl.BasicState([0, 1, 0])

    mem_p1 = np.zeros(memory_depth)
    mem_p2 = np.zeros(memory_depth)
    preds  = np.zeros_like(encoded_phases)

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
        probs = Sampler(proc).probs(1000)["results"]
        p001 = probs.get(state_001, 0.0)
        p010 = probs.get(state_010, 0.0)

        preds[i] = p001
        mem_p1[t], mem_p2[t] = p010, p001

    return preds

###############################################################################
# 4.  Corrected PSR‑Coefficients (Torch)                                       #
###############################################################################

@lru_cache(maxsize=None)
def photonic_psr_coeffs_torch(n: int) -> tuple[Tensor, Tensor]:
    """Return (shifts, c_p) in float64 on CPU with corrected signs."""
    P = 2 * n
    shifts = 2 * np.pi * np.arange(1, P + 1) / (2 * n + 1)

    # *** Sign fix:  -1j instead of  +1j  ***
    grad_vec = -1j * np.concatenate((np.arange(1, n + 1), -np.arange(n, 0, -1)))
    cp_full  = np.fft.ifft(np.concatenate(([0], grad_vec)))

    shifts_t = torch.from_numpy(shifts).double()
    cp_t     = torch.from_numpy(np.real_if_close(cp_full[1:])).double()
    return shifts_t, cp_t

###############################################################################
# 5.  Autograd Function   (PSR  +  Chain rule)                                 #
###############################################################################

class MemristorLossPSR(torch.autograd.Function):
    """Forward: MSE‑Loss; Backward: analytic PSR + FD."""

    @staticmethod
    def forward(ctx, theta: Tensor, enc_phases: Tensor, y: Tensor,
                memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int]) -> Tensor:
        theta_np, enc_np, y_np = map(lambda t: t.detach().cpu().double().numpy(),
                                     (theta, enc_phases, y))
        preds = run_simulation_sequence_np(theta_np, enc_np, memory_depth)
        loss  = 0.5 * ((preds - y_np) ** 2).mean()

        ctx.save_for_backward(theta.detach(), enc_phases.detach(), y.detach())
        ctx.memory_depth, ctx.phase_idx, ctx.n_photons, ctx.preds_np = memory_depth, list(phase_idx), list(n_photons), preds
        return torch.tensor(loss, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def backward(ctx, g_out: Tensor):
        theta, enc_phases, y = ctx.saved_tensors
        N = y.numel()

        theta_np = theta.cpu().double().numpy()
        enc_np   = enc_phases.cpu().double().numpy()
        y_np     = y.cpu().double().numpy()
        preds    = ctx.preds_np
        dL_df    = (preds - y_np) / N                      

        grads = np.zeros_like(theta_np)

        # --- PSR for photonic phases ---
        for gate_i, p_idx in enumerate(ctx.phase_idx):
            shifts, cp = photonic_psr_coeffs_torch(ctx.n_photons[gate_i])
            df_dtheta = np.zeros_like(preds)
            for s, c in zip(shifts.numpy(), cp.numpy()):
                th_shift = theta_np.copy(); th_shift[p_idx] += s
                df_dtheta += c * run_simulation_sequence_np(th_shift, enc_np, ctx.memory_depth)
            grads[p_idx] = np.real(np.dot(dL_df, df_dtheta))

        # --- Finite Difference for the memristor weight ---
        eps = 1e-3
        for idx in set(range(len(theta_np))) - set(ctx.phase_idx):
            th_plus, th_minus = theta_np.copy(), theta_np.copy()
            th_plus[idx]  += eps; th_minus[idx] -= eps
            th_plus[idx]  = np.clip(th_plus[idx], 0.01, 1)
            th_minus[idx] = np.clip(th_minus[idx], 0.01, 1)
            loss_p = 0.5 * ((run_simulation_sequence_np(th_plus,  enc_np, ctx.memory_depth) - y_np) ** 2).mean()
            loss_m = 0.5 * ((run_simulation_sequence_np(th_minus, enc_np, ctx.memory_depth) - y_np) ** 2).mean()
            grads[idx] = (loss_p - loss_m) / (2 * eps)

        return g_out * torch.from_numpy(grads).to(theta), None, None, None, None, None

###############################################################################
# 6.  Model‑Klasse                                                             #
###############################################################################

class PhotonicModel(torch.nn.Module):
    def __init__(self, init_theta: Sequence[float], enc_np: np.ndarray, y_np: np.ndarray,
                 memory_depth: int, phase_idx: Sequence[int], n_photons: Sequence[int]):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.register_buffer("y",   torch.from_numpy(y_np).double())
        self.memory_depth, self.phase_idx, self.n_photons = memory_depth, phase_idx, n_photons

    def forward(self):
        return MemristorLossPSR.apply(self.theta, self.enc, self.y,
                                      self.memory_depth, self.phase_idx, self.n_photons)

###############################################################################
# 7.  Training (discrete & continuous)                                         #
###############################################################################

def _init_theta(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(0.01, 1) * 2 * np.pi,   # φ1
                     rng.uniform(0.01, 1) * 2 * np.pi,   # φ3
                     rng.uniform(0.01, 1)])              # w


def train_pytorch_generic(enc_np: np.ndarray, y_np: np.ndarray, *, memory_depth=2,
                          lr=0.03, epochs=150, phase_idx=(0, 1), n_photons=(1, 1),
                          seed: int = 42):
    rng = np.random.default_rng(seed)
    init_theta = _init_theta(rng)

    model = PhotonicModel(init_theta, enc_np, y_np, memory_depth, phase_idx, n_photons)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist  = []

    for _ in tqdm(range(epochs), desc="Training", ncols=100):
        optim.zero_grad(); loss = model(); loss.backward(); optim.step()
        with torch.no_grad():
            model.theta.data[2].clamp_(0.01, 1.0)        # w ∈ [0.01,1]
            model.theta.data[:2].remainder_(2 * np.pi)   # Phasen ∈ [0,2π)
        hist.append(loss.item())
    return model.theta.detach().cpu().numpy(), hist


def train_pytorch(X: np.ndarray, y: np.ndarray, **kwargs):
    """Original discrete training path (back‑compat wrapper)."""
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(enc, y, **kwargs)


def train_pytorch_cont(X: np.ndarray, y: np.ndarray, *, n_swipe=11, swipe_span=np.pi/20, **kwargs):
    """Continuous‑swipe training path.

    Expands each (X, y) into *n_swipe* neighbouring phase points and then
    hands off to the generic trainer.
    """
    enc_swipe, y_swipe = get_cont_swipe_data(X, y, n_swipe=n_swipe, swipe_span=swipe_span)
    return train_pytorch_generic(enc_swipe, y_swipe, **kwargs)

###############################################################################
# 8.  Gradient‑Check (unchanged)                                               #
###############################################################################

def gradient_check():
    X, y, *_ = get_data(60, 0.0)
    enc = 2 * np.arccos(X)
    theta0 = np.array([1.2, 2.3, 0.5])
    mem_depth = 2

    def L(params):
        return 0.5 * ((run_simulation_sequence_np(params, enc, mem_depth) - y) ** 2).mean()

    # Finite Difference
    eps = 1e-5; num_grad = np.zeros_like(theta0)
    for k in range(len(theta0)):
        p_plus, p_minus = theta0.copy(), theta0.copy()
        p_plus[k] += eps; p_minus[k] -= eps
        num_grad[k] = (L(p_plus) - L(p_minus)) / (2 * eps)

    # PSR / Autograd
    th_t = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    loss = MemristorLossPSR.apply(th_t, torch.from_numpy(enc).double(), torch.from_numpy(y).double(),
                                  mem_depth, (0, 1), (1, 1))
    loss.backward(); psr_grad = th_t.grad.detach().cpu().numpy()

    print("Finite‑diff  :", num_grad)
    print("PSR / Torch :", psr_grad)
    print("Abs‑error   :", np.abs(num_grad - psr_grad))
    print("Max‑error   :", np.abs(num_grad - psr_grad).max())

###############################################################################
# 9.  Main                                                                     #
###############################################################################

# ===================== CONFIGURATION =====================
config = {
    # Data generation
    'n_data': 100,
    'sigma_noise': 0.1,
    'datafunction': quartic_data,
    'memory_depth': 2,

    # Continuous swipe
    'use_continuous': False,
    'n_swipe': 11,
    'swipe_span': np.pi/20,

    # Training
    'lr': 0.03,
    'epochs': 1,
    'phase_idx': (0, 1),
    'n_photons': (1, 1),

    # Model initialization
    'init_theta': None,  # Will be set in trainer
}
# =========================================================



def _run_training(X_train, y_train, X_test, y_test, *, cont: bool):
    """Train model and produce three diagnostic plots."""

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
            swipe_span=config['swipe_span']
        )
    else:
        theta_opt, history = train_pytorch(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            phase_idx=config['phase_idx'],
            n_photons=config['n_photons']
        )

    print("Optimized θ:", theta_opt)

    # ── predictions on dense grid ──
    enc_test = 2 * np.arccos(X_test)
    preds = run_simulation_sequence_np(theta_opt, enc_test, memory_depth=config['memory_depth'])

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
        plt.scatter(X_swipe, np.repeat(y_train, config['n_swipe']), s=8, alpha=0.35, label=f'Swipe (n={config["n_swipe"]})', zorder=2)

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


def main(measured_data: str | None = None, use_continuous: bool = False):
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

    _run_training(X_train, y_train, X_test, y_test, cont=use_continuous)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--check", action="store_true", help="Run gradient check and exit")
    argp.add_argument("--cont",  action="store_true", help="Use continuous‑swipe training")
    argp.add_argument("--data",  type=str, help="Pickle file with measured (X, y) traces")
    args, _ = argp.parse_known_args()

    if args.check:
        gradient_check()
    else:
        main(measured_data=args.data, use_continuous=args.cont or config['use_continuous'])
