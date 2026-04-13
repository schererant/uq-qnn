from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .circuits import normalize_memristive_phase_idx
from .config import SimConfig, validate_sim_config
from .logging_config import get_logger, log_params
from .loss import PhotonicModel, TrainedPhotonicState

logger = get_logger(__name__)


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
    memristive_indices = normalize_memristive_phase_idx(
        memristive_phase_idx, n_modes, expected_phases
    )
    if len(memristive_indices) == 0:
        return phases
    weights = rng.uniform(0.01, 1, size=len(memristive_indices))
    return np.concatenate([phases, weights])


def train_pytorch_generic(
    enc_np: np.ndarray,
    y_np: np.ndarray,
    *,
    sim_cfg: SimConfig,
    lr: float,
    epochs: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[TrainedPhotonicState, List[float], PhotonicModel]:
    """
    Trains the photonic model and returns serialized state, loss history, and model.
    Args:
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        sim_cfg (SimConfig): Circuit and simulation parameters.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
        verbose (bool): If True, print per-epoch loss and final parameters.
    Returns:
        Serialized trained state, loss history, and trained :class:`PhotonicModel`.
    """
    sim_cfg_work = sim_cfg
    # Validate classification setup (singles only: target_mode lists output modes per class).
    # Coincidence classification uses C(W, N) channels from working_detectors; target_mode
    # does not define the class basis (see validate_sim_config + PhotonicModel).
    if sim_cfg_work.loss_type == "cross_entropy":
        if sim_cfg_work.output_mode == "coincidence":
            pass
        elif sim_cfg_work.target_mode is None:
            if sim_cfg_work.n_classes > sim_cfg_work.n_modes:
                raise ValueError(
                    f"For {sim_cfg_work.n_classes} classes, need at least "
                    f"{sim_cfg_work.n_classes} modes, got {sim_cfg_work.n_modes}"
                )
            sim_cfg_work = sim_cfg_work.replace(
                target_mode=tuple(range(sim_cfg_work.n_classes))
            )
        elif len(sim_cfg_work.target_mode) != sim_cfg_work.n_classes:
            raise ValueError(
                f"For classification with n_classes={sim_cfg_work.n_classes}, "
                f"target_mode must have {sim_cfg_work.n_classes} elements, "
                f"got {len(sim_cfg_work.target_mode)}"
            )

    validate_sim_config(sim_cfg_work)

    if verbose:
        params = {
            **sim_cfg_work.to_dict(),
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
        }
        log_params(logger, params)

    rng = np.random.default_rng(seed)
    init_theta = _init_theta(
        rng, sim_cfg_work.n_modes, sim_cfg_work.memristive_phase_idx
    )

    expected_phases = sim_cfg_work.n_modes * (sim_cfg_work.n_modes - 1)
    memristive_indices = normalize_memristive_phase_idx(
        sim_cfg_work.memristive_phase_idx, sim_cfg_work.n_modes, expected_phases
    )
    enc_pi = int(sim_cfg_work.encoding_phase_idx)
    phase_idx = tuple(
        i for i in range(expected_phases) if i not in memristive_indices and i != enc_pi
    )

    model = PhotonicModel(
        init_theta.tolist(),
        enc_np,
        phase_idx,
        sim_cfg_work,
    )
    if sim_cfg_work.loss_type == "cross_entropy":
        if y_np.ndim == 2:
            y_t = torch.from_numpy(np.argmax(y_np, axis=1)).long()
        else:
            y_t = torch.from_numpy(y_np).long()
    else:
        y_t = torch.from_numpy(y_np).double()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    pbar = tqdm(range(epochs), desc="Training", ncols=100)
    for e in pbar:
        optim.zero_grad()
        loss = model(
            y=y_t,
            n_samples=sim_cfg_work.n_samples,
            n_swipe=sim_cfg_work.n_swipe,
            swipe_span=sim_cfg_work.swipe_span,
        )
        loss.backward()
        optim.step()
        with torch.no_grad():
            # Only appended memristor weights are clamped; all mesh phases stay on the circle.
            weight_idxs = range(expected_phases, len(model.theta))
            for idx in weight_idxs:
                model.theta.data[idx].clamp_(0.01, 1.0)
            for idx in range(expected_phases):
                model.theta.data[idx].remainder_(2 * np.pi)

        hist.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    trained_state = model.export_state()
    if verbose:
        logger.info("Final parameters: %s", trained_state.theta)
    return trained_state, hist, model


def train_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sim_cfg: SimConfig,
    lr: float,
    epochs: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[TrainedPhotonicState, List[float], PhotonicModel]:
    """
    Unified training path for both discrete and continuous modes.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        sim_cfg (SimConfig): Circuit and simulation parameters.
        **kwargs: Additional arguments for training.
    Returns:
        Serialized trained state, loss history, and trained :class:`PhotonicModel`.
    """
    enc = 2 * np.arccos(X)
    return train_pytorch_generic(
        enc,
        y,
        sim_cfg=sim_cfg,
        lr=lr,
        epochs=epochs,
        seed=seed,
        verbose=verbose,
    )


def gradient_check(
    n_modes: int = 3, memristive_phase_idx: Optional[Union[int, Sequence[int]]] = None
) -> None:
    """
    Performs a gradient check comparing finite difference and analytic gradients.
    """
    from .autograd import MemristorLossPSR
    from .data import get_data
    from .simulation import run_simulation_sequence_np

    X, y, *_ = get_data(60, 0.0)
    enc = 2 * np.arccos(X)
    n_phases = n_modes * (n_modes - 1)
    memristive_indices = normalize_memristive_phase_idx(
        memristive_phase_idx, n_modes, n_phases
    )
    n_memristive = len(memristive_indices)
    non_mem = [i for i in range(n_phases) if i not in memristive_indices]
    enc_slot = non_mem[0]
    phase_idx = tuple(i for i in non_mem if i != enc_slot)

    theta0 = np.random.rand(n_phases + n_memristive)
    theta0[:n_phases] *= 2 * np.pi
    theta0[n_phases:] *= 0.5
    mem_depth = 2
    n_samples = 5

    cfg = SimConfig(
        n_modes=n_modes,
        input_state=tuple(1 if i == 0 else 0 for i in range(n_modes)),
        encoding_phase_idx=enc_slot,
        photon_distinguishability=None,
        target_mode=(n_modes - 1,) if n_modes else None,
        memristive_phase_idx=(
            None
            if memristive_phase_idx is None
            else int(memristive_phase_idx)
            if isinstance(memristive_phase_idx, int)
            else tuple(int(idx) for idx in memristive_phase_idx)
        ),
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=n_samples,
        memory_depth=mem_depth,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="mse",
        n_classes=1,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )
    validate_sim_config(cfg)

    def L(params):
        return 0.5 * ((run_simulation_sequence_np(params, enc, cfg) - y) ** 2).mean()

    eps = 1e-5
    num_grad = np.zeros_like(theta0)
    for k in range(len(theta0)):
        p_plus, p_minus = theta0.copy(), theta0.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        num_grad[k] = (L(p_plus) - L(p_minus)) / (2 * eps)

    th_t = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    feats = MemristorLossPSR.apply(
        th_t,
        torch.from_numpy(enc).double(),
        phase_idx,
        cfg,
    )
    loss = 0.5 * torch.mean((feats[:, -1] - torch.from_numpy(y).double()) ** 2)
    loss.backward()
    assert th_t.grad is not None
    psr_grad = th_t.grad.detach().cpu().numpy()
    # PSR omits the frozen encoding slot; skip it in the FD comparison.
    compare_mask = np.ones(len(theta0), dtype=bool)
    compare_mask[enc_slot] = False
    logger.info("Finite‑diff  : %s", num_grad)
    logger.info("PSR / Torch : %s", psr_grad)
    logger.info("Abs‑error   : %s", np.abs(num_grad - psr_grad))
    logger.info(
        "Max‑error (trainable) : %s",
        np.abs(num_grad[compare_mask] - psr_grad[compare_mask]).max(),
    )
