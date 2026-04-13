#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expressivity improvement strategies for photonic QNN regression.

Diagnoses and addresses why multi-modal, step, and sinusoidal targets are
harder to fit than quartic/cubic.  Four strategies are compared:

  baseline      — 6-mode singles, 150 epochs, raw targets
  normalized    — 6-mode singles, 150 epochs, targets scaled to [0.05, 0.95]
                  (aligns label range with Born-rule output distribution)
  large_circuit — 12-mode singles, 150 epochs, raw targets
                  (doubles the number of photonic basis functions: 6 → 12)
  combined      — 12-mode singles, 300 epochs, normalized targets
                  (all improvements together)

The photonic model always uses ALL output-mode probabilities as features
(via MemristorLossPSR._feature_cfg), so:
  • 6-mode  → Linear(6,  1) readout head
  • 12-mode → Linear(12, 1) readout head

Produced artefacts
------------------
predictions_grid.png   Fit + 95 % CI for every (function, strategy) pair
r2_heatmap.png         R² heatmap — quick expressivity overview
training_loss.png      Log-scale convergence curves per function
"""

import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimConfig
from src.data import get_data
from src.experiment import Experiment
from src.logging_config import get_logger
from src.loss import PhotonicModel
from src.training import train_pytorch_generic

logger = get_logger(__name__)

# ── Target functions ────────────────────────────────────────────────────────────
FUNCTIONS = ["quartic_data", "sinusoid_data", "multi_modal_data", "step_function_data"]
FUNC_LABELS = {
    "quartic_data": r"$x^4$  (easy baseline)",
    "sinusoid_data": r"$\sin(0.7\pi x)$  (non-monotonic)",
    "multi_modal_data": "Multi-modal Gaussians  (peaked)",
    "step_function_data": "Smooth step  (sharp transition)",
}

# ── Normalization range ─────────────────────────────────────────────────────────
# Born-rule probabilities for a 6-mode circuit average ~1/6 ≈ 0.17, rarely
# reaching 0 or 1.  Squeezing targets into [NORM_LOW, NORM_HIGH] reduces the
# linear head's required weight magnitude and improves gradient flow.
NORM_LOW: float = 0.05
NORM_HIGH: float = 0.95

# ── Common hyperparameters shared across all strategies ─────────────────────────
_SHARED = dict(
    photon_distinguishability=None,
    memristive_phase_idx=None,
    memristive_output_modes=None,
    output_mode="singles",
    working_detectors=None,
    loss_type="mse",
    n_classes=1,
    n_samples=0,
    memory_depth=1,
    n_swipe=0,
    swipe_span=0.0,
    noise_std=None,
    sim_backend="numpy",
    n_data=500,
    sigma_noise=0.02,
    lr=0.05,
    seed=42,
    unc_n_passes=15,
    unc_noise_std=0.05,
)

# ── Strategy definitions ────────────────────────────────────────────────────────
# Keys normalize_targets / epochs / label / color are experiment-only and are
# stripped before passing to SimConfig.
STRATEGIES: Dict[str, Dict] = {
    "baseline": dict(
        n_modes=6,
        input_state=tuple(1 if i == 0 else 0 for i in range(6)),
        encoding_phase_idx=26,
        target_mode=(2,),
        epochs=150,
        normalize_targets=False,
        label="Baseline\n(6-mode, 150 ep)",
        color="#1f77b4",
    ),
    "normalized": dict(
        n_modes=6,
        input_state=tuple(1 if i == 0 else 0 for i in range(6)),
        encoding_phase_idx=26,
        target_mode=(2,),
        epochs=150,
        normalize_targets=True,
        label="Normalized targets\n(6-mode, 150 ep)",
        color="#ff7f0e",
    ),
    "large_circuit": dict(
        n_modes=12,
        input_state=tuple(1 if i == 0 else 0 for i in range(12)),
        encoding_phase_idx=26,
        target_mode=(6,),
        epochs=150,
        normalize_targets=False,
        label="Large circuit\n(12-mode, 150 ep)",
        color="#2ca02c",
    ),
    "combined": dict(
        n_modes=12,
        input_state=tuple(1 if i == 0 else 0 for i in range(12)),
        encoding_phase_idx=26,
        target_mode=(6,),
        epochs=300,
        normalize_targets=True,
        label="Combined\n(12-mode, norm, 300 ep)",
        color="#d62728",
    ),
}

# CONFIG dict satisfies Experiment's _REQUIRED_CONFIG_KEYS; uses baseline circuit.
CONFIG = {
    **_SHARED,
    "n_modes": 6,
    "input_state": tuple(1 if i == 0 else 0 for i in range(6)),
    "encoding_phase_idx": 26,
    "photon_distinguishability": None,
    "target_mode": (2,),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "singles",
    "working_detectors": None,
    "loss_type": "mse",
    "n_classes": 1,
    "epochs": 150,
    "strategies": list(STRATEGIES.keys()),
    "data_functions": FUNCTIONS,
}

# ── Normalisation helpers ───────────────────────────────────────────────────────


def _scale_targets(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Scale y to [NORM_LOW, NORM_HIGH]; returns (y_norm, y_min, y_max)."""
    y_min, y_max = float(y.min()), float(y.max())
    span = y_max - y_min
    if span < 1e-9:
        return np.full_like(y, (NORM_LOW + NORM_HIGH) / 2.0), y_min, y_max
    return NORM_LOW + (y - y_min) / span * (NORM_HIGH - NORM_LOW), y_min, y_max


def _unscale_predictions(
    preds: np.ndarray, y_min: float, y_max: float
) -> np.ndarray:
    """Invert _scale_targets."""
    return y_min + (preds - NORM_LOW) / (NORM_HIGH - NORM_LOW) * (y_max - y_min)


# ── SimConfig builder ────────────────────────────────────────────────────────────


def _build_sim_cfg(strategy_name: str) -> SimConfig:
    strat = STRATEGIES[strategy_name]
    cfg = {**_SHARED, **strat}
    for key in ("normalize_targets", "epochs", "label", "color"):
        cfg.pop(key, None)
    return SimConfig.from_experiment_config(cfg)


# ── Uncertainty estimation ───────────────────────────────────────────────────────


def _run_uncertainty(
    model: PhotonicModel,
    enc_test: np.ndarray,
    *,
    n_passes: int,
    noise_std: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta_base = model.theta.detach().cpu().numpy()
    all_preds = np.zeros((len(enc_test), n_passes))
    for i in range(n_passes):
        perturbed = theta_base + rng.normal(0.0, noise_std, size=len(theta_base))
        all_preds[:, i] = model.predict(enc_test, theta_np=perturbed)
    return {"mean": np.mean(all_preds, axis=1), "std": np.std(all_preds, axis=1)}


# ── Metrics ──────────────────────────────────────────────────────────────────────


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ── Core training ────────────────────────────────────────────────────────────────


def train_strategy(
    strategy_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    strat = STRATEGIES[strategy_name]
    normalize = bool(strat["normalize_targets"])
    epochs = int(strat["epochs"])
    seed = int(_SHARED["seed"])

    y_min: Optional[float] = None
    y_max: Optional[float] = None
    if normalize:
        y_tr, y_min, y_max = _scale_targets(y_train)
    else:
        y_tr = y_train

    sc = _build_sim_cfg(strategy_name)
    enc_train = 2.0 * np.arccos(np.clip(X_train, 0.0, 1.0))
    enc_test = 2.0 * np.arccos(np.clip(X_test, 0.0, 1.0))

    _, history, model = train_pytorch_generic(
        enc_train,
        y_tr,
        sim_cfg=sc,
        lr=float(_SHARED["lr"]),
        epochs=epochs,
        seed=seed,
    )

    unc = _run_uncertainty(
        model,
        enc_test,
        n_passes=int(_SHARED["unc_n_passes"]),
        noise_std=float(_SHARED["unc_noise_std"]),
        seed=seed,
    )

    mean_preds = unc["mean"]
    std_preds = unc["std"]

    if normalize:
        assert y_min is not None and y_max is not None
        mean_preds = _unscale_predictions(mean_preds, y_min, y_max)
        # std scales linearly with the same factor
        std_preds = std_preds * (y_max - y_min) / (NORM_HIGH - NORM_LOW)

    r2 = _r2(y_test, mean_preds)
    rmse = float(np.sqrt(np.mean((mean_preds - y_test) ** 2)))
    logger.info(
        "  %-14s  R²=%+.3f  RMSE=%.4f  loss=%.6f",
        strategy_name,
        r2,
        rmse,
        float(history[-1]),
    )
    return {
        "history": history,
        "mean_preds": mean_preds,
        "std_preds": std_preds,
        "r2": r2,
        "rmse": rmse,
    }


# ── Plots ────────────────────────────────────────────────────────────────────────


def plot_predictions_grid(
    results: Dict,
    data: Dict,
    functions: list,
    strategies: list,
) -> plt.Figure:
    n_f, n_s = len(functions), len(strategies)
    fig, axes = plt.subplots(n_f, n_s, figsize=(5 * n_s, 3.5 * n_f))
    axes = np.atleast_2d(axes)

    for row, func in enumerate(functions):
        X_train, y_train, X_test, y_test = data[func]
        for col, strat_name in enumerate(strategies):
            ax = axes[row, col]
            r = results[func][strat_name]
            color = STRATEGIES[strat_name]["color"]

            ax.scatter(X_train, y_train, s=8, alpha=0.30, color="dimgray", zorder=1)
            ax.plot(X_test, y_test, "k--", lw=1.2, label="Ground truth", zorder=2)
            ax.plot(
                X_test, r["mean_preds"], color=color, lw=2.0, label="Prediction", zorder=3
            )
            ax.fill_between(
                X_test,
                r["mean_preds"] - 1.96 * r["std_preds"],
                r["mean_preds"] + 1.96 * r["std_preds"],
                color=color,
                alpha=0.18,
                label="95 % CI",
                zorder=0,
            )
            title = (
                f"{FUNC_LABELS[func]}\n"
                f"{STRATEGIES[strat_name]['label'].replace(chr(10), ' · ')}\n"
                f"R²={r['r2']:.3f}   RMSE={r['rmse']:.4f}"
            )
            ax.set_title(title, fontsize=7.5)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.suptitle(
        "Expressivity Improvements — Photonic QNN Regression",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_r2_heatmap(
    results: Dict, functions: list, strategies: list
) -> plt.Figure:
    matrix = np.array(
        [[results[f][s]["r2"] for s in strategies] for f in functions]
    )
    fig, ax = plt.subplots(figsize=(len(strategies) * 2.2 + 1.2, len(functions) + 1.0))
    im = ax.imshow(matrix, vmin=-0.2, vmax=1.0, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="R²", fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(
        [STRATEGIES[s]["label"].replace("\n", " · ") for s in strategies],
        rotation=18,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(len(functions)))
    ax.set_yticklabels([FUNC_LABELS[f] for f in functions], fontsize=9)
    ax.set_title("R² Score — Functions × Strategies\n(green = good fit)", fontsize=11)

    for i in range(len(functions)):
        for j in range(len(strategies)):
            v = matrix[i, j]
            txt_color = "white" if abs(v) > 0.6 else "black"
            ax.text(
                j, i, f"{v:.3f}", ha="center", va="center",
                color=txt_color, fontsize=11, fontweight="bold",
            )

    fig.tight_layout()
    return fig


def plot_training_loss(
    results: Dict, functions: list, strategies: list
) -> plt.Figure:
    n_f = len(functions)
    fig, axes = plt.subplots(1, n_f, figsize=(5 * n_f, 4))
    axes = np.atleast_1d(axes)

    for ax, func in zip(axes, functions):
        for strat_name in strategies:
            ax.plot(
                results[func][strat_name]["history"],
                color=STRATEGIES[strat_name]["color"],
                lw=1.6,
                alpha=0.85,
                label=STRATEGIES[strat_name]["label"].replace("\n", " "),
            )
        ax.set_yscale("log")
        ax.set_title(FUNC_LABELS[func].split(" ")[0], fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Training Convergence by Strategy", fontsize=12)
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    with Experiment("Function Expressivity Improvements", config=CONFIG) as exp:
        np.random.seed(int(CONFIG["seed"]))

        data: Dict = {}
        for func_name in FUNCTIONS:
            data[func_name] = get_data(
                int(CONFIG["n_data"]),
                float(CONFIG["sigma_noise"]),
                func_name,
            )
            logger.info(
                "Loaded %-24s  train=%d  test=%d",
                func_name,
                len(data[func_name][0]),
                len(data[func_name][2]),
            )

        strategy_names = list(STRATEGIES.keys())
        results: Dict = {}
        all_metrics: Dict = {}

        for func_name in FUNCTIONS:
            results[func_name] = {}
            all_metrics[func_name] = {}
            X_train, y_train, X_test, y_test = data[func_name]
            logger.info("── %s ──", func_name)
            for strat_name in strategy_names:
                r = train_strategy(strat_name, X_train, y_train, X_test, y_test)
                results[func_name][strat_name] = r
                all_metrics[func_name][strat_name] = {
                    "r2": r["r2"],
                    "rmse": r["rmse"],
                    "final_loss": float(r["history"][-1]),
                }

        exp.save_metrics({"by_function_and_strategy": all_metrics})

        figs = [
            (
                "predictions_grid.png",
                plot_predictions_grid(results, data, FUNCTIONS, strategy_names),
            ),
            (
                "r2_heatmap.png",
                plot_r2_heatmap(results, FUNCTIONS, strategy_names),
            ),
            (
                "training_loss.png",
                plot_training_loss(results, FUNCTIONS, strategy_names),
            ),
        ]
        for fname, fig in figs:
            exp.savefig(fig, fname, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", fname)

        logger.info("Done → %s", exp.run_dir)


if __name__ == "__main__":
    main()
