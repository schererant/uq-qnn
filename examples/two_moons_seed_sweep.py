#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick sweep of training seeds for two-moons classification.

Uses the same circuit layout as ``two_moons_classification.py`` with a short
epoch budget to rank seeds by final cross-entropy and approximate convergence
speed (loss drop per epoch).  Data split is fixed so differences come only from
initialization (mesh + linear head).

Run:
    uv run python examples/two_moons_seed_sweep.py
    uv run python examples/two_moons_seed_sweep.py --seed-min 0 --seed-max 63 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimConfig
from src.data import encode_2d_to_phases_multi, get_two_moons_data
from src.experiment import Experiment
from src.logging_config import get_logger
from src.training import train_pytorch_generic

logger = get_logger(__name__)

N_LAYERS = 3
N_MODES = 4
PHASES_PER_LAYER = N_MODES * (N_MODES - 1)
ENCODING_PHASE_IDX = tuple(
    layer * PHASES_PER_LAYER + s for layer in range(N_LAYERS) for s in (0, 1)
)

BASE_CONFIG: dict[str, Any] = {
    "n_modes": N_MODES,
    "n_layers": N_LAYERS,
    "input_state": (1, 0, 0, 0),
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "n_enc_features": 2,
    "photon_distinguishability": None,
    "target_mode": (1, 2),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "singles",
    "working_detectors": None,
    "loss_type": "cross_entropy",
    "n_classes": 2,
    "lr": 0.04,
    "epochs": 80,
    "n_samples": 2000,
    "memory_depth": 3,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 0,
    "sim_backend": "numpy",
    "data_n_samples": 1000,
    "data_noise": 0.05,
    "unc_n_passes": 0,
    "unc_noise_std": 0.05,
    "data_seed": 17,
    "seed_min": 0,
    "seed_max": 31,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep training seeds for two moons.")
    p.add_argument("--seed-min", type=int, default=BASE_CONFIG["seed_min"])
    p.add_argument("--seed-max", type=int, default=BASE_CONFIG["seed_max"])
    p.add_argument("--epochs", type=int, default=BASE_CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=BASE_CONFIG["lr"])
    p.add_argument(
        "--data-seed",
        type=int,
        default=BASE_CONFIG["data_seed"],
        help="Fixed sklearn split (independent of training seed).",
    )
    p.add_argument("--top-k-curves", type=int, default=5)
    return p.parse_args()


def _sim_cfg(epochs: int, lr: float, train_seed: int) -> SimConfig:
    cfg = {**BASE_CONFIG, "epochs": epochs, "lr": lr, "seed": train_seed}
    return SimConfig.from_experiment_config(cfg)


def _evaluate_seed(
    train_seed: int,
    *,
    enc_train: np.ndarray,
    y_train: np.ndarray,
    enc_test: np.ndarray,
    y_test: np.ndarray,
    sim_cfg: SimConfig,
    lr: float,
    epochs: int,
) -> dict[str, Any]:
    trained_state, history, model = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=lr,
        epochs=epochs,
        seed=train_seed,
    )
    hist = [float(x) for x in history]
    initial_loss = hist[0]
    final_loss = hist[-1]
    loss_drop = initial_loss - final_loss
    loss_per_epoch = loss_drop / max(epochs, 1)

    preds = np.argmax(model.predict(enc_test), axis=1)
    del model
    test_acc = float(accuracy_score(y_test, preds))

    return {
        "train_seed": train_seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": loss_drop,
        "loss_per_epoch": loss_per_epoch,
        "test_accuracy": test_acc,
        "history": hist,
        "trained_state": trained_state,
    }


def _print_ranking(rows: list[dict[str, Any]], epochs: int) -> None:
    by_final = sorted(rows, key=lambda r: r["final_loss"])
    by_speed = sorted(rows, key=lambda r: r["loss_per_epoch"], reverse=True)

    print(f"\n=== Seed sweep ({len(rows)} seeds, {epochs} epochs) ===\n")
    print("Lowest final loss (best at end of short run):")
    print(f"{'seed':>6}  {'final_loss':>12}  {'loss/ep':>10}  {'test_acc':>9}")
    for r in by_final[:10]:
        print(
            f"{r['train_seed']:6d}  {r['final_loss']:12.6f}  "
            f"{r['loss_per_epoch']:10.6f}  {r['test_accuracy']:9.4f}"
        )

    print("\nFastest loss drop per epoch (steepest short-run descent):")
    print(f"{'seed':>6}  {'loss/ep':>10}  {'final_loss':>12}  {'test_acc':>9}")
    for r in by_speed[:10]:
        print(
            f"{r['train_seed']:6d}  {r['loss_per_epoch']:10.6f}  "
            f"{r['final_loss']:12.6f}  {r['test_accuracy']:9.4f}"
        )

    best = by_final[0]
    print(
        f"\nSuggested seed for a full run: {best['train_seed']} "
        f"(final_loss={best['final_loss']:.6f}, test_acc={best['test_accuracy']:.4f})"
    )


def _plot_results(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    epochs: int,
) -> plt.Figure:
    seeds = [r["train_seed"] for r in rows]
    final_losses = [r["final_loss"] for r in rows]
    loss_per_ep = [r["loss_per_epoch"] for r in rows]

    by_final = sorted(rows, key=lambda r: r["final_loss"])
    top = by_final[:top_k]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax0 = axes[0, 0]
    order = np.argsort(final_losses)
    ax0.bar(
        [seeds[i] for i in order],
        [final_losses[i] for i in order],
        color="steelblue",
        width=0.8,
    )
    ax0.set(xlabel="Training seed", ylabel="Final loss", title="Final cross-entropy")
    ax0.grid(True, axis="y", alpha=0.3)

    ax1 = axes[0, 1]
    sc = ax1.scatter(seeds, final_losses, c=loss_per_ep, cmap="viridis", s=36)
    fig.colorbar(sc, ax=ax1, label="Loss drop / epoch")
    ax1.set(xlabel="Training seed", ylabel="Final loss", title="Final loss vs seed")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1, 0]
    for r in top:
        ax2.plot(r["history"], label=f"seed {r['train_seed']}", alpha=0.85)
    ax2.set_yscale("log")
    ax2.set(
        xlabel="Epoch",
        ylabel="Loss",
        title=f"Top {top_k} seeds by final loss",
    )
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 1]
    speed_order = np.argsort(loss_per_ep)[::-1]
    ax3.bar(
        [seeds[i] for i in speed_order],
        [loss_per_ep[i] for i in speed_order],
        color="darkorange",
        width=0.8,
    )
    ax3.set(
        xlabel="Training seed",
        ylabel="(initial − final) / epoch",
        title="Approx. convergence speed",
    )
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Two moons seed sweep ({epochs} epochs, fixed data split)", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed_min, args.seed_max + 1))
    if not seeds:
        raise ValueError("Empty seed range")

    sweep_config = {
        **BASE_CONFIG,
        "epochs": args.epochs,
        "lr": args.lr,
        "data_seed": args.data_seed,
        "seed_min": args.seed_min,
        "seed_max": args.seed_max,
    }

    X_train, y_train, X_test, y_test = get_two_moons_data(
        n_samples=int(sweep_config["data_n_samples"]),
        noise=float(sweep_config["data_noise"]),
        random_state=args.data_seed,
        return_one_hot=False,
    )
    enc_train = encode_2d_to_phases_multi(X_train, n_layers=N_LAYERS)
    enc_test = encode_2d_to_phases_multi(X_test, n_layers=N_LAYERS)

    with Experiment("Two Moons Seed Sweep", config=sweep_config) as exp:
        rows: list[dict[str, Any]] = []
        for train_seed in seeds:
            logger.info("Training seed %d / %d", train_seed, seeds[-1])
            sim_cfg = _sim_cfg(args.epochs, args.lr, train_seed)
            row = _evaluate_seed(
                train_seed,
                enc_train=enc_train,
                y_train=y_train,
                enc_test=enc_test,
                y_test=y_test,
                sim_cfg=sim_cfg,
                lr=args.lr,
                epochs=args.epochs,
            )
            rows.append(row)

        _print_ranking(rows, args.epochs)

        by_final = sorted(rows, key=lambda r: r["final_loss"])
        best_seed = int(by_final[0]["train_seed"])

        summary_rows = [
            {k: v for k, v in r.items() if k not in ("history", "trained_state")}
            for r in rows
        ]
        results_path = exp.run_dir / "seed_sweep_results.json"
        results_path.write_text(
            json.dumps(summary_rows, indent=2) + "\n",
            encoding="utf-8",
        )
        exp.artifacts.append(str(results_path.resolve()))

        best_row = by_final[0]
        best_path = exp.run_dir / f"trained_state_seed{best_seed}.json"
        best_row["trained_state"].save_json(best_path)
        exp.artifacts.append(str(best_path.resolve()))
        del best_row["trained_state"]

        exp.save_metrics(
            {
                "sweep_epochs": args.epochs,
                "sweep_lr": args.lr,
                "data_seed": args.data_seed,
                "n_seeds": len(seeds),
                "best_seed_final_loss": best_row["final_loss"],
                "best_seed_loss_per_epoch": best_row["loss_per_epoch"],
                "best_seed_test_accuracy": best_row["test_accuracy"],
                "recommended_train_seed": best_seed,
            }
        )

        fig = _plot_results(rows, top_k=args.top_k_curves, epochs=args.epochs)
        exp.savefig(fig, "two_moons_seed_sweep.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
