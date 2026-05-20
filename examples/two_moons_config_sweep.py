#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two Moons configuration sweep.

Compares alternative photonic-circuit / training settings against the current
``two_moons_classification.py`` baseline. Each variation is run across a small
seed set so the ranking reflects the variation, not initialization luck.

Axes covered:
  * mesh width   (n_modes ∈ {4, 5, 6})
  * depth        (n_layers ∈ {3, 4, 5})
  * lr schedule  (lr=0.04/1000ep vs lr=0.015/2500ep)
  * data noise   (data_noise ∈ {0.05, 0.08})
  * encoding density (2 slots/layer vs 4 slots/layer)

Note on coincidences: 2-photon cross-entropy classification is *not* a free
axis here — config validation requires ``n_classes == C(W, N)`` for
``output_mode='coincidence'``, which has no integer solution for binary
labels. To explore coincidences, switch the task (e.g. multi-class
classification with 3 classes and W=3, N=2 → C(3,2)=3).

Run:
    uv run python examples/two_moons_config_sweep.py
    uv run python examples/two_moons_config_sweep.py --seeds 0,1,2,3,4
    uv run python examples/two_moons_config_sweep.py --variations baseline,wide_n5,deep_l5
    uv run python examples/two_moons_config_sweep.py --epochs-scale 0.5  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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


# ─────────────────────────────────────────────────────────────────────────────
# Variation definitions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Variation:
    """One photonic-circuit / training config to compare.

    ``encoding_slots`` controls how many *distinct* phase slots per layer get the
    input. Default 2 matches ``encode_2d_to_phases_multi`` (x₁→slot 0, x₂→slot 1).
    Setting 4 means each input is uploaded into two phase slots per layer
    (x₁→slots 0 and ``n_modes-1``; x₂→slots 1 and ``n_modes``), which doubles
    the entry points for the data signal without changing data dimensionality.
    """

    name: str
    n_modes: int
    n_layers: int
    lr: float
    epochs: int
    data_noise: float
    encoding_slots: int = 2  # 2 = default Mauser block; 4 = duplicated block
    note: str = ""


def _phases_per_layer(n_modes: int) -> int:
    return n_modes * (n_modes - 1)


def _encoding_phase_idx(n_modes: int, n_layers: int, slots: int) -> tuple[int, ...]:
    ppl = _phases_per_layer(n_modes)
    if slots == 2:
        local = (0, 1)
    elif slots == 4:
        # Pair two non-adjacent phase slots per input. The first beam-splitter
        # column in a Clements mesh has its phases at indices 0..n_modes-1; we
        # pick two of them per input so the encoded phase has two entry points
        # into the unitary per re-uploading layer.
        if n_modes < 4:
            raise ValueError("encoding_slots=4 requires n_modes>=4")
        local = (0, 1, n_modes - 2, n_modes - 1)
    else:
        raise ValueError(f"Unsupported encoding_slots={slots}")
    return tuple(layer * ppl + s for layer in range(n_layers) for s in local)


def _encode_inputs(X_2d: np.ndarray, slots: int, n_layers: int) -> np.ndarray:
    """Encode (n, 2) → (n, slots*n_layers) phases via 2*arccos.

    For slots=2: identical to ``encode_2d_to_phases_multi``.
    For slots=4: (x₁, x₂) → (φ₁, φ₂, φ₁, φ₂), repeated per layer.
    """
    base = encode_2d_to_phases_multi(X_2d, n_layers=1)  # shape (n, 2)
    if slots == 2:
        block = base
    elif slots == 4:
        block = np.concatenate([base, base], axis=1)  # (n, 4)
    else:
        raise ValueError(f"Unsupported encoding_slots={slots}")
    return np.tile(block, (1, n_layers))


# Baseline matches examples/two_moons_classification.py.
VARIATIONS: list[Variation] = [
    Variation(
        name="baseline",
        n_modes=4,
        n_layers=3,
        lr=0.04,
        epochs=1000,
        data_noise=0.08,
        encoding_slots=2,
        note="current two_moons_classification.py config",
    ),
    Variation(
        name="wide_n5",
        n_modes=5,
        n_layers=3,
        lr=0.04,
        epochs=1000,
        data_noise=0.08,
        encoding_slots=2,
        note="wider mesh (5 modes, 60 phases)",
    ),
    Variation(
        name="wide_n6",
        n_modes=6,
        n_layers=3,
        lr=0.04,
        epochs=1000,
        data_noise=0.08,
        encoding_slots=2,
        note="wider mesh (6 modes, 90 phases)",
    ),
    Variation(
        name="deep_l4",
        n_modes=4,
        n_layers=4,
        lr=0.04,
        epochs=1000,
        data_noise=0.08,
        encoding_slots=2,
        note="4 re-uploading layers",
    ),
    Variation(
        name="deep_l5",
        n_modes=4,
        n_layers=5,
        lr=0.04,
        epochs=1000,
        data_noise=0.08,
        encoding_slots=2,
        note="5 re-uploading layers",
    ),
    Variation(
        name="enc_4slots",
        n_modes=4,
        n_layers=3,
        lr=0.04,
        epochs=1000,
        data_noise=0.08,
        encoding_slots=4,
        note="duplicate each input into 2 phase slots per layer",
    ),
    Variation(
        name="slow_lr_long",
        n_modes=4,
        n_layers=3,
        lr=0.015,
        epochs=2500,
        data_noise=0.08,
        encoding_slots=2,
        note="lower lr, longer training",
    ),
    Variation(
        name="clean_data",
        n_modes=4,
        n_layers=3,
        lr=0.04,
        epochs=1000,
        data_noise=0.05,
        encoding_slots=2,
        note="cleaner data (noise=0.05) — diagnoses model-vs-data ceiling",
    ),
    Variation(
        name="wide_deep",
        n_modes=5,
        n_layers=4,
        lr=0.025,
        epochs=2000,
        data_noise=0.08,
        encoding_slots=2,
        note="5 modes × 4 layers @ lower lr — combined capacity bump",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def _build_config(v: Variation, train_seed: int, data_seed: int) -> dict[str, Any]:
    return {
        "n_modes": v.n_modes,
        "n_layers": v.n_layers,
        "input_state": tuple([1] + [0] * (v.n_modes - 1)),
        "encoding_phase_idx": _encoding_phase_idx(v.n_modes, v.n_layers, v.encoding_slots),
        "n_enc_features": v.encoding_slots,
        "photon_distinguishability": None,
        "target_mode": (1, 2),
        "memristive_phase_idx": None,
        "memristive_output_modes": None,
        "output_mode": "singles",
        "working_detectors": None,
        "loss_type": "cross_entropy",
        "n_classes": 2,
        "lr": v.lr,
        "epochs": v.epochs,
        "n_samples": 2000,
        "memory_depth": 3,
        "n_swipe": 0,
        "swipe_span": 0.0,
        "noise_std": None,
        "seed": train_seed,
        "sim_backend": "numpy",
        "data_n_samples": 2000,
        "data_noise": v.data_noise,
        "unc_n_passes": 0,
        "unc_noise_std": 0.05,
        "data_seed": data_seed,
    }


def _train_eval(
    v: Variation,
    *,
    train_seed: int,
    data_seed: int,
    epochs_scale: float,
) -> dict[str, Any]:
    epochs = max(1, int(round(v.epochs * epochs_scale)))
    cfg = _build_config(v, train_seed=train_seed, data_seed=data_seed)
    cfg["epochs"] = epochs
    sim_cfg = SimConfig.from_experiment_config(cfg)

    X_train, y_train, X_test, y_test = get_two_moons_data(
        n_samples=int(cfg["data_n_samples"]),
        noise=float(cfg["data_noise"]),
        random_state=data_seed,
        return_one_hot=False,
    )
    enc_train = _encode_inputs(X_train, v.encoding_slots, v.n_layers)
    enc_test = _encode_inputs(X_test, v.encoding_slots, v.n_layers)

    trained_state, history, model = train_pytorch_generic(
        enc_train,
        y_train,
        sim_cfg=sim_cfg,
        lr=v.lr,
        epochs=epochs,
        seed=train_seed,
    )
    hist = [float(x) for x in history]
    preds = np.argmax(model.predict(enc_test), axis=1)
    del model

    return {
        "variation": v.name,
        "train_seed": train_seed,
        "epochs": epochs,
        "final_loss": hist[-1],
        "initial_loss": hist[0],
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "history": hist,
        "n_trainable_phases": len(sim_cfg.theta_init_phase_idx)
        if hasattr(sim_cfg, "theta_init_phase_idx")
        else _phases_per_layer(v.n_modes) * v.n_layers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation, printing, plotting
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_var: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_var.setdefault(r["variation"], []).append(r)
    summary = []
    for name, group in by_var.items():
        accs = np.array([r["test_accuracy"] for r in group])
        losses = np.array([r["final_loss"] for r in group])
        summary.append(
            {
                "variation": name,
                "n_seeds": len(group),
                "acc_mean": float(accs.mean()),
                "acc_std": float(accs.std(ddof=0)),
                "acc_max": float(accs.max()),
                "acc_min": float(accs.min()),
                "loss_mean": float(losses.mean()),
                "loss_std": float(losses.std(ddof=0)),
                "loss_min": float(losses.min()),
                "epochs": group[0]["epochs"],
                "n_trainable_phases": group[0]["n_trainable_phases"],
            }
        )
    summary.sort(key=lambda r: -r["acc_mean"])
    return summary


def _print_ranking(summary: list[dict[str, Any]], variations: list[Variation]) -> None:
    notes = {v.name: v.note for v in variations}
    print("\n=== Two-moons config sweep (mean ± std over seeds) ===\n")
    print(
        f"{'variation':<14} {'phases':>7} {'epochs':>7} "
        f"{'acc_mean':>9} {'acc_std':>8} {'acc_max':>8} "
        f"{'loss_mean':>10} {'loss_min':>10}  note"
    )
    for r in summary:
        print(
            f"{r['variation']:<14} {r['n_trainable_phases']:7d} {r['epochs']:7d} "
            f"{r['acc_mean']:9.4f} {r['acc_std']:8.4f} {r['acc_max']:8.4f} "
            f"{r['loss_mean']:10.4f} {r['loss_min']:10.4f}  {notes.get(r['variation'], '')}"
        )
    best = summary[0]
    print(
        f"\nBest by mean test accuracy: {best['variation']} "
        f"(acc={best['acc_mean']:.4f} ± {best['acc_std']:.4f}, "
        f"loss_mean={best['loss_mean']:.4f})"
    )


def _plot(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    *,
    output_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    names = [s["variation"] for s in summary]

    # accuracy bar (sorted high → low)
    ax = axes[0, 0]
    means = [s["acc_mean"] for s in summary]
    stds = [s["acc_std"] for s in summary]
    ax.bar(names, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set(ylabel="Test accuracy", title="Test accuracy (mean ± std)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # final loss bar
    ax = axes[0, 1]
    lmeans = [s["loss_mean"] for s in summary]
    lstds = [s["loss_std"] for s in summary]
    ax.bar(names, lmeans, yerr=lstds, capsize=4, color="darkorange", alpha=0.85)
    ax.set(ylabel="Final cross-entropy", title="Final loss (mean ± std)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    # mean loss curve per variation
    ax = axes[1, 0]
    by_var: dict[str, list[list[float]]] = {}
    for r in rows:
        by_var.setdefault(r["variation"], []).append(r["history"])
    for name in names:
        runs = by_var.get(name, [])
        if not runs:
            continue
        max_len = max(len(h) for h in runs)
        padded = np.array(
            [h + [h[-1]] * (max_len - len(h)) for h in runs], dtype=float
        )
        mean_hist = padded.mean(axis=0)
        ax.plot(mean_hist, label=name, alpha=0.85)
    ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Mean training-loss curves")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # accuracy vs trainable-phase count
    ax = axes[1, 1]
    xs = [s["n_trainable_phases"] for s in summary]
    ys = [s["acc_mean"] for s in summary]
    ax.scatter(xs, ys, c="seagreen", s=70)
    for s in summary:
        ax.annotate(
            s["variation"],
            (s["n_trainable_phases"], s["acc_mean"]),
            fontsize=8,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Trainable mesh phases",
        ylabel="Test accuracy (mean)",
        title="Capacity vs accuracy",
    )
    ax.grid(True, alpha=0.3)

    fig.suptitle("Two moons configuration sweep", y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep two-moons configs.")
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated training seeds (default: 0,1,2).",
    )
    p.add_argument(
        "--variations",
        type=str,
        default="",
        help="Optional comma-separated subset of variation names. Empty = run all.",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=17,
        help="Fixed sklearn split seed for fair comparison.",
    )
    p.add_argument(
        "--epochs-scale",
        type=float,
        default=1.0,
        help="Scale factor on each variation's epoch budget (use <1 for quick smoke tests).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Preset: 1 seed, 30%% epochs, drop the heaviest variations "
        "(wide_n6, wide_deep). Useful for a ~1.5h first look.",
    )
    return p.parse_args()


QUICK_DROP = {"wide_n6", "wide_deep"}


def main() -> None:
    args = _parse_args()
    if args.quick:
        seeds = [0]
        epochs_scale = 0.3
        selected = [v for v in VARIATIONS if v.name not in QUICK_DROP]
        logger.info(
            "Quick mode: 1 seed, 30%% epochs, dropping %s", sorted(QUICK_DROP)
        )
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("At least one seed required")
        epochs_scale = args.epochs_scale
        selected = VARIATIONS
        if args.variations.strip():
            wanted = {s.strip() for s in args.variations.split(",") if s.strip()}
            selected = [v for v in VARIATIONS if v.name in wanted]
            if not selected:
                raise ValueError(f"No variations matched {wanted}")

    # Experiment requires a full sim config; use the first variation as the
    # representative baseline and tack on sweep metadata.
    base_cfg = _build_config(selected[0], train_seed=seeds[0], data_seed=args.data_seed)
    sweep_config = {
        **base_cfg,
        "sweep_variations": [v.name for v in selected],
        "sweep_seeds": seeds,
        "sweep_epochs_scale": epochs_scale,
    }

    with Experiment("Two Moons Config Sweep", config=sweep_config) as exp:
        rows: list[dict[str, Any]] = []
        for v in selected:
            for s in seeds:
                logger.info("Variation=%s  seed=%d", v.name, s)
                row = _train_eval(
                    v,
                    train_seed=s,
                    data_seed=args.data_seed,
                    epochs_scale=epochs_scale,
                )
                rows.append(row)

        summary = _aggregate(rows)
        _print_ranking(summary, selected)

        results_path = exp.run_dir / "config_sweep_results.json"
        results_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "rows": [
                        {k: v for k, v in r.items() if k != "history"} for r in rows
                    ],
                    "variations": [
                        {
                            "name": v.name,
                            "n_modes": v.n_modes,
                            "n_layers": v.n_layers,
                            "lr": v.lr,
                            "epochs": v.epochs,
                            "data_noise": v.data_noise,
                            "encoding_slots": v.encoding_slots,
                            "note": v.note,
                        }
                        for v in selected
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        exp.artifacts.append(str(results_path.resolve()))

        exp.save_metrics(
            {
                "n_variations": len(selected),
                "n_seeds": len(seeds),
                "best_variation": summary[0]["variation"],
                "best_acc_mean": summary[0]["acc_mean"],
                "best_loss_mean": summary[0]["loss_mean"],
            }
        )

        fig = _plot(rows, summary)
        exp.savefig(fig, "two_moons_config_sweep.png", bbox_inches="tight")
        plt.close(fig)

        logger.info("Report saved to %s", exp.run_dir)


if __name__ == "__main__":
    main()
