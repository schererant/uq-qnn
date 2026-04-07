#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark the fast NumPy memristive backend against the legacy reference loop."""

import os
import sys
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.circuits import (
    normalize_memristive_output_modes,
    normalize_memristive_phase_idx,
)
from src.config import SimConfig
from src.experiment import Experiment
from src.logging_config import get_logger
from src.numpy_backend import singles_prob_from_unitary, unitary_for_point
from src.simulation import MemristiveState, run_simulation_sequence_np

logger = get_logger(__name__)

BASE_CONFIG: Dict = {
    "n_modes": 6,
    "input_state": (0,),
    "encoding_phase_idx": 0,
    "photon_distinguishability": None,
    "target_mode": (5,),
    "memristive_phase_idx": (6,),
    "memristive_output_modes": None,
    "output_mode": "singles",
    "working_detectors": None,
    "noise_std": None,
    "n_samples": 300,
    "memory_depth": 3,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "sim_backend": "numpy",
    "loss_type": "mse",
    "n_classes": 1,
    "lr": 0.01,
    "epochs": 1,
    "seed": 42,
}

CASES = [
    {
        "name": "discrete",
        "n_points": 128,
        "n_swipe": 0,
        "swipe_span": 0.0,
        "memory_depth": 3,
        "encoding_phase_idx": 0,
        "repeats": 8,
    },
    {
        "name": "swipe",
        "n_points": 96,
        "n_swipe": 7,
        "swipe_span": float(np.pi / 18),
        "memory_depth": 3,
        "encoding_phase_idx": 0,
        "repeats": 6,
    },
    {
        "name": "swipe_inline",
        "n_points": 96,
        "n_swipe": 7,
        "swipe_span": float(np.pi / 18),
        "memory_depth": 3,
        "encoding_phase_idx": 7,
        "repeats": 6,
    },
]


def _sim_cfg(**overrides: object) -> SimConfig:
    cfg = BASE_CONFIG.copy()
    cfg.update(overrides)
    return SimConfig.from_experiment_config(cfg)


def _legacy_memristive_reference(
    params: np.ndarray,
    encoded_phases: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    memristive_indices = normalize_memristive_phase_idx(
        cfg.memristive_phase_idx, cfg.n_modes, n_phases
    )
    output_modes = normalize_memristive_output_modes(
        cfg.memristive_output_modes, memristive_indices, cfg.n_modes
    )
    weights = params[-len(memristive_indices) :]
    target_mode = cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)

    mem_state = MemristiveState(
        n_indices=len(memristive_indices),
        memory_depth=cfg.memory_depth,
        output_modes=output_modes,
        singles_input_mode=cfg.singles_input_mode,
    )

    preds = np.zeros(len(encoded_phases), dtype=float)
    if cfg.n_swipe > 0:
        offsets = np.linspace(
            -cfg.swipe_span / 2,
            cfg.swipe_span / 2,
            cfg.n_swipe,
            dtype=np.float64,
        )
    else:
        offsets = np.array([0.0], dtype=np.float64)

    for i, enc_phi in enumerate(encoded_phases):
        mem_phis = mem_state.current_phases(weights, i)
        p1_swipe = np.empty((len(offsets), len(memristive_indices)), dtype=float)
        p2_swipe = np.empty((len(offsets), len(memristive_indices)), dtype=float)
        target_swipe = np.empty(len(offsets), dtype=float)

        for k, off in enumerate(offsets):
            phases_loc = params[: -len(memristive_indices)].copy()
            for j, idx in enumerate(memristive_indices):
                phases_loc[idx] = mem_phis[j]
            u = unitary_for_point(
                phases_loc,
                float(enc_phi + off),
                cfg.n_modes,
                cfg.encoding_phase_idx,
            )
            for j, (m1, m2) in enumerate(output_modes):
                p1_swipe[k, j] = float(np.abs(u[m1, cfg.singles_input_mode]) ** 2)
                p2_swipe[k, j] = float(np.abs(u[m2, cfg.singles_input_mode]) ** 2)
            target_swipe[k] = singles_prob_from_unitary(
                u, cfg.singles_input_mode, target_mode
            )

        mem_state.update_from_prob_arrays(i, p1_swipe.mean(axis=0), p2_swipe.mean(axis=0))
        preds[i] = float(target_swipe.mean())

    return preds


def _time_call(fn, *args, repeats: int) -> np.ndarray:
    samples = np.zeros(repeats, dtype=float)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        samples[i] = time.perf_counter() - t0
    return samples


def _benchmark_case(case: Dict[str, object]) -> Dict[str, float]:
    cfg = _sim_cfg(
        n_swipe=int(case["n_swipe"]),
        swipe_span=float(case["swipe_span"]),
        memory_depth=int(case["memory_depth"]),
        encoding_phase_idx=case["encoding_phase_idx"],
    )
    rng = np.random.default_rng(1000 + int(case["n_points"]))
    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    params = np.concatenate(
        [rng.uniform(0.0, 2 * np.pi, size=n_phases), np.array([0.43], dtype=float)]
    )
    encoded = np.linspace(0.12, 1.24, int(case["n_points"]), dtype=float)

    fast_preds = run_simulation_sequence_np(params, encoded, cfg)
    legacy_preds = _legacy_memristive_reference(params, encoded, cfg)
    np.testing.assert_allclose(fast_preds, legacy_preds, rtol=0.0, atol=1e-12)

    repeats = int(case["repeats"])
    fast_times = _time_call(run_simulation_sequence_np, params, encoded, cfg, repeats=repeats)
    legacy_times = _time_call(_legacy_memristive_reference, params, encoded, cfg, repeats=repeats)

    return {
        "n_points": float(len(encoded)),
        "n_swipe": float(cfg.n_swipe),
        "fast_mean_ms": float(fast_times.mean() * 1000.0),
        "fast_std_ms": float(fast_times.std(ddof=0) * 1000.0),
        "legacy_mean_ms": float(legacy_times.mean() * 1000.0),
        "legacy_std_ms": float(legacy_times.std(ddof=0) * 1000.0),
        "speedup": float(legacy_times.mean() / fast_times.mean()),
    }


def _plot_results(results: Dict[str, Dict[str, float]]) -> plt.Figure:
    labels = list(results.keys())
    fast_means = [results[k]["fast_mean_ms"] for k in labels]
    legacy_means = [results[k]["legacy_mean_ms"] for k in labels]
    fast_std = [results[k]["fast_std_ms"] for k in labels]
    legacy_std = [results[k]["legacy_std_ms"] for k in labels]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, legacy_means, width, yerr=legacy_std, label="Legacy reference", color="#9aa0a6")
    ax.bar(x + width / 2, fast_means, width, yerr=fast_std, label="Fast NumPy path", color="#1f77b4")

    for i, key in enumerate(labels):
        ax.text(
            x[i],
            max(legacy_means[i], fast_means[i]) * 1.03,
            f"x{results[key]['speedup']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime [ms]")
    ax.set_title("Memristive NumPy Backend Benchmark")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    with Experiment("Memristive Backend Benchmark", config=BASE_CONFIG) as exp:
        results: Dict[str, Dict[str, float]] = {}
        for case in CASES:
            name = str(case["name"])
            logger.info("Benchmarking case: %s", name)
            results[name] = _benchmark_case(case)

        exp.save_metrics({"benchmark": results})

        print("\nMemristive backend benchmark")
        print("=" * 72)
        print(
            f"{'Case':15s} {'Pts':>6s} {'Swipe':>6s} {'Fast [ms]':>12s} "
            f"{'Legacy [ms]':>14s} {'Speedup':>9s}"
        )
        print("-" * 72)
        for name, metrics in results.items():
            print(
                f"{name:15s} {int(metrics['n_points']):6d} {int(metrics['n_swipe']):6d} "
                f"{metrics['fast_mean_ms']:12.3f} {metrics['legacy_mean_ms']:14.3f} "
                f"{metrics['speedup']:9.2f}x"
            )

        fig = _plot_results(results)
        exp.savefig(fig, "memristive_backend_benchmark.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved benchmark figure to %s", exp.run_dir)


if __name__ == "__main__":
    main()
