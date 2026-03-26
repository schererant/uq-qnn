# -*- coding: utf-8 -*-
"""
Standardized Experiment class for the UQ-QNN framework.

Provides a context manager that handles:
- Timestamped run directories under reports/
- Full-fidelity run.log via the package logging configuration (no stdout tee)
- Config validation and serialization
- Training, prediction, and uncertainty analysis via config
- Artifact tracking and run_summary.json

No default config values — every parameter must be set explicitly
in the experiment script's CONFIG dict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.simulation import sim_logger
from src.logging_config import get_logger, add_file_handler, remove_file_handler

logger = get_logger(__name__)

_REQUIRED_CONFIG_KEYS = frozenset(
    {
        "n_modes",
        "memory_depth",
        "lr",
        "epochs",
        "n_samples",
        "n_swipe",
        "swipe_span",
        "encoding_mode",
        "target_mode",
        "n_photons",
        "memristive_phase_idx",
        "memristive_output_modes",
        "output_mode",
        "loss_type",
        "n_classes",
        "sim_backend",
        "seed",
    }
)


class Experiment:
    """Context manager for a single UQ-QNN experiment run.

    Args:
        name: Experiment name (used for report subdirectory).
        config: Complete experiment configuration dict.  All circuit,
            training, and task parameters must be specified explicitly —
            there are no hidden defaults.
    """

    def __init__(self, name: str, *, config: dict[str, Any]):
        self.name = name
        self.config = config.copy()

        missing = _REQUIRED_CONFIG_KEYS - self.config.keys()
        if missing:
            raise ValueError(f"Missing required config keys: {sorted(missing)}")

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.project_root = Path(__file__).resolve().parent.parent
        slug = name.lower().replace(" ", "_")
        self.run_dir = self.project_root / "reports" / slug / self.timestamp

        self.metrics: dict[str, Any] = {}
        self.artifacts: list[str] = []
        self._file_handler: Any = None

    # ── lifecycle ──────────────────────────────────────────────

    def __enter__(self) -> Experiment:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.run_dir / "run.log"
        self._file_handler = add_file_handler(log_path)
        logger.info("=== Experiment: %s ===", self.name)
        logger.info("Run directory: %s", self.run_dir.resolve())
        logger.info("Timestamp: %s", self.timestamp)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Experiment failed with error: %s", exc_val)
            self.metrics["status"] = "failed"
            self.metrics["error"] = str(exc_val)
        else:
            self.metrics["status"] = "completed"

        sim_stats = sim_logger.stats_dict()
        if sim_stats:
            sim_logger.report()

        self._write_run_summary(simulation_stats=sim_stats)
        logger.info("Experiment %s finished.", self.name)

        if self._file_handler:
            remove_file_handler(self._file_handler)
            self._file_handler = None

    # ── training ───────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        encoded: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        """Train the photonic model using parameters from self.config.

        Args:
            X: Input data (raw values in [0,1]) or pre-encoded phases.
            y: Target values / labels.
            encoded: If True, X contains pre-encoded phases and is passed
                directly to train_pytorch_generic.  Otherwise the standard
                ``2 * arccos(X)`` encoding is applied first.

        Returns:
            ``(theta_opt, loss_history)`` tuple.
        """
        from src.training import train_pytorch_generic

        if not encoded:
            X = 2 * np.arccos(X)

        return train_pytorch_generic(X, y, **self._training_kwargs())

    def _training_kwargs(self) -> dict[str, Any]:
        c = self.config
        return {
            "memory_depth": c["memory_depth"],
            "lr": c["lr"],
            "epochs": c["epochs"],
            "n_samples": c["n_samples"],
            "n_swipe": c["n_swipe"],
            "swipe_span": c["swipe_span"],
            "n_modes": c["n_modes"],
            "encoding_mode": c["encoding_mode"],
            "n_photons": c["n_photons"],
            "target_mode": c["target_mode"],
            "memristive_phase_idx": c["memristive_phase_idx"],
            "memristive_output_modes": c["memristive_output_modes"],
            "encoding_phase_idx": c.get("encoding_phase_idx"),
            "output_mode": c["output_mode"],
            "input_modes": c.get("input_modes"),
            "working_detectors": c.get("working_detectors"),
            "noise_std": c.get("noise_std"),
            "loss_type": c["loss_type"],
            "n_classes": c["n_classes"],
            "seed": c["seed"],
            "backend": c["sim_backend"],
        }

    # ── prediction ─────────────────────────────────────────────

    def predict(
        self,
        theta: np.ndarray,
        encoded_phases: np.ndarray,
        *,
        return_class_probs: bool = False,
    ) -> np.ndarray:
        """Run a single forward pass through the trained circuit.

        Args:
            theta: Optimized parameter vector.
            encoded_phases: Phase-encoded input data.
            return_class_probs: If True, return per-class probability vectors.
        """
        from src.simulation import run_simulation_sequence_np

        c = self.config
        return run_simulation_sequence_np(
            theta,
            c["memory_depth"],
            c["n_samples"],
            encoded_phases=encoded_phases,
            n_swipe=c["n_swipe"],
            swipe_span=c["swipe_span"],
            n_modes=c["n_modes"],
            encoding_mode=c["encoding_mode"],
            target_mode=c["target_mode"],
            memristive_phase_idx=c["memristive_phase_idx"],
            memristive_output_modes=c["memristive_output_modes"],
            encoding_phase_idx=c.get("encoding_phase_idx"),
            output_mode=c["output_mode"],
            input_modes=c.get("input_modes"),
            working_detectors=c.get("working_detectors"),
            noise_std=c.get("noise_std"),
            backend=c["sim_backend"],
            return_class_probs=return_class_probs,
        )

    # ── uncertainty ────────────────────────────────────────────

    def run_uncertainty_analysis(
        self,
        theta: np.ndarray,
        encoded_phases: np.ndarray,
        *,
        n_passes: int,
        noise_std: float,
        return_class_probs: bool = False,
    ) -> dict[str, np.ndarray]:
        """Multi-pass uncertainty estimation via parameter perturbation.

        Args:
            theta: Optimized parameter vector.
            encoded_phases: Phase-encoded test inputs.
            n_passes: Number of noisy forward passes.
            noise_std: Std-dev of Gaussian noise added to phase parameters
                on each pass.
            return_class_probs: If True, collect per-class probabilities
                (inferred automatically when loss_type is cross_entropy).

        Returns:
            Dict with ``'mean'``, ``'std'``, and ``'all_preds'`` arrays.
        """
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        from src.simulation import uncertainty_forward_pass

        c = self.config
        n_classes = c["n_classes"]

        is_classification = c["loss_type"] == "cross_entropy" or return_class_probs

        logger.info("Estimating uncertainty with %d forward passes…", n_passes)

        if is_classification and n_classes > 1:
            all_preds = np.zeros((len(encoded_phases), n_classes, n_passes))
        else:
            all_preds = np.zeros((len(encoded_phases), n_passes))

        unc_cfg = {
            "memory_depth": c["memory_depth"],
            "n_swipe": 0,
            "swipe_span": 0.0,
            "n_modes": c["n_modes"],
            "encoding_mode": c["encoding_mode"],
            "target_mode": c["target_mode"],
            "memristive_phase_idx": c["memristive_phase_idx"],
            "memristive_output_modes": c["memristive_output_modes"],
            "encoding_phase_idx": c.get("encoding_phase_idx"),
            "output_mode": c["output_mode"],
            "input_modes": c.get("input_modes"),
            "working_detectors": c.get("working_detectors"),
            "noise_std": None,
            "backend": c["sim_backend"],
            "return_class_probs": is_classification,
        }

        rng = np.random.default_rng(c["seed"])
        n_samples_base = c["n_samples"]
        n_memristive = (
            len(c["memristive_phase_idx"]) if c["memristive_phase_idx"] else 0
        )

        jobs = []
        for _ in range(n_passes):
            sample_count = max(100, n_samples_base + int(rng.integers(-100, 101)))
            perturbed = theta.copy()
            if n_memristive > 0:
                perturbed[:-n_memristive] += rng.normal(
                    0, noise_std, size=len(perturbed) - n_memristive
                )
            else:
                perturbed += rng.normal(0, noise_std, size=len(perturbed))
            jobs.append((perturbed, sample_count, encoded_phases, unc_cfg))

        max_workers = min(n_passes, os.cpu_count() or 2)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(
                tqdm(
                    pool.map(uncertainty_forward_pass, jobs),
                    total=n_passes,
                    desc="UQ passes",
                )
            )

        for i, preds in enumerate(results):
            if is_classification and n_classes > 1:
                all_preds[:, :, i] = preds
            else:
                all_preds[:, i] = preds

        return {
            "mean": np.mean(all_preds, axis=-1),
            "std": np.std(all_preds, axis=-1),
            "all_preds": all_preds,
        }

    # ── artifacts ──────────────────────────────────────────────

    def save_metrics(self, metrics: dict[str, Any]):
        """Merge additional metrics into the run summary."""
        self.metrics.update(metrics)

    def savefig(self, fig, name: str, *, dpi: int = 300, **kwargs) -> Path:
        """Save a matplotlib figure to run_dir and track it as an artifact."""
        path = self.run_dir / name
        fig.savefig(path, dpi=dpi, **kwargs)
        self.artifacts.append(str(path))
        return path

    # ── internals ──────────────────────────────────────────────

    def _get_git_sha(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def _json_safe(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(x) for x in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _write_run_summary(self, simulation_stats: Optional[dict] = None):
        summary = {
            "schema": "uq-qnn.experiment_run.v1",
            "name": self.name,
            "run_dir": str(self.run_dir.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "git_commit": self._get_git_sha(),
            "config": self._json_safe(self.config),
            "metrics": self._json_safe(self.metrics),
            "artifacts": self.artifacts,
        }
        if simulation_stats:
            summary["simulation"] = self._json_safe(simulation_stats)

        out_path = self.run_dir / "run_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
