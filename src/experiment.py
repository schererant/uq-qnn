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
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from src.circuit import PhotonicCircuit
from src.coincidence import (
    apply_noise_to_outcomes,
    get_cc_labels,
    postselect_measurement,
    working_detectors_to_cc_indices,
)
from src.config import SimConfig
from src.hardware import HardwareProfile, get_profile
from src.logging_config import add_file_handler, get_logger, remove_file_handler
from src.simulation import sim_logger

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
        hardware: Optional hardware profile. Can be a ``HardwareProfile``
            instance, a profile name string (looked up via ``get_profile``),
            or ``None`` (no hardware profile applied).  When provided, the
            profile's backend is merged into the config (explicit config
            keys take precedence), and the profile's noise model is applied
            to ``predict()`` and ``run_uncertainty_analysis()`` outputs.
    """

    def __init__(
        self,
        name: str,
        *,
        config: dict[str, Any],
        hardware: Optional[Union[HardwareProfile, str]] = None,
    ):
        self.name = name
        self.config = config.copy()

        # Resolve hardware profile
        if isinstance(hardware, str):
            self.hardware: Optional[HardwareProfile] = get_profile(hardware)
        else:
            self.hardware = hardware

        # Apply hardware profile to config (explicit keys take precedence)
        if self.hardware is not None:
            self.config = self.hardware.apply_to_config(self.config)
            # If profile has noise, disable SimConfig-level noise_std to avoid double noise
            if (
                self.hardware.noise is not None
                and self.config.get("noise_std") is not None
            ):
                logger.warning(
                    "HardwareProfile %r has a noise model and config has noise_std set. "
                    "Profile noise takes precedence; setting noise_std=None.",
                    self.hardware.name,
                )
                self.config["noise_std"] = None

        missing = _REQUIRED_CONFIG_KEYS - self.config.keys()
        if missing:
            raise ValueError(f"Missing required config keys: {sorted(missing)}")

        self.sim_cfg = SimConfig.from_experiment_config(self.config)

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
        if self.hardware is not None:
            logger.info("Hardware profile: %s", self.hardware.name)
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
            "sim_cfg": self.sim_cfg,
            "lr": c["lr"],
            "epochs": c["epochs"],
            "seed": c["seed"],
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

        If a hardware profile with a noise model is set, noise is applied
        to the raw simulation outputs.

        Args:
            theta: Optimized parameter vector.
            encoded_phases: Phase-encoded input data.
            return_class_probs: If True, return per-class probability vectors.
        """
        from src.simulation import run_simulation_sequence_np

        if self._can_use_photonic_circuit():
            raw = self._predict_with_photonic_circuit(
                theta,
                encoded_phases,
                return_class_probs=return_class_probs,
            )
        else:
            raw = run_simulation_sequence_np(
                theta,
                encoded_phases,
                self.sim_cfg,
                return_class_probs=return_class_probs,
            )
        if self.hardware is not None and self.hardware.noise is not None:
            raw = self._apply_hardware_noise(raw)
        return raw

    def _apply_hardware_noise(
        self,
        preds: np.ndarray,
        *,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Apply the hardware profile's noise model to prediction outputs.

        For multi-class outputs (2D array), noise is applied row-by-row.
        For scalar outputs (1D array), each value is embedded into a
        2-channel distribution before noise is applied.
        """
        noise = self.hardware.noise
        rng = np.random.default_rng(seed or self.config.get("seed", 42))

        if preds.ndim == 2:
            # Multi-class: each row is a probability vector
            n_channels = preds.shape[1]
            working_indices = tuple(range(n_channels))
            labels = [f"C{i}" for i in range(n_channels)]
            result = np.empty_like(preds)
            for i in range(len(preds)):
                result[i] = noise(preds[i], working_indices, labels, rng=rng)
            return result
        else:
            # Scalar outputs are embedded into a 2-channel distribution so the
            # hardware noise models can perturb them without collapsing them to 1.
            working_indices = (0, 1)
            labels = ["C0", "C1"]
            result = np.empty_like(preds)
            for i in range(len(preds)):
                value = float(np.clip(preds[i], 0.0, 1.0))
                row = np.array([value, 1.0 - value], dtype=float)
                noised = noise(row, working_indices, labels, rng=rng)
                result[i] = noised[0]
            return result

    # ── photonic circuit fast path -------------------------------------------

    def _can_use_photonic_circuit(self) -> bool:
        cfg = self.sim_cfg
        if cfg.backend != "numpy":
            return False
        if cfg.n_swipe != 0:
            return False
        return not self._has_memristive_phases()

    def _has_memristive_phases(self) -> bool:
        mpi = self.sim_cfg.memristive_phase_idx
        if mpi is None:
            return False
        if isinstance(mpi, int):
            return True
        return len(tuple(mpi)) > 0

    def _predict_with_photonic_circuit(
        self,
        theta: np.ndarray,
        encoded_phases: np.ndarray,
        *,
        return_class_probs: bool,
    ) -> np.ndarray:
        circuit = self._build_photonic_circuit(theta)
        if self.sim_cfg.output_mode == "coincidence":
            return self._predict_coincidence_with_circuit(
                circuit,
                encoded_phases,
                return_class_probs=return_class_probs,
            )
        return self._predict_singles_with_circuit(
            circuit,
            encoded_phases,
            return_class_probs=return_class_probs,
        )

    def _build_photonic_circuit(self, theta: np.ndarray) -> PhotonicCircuit:
        cfg = self.sim_cfg
        n_phases = cfg.n_modes * (cfg.n_modes - 1)
        phases = np.asarray(theta[:n_phases], dtype=np.float64)
        return PhotonicCircuit(
            n_modes=cfg.n_modes,
            phases=phases,
            encoding_mode=cfg.encoding_mode,
            encoding_phase_idx=cfg.encoding_phase_idx,
        )

    def _predict_singles_with_circuit(
        self,
        circuit: PhotonicCircuit,
        encoded_phases: np.ndarray,
        *,
        return_class_probs: bool,
    ) -> np.ndarray:
        cfg = self.sim_cfg
        target_mode = (
            cfg.target_mode if cfg.target_mode is not None else (cfg.n_modes - 1,)
        )
        indices = tuple(int(m) for m in target_mode)
        singles = circuit.singles_batch(encoded_phases)
        if return_class_probs and len(indices) > 1:
            return singles[:, indices]
        selected = singles[:, indices]
        if len(indices) > 1:
            return selected.mean(axis=1)
        return selected[:, 0]

    def _predict_coincidence_with_circuit(
        self,
        circuit: PhotonicCircuit,
        encoded_phases: np.ndarray,
        *,
        return_class_probs: bool,
    ) -> np.ndarray:
        cfg = self.sim_cfg
        in_modes = self._coincidence_input_modes()
        working_detectors = (
            tuple(cfg.working_detectors)
            if cfg.working_detectors is not None
            else (0, 1, 5)
        )
        working_cc_indices = working_detectors_to_cc_indices(
            working_detectors, cfg.n_modes
        )
        cc_labels = get_cc_labels(cfg.n_modes)
        add_noise = self._should_add_noise(cfg.noise_std)
        coinc = circuit.coincidences_batch(encoded_phases, input_modes=in_modes)
        n_data = coinc.shape[0]
        n_classes = len(working_cc_indices)
        if return_class_probs and n_classes > 1:
            preds = np.zeros((n_data, n_classes), dtype=float)
        else:
            preds = np.zeros(n_data, dtype=float)
        for i in range(n_data):
            out = postselect_measurement(
                coinc[i],
                working_cc_indices,
                cc_labels,
                fallback_uniform=True,
            )
            if add_noise:
                out = apply_noise_to_outcomes(
                    out,
                    cfg.noise_std,
                    working_cc_indices,
                    cc_labels,
                    seed=i,
                )
            if return_class_probs and n_classes > 1:
                preds[i, :] = out[list(working_cc_indices)]
            else:
                preds[i] = out[working_cc_indices[0]] if working_cc_indices else 0.0
        return preds

    def _coincidence_input_modes(self) -> Tuple[int, int]:
        cfg = self.sim_cfg
        if cfg.input_modes is not None:
            in_modes = tuple(int(m) for m in cfg.input_modes)
        elif cfg.n_modes >= 6:
            in_modes = (1, 4)
        else:
            in_modes = (0, 1)
        if len(in_modes) != 2:
            raise ValueError(
                f"Coincidence mode requires exactly 2 input modes, got {in_modes}"
            )
        for m in in_modes:
            if m < 0 or m >= cfg.n_modes:
                raise ValueError(
                    f"input_modes {in_modes} out of range [0, {cfg.n_modes - 1}]"
                )
        return in_modes

    def _should_add_noise(
        self,
        noise_std: Optional[Union[float, Sequence[float]]],
    ) -> bool:
        if noise_std is None:
            return False
        if isinstance(noise_std, (int, float)):
            return float(noise_std) > 0
        try:
            return any(float(x) > 0 for x in noise_std)  # type: ignore[arg-type]
        except TypeError:
            return False

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

        if n_passes <= 0:
            raise ValueError("n_passes must be a positive integer")

        logger.info("Estimating uncertainty with %d forward passes…", n_passes)

        if is_classification and n_classes > 1:
            all_preds = np.zeros((len(encoded_phases), n_classes, n_passes))
        else:
            all_preds = np.zeros((len(encoded_phases), n_passes))

        unc_sim_cfg = self.sim_cfg.replace(n_swipe=0, swipe_span=0.0, noise_std=None)

        rng = np.random.default_rng(c["seed"])
        n_samples_base = c["n_samples"]
        memristive_phase_idx = c["memristive_phase_idx"]
        if memristive_phase_idx is None:
            n_memristive = 0
        elif isinstance(memristive_phase_idx, int):
            n_memristive = 1
        else:
            n_memristive = len(memristive_phase_idx)

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
            jobs.append(
                (
                    perturbed,
                    sample_count,
                    encoded_phases,
                    unc_sim_cfg.to_dict(),
                    is_classification,
                )
            )

        def _run_sequential():
            return [
                uncertainty_forward_pass(job)
                for job in tqdm(jobs, total=n_passes, desc="UQ passes")
            ]

        results: list[np.ndarray]
        if n_passes == 1:
            results = _run_sequential()
        else:
            max_workers = min(n_passes, os.cpu_count() or 2)
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    results = list(
                        tqdm(
                            pool.map(uncertainty_forward_pass, jobs),
                            total=n_passes,
                            desc="UQ passes",
                        )
                    )
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "ProcessPool unavailable (%s); falling back to sequential UQ passes",
                    exc,
                )
                results = _run_sequential()

        has_hw_noise = self.hardware is not None and self.hardware.noise is not None

        for i, preds in enumerate(results):
            if has_hw_noise:
                preds = self._apply_hardware_noise(
                    np.atleast_2d(preds)
                    if (is_classification and n_classes > 1)
                    else preds,
                    seed=(c["seed"] + i + 1),
                )
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
            "hardware": self.hardware.to_dict() if self.hardware is not None else None,
            "metrics": self._json_safe(self.metrics),
            "artifacts": self.artifacts,
        }
        if simulation_stats:
            summary["simulation"] = self._json_safe(simulation_stats)

        out_path = self.run_dir / "run_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
