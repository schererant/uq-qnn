"""
Hardware abstraction layer for the UQ-QNN framework.

Provides:
- ``NoiseModel`` protocol and concrete implementations (Gaussian, shot, dark count, composite)
- ``TimingParams`` for hardware timing constraints (heater, laser, detector)
- ``HardwareProfile`` bundles noise + timing + backend into a named configuration
- ``HardwareBackend`` protocol and ``RealHardwareBackend`` placeholder for real photonic hardware
- Built-in profiles: IDEAL, LAB_6MODE, NOISY_PROTOTYPE
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np


# ── Noise models ──────────────────────────────────────────────


@runtime_checkable
class NoiseModel(Protocol):
    """Callable that perturbs measurement outcomes to simulate hardware noise.

    Implementations must be frozen dataclasses with a ``__call__`` method
    and a ``to_dict`` method for JSON serialization.
    """

    def __call__(
        self,
        outcomes: np.ndarray,
        working_indices: Tuple[int, ...],
        labels: list[str],
        *,
        rng: np.random.Generator,
    ) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


def _clip_and_renormalize(vals: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and renormalize to sum to 1."""
    vals = np.clip(vals, 0.0, 1.0)
    total = vals.sum()
    if total > 0:
        vals = vals / total
    return vals


@dataclass(frozen=True)
class GaussianNoise:
    """Additive Gaussian noise on measurement outcomes.

    This matches the existing ``apply_noise_to_outcomes`` behavior in
    ``coincidence.py``: Gaussian noise is added to working channels,
    then clipped and renormalized.

    Args:
        std: Standard deviation. Scalar (same for all channels) or
            per-channel tuple matching len(working_indices).
    """

    std: Union[float, Tuple[float, ...]]

    def __call__(
        self,
        outcomes: np.ndarray,
        working_indices: Tuple[int, ...],
        labels: list[str],
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        vals = outcomes.copy()
        if isinstance(self.std, (list, tuple)):
            stds = self.std
        else:
            stds = [self.std] * len(working_indices)
        for k, idx in enumerate(working_indices):
            vals[idx] += rng.normal(0, stds[k])
        return _clip_and_renormalize(vals)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "gaussian", "std": self.std}


@dataclass(frozen=True)
class ShotNoise:
    """Poisson-distributed shot noise scaled by sample count.

    Treats outcomes as probabilities, draws multinomial samples, and
    converts back to normalized probabilities.

    Args:
        n_samples: Number of photon detection events to simulate.
    """

    n_samples: int

    def __call__(
        self,
        outcomes: np.ndarray,
        working_indices: Tuple[int, ...],
        labels: list[str],
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        vals = outcomes.copy()
        probs = np.zeros_like(vals)
        probs[list(working_indices)] = vals[list(working_indices)]
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs[list(working_indices)] = 1.0 / len(working_indices)
        counts = rng.multinomial(self.n_samples, probs)
        total_counts = counts.sum()
        if total_counts > 0:
            vals = counts / total_counts
        return vals.astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "shot", "n_samples": self.n_samples}


@dataclass(frozen=True)
class DarkCountNoise:
    """Constant dark-count rate added per detector channel.

    Adds a uniform baseline to each working detector, then renormalizes.

    Args:
        rate_per_detector: Dark count probability added to each working channel.
    """

    rate_per_detector: float

    def __call__(
        self,
        outcomes: np.ndarray,
        working_indices: Tuple[int, ...],
        labels: list[str],
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        vals = outcomes.copy()
        for idx in working_indices:
            vals[idx] += self.rate_per_detector
        return _clip_and_renormalize(vals)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "dark_count", "rate_per_detector": self.rate_per_detector}


@dataclass(frozen=True)
class CompositeNoise:
    """Applies multiple noise models in sequence.

    Args:
        models: Tuple of noise models applied in order (first to last).
    """

    models: Tuple[Any, ...]  # Tuple[NoiseModel, ...] but Any avoids Protocol in generic

    def __call__(
        self,
        outcomes: np.ndarray,
        working_indices: Tuple[int, ...],
        labels: list[str],
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        vals = outcomes.copy()
        for model in self.models:
            vals = model(vals, working_indices, labels, rng=rng)
        return vals

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "composite",
            "models": [m.to_dict() for m in self.models],
        }


# ── Timing parameters ────────────────────────────────────────


@dataclass(frozen=True)
class TimingParams:
    """Hardware timing constraints for continuous-swipe mode.

    These translate physical hardware limits (heater settle time, laser
    repetition rate, detector window) into a safe swipe count.

    Args:
        t_phase_ms: Heater settle time in milliseconds.
        f_laser_khz: Laser repetition rate in kHz.
        det_window_us: Detector integration window in microseconds.
        max_swipe: Maximum allowed swipe count (caps memory footprint).
    """

    t_phase_ms: float
    f_laser_khz: float
    det_window_us: float
    max_swipe: int

    def compute_n_swipe(self) -> int:
        """Compute the number of phase sweep points from hardware timing.

        Delegates to ``data.compute_n_swipe`` with this instance's values.
        """
        from .data import compute_n_swipe

        return compute_n_swipe(
            self.t_phase_ms,
            self.f_laser_khz,
            self.det_window_us,
            self.max_swipe,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_phase_ms": self.t_phase_ms,
            "f_laser_khz": self.f_laser_khz,
            "det_window_us": self.det_window_us,
            "max_swipe": self.max_swipe,
        }


# ── Hardware profile ──────────────────────────────────────────


@dataclass(frozen=True)
class HardwareProfile:
    """Named hardware configuration bundling backend, noise, and timing.

    A profile can be passed to ``Experiment`` to configure the simulation
    environment. Explicit keys in the experiment CONFIG dict always take
    precedence over profile defaults.

    Args:
        name: Human-readable profile name (e.g. "ideal", "lab_6mode").
        backend: Simulation backend ("numpy", "perceval", or a custom name).
        noise: Noise model to apply to measurement outcomes, or None for ideal.
        timing: Hardware timing parameters, or None for no timing constraints.
        coincidence_window_ns: Coincidence window in nanoseconds (for accidental correction).
        coincidence_window_factor: Hardware-dependent window factor in seconds.
    """

    name: str
    backend: str
    noise: Optional[Any]  # Optional[NoiseModel]
    timing: Optional[TimingParams]
    coincidence_window_ns: Optional[float]
    coincidence_window_factor: Optional[float]

    def apply_to_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Merge profile values into an experiment config dict.

        Explicit keys already present in ``cfg`` take precedence over
        profile defaults. Returns a new dict (does not mutate ``cfg``).
        """
        merged = cfg.copy()

        # Backend — only set if not explicitly specified
        if "sim_backend" not in merged and "backend" not in merged:
            merged["sim_backend"] = self.backend

        return merged

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "backend": self.backend,
            "noise": self.noise.to_dict() if self.noise is not None else None,
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "coincidence_window_ns": self.coincidence_window_ns,
            "coincidence_window_factor": self.coincidence_window_factor,
        }
        return result


# ── Hardware backend protocol ─────────────────────────────────


@runtime_checkable
class HardwareBackend(Protocol):
    """Interface for custom simulation or real-hardware backends.

    Implement this protocol to connect to actual photonic hardware
    or alternative simulators.
    """

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def run_circuit(
        self,
        params: np.ndarray,
        encoded_phases: np.ndarray,
        cfg: Any,  # SimConfig — use Any to avoid circular import
        *,
        return_class_probs: bool = False,
    ) -> np.ndarray: ...


class RealHardwareBackend:
    """Placeholder backend for connecting to actual photonic hardware.

    Subclass this and implement ``run_circuit()`` to interface with
    a real photonic chip. The ``is_available()`` method should return
    True when the hardware connection is established.
    """

    @property
    def name(self) -> str:
        return "real_hardware"

    def is_available(self) -> bool:
        return False

    def run_circuit(
        self,
        params: np.ndarray,
        encoded_phases: np.ndarray,
        cfg: Any,
        *,
        return_class_probs: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Real hardware backend is not yet implemented. "
            "Subclass RealHardwareBackend and implement run_circuit() "
            "to connect to your photonic hardware."
        )


# ── Backend registry ──────────────────────────────────────────

_BACKENDS: dict[str, HardwareBackend] = {}


def register_backend(name: str, backend: HardwareBackend) -> None:
    """Register a custom hardware backend by name.

    Once registered, the backend name can be used as the ``backend``
    field in a ``HardwareProfile`` or ``SimConfig``.
    """
    _BACKENDS[name] = backend


def get_backend(name: str) -> HardwareBackend:
    """Look up a registered custom backend by name.

    Raises:
        KeyError: If the backend name is not registered.
    """
    if name not in _BACKENDS:
        raise KeyError(
            f"Unknown hardware backend {name!r}. "
            f"Registered backends: {sorted(_BACKENDS)}. "
            f"Use register_backend() to add custom backends."
        )
    return _BACKENDS[name]


# ── Built-in profiles ─────────────────────────────────────────

IDEAL = HardwareProfile(
    name="ideal",
    backend="numpy",
    noise=None,
    timing=None,
    coincidence_window_ns=None,
    coincidence_window_factor=None,
)

LAB_6MODE = HardwareProfile(
    name="lab_6mode",
    backend="numpy",
    noise=GaussianNoise(std=0.02),
    timing=TimingParams(
        t_phase_ms=10.0,
        f_laser_khz=50.0,
        det_window_us=10.0,
        max_swipe=21,
    ),
    coincidence_window_ns=170.0,
    coincidence_window_factor=27e-12,
)

NOISY_PROTOTYPE = HardwareProfile(
    name="noisy_prototype",
    backend="numpy",
    noise=CompositeNoise(
        models=(
            GaussianNoise(std=0.05),
            DarkCountNoise(rate_per_detector=1e-4),
        )
    ),
    timing=TimingParams(
        t_phase_ms=5.0,
        f_laser_khz=100.0,
        det_window_us=5.0,
        max_swipe=41,
    ),
    coincidence_window_ns=170.0,
    coincidence_window_factor=27e-12,
)


PROFILES: dict[str, HardwareProfile] = {
    "ideal": IDEAL,
    "lab_6mode": LAB_6MODE,
    "noisy_prototype": NOISY_PROTOTYPE,
}


def get_profile(name: str) -> HardwareProfile:
    """Look up a built-in hardware profile by name.

    Raises:
        KeyError: If the profile name is not found.
    """
    if name not in PROFILES:
        raise KeyError(
            f"Unknown hardware profile {name!r}. "
            f"Available: {sorted(PROFILES)}"
        )
    return PROFILES[name]
