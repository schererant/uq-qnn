"""
Simulation configuration dataclass for the UQ-QNN framework.

``SimConfig`` is the single object that flows through the entire
simulation → training → autograd stack, replacing ~18 loose keyword
arguments.

IMPORTANT: No field has a default value.  Every parameter must be
set explicitly by the caller.  This mirrors the project's "no hidden
defaults" policy enforced by _REQUIRED_CONFIG_KEYS in experiment.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace as _dc_replace
from typing import Any, Optional, Tuple, Union


def _seq_as_tuples(v: Any) -> Any:
    """Recursively convert lists to tuples for frozen-config round-trips."""
    if isinstance(v, list):
        return tuple(_seq_as_tuples(x) for x in v)
    return v


@dataclass(frozen=True)
class SimConfig:
    """Immutable bundle of circuit / simulation / task parameters.

    Every field that previously appeared as a keyword argument in
    ``run_simulation_sequence_np``, ``train_pytorch_generic``,
    ``PhotonicModel``, and ``MemristorLossPSR`` is collected here.

    The class is *frozen* so it can be safely shared across threads,
    stored on ``torch.autograd`` ctx objects, and serialised to JSON.

    **No field has a default value.**  All must be supplied explicitly.
    """

    # ── circuit geometry ───────────────────────────────────────
    n_modes: int
    encoding_mode: int
    target_mode: Optional[Tuple[int, ...]]
    memristive_phase_idx: Optional[Union[int, Tuple[int, ...]]]
    memristive_output_modes: Optional[Tuple[Tuple[int, int], ...]]
    encoding_phase_idx: Optional[int]

    # ── measurement mode ───────────────────────────────────────
    output_mode: str  # "singles" | "coincidence"
    input_modes: Optional[Tuple[int, ...]]  # for coincidence
    working_detectors: Optional[Tuple[int, ...]]  # for coincidence
    noise_std: Optional[Union[float, Tuple[float, ...]]]

    # ── simulation ─────────────────────────────────────────────
    n_samples: int
    memory_depth: int
    n_swipe: int
    swipe_span: float
    n_photons: Optional[Tuple[int, ...]]
    backend: str  # "numpy" | "perceval"

    # ── loss / task ────────────────────────────────────────────
    loss_type: str  # "mse" | "cross_entropy"
    n_classes: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (for run_summary, uncertainty_forward_pass, etc.)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimConfig:
        """Construct from a plain dict, ignoring unknown keys.

        Converts mutable sequences (lists) to tuples so the result is
        hashable / frozen-safe (including nested structures from
        :func:`dataclasses.asdict`).

        Raises TypeError if any required field is missing — there are
        no defaults to fall back on.
        """
        valid = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {}
        for k, v in d.items():
            if k not in valid:
                continue
            filtered[k] = _seq_as_tuples(v) if isinstance(v, list) else v
        return cls(**filtered)

    @classmethod
    def from_experiment_config(cls, cfg: dict[str, Any]) -> SimConfig:
        """Build from an Experiment CONFIG dict.

        Handles the key rename ``sim_backend`` → ``backend`` and drops
        experiment-only keys (lr, epochs, seed, n_data, sigma_noise,
        unc_n_passes, unc_noise_std, …).
        """

        backend = cfg.get("sim_backend")
        if backend is None:
            backend = cfg.get("backend")
        if backend is None:
            raise KeyError("experiment config must include 'sim_backend' or 'backend'")

        mom_raw = cfg.get("memristive_output_modes")
        mom: Optional[Tuple[Tuple[int, int], ...]]
        if mom_raw is None:
            mom = None
        else:
            mom = tuple(tuple(int(a) for a in p) for p in mom_raw)

        tm_raw = cfg.get("target_mode")
        if tm_raw is None:
            tm = None
        elif isinstance(tm_raw, list):
            tm = tuple(int(x) for x in tm_raw)
        else:
            tm = tuple(int(x) for x in tm_raw)

        def as_int_tuple_opt(key: str) -> Optional[Tuple[int, ...]]:
            v = cfg.get(key)
            if v is None:
                return None
            if isinstance(v, list):
                return tuple(int(x) for x in v)
            return tuple(int(x) for x in v)

        np_raw = cfg.get("n_photons")
        n_photons: Optional[Tuple[int, ...]]
        if np_raw is None:
            n_photons = None
        elif isinstance(np_raw, list):
            n_photons = tuple(int(x) for x in np_raw)
        else:
            n_photons = tuple(int(x) for x in np_raw)

        return cls(
            n_modes=int(cfg["n_modes"]),
            encoding_mode=int(cfg["encoding_mode"]),
            target_mode=tm,
            memristive_phase_idx=cfg.get("memristive_phase_idx"),
            memristive_output_modes=mom,
            encoding_phase_idx=cfg.get("encoding_phase_idx"),
            output_mode=str(cfg["output_mode"]),
            input_modes=as_int_tuple_opt("input_modes"),
            working_detectors=as_int_tuple_opt("working_detectors"),
            noise_std=cfg.get("noise_std"),
            n_samples=int(cfg["n_samples"]),
            memory_depth=int(cfg["memory_depth"]),
            n_swipe=int(cfg["n_swipe"]),
            swipe_span=float(cfg["swipe_span"]),
            n_photons=n_photons,
            backend=str(backend),
            loss_type=str(cfg["loss_type"]),
            n_classes=int(cfg["n_classes"]),
        )

    def replace(self, **overrides: Any) -> SimConfig:
        """Return a new SimConfig with selected fields replaced.

        Useful for uncertainty analysis where n_swipe/noise_std differ::

            unc_cfg = sim_cfg.replace(n_swipe=0, swipe_span=0.0, noise_std=None)
        """
        return _dc_replace(self, **overrides)
