"""
Simulation configuration dataclass for the UQ-QNN framework.

``SimConfig`` is the single object that flows through the entire
simulation → training → autograd stack.

IMPORTANT: No field has a default value.  Every parameter must be
set explicitly by the caller.  This mirrors the project's "no hidden
defaults" policy enforced by _REQUIRED_CONFIG_KEYS in experiment.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from dataclasses import replace as _dc_replace
from typing import Any, Literal, Optional, Tuple, Union

PhotonDistinguishability = Literal["indistinguishable", "distinguishable"]


def validate_sim_config(cfg: "SimConfig") -> None:
    """Validate physics and readout semantics. Call before simulation/training."""

    if cfg.n_modes < 2:
        raise ValueError(f"n_modes must be >= 2, got {cfg.n_modes}")

    n_phases = cfg.n_modes * (cfg.n_modes - 1)
    enc_idx = int(cfg.encoding_phase_idx)
    if enc_idx < 0 or enc_idx >= n_phases:
        raise ValueError(
            f"encoding_phase_idx must be in [0, {n_phases - 1}] for {cfg.n_modes} modes, "
            f"got {enc_idx}"
        )

    st = tuple(int(x) for x in cfg.input_state)
    if cfg.output_mode == "singles":
        if len(st) != 1:
            raise ValueError(
                f"singles output_mode requires input_state of length 1, got {st!r}"
            )
    elif cfg.output_mode == "coincidence":
        if len(st) != 2:
            raise ValueError(
                f"coincidence output_mode requires input_state of length 2, got {st!r}"
            )
        if st[0] == st[1]:
            raise ValueError(
                f"coincidence input_state must be two distinct modes (collision-free), got {st!r}"
            )
    else:
        raise ValueError(
            "output_mode must be either 'singles' or 'coincidence', "
            f"got {cfg.output_mode!r}"
        )

    for m in st:
        if m < 0 or m >= cfg.n_modes:
            raise ValueError(
                f"input_state modes must lie in [0, {cfg.n_modes - 1}], got {st!r}"
            )

    if len(st) == 1:
        if cfg.photon_distinguishability is not None:
            raise ValueError(
                "photon_distinguishability must be None for single-photon input_state"
            )
    else:
        if cfg.photon_distinguishability is None:
            raise ValueError(
                "photon_distinguishability is required for two-photon input_state "
                "(use 'indistinguishable' or 'distinguishable')"
            )
        if cfg.photon_distinguishability not in (
            "indistinguishable",
            "distinguishable",
        ):
            raise ValueError(
                "photon_distinguishability must be 'indistinguishable' or 'distinguishable', "
                f"got {cfg.photon_distinguishability!r}"
            )
        if cfg.photon_distinguishability == "distinguishable":
            raise NotImplementedError(
                "Two-photon distinguishable simulation is not implemented "
                f"(backend={cfg.backend!r}); use 'indistinguishable'."
            )

    if cfg.output_mode == "coincidence":
        if cfg.working_detectors is None or len(cfg.working_detectors) == 0:
            raise ValueError(
                "coincidence output_mode requires a non-empty working_detectors tuple"
            )
        for wd in cfg.working_detectors:
            if wd < 0 or wd >= cfg.n_modes:
                raise ValueError(
                    "working_detectors indices must lie within the circuit modes, "
                    f"got {cfg.working_detectors!r}"
                )

    if cfg.output_mode not in ("singles", "coincidence"):
        raise ValueError(
            "output_mode must be either 'singles' or 'coincidence', "
            f"got {cfg.output_mode!r}"
        )

    # Readout: scalar coincidence regression needs a pair of output modes -> CC index
    if cfg.output_mode == "coincidence" and cfg.loss_type == "mse":
        if cfg.target_mode is None or len(cfg.target_mode) != 2:
            raise ValueError(
                "coincidence regression (loss_type='mse') requires target_mode as a pair "
                "(j, k) of output modes naming the coincidence channel"
            )

    if cfg.output_mode == "coincidence" and cfg.loss_type == "cross_entropy":
        if cfg.n_classes > 1:
            from .coincidence import working_detectors_to_cc_indices

            n_cc = len(working_detectors_to_cc_indices(cfg.working_detectors, cfg.n_modes))
            if cfg.n_classes != n_cc:
                raise ValueError(
                    f"coincidence cross-entropy with n_classes={cfg.n_classes} requires "
                    f"n_classes to match the number of coincidence channels from "
                    f"working_detectors ({n_cc})"
                )


def psr_photon_counts_for_phases(sim_cfg: "SimConfig", n_phase_params: int) -> Tuple[int, ...]:
    """Per-trainable-phase photon count for PSR (training-only; not physical config)."""
    n = 2 if sim_cfg.output_mode == "coincidence" else 1
    return tuple(n for _ in range(n_phase_params))


@dataclass(frozen=True)
class CircuitConfig:
    """Circuit geometry + input + encoding + measurement (same field set as SimConfig slice).

    No optional defaults on physics fields: callers must set input_state,
    encoding_phase_idx, and photon_distinguishability consistently.
    """

    n_modes: int
    input_state: Tuple[int, ...]
    encoding_phase_idx: int
    photon_distinguishability: Optional[str]
    output_mode: str
    working_detectors: Optional[Tuple[int, ...]]

    @property
    def n_phases(self) -> int:
        """Total number of Clements mesh phases for ``n_modes``."""

        return self.n_modes * (self.n_modes - 1)

    @property
    def n_coincidences(self) -> int:
        """Number of unique coincidence channels for ``n_modes``."""

        return self.n_modes * (self.n_modes - 1) // 2

    @property
    def singles_input_mode(self) -> int:
        """Input mode index for single-photon paths."""

        return int(self.input_state[0])

    def validate(self) -> None:
        """Validate circuit description (delegates to full SimConfig-compatible checks)."""

        # Minimal checks without full SimConfig; mirror validate_sim_config logic for circuit fields
        if self.n_modes < 2:
            raise ValueError(f"n_modes must be >= 2, got {self.n_modes}")
        n_phases = self.n_phases
        idx = int(self.encoding_phase_idx)
        if idx < 0 or idx >= n_phases:
            raise ValueError(
                f"encoding_phase_idx must be in [0, {n_phases - 1}], got {idx}"
            )
        if self.output_mode not in ("singles", "coincidence"):
            raise ValueError(
                "output_mode must be either 'singles' or 'coincidence', "
                f"got {self.output_mode!r}"
            )
        st = self.input_state
        if self.output_mode == "singles" and len(st) != 1:
            raise ValueError(f"singles requires input_state length 1, got {st!r}")
        if self.output_mode == "coincidence":
            if len(st) != 2 or st[0] == st[1]:
                raise ValueError(
                    f"coincidence requires two distinct modes in input_state, got {st!r}"
                )
            if (
                self.working_detectors is None
                or len(self.working_detectors) == 0
            ):
                raise ValueError("coincidence requires non-empty working_detectors")
        for m in st:
            if m < 0 or m >= self.n_modes:
                raise ValueError(f"input_state out of range: {st!r}")
        if len(st) == 1:
            if self.photon_distinguishability is not None:
                raise ValueError("photon_distinguishability must be None for singles")
        else:
            if self.photon_distinguishability not in (
                "indistinguishable",
                "distinguishable",
            ):
                raise ValueError(
                    "two-photon input requires photon_distinguishability "
                    "'indistinguishable' or 'distinguishable'"
                )
        if self.working_detectors is not None:
            for wd in self.working_detectors:
                if wd < 0 or wd >= self.n_modes:
                    raise ValueError(f"working_detectors out of range: {self.working_detectors!r}")

    @classmethod
    def from_sim_config(cls, cfg: "SimConfig") -> "CircuitConfig":
        """Extract a :class:`CircuitConfig` from an existing ``SimConfig``."""

        return cls(
            n_modes=cfg.n_modes,
            input_state=cfg.input_state,
            encoding_phase_idx=cfg.encoding_phase_idx,
            photon_distinguishability=cfg.photon_distinguishability,
            output_mode=cfg.output_mode,
            working_detectors=cfg.working_detectors,
        )


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
    input_state: Tuple[int, ...]
    encoding_phase_idx: int
    photon_distinguishability: Optional[str]
    target_mode: Optional[Tuple[int, ...]]
    memristive_phase_idx: Optional[Union[int, Tuple[int, ...]]]
    memristive_output_modes: Optional[Tuple[Tuple[int, int], ...]]

    # ── measurement mode ───────────────────────────────────────
    output_mode: str  # "singles" | "coincidence"
    working_detectors: Optional[Tuple[int, ...]]  # required for coincidence
    noise_std: Optional[Union[float, Tuple[float, ...]]]

    # ── simulation ─────────────────────────────────────────────
    n_samples: int
    memory_depth: int
    n_swipe: int
    swipe_span: float
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
            mom = tuple((int(p[0]), int(p[1])) for p in mom_raw)

        tm_raw = cfg.get("target_mode")
        if tm_raw is None:
            tm = None
        elif isinstance(tm_raw, list):
            tm = tuple(int(x) for x in tm_raw)
        else:
            tm = tuple(int(x) for x in tm_raw)

        def as_int_tuple(key: str) -> Tuple[int, ...]:
            v = cfg[key]
            if isinstance(v, list):
                return tuple(int(x) for x in v)
            return tuple(int(x) for x in v)

        def as_int_tuple_opt(key: str) -> Optional[Tuple[int, ...]]:
            v = cfg.get(key)
            if v is None:
                return None
            if isinstance(v, list):
                return tuple(int(x) for x in v)
            return tuple(int(x) for x in v)

        dist = cfg.get("photon_distinguishability")
        if dist is not None:
            dist = str(dist)

        return cls(
            n_modes=int(cfg["n_modes"]),
            input_state=as_int_tuple("input_state"),
            encoding_phase_idx=int(cfg["encoding_phase_idx"]),
            photon_distinguishability=dist,
            target_mode=tm,
            memristive_phase_idx=cfg.get("memristive_phase_idx"),
            memristive_output_modes=mom,
            output_mode=str(cfg["output_mode"]),
            working_detectors=as_int_tuple_opt("working_detectors"),
            noise_std=cfg.get("noise_std"),
            n_samples=int(cfg["n_samples"]),
            memory_depth=int(cfg["memory_depth"]),
            n_swipe=int(cfg["n_swipe"]),
            swipe_span=float(cfg["swipe_span"]),
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

    @property
    def circuit_config(self) -> CircuitConfig:
        """Lightweight :class:`CircuitConfig` view of the circuit-specific fields."""

        return CircuitConfig(
            n_modes=self.n_modes,
            input_state=self.input_state,
            encoding_phase_idx=self.encoding_phase_idx,
            photon_distinguishability=self.photon_distinguishability,
            output_mode=self.output_mode,
            working_detectors=self.working_detectors,
        )

    @property
    def singles_input_mode(self) -> int:
        """Input mode index for single-photon paths (first entry of input_state)."""

        return int(self.input_state[0])
