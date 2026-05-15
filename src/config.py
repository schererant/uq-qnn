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

from .clements_geometry import normalize_encoding_phase_idx, normalize_memristive_phase_idx

PhotonDistinguishability = Literal["indistinguishable", "distinguishable"]

# N-fold coincidence guardrails (exact simulation supported, but not always practical).
_MAX_NFOLD_PHOTONS = 8
_MAX_NFOLD_CHANNELS = 500


def validate_sim_config(cfg: "SimConfig") -> None:
    """Validate physics and readout semantics. Call before simulation/training."""

    if cfg.n_modes < 2:
        raise ValueError(f"n_modes must be >= 2, got {cfg.n_modes}")

    if cfg.n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {cfg.n_layers}")

    n_phases_per_layer = cfg.n_modes * (cfg.n_modes - 1)
    enc_slots = normalize_encoding_phase_idx(
        cfg.encoding_phase_idx, cfg.n_layers, cfg.n_modes
    )
    n_enc = len(enc_slots)

    memristive_indices = normalize_memristive_phase_idx(
        cfg.memristive_phase_idx, cfg.n_modes, n_phases_per_layer
    )
    if memristive_indices and cfg.n_layers > 1:
        raise ValueError(
            "multi-layer re-uploading (n_layers > 1) is not supported with memristive phases; "
            "use n_layers=1"
        )
    overlap = set(enc_slots) & set(memristive_indices)
    if overlap:
        raise ValueError(
            f"encoding_phase_idx must not overlap memristive_phase_idx, got {overlap}"
        )

    if cfg.n_layers > 1:
        if cfg.n_enc_features is not None:
            expected_enc = cfg.n_enc_features * cfg.n_layers
            if n_enc != expected_enc:
                raise ValueError(
                    f"encoding_phase_idx length ({n_enc}) must equal "
                    f"n_enc_features * n_layers ({expected_enc})"
                )
        elif n_enc % cfg.n_layers != 0:
            raise ValueError(
                f"encoding_phase_idx length ({n_enc}) must be divisible by n_layers "
                f"({cfg.n_layers})"
            )

    if cfg.output_mode not in ("singles", "coincidence"):
        raise ValueError(
            "output_mode must be either 'singles' or 'coincidence', "
            f"got {cfg.output_mode!r}"
        )

    from .coincidence import (
        migration_hint_legacy_input_state_pair,
        nfold_channel_count,
        nfold_tuple_to_channel_index,
        nfold_working_detector_tuples,
    )

    st = tuple(int(x) for x in cfg.input_state)
    if len(st) != cfg.n_modes:
        hint = migration_hint_legacy_input_state_pair(st, cfg.n_modes)
        msg = (
            f"input_state must be a length-{cfg.n_modes} occupation vector "
            f"(non-negative integers, sum = photon number), got length {len(st)!r}: {st!r}."
        )
        if hint:
            msg = f"{msg} {hint}"
        raise ValueError(msg)

    if any(x < 0 for x in st):
        raise ValueError(f"input_state occupations must be non-negative, got {st!r}")

    n_photons = int(sum(st))
    if n_photons < 1:
        raise ValueError(f"input_state must contain at least one photon, got {st!r}")

    if cfg.output_mode == "singles":
        if n_photons != 1:
            raise ValueError(
                f"singles output_mode requires exactly one photon (sum(input_state)==1), "
                f"got sum={n_photons} for {st!r}"
            )
        if max(st) > 1:
            raise ValueError(
                f"singles requires a single photon in one mode, got {st!r}"
            )
    elif cfg.output_mode == "coincidence":
        if n_photons < 2:
            raise ValueError(
                f"coincidence output_mode requires at least two photons "
                f"(sum(input_state) >= 2), got sum={n_photons} for {st!r}"
            )

    if n_photons == 1:
        if cfg.photon_distinguishability is not None:
            raise ValueError(
                "photon_distinguishability must be None for single-photon input_state"
            )
    else:
        if cfg.photon_distinguishability is None:
            raise ValueError(
                "photon_distinguishability is required when sum(input_state) >= 2 "
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
                "Multi-photon distinguishable simulation is not implemented "
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
        wd_sorted = sorted(int(x) for x in cfg.working_detectors)
        if len(wd_sorted) != len(set(wd_sorted)):
            raise ValueError(
                f"working_detectors must have unique indices, got {cfg.working_detectors!r}"
            )
        w = len(wd_sorted)
        if n_photons > w:
            raise ValueError(
                f"cannot form N-fold coincidence channels: sum(input_state)={n_photons} "
                f"exceeds len(working_detectors)={w}"
            )
        n_ch = nfold_channel_count(w, n_photons)
        if n_ch > _MAX_NFOLD_CHANNELS or n_photons > _MAX_NFOLD_PHOTONS:
            raise ValueError(
                f"N-fold coincidence exceeds default guardrails (exact but expensive): "
                f"N={n_photons}, C(W,N)={n_ch} with W={w}. "
                f"Reduce photon number or working detectors (limits: "
                f"N<={_MAX_NFOLD_PHOTONS}, channels<={_MAX_NFOLD_CHANNELS})."
            )

    if cfg.output_mode == "coincidence" and cfg.loss_type == "mse":
        if cfg.target_mode is None:
            raise ValueError(
                "coincidence regression (loss_type='mse') requires target_mode as a tuple of "
                f"{n_photons} distinct detector indices in working_detectors"
            )
        if len(cfg.target_mode) != n_photons:
            raise ValueError(
                f"coincidence regression requires len(target_mode)==sum(input_state) "
                f"({n_photons}), got len={len(cfg.target_mode)!r} for target_mode={cfg.target_mode!r}"
            )
        try:
            nfold_tuple_to_channel_index(
                cfg.target_mode, cfg.working_detectors, n_photons
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid coincidence regression target_mode={cfg.target_mode!r}: {exc}"
            ) from exc

    if cfg.output_mode == "coincidence" and cfg.loss_type == "cross_entropy":
        if cfg.n_classes > 1:
            assert cfg.working_detectors is not None
            n_cc = len(nfold_working_detector_tuples(cfg.working_detectors, n_photons))
            if cfg.n_classes != n_cc:
                raise ValueError(
                    f"coincidence cross-entropy requires n_classes=C(W,N)={n_cc} "
                    f"(W=len(working_detectors), N=sum(input_state)), got n_classes={cfg.n_classes}"
                )

    # ── feedback_mode validation ───────────────────────────────────────────────
    if cfg.feedback_mode not in ("internal_arm", "chip_output"):
        raise ValueError(
            f"feedback_mode must be 'internal_arm' or 'chip_output', got {cfg.feedback_mode!r}"
        )

    if cfg.feedback_mode == "chip_output":
        if cfg.feedback_modes is None or len(cfg.feedback_modes) == 0:
            raise ValueError(
                "feedback_mode='chip_output' requires a non-empty feedback_modes tuple "
                "(one (port_a, port_b) pair per memristive index)"
            )
        for pair in cfg.feedback_modes:
            for m in pair:
                if int(m) < 0 or int(m) >= cfg.n_modes:
                    raise ValueError(
                        f"feedback_modes index {m} out of range for n_modes={cfg.n_modes}"
                    )
        if cfg.output_mode == "coincidence" and cfg.loss_type == "mse":
            if cfg.computation_modes is None:
                raise ValueError(
                    "feedback_mode='chip_output' with coincidence regression (loss_type='mse') "
                    "requires computation_modes=(port_c, port_d)"
                )
            for m in cfg.computation_modes:
                if int(m) < 0 or int(m) >= cfg.n_modes:
                    raise ValueError(
                        f"computation_modes index {m} out of range for n_modes={cfg.n_modes}"
                    )


def psr_photon_counts_for_phases(
    sim_cfg: "SimConfig", n_phase_params: int
) -> Tuple[int, ...]:
    """Per-trainable-phase photon count for PSR (training-only; not physical config)."""
    n = int(sum(sim_cfg.input_state))
    return tuple(n for _ in range(n_phase_params))


@dataclass(frozen=True)
class CircuitConfig:
    """Circuit geometry + input + encoding + measurement (same field set as SimConfig slice).

    No optional defaults on physics fields: callers must set input_state,
    encoding_phase_idx, and photon_distinguishability consistently.
    """

    n_modes: int
    input_state: Tuple[int, ...]
    encoding_phase_idx: Union[int, Tuple[int, ...]]
    photon_distinguishability: Optional[str]
    output_mode: str
    working_detectors: Optional[Tuple[int, ...]]

    @property
    def n_phases(self) -> int:
        """Total Clements mesh phases per layer for ``n_modes``."""

        return self.n_modes * (self.n_modes - 1)

    @property
    def n_coincidences(self) -> int:
        """Legacy: number of two-photon collision-free pairs on the full ``n_modes`` mesh."""

        return self.n_modes * (self.n_modes - 1) // 2

    @property
    def singles_input_mode(self) -> int:
        """Mode index of the single photon (requires sum(input_state)==1)."""

        st = self.input_state
        if sum(int(x) for x in st) != 1:
            raise ValueError(
                "singles_input_mode requires exactly one photon in input_state"
            )
        for i, v in enumerate(st):
            if int(v) == 1:
                return int(i)
            if int(v) > 1:
                raise ValueError(
                    f"singles_input_mode requires a single photon in one mode, got {st!r}"
                )
        raise ValueError(f"singles_input_mode: no photon found in {st!r}")

    def validate(self) -> None:
        """Validate circuit description (delegates to full SimConfig-compatible checks)."""

        from .coincidence import migration_hint_legacy_input_state_pair

        if self.n_modes < 2:
            raise ValueError(f"n_modes must be >= 2, got {self.n_modes}")
        normalize_encoding_phase_idx(
            self.encoding_phase_idx, 1, self.n_modes
        )
        if self.output_mode not in ("singles", "coincidence"):
            raise ValueError(
                "output_mode must be either 'singles' or 'coincidence', "
                f"got {self.output_mode!r}"
            )
        st = tuple(int(x) for x in self.input_state)
        if len(st) != self.n_modes:
            hint = migration_hint_legacy_input_state_pair(st, self.n_modes)
            msg = f"input_state must have length n_modes={self.n_modes}, got {len(st)!r}: {st!r}."
            if hint:
                msg = f"{msg} {hint}"
            raise ValueError(msg)
        if any(x < 0 for x in st):
            raise ValueError(f"input_state must be non-negative, got {st!r}")
        n_photons = int(sum(st))
        if n_photons < 1:
            raise ValueError(
                f"input_state must contain at least one photon, got {st!r}"
            )
        if self.output_mode == "singles":
            if n_photons != 1 or max(st) > 1:
                raise ValueError(
                    f"singles requires exactly one photon in one mode, got {st!r}"
                )
        elif self.output_mode == "coincidence":
            if n_photons < 2:
                raise ValueError(
                    f"coincidence requires sum(input_state) >= 2, got {st!r}"
                )
            if self.working_detectors is None or len(self.working_detectors) == 0:
                raise ValueError("coincidence requires non-empty working_detectors")
        if n_photons == 1:
            if self.photon_distinguishability is not None:
                raise ValueError("photon_distinguishability must be None for singles")
        else:
            if self.photon_distinguishability not in (
                "indistinguishable",
                "distinguishable",
            ):
                raise ValueError(
                    "multi-photon input requires photon_distinguishability "
                    "'indistinguishable' or 'distinguishable'"
                )
        if self.working_detectors is not None:
            for wd in self.working_detectors:
                if wd < 0 or wd >= self.n_modes:
                    raise ValueError(
                        f"working_detectors out of range: {self.working_detectors!r}"
                    )

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
    ``run_simulation_sequence`` / ``run_simulation_sequence_np``,
    ``train_pytorch_generic``, ``PhotonicModel``, and ``MemristorLossPSR`` is
    collected here.

    The class is *frozen* so it can be safely shared across threads,
    stored on ``torch.autograd`` ctx objects, and serialised to JSON.

    **No field has a default value.**  All must be supplied explicitly.

    Experiment ``CONFIG`` dicts use the key ``sim_backend`` for the simulation
    engine; that value is stored on this object as the field ``backend``. Use
    :attr:`sim_backend` when you want a name that matches the config file.
    """

    # ── circuit geometry ───────────────────────────────────────
    n_modes: int
    n_layers: int
    input_state: Tuple[int, ...]
    encoding_phase_idx: Union[int, Tuple[int, ...]]
    n_enc_features: Optional[int]
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

    # ── memristive feedback mode ────────────────────────────────
    # "internal_arm"  — legacy: feedback from single-photon Born-rule column of U
    # "chip_output"   — hardware-realistic: feedback from marginal click probabilities
    #                   at dedicated detector ports (requires backend="perceval")
    feedback_mode: str  # "internal_arm" | "chip_output"
    feedback_modes: Optional[Tuple[Tuple[int, int], ...]]  # per-memristor port pairs; required for chip_output
    computation_modes: Optional[Tuple[int, int]]  # output detector pair for coincidence regression; required for chip_output

    @property
    def sim_backend(self) -> str:
        """Same as ``backend``; mirrors the ``sim_backend`` key in experiment configs."""

        return self.backend

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
        # Migration defaults for fields added after initial serialisation.
        filtered.setdefault("feedback_mode", "internal_arm")
        filtered.setdefault("feedback_modes", None)
        filtered.setdefault("computation_modes", None)
        filtered.setdefault("n_layers", 1)
        filtered.setdefault("n_enc_features", None)
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

        fm_raw = cfg.get("feedback_modes")
        fm: Optional[Tuple[Tuple[int, int], ...]]
        if fm_raw is None:
            fm = None
        else:
            fm = tuple((int(p[0]), int(p[1])) for p in fm_raw)

        cm_raw = cfg.get("computation_modes")
        cm: Optional[Tuple[int, int]]
        if cm_raw is None:
            cm = None
        else:
            cm = (int(cm_raw[0]), int(cm_raw[1]))

        enc_raw = cfg["encoding_phase_idx"]
        if isinstance(enc_raw, int):
            enc_idx: Union[int, Tuple[int, ...]] = int(enc_raw)
        elif isinstance(enc_raw, (list, tuple)):
            enc_idx = tuple(int(x) for x in enc_raw)
        else:
            enc_idx = int(enc_raw)

        n_enc_feat = cfg.get("n_enc_features")
        if n_enc_feat is not None:
            n_enc_feat = int(n_enc_feat)

        return cls(
            n_modes=int(cfg["n_modes"]),
            n_layers=int(cfg.get("n_layers", 1)),
            input_state=as_int_tuple("input_state"),
            encoding_phase_idx=enc_idx,
            n_enc_features=n_enc_feat,
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
            feedback_mode=str(cfg.get("feedback_mode", "internal_arm")),
            feedback_modes=fm,
            computation_modes=cm,
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
    def n_phases_per_layer(self) -> int:
        """Clements mesh phase count for one layer."""

        return self.n_modes * (self.n_modes - 1)

    @property
    def total_mesh_phases(self) -> int:
        """Total trainable mesh phase slots across all re-uploading layers."""

        return self.n_layers * self.n_phases_per_layer

    @property
    def encoding_slots(self) -> Tuple[int, ...]:
        """Flat indices into the stacked phase vector that receive data."""

        return normalize_encoding_phase_idx(
            self.encoding_phase_idx, self.n_layers, self.n_modes
        )

    @property
    def singles_input_mode(self) -> int:
        """Mode index of the single photon (requires sum(input_state)==1)."""

        st = self.input_state
        if sum(int(x) for x in st) != 1:
            raise ValueError(
                "singles_input_mode requires exactly one photon in input_state"
            )
        for i, v in enumerate(st):
            if int(v) == 1:
                return int(i)
            if int(v) > 1:
                raise ValueError(
                    f"singles_input_mode requires a single photon in one mode, got {st!r}"
                )
        raise ValueError(f"singles_input_mode: no photon found in {st!r}")
