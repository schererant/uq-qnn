# Changelog

## 2026-03-27 — Add ty type-checking support and clean core diagnostics

### Added

- **`ty.toml`** — Added repo-level `ty` configuration to type-check the maintained package and test suite by default while excluding legacy notebooks, playground code, generated reports, and one-off example scripts.
- **`pyproject.toml`** — Added **`ty>=0.0.26`** to the development dependency group.

### Changed

- **`src/coincidence.py`** — Added overloads and stricter local typing so coincidence postselection/noise helpers return precise types for array vs. dict inputs.
- **`src/circuit.py`**, **`src/experiment.py`**, **`src/numpy_backend.py`**, **`src/simulation/runner.py`**, **`src/training.py`** — Tightened annotations and assertions so the core runtime passes `ty` cleanly without changing behavior.
- **`tests/test_autograd.py`**, **`tests/test_experiment.py`**, **`tests/test_hardware.py`**, **`tests/test_photonic_circuit.py`** — Adjusted tests to satisfy stricter typing around frozen dataclasses, optional gradients, and config value narrowing.

## 2026-03-27 — Fix regression hardware noise, memristive UQ edge case, and gradient coverage

### Changed

- **`src/experiment.py`** — Fixed hardware-profile noise for scalar regression outputs by embedding each scalar into a 2-channel distribution before applying noise, preventing predictions from collapsing to `1.0`.
- **`src/experiment.py`** — `run_uncertainty_analysis()` now accepts `memristive_phase_idx` as `None`, a single `int`, or a sequence, matching the rest of the config API.
- **`src/autograd.py`** — Finite-difference gradients now clip every memristive weight parameter to `[0.01, 1.0]` instead of treating all but the final weight like wrapped phase parameters.

### Added

- **`tests/test_experiment.py`** — Regression tests for hardware-profile noise on scalar predictions and uncertainty analysis with `memristive_phase_idx` provided as an `int`.
- **`tests/test_autograd.py`** — Regression test that compares multi-memristive weight gradients against an external finite-difference calculation.

## 2026-03-27 — PhotonicCircuit core and simulation package cleanup

### Added

- **`src/config.py`** — Introduced `CircuitConfig` (geometry + measurement) with validation helpers and exposed it via `SimConfig.circuit_config`.
- **`src/circuit.py`** — New `PhotonicCircuit` façade that wraps the NumPy backend; provides singles/coincidence (batched) outputs, cached mesh unitaries, and convenience helpers (`target_probability`, `with_phases`, `random`).
- **`src/simulation/logger.py`**, **`src/simulation/memristive.py`**, **`src/simulation/__init__.py`** — Broke out the global `SimulationLogger`, introduced a `MemristiveState` buffer manager, and re-exported the public simulation API from the new package namespace.
- **`tests/test_photonic_circuit.py`** — Regression tests covering phase validation, batch/vectorized paths, coincidence math, and parity with `run_simulation_sequence_np`.

### Changed

- **`src/simulation/runner.py`** — Former `simulation.py` is now the runner module; imports the new logger/memristive helpers, keeps the existing function signatures, and continues to support NumPy and Perceval backends.
- **`src/circuits.py`** — Promoted `_clements_mzi_pairs` / memristive normalizers to public functions (with backwards-compatible aliases) so downstream code (training, visualization, etc.) no longer imports private helpers from the simulation module.
- **`src/numpy_backend.py`** — Added `all_singles_from_unitary` / `all_coincidences_from_unitary` conveniences, updated imports to the public circuit helpers.
- **`src/training.py`**, **`src/circuit_visualization.py`**, **`src/experiment.py`** — Updated to consume `normalize_memristive_phase_idx` from `circuits`; `Experiment.predict` now short-circuits through `PhotonicCircuit` when the config is a plain NumPy singles/coincidence run.
- **`src/coincidence.py`** — Made the Perceval import lazy so basic coincidence helpers work without Perceval installed.
- **`README.md`** — Documented the new `PhotonicCircuit` workflow for quick, SimConfig-free simulations.

### Removed

- **`src/inference.py`** — Dead, broken module replaced by `PhotonicCircuit` and the `Experiment` API.

## 2026-03-26 — Hardware abstraction layer, noise models, and legacy cleanup

### Added

- **`src/hardware.py`** — Full hardware abstraction layer:
    - **`NoiseModel` protocol** — Structural typing for noise callables `(outcomes, working_indices, labels, *, rng) -> np.ndarray`.
    - **`GaussianNoise(std)`** — Additive Gaussian noise, clipped to `[0, 1]` and renormalized. Supports per-channel `std` as a tuple.
    - **`ShotNoise(n_samples)`** — Poisson-distributed shot noise from multinomial sampling.
    - **`DarkCountNoise(rate_per_detector)`** — Constant dark count baseline added to each detector, renormalized.
    - **`CompositeNoise(models)`** — Chains multiple noise models in sequence.
    - **`TimingParams`** — Frozen dataclass for hardware timing: `t_phase_ms`, `f_laser_khz`, `det_window_us`, `max_swipe`. `compute_n_swipe()` delegates to `data.compute_n_swipe()`.
    - **`HardwareProfile`** — Frozen dataclass bundling `name`, `backend`, `noise`, `timing`, and coincidence window constants. `apply_to_config(cfg)` merges profile defaults into a config dict (explicit keys in the dict take precedence). `to_dict()` returns a JSON-serializable representation.
    - **`HardwareBackend` protocol** and **`RealHardwareBackend`** placeholder — Interface for future physical hardware connection; `run_circuit()` raises `NotImplementedError` until implemented.
    - **Backend registry** — `register_backend(name, backend)` / `get_backend(name)` for custom hardware backends.
    - **Built-in profiles** — `IDEAL` (no noise), `LAB_6MODE` (Gaussian std=0.02), `NOISY_PROTOTYPE` (Gaussian std=0.05 + dark counts rate=0.002).
    - **Profile registry** — `get_profile(name)` looks up profiles by string name; raises `KeyError` with a clear message on unknown names.
- **`examples/hardware_profile_comparison.py`** — Trains the same 6-mode quartic regression under `IDEAL`, `LAB_6MODE`, and `NOISY_PROTOTYPE` profiles; plots side-by-side predictions with 95% CI.
- **`tests/test_hardware.py`** — 27 tests covering noise model contracts (shape, clipping, seed reproducibility, per-channel std), `CompositeNoise` chaining, `HardwareProfile.to_dict()` and `apply_to_config()` merge precedence, `TimingParams`, `get_profile`, `RealHardwareBackend`, and backend registry.

### Changed

- **`src/experiment.py`** — `Experiment.__init__` accepts `hardware: HardwareProfile | str | None`. Profile is resolved, merged into config via `apply_to_config()`, and applied as a post-processing step after `predict()` and each UQ pass. `run_summary.json` includes `hardware.to_dict()`. When both a profile noise model and `config["noise_std"]` are set, the profile takes precedence and a warning is logged.
- **`src/utils.py`** — Stripped to ~40 lines. Removed `config` dict, `main()`, `_run_training()`, `run_cli()`, and all associated imports. Kept `print_run_params()` (deprecated wrapper) and `resolve_n_swipe(timing, n_swipe_override)` (now accepts `TimingParams` instead of reading a global dict).
- **`src/__init__.py`** — Registers `hardware` module; exports all public hardware types; removes dead utils exports (`main`, `config`).
- **`examples/circuit_comparison.py`**, **`examples/circuit_comparison_quartic.py`**, **`examples/circuit_visualization_training.py`**, **`examples/function_comparison.py`**, **`examples/quartic_regression_comparison.py`**, **`examples/memristor_circuit_visualization.py`** — Replaced `from src.utils import config` with explicit local `CONFIG` / constant dicts. No more shared mutable global.

### Removed

- **`main.py`** — Legacy CLI entry point removed. Use `Experiment`-based scripts in `examples/` instead.
- **`config/`** directory — `config.yaml` and `quantum/pcvl.yaml` removed; settings are now passed directly via `CONFIG` dicts in each script.

---

## 2026-03-26 — Experiment: script-level CONFIG, API bridge, no hidden defaults

### Added

- **`Experiment` API** (`src/experiment.py`) — Context manager with:
    - **`config` dict only** — No merge with `src.utils.config`; required keys are validated at construction.
    - **`train(X, y, *, encoded=False)`** — Maps config to `train_pytorch_generic` (standard `2 * arccos(X)` encoding unless `encoded=True` for pre-encoded phases).
    - **`predict(theta, encoded_phases, *, return_class_probs=False)`** — Single forward pass via `run_simulation_sequence_np` using the same config.
    - **`run_uncertainty_analysis(theta, encoded_phases, *, n_passes, noise_std)`** — Parallel UQ via `uncertainty_forward_pass`; classification uses `return_class_probs` when `loss_type == "cross_entropy"`.
    - **`savefig(fig, name, ...)`** — Saves under `run_dir` and records the path in `run_summary.json` artifacts.
    - Run directories: **`reports/<slug>/<YYYY-mm-dd_HHMMSS>/`** (slug from experiment name).
- **`src/__init__.py`** — Exports **`Experiment`**.

### Changed

- **Example scripts** — Each file starts with an explicit **`CONFIG = { ... }`** block (all circuit, training, task, data, and UQ-related values used by that script). `main()` calls `exp.train` / `exp.predict` / `exp.run_uncertainty_analysis` instead of repeating long keyword lists to `train_pytorch` / `run_simulation_sequence_np`.
    - Updated: **`examples/simple_regression.py`**, **`examples/simple_classification.py`**, **`examples/coincidence_regression.py`**, **`examples/two_moons_classification.py`**, **`examples/multi_class_classification.py`**, **`examples/simple_regression_test.py`**.
- **`uncertainty_forward_pass`** (`src/simulation.py`) — Forwards **`return_class_probs`** from the job `cfg` into `run_simulation_sequence_np`, so UQ matches classification inference without a duplicate worker in `experiment.py`.

### Fixed

- **Broken examples** — **`two_moons_classification.py`**, **`multi_class_classification.py`**, and **`simple_regression_test.py`** no longer import the removed **`examples/reporting.py`**; they use **`Experiment`** like the other examples.

---

## 2026-03-25 — Experiment lifecycle, reporting, and explicit training kwargs

### Added

- **`src/experiment.py`** (initial iteration) — Context manager: timestamped `reports/`, stdout tee to `run.log`, `run_summary.json`, UQ helper.
- **`SimulationLogger.stats_dict()`** in `src/simulation.py` — JSON-serializable simulation statistics for reporting.

### Changed

- **`src/training.py`** — Removed default values from `train_pytorch` and `train_pytorch_generic` keyword-only arguments so callers (and later `Experiment`) pass parameters explicitly.
- **`examples/simple_regression.py`**, **`examples/simple_classification.py`**, **`examples/coincidence_regression.py`** — First pass: adopt `Experiment` (superseded by the 2026-03-26 CONFIG-centric layout above).

### Fixed

- **`src/numpy_backend.py`** — `ValueError` in `unitary_for_point` when a pre-computed encoding unitary was passed into batch code (broadcasting mismatch).

### Removed

- **`examples/reporting.py`** — Replaced by `src/experiment.py`.

## 2026-03-25 — Simulation performance: NumPy backend, parametric Perceval, parallel uncertainty

### Added

- **`src/numpy_backend.py`** — Fast path that builds the Clements unitary in NumPy and evaluates singles / collision-free coincidence probabilities via the Born rule (2×2 permanents). For non-memristive discrete runs, **all data points are vectorized** in one pass (separate or inline mesh encoding).
- **`backend` parameter** — `run_simulation_sequence_np(..., backend="numpy" | "perceval")`. Default **`numpy`**. Threaded through `MemristorLossPSR`, `PhotonicModel`, `train_pytorch` / `train_pytorch_generic`.
- **CLI** — `main.py`: `--backend {numpy,perceval}` (stored in `config["sim_backend"]` alongside existing config).
- **`build_parametric_circuit` / `encoding_circuit_parametric`** in `src/circuits.py` — Perceval `pcvl.P` encoding phase for reuse; requires `encoding_phase_idx=None` (legacy separate encoding block).
- **`SimulationLogger.log_circuits(elapsed, count)`** — Batch-friendly logging for vectorized runs.
- **`uncertainty_forward_pass(job)`** in `src/simulation.py` — Picklable `(params, n_samples, encoded_phases, cfg)` worker for multiprocessing.
- **`tests/test_numpy_perceval_agreement.py`** — Asserts NumPy vs Perceval outputs match within tight tolerance (singles, coincidence, inline encoding).

### Changed

- **`src/simulation.py`** — Dispatches on `backend`; Perceval path reuses one Processor/Sampler with `set_value` on the encoding parameter when discrete, non-memristive, and separate encoding; swipe inner loop uses the same pattern when applicable.
- **`src/circuits.py`** — `@lru_cache` on `_clements_mzi_pairs` (returns `tuple`).
- **`examples/coincidence_regression.py`**, **`examples/simple_regression.py`** — Uncertainty estimation uses **`ProcessPoolExecutor`** + **`uncertainty_forward_pass`** (later wired through **`Experiment.run_uncertainty_analysis`**; see 2026-03-26 entry).
- **`src/__init__.py`** — Exports `numpy_backend`, `build_parametric_circuit`, `encoding_circuit_parametric`, `uncertainty_forward_pass`.
- **`src/utils.py`** — `config["sim_backend"]` default `"numpy"`.
- **`tests/test_imports.py`** — Imports `numpy_backend` module.

### Notes

- **NumPy + memristive** is supported for **singles** only (sequential loop). Coincidence + memristive remains unsupported; use **`backend="perceval"`** if that path is added later.
- **Perceval** remains the full SLOS pipeline when `backend="perceval"`; parametric reuse does not apply when `encoding_phase_idx` is set (inline mesh encoding).

### Example run reports

- **Figures and summaries** — Example scripts save outputs under **`reports/<slug>/<YYYY-mm-dd_HHMMSS>/`** via **`Experiment`** (`run_summary.json` schema **`uq-qnn.experiment_run.v1`**). Repo **`.gitignore`** ignores generated run folders; **`reports/README.md`** and **`reports/.gitkeep`** stay tracked.

## 2026-03-24 — Fix broken PSR gradients for coincidence (2-photon) mode

### Bug: Wrong `n_photons` in Parameter Shift Rule (Critical)

**Files changed:** `examples/coincidence_regression.py`, `main.py`, `src/utils.py`

The photonic parameter-shift rule (PSR) from arXiv:2410.02726 computes exact
gradients for phase parameters in linear optical circuits. The number of shift
terms depends on the photon count `n` through each phase shifter: the output
probability is an `n`-th order trigonometric polynomial, and the PSR needs `2n`
shifted evaluations to capture all harmonics.

The coincidence mode injects **2 photons** (e.g. modes 1 and 4), so each phase
can carry up to 2 photons in superposition. The coincidence probability
P(CC_jk) is therefore a **degree-2** Fourier series in each phase:

    P(theta_k) = a0 + a1*cos(theta_k) + b1*sin(theta_k)
                    + a2*cos(2*theta_k) + b2*sin(2*theta_k)

`n_photons` was hardcoded to `1` everywhere. With n=1, the PSR uses only 2
shift terms (at 2pi/3 and 4pi/3) and captures only the 1st harmonic. The 2nd
harmonic derivative terms — which carry the dominant x^4 signal via
cos(2*arccos(x)) = 8x^4 - 8x^2 + 1 — were completely invisible to the
optimizer. The computed gradients were systematically wrong, explaining why the
loss curve was effectively flat and the model output a near-constant prediction.

**Fix:** `n_photons` is now derived from the number of input photons:
- `examples/coincidence_regression.py`: `n_photons = tuple([len(input_modes)] * n_phases)` — evaluates to 2 for 2-photon coincidence.
- `main.py` (`update_config_from_args`): `n_photons` is set *after* `output_mode` is resolved. Coincidence mode gets `n=2`, singles mode keeps `n=1`.
- `src/utils.py`: Added clarifying comment on the default `n_photons` value.

### Bug: Insufficient training epochs

**File changed:** `examples/coincidence_regression.py`

`epochs` was set to 5 for a 30-parameter Clements circuit. Five gradient steps
cannot meaningfully navigate a 30-dimensional optimization landscape, regardless
of gradient quality. Increased to 60 to give Adam enough iterations to converge.

### Cleanup: Stale docstring

**File changed:** `examples/coincidence_regression.py`

The module docstring referenced "Working detectors: modes 0, 1, 5" but the code
sets `working_detectors = tuple(range(n_modes))` (all 6 detectors active).
Updated to match the actual behavior.
