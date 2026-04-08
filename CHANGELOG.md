# Changelog

## 2026-04-08 — Occupation-vector `input_state` and N-fold training coincidence (breaking)

### Why

`input_state` previously mixed meanings (length-1 or length-2 tuples of mode indices vs Fock occupation). Training and `SimConfig` now follow the same occupation-vector convention as `PhotonicCircuit`, with **N-fold postselected coincidence** over `working_detectors` (`C(W, N)` channels) instead of only two-photon pairs embedded in a full `n*(n-1)/2` layout.

### Breaking

- **`input_state`** must be a **length `n_modes` tuple/list** of non-negative integers whose **sum is the photon number**.
  - Singles: `sum(input_state) == 1`.
  - Coincidence: `sum(input_state) >= 2`.
- Legacy injection pairs like `(j, k)` are **rejected** with a migration hint pointing to an occupation vector.
- **Coincidence regression** `target_mode` is an **`N`-tuple of distinct detector indices** (subset of `working_detectors`), not only a length-2 pair in the old sense.
- **Coincidence classification**: `n_classes` must equal **`C(W, N)`** with `W = len(working_detectors)`, `N = sum(input_state)`; class order matches **lexicographic** `combinations(sorted(working_detectors), N)`.
- **PSR** photon counts use **`sum(input_state)`** (not `output_mode`).

### Added / changed (high level)

- **`src/coincidence.py`** — N-fold helpers: channel enumeration, canonical detector tuples, Perceval `probs_to_nfold_coincidences`, migration hint for legacy pairs.
- **`src/config.py`** — Validation, guardrails on `N` / channel count, `psr_photon_counts_for_phases` from occupation sum; `CircuitConfig` / `SimConfig` **`singles_input_mode`** from occupation.
- **`src/numpy_backend.py`** — `_coincidence_nfold_raw_batch` (Ryser path; keeps **vectorized N=2** closed form when input is two photons in **distinct** modes).
- **`src/simulation/runner.py`** (Perceval) — `BasicState` from occupation vector; N-fold extraction aligned with NumPy.
- **`src/circuit.py`**, **`src/circuit_visualization.py`** — Legacy / singles paths updated for occupation `input_state`.
- **`src/loss.py`** — Coincidence classification validates `n_classes` vs `C(W, N)`.
- **Examples and tests** migrated to occupation vectors.

### Fixed

- **`src/numpy_backend.py`**, **`src/circuit.py`** — Bosonic transition probability divides by **∏ n_in! ∏ n_out!** (not output factorials only), correcting repeated-input Fock states (e.g. two photons in one mode).
- **`src/numpy_backend.py`**, **`src/circuit.py`** — Ryser submatrix uses **rows = expanded output modes, columns = expanded input modes**, matching Perceval SLOS and the two-photon coincidence formula (reversed indexing matched Perceval only when input/output index sets were identical).
- **`src/training.py`** — `train_pytorch_generic` skips singles-style `target_mode` auto-fill and `len(target_mode) == n_classes` logic when **`loss_type="cross_entropy"`** and **`output_mode="coincidence"`**; **`gradient_check`** uses a valid occupation-vector **`input_state`**.
- **`README.md`** — Aligned `PhotonicCircuit` / `SimConfig` examples and coincidence wording with the occupation-vector and N-fold API.
- **`src/simulation/runner.py`** — Scalar coincidence error messages refer to an **N-tuple** of detector indices, not a two-mode “pair”.
- **Tests** — NumPy vs Perceval **N=3** coincidence agreement; bunched-input boson probability checks; coincidence cross-entropy training regression (**`target_mode`** left **`None`**).

### Cost / semantics note

- **Exact** N-fold simulation is supported for moderate `N` and `C(W,N)`; defaults **error** if `N > 8` or channels `> 500`. **Regression** (one channel) is cheaper than **full CE** over all N-fold channels; **PSR** scales with **`2N` per phase**.

## 2026-04-07 — Add full N-fold coincidence distributions to `PhotonicCircuit`

### Why

`PhotonicCircuit.coincidences()` only supported a 2-photon, collision-free vector API tied to an encoding phase. That was not sufficient for general multi-photon coincidence analysis (N > 2), PNR-vs-click detector modeling, or validating N-fold normalization behavior needed for Monte Carlo/UQ studies on fixed-noise unitary samples.

### Added

- **`src/circuit.py`** — N-fold coincidence path:
  - `PhotonicCircuit.coincidences(input_state, detector_mode="click"|"pnr", unitary=...)` now returns a dict mapping output Fock states to probabilities.
  - `detector_mode="click"` enumerates collision-free outputs only (post-selected subset).
  - `detector_mode="pnr"` enumerates all number-conserving output occupations (full distribution).
- **`src/circuit.py`** — Exact permanent evaluation via **Ryser's formula** (`_ryser_permanent`) for small-N submatrices.
- **`src/circuit.py`** — `PhotonicCircuit.accidentals_rate(singles_rates, tau, N)` utility for expected N-fold accidental rate scaling with `tau^(N-1)`.
- **`tests/test_photonic_circuit.py`** — New coverage for:
  - HOM behavior on a 50/50 beamsplitter subspace (PNR bunching and click-mode dip),
  - 3-photon PNR normalization on a random unitary,
  - click-mode as a strict collision-free subset of PNR outcomes.

### Changed

- **`src/circuit.py`** — `coincidences()` remains backward-compatible with legacy calls:
  - `coincidences(encoding_phase)` and `coincidences(encoding_phase=...)` still return the previous two-photon coincidence vector.
- **`examples/coincidence_regression.py`** — Fixed coincidence config to use a valid two-photon input state (`input_state=(0, 1)`) for `output_mode="coincidence"` validation.

### Behaviour

- N-fold coincidence probabilities are now available directly from `PhotonicCircuit` without embedding Monte Carlo logic in the method itself (callers can perturb the unitary externally per sample).
- PNR distributions are normalized (sum to ~1); click-mode distributions represent post-selected collision-free mass (sum generally < 1).

## 2026-04-07 — Fix frozen encoding-slot handling and dead comparison slots

### Why

After freezing `encoding_phase_idx` for training, the optimization code still treated that phase like a memristor weight during post-step clamping and finite-difference fallback. That could silently force the encoding slot into the weight range `[0.01, 1.0]` even though it is a mesh phase and should remain on the circle.

At the experiment level, `examples/function_comparison.py` used `encoding_phase_idx=5` for both singles and coincidence. For the chosen input/readout pairs, that slot was effectively dead, so the forward pass became input-independent, predictions collapsed to a constant, and calibration reporting emitted `ConstantInputWarning` / `NaN`.

### Changed

- **`src/training.py`** — Post-step parameter handling now clamps only appended memristive weights and wraps all mesh phases modulo `2π`, including the frozen encoding slot.
- **`src/autograd.py`** — Finite-difference fallback now applies only to true memristive weight parameters, not to the frozen encoding phase.
- **`examples/function_comparison.py`** — Split the shared encoding slot into per-mode choices: singles now uses a live slot for its `(input_state=(0,), target_mode=(2,))` setup, and coincidence now reuses the validated coincidence slot from `examples/coincidence_regression.py`.
- **`examples/function_comparison.py`** — Calibration correlation now short-circuits constant-input cases instead of letting `scipy.stats.spearmanr` warn during reporting.
- **`tests/test_training.py`** — Added a regression test that verifies the frozen encoding slot is not clamped like a weight.

### Behaviour

- `examples/function_comparison.py` no longer collapses to constant predictions for every function/mode pair.
- Calibration metrics are now informative for the repaired runs; constant-input cases remain explicitly reported as `NaN` without a warning.

## 2026-04-07 — Allow `n_samples=0` on NumPy backend

### Why

`examples/coincidence_regression.py` and similar NumPy-only runs can compute analytic probabilities without Perceval sampling, but the shared simulation runner still rejected `n_samples=0` before backend dispatch.

### Changed

- **`src/simulation/runner.py`** — Backend validation now runs before `n_samples` validation. `backend="numpy"` accepts `n_samples >= 0`, while `backend="perceval"` still requires a positive integer.

### Behaviour

- `n_samples=0` is now valid for NumPy simulation and training paths.
- Perceval sampling semantics are unchanged.

## 2026-04-07 — Simulator backend architecture cleanup

### Why

The simulation runner and NumPy backend were **tightly coupled to Perceval at import time** (`runner.py` and `numpy_backend → circuits`), and NumPy coincidence runs still built **`pcvl.BasicState`** before hitting the vectorized path. **`Experiment.predict()`** duplicated the discrete NumPy coincidence path already implemented in **`run_vectorized_non_memristive`**. Experiment **`CONFIG`** uses **`sim_backend`** while **`SimConfig`** stores **`backend`**, which was easy to confuse when reading code.

### Added

- **`src/clements_geometry.py`** — Pure layout helpers: **`clements_mzi_pairs`**, **`get_mzi_modes_for_phase`**, **`normalize_memristive_phase_idx`**, **`normalize_memristive_output_modes`** (no Perceval).
- **`src/simulation/runner.py`** — **`run_simulation_sequence`** as the primary forward API; **`run_simulation_sequence_np`** remains a **backward-compatible alias** to the same function.
- **`src/config.py`** — **`SimConfig.sim_backend`** property (alias of **`backend`**) and docstring note on the **`sim_backend`** config key vs dataclass field.
- **`tests/test_simulation_backend_imports.py`** — Subprocess check that **`src.simulation.runner`** can be imported **without loading Perceval** (stub package layout avoids **`src/__init__.py`**); alias assertion.
- **`tests/test_coincidence_regression_numpy_backend.py`** — Reference **`SimConfig`** aligned with **`examples/coincidence_regression.py`**, parity runner vs vectorized path, golden values, **`sim_backend`** property.

### Changed

- **`src/circuits.py`** — Re-exports geometry helpers from **`clements_geometry`** under the same public names.
- **`src/numpy_backend.py`** — Imports **`clements_mzi_pairs`** from **`clements_geometry`** instead of **`circuits`**.
- **`src/simulation/runner.py`** — **Lazy-imports** Perceval, **`Sampler`**, and **`build_circuit`** only on the **`backend == "perceval"`** branch; **NumPy vectorized and memristive-fast returns run before** any **`BasicState`** construction.
- **`src/simulation/__init__.py`** — Normalize re-exports from **`clements_geometry`**; exports **`run_simulation_sequence`**.
- **`src/__init__.py`** — Re-exports **`run_simulation_sequence`**.
- **`src/experiment.py`** — Discrete NumPy **`predict()`** uses **`run_vectorized_non_memristive`** when eligible; otherwise **`run_simulation_sequence`**; removed PhotonicCircuit-based predict helpers; **`_can_use_vectorized_numpy_discrete`** replaces **`_can_use_photonic_circuit`**.
- **`examples/coincidence_regression.py`** — Calls **`run_simulation_sequence`**.
- **`tests/test_imports.py`** — Asserts **`run_simulation_sequence`** is importable.
- **`src/circuits.py`**, **`src/circuit_visualization.py`** — **Lazy-import Perceval** inside Perceval-specific APIs so **`import src`** does not load **`perceval`** until **`build_circuit`**, **`encoding_circuit`**, etc., or annotated visualization runs.
- **`tests/test_package_import_hygiene.py`** — Subprocess checks that **`import src`** and **`from src.simulation import run_simulation_sequence`** stay Perceval-free, NumPy forwards via **`src.run_simulation_sequence`** stay Perceval-free, and **`src.build_circuit`** loads Perceval when invoked.

### Behaviour / limitations

- **Perceval** loads when any **Perceval circuit builder or annotated visualization** runs (e.g. **`build_circuit`**, **`display_circuit_annotated`**), not at **`import src`**.

---

## 2026-04-07 — Canonical photonic configuration, internal encoding, and NumPy batching fix

### Why

Circuit and experiment settings had grown ambiguous: **`encoding_mode`** mixed “which mode gets the photon” with “where the data-driven encoding sits,” coincidence runs relied on **implicit defaults** for input modes and working detectors, **`n_photons` on `SimConfig`** duplicated information already fixed by `output_mode` / `input_state`, and the **encoding mesh phase could be trained** alongside other phases. The goal is **one explicit injection** (`input_state`), **one explicit encoding placement** (`encoding_phase_idx` on the Clements mesh), **fail-fast validation** for coincidence readout and two-photon distinguishability, and **PSR/training metadata derived** from the same physics picture.

A follow-on issue appeared after switching the NumPy path to **internal** encoding: `_unitary_batch_internal_encoding` rebuilt a **full `clements_unitary` per data point**, while the old **external** encoding path had used **one mesh unitary plus a batched 2×2 encoding block**. That accidentally made coincidence training roughly **O(n_data)** full meshes per call instead of **one batched pass over MZIs**, which dominated wall time (e.g. ~20 s → ~20 min for `examples/coincidence_regression.py`). **`clements_unitary_batch`** restores vectorized mesh construction for all rows that share the same phase vector except at `encoding_phase_idx`.

### Added

- **`src/config.py`** — `validate_sim_config()` as the single validation entry point; `psr_photon_counts_for_phases()` for PSR photon counts derived from `output_mode` / phase count; `SimConfig.singles_input_mode` helper; **`input_state`**, **`photon_distinguishability`**, required **`encoding_phase_idx: int`**. **`CircuitConfig`** aligned with the same circuit field set (no misleading optional defaults on physics fields).
- **`src/numpy_backend.py`** — **`_mzi_unitary_batch`**, **`clements_unitary_batch`** — batched Clements products so internal-encoding batches are not implemented as a Python loop of full `clements_unitary` per point.
- **`tests/test_sim_config_validation.py`** — Validation, distinguishability, coincidence readout, and related config rules.
- **`tests/test_numpy_perceval_agreement.py`** — **`test_clements_unitary_batch_matches_serial`** — numerical parity of batched vs serial unitaries and `_unitary_batch_internal_encoding` vs `unitary_for_point`.

### Changed

- **`src/circuits.py`** — **`build_circuit`** uses **internal mesh encoding only** (phase offset at `encoding_phase_idx`); removed the default public **external prepended encoding** path and **`build_parametric_circuit`** / parametric Perceval reuse tied to external encoding.
- **`src/numpy_backend.py`** — Forward paths build from **`input_state`** and internal encoding; **`_unitary_batch_internal_encoding`** now delegates to **`clements_unitary_batch`**; coincidence scalar readout **raises** if the CC channel cannot be resolved (no silent index `0`); **`unitary_for_point`** takes **`encoding_phase_idx: int`** only.
- **`src/simulation/runner.py`** — Validates config up front; **`BasicState`** from **`input_state`**; coincidence without hidden **`input_modes` / working-detector** fallbacks; Perceval discrete path **rebuilds `build_circuit` per evaluation** when using internal encoding (documented limitation vs old parametric reuse).
- **`src/simulation/memristive.py`** — **`singles_input_mode`** (from `input_state[0]`) replaces **`encoding_mode`** for memristor monitor semantics.
- **`src/training.py`** — **`phase_idx`** excludes **`encoding_phase_idx`** and memristive indices; removed **`_resolve_n_photons`** in favor of config-driven PSR counts inside autograd.
- **`src/autograd.py`**, **`src/loss.py`** — **`MemristorLossPSR`** / **`PhotonicModel`** no longer take user-facing **`n_photons`**; counts come from **`psr_photon_counts_for_phases(sim_cfg, …)`**.
- **`src/circuit.py`**, **`src/circuit_visualization.py`** — **`PhotonicCircuit`** takes **`circuit_config: CircuitConfig`**; coincidence APIs use config’s **`input_state`** (no default `(0, 1)`).
- **`src/experiment.py`** — **`_REQUIRED_CONFIG_KEYS`** includes **`input_state`**, **`encoding_phase_idx`**, **`photon_distinguishability`**, **`working_detectors`** where required; drops **`encoding_mode`**, **`n_photons`**; coincidence prediction paths **do not** reintroduce hidden defaults.
- **`src/__init__.py`** — Exports **`validate_sim_config`**; drops removed parametric circuit exports.
- **`examples/`** — All experiment **`CONFIG`** blocks migrated to **`input_state` + `encoding_phase_idx` + `photon_distinguishability`**; coincidence scripts set explicit **`working_detectors`** and readout conventions; captions/logs updated.
- **`scripts/perceval_pytorch.py`** — Standalone **`MemristorLossPSR` / `PhotonicModel`** updated to drop **`n_photons`** from the public API (PSR counts fixed for that singles memristor demo).
- **`tests/`** — **`test_autograd`**, **`test_circuits`**, **`test_experiment`**, **`test_memristive_numpy_backend`**, **`test_photonic_circuit`**, **`test_training`**, etc., updated for the new config and APIs.

### Removed

- **`SimConfig` / experiment config** — **`encoding_mode`**, **`input_modes`**, **`n_photons`** as user-facing fields.
- **`src/circuits.py`** — **`build_parametric_circuit`** (and related parametric encoding export) removed from the supported API.

### Behaviour / limitations (documented in code and tests)

- **`photon_distinguishability`**: **`None`** only for **`len(input_state) == 1`**; **required** for two-photon configs; **`"distinguishable"`** raises **`NotImplementedError` / `ValueError`** (bosonic NumPy and Perceval paths implement **`"indistinguishable"`** only).
- **Coincidence** — **`working_detectors`** required when **`output_mode == "coincidence"`**; scalar MSE regression expects **`target_mode`** as an **output mode pair** **`(j, k)`** mapping to a CC index via **`mode_pair_to_cc_index`**.
- **Perceval** — No **`build_parametric_circuit`** shortcut when encoding is internal; discrete runs pay **full circuit rebuild** cost by design.

---

## 2026-04-06 — Coincidence output-pair selection and comparison experiments

### Added

- **`src/coincidence.py`** — `mode_pair_to_cc_index(j, k, n_modes)` helper that maps an output mode pair `(j, k)` to its canonical CC channel index.
- **`src/numpy_backend.py`**, **`src/simulation/runner.py`** — Coincidence regression can now read an arbitrary CC channel via `target_mode=(j, k)`. Previously the scalar output was always `working_cc_indices[0]` (i.e. the first CC pair formed by the working detectors). Now, when `target_mode` is a 2-tuple in coincidence mode, the specified CC pair is read after postselection while the full normalization context is preserved.
- **`examples/output_mode_comparison.py`** — Experiment sweeping all singles output detectors (modes 0–N-1) for one regression function.
- **`examples/coincidence_output_pair_comparison.py`** — Experiment sweeping all 15 output CC pairs via `target_mode=(j, k)` with fixed `input_modes=(1, 4)` and all detectors working.
- **`examples/coincidence_input_pair_comparison.py`** — Experiment sweeping all 15 input-mode pairs with fixed detectors and readout channel.

## 2026-04-01 — Faster NumPy path for memristive singles runs

### Added

- **`src/numpy_backend.py`** — Added a dedicated memristive singles fast path that propagates a single-photon state vector through the Clements mesh instead of rebuilding the full unitary at every time step. This keeps the memristive feedback recurrence intact while substantially reducing per-sample overhead for NumPy singles runs in both discrete and swipe modes.
- **`tests/test_memristive_numpy_backend.py`** — Added regression tests that compare the new fast memristive NumPy path against the legacy unitary-based loop for separate encoding, inline encoding / multi-target outputs, and swipe-mode execution.
- **`examples/function_memristor_comparison.py`** — Added a function-by-function architecture comparison experiment for a 6-mode standard Clements mesh versus a 6-mode memristive circuit.
- **`examples/smooth_step_memristor_placement_comparison.py`** — Added a smooth-step regression experiment that compares several single-memristor placements across the 6-mode Clements mesh against a standard no-memristor baseline.
- **`examples/smooth_step_multi_memristor_comparison.py`** — Added a smooth-step regression experiment that compares no memristor, one-memristor, and two-memristor configurations on phases `6` and `8`.
- **`examples/benchmark_memristive_backend.py`** — Added a benchmark script that times the fast memristive NumPy path against a local legacy reference implementation for discrete and swipe workloads.

### Changed

- **`src/simulation/runner.py`** — NumPy-backed singles runs with memristive phases now dispatch to the new optimized backend path instead of the older generic per-point unitary loop, including swipe-mode execution.
- **`README.md`** — Updated backend and architecture documentation to note that NumPy now supports an optimized memristive singles path, and documented the new function-vs-memristor comparison example.

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
