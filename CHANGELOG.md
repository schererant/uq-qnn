# Changelog

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
- **`examples/coincidence_regression.py`**, **`examples/simple_regression.py`** — `SIM_BACKEND` constant; uncertainty estimation uses **`ProcessPoolExecutor`** + `uncertainty_forward_pass`.
- **`src/__init__.py`** — Exports `numpy_backend`, `build_parametric_circuit`, `encoding_circuit_parametric`, `uncertainty_forward_pass`.
- **`src/utils.py`** — `config["sim_backend"]` default `"numpy"`.
- **`tests/test_imports.py`** — Imports `numpy_backend` module.

### Notes

- **NumPy + memristive** is supported for **singles** only (sequential loop). Coincidence + memristive remains unsupported; use **`backend="perceval"`** if that path is added later.
- **Perceval** remains the full SLOS pipeline when `backend="perceval"`; parametric reuse does not apply when `encoding_phase_idx` is set (inline mesh encoding).

### Example run reports

- **`examples/reporting.py`** — `make_run_dir(__file__)`, `write_run_summary` (writes `run_summary.json` with schema `uq-qnn.example_run.v1`), optional **`tee_stdout`** for text-only examples.
- **All example scripts** — Save figures and optional metrics under **`reports/<example_stem>/<YYYY-mm-dd_HHMMSS>/`**; repo **`.gitignore`** ignores generated run folders but keeps **`reports/README.md`** and **`reports/.gitkeep`**.

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
