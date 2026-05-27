# UQ-QNN -- Uncertainty Quantification for Photonic Quantum Neural Networks

A framework for training photonic quantum neural networks (QNNs) on integrated photonic circuits, with built-in uncertainty quantification via multi-pass parameter perturbation.

---

## Overview

UQ-QNN models integrated photonic circuits (Clements mesh, photonic memristors) as differentiable quantum layers and trains them with PyTorch via the **Parameter-Shift Rule (PSR)**. It supports regression and classification tasks, two simulation backends (fast NumPy and full Perceval SLOS), and a structured `Experiment` class for reproducible runs.

### Core data flow

```
Input x --> phase encoding (2*arccos(x)) --> photonic circuit --> Born-rule probabilities --> loss --> PSR gradients --> Adam update
```

---

## Installation

Requires **Python >= 3.13** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/username/uq-qnn.git
cd uq-qnn
uv sync
```

This installs all dependencies (PyTorch, Perceval, NumPy, scikit-learn, matplotlib, tqdm, etc.) into a virtual environment managed by `uv`.

---

## Quick start

### Example scripts

| Script | Description |
|---|---|
| `examples/simple_regression.py` | Regression + UQ on a quartic function |
| `examples/simple_classification.py` | Binary classification + UQ |
| `examples/multi_class_classification.py` | 3-class Clements circuit |
| `examples/two_moons_classification.py` | 2D half-moons dataset (L=3 Mauser re-uploading) |
| `examples/simple_regression_reuploading.py` | Quartic regression with L-layer data re-uploading |
| `examples/two_moons_config_sweep.py` | Sweep mesh width / layer count for two moons |
| `examples/two_moons_seed_sweep.py` | Rank training seeds for two moons convergence |
| `examples/circuit_comparison.py` | Memristor vs. Clements on the same data |
| `examples/circuit_comparison_quartic.py` | Architecture comparison on quartic |
| `examples/quartic_regression_comparison.py` | Quartic regression ablation |
| `examples/function_comparison.py` | Model performance across synthetic functions |
| `examples/function_memristor_comparison.py` | Standard vs memristor comparison across synthetic functions |
| `examples/smooth_step_memristor_placement_comparison.py` | Compare smooth-step fitting across memristor placements |
| `examples/smooth_step_multi_memristor_comparison.py` | Compare smooth-step fitting with one vs two memristors |
| `examples/benchmark_memristive_backend.py` | Benchmark fast memristive NumPy runs against the legacy loop |
| `examples/circuit_visualization_training.py` | Live circuit training visualization |
| `examples/memristor_circuit_visualization.py` | Memristor circuit diagram |
| `examples/hardware_profile_comparison.py` | Compare ideal vs noisy hardware profiles |
| `examples/coincidence_regression.py` | Two-photon coincidence regression |
| `examples/coincidence_gaussian_bump.py` | Static vs memristive coincidence comparison on a single Gaussian bump |
| `examples/coincidence_hard_function_search.py` | Preset sweep for multi-modal, step, and oscillatory coincidence regression |
| `examples/6x6.py` | 6-mode circuit example |

```bash
uv run python examples/simple_regression.py
uv run python examples/simple_classification.py
```

### Simple circuit simulation via `PhotonicCircuit`

Need a quick probability vector without constructing a full `SimConfig`? Use `PhotonicCircuit` with a [`CircuitConfig`](src/config.py) (same circuit fields as `SimConfig`, without training/sim-only keys):

```python
import numpy as np
from src import PhotonicCircuit
from src.config import CircuitConfig

# 6-mode Clements mesh (30 phases)
phases = np.random.uniform(0, 2 * np.pi, 6 * (6 - 1))
singles_cfg = CircuitConfig(
    n_modes=6,
    input_state=tuple(1 if i == 0 else 0 for i in range(6)),
    encoding_phase_idx=5,
    photon_distinguishability=None,
    output_mode="singles",
    working_detectors=None,
)
circuit = PhotonicCircuit(n_modes=6, phases=phases, circuit_config=singles_cfg)

encoded = np.linspace(0, np.pi, 64)
singles = circuit.singles_batch(encoded)  # shape (64, 6)

# Legacy-style 2-photon coincidence vector (requires two photons in distinct modes in circuit_config.input_state)
cc_cfg = CircuitConfig(
    n_modes=6,
    input_state=tuple(1 if i in (1, 4) else 0 for i in range(6)),
    encoding_phase_idx=5,
    photon_distinguishability="indistinguishable",
    output_mode="coincidence",
    working_detectors=tuple(range(6)),
)
cc_circ = PhotonicCircuit(n_modes=6, phases=phases, circuit_config=cc_cfg)
coinc = cc_circ.coincidences(0.25)  # shape (15,) — full mesh CC basis

# General Fock analysis: pass an occupation vector (length n_modes); click vs PNR dict output
u = cc_circ.unitary(0.25)
probs_click = cc_circ.coincidences(
    (1, 1, 0, 0, 0, 0), detector_mode="click", unitary=u
)

print(circuit.config.n_phases)  # -> 30
```

Training and experiment code should use **occupation-vector** `input_state` and N-fold coincidence via `run_simulation_sequence_np` / `Experiment` (see [Config reference](#config-reference)).

---

## Creating experiments with the `Experiment` class

`Experiment` is the recommended entry point for all experiment scripts. It is a context manager that:

- Creates a timestamped run directory under `reports/<name>/<timestamp>/`
- Writes a structured `run.log` via the package logger
- Validates that all required config keys are present (no hidden defaults)
- Writes a `run_summary.json` (config, metrics, artifacts, git SHA) on exit
- After `train()`, saves `trained_state.json` (mesh + linear head + `sim_cfg`) and `loss_history.json` under the run directory
- Exposes `train`, `run_uncertainty_analysis`, and `savefig` helpers, plus a deprecated `predict` wrapper
- Builds a frozen `SimConfig` from your config dict automatically

### Step-by-step guide

1. **Define a CONFIG dict** with all required keys (see [Config reference](#config-reference) below).
2. **Prepare your data** using the built-in generators or your own arrays.
3. **Open an `Experiment` context** -- this creates the run directory and starts logging.
4. **Train** with `exp.train(X, y)` -- phase encoding is applied automatically; you get back a `trained_state` and the run directory also receives `trained_state.json` and `loss_history.json`.
5. **Predict** with `trained_state.predict(encoded_phases)` on test data.
6. **Run UQ** with `exp.run_uncertainty_analysis(trained_state, ...)` for uncertainty estimates.
7. **Save figures** with `exp.savefig(fig, "name.png")`.
8. **Exit the context** -- `run_summary.json` is written automatically (includes artifact paths and metric keys `trained_state` / `loss_history`).

To **reload a completed run** without retraining:

```python
from pathlib import Path
from src.loss import TrainedPhotonicState

run_dir = Path("reports/simple_regression/2026-05-18_120000")
state = TrainedPhotonicState.load_json(run_dir / "trained_state.json")
preds = state.predict(encoded_test)  # same encoding as training
```

### Full regression example

```python
import numpy as np
import matplotlib.pyplot as plt
from src.data import get_data
from src.experiment import Experiment

CONFIG = {
    # -- circuit geometry --
    "n_modes": 6,
    # Fock occupation vector length n_modes; exactly one photon (here in mode 4)
    "input_state": tuple(1 if i == 4 else 0 for i in range(6)),
    "encoding_phase_idx": 10,
    "photon_distinguishability": None,
    "target_mode": (4,),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,

    # -- measurement --
    "output_mode": "singles",
    "working_detectors": None,
    "noise_std": None,

    # -- simulation --
    "n_samples": 20,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "sim_backend": "numpy",

    # -- task / loss --
    "loss_type": "mse",
    "n_classes": 1,

    # -- training --
    "lr": 0.05,
    "epochs": 100,
    "seed": 42,

    # -- data (experiment-only) --
    "n_data": 20,
    "sigma_noise": 0.005,

    # -- uncertainty (experiment-only) --
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}


def main():
    X_train, y_train, X_test, y_test = get_data(
        CONFIG["n_data"], CONFIG["sigma_noise"], "quartic_data"
    )

    with Experiment("my_regression", config=CONFIG) as exp:
        # Train -- exp.train applies 2*arccos(X) encoding automatically
        trained_state, history, _ = exp.train(X_train, y_train)
        exp.save_metrics({"final_loss": history[-1]})

        # Predict on test data (must encode manually)
        enc_test = 2 * np.arccos(X_test)
        preds = trained_state.predict(enc_test)

        # Uncertainty analysis (parallel forward passes with perturbed params)
        unc = exp.run_uncertainty_analysis(
            trained_state, enc_test,
            n_passes=CONFIG["unc_n_passes"],
            noise_std=CONFIG["unc_noise_std"],
        )
        mean_preds = unc["mean"]   # shape (n_test,)
        std_preds  = unc["std"]    # shape (n_test,)

        mse = float(np.mean((mean_preds - y_test) ** 2))
        exp.save_metrics({"test_mse": mse})

        # Save a figure to the run directory
        fig, ax = plt.subplots()
        ax.plot(X_test, y_test, "k--", label="Ground truth")
        ax.plot(X_test, mean_preds, "r-", label="Prediction")
        ax.fill_between(X_test,
                        mean_preds - 2 * std_preds,
                        mean_preds + 2 * std_preds,
                        color="r", alpha=0.2, label="95% CI")
        ax.legend()
        exp.savefig(fig, "fit.png")


if __name__ == "__main__":
    main()
```

### Full classification example

```python
import numpy as np
from src.data import get_classification_data
from src.experiment import Experiment

CONFIG = {
    "n_modes": 3,
    "input_state": (1, 0, 0),
    "encoding_phase_idx": 0,
    "photon_distinguishability": None,
    "target_mode": (1, 2),          # one output mode per class (singles CE only)
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "output_mode": "singles",
    "working_detectors": None,
    "noise_std": None,
    "n_samples": 500,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "sim_backend": "numpy",
    "loss_type": "cross_entropy",
    "n_classes": 2,
    "lr": 0.03,
    "epochs": 30,
    "seed": 42,
    "n_data": 80,
    "sigma_noise": 0.05,
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}

X_train, y_train, X_test, y_test = get_classification_data(
    CONFIG["n_data"], "binary_threshold"
)

with Experiment("binary_classification", config=CONFIG) as exp:
    trained_state, history, _ = exp.train(X_train, y_train)

    enc_test = 2 * np.arccos(X_test)
    unc = exp.run_uncertainty_analysis(
        trained_state, enc_test,
        n_passes=CONFIG["unc_n_passes"],
        noise_std=CONFIG["unc_noise_std"],
    )
    mean_probs = unc["mean"]        # shape (n_test, n_classes)
    predicted_labels = np.argmax(mean_probs, axis=1)

    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-15), axis=1)
    exp.save_metrics({"entropy_mean": float(entropy.mean())})
```

---

## Config reference

Every key listed below is **required** by `Experiment`. There are no hidden defaults -- all parameters must be set explicitly. The config dict is split into two groups: keys that flow into `SimConfig` (the frozen dataclass used throughout the simulation/training stack) and experiment-only keys consumed by your script.

### Circuit geometry

| Key | Type | Description |
|---|---|---|
| `n_modes` | `int` | Number of waveguide modes in the Clements mesh. Per-layer depth: `n_modes * (n_modes - 1)` trainable phases. |
| `n_layers` | `int` | Number of stacked re-uploading Clements blocks (default `1`). Total mesh phases = `n_layers * n_modes * (n_modes - 1)`. Not compatible with memristive phases when `> 1`. |
| `n_enc_features` | `int` or `None` | Features encoded per layer (e.g. `2` for two-moons x₁/x₂). When set, `len(encoding_phase_idx)` must equal `n_enc_features * n_layers`. |
| `input_state` | `tuple[int, ...]` | **Fock occupation vector** of length `n_modes`: non-negative integers, `sum(input_state)` = total photon number. Singles: sum `1`. Coincidence: sum `≥ 2` (including bunching, e.g. `(2, 0, 0, …)`). |
| `encoding_phase_idx` | `int` or `tuple[int, ...]` | Global mesh phase index/indices where data-encoded phases are added (internal encoding). For multi-layer runs, indices span all stacked blocks — set `n_layers` explicitly. |
| `photon_distinguishability` | `str` or `None` | `None` when `sum(input_state)==1`. Required when `sum(input_state)≥2` (training stack: `"indistinguishable"` supported on NumPy). |
| `target_mode` | `tuple[int, ...]` or `None` | **Singles:** output mode(s) for Born-rule readout; regression often `(m,)`. **Singles CE:** one mode index per class. **Coincidence regression:** `N` distinct detector indices in `working_detectors` (`N = sum(input_state)`). **Coincidence CE:** class layout is `C(W, N)` N-fold channels — use `target_mode=None` or any placeholder; `n_classes` must match. |
| `memristive_phase_idx` | `int`, `tuple[int, ...]`, or `None` | Phase index/indices with memristive feedback. `None` = pure Clements mesh. |
| `memristive_output_modes` | `tuple[tuple[int, int], ...]` or `None` | Output mode pairs feeding memristive phases. |

### Measurement

| Key | Type | Description |
|---|---|---|
| `output_mode` | `str` | `"singles"` — single-photon Born probabilities. `"coincidence"` — **N-fold** post-selected coincidence distribution over `working_detectors` (`C(W, N)` channels, `N = sum(input_state)`). |
| `working_detectors` | `tuple[int, ...]` or `None` | **Coincidence:** non-empty tuple of functioning detector indices. **Singles:** `None`. |
| `noise_std` | `float`, `tuple[float, ...]`, or `None` | Gaussian noise on channel probabilities (then renormalized). `None` = noiseless. |

### Simulation

| Key | Type | Description |
|---|---|---|
| `n_samples` | `int` | Samples per data point (Perceval); `0` allowed on NumPy for analytic discrete runs. |
| `memory_depth` | `int` | Memristor buffer depth. |
| `n_swipe` | `int` | Continuous swipe sample count; `0` = discrete. |
| `swipe_span` | `float` | Swipe range in radians when `n_swipe > 0`. |
| `sim_backend` | `str` | `"numpy"` or `"perceval"`. |

### Task / loss

| Key | Type | Description |
|---|---|---|
| `loss_type` | `str` | `"mse"` or `"cross_entropy"`. |
| `n_classes` | `int` | `1` for regression. For **singles** CE, must match `len(target_mode)` (or auto-filled modes). For **coincidence** CE, must equal `C(W, N)` over `working_detectors` and `N = sum(input_state)`. |

**PSR:** photon count per phase shift is derived from `sum(input_state)` (not from `output_mode` alone).

### Training

| Key | Type | Description |
|---|---|---|
| `lr` | `float` | Adam optimizer learning rate. |
| `epochs` | `int` | Number of full passes over the training set. |
| `seed` | `int` | RNG seed for parameter initialization and UQ pass noise. |

### Data (experiment-only)

These keys are consumed by the experiment script and are **not** forwarded to `SimConfig`.

| Key | Type | Description |
|---|---|---|
| `n_data` | `int` | Number of synthetic training/test samples to generate. |
| `sigma_noise` | `float` | Label noise standard deviation for synthetic regression datasets. |

### Uncertainty (experiment-only)

| Key | Type | Description |
|---|---|---|
| `unc_n_passes` | `int` | Number of noisy forward passes for uncertainty estimation. `0` skips UQ (`run_uncertainty_analysis` returns `None`). |
| `unc_noise_std` | `float` | Standard deviation of Gaussian noise added to phase parameters on each UQ pass. Models parameter uncertainty / shot-noise variability. |

---

## The `SimConfig` dataclass

`SimConfig` (`src/config.py`) is a frozen dataclass that bundles all circuit, simulation, and task parameters into a single immutable object. It replaces the loose keyword arguments that previously flowed through the stack.

- **Frozen** -- safe to share across threads and store on `torch.autograd` ctx objects
- **No defaults** -- all fields must be supplied explicitly
- Created automatically by `Experiment` via `SimConfig.from_experiment_config(config)`
- Can also be constructed directly via `SimConfig(...)` or `SimConfig.from_dict(d)`
- Use `sim_cfg.replace(field=value)` to derive a modified copy (e.g. for UQ passes)
- `sim_cfg.to_dict()` returns a JSON-safe dict for serialization

The fields on `SimConfig` map directly to the config keys above (circuit geometry, measurement, simulation, task/loss), with one rename: the config key `sim_backend` maps to the `SimConfig.backend` field.

---

## Hardware profiles

`HardwareProfile` bundles a simulation backend, noise model, and timing parameters into a named, frozen dataclass. Use profiles to simulate different hardware conditions without changing your experiment config.

### Built-in profiles

| Profile | Backend | Noise | Description |
|---|---|---|---|
| `IDEAL` | numpy | None | Perfect noiseless simulation |
| `LAB_6MODE` | numpy | Gaussian (std=0.02) | Typical 6-mode lab setup |
| `NOISY_PROTOTYPE` | numpy | Gaussian (std=0.05) + dark counts | Early prototype with high noise |

### Using hardware profiles

Pass a `hardware` argument to `Experiment`:

```python
from src.experiment import Experiment
from src.hardware import LAB_6MODE, NOISY_PROTOTYPE

# By object
with Experiment("my_run", config=CONFIG, hardware=LAB_6MODE) as exp:
    trained_state, history, _ = exp.train(X_train, y_train)

# By name
with Experiment("my_run", config=CONFIG, hardware="noisy_prototype") as exp:
    trained_state, history, _ = exp.train(X_train, y_train)
```

The profile's noise model is applied automatically to `predict()` and `run_uncertainty_analysis()` outputs. If the profile specifies a backend (e.g. `"numpy"`), it is merged into your config (explicit `sim_backend` in the config takes precedence).

### Noise models

Noise models implement a callable protocol and are applied post-simulation (modelling detector imperfections):

| Class | Parameters | Description |
|---|---|---|
| `GaussianNoise` | `std: float \| tuple` | Additive Gaussian noise, clipped and renormalized |
| `ShotNoise` | `n_samples: int` | Poisson-distributed shot noise |
| `DarkCountNoise` | `rate_per_detector: float` | Constant dark count baseline per detector |
| `CompositeNoise` | `models: tuple[NoiseModel, ...]` | Chains multiple noise models in order |

### Custom profiles

```python
from src.hardware import HardwareProfile, GaussianNoise, CompositeNoise, DarkCountNoise, TimingParams

my_profile = HardwareProfile(
    name="my_lab",
    backend="numpy",
    noise=CompositeNoise(models=(
        GaussianNoise(std=0.03),
        DarkCountNoise(rate_per_detector=0.005),
    )),
    timing=TimingParams(
        t_phase_ms=10.0,
        f_laser_khz=50.0,
        det_window_us=10.0,
        max_swipe=21,
    ),
)
```

### Real hardware (placeholder)

`RealHardwareBackend` provides the interface for connecting to physical photonic hardware. It currently raises `NotImplementedError` -- implement `run_circuit()` to bridge to your lab control software:

```python
from src.hardware import RealHardwareBackend, register_backend

backend = RealHardwareBackend()
backend.is_available()  # False (placeholder)
register_backend("my_chip", backend)
```

---

## Source modules

| Module | Role |
|---|---|
| `src/config.py` | `SimConfig` frozen dataclass -- the single config object flowing through the entire stack |
| `src/hardware.py` | Hardware abstraction -- `HardwareProfile`, noise models, `TimingParams`, backend registry |
| `src/experiment.py` | `Experiment` context manager -- run directories, logging, train/UQ helpers, deprecated predict wrapper |
| `src/circuits.py` | Perceval circuit builders: `build_circuit()`, `build_parametric_circuit()` |
| `src/simulation.py` | Central orchestrator `run_simulation_sequence_np()`; routes to backends; handles memristive feedback, noise, swipe |
| `src/numpy_backend.py` | Fast analytic backend -- vectorized plain Clements runs, optimized memristive singles scans, 2x2 permanents for coincidences |
| `src/autograd.py` | PSR gradient engine -- `photonic_psr_coeffs_torch()`, `MemristorLossPSR` autograd Function |
| `src/loss.py` | `PhotonicModel(nn.Module)` -- wraps circuit + PSR; supports MSE and cross-entropy |
| `src/training.py` | `train_pytorch_generic()` -- main Adam training loop |
| `src/data.py` | Synthetic dataset generators and `encode_2d_to_phase()` for 2D inputs |
| `src/coincidence.py` | Multi-photon coincidence indexing, postselection, noise/accidental-correction |
| `src/logging_config.py` | Structured logging setup with file handler support |
| `src/circuit_visualization.py` | Annotated circuit display and export utilities |

---

## Circuit architectures

### Clements mesh

Rectangular mesh of Mach-Zehnder Interferometers (MZIs). Scales to arbitrary size:

- `n_modes * (n_modes - 1)` phase parameters
- Default architecture when `memristive_phase_idx=None`
- Supports singles and coincidence output modes
- Implemented analytically in `numpy_backend.py` for speed

### Memristor circuit

Compact photonic memristor with history-dependent phase feedback:

- Requires `memristive_phase_idx` to be set
- Supports `sim_backend="numpy"` for singles runs via an optimized sequential state-propagation path (discrete and swipe)
- `sim_backend="perceval"` remains available for full SLOS execution and unsupported NumPy cases
- Memory buffer length controlled by `memory_depth`
- Photon feedback modes specified via `memristive_output_modes`

---

## Gradient computation (PSR)

The Parameter-Shift Rule computes **exact** gradients without finite differences:

- Each phase parameter contributes **2n shift terms** (n = photon count)
- Phase parameters: exact PSR gradients
- Memristor weights: finite differences (non-unitary parameters)
- PSR photon count is taken from **`sum(input_state)`** for each trainable mesh phase — it must match the physical input photon number or gradients are wrong

---

## Uncertainty quantification

`Experiment.run_uncertainty_analysis` runs `n_passes` forward passes, each with Gaussian noise `~ N(0, unc_noise_std^2)` added to the phase parameters. Pass a `trained_state` to keep the learned linear head attached during every pass.

Passes run in parallel via `ProcessPoolExecutor` up to `os.cpu_count()` workers.

**Regression:**
```python
unc = exp.run_uncertainty_analysis(trained_state, enc_test, n_passes=20, noise_std=0.05)
mean = unc["mean"]   # (n_test,)
std  = unc["std"]    # (n_test,)  -- predictive uncertainty
```

**Classification:**
```python
unc = exp.run_uncertainty_analysis(trained_state, enc_test, n_passes=20, noise_std=0.05)
mean_probs = unc["mean"]   # (n_test, n_classes)
entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-15), axis=1)
```

The returned dict contains:
- `"mean"` -- mean prediction across all passes
- `"std"` -- standard deviation across all passes
- `"all_preds"` -- raw predictions from every pass

---

## Synthetic datasets

### Regression functions

Available via `src.data.get_data(n_data, sigma_noise, function_name)`:

| Name | Description |
|---|---|
| `quartic_data` | x^4 |
| `sinusoid_data` | sin(2*pi*x) * 0.5 + 0.5 |
| `multi_modal_data` | Sum of Gaussian peaks |
| `gaussian_bump_data` | Single smooth Gaussian bump |
| `step_function_data` | Smooth tanh step |
| `oscillating_poly_data` | x^3 - 0.5*x^2 + 0.1*sin(15x) |
| `damped_cosine_data` | Damped cosine wave |
| `neg_quadratic_data` | Negative quadratic |
| `neg_qubic_data` | Negative cubic |

### Classification datasets

Via `src.data.get_classification_data(n_data, data_type)`:

| `data_type` | Description |
|---|---|
| `binary_threshold` | Threshold at x=0.5 (2 classes) |
| `multi_class_regions` | Three equal regions (3 classes) |
| `sinusoidal` | Classes from sign of sin(2*pi*x) (2 classes) |

### 2D datasets

`get_two_moons_data()` generates the scikit-learn `make_moons` dataset. Use `encode_2d_to_phase()` to reduce 2D inputs to a single phase value before training.

---

## Development

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_circuits.py
```
