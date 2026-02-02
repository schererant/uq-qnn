# Uncertainty Quantification with Quantum Neural Networks on Integrated Photonic Circuits

#TODO:
- coverage is not calculated correctly



## Overview

This repository implements a modular framework for training photonic quantum neural networks with uncertainty quantification capabilities. The code is based on the photonic quantum memristor paper and provides both discrete-phase and continuous-swipe implementations for circuit simulation.

## Modular Structure

The codebase is organized into the following modules:

- `src/autograd.py`: Implements parameter-shift rule (PSR) for photonic circuits (regression and classification)
- `src/circuits.py`: Circuit construction for encoding and memristor components (array-based parameter design)
- `src/data.py`: Data generation and processing utilities (regression and classification datasets)
- `src/loss.py`: Custom loss functions and PyTorch model implementations (MSE and cross-entropy)
- `src/simulation.py`: Circuit simulation with discrete and continuous modes
- `src/training.py`: Training algorithms with PyTorch optimization
- `src/utils.py`: Configuration and utility functions

## Installation

```bash
# Clone the repository
git clone https://github.com/username/uq-qnn.git
cd uq-qnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to train the quantum neural network:

```bash
python main.py --n-samples 1000
```

### Key Options

- `--continuous`: Use continuous-swipe training mode
- `--n-samples`: Number of samples for circuit simulation
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--measured-data`: Path to measured data pickle file
- `--datafunction`: Synthetic data function to use (see below)
- `--n-phases`: Number of phase parameters in the memristor circuit (default: 2)
- `--circuit-type`: Circuit architecture to use ('memristor' or 'clements')
- `--n-modes`: Number of modes for Clements architecture (default: 3)
- `--encoding-mode`: Mode to apply encoding to (default: 0)
- `--target-mode`: Target output mode(s) as comma-separated list (e.g., '2,3')

See `python main.py --help` for all available options.

### Synthetic Data Functions

The framework includes multiple synthetic data functions for regression tasks:

- `quartic_data`: Standard x⁴ function
- `sinusoid_data`: Sinusoidal function (sin(2πx) * 0.5 + 0.5)
- `multi_modal_data`: Sum of Gaussian peaks
- `step_function_data`: Smooth step function using tanh
- `oscillating_poly_data`: Oscillating polynomial (x³ - 0.5x² + 0.1sin(15x))
- `damped_cosine_data`: Damped cosine wave

Run the `examples/function_comparison.py` script to compare model performance across all functions.

### Example Scripts

The repository includes several example scripts:

**Regression Examples:**
1. `examples/simple_regression.py` - Basic regression with uncertainty quantification
2. `examples/function_comparison.py` - Compare performance across different synthetic functions
3. `examples/circuit_comparison.py` - Compare memristor vs. Clements circuit architectures

**Classification Examples:**
4. `examples/simple_classification.py` - Binary classification with uncertainty quantification
5. `examples/multi_class_classification.py` - Multi-class classification (3+ classes) demonstration
6. `examples/two_moons_classification.py` - Two Moons (half-moons) 2D classification dataset

For the Clements architecture example, run:
```bash
python examples/circuit_comparison.py
```

This script demonstrates how to use both architectures on the same dataset and compares their performance.

### Circuit Architectures

The framework supports two distinct circuit architectures:

#### 1. Memristor Architecture

The photonic memristor circuit implementation uses an array-based approach:

- `encoding_circuit`: Builds a 2-mode encoding circuit with a phase shifter
- `memristor_circuit`: Takes an array of phases instead of individual parameters
- `build_circuit`: Combines encoding and memristor circuits with array-based parameters

Parameter structure for memristor circuit:
```
params = [phi1, phi3, w]
```
where `phi1` and `phi3` are phase parameters and `w` is the memory weight parameter.

#### 2. Clements (Rectangular) Architecture

The Clements architecture provides a more flexible, scalable approach:

- Configurable number of modes (use `--n-modes` option)
- Mesh of Mach-Zehnder Interferometers (MZIs) in a rectangular grid pattern
- Each MZI has two phase shifters (internal and external)
- Supports arbitrary-sized photonic neural networks

Parameter structure for Clements circuit:
```
params = [phi1_int, phi1_ext, phi2_int, phi2_ext, ..., phiN_int, phiN_ext, w]
```
where each MZI has an internal phase (`phi_int`) and external phase (`phi_ext`), and `w` is the memory weight parameter.

The number of phase parameters is automatically calculated as `n_modes * (n_modes - 1)` based on the number of modes.

> **Note:** For Clements architecture, you must ensure that `n_modes` ≥ 2 and `encoding_mode` < `n_modes`. The target mode(s) must also be valid for the given number of modes.

## Classification Tasks

The framework now supports both regression and classification tasks using the Photonic Parameter Shift Rule (PSR) for exact gradient computation.

### Classification Usage

For binary classification (2 classes):
```python
from src.data import get_classification_data
from src.training import train_pytorch

# Generate binary classification data
X_train, y_train, X_test, y_test = get_classification_data(
    n_data=80,
    n_classes=2,
    data_type='binary_threshold',
    noise_level=0.05,
    return_one_hot=False
)

# Train with cross-entropy loss
theta, history = train_pytorch(
    X_train, y_train,
    loss_type='cross_entropy',
    n_classes=2,
    target_mode=(1, 2),  # Use modes 1 and 2 for binary classification
    epochs=30,
    n_samples=500
)
```

For multi-class classification (3+ classes):
```python
# Generate multi-class data
X_train, y_train, X_test, y_test = get_classification_data(
    n_data=100,
    n_classes=3,
    data_type='multi_class_regions',
    noise_level=0.05,
    return_one_hot=False
)

# Train with Clements architecture (more modes needed)
theta, history = train_pytorch(
    X_train, y_train,
    circuit_type='clements',
    n_modes=4,  # Need at least n_classes modes
    loss_type='cross_entropy',
    n_classes=3,
    target_mode=(0, 1, 2),  # Use first 3 modes for 3 classes
    epochs=40,
    n_samples=500
)
```

### Classification Data Types

The framework supports several classification datasets:

**1D Synthetic Datasets** (via `get_classification_data()`):
- `'binary_threshold'`: Simple threshold at x=0.5 (2 classes only)
- `'multi_class_regions'`: Three regions [0,0.33], [0.33,0.66], [0.66,1.0] (3 classes only)
- `'sinusoidal'`: Classes based on sin(2πx) sign (2 classes only)

**2D Datasets**:
- **Two Moons** (via `get_two_moons_data()`): Classic 2D binary classification dataset with two interleaving half-circles. Uses `sklearn.datasets.make_moons` under the hood. The 2D features are encoded to a single phase value using `encode_2d_to_phase()` with methods:
  - `'weighted_sum'`: Linear combination of both dimensions (default)
  - `'first_dim'`: Use only first dimension
  - `'radial'`: Use radial distance from center

### Classification PSR

The classification PSR implements Equation (15) from the paper:

```
∂L/∂θ = -(1/K) Σ_q c_q Σ_c (y_c / F^c_Θ(x)) · F^c_{Θ+Θ^q}(x)
```

where:
- `F^c_Θ(x)` is the probability for class `c`
- `y_c` is the one-hot encoded target for class `c`
- The same shift angles and coefficients are used as in regression PSR

### Uncertainty Quantification for Classification

For classification tasks, uncertainty is quantified using:
- **Entropy**: `H = -Σ_c p_c * log(p_c)` where `p_c` is the predicted probability for class `c`
- **Multiple forward passes**: Similar to regression, run multiple passes with parameter perturbations
- **Class probability variance**: Standard deviation of class probabilities across forward passes

See `examples/simple_classification.py`, `examples/multi_class_classification.py`, and `examples/two_moons_classification.py` for complete examples.

### Two Moons Dataset Example
Taking implementation from https://github.com/lightning-uq-box/lightning-uq-box/blob/main/lightning_uq_box/datamodules/toy_half_moons.py


The Two Moons dataset is a classic 2D binary classification benchmark:

```python
from src.data import get_two_moons_data, encode_2d_to_phase
from src.training import train_pytorch_generic

# Generate Two Moons dataset
X_train, y_train, X_test, y_test = get_two_moons_data(
    n_samples=1000,
    noise=0.1,
    random_state=42,
    return_one_hot=False
)

# Encode 2D features to phase values
enc_train = encode_2d_to_phase(X_train, method='weighted_sum')
enc_test = encode_2d_to_phase(X_test, method='weighted_sum')

# Train with cross-entropy loss
theta, history = train_pytorch_generic(
    enc_train, y_train,
    loss_type='cross_entropy',
    n_classes=2,
    target_mode=(1, 2),
    epochs=50,
    n_samples=500
)
```

The example includes visualization of the decision boundary and uncertainty maps.

## Tasks

- initial study of QNNs with UQ in simulation, use 1-d regression function from photonic quantum memristor paper from Iris (ask Iris about simulations)

- use existing QNNs works for regression and classification to check if "inherent Quantum"-UQ adds a benefit:
    - **classification task:** over multiple forward passes compute the mean of logits and then take the softmax, as e.g. in [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L166)
    - compute Entropy over softmax outputs as in [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L186)
    - on a validation set (it should kinda all be iid and similar splits) compute the quantiles of the entropies (per prediction) as in [Pandas Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html). An example is given in [Link](https://github.com/nilsleh/tropical_cyclone_uq/blob/main/src/class_results_analysis.ipynb) - code box 12, titled "Selective Prediction"

:::info
**Selective prediction in a nutshell:**  UQ evaluation with selective prediction, as introduced in [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf). Here, samples with with a predictive uncertainty **(classification: entropies, regression: standard deviation)** above a given threshold are omitted from prediction and referred to an expert and optionally another method. If the corresponding UQ method has higher uncertainties for inaccurate predictions, leaving out the predictions for these samples should increase the overall accuracy. This could resemble a deployment scenario, where predictions are monitored and if the predictive uncertainties surpass a given threshold, the sample is referred to an expert and/or additionally evaluated with another method. Instead of a fixed threshold on the predictive uncertainties across methods, one can chose a UQ specific threshold based on the 0.8 quantile of predictive uncertainties computed on a held out validation dataset for each method. These method-specific thresholds are then utilized on the separate test set for which we report results.

:::

- **regression task:** over multiple forward passes compute the mean of predictions [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L118) and standard deviation, as e.g. in  [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L151)
-  on a validation set (it should kinda all be iid and similar splits) compute the quantiles of the standard deviations (per prediction) as in [Pandas Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html). An example is given in [Link](https://github.com/nilsleh/tropical_cyclone_uq/blob/main/src/results_tropical.ipynb) - code box 6, titled "selective prediction thresholds based on validation set"


- compute accuracy according to different quantile UQ thresholds on test datasets, plot should look like:

![](https://s3.desy.de/hackmd/uploads/b40060f6-6324-4a3a-bbfa-5f042bf76474.png)
Above is the RMSE with Selective Prediction Results across Quantiles for the Tropical Cyclone Dataset.
