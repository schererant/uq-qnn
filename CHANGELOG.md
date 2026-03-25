# Changelog

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
