# Design Note: Occupation-Vector Inputs and N-Fold Coincidence Readout

## Why this change exists

The codebase currently uses two different meanings for `input_state`:

- In the training and simulation stack, coincidence mode treats it as a tuple of occupied input mode indices such as `(1, 4)`.
- In the circuit-level API, the natural representation is already a Fock occupation vector such as `(0, 1, 0, 0, 1, 0)`.

This mismatch creates three problems:

1. The public API is inconsistent.
2. The training stack is implicitly hard-wired to the special case of two photons entering distinct modes.
3. Readout semantics become unclear as soon as we want `sum(input_state) > 2`.

The goal of this change is to unify the input representation, preserve the fast paths that already work well, and define multi-photon readout in a way that is both hardware-realistic and still trainable.

---

## What we want to optimize for

We want two things at once:

1. Simulate the intended hardware as closely as possible.
2. Keep simulations fast enough to remain useful for training and sweeps.

Those goals are in tension. Full multi-photon Fock readout is more general, but it is slower and not obviously the right abstraction for the current hardware layer. The existing hardware abstractions already lean toward coincidence counting with working detectors, detector noise, and coincidence windows rather than full photon-number-resolving hardware.

That makes the real design question:

> What should `output_mode="coincidence"` mean once the input is allowed to contain an arbitrary number of photons?

---

## Current observations from the codebase

- The training stack still validates coincidence mode as exactly two distinct input modes in [`src/config.py`](../src/config.py).
- The NumPy backend has a very fast two-photon collision-free path in [`src/numpy_backend.py`](../src/numpy_backend.py).
- The Perceval runner also assumes two-photon coincidence channels indexed by detector pairs in [`src/simulation/runner.py`](../src/simulation/runner.py).
- PSR metadata currently infers photon count from `output_mode`, not from the actual total photon number.
- The circuit-level helper in [`src/circuit.py`](../src/circuit.py) already contains useful building blocks:
  - occupation-vector inputs,
  - expansion from occupation vectors to repeated input rows,
  - output-state enumeration,
  - Ryser permanents for general multi-photon transition probabilities.

One important nuance:

- `PhotonicCircuit.coincidences(..., detector_mode="click")` is not a full threshold-detector aggregation over all PNR outputs.
- It enumerates collision-free binary output patterns with exactly `N` clicks.

That is acceptable for **N-fold coincidence postselection**, but it is not the same thing as a general threshold-detector output model.

---

## Options considered

### Option 1: Keep pair-style `input_state` and patch special cases

This is the weakest option.

- It preserves the current ambiguity.
- It does not scale to repeated occupancy such as `(2, 0, 0, 0, ...)`.
- It keeps the training stack structurally tied to two-photon coincidence channels.

Decision: reject.

### Option 2: Use occupation vectors everywhere, but keep coincidence readout two-photon-only

This improves input semantics, but it does not solve the actual design problem.

- We would still have no clear story for `sum(input_state) > 2`.
- PSR, channel indexing, and training outputs would remain special-cased.

Decision: reject.

### Option 3: Generalize training to full PNR output distributions

This is the most general physical model, but it is not the best fit here.

Pros:

- Physically complete for indistinguishable bosons.
- Natural if the target hardware has genuine PNR detectors.

Cons:

- Channel count grows as `C(W + N - 1, N)` over `W` working detectors.
- Much slower than the existing two-photon fast path.
- Less aligned with the current coincidence-oriented hardware layer.
- Harder to use cleanly for regression/classification tasks.

Decision: keep available for circuit-level analysis, but do not make this the training readout.

### Option 4: Generalize training to N-fold postselected coincidence channels

This keeps the spirit of the current system and gives the best speed/fidelity tradeoff.

Pros:

- Matches the current meaning of coincidence counting: distinct detector clicks inside a coincidence window.
- Preserves the current two-photon semantics as the `N = 2` case.
- Channel count grows as `C(W, N)`, which is much smaller than full PNR.
- Compatible with current `working_detectors`, postselection, dark-count, and coincidence-window abstractions.
- Lets us preserve the current optimized 2-photon path.

Cons:

- It is a postselected coincidence model, not a full threshold-detector model.
- It deliberately discards bunching outcomes from the training readout basis.

Decision: adopt.

---

## Decision

The training and simulation stack should generalize to **N-fold postselected coincidence readout**, not to full PNR readout.

Concretely:

- `input_state` becomes a Fock occupation vector of length `n_modes` everywhere.
- Photon number is always `sum(input_state)`.
- `output_mode="singles"` remains the single-photon path and is valid only when `sum(input_state) == 1`.
- `output_mode="coincidence"` becomes the **N-fold coincidence path** and is valid when `sum(input_state) >= 2`.
- Coincidence channels are **collision-free click patterns** over `working_detectors`, i.e. binary occupation vectors with exactly `N` ones.
- Full PNR output remains available at the circuit level for analysis and validation, but not as the default training abstraction.

This is the best compromise between hardware realism and runtime.

---

## Semantics after the redesign

### 1. Input state

`input_state` is always:

- a tuple of length `n_modes`,
- each entry is a non-negative integer,
- the sum is the total photon number.

Examples:

- Singles: `(0, 0, 1, 0, 0, 0)`
- Two photons in distinct inputs: `(0, 1, 0, 0, 1, 0)`
- Two photons in the same input mode: `(2, 0, 0, 0, 0, 0)`
- Three photons: `(1, 1, 0, 0, 1, 0)`

Legacy pair-style inputs such as `(1, 4)` should be rejected with a migration-focused error message that points to the occupation-vector form.

### 2. Photon count and distinguishability

Let `n_photons = sum(input_state)`.

- If `n_photons == 1`, `photon_distinguishability` must be `None`.
- If `n_photons >= 2`, `photon_distinguishability` is required.
- In the first generalized release, the NumPy backend should still support only `"indistinguishable"` multi-photon simulation.
- Distinguishable or partially distinguishable multi-photon simulation should remain future work or Perceval-only.

### 3. Coincidence channel basis

For `output_mode="coincidence"` with `N = sum(input_state)`:

- choose the detector subset `working_detectors`,
- enumerate all size-`N` subsets of those detectors,
- convert each subset into a binary occupation vector of length `n_modes`,
- compute the probability of each such collision-free output pattern,
- postselect and renormalize over exactly those channels.

This is the `N`-photon generalization of the current two-photon pair basis.

Channel count:

- coincidence basis: `C(W, N)`
- full PNR basis: `C(W + N - 1, N)`

where `W = len(working_detectors)`.

Example for `W = 6`, `N = 3`:

- N-fold coincidence channels: `C(6, 3) = 20`
- PNR channels: `C(8, 3) = 56`

This difference is exactly why the coincidence basis is the better training default.

### 4. Regression semantics

For coincidence regression:

- the scalar prediction is the postselected probability of one explicit N-fold coincidence channel.

To minimize API churn, keep `target_mode` for this purpose, but redefine it as:

- a tuple of detector indices,
- length exactly `N`,
- all entries distinct,
- all entries contained in `working_detectors`.

Examples:

- `N = 2`: `target_mode = (0, 4)`
- `N = 3`: `target_mode = (0, 2, 5)`

This is the direct generalization of the current pair readout.

### 5. Classification semantics

For coincidence classification:

- the output vector is the postselected distribution over all admissible N-fold coincidence channels from `working_detectors`,
- channel order is canonical lexicographic order of the detector-index tuples,
- `n_classes` must equal the number of such channels, i.e. `C(W, N)`.

For this mode:

- `target_mode` should not be used to define the class basis,
- the class basis should be derived from `working_detectors` and `N`.

This requires a cleanup in training-side validation, because current classification checks are singles-shaped and assume `target_mode` defines the classes.

### 6. PSR semantics

PSR photon counts must come from the physical input, not from `output_mode`.

Therefore:

- `psr_photon_counts_for_phases()` should use `sum(input_state)`.

This is required for correctness.

---

## Backend strategy

### Shared helper layer

The generalized multi-photon logic should not be duplicated separately in each backend.

Extract shared helpers for:

- expanding occupation vectors to repeated mode indices,
- enumerating N-fold coincidence channels from `working_detectors`,
- converting detector-index tuples to labels and indices,
- mapping a coincidence readout target to its channel index.

The existing helper code in [`src/circuit.py`](../src/circuit.py) is the right starting point.

### NumPy backend

Keep the current fast path:

- `N = 2`,
- collision-free two-photon coincidence channels,
- non-memristive,
- discrete mode,
- indistinguishable photons.

Add a generalized exact path for:

- arbitrary `N >= 2`,
- occupation-vector inputs,
- N-fold coincidence outputs,
- Ryser permanent per output channel.

It is important to be precise here:

- for `N = 2`, the current implementation is a true fast path because the collision-free permanent reduces to a simple vectorized closed form,
- for `N > 2`, exact simulation remains feasible in NumPy, but it is no longer a comparably cheap fast path,
- for `N = 3`, a specialized exact path may still be worth implementing because it is likely to remain practical for small-mode experiments,
- for `N >= 4`, the NumPy path should be treated as exact but increasingly expensive,
- for large `N`, this must be presented as a guarded slow path rather than a fast path.

Even so, the N-fold coincidence basis remains preferable to full PNR as the default training readout, because the number of output channels is still much smaller.

### Perceval backend

Construct `BasicState(input_state)` directly from the occupation vector.

Then:

- extract only the N-fold coincidence output states from the SLOS result,
- apply postselection and optional noise in the same canonical channel order as the NumPy backend.

### Guardrails

Performance degrades rapidly with photon number because permanent evaluation is exponential in `N`.

We should therefore keep explicit guardrails:

- document expected scaling,
- preserve the 2-photon optimized path,
- treat `N = 3` as the only plausible next candidate for a specialized exact NumPy optimization,
- distinguish clearly in docs and errors between "exact supported" and "fast supported",
- warn or fail clearly for cases that exceed practical limits,
- do not silently route large-N training jobs into extremely slow paths.

Training-time cost must be called out explicitly:

- regression is cheaper because it only needs one readout channel,
- classification is more expensive because it needs the full `C(W, N)` coincidence distribution,
- PSR multiplies the simulation cost by `2N` per trainable phase,
- so an `N > 2` forward path that is acceptable for analysis may still be too expensive for gradient-based training.

---

## Implications for hardware realism

This design matches the current hardware layer better than full PNR training would.

Why:

- coincidence windows already exist in [`src/coincidence.py`](../src/coincidence.py),
- working detectors already model detector availability,
- dark-count and Gaussian/shot noise are already phrased in terms of detector channels,
- the current coincidence workflow is built around postselected channel distributions, not raw Fock tomography.

So the right interpretation is:

- we are modeling a coincidence-counting experiment with postselection over valid N-fold detector events,
- not a general detector array returning the full threshold-collapsed output distribution,
- and not a genuine PNR instrument unless we explicitly opt into that mode later.

If a future hardware target uses true PNR detectors, that should be introduced as a separate readout mode, not folded into coincidence mode.

---

## Non-goals for this change

This redesign should **not** try to solve everything at once.

Out of scope for the first pass:

- legacy pair-format support,
- full threshold-detector aggregation over all click cardinalities,
- full PNR training readout,
- partial distinguishability models,
- memristive coincidence generalization.

Those can be added later once the occupation-vector and N-fold coincidence semantics are stable.

---

## Recommended implementation order

1. Re-spec `SimConfig` and `CircuitConfig` so `input_state` is always an occupation vector.
2. Replace all `len(input_state)` photon semantics with `sum(input_state)`.
3. Add shared helpers for N-fold coincidence channel enumeration and indexing.
4. Update NumPy and Perceval backends to decode occupation vectors and emit N-fold coincidence channels.
5. Fix training, loss, and autograd validation so coincidence classification derives its class basis from `working_detectors`.
6. Keep the existing 2-photon NumPy fast path intact.
7. Update examples, tests, README, and changelog.

---

## Final recommendation

Unify the entire stack on occupation-vector inputs and define coincidence mode as **postselected N-fold coincidence readout over working detectors**.

This choice:

- fixes the API inconsistency,
- preserves the meaning of the current two-photon experiments,
- scales naturally to repeated occupancy and `N > 2`,
- matches the present hardware abstractions,
- keeps the most important simulation fast path intact,
- and gives clean semantics for both regression and classification.

That should be the design target for the migration.
