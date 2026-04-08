# Presentation Assets

This folder contains:

- `repository_overview.tex` - article-style LaTeX document describing the repository, its goals, development stages, changes, and rerun experiment analysis
- `rerun_experiments_slides.tex` - Beamer presentation focused on the rerun experiment results

The TeX sources are kept in line with the current training API: **`input_state` as a length-`n_modes` occupation vector**, **N-fold coincidence** over `working_detectors` (`C(W, N)` channels), **`PhotonicCircuit` + `CircuitConfig`**, and **PSR** scaling with **`sum(input_state)`** (see root `CHANGELOG.md` and `README.md`).

Build from the repository root or from this directory:

```bash
cd presentation
pdflatex repository_overview.tex
pdflatex rerun_experiments_slides.tex
```

The sources reference figures already generated under `../reports/`.
