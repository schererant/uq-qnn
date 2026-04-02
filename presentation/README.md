# Presentation Assets

This folder contains:

- `repository_overview.tex` - article-style LaTeX document describing the repository, its goals, development stages, changes, and rerun experiment analysis
- `rerun_experiments_slides.tex` - Beamer presentation focused on the rerun experiment results

Build from the repository root or from this directory:

```bash
cd presentation
pdflatex repository_overview.tex
pdflatex rerun_experiments_slides.tex
```

The sources reference figures already generated under `../reports/`.
