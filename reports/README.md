# Example run outputs

Running any script under `examples/` writes artifacts here in a consistent layout:

```
reports/<example_script_name>/<YYYY-mm-dd_HHMMSS>/
  run_summary.json    # metadata, optional metrics, artifact paths
  *.png               # figures (and other files produced by that example)
```

The `run_summary.json` file uses schema `uq-qnn.example_run.v1` and includes UTC time, Python version, and a short git commit when available.

Generated run folders are gitignored; this file and `.gitkeep` remain in version control so the directory exists in clones.
