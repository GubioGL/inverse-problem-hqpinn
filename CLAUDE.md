# Notes for Claude

Inverse problem of the HQPINN project: learn the viscosity ν of Burgers' equation with a PINN that has a
fixed quantum layer, against a classical PINN. The author writes notes in Portuguese; code comments and
the paper are in English. Companion repo: GubioGL/forward-problem-hqpinn (same `src/main/` package).

## Layout rules
- `src/` = importable library (`main/` package, `paths.py`). `run/` = entry points; they may import each other
  (e.g. `Inverser_quantum_hardware_pinn.py` imports `Inverser_quantum_exact_pinn_paralelo.py`).
- Every script in `run/` starts with the repository header block (puts `run/`, `src/`, `src/quantum_hardware/`
  and the script's own folder on `sys.path`, then `chdir`s to the repo root). Keep it when adding scripts.
- Outputs of training runs go under `data/Inverso/...` or `data/Inverso_110/...`.
- Figures: always write to `paper/figures/` as PDF via `paths.save_figure(fig, name)`. Never commit PNGs.

## Paper / Overleaf
- `paper/` is synced two-way with Overleaf project `6ad4df833716c9db1f4aabee` by
  `.github/workflows/overleaf-sync.yml` (script: `.github/scripts/overleaf_sync.sh`, state in
  `.github/overleaf-sync-state` — do not edit by hand).
- To change the paper: edit `paper/` and push to `main`; the workflow pushes to Overleaf.
- Branch `overleaf-conflict` = unresolved same-line edits; merge it into `main` and push.

## Not in the repository
- `src/quantum_hardware/regression_hardware_nqubits.py` and `run/benchmark_viscosidade.py` (local only).
- `losshist_*.npz` of `data/Inverso/{classical_inverse,quantum_inverse}`: `data/fetch_full_histories.sh`.
  Use the `histories_summary.csv` in each folder for final ν and loss values.
