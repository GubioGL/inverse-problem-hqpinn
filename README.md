# Inverse problem: learning the viscosity of Burgers' equation with a hybrid quantum PINN

Code, data and manuscript for estimating the unknown viscosity ν of
u_t + u u_x = ν u_xx with a **hybrid quantum physics-informed neural network (HQPINN)**, a PINN with a
fixed quantum layer, compared with a classical PINN trained the same way.

Main result so far (`data/Inverso_110/README.md`): with only **110 epochs** and the quantum layer
evaluated on the noisy FakeBerlin emulator (1024 shots), the HQPINN estimates ν = 0.1 with a median
relative error of **1.5 %** on the test seeds versus 8.4 % for the classical PINN. On 60 validation
runs the medians are 2.2 % versus 4.0 %.

## Repository layout

```text
inverse-problem-hqpinn/
├── src/                     shared library code
│   ├── main/                PINN models, quantum circuits, physics, training loops
│   └── paths.py             repository paths + save_figure()
├── run/                     everything you execute
│   ├── requirements.txt
│   ├── Inverser_classical_pinn*.py           classical PINN (single, parallel sweep, 110 epochs)
│   ├── Inverser_quantum_pinn*.py             quantum PINN, PennyLane
│   ├── Inverser_quantum_exact_pinn_paralelo.py  exact quantum layer, parallel sweep
│   ├── Inverser_quantum_hardware_pinn*.py    quantum layer on FakeBerlin / IBM hardware
│   ├── run_quantum_hardware.py               hardware helpers shared with the forward problem
│   ├── diagnostico_checkpoints.py            checkpoint diagnostics
│   └── plots/                                figures -> paper/figures/
├── data/
│   ├── Inverso/             long-training sweeps (classical_inverse, quantum_inverse)
│   └── Inverso_110/         110-epoch comparison, classical vs FakeBerlin (see its README)
├── paper/                   LaTeX manuscript, synced with Overleaf
│   └── figures/
└── .github/                 Overleaf sync workflow
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r run/requirements.txt

python run/plots/plot_inverse_E110.py                  # 110-epoch figures -> paper/figures/
python run/Inverser_classical_pinn_E110.py --help      # example: a training run
```

Scripts can be started from any folder. Each one puts `src/` and `run/` on the import path and
resolves `data/...` from the repository root.

## Figures go straight into the paper

Every plotting script writes its PDF into `paper/figures/`, the folder that LaTeX reads and that is
synced with Overleaf. For new plots:

```python
from paths import save_figure

save_figure(fig, "nu_vs_epoch")   # -> paper/figures/nu_vs_epoch.pdf (+ a local .png preview)
```

## Paper and Overleaf

`paper/` is synced in both directions with the Overleaf project through Overleaf's Git integration
(`.github/workflows/overleaf-sync.yml`):

| Direction | When |
|---|---|
| GitHub → Overleaf | every push to `main` that changes `paper/` |
| Overleaf → GitHub | every 3 hours, or right away from **Actions → Overleaf sync → Run workflow** |

Edits to different parts of the text on both sides are merged automatically. If the same lines were
edited in both places, the run stops and puts the Overleaf version on the branch `overleaf-conflict`.
Merge it into `main`, push, and the sync completes.

**One-time setup:** in Overleaf, open *Account Settings → Git integration* and generate a token. Here,
open *Settings → Secrets and variables → Actions → New repository secret*, name it `OVERLEAF_TOKEN` and
paste the token. Tokens expire after one year.

`paper/` starts with a short starter manuscript (`main.tex`, `references.bib`) and the 110-epoch
figures. On the first sync:

- if the Overleaf project already holds a draft, the draft replaces the starter, and the figures are
  kept and sent to Overleaf;
- if the Overleaf project is still blank, the starter goes to Overleaf.

No branch is created; every earlier Overleaf version stays in Overleaf's *History*.

## Data

All result files the scripts read are included, with two exceptions:

- **Per-epoch loss histories of the long sweeps** (`data/Inverso/{classical_inverse,quantum_inverse}/**/losshist_*.npz`,
  about 2 GB). Each folder has a `histories_summary.csv` with the final and minimum loss and ν of
  every run. The full histories are in [GubioGL/CBPF-Geral](https://github.com/GubioGL/CBPF-Geral);
  `bash data/fetch_full_histories.sh` copies them back.
- **Files kept on your own machine:** `src/quantum_hardware/` (with `regression_hardware_nqubits.py`)
  and `run/benchmark_viscosidade.py`. The hardware scripts and `diagnostico_checkpoints.py` need them.
  `src/quantum_hardware/` is git-ignored.

## Origin

Split from [GubioGL/CBPF-Geral](https://github.com/GubioGL/CBPF-Geral) (commit `5776ab2`), which keeps
the full history and serves as the archive. The forward problem lives in
[GubioGL/forward-problem-hqpinn](https://github.com/GubioGL/forward-problem-hqpinn).
