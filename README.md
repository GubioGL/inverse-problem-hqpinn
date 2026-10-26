# Inverse problem: learning the viscosity of Burgers' equation with a hybrid quantum PINN

Code, data and manuscript for estimating the unknown viscosity ν of
u_t + u u_x = ν u_xx with a **hybrid quantum physics-informed neural network (HQPINN)**, a PINN with a
fixed quantum layer, compared with a classical PINN trained the same way.

Main result so far (`data/Inverso_110/README.md`): ν = 0.01, only **110 epochs** with 10 collocation
points per epoch, and the quantum layer evaluated on the noisy FakeBerlin emulator (1024 shots).
Results are over 33 runs per model (3/7/10 neurons or qubits × seeds 0–10):

| Median relative error of ν | Classical PINN | HQPINN (fake QPU) |
|---|---|---|
| T = 0.1 | 7.3 % | 6.0 % (tie: p = 0.11) |
| T = 1 | **21.5 %** | 30.2 % (p = 0.046) |

The classical PINN also fits u(x, T) better in both cases. At T = 1 a shock forms before the final
time, and both models use an adjusted training: u(x,0) as a soft loss instead of a hard constraint, 10
warm-up epochs on the data only, and a plateau learning rate.

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

On Debian/Ubuntu, install the virtual-environment support once if `python3 -m
venv` reports that `ensurepip` is unavailable:

```bash
sudo apt update
sudo apt install python3-venv
```

Then, from the repository root, create and activate the environment. Use
`python3` explicitly on systems where the `python` command is not defined:

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r run/requirements.txt

python run/plots/plot_inverse_E110.py                 # 110-epoch figures -> paper/figures/
python run/Inverser_classical_pinn_E110.py --help     # example: a training run
```

Scripts can be started from any folder. Each one puts `src/` and `run/` on the import path and
resolves `data/...` from the repository root.

If the environment already exists, activate it before running a script:

```bash
source .venv/bin/activate
python run/Inverser_classical_pinn_E110.py --help
```

## Running the 110-epoch cases (ν = 0.01, T = 0.1 and T = 1)

Each script trains 3, 7 and 10 neurons (classical) or qubits (quantum) × seeds 0–10, 110 epochs with
10 random collocation points per epoch, and writes to `data/Inverso_110/<classico|quantico>/T_<T>_nu_0.01/`
(one `run_*.json`, `losshist_*.npz`, `validation_*.npz` and `model_*.pt` per run, plus `summary.json`).

**Classical** (a few seconds per run):

```bash
python run/Inverser_classical_pinn_E110.py --tfin 0.1 --nu 0.01
python run/Inverser_classical_pinn_E110.py --tfin 1 --nu 0.01
```

**Quantum on the FakeBerlin emulator** (5–15 min per run; the runs go in parallel):

1. The script imports `regression_hardware_nqubits.py`, which is not in git. For the fake only, the
   stand-in in `../CBPF-Geral/quantum_hardware/` is enough (it reads the recorded chip chains; the
   calibrations in `data/Inverso_110/quantico/` are reused). For `--real` (IBM hardware), use the
   original file, which sets `INSTANCE`.
2. In a pip `.venv`, `qiskit_aer` loads only with its own `libgomp` preloaded (otherwise: *cannot
   allocate memory in static TLS block*).

```bash
export LD_PRELOAD=$(ls .venv/lib/python3.*/site-packages/qiskit_aer.libs/libgomp*.so*)
export PYTHONPATH=../CBPF-Geral/quantum_hardware
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01 --workers 12
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 1 --nu 0.01 --workers 12
```

`--workers 12` fits a 16-core, 16 GB machine; each case (33 runs) takes about 35–45 min. Both scripts
take `--seeds` (default 0–10); the quantum one also `--qubits`.

**What changes at T = 1.** With ν = 0.01 a shock forms before T = 1. For this case only
(`ADJUSTED_CASES` in `run/Inverser_classical_pinn_E110.py`), both scripts apply three adjustments
(`ADJUSTMENTS`, the same for both models), with no extra flag:

| Adjustment | T = 0.1 (unchanged) | T = 1 |
|---|---|---|
| Initial condition | hard: u = sin(2πx) + (t/T)·network | soft: u = network, plus an MAE loss on 10 random points at t = 0 per epoch (weight 10) |
| First epochs | the residual weight ramps up from epoch 1 | 10 epochs with the data loss only and ν frozen, then the ramp |
| Learning rate of the network | cosine decay | constant, divided by 1.5 after 5 epochs without a 1 % drop of the loss |

On the quantum side, the t = 0 points are measured in the same job as the collocation points, so the
number of jobs (110) does not change. `--treino-antigo` runs T = 1 without the adjustments (into
`treino_antigo/`). Why these and not others: `data/Inverso_110/README.md`.

**Figures** (into `paper/figures/`):

```bash
python run/plots/plot_inverse_E110.py
```

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
  `src/quantum_hardware/` is git-ignored. For the fake QPU, a stand-in of `regression_hardware_nqubits.py`
  in `../CBPF-Geral/quantum_hardware/` works (see *Running the 110-epoch cases*).

## Origin

Split from [GubioGL/CBPF-Geral](https://github.com/GubioGL/CBPF-Geral) (commit `5776ab2`), which keeps
the full history and serves as the archive. The forward problem lives in
[GubioGL/forward-problem-hqpinn](https://github.com/GubioGL/forward-problem-hqpinn).
