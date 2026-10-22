"""Summary figures of the 110-epoch inverse runs (Inverser_quantum_hardware_pinn_E110.py and
Inverser_classical_pinn_E110.py), made only from the files those scripts save -- nothing is trained.

For each case folder <case> = T_<T>[_nu_<nu>] in OUT_DIR/quantico/ and OUT_DIR/classico/, every
figure saved in OUT_DIR:
  * nu_summary_<model>_<case>_E110.png: learned nu vs epoch, one panel per size (qubits or
    neurons) with all the seeds done, one line per seed (losshist_*.npz);
  * comparacao_classico_quantico_<case>_E110.png (when both models have the case): the run of
    each model with the smallest nu error, learned nu vs epoch and u(x, T) (validation_*.npz).

Usage (any env with numpy + matplotlib):
    python plot_inverse_E110.py                        # every case found in OUT_DIR
    python plot_inverse_E110.py --tfin 0.1             # only T = 0.1, nu = 0.1
    python plot_inverse_E110.py --tfin 0.1 --nu 0.01   # only T_0.1_nu_0.01
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _ROOT / "src" / "quantum_hardware", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------
from paths import save_figure  # noqa: E402

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join("data", "Inverso_110")
N_COLLOCATION = 10
EPOCHS = 110
# model folder -> (key of the size in run_*.json, prefix of the size in the file names, size label)
MODELS = {"quantico": ("n_qubits", "NQ", "qubits"), "classico": ("neurons", "NE", "neurons")}

BLUE, ORANGE, GREEN, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a64"   # categorical slots 1-3, fixed order


def case_tag(tfin, nu):
    """T_<tfin>, plus _nu_<nu> when nu is not the reference 0.1 (as in the run scripts)."""
    return f"T_{tfin}" if nu == 0.1 else f"T_{tfin}_nu_{nu:g}"


def relative_l2_error(u_pred, u_ref):
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))


def run_tag(run, size_key, prefix):
    return f"{N_COLLOCATION}_{prefix}{run[size_key]}_{run['seed']}_E{EPOCHS}"


def load_runs(folder, model):
    """run_*.json of `folder` whose losshist and validation files exist (the .npz are not in git)."""
    size_key, prefix, _ = MODELS[model]
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(folder, f"run_*_E{EPOCHS}.json"))]
    have = [r for r in runs if all(os.path.exists(os.path.join(folder, f"{kind}_{run_tag(r, size_key, prefix)}.npz"))
                                   for kind in ("losshist", "validation"))]
    if len(have) < len(runs):
        print(f"  {folder}: {len(runs) - len(have)} of {len(runs)} runs without losshist/validation .npz -- skipped")
    return have


def style(ax):
    ax.grid(True, linestyle="--", alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ----------------------------------------------------------------------
# nu vs epoch of every run of one model
# ----------------------------------------------------------------------
def plot_nu_summary(case, model, nu_true):
    """One panel per size with all the seeds done, one line per seed."""
    folder = os.path.join(OUT_DIR, model, case)
    size_key, prefix, size_label = MODELS[model]
    runs = load_runs(folder, model)
    seeds = sorted({r["seed"] for r in runs})
    sizes = sorted(k for k in {r[size_key] for r in runs}
                   if sum(r[size_key] == k for r in runs) == len(seeds))
    if not sizes:
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.6), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for color, s in zip((BLUE, ORANGE, GREEN), seeds):
            tag = run_tag({size_key: size, "seed": s}, size_key, prefix)
            nu = np.load(os.path.join(folder, f"losshist_{tag}.npz"))["nu"]
            ax.plot(np.arange(1, len(nu) + 1), nu, color=color, linewidth=2.0, label=f"seed {s}: nu = {nu[-1]:.4f}")
        ax.axhline(nu_true, color=INK, linestyle="--", linewidth=1.2, label=f"target {nu_true:g}")
        ax.set_yscale("log")
        ax.set_title(f"{size} {size_label}", fontsize=12)
        ax.set_xlabel("Epochs")
        style(ax)
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("learned nu")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"nu_summary_{model}_{case}_E{EPOCHS}.png")
    path = save_figure(fig, path)
    plt.close(fig)
    print(f"  {path}")


# ----------------------------------------------------------------------
# Comparison: best classical run vs best quantum run
# ----------------------------------------------------------------------
def best_run(folder, model):
    """Run of `folder` with the smallest nu error, with its nu history and u(x, T)."""
    size_key, prefix, _ = MODELS[model]
    best = min(load_runs(folder, model), key=lambda r: r["nu_rel_err"])
    tag = run_tag(best, size_key, prefix)
    val = np.load(os.path.join(folder, f"validation_{tag}.npz"))
    u = val["u_qpu"] if "u_qpu" in val.files else val["u_pred"]
    return best, np.load(os.path.join(folder, f"losshist_{tag}.npz"))["nu"], val["x"], u, val["u_exact"]


def plot_comparison(case, tfin, nu_true):
    classical, nu_c, x, u_c, u_ref = best_run(os.path.join(OUT_DIR, "classico", case), "classico")
    quantum, nu_q, x_q, u_q, _ = best_run(os.path.join(OUT_DIR, "quantico", case), "quantico")
    l2_c, l2_q = relative_l2_error(u_c, u_ref), relative_l2_error(u_q, u_ref)
    label_c = (f"classical ({classical['neurons']} neurons, seed {classical['seed']}): "
               f"nu = {classical['nu']:.5f} ({100 * classical['nu_rel_err']:.2f}%)")
    label_q = (f"quantum, fake QPU ({quantum['n_qubits']} qubits, seed {quantum['seed']}): "
               f"nu = {quantum['nu']:.5f} ({100 * quantum['nu_rel_err']:.2f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax1.plot(np.arange(1, len(nu_c) + 1), nu_c, color=BLUE, linewidth=2.0, label=label_c)
    ax1.plot(np.arange(1, len(nu_q) + 1), nu_q, color=ORANGE, linewidth=2.0, label=label_q)
    ax1.axhline(nu_true, color=INK, linestyle="--", linewidth=1.2, label=f"target nu = {nu_true:g}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("learned nu")
    ax1.set_title("Learned viscosity")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    ax2.plot(x.ravel(), u_ref.ravel(), color=INK, linewidth=3.0, label="exact (Cole-Hopf)")
    ax2.plot(x.ravel(), u_c.ravel(), color=BLUE, linewidth=2.0, linestyle="--", label=f"classical: rel. L2 = {l2_c:.2e}")
    ax2.plot(x_q.ravel(), u_q.ravel(), color=ORANGE, linewidth=2.0, linestyle="--",
             label=f"quantum on the fake QPU: rel. L2 = {l2_q:.2e}")
    ax2.set_xlabel("x")
    ax2.set_ylabel(f"u(x, T={tfin})")
    ax2.set_title(f"Solution at the final time T = {tfin}")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        style(ax)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"comparacao_classico_quantico_{case}_E{EPOCHS}.png")
    path = save_figure(fig, path)
    plt.close(fig)
    print(f"  {path}")
    print(f"    best classical: {label_c}, rel. L2 of u(x,T) {l2_c:.2e}")
    print(f"    best quantum:   {label_q}, rel. L2 of u(x,T) {l2_q:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tfin", type=float, help="só o caso com tempo final T (padrão: todos os casos)")
    parser.add_argument("--nu", type=float, default=0.1, help="viscosidade verdadeira do caso (com --tfin; padrão 0.1)")
    args = parser.parse_args()

    if args.tfin is not None:
        cases = [case_tag(args.tfin, args.nu)]
    else:
        cases = sorted({os.path.basename(os.path.dirname(p))
                        for p in glob.glob(os.path.join(OUT_DIR, "*", "T_*", "summary.json"))})
    for case in cases:
        print(case)
        summaries = {}
        for model in MODELS:
            path = os.path.join(OUT_DIR, model, case, "summary.json")
            if os.path.exists(path) and load_runs(os.path.dirname(path), model):
                summaries[model] = json.load(open(path))
                plot_nu_summary(case, model, summaries[model]["nu_true"])
        if not summaries:
            print("  sem resultados")
        elif len(summaries) == len(MODELS):
            s = summaries["classico"]
            plot_comparison(case, s["Tfin"], s["nu_true"])
