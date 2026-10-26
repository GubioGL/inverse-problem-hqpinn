"""Summary figures of the 110-epoch inverse runs (Inverser_quantum_hardware_pinn_E110.py and
Inverser_classical_pinn_E110.py), made only from the files those scripts save -- nothing is trained.

For each case folder <case> = T_<T>[_nu_<nu>] in OUT_DIR/quantico/ and OUT_DIR/classico/, every
figure saved in OUT_DIR:
  * nu_summary_<model>_<case>_E110.png: learned nu vs epoch, one panel per size (qubits or
    neurons), a thin line per seed and their median (losshist_*.npz);
  * comparacao_classico_quantico_<case>_E110.png (when both models have the case): the run of
    each model with the smallest nu error, learned nu vs epoch and u(x, T) (validation_*.npz);
  * antigo_vs_novo_<case>_E110.png (cases with the adjusted training, whose old runs are in
    <model>/<case>/treino_antigo/): nu error and rel. L2 of u(x, T) of every run, old -> new;
  * comparacao_classico_quantico_nu_<nu>_E110.png (every case of that nu with both models, one row
    per T): nu error and rel. L2 of u(x, T) of every run per size, and the median nu vs epoch with
    its interquartile band, classical vs quantum.

Usage (any env with numpy + matplotlib):
    python plot_inverse_E110.py                 # every case found in OUT_DIR
    python plot_inverse_E110.py --tfin 0.1      # only T = 0.1 (nu = 0.01)
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
def nu_histories(folder, model, runs):
    """(runs, epochs) array of learned nu."""
    size_key, prefix, _ = MODELS[model]
    return np.array([np.load(os.path.join(folder, f"losshist_{run_tag(r, size_key, prefix)}.npz"))["nu"] for r in runs])


def plot_nu_summary(case, model, nu_true):
    """One panel per size: a thin line per seed and the median over the seeds."""
    folder = os.path.join(OUT_DIR, model, case)
    size_key, prefix, size_label = MODELS[model]
    runs = load_runs(folder, model)
    sizes = sorted({r[size_key] for r in runs})
    if not sizes:
        return
    color = BLUE if model == "classico" else ORANGE
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.6), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        rs = [r for r in runs if r[size_key] == size]
        nus = nu_histories(folder, model, rs)
        epochs = np.arange(1, nus.shape[1] + 1)
        for nu in nus:
            ax.plot(epochs, nu, color=color, linewidth=0.8, alpha=0.35)
        ax.plot(epochs, np.median(nus, 0), color=color, linewidth=2.2, label=f"median of {len(rs)} seeds")
        ax.axhline(nu_true, color=INK, linestyle="--", linewidth=1.2, label=f"target {nu_true:g}")
        ax.set_yscale("log")
        err = np.median([100 * r["nu_rel_err"] for r in rs])
        ax.set_title(f"{size} {size_label}: median error of the final nu {err:.1f}%", fontsize=10)
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
# Adjusted cases: old (treino_antigo/) vs new training, run by run
# ----------------------------------------------------------------------
def run_errors(folder, model):
    """{(size, seed): (nu error %, rel. L2 of u(x,T))} of the runs of `folder`; u on the QPU for the quantum model."""
    size_key, prefix, _ = MODELS[model]
    out = {}
    for r in load_runs(folder, model):
        val = np.load(os.path.join(folder, f"validation_{run_tag(r, size_key, prefix)}.npz"))
        u = val["u_qpu"] if "u_qpu" in val.files else val["u_pred"]
        out[(r[size_key], r["seed"])] = (100 * r["nu_rel_err"], relative_l2_error(u, val["u_exact"]))
    return out


def plot_old_vs_new(case):
    """Cases with a treino_antigo/ subfolder: one row per model, nu error and rel. L2 of u(x,T) of every
    run (size x seed), old training (gray) -> new (model color)."""
    rows = []
    for model, color in (("classico", BLUE), ("quantico", ORANGE)):
        folder = os.path.join(OUT_DIR, model, case)
        if os.path.isdir(os.path.join(folder, "treino_antigo")):
            new, old = run_errors(folder, model), run_errors(os.path.join(folder, "treino_antigo"), model)
            keys = sorted(set(new) & set(old), key=lambda k: (k[1], k[0]))
            if keys:
                rows.append((model, color, keys, old, new))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.2 * len(rows)), squeeze=False)
    for (model, color, keys, old, new), (ax_nu, ax_l2) in zip(rows, axes):
        _, _, size_label = MODELS[model]
        name = "classical" if model == "classico" else "quantum, fake QPU"
        for col, ax, unit in ((0, ax_nu, "%"), (1, ax_l2, "")):
            for i, k in enumerate(keys):
                y = len(keys) - 1 - i
                v_old, v_new = old[k][col], new[k][col]
                ax.plot([v_old, v_new], [y, y], color=INK, alpha=0.25, linewidth=3, zorder=1)
                ax.scatter([v_old], [y], s=55, color=INK, edgecolors="white", linewidths=1.5, zorder=2)
                ax.scatter([v_new], [y], s=55, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            ax.set_yticks(range(len(keys))[::-1], [f"{k[0]} {size_label}, seed {k[1]}" for k in keys], fontsize=8)
            med_old = np.median([old[k][col] for k in keys])
            med_new = np.median([new[k][col] for k in keys])
            fmt = (lambda v: f"{v:.1f}%") if col == 0 else (lambda v: f"{v:.3f}")
            what = "error of the learned nu (%)" if col == 0 else "rel. L2 of u(x, T)"
            ax.set_title(f"{name}: {what}, median {fmt(med_old)} -> {fmt(med_new)}", fontsize=10)
            ax.set_xlim(left=0)
            style(ax)
        ax_nu.set_xlabel("relative error of nu (%)")
        ax_l2.set_xlabel("relative L2 error of u(x, T)")
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=INK, label="old training"),
                   plt.Line2D([], [], marker="o", linestyle="", color=color, label="new training (adjusted)")]
        ax_l2.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = save_figure(fig, os.path.join(OUT_DIR, f"antigo_vs_novo_{case}_E{EPOCHS}.png"))
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


# ----------------------------------------------------------------------
# Final comparison: every run of both models, one row per final time
# ----------------------------------------------------------------------
def run_metrics(folder, model):
    """runs of `folder` with "err" (nu error %) and "l2" (rel. L2 of u(x,T); on the QPU for the quantum model)."""
    size_key, prefix, _ = MODELS[model]
    runs = load_runs(folder, model)
    for r in runs:
        val = np.load(os.path.join(folder, f"validation_{run_tag(r, size_key, prefix)}.npz"))
        u = val["u_qpu"] if "u_qpu" in val.files else val["u_pred"]
        r["err"], r["l2"], r["size"] = 100 * r["nu_rel_err"], relative_l2_error(u, val["u_exact"]), r[size_key]
    return runs


def strip(ax, x, values, color):
    """Jittered points of one group and its median as a short bar."""
    rng = np.random.default_rng(int(1000 * (x + 1)))
    ax.scatter(x + rng.uniform(-0.12, 0.12, len(values)), values, s=16, color=color, alpha=0.55, linewidths=0, zorder=2)
    ax.plot([x - 0.17, x + 0.17], [np.median(values)] * 2, color=color, linewidth=3, solid_capstyle="round", zorder=3)


def plot_final_comparison(nu_true):
    """One row per case of this nu with both models: (a) nu error per size, (b) median nu vs epoch with
    the interquartile band, (c) rel. L2 of u(x, T) per size -- classical (blue) vs quantum (orange)."""
    tag = f"_nu_{nu_true:g}"
    cases = sorted(c for c in {os.path.basename(os.path.dirname(p))
                               for p in glob.glob(os.path.join(OUT_DIR, "*", "T_*", "summary.json"))}
                   if c.endswith(tag) and all(os.path.isdir(os.path.join(OUT_DIR, m, c)) for m in MODELS))
    if not cases:
        return
    models = (("classico", "classical", BLUE, -0.2), ("quantico", "quantum, fake QPU", ORANGE, 0.2))
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4.3 * len(cases)), squeeze=False,
                             gridspec_kw=dict(width_ratios=(1, 1.25, 1)))
    print(f"  {'case':16s} {'model':10s} {'runs':>4s} {'nu err median/mean/p90':>24s} {'L2 median':>9s}   per size (median nu err)")
    for case, (ax_e, ax_nu, ax_l2) in zip(cases, axes):
        tfin = case.split("_")[1]
        for model, name, color, dx in models:
            folder = os.path.join(OUT_DIR, model, case)
            runs = run_metrics(folder, model)
            sizes = sorted({r["size"] for r in runs})
            for i, size in enumerate(sizes):
                strip(ax_e, i + dx, [r["err"] for r in runs if r["size"] == size], color)
                strip(ax_l2, i + dx, [r["l2"] for r in runs if r["size"] == size], color)
            nus = nu_histories(folder, model, runs)
            epochs = np.arange(1, nus.shape[1] + 1)
            q1, med, q3 = np.percentile(nus, [25, 50, 75], axis=0)
            ax_nu.fill_between(epochs, q1, q3, color=color, alpha=0.18, linewidth=0)
            e = np.array([r["err"] for r in runs])
            ax_nu.plot(epochs, med, color=color, linewidth=2.0, label=f"{name}: {len(runs)} runs, final nu error median {np.median(e):.1f}%")
            per_size = "  ".join(f"{s}: {np.median([r['err'] for r in runs if r['size'] == s]):.1f}%" for s in sizes)
            print(f"  {case:16s} {model:10s} {len(runs):4d} {np.median(e):8.1f} / {e.mean():5.1f} / {np.percentile(e, 90):5.1f} "
                  f"{np.median([r['l2'] for r in runs]):9.3f}   {per_size}")
        for ax, what in ((ax_e, "relative error of the learned nu (%)"), (ax_l2, "rel. L2 error of u(x, T)")):
            ax.set_xticks(range(len(sizes)), [f"{s}" for s in sizes])
            ax.set_xlabel("size (neurons / qubits)")
            ax.set_ylabel(what)
            ax.set_ylim(bottom=0)
        ax_e.set_title(f"T = {tfin}: error of nu per run (bar = median)", fontsize=10)
        ax_l2.set_title(f"T = {tfin}: error of u(x, T) per run", fontsize=10)
        ax_nu.axhline(nu_true, color=INK, linestyle="--", linewidth=1.2, label=f"target {nu_true:g}")
        ax_nu.set_yscale("log")
        ax_nu.set_xlabel("Epochs")
        ax_nu.set_ylabel("learned nu")
        ax_nu.set_title(f"T = {tfin}: median learned nu, band = quartiles over the runs", fontsize=10)
        ax_nu.legend(fontsize=8, frameon=False, loc="upper right")
        for ax in (ax_e, ax_nu, ax_l2):
            style(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=n) for _, n, c, _ in models]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = save_figure(fig, os.path.join(OUT_DIR, f"comparacao_classico_quantico{tag}_E{EPOCHS}.png"))
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tfin", type=float, help="só o caso com tempo final T (padrão: todos os casos)")
    parser.add_argument("--nu", type=float, default=0.01, help="viscosidade verdadeira do caso (padrão 0.01)")
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
        plot_old_vs_new(case)
        if not summaries:
            print("  sem resultados")
        elif len(summaries) == len(MODELS):
            s = summaries["classico"]
            plot_comparison(case, s["Tfin"], s["nu_true"])
    plot_final_comparison(args.nu)
