"""
Plots the estimation results of the nu parameter (quantum inverse PINN) against
the target value, using the seeds available in data/quantum_simple.

Generates 4 complementary figures, in "scienceplots" style:
  1. Convergence trajectories of nu (all seeds) towards the target.
  2. Scatter of nu_init -> nu_learned vs. target (shows that, starting from
     very different initial guesses, the model converges close to the target).
  3. Absolute error of the final estimate per seed (bars, sorted), highlighting
     which seeds fell within tolerance.
  4. Mean squared error of the final estimate per seed (bars, sorted), same
     highlighting as above.

Outputs in data/quantum_simple/:
  - nu_convergence_all_seeds.png
  - nu_init_vs_learned_scatter.png
  - nu_estimation_error_bar_abs.png
  - nu_estimation_error_bar_mse.png
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/Inverso/..." are resolved from the repository root.
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

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

DATA_DIR = "data/Inverso/quantum_simple"
PREFIX = "inverse_Qmodel_linear_cnot_2q_random_100_"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

COLOR_LEARNED = "#4C72B0"
COLOR_TARGET = "#C44E52"
COLOR_OK = "#55A868"
COLOR_BAD = "#C44E52"

NU_TOL = 0.001


def load_runs():
    runs = []
    config_files = sorted(
        glob.glob(os.path.join(DATA_DIR, f"{PREFIX}*_config.json")),
        key=lambda f: int(f.split(PREFIX)[1].split("_config.json")[0]),
    )
    for cfg_path in config_files:
        seed = int(cfg_path.split(PREFIX)[1].split("_config.json")[0])
        with open(cfg_path) as f:
            cfg = json.load(f)["training_params"]
        hist_path = os.path.join(DATA_DIR, f"{PREFIX}{seed}_history.npz")
        hist = np.load(hist_path)
        runs.append({
            "seed": seed,
            "nu_init": cfg["NU_INIT"],
            "nu_learned": cfg["NU_LEARNED"],
            "nu_target": cfg["NU_TARGET"],
            "nu_tol": cfg["NU_TOL"],
            "within_tol": cfg["NU_WITHIN_TOL"],
            "nu_history": hist["nu"],
        })
    return runs


def plot_convergence(runs, save_path):
    target = runs[0]["nu_target"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    n = len(runs)
    for i, r in enumerate(runs):
        ax.plot(
            r["nu_history"], color=cmap(i / max(n - 1, 1)),
            linewidth=1.2, alpha=0.85,
        )
    ax.axhline(
        target, color=COLOR_TARGET, linestyle="--", linewidth=2.0, zorder=5,
        label=fr"target $\nu$ = {target:g}",
    )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"learned $\nu$")
    ax.set_title(fr"Convergence of $\nu$ across {n} seeds")
    ax.legend(loc="upper right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("seed")
    fig.tight_layout()
    ax.set_yscale("log")
    save_figure(fig, save_path)
    plt.close(fig)


def plot_init_vs_learned(runs, save_path):
    target = runs[0]["nu_target"]
    seeds = [r["seed"] for r in runs]
    inits = [r["nu_init"] for r in runs]
    learned = [r["nu_learned"] for r in runs]
    within = [abs(r["nu_learned"] - r["nu_target"]) <= NU_TOL for r in runs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(target, color=COLOR_TARGET, linestyle="--", linewidth=2.0,
               label=fr"target $\nu$ = {target:g}", zorder=1)
    colors = [COLOR_OK if ok else COLOR_LEARNED for ok in within]
    ax.scatter(inits, learned, c=colors, s=70, edgecolor="black",
               linewidth=0.6, zorder=3)
    for x, y, s in zip(inits, learned, seeds):
        ax.annotate(str(s), (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="dimgray")
    ax.set_xlabel(r"$\nu_{init}$ (initial guess)")
    ax.set_ylabel(r"$\nu_{learned}$ (final estimate)")
    ax.set_title(r"Learned $\nu$ vs. initial guess, by seed")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OK,
                   markeredgecolor="black", markersize=9, label="within tolerance"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_LEARNED,
                   markeredgecolor="black", markersize=9, label="outside tolerance"),
        plt.Line2D([0], [0], color=COLOR_TARGET, linestyle="--", linewidth=2.0,
                   label=fr"target $\nu$ = {target:g}"),
    ]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def _plot_error_bar(runs, save_path, kind):
    """Bar chart of the per-seed estimation error, sorted ascending.

    kind="abs" plots the absolute error |nu_learned - nu_target|.
    kind="mse" plots the squared error (nu_learned - nu_target)^2, i.e. the
    per-seed mean squared error of the estimated parameter(s) (here a single
    scalar parameter, nu).
    """
    target = runs[0]["nu_target"]
    tol = NU_TOL

    def error_of(r):
        diff = r["nu_learned"] - r["nu_target"]
        return diff ** 2 if kind == "mse" else abs(diff)

    tol_line = tol ** 2 if kind == "mse" else tol

    runs_sorted = sorted(runs, key=error_of)
    seeds = [r["seed"] for r in runs_sorted]
    errors = [error_of(r) for r in runs_sorted]
    within = [e <= tol_line for e in errors]
    colors = [COLOR_OK if ok else COLOR_BAD for ok in within]

    if kind == "mse":
        ylabel = r"$(\nu_{learned} - \nu_{target})^2$"
        title = fr"Mean squared error of $\nu$ estimate per seed (target = {target:g})"
        tol_label = f"tolerance$^2$ = {tol_line:g}"
    else:
        ylabel = r"$|\nu_{learned} - \nu_{target}|$"
        title = fr"Absolute error of $\nu$ estimate per seed (target = {target:g})"
        tol_label = f"tolerance = {tol_line:g}"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(seeds))
    ax.bar(x, errors, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
    ax.axhline(tol_line, color="black", linestyle=":", linewidth=1.5, label=tol_label)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=9)
    ax.set_xlabel("seed (sorted by increasing error)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    n_ok = sum(within)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OK, ec="black",
                      label=f"within tolerance ({n_ok}/{len(runs)})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BAD, ec="black",
                      label=f"outside tolerance ({len(runs) - n_ok}/{len(runs)})"),
        plt.Line2D([0], [0], color="black", linestyle=":", linewidth=1.5, label=tol_label),
    ]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_error_bar_abs(runs, save_path):
    _plot_error_bar(runs, save_path, kind="abs")


def plot_error_bar_mse(runs, save_path):
    _plot_error_bar(runs, save_path, kind="mse")


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs from {DATA_DIR}")

    p1 = os.path.join(DATA_DIR, "nu_convergence_all_seeds.png")
    p2 = os.path.join(DATA_DIR, "nu_init_vs_learned_scatter.png")
    p3 = os.path.join(DATA_DIR, "nu_estimation_error_bar_abs.png")
    p4 = os.path.join(DATA_DIR, "nu_estimation_error_bar_mse.png")

    plot_convergence(runs, p1)
    plot_init_vs_learned(runs, p2)
    plot_error_bar_abs(runs, p3)
    plot_error_bar_mse(runs, p4)

    print("Figures saved to:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print(" ", p4)


if __name__ == "__main__":
    main()
