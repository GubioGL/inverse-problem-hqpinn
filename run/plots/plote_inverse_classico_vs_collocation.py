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

import os
import matplotlib.pyplot as plt
from main.functions import (
    compute_classical_inverse_nu_error_stats_vs_collocation,
    compute_quantum_inverse_nu_error_stats_vs_collocation,
)

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

CLASSICAL_INVERSE_DIR = "data/Inverso/classical_inverse"
QUANTUM_INVERSE_DIR = "data/Inverso/quantum_inverse_exact"
T_LABELS = ["0.1", "1"]
INTRA_NEURO_LIST = [3, 7, 15]  # widths with a full N_COLLOCATION sweep
N_QUBITS_LIST = [3, 7,15]  # quantum widths with a full N_COLLOCATION sweep
INTRA_NEURO_COLORS = {
    3: "#e34948",
    7: "#3aa15c",
    15: "#8a4fd1",
}
QUANTUM_COLOR = "black"
QUANTUM_LINESTYLES = {3: "--", 7: ":", 15: "-."}
# Fixed marker per width (INTRA_NEURO / N_QUBITS), shared between classical
# and quantum -- only color/linestyle distinguish the two.
MARKER_BY_PARAM = {3: "o", 7: "D", 15: "v"}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def plot_nu_error_vs_collocation(series, Tfin, save_path):
    """series: list of (kind, param, n_collocation, mse_mean, mse_var), kind
    in {"Classical", "Quantum"} and param = INTRA_NEURO (classical) or
    N_QUBITS (quantum). One curve per (kind, param): fixed marker per param
    (MARKER_BY_PARAM), classical colored by param (INTRA_NEURO_COLORS) with
    a solid line, quantum always black with its own dashed/dotted linestyle
    (QUANTUM_LINESTYLES) -- for direct comparison against the classical
    curves. X-axis positions are evenly spaced (union of the N_COLLOCATION
    values across all series)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n = sorted({n for _, _, n_list, _, _ in series for n in n_list})
    positions = {n: i for i, n in enumerate(all_n)}

    for kind, param, n_collocation, mse_mean, mse_var in series:
        pos = [positions[n] for n in n_collocation]
        marker = MARKER_BY_PARAM.get(param, "o")
        if kind == "Quantum":
            color = QUANTUM_COLOR
            linestyle = QUANTUM_LINESTYLES.get(param, "--")
            label = f"{param} - Qubits"
        else:
            color = INTRA_NEURO_COLORS.get(param)
            linestyle = "-"
            label = f"{param} - Neurons"
        ax.errorbar(pos, mse_mean, yerr=mse_var, color=color, linestyle=linestyle,
                     marker=marker, linewidth=2.0, markersize=6, capsize=4, label=label)

    ax.set_xlabel("N° of discretization points", fontsize=14)
    ax.set_ylabel(r"MSE($\nu$)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.set_title(f"T = {Tfin}", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_INVERSE_DIR, f"T_{t_label}")
        quantum_t_dir = os.path.join(QUANTUM_INVERSE_DIR, f"T_{t_label}")

        error_series = []
        Tfin_error = None
        for intra_neuron in INTRA_NEURO_LIST:
            (n_collocation, mse_mean, mse_std, mse_var, mse_min, mse_max, abs_mean, abs_std,
             abs_var, abs_min, abs_max, n_samples, Tfin) = compute_classical_inverse_nu_error_stats_vs_collocation(
                t_dir, intra_neuron=intra_neuron
            )

            if not n_collocation:
                print(f"No classical inverse models found in {t_dir} for INTRA_NEURO={intra_neuron}, skipping.")
                continue

            error_series.append(("Classical", intra_neuron, n_collocation, mse_mean, mse_var))
            Tfin_error = Tfin

        for n_qubits in N_QUBITS_LIST:
            (n_collocation, mse_mean, mse_std, mse_var, mse_min, mse_max, abs_mean, abs_std,
             abs_var, abs_min, abs_max, n_samples, Tfin) = compute_quantum_inverse_nu_error_stats_vs_collocation(
                quantum_t_dir, n_qubits=n_qubits
            )

            if not n_collocation:
                print(f"No quantum inverse models found in {quantum_t_dir} for N_QUBITS={n_qubits}, skipping.")
                continue

            error_series.append(("Quantum", n_qubits, n_collocation, mse_mean, mse_var))
            Tfin_error = Tfin_error or Tfin

        if not error_series:
            print(f"No inverse stats found for T={t_label}, skipping.")
            continue

        neuro_suffix = "-".join(str(n) for n in INTRA_NEURO_LIST)
        save_path = os.path.join(
            CLASSICAL_INVERSE_DIR, f"NuError_vs_N_COLLOCATION_T_{t_label}_NEURO{neuro_suffix}.png"
        )
        plot_nu_error_vs_collocation(error_series, Tfin_error, save_path)
        print(f"  -> saved {save_path}")
