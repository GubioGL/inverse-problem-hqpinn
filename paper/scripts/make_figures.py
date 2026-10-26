"""Paper figures for the inverse (H)QPINN on Burgers' equation.

Reads the output folders written by Inverser_quantum_hardware_pinn_E110.py and
Inverser_classical_pinn_E110.py (data/Inverso_110/ by default), one per case <case> = T_<T>_nu_<nu>:

    <DATA>/quantico/<case>/summary.json, losshist_10_NQ<n>_<seed>_E110.npz, validation_10_NQ<n>_<seed>_E110.npz
    <DATA>/classico/<case>/summary.json, losshist_10_NE<k>_<seed>_E110.npz, validation_10_NE<k>_<seed>_E110.npz

and writes vector PDFs to figures/, one set per case: fig_nu_statistics_T<T>.pdf,
fig_solution_T<T>.pdf and fig_l2_device_vs_ideal_T<T>.pdf. Real-QPU runs in
<DATA>/quantico_ibm_*/<case>/ are added to the first and last figures when present.

Usage (from the repository root):
    python paper/scripts/make_figures.py --data data/Inverso_110 --out paper/figures
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPOCHS, NCOL = 110, 10
SIZES = (3, 7, 10)
CASES = ("T_0.1_nu_0.01", "T_1.0_nu_0.01")
# fixed categorical order (validated: CVD dE 23.5, normal-vision dE 32.7 on white)
C_CLASS, C_QUANT = "#2A7BD8", "#E8603C"
C_REAL = "#7B4FC9"                      # reserved for the real-QPU series
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#d9d8d3"
COL_W, FULL_W = 3.375, 7.0              # APS column / page width [in]

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)


def panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)


def load_model(folder, prefix, size_key):
    """Runs of one model and case, sorted by (size, seed), and the true viscosity."""
    summ = json.load(open(os.path.join(folder, "summary.json")))
    nu_true = summ["nu_true"]
    runs = sorted(summ["runs"], key=lambda r: (r[size_key], r["seed"]))
    for r in runs:
        r.setdefault("nu_rel_err", abs(r["nu"] - nu_true) / nu_true)
        r["nu_hist"] = np.load(os.path.join(folder, f"losshist_{NCOL}_{prefix}{r[size_key]}_{r['seed']}_E{EPOCHS}.npz"))["nu"]
        r["size"] = r[size_key]
    return runs, nu_true


def fig_nu(models, nu_true, tag, out):
    """Fig. 2: (a) nu_k median + 10-90% band, (b) median relative error with IQR, (c) final error per run."""
    NU_TRUE = nu_true
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.25), gridspec_kw=dict(width_ratios=[1, 1, 0.85], wspace=0.38))
    ep = np.arange(1, EPOCHS + 1)
    for name, runs, color in models:
        H = np.array([r["nu_hist"] for r in runs])
        E = np.abs(H - NU_TRUE) / NU_TRUE
        a = axes[0]
        p10, p90 = np.percentile(H, [10, 90], axis=0)
        a.fill_between(ep, p10, p90, color=color, alpha=0.15, linewidth=0)
        a.plot(ep, np.median(H, 0), color=color, label=name)
        b = axes[1]
        q1, q3 = np.percentile(E, [25, 75], axis=0)
        b.fill_between(ep, q1, q3, color=color, alpha=0.18, linewidth=0)
        b.plot(ep, np.median(E, 0), color=color, label=name)
    axes[0].axhline(NU_TRUE, color=MUTED, ls="--", lw=0.9)
    axes[0].text(EPOCHS, NU_TRUE * 1.18, r"$\nu_{\rm true}$", ha="right", va="bottom", color=MUTED, fontsize=7)
    axes[0].set(yscale="log", xlabel="epoch $k$", ylabel=r"learned $\nu_k$", xlim=(1, EPOCHS), ylim=(NU_TRUE / 20, 1.2))
    axes[1].axhline(0.05, color=MUTED, ls=":", lw=0.9)
    axes[1].text(3, 0.055, "5%", color=MUTED, fontsize=7, va="bottom")
    axes[1].set(yscale="log", xlabel="epoch $k$", ylabel=r"$|\nu_k-\nu_{\rm true}|/\nu_{\rm true}$", xlim=(1, EPOCHS), ylim=(3e-3, 120))
    axes[0].legend(loc="lower right", handlelength=1.6)

    c = axes[2]
    offs = np.linspace(-0.22, 0.22, len(models))
    rng = np.random.default_rng(0)
    for (name, runs, color), dx in zip(models, offs):
        for r in runs:
            xi = SIZES.index(r["size"]) + dx + rng.uniform(-0.07, 0.07)
            c.plot(xi, 100 * r["nu_rel_err"], marker="o", ms=2.6, color=color, alpha=0.7, mec="white", mew=0.3, ls="none")
        med = [np.median([100 * r["nu_rel_err"] for r in runs if r["size"] == s]) for s in SIZES]
        c.plot(np.arange(len(SIZES)) + dx, med, color=color, lw=0.9, alpha=0.8, marker="_", ms=9)
    c.set_xticks(range(len(SIZES)), [f"{s}" for s in SIZES])
    c.set(xlabel="qubits $n$ / neurons $k$", ylabel=r"final $\epsilon_\nu$ [%]", ylim=(0, None))
    for ax, l in zip(axes, "abc"):
        style(ax); panel(ax, l)
    fig.savefig(os.path.join(out, f"fig_nu_statistics_{tag}.pdf"))
    plt.close(fig)


def fig_solution(qdir, cdir, tag, out, n=10, seed=0):
    """Fig. 3: u(x,T) for one paired seed, and the pointwise errors (QPU vs ideal circuit, same weights)."""
    vq = np.load(os.path.join(qdir, f"validation_{NCOL}_NQ{n}_{seed}_E{EPOCHS}.npz"))
    vc = np.load(os.path.join(cdir, f"validation_{NCOL}_NE{n}_{seed}_E{EPOCHS}.npz"))
    x, ue = vq["x"].ravel(), vq["u_exact"].ravel()
    uq, ui, uc = vq["u_qpu"].ravel(), vq["u_ideal_circuit"].ravel(), vc["u_pred"].ravel()
    l2 = lambda u: np.linalg.norm(u - ue) / np.linalg.norm(ue)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.12))
    a.plot(x, ue, color=INK, lw=2.4, alpha=0.35, label="exact (Cole–Hopf)")
    a.plot(x, uc, color=C_CLASS, ls="--", label=fr"classical $k={n}$ ($\epsilon_{{L_2}}={l2(uc):.3f}$)")
    a.plot(x, uq, color=C_QUANT, ls="-", lw=1.2, label=fr"HQPINN $n={n}$, device ($\epsilon_{{L_2}}={l2(uq):.3f}$)")
    a.set(ylabel=r"$u(x,T)$")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.3, borderaxespad=0.2)
    b.plot(x, np.abs(uc - ue), color=C_CLASS, ls="--", label="classical")
    b.plot(x, np.abs(uq - ue), color=C_QUANT, label="HQPINN, device")
    b.plot(x, np.abs(ui - ue), color=MUTED, ls=":", lw=1.3, label=fr"same weights, ideal circuit ($\epsilon_{{L_2}}={l2(ui):.3f}$)")
    top = max(np.abs(uc - ue).max(), np.abs(uq - ue).max(), np.abs(ui - ue).max())
    b.set(xlabel="$x$", ylabel=r"$|u_\Theta-u|$", xlim=(0, 1), ylim=(0, 1.7 * top))
    b.legend(loc="upper left", fontsize=6.3, ncol=1)
    style(a); style(b); panel(b, "b")
    a.text(-0.02, 1.02, "(a)", transform=a.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, f"fig_solution_{tag}.pdf"))
    plt.close(fig)


def fig_l2(qruns, cruns, tag, out):
    """Fig. 4: relative L2 error of u(x,T) per run; the HQPINN evaluated on the device, on the
    surrogate it was trained through, and with the same weights on the ideal circuit."""
    cols = [("classical", [r["rel_l2_T"] for r in cruns], C_CLASS),
            ("HQPINN\nsurrogate", [r["rel_l2_T_measured_model"] for r in qruns], C_QUANT),
            ("HQPINN\ndevice", [r["rel_l2_T_qpu"] for r in qruns], C_QUANT),
            ("HQPINN\nideal circuit", [r["rel_l2_T_ideal_circuit"] for r in qruns], MUTED)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    rng = np.random.default_rng(1)
    jit = rng.uniform(-0.12, 0.12, len(qruns))
    for i, (lab, vals, color) in enumerate(cols):
        runs = cruns if i == 0 else qruns
        for r, v, j in zip(runs, vals, jit):
            ax.plot(i + j, v, marker="o", color=color, ms=2.8, ls="none", mec="white", mew=0.3,
                    alpha=0.8 if i != 1 else 0.5)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=INK, lw=1.0)
    for r, j in zip(qruns, jit):  # pair device <-> ideal circuit, same weights
        ax.plot([2 + j, 3 + j], [r["rel_l2_T_qpu"], r["rel_l2_T_ideal_circuit"]], color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols])
    ax.set(ylabel=r"$\epsilon_{L_2}$ of $u(x,T)$", ylim=(0, None))
    style(ax)
    fig.savefig(os.path.join(out, f"fig_l2_device_vs_ideal_{tag}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "Inverso_110"))
    p.add_argument("--out", default="figures")
    p.add_argument("--cases", nargs="+", default=list(CASES))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for case in a.cases:
        tag = case.split("_nu_")[0].replace("_", "")          # T_0.1_nu_0.01 -> T0.1
        qdir, cdir = os.path.join(a.data, "quantico", case), os.path.join(a.data, "classico", case)
        (qruns, nu_true), (cruns, _) = load_model(qdir, "NQ", "n_qubits"), load_model(cdir, "NE", "neurons")
        models = [("classical PINN", cruns, C_CLASS), ("HQPINN, fake device", qruns, C_QUANT)]
        for d in sorted(glob.glob(os.path.join(a.data, "quantico_ibm_*", case))):   # real-QPU runs, if any
            models.append((f"HQPINN, {d.split(os.sep)[-2][9:]}", load_model(d, "NQ", "n_qubits")[0], C_REAL))
        fig_nu(models, nu_true, tag, a.out)
        fig_solution(qdir, cdir, tag, a.out)
        fig_l2(qruns, cruns, tag, a.out)
        print(case, "->", a.out)
