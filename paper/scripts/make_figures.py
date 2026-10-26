"""Paper figures for the inverse (H)QPINN on Burgers' equation.

Reads the output folder written by Inverser_quantum_hardware_pinn_E110.py and
Inverser_classical_pinn_E110.py (data/Inverso_110/ by default):

    <DATA>/quantico_<backend>/summary.json, losshist_10_NQ<n>_<seed>_E110.npz, validation_10_NQ<n>_<seed>_E110.npz
    <DATA>/classico/summary.json,           losshist_10_NE<k>_<seed>_E110.npz, validation_10_NE<k>_<seed>_E110.npz

and writes vector PDFs to figures/. Extra hardware folders (e.g. quantico_ibm_berlin from a
--real run) are added automatically to Fig. 2 and Fig. 4 when present.

Usage:  python scripts/make_figures.py --data data/Inverso_110 --out figures
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NU_TRUE, EPOCHS, NCOL = 0.1, 110, 10
SIZES, SEEDS = (3, 7, 10), (0, 1, 2)
# fixed categorical order (validated: CVD dE 23.5, normal-vision dE 32.7 on white)
C_CLASS, C_QUANT = "#2A7BD8", "#E8603C"
C_REAL = "#7B4FC9"                      # reserved for the real-QPU series
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#d9d8d3"
SEED_MARKERS = {0: "o", 1: "s", 2: "^"}
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
    summ = json.load(open(os.path.join(folder, "summary.json")))
    runs = sorted(summ["runs"], key=lambda r: (r[size_key], r["seed"]))
    for r in runs:
        r.setdefault("nu_rel_err", abs(r["nu"] - NU_TRUE) / NU_TRUE)
        r["nu_hist"] = np.load(os.path.join(folder, f"losshist_{NCOL}_{prefix}{r[size_key]}_{r['seed']}_E{EPOCHS}.npz"))["nu"]
        r["size"] = r[size_key]
    return runs


def fig_nu(models, out):
    """Fig. 2: (a) nu_k median + min/max band, (b) median relative error with IQR, (c) final error per run."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.25), gridspec_kw=dict(width_ratios=[1, 1, 0.85], wspace=0.38))
    ep = np.arange(1, EPOCHS + 1)
    for name, runs, color in models:
        H = np.array([r["nu_hist"] for r in runs])
        E = np.abs(H - NU_TRUE) / NU_TRUE
        a = axes[0]
        a.fill_between(ep, H.min(0), H.max(0), color=color, alpha=0.15, linewidth=0)
        a.plot(ep, np.median(H, 0), color=color, label=name)
        b = axes[1]
        q1, q3 = np.percentile(E, [25, 75], axis=0)
        b.fill_between(ep, q1, q3, color=color, alpha=0.18, linewidth=0)
        b.plot(ep, np.median(E, 0), color=color, label=name)
    axes[0].axhline(NU_TRUE, color=MUTED, ls="--", lw=0.9)
    axes[0].text(EPOCHS, NU_TRUE * 1.18, r"$\nu_{\rm true}$", ha="right", va="bottom", color=MUTED, fontsize=7)
    axes[0].set(yscale="log", xlabel="epoch $k$", ylabel=r"learned $\nu_k$", xlim=(1, EPOCHS), ylim=(8e-5, 0.4))
    axes[1].axhline(0.05, color=MUTED, ls=":", lw=0.9)
    axes[1].text(3, 0.055, "5%", color=MUTED, fontsize=7, va="bottom")
    axes[1].set(yscale="log", xlabel="epoch $k$", ylabel=r"$|\nu_k-\nu_{\rm true}|/\nu_{\rm true}$", xlim=(1, EPOCHS), ylim=(3e-3, 1.5))
    axes[0].legend(loc="lower right", handlelength=1.6)

    c = axes[2]
    offs = np.linspace(-0.22, 0.22, len(models))
    for (name, runs, color), dx in zip(models, offs):
        for r in runs:
            xi = SIZES.index(r["size"]) + dx
            c.plot(xi, 100 * r["nu_rel_err"], marker=SEED_MARKERS[r["seed"]], ms=4.2, color=color,
                   mec="white", mew=0.5, ls="none")
        med = [np.median([100 * r["nu_rel_err"] for r in runs if r["size"] == s]) for s in SIZES]
        c.plot(np.arange(len(SIZES)) + dx, med, color=color, lw=0.9, alpha=0.8, marker="_", ms=9)
    c.set_xticks(range(len(SIZES)), [f"{s}" for s in SIZES])
    c.set(xlabel="qubits $n$ / neurons $k$", ylabel=r"final $\epsilon_\nu$ [%]", ylim=(-0.5, 19.5))
    for s, m in SEED_MARKERS.items():
        c.plot([], [], marker=m, color=MUTED, ls="none", ms=3.8, label=f"seed {s}")
    c.legend(loc="upper center", ncol=3, handletextpad=0.1, columnspacing=0.5, fontsize=6.3, borderaxespad=0.1)
    for ax, l in zip(axes, "abc"):
        style(ax); panel(ax, l)
    fig.savefig(os.path.join(out, "fig_nu_statistics.pdf"))
    plt.close(fig)


def fig_solution(qdir, cdir, out, n=10, seed=0):
    """Fig. 3: u(x,T) for one paired seed, and the pointwise errors (QPU vs ideal circuit, same weights)."""
    vq = np.load(os.path.join(qdir, f"validation_{NCOL}_NQ{n}_{seed}_E{EPOCHS}.npz"))
    vc = np.load(os.path.join(cdir, f"validation_{NCOL}_NE{n}_{seed}_E{EPOCHS}.npz"))
    x, ue = vq["x"].ravel(), vq["u_exact"].ravel()
    uq, ui, uc = vq["u_qpu"].ravel(), vq["u_ideal_circuit"].ravel(), vc["u_pred"].ravel()
    l2 = lambda u: np.linalg.norm(u - ue) / np.linalg.norm(ue)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.12))
    a.plot(x, ue, color=INK, lw=2.4, alpha=0.35, label="exact (Cole–Hopf)")
    a.plot(x, uc, color=C_CLASS, ls="--", label=fr"classical $k={n}$ ($\epsilon_{{L_2}}={l2(uc):.3f}$)")
    a.plot(x, uq, color=C_QUANT, ls="-", lw=1.2, label=fr"HQPINN $n={n}$, device ($\epsilon_{{L_2}}={l2(uq):.3f}$)")
    a.set(ylabel=r"$u(x,T)$", ylim=(-1.12, 0.82))
    a.legend(loc="lower left", fontsize=6.3, borderaxespad=0.2)
    b.plot(x, np.abs(uc - ue), color=C_CLASS, ls="--", label="classical")
    b.plot(x, np.abs(uq - ue), color=C_QUANT, label="HQPINN, device")
    b.plot(x, np.abs(ui - ue), color=MUTED, ls=":", lw=1.3, label=fr"same weights, ideal circuit ($\epsilon_{{L_2}}={l2(ui):.3f}$)")
    b.set(xlabel="$x$", ylabel=r"$|u_\Theta-u|$", xlim=(0, 1))
    b.legend(loc="upper center", fontsize=6.3, ncol=1)
    for ax, l in zip((a, b), "ab"):
        style(ax); panel(ax, l)
    fig.savefig(os.path.join(out, "fig_solution_T.pdf"))
    plt.close(fig)


def fig_l2(qruns, cruns, out):
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
            ax.plot(i + j, v, marker=SEED_MARKERS[r["seed"]], color=color, ms=4, ls="none", mec="white", mew=0.5,
                    alpha=0.9 if i != 1 else 0.55)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=INK, lw=1.0)
    for r, j in zip(qruns, jit):  # pair device <-> ideal circuit, same weights
        ax.plot([2 + j, 3 + j], [r["rel_l2_T_qpu"], r["rel_l2_T_ideal_circuit"]], color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols])
    ax.set(ylabel=r"$\epsilon_{L_2}$ of $u(x,T)$", ylim=(0, 0.085))
    style(ax)
    fig.savefig(os.path.join(out, "fig_l2_device_vs_ideal.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "Inverso_110"))
    p.add_argument("--out", default="figures")
    p.add_argument("--fake", default="quantico_fake_berlin")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qdir, cdir = os.path.join(a.data, a.fake), os.path.join(a.data, "classico")
    qruns, cruns = load_model(qdir, "NQ", "n_qubits"), load_model(cdir, "NE", "neurons")
    models = [("classical PINN", cruns, C_CLASS), ("HQPINN, fake device", qruns, C_QUANT)]
    for d in sorted(glob.glob(os.path.join(a.data, "quantico_ibm_*"))):   # real-QPU runs, if any
        models.append((f"HQPINN, {os.path.basename(d)[9:]}", load_model(d, "NQ", "n_qubits"), C_REAL))
    fig_nu(models, a.out)
    fig_solution(qdir, cdir, a.out)
    fig_l2(qruns, cruns, a.out)
    print("figures written to", a.out)
