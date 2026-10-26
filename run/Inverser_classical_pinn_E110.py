"""Inverse classical PINN for viscous Burgers trained in EPOCHS = 110 epochs: learns nu jointly
with u(x,t). Classical counterpart of Inverser_quantum_hardware_pinn_E110.py: same problem, same
data, same training (CONFIG "fixed_varpro"); the quantum layer is replaced by a trainable
classical bottleneck of k neurons (k = 3, 7, 10, like the qubits).

Problem: u_t + u u_x = nu u_xx on [0, 1) periodic, t in [0, TFIN], nu = NU_TRUE, with
u(x,0) = sin(2 pi x) imposed exactly by the network. Model (ClassicalPINN of
Inverser_classical_pinn_paralelo.py, the same as Inverser_classical_pinn.py):
pre-net (3 -> 10 -> k) -> net (k -> 10 -> k) -> post-net (k -> 10 -> 1), tanh, plus the learnable nu.

Every epoch, N_COLLOCATION fresh random points (x_c, t_c) are used for the PDE residual
(finite differences on the 5-point stencil) AND as data (the exact Cole-Hopf solution there).
Training: AdamW for the network, loss = ramp * MAE(residual) + W_D * MAE(data), inner_steps steps
per epoch on the epoch's points; "varpro" for nu
alone, in log space: log nu relaxed towards the |B|-weighted median of A/B (residual = A - nu*B).

T = 1, nu = 0.01 (ADJUSTED_CASES): the training has the ADJUSTMENTS found by the training tests
(data/Inverso_110/README.md) -- u(x,0) as a soft loss instead of the hard constraint, 10 epochs
with the data loss only and nu frozen, and the lr of the network cut on plateaus instead of the cosine.
--treino-antigo runs that case with the training above, unadjusted.

Outputs: OUT_DIR/classico/T_<TFIN>/. The summary figures (nu_summary, comparison with the quantum
model) are made by plot_inverse_E110.py.

Usage (conda env `qiskit` or any env with torch + scipy):
    python run/Inverser_classical_pinn_E110.py               # 3/7/10 neurons x seeds 0..10, T = TFIN
    python run/Inverser_classical_pinn_E110.py --tfin 0.1    # same, final time T = 0.1
    python run/Inverser_classical_pinn_E110.py --nu 0.01     # same, true viscosity 0.01 (-> classico/T_1.0_nu_0.01, adjusted)
    python run/Inverser_classical_pinn_E110.py --nu 0.01 --treino-antigo   # T = 1, nu = 0.01 unadjusted (-> .../treino_antigo)
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

import argparse
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch as tc

from Inverser_classical_pinn_paralelo import (
    HT, HX, L, build_model, exact_solution, initial_condition, plot_loss, plot_parameter_history, plot_solution,
    predict, relative_l2_error, residual_terms, set_seed, stencil_inputs,
)

# ---- problem --------------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 1.0
N_COLLOCATION = 10        # random points per epoch: residual AND data
EPOCHS = 110
NEURONS = (3, 7, 10)      # width k of the trainable bottleneck
SEEDS = tuple(range(11))   # 0..10
VAL_POINTS = 100          # u(x, TFIN) evaluated on the same x points as the quantum validation

# ---- training: "fixed_varpro", same scheme as Inverser_quantum_hardware_pinn_E110.py, values
# searched for this classical model with the same protocol (data/Inverso_110/README.md) ----------
CONFIG = dict(
    lr_net=2.53e-2, lr_net_min=3.2e-4, betas_net=(0.73, 0.99),   # AdamW, cosine decay
    weight_decay=9.3e-4, grad_clip=10.0,
    r_ramp=10,                # epochs over which the residual weight ramps up from 0
    w_d=37.8,                 # data weight (the residual weight is 1)
    lr_nu=0.50, lr_nu_min=0.163,   # varpro relaxation factor beta, cosine decay
    inner_steps=16,           # AdamW steps per epoch, all on the epoch's points (as in the quantum script)
)

# ---- T = 1, nu = 0.01: adjustments of the training, chosen on 60 validation runs (seeds 200-219) by
# the training tests (scripts removed, in the git history); median nu error there 38.5% -> 12.7% (data/Inverso_110/README.md) --
ADJUSTMENTS = dict(
    hard_ic=False, n_ic=10, w_ic=10.0,   # u = network output; u(x,0) = u0 as a loss on 10 random points per epoch
    warmup=10,                           # first 10 epochs: data loss only and nu frozen; then the residual ramp
    plateau=dict(patience=5, factor=1.5, threshold=0.01),   # lr constant, /1.5 after 5 epochs without a 1% drop
)
ADJUSTED_CASES = ((1.0, 0.01),)   # (TFIN, NU_TRUE)

OUT_DIR = os.path.join("data", "Inverso_110")


def adjustments_for(tfin, nu, old=False):
    """ADJUSTMENTS for the cases of ADJUSTED_CASES, None (the training above) otherwise or with old=True."""
    return None if old or (tfin, nu) not in ADJUSTED_CASES else ADJUSTMENTS


def case_tag(tfin, nu):
    """T_<tfin>, plus _nu_<nu> when nu is not the reference 0.1."""
    return f"T_{tfin}" if nu == 0.1 else f"T_{tfin}_nu_{nu:g}"


def results_dir(model_kind, tfin, nu, old=False):
    """OUT_DIR/<classico|quantico>/<case_tag>: the runs of one model for one (T, nu); the unadjusted
    training of an adjusted case (old=True) goes to its subfolder treino_antigo/."""
    path = os.path.join(OUT_DIR, model_kind, case_tag(tfin, nu))
    return os.path.join(path, "treino_antigo") if old and (tfin, nu) in ADJUSTED_CASES else path


# ----------------------------------------------------------------------
# Training: AdamW for the network, varpro for nu
# ----------------------------------------------------------------------
def cosine(lr0, lr_min, progress):
    """Cosine decay lr0 -> lr_min, progress in [0, 1]."""
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))


def exact_at(x, t):
    """Exact solution u(x, t) (Cole-Hopf, nu = NU_TRUE): the data, shaped like x."""
    return tc.tensor(exact_solution(x.numpy(), t.numpy(), nu=NU_TRUE, L=L), dtype=x.dtype)


def loss_terms(model, xt, n_f, u_data):
    """Residual split r = A - nu*B (A, B vectors) and the MAE of the residual and of the data
    misfit at the stencil centres (the collocation points themselves)."""
    u = model(xt)
    a, b = residual_terms(u, n_f, HX, HT)
    a, b = a.view(-1), b.view(-1)
    loss_r = (a - model.learnable_parameter * b).abs().mean()
    loss_d = (u[:n_f] - u_data).view(-1).abs().mean()
    return a, b, loss_r, loss_d


def nu_minimizer(a, b):
    """nu minimizing mean|A - nu*B| with the network fixed: the |B|-weighted median of A/B."""
    a, b = a.detach().double(), b.detach().double()
    ratio, weight = a / b, b.abs()
    order = tc.argsort(ratio)
    cum = tc.cumsum(weight[order], 0)
    return float(ratio[order][tc.searchsorted(cum, 0.5 * cum[-1])])


def train(model, seed, cfg=CONFIG, Tfin=TFIN, n_col=N_COLLOCATION, epochs=EPOCHS, adj=None):
    """Trains `model` for `epochs` epochs, one batch of n_col fresh random points per epoch.
    adj: ADJUSTMENTS (model.hard_ic must already be False when adj["hard_ic"] is), or None.
    Returns the history of the loss terms and of nu."""
    adj = adj or {}
    n_ic, warmup, plateau = adj.get("n_ic", 0), adj.get("warmup", 0), adj.get("plateau")
    nu = model.learnable_parameter
    net_params = [p for k, p in model.named_parameters() if k != "learnable_parameter"]
    opt = tc.optim.AdamW(net_params, lr=cfg["lr_net"], betas=cfg["betas_net"], weight_decay=cfg["weight_decay"])
    log_nu = tc.log(nu.detach().clone())
    generator = tc.Generator().manual_seed(seed)
    ic_generator = tc.Generator().manual_seed(seed + 10_000)   # IC points: the collocation points stay the same
    lr_scale, best_loss, wait = 1.0, math.inf, 0                # plateau
    hist = {k: [] for k in ("loss_r", "loss_d", "nu")}

    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        for g in opt.param_groups:
            g["lr"] = cfg["lr_net"] * lr_scale if plateau else cosine(cfg["lr_net"], cfg["lr_net_min"], progress)
        beta = cosine(cfg["lr_nu"], cfg["lr_nu_min"], progress)

        x_f = tc.rand((n_col, 1), generator=generator) * L
        t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
        xt = stencil_inputs(x_f, t_f, L, HX, HT)   # 5 blocks of n_col rows, the centres first
        u_data = exact_at(x_f, t_f)
        if n_ic:
            x_ic = tc.rand((n_ic, 1), generator=ic_generator) * L
            xt_ic, u_ic = tc.cat([x_ic, tc.zeros_like(x_ic)], dim=1), initial_condition(x_ic, L=L)

        ramp = 0.0 if epoch < warmup else min(1.0, (epoch + 1 - warmup) / cfg["r_ramp"])
        for _ in range(cfg.get("inner_steps", 1)):
            a, b, loss_r, loss_d = loss_terms(model, xt, n_col, u_data)
            loss = ramp * loss_r + cfg["w_d"] * loss_d
            if n_ic:
                loss = loss + adj["w_ic"] * (model(xt_ic) - u_ic).abs().mean()
            opt.zero_grad()
            loss.backward(inputs=net_params)
            tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
            opt.step()

        if plateau and epoch >= warmup:   # lr / factor when the epoch loss stalls for `patience` epochs
            epoch_loss = ramp * float(loss_r.detach()) + cfg["w_d"] * float(loss_d.detach())
            if epoch_loss < best_loss * (1.0 - plateau["threshold"]):
                best_loss, wait = epoch_loss, 0
            else:
                wait += 1
                if wait > plateau["patience"]:
                    lr_scale, wait = max(lr_scale / plateau["factor"], 0.01), 0

        if epoch >= warmup:               # nu stays at its initial value during the warmup
            nu_star = min(max(nu_minimizer(a, b), 1.0e-4), 10.0)
            log_nu += beta * (math.log(nu_star) - log_nu)
            with tc.no_grad():
                nu.copy_(tc.exp(log_nu))

        hist["loss_r"].append(float(loss_r.detach()))
        hist["loss_d"].append(float(loss_d.detach()))
        hist["nu"].append(float(nu.detach()))
    return hist


# ----------------------------------------------------------------------
# Runs, outputs
# ----------------------------------------------------------------------
def run_one(neurons, seed, out_dir, adj=None):
    """Same initialization as the quantum script: set_seed, nu_init = np.random.random(), network."""
    tc.set_num_threads(1)
    t0 = time.time()
    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(neurons, TFIN, nu_init)
    model.hard_ic = (adj or {}).get("hard_ic", True)
    hist = train(model, seed, Tfin=TFIN, adj=adj)

    x_val = np.linspace(0.0, L, VAL_POINTS, endpoint=False).reshape(-1, 1)
    u_pred, u_ref = predict(model, x_val, TFIN), exact_solution(x_val, TFIN, nu=NU_TRUE, L=L)
    nu_learned = float(model.learnable_parameter.detach())
    out = dict(neurons=neurons, seed=seed, nu_init=nu_init, nu=nu_learned,
               nu_rel_err=abs(nu_learned - NU_TRUE) / NU_TRUE, rel_l2_T=relative_l2_error(u_pred, u_ref),
               adjustments=adj, time_s=time.time() - t0)

    tag = f"{N_COLLOCATION}_NE{neurons}_{seed}_E{EPOCHS}"
    tc.save(dict(state_dict=model.state_dict(), cfg=CONFIG, result=out), os.path.join(out_dir, f"model_{tag}.pt"))
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"), **{k: np.array(v) for k, v in hist.items()})
    np.savez(os.path.join(out_dir, f"validation_{tag}.npz"), x=x_val, u_pred=u_pred, u_exact=u_ref)
    plot_loss({"residual": hist["loss_r"], "data": hist["loss_d"]},
              title=f"Classical PINN (inverse, {EPOCHS} epochs) -- {neurons} neurons, seed {seed}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(hist["nu"], NU_TRUE,
                           title=f"Learned nu = {nu_learned:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"{neurons} neurons, seed {seed}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(x_val, {"Classical PINN": u_pred}, ref=u_ref,
                  title=f"T={TFIN}, {neurons} neurons, seed {seed}: rel. L2 = {out['rel_l2_T']:.2e}",
                  save_path=os.path.join(out_dir, f"solution_{tag}.png"))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tfin", type=float, default=TFIN, help=f"tempo final T (padrão {TFIN})")
    parser.add_argument("--nu", type=float, default=NU_TRUE, help=f"viscosidade verdadeira (padrão {NU_TRUE})")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="seeds (padrão 0..10)")
    parser.add_argument("--treino-antigo", action="store_true",
                        help="nos casos ajustados (T = 1, nu = 0.01), o treino sem os ajustes (-> subpasta treino_antigo)")
    args = parser.parse_args()
    TFIN, NU_TRUE = args.tfin, args.nu
    adj = adjustments_for(TFIN, NU_TRUE, args.treino_antigo)

    out_dir = results_dir("classico", TFIN, NU_TRUE, args.treino_antigo)
    os.makedirs(out_dir, exist_ok=True)
    runs = [run_one(k, s, out_dir, adj) for k in NEURONS for s in args.seeds]
    err = np.array([r["nu_rel_err"] for r in runs])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(config=CONFIG, adjustments=adj, nu_true=NU_TRUE, Tfin=TFIN, n_collocation=N_COLLOCATION, epochs=EPOCHS,
                       data="exact solution at the collocation points", nu_rel_err_median=float(np.median(err)),
                       nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()), runs=runs), f, indent=1)
    print(f"classical: NU={NU_TRUE}, T={TFIN}, N_col={N_COLLOCATION}, {EPOCHS} epochs, "
          f"{'adjusted' if adj else 'unadjusted'} training -- learned nu (relative error)")
    for k in NEURONS:
        print(f"  {k:2d} neurons " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                            for r in runs if r["neurons"] == k))
