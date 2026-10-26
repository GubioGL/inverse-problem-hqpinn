"""Inverse hybrid PINN for viscous Burgers with the quantum layer on an IBM QPU, trained in
EPOCHS = 110 epochs with the QPU in the loop: learns nu jointly with u(x,t).

Problem: u_t + u u_x = nu u_xx on [0, 1) periodic, t in [0, TFIN], nu = NU_TRUE, with
u(x,0) = sin(2 pi x) imposed exactly by the network. Model (Inverser_quantum_exact_pinn_paralelo.py):
pre-net -> RY angles -> fixed brickwork CNOT layer -> <Z_i> -> post-net, plus the learnable nu.

Every epoch, N_COLLOCATION fresh random points (x_c, t_c) are used twice:
  * PDE residual: r = A - nu*B (A = u_t + u u_x, B = u_xx), finite differences on the 5-point
    stencil of each point;
  * data: the exact (Cole-Hopf) solution at the same points, u(x_c, t_c).
So the QPU measures only the stencil centres: the same <Z> feeds the residual and the data.

Training (CONFIG, "fixed_varpro"), two optimizers:
  * AdamW for the network, loss = ramp * MAE(residual) + W_D * MAE(data), the residual weight
    ramping up from 0 over the first epochs; inner_steps AdamW steps per epoch, all on the
    epoch's points and its single QPU measurement (the straight-through correction is kept);
  * "varpro" for nu alone, in log space: with the network fixed, the nu minimizing the batch
    residual is the |B|-weighted median of A/B, and log nu is relaxed towards it by a factor
    beta that decays over the training.

QPU in the loop (machinery of Inverser_quantum_hardware_pinn.py):
  1. calibration (1 job per qubit count, cached): random angles on every n-qubit chain of the
     chip -> least squares for the measured response, a degree-1 trigonometric polynomial in
     each angle of the causal cone (HardwareSurrogateLayer);
  2. training (1 job per epoch): <Z> measured at the current angles of the stencil centres;
     the forward pass uses the QPU values and the gradient goes through the polynomial
     (straight-through), q(phi) = poly(phi) + stopgrad(q_QPU(phi_c) - poly(phi_c)), with the
     same correction on the 5 stencil points of centre c. Every RECAL_EVERY epochs the
     polynomial is refitted with the calibration + all measurements;
  3. validation (1 job): u(x, TFIN) of the trained network running on the QPU vs the exact solution.

T = 1, nu = 0.01: the training has the adjustments found for the classical model
(ADJUSTMENTS of Inverser_classical_pinn_E110.py, data/Inverso_110/README.md) -- u(x,0) as a
soft loss on N_IC points at t = 0 per epoch instead of the hard constraint (their <Z> measured in the
epoch's job, with the stencil centres), 10 epochs with the data loss only and nu frozen, and the lr of
the network cut on plateaus instead of the cosine. --treino-antigo runs that case unadjusted.

Fake QPU (default): the noise model of BACKEND simulated locally with Aer (FakeChainCircuit),
no IBM job. Real QPU (--real): PackedCircuit in a Session -- not run in the verification.

Usage (conda env `qiskit`):
    python Inverser_quantum_hardware_pinn_E110.py                        # fake: 3/7/10 qubits x seeds 0..10
    python Inverser_quantum_hardware_pinn_E110.py --qubits 7 10
    python Inverser_quantum_hardware_pinn_E110.py --tfin 0.1                         # final time T = 0.1
    python Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01               # -> quantico/T_0.1_nu_0.01
    python Inverser_quantum_hardware_pinn_E110.py --nu 0.01                          # T = 1, adjusted -> quantico/T_1.0_nu_0.01
    python Inverser_quantum_hardware_pinn_E110.py --nu 0.01 --treino-antigo          # unadjusted -> .../treino_antigo
    python Inverser_quantum_hardware_pinn_E110.py --real --qubits 3 --seeds 0
Outputs: calibrations (independent of T, shared) in OUT_DIR/quantico/; runs and summary.json in
OUT_DIR/quantico/T_<TFIN>/. The summary figures (nu_summary, comparison with the classical model) are
made by plot_inverse_E110.py.
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
import glob
import json
import math
import multiprocessing
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import numpy as np
import torch as tc

sys.path.insert(0, str(_ROOT / "src" / "quantum_hardware"))
from qiskit_ibm_runtime import QiskitRuntimeService, Session  # noqa: E402

from Inverser_quantum_exact_pinn_paralelo import (  # noqa: E402  -- the model and the FD residual
    FIXED_LAYER, FIXED_LAYER_REPEATS, HT, HX, INPUT_ROTATION_AXIS, L, ExactQuantumLayer, build_model,
    exact_solution, initial_condition, plot_loss, plot_parameter_history, plot_solution, relative_l2_error,
    residual_terms, set_seed, stencil_inputs,
)
from Inverser_classical_pinn_E110 import ADJUSTED_CASES, adjustments_for  # noqa: E402  -- T = 1 adjustments
from Inverser_quantum_hardware_pinn import (  # noqa: E402  -- the hardware machinery
    FAKE_BACKENDS, MITIGATION, MITIGATIONS, HardwareSurrogateLayer, PackedCircuit, causal_cones,
    cnot_gate_list, find_chains, fit_surrogate, network_angles, network_output, screen_chains,
)
from main.functions import repeated_cnot_network_matrix  # noqa: E402
from regression_hardware_nqubits import INSTANCE  # noqa: E402
from run_quantum_hardware import FakeChainCircuit  # noqa: E402

# ---- problem --------------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 1.0
N_COLLOCATION = 10        # random points per epoch: residual AND data
EPOCHS = 110
QUBITS = (3, 7, 10)
SEEDS = tuple(range(11))   # 0..10

# ---- training: "fixed_varpro" (data README, data/Inverso_110/README.md) -----------
CONFIG = dict(
    lr_net=3.07e-2, lr_net_min=1.2e-3, betas_net=(0.77, 0.99),   # AdamW, cosine decay
    weight_decay=6.4e-6, grad_clip=10.0,
    r_ramp=20,                # epochs over which the residual weight ramps up from 0
    w_d=85.8,                 # data weight (the residual weight is 1)
    lr_nu=0.97, lr_nu_min=0.128,   # varpro relaxation factor beta, cosine decay
    inner_steps=16,           # AdamW steps per epoch, all on the epoch's single QPU measurement
)

# ---- QPU ------------------------------------------------------------------------
BACKEND = "ibm_berlin"            # the fake is FAKE_BACKENDS[BACKEND]
PRECISION = 1 / 32                # per expectation value -> 1024 shots per circuit
CAL_SAMPLES = 1024                # calibration samples per output (~ chains x parameter sets)
MAX_CHAIN_SCORE_PER_QUBIT = 0.05  # chains with a worse error score (find_chains) are dropped
CHAIN_OUTLIER = 3.0               # ... and so are those with calibration misfit > 3 x the median
RECAL_EVERY = 10                  # refit the polynomial with the training measurements
VAL_POINTS = 100                  # validation: x points at t = TFIN, one replica per chain
SESSION_MAX_TIME = "2h"           # --real: safety limit of the Session

OUT_DIR = os.path.join("data", "Inverso_110")
CAL_DIR = os.path.join(OUT_DIR, "quantico")   # calibrations: shared by every final time


def case_tag(tfin, nu):
    """T_<tfin>, plus _nu_<nu> when nu is not the reference 0.1."""
    return f"T_{tfin}" if nu == 0.1 else f"T_{tfin}_nu_{nu:g}"


def results_dir(tfin, nu, old=False):
    """OUT_DIR/quantico/<case_tag>: the runs for one (T, nu); the unadjusted training of an adjusted
    case (old=True) goes to its subfolder treino_antigo/."""
    path = os.path.join(CAL_DIR, case_tag(tfin, nu))
    return os.path.join(path, "treino_antigo") if old and (tfin, nu) in ADJUSTED_CASES else path


# ----------------------------------------------------------------------
# Training: AdamW for the network, varpro for nu
# ----------------------------------------------------------------------
def cosine(lr0, lr_min, progress):
    """Cosine decay lr0 -> lr_min, progress in [0, 1]."""
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))


def exact_at(x, t, nu_true):
    """Exact solution u(x, t) (Cole-Hopf, nu = nu_true): the data, shaped like x."""
    return tc.tensor(exact_solution(x.numpy(), t.numpy(), nu=nu_true, L=L), dtype=x.dtype)


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


def train(model, seed, measure=None, cfg=CONFIG, Tfin=TFIN, nu_true=NU_TRUE, n_col=N_COLLOCATION, epochs=EPOCHS,
          adj=None):
    """Trains `model` for `epochs` epochs, one batch of n_col fresh random points per epoch.
    measure(epoch, xt, xt_ic), optional (QPU in the loop): gets the epoch's stencil points (and the
    IC points, or None) before the forward pass and returns the offsets added to the output of
    `model.quantum` for each (None for the IC when there are none).
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
        xt_ic = None
        if n_ic:
            x_ic = tc.rand((n_ic, 1), generator=ic_generator) * L
            xt_ic, u_ic = tc.cat([x_ic, tc.zeros_like(x_ic)], dim=1), initial_condition(x_ic, L=L)
        offset = offset_ic = None
        if measure is not None:
            offset, offset_ic = measure(epoch, xt, xt_ic)
        u_data = exact_at(x_f, t_f, nu_true)

        ramp = 0.0 if epoch < warmup else min(1.0, (epoch + 1 - warmup) / cfg["r_ramp"])
        for _ in range(cfg.get("inner_steps", 1)):   # the extra steps reuse the epoch's QPU offsets
            if measure is not None:
                model.quantum.offset = offset
            a, b, loss_r, loss_d = loss_terms(model, xt, n_col, u_data)
            loss = ramp * loss_r + cfg["w_d"] * loss_d
            if n_ic:
                if measure is not None:
                    model.quantum.offset = offset_ic
                loss = loss + adj["w_ic"] * (model(xt_ic) - u_ic).abs().mean()
            opt.zero_grad()
            loss.backward(inputs=net_params)
            tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
            opt.step()
        if measure is not None:
            model.quantum.offset = None

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
# QPU: chains, calibration
# ----------------------------------------------------------------------
def get_backend(real):
    if real:
        return QiskitRuntimeService(instance=INSTANCE).backend(BACKEND)
    return FAKE_BACKENDS[BACKEND]()


def make_qpu(backend, chains, n_qubits, real, threads=0):
    gates = cnot_gate_list(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS)
    if real:
        return PackedCircuit(backend, chains, n_qubits, gates, PRECISION, False, MITIGATIONS[MITIGATION])
    qpu = FakeChainCircuit(backend, chains, n_qubits, gates, PRECISION, None)
    qpu.estimator.backend.set_options(max_parallel_threads=threads)
    return qpu


def circuit_structure(n_qubits):
    gates = cnot_gate_list(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS)
    return repeated_cnot_network_matrix(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS), causal_cones(n_qubits, gates)


def calibrate(backend, n_qubits, real, out_dir):
    """Chains of n qubits + calibration measurements (1 job), cached in out_dir. Chains with a
    bad error score or that do not follow the chip-average polynomial are dropped."""
    path = os.path.join(out_dir, f"calibration_NQ{n_qubits}_{backend.name}.npz")
    if os.path.exists(path):
        cal = np.load(path)
        return cal["chains"].tolist(), cal["angles"], cal["evs"]
    M, cones = circuit_structure(n_qubits)
    chains, scores, _ = find_chains(backend, n_qubits)
    chains = [c for c, s in zip(chains, scores) if s <= MAX_CHAIN_SCORE_PER_QUBIT * n_qubits]
    qpu = make_qpu(backend, chains, n_qubits, real)
    rng = np.random.default_rng(0)
    angle_sets = rng.uniform(-np.pi, np.pi, size=(math.ceil(CAL_SAMPLES / qpu.B), qpu.B, n_qubits))
    angles, evs = angle_sets.reshape(-1, n_qubits), qpu.run(angle_sets, [], verbose=False).reshape(-1, n_qubits)
    keep, _ = screen_chains(angles, evs, qpu.B, cones, M, CHAIN_OUTLIER)
    rows = keep[np.arange(len(angles)) % qpu.B]
    chains = [c for c, k in zip(chains, keep) if k]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, chains=np.array(chains), angles=angles[rows], evs=evs[rows], backend=backend.name)
    return chains, angles[rows], evs[rows]


# ----------------------------------------------------------------------
# One run: training with the QPU in the loop + validation on the QPU
# ----------------------------------------------------------------------
def run_one(n_qubits, seed, real=False, threads=0, tfin=TFIN, nu_true=NU_TRUE, cfg=CONFIG, epochs=EPOCHS, adj=None):
    """Initialization as Inverser_quantum_exact_pinn_paralelo.run_one (set_seed,
    nu_init = np.random.random(), then the network). adj: ADJUSTMENTS or None (unadjusted)."""
    tc.set_num_threads(1)
    t0 = time.time()
    backend = get_backend(real)
    out_dir = results_dir(tfin, nu_true, old=adj is None)
    os.makedirs(out_dir, exist_ok=True)
    chains, cal_angles, cal_evs = calibrate(backend, n_qubits, real, CAL_DIR)
    M, cones = circuit_structure(n_qubits)
    qpu = make_qpu(backend, chains, n_qubits, real, threads)
    coefs, fit_rms, _ = fit_surrogate(cal_angles, cal_evs, cones, M)

    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(n_qubits, tfin, nu_init)
    model.hard_ic = (adj or {}).get("hard_ic", True)
    model.quantum = HardwareSurrogateLayer(cones, coefs)
    job_ids, meas_angles, meas_evs = [], [cal_angles], [cal_evs]

    def measure(epoch, xt, xt_ic=None):
        """<Z> on the QPU at the stencil centres (and the IC points, same job) -> offset of their
        5 stencil points (and of the IC points, or None)."""
        if epoch and epoch % RECAL_EVERY == 0:
            model.quantum.set_coefs(fit_surrogate(np.concatenate(meas_angles), np.concatenate(meas_evs), cones, M)[0])
        points = xt[:N_COLLOCATION] if xt_ic is None else tc.cat([xt[:N_COLLOCATION], xt_ic])
        with tc.no_grad():
            angles = network_angles(model, points[:, :1], points[:, 1:])
            q_poly = model.quantum(angles)
        q_qpu, slot_angles, slot_evs = qpu.evaluate(angles.numpy(), job_ids, shift=epoch, verbose=False)
        meas_angles.append(slot_angles)
        meas_evs.append(slot_evs)
        delta = tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly
        return delta[:N_COLLOCATION].repeat(5, 1), (None if xt_ic is None else delta[N_COLLOCATION:])

    with (Session(backend=backend, max_time=SESSION_MAX_TIME) if real else nullcontext()) as session:
        if real:
            qpu.set_mode(session)
        hist = train(model, seed, measure=measure, cfg=cfg, Tfin=tfin, nu_true=nu_true, epochs=epochs, adj=adj)

        # ---------- validation: the trained network running on the QPU ----------
        x_val = np.linspace(0.0, L, VAL_POINTS, endpoint=False).reshape(-1, 1)
        x_t = tc.tensor(x_val, dtype=tc.get_default_dtype())
        t_t = tc.full_like(x_t, tfin)
        with tc.no_grad():
            angles = network_angles(model, x_t, t_t)
            u_sur = model(tc.cat([x_t, t_t], 1)).numpy()
            q_exact = ExactQuantumLayer(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS, INPUT_ROTATION_AXIS)(angles)
            u_ideal = network_output(model, x_t, t_t, q_exact).numpy()
        q_qpu, _, _ = qpu.evaluate(angles.numpy(), job_ids, replicas=qpu.B, verbose=False)
    with tc.no_grad():
        u_qpu = network_output(model, x_t, t_t, q_qpu).numpy()
    u_ref = exact_solution(x_val, tfin, nu=nu_true, L=L)

    nu_learned = float(model.learnable_parameter.detach())
    out = dict(n_qubits=n_qubits, seed=seed, backend=backend.name, Tfin=tfin, nu_true=nu_true, nu_init=nu_init,
               nu=nu_learned, nu_rel_err=abs(nu_learned - nu_true) / nu_true,
               rel_l2_T_qpu=relative_l2_error(u_qpu, u_ref), rel_l2_T_measured_model=relative_l2_error(u_sur, u_ref),
               rel_l2_T_ideal_circuit=relative_l2_error(u_ideal, u_ref), chains=len(chains),
               fit_rms_calibration=float(np.mean(fit_rms)), n_jobs=len(job_ids), adjustments=adj,
               time_s=time.time() - t0)

    # ---------- outputs ----------
    tag = f"{N_COLLOCATION}_NQ{n_qubits}_{seed}_E{epochs}"
    tc.save(dict(state_dict=model.state_dict(), config=model.config, cfg=cfg, cones=cones,
                 coefs=[c.tolist() for c in model.quantum.coef.numpy()], chains=chains, result=out),
            os.path.join(out_dir, f"model_{tag}.pt"))
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(dict(out, cfg=cfg, job_ids=job_ids, chain_list=chains), f, indent=1)
    np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"), **{k: np.array(v) for k, v in hist.items()})
    np.savez(os.path.join(out_dir, f"validation_{tag}.npz"), x=x_val, angles=angles.numpy(), q_qpu=q_qpu,
             u_qpu=u_qpu, u_measured_model=u_sur, u_ideal_circuit=u_ideal, u_exact=u_ref)
    plot_loss({"residual (QPU values)": hist["loss_r"], "data (QPU values)": hist["loss_d"]},
              title=f"Hardware PINN (inverse, {epochs} epochs, {backend.name}) -- NQ={n_qubits}, seed {seed}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(hist["nu"], nu_true,
                           title=f"Learned nu = {nu_learned:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"{backend.name}, NQ={n_qubits}, seed {seed}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(x_val, {f"QPU ({backend.name})": u_qpu, "measured model": u_sur,
                          "same weights, ideal circuit": u_ideal}, ref=u_ref,
                  title=f"T={tfin}, NQ={n_qubits}, seed {seed}: rel. L2 QPU = {out['rel_l2_T_qpu']:.2e}",
                  save_path=os.path.join(out_dir, f"solution_{tag}.png"))
    return out


def summarize(out_dir, tfin, nu_true, adj=None):
    """summary.json of every run saved in out_dir; returns the runs."""
    runs = sorted((json.load(open(p)) for p in glob.glob(os.path.join(out_dir, f"run_*_E{EPOCHS}.json"))),
                  key=lambda r: (r["n_qubits"], r["seed"]))
    for r in runs:
        for k in ("cfg", "job_ids", "chain_list"):
            r.pop(k, None)
    err = np.array([r["nu_rel_err"] for r in runs])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(config=CONFIG, adjustments=adj, nu_true=nu_true, Tfin=tfin, n_collocation=N_COLLOCATION, epochs=EPOCHS,
                       data="exact solution at the collocation points", nu_rel_err_median=float(np.median(err)),
                       nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()), runs=runs), f, indent=1)
    return runs


def _run_job(job):
    return run_one(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=list(QUBITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--real", action="store_true", help=f"QPU real {BACKEND} (padrão: o fake, local)")
    parser.add_argument("--workers", type=int, default=6, help="runs em paralelo no fake")
    parser.add_argument("--tfin", type=float, default=TFIN, help=f"tempo final T (padrão {TFIN})")
    parser.add_argument("--nu", type=float, default=NU_TRUE, help=f"viscosidade verdadeira (padrão {NU_TRUE})")
    parser.add_argument("--treino-antigo", action="store_true",
                        help="nos casos ajustados (T = 1, nu = 0.01), o treino sem os ajustes (-> subpasta treino_antigo)")
    args = parser.parse_args()
    adj = adjustments_for(args.tfin, args.nu, args.treino_antigo)

    backend = get_backend(args.real)
    out_dir = results_dir(args.tfin, args.nu, old=adj is None)
    for n in args.qubits:   # once per qubit count, before the runs that share it
        calibrate(backend, n, args.real, CAL_DIR)
    jobs = [(n, s) for n in args.qubits for s in args.seeds]
    if args.real:           # one Session per run, one after the other
        for n, s in jobs:
            run_one(n, s, real=True, tfin=args.tfin, nu_true=args.nu, adj=adj)
    else:
        threads = max(1, (os.cpu_count() or 2) // args.workers)
        # spawn, not fork (the Linux default): the parent already holds OpenMP threads from the
        # calibration, and forked workers deadlock on their locks
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(_run_job, [(n, s, False, threads, args.tfin, args.nu, CONFIG, EPOCHS, adj) for n, s in jobs]))

    runs = summarize(out_dir, args.tfin, args.nu, adj)
    print(f"{backend.name}: NU={args.nu}, T={args.tfin}, N_col={N_COLLOCATION}, {EPOCHS} epochs, "
          f"{'adjusted' if adj else 'unadjusted'} training -- learned nu (relative error)")
    for n in sorted({r["n_qubits"] for r in runs}):
        print(f"  NQ{n:<2d} " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                       for r in runs if r["n_qubits"] == n))
