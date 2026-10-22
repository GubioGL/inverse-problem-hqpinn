"""Hybrid forward PINN for viscous Burgers (nu known, only u(x,t) is learned) trained
with the quantum bottleneck on a fake IBM QPU (FakeBerlin / FakeAachen noise model,
simulated locally with Aer) -- QPU in the loop from the first epoch, several seeds
trained together as one batch.

Same problem and model as run_quantumExact.ipynb (QuantumExactPINN: pre-net -> fixed
CNOT feature map -> post-net, MSE of the FD-stencil residual, FDM reference), but the
quantum layer is the response MEASURED on the QPU. The hardware machinery is the one of
Inverser_quantum_hardware_pinn.py (imported from there):

  1. calibration (1 job): random angles on every n-qubit chain of the chip -> the
     measured <Z_i> is a degree-1 trigonometric polynomial in each angle of the causal
     cone of qubit i (<= 4 qubits for the brickwork layer -> 81 coefficients), fitted by
     least squares (HardwareSurrogateLayer). It is not a training phase: it measures the
     derivative of the device response, which the gradient needs (below);
  2. training, QPU in the loop (1 job per epoch for ALL seeds): from the random
     initialization, every AdamW step measures <Z> at the current angles of the
     collocation points of every seed. The forward pass uses the QPU values and the
     gradient goes through the polynomial (straight-through), with the same correction
     on the 5 points of the FD stencil of each collocation point:

         q(phi) = poly(phi) + stopgrad(q_QPU(phi_c) - poly(phi_c)).

     Measuring the stencil neighbours on the QPU instead would put the shot noise on
     u_xx times HX^-2 ~ 2.6e5. Every RECAL_EVERY epochs the polynomial is refitted with
     the calibration + every measurement of the training. Learning rate: LR0, minus
     LR_STEP every LR_EVERY epochs, down to LR_MIN;
  3. validation (1 job): u(x, Tfin) of every seed from the network running on the QPU,
     compared with the FDM reference (MSE / MAE / relative L2, as in the notebook).

Closeness to the hardware: every <Z> of the training and of the validation comes from
the fake QPU with its noise model (gate errors, T1/T2 relaxation during the gates, readout
errors; T1/T2 on the idle qubits too with SCHEDULING = 'alap') and 1/precision^2 shots per
circuit; the chains, layouts and ISA gates are the ones the real chip would run. What Aer
cannot reproduce: the runtime error suppression/mitigation (the fake runs the raw
circuits, like --mitigation none), crosstalk and drift in time.
The chains are independent, so each one is a separate n-qubit circuit (FakeChainCircuit):
Aer simulates it exactly with a density matrix and then samples the shots, instead of
one 90+-qubit noisy circuit.

Uso (ambiente conda `qiskit`):
    python run_quantum_hardware.py                          # FakeBerlin, T=0.1, N_col=10, 10 qubits, seeds 0-4
    python run_quantum_hardware.py --backend ibm_aachen     # FakeAachen
    python run_quantum_hardware.py --seeds 0                # uma seed só
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
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")  # no window: figures are only saved

import numpy as np
import torch as tc

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

# Hardware machinery (chains, calibration polynomial, point grid) of the inverse script
from Inverser_quantum_hardware_pinn import (
    FAKE_BACKENDS, HardwareSurrogateLayer, PackedCircuit,
    causal_cones, cnot_gate_list, find_chains, fit_surrogate, screen_chains,
)
from main.functions import (
    cnot_cosine_expectations, plot_loss, plot_solution, reference_at_time,
    relative_l2_error, repeated_cnot_network_matrix, sample_collocation_points, set_seed,
    solve_burgers_fd,
)
from main.models import QuantumExactPINN

ROOT = _ROOT

# ---- fake QPU -------------------------------------------------------------------
BACKEND = "ibm_berlin"            # simulated with FAKE_BACKENDS[BACKEND]
# find_chains score = readout + sx errors of the chain + its CZ errors, so it grows with the
# chain length; on fake_berlin the 10-qubit chains score 0.14-0.6, plus two with a broken
# qubit (> 1) that would bias the chip-average polynomial
MAX_CHAIN_SCORE_PER_QUBIT = 0.05
CHAIN_OUTLIER = 3.0               # see Inverser_quantum_hardware_pinn.py (0 = keep all)
# 'alap' schedules the circuits as on the device: the idle qubits get delays -> T1/T2 noise on
# them. The circuit lasts ~0.3 us against T1 ~ 100 us: with 65536 shots the scheduled and
# unscheduled <Z> differ by RMS 0.0049, the shot noise of the comparison alone (0.0055), and
# Aer gets ~3x slower (the relaxation channels break its gate fusion) -> off (None) by default
SCHEDULING = None
PRECISION = 1 / 32                # per expectation value -> 1024 shots per circuit
CAL_SAMPLES = 1024                # calibration samples per output (~ chains x parameter sets)

OUT_DIR = ROOT / "data" / "quantum_forward_hardware"

# ---- physical parameters (same as run_quantumExact.ipynb) ---------------------
L, NU = 1.0, 0.01
HX, HT = L / 512, 1.0e-3
# ---- architecture ---------------------------------------------------------------
N_QUBITS = 10
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
# the hardware encodes with RY; for <Z> it is the same feature map as the notebook's 'rx'
INPUT_ROTATION_AXIS = "ry"
ANGLE_SCALE = None
# chain networks only (no SWAPs); brickwork has the smallest causal cone (<= 4 qubits)
FIXED_LAYER = "brickwork_cnot"
FIXED_LAYER_REPEATS = 1
MAX_CONE = 6                      # 3^6 = 729 coefficients per output
# ---- training (QPU in the loop) ---------------------------------------------------
N_COLLOCATION = 10
TFIN = 0.1
SEEDS = (0, 1, 2, 3, 4)           # trained together: their points share each QPU job
EPOCHS = 110
LR0, LR_STEP, LR_EVERY, LR_MIN = 0.1, 0.01, 10, 0.01   # 0.1 in epochs 1-10, ..., 0.01 from epoch 91 on
WEIGHT_DECAY = 1.0e-6
GRAD_CLIP = 1.0
RECAL_EVERY = 10                  # refit the polynomial with the training measurements (0 = never)
COLLOCATION_DISTRIBUTION = "random"
N_EVALUATION_POINTS = 200


def learning_rate(epoch):
    """LR of `epoch` (1-based)."""
    return max(LR0 - LR_STEP * ((epoch - 1) // LR_EVERY), LR_MIN)


class QuantumHardwarePINN(QuantumExactPINN):
    """QuantumExactPINN whose quantum layer can be swapped: `quantum = None` is the exact
    cosine-product formula (identical to the parent), otherwise a module angles -> <Z>
    (HardwareSurrogateLayer)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantum = None

    def angles(self, xt):
        """Rotation angles the pre-net sends to the circuit (same as the parent's forward)."""
        parameter = self.pre[0].weight
        xt = xt.to(dtype=parameter.dtype, device=parameter.device)
        input_features, _ = self.Input_angular(xt)
        raw_angles = self.pre(input_features)
        return raw_angles if self.angle_scale is None else self.angle_scale * tc.tanh(raw_angles)

    def from_features(self, xt, q):
        """u(x,t) from given quantum features q (the parent's forward after the circuit)."""
        parameter = self.pre[0].weight
        xt = xt.to(dtype=parameter.dtype, device=parameter.device)
        input_features, x = self.Input_angular(xt)
        q = tc.as_tensor(q, dtype=parameter.dtype, device=parameter.device)
        return self.Hard_constrain(x, input_features[:, 2:3], self.post(q))

    def forward(self, xt):
        angles = self.angles(xt)
        q = cnot_cosine_expectations(angles, self.cnot_matrix) if self.quantum is None else self.quantum(angles)
        return self.from_features(xt, q)


def build_model(n_qubits, Tfin):
    return QuantumHardwarePINN(
        n_qubits=n_qubits, pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
        input_rotation_axis=INPUT_ROTATION_AXIS, angle_scale=ANGLE_SCALE, L=L, Tfin=Tfin,
    )


# ----------------------------------------------------------------------
# Fake QPU: one n-qubit circuit per chain (same interface as PackedCircuit)
# ----------------------------------------------------------------------
class FakeChainCircuit:
    """Stand-in for PackedCircuit on a fake backend. Every chain is transpiled on its own
    physical qubits (same ISA gates as the packed circuit, scheduled as on the device),
    and all chains go in one BackendEstimatorV2 job on AerSimulator.from_backend -- what
    the runtime local mode runs, minus its deep copy of the backend's noise model on
    every job (~5 s)."""

    evaluate = PackedCircuit.evaluate

    def __init__(self, backend, chains, n, gates, precision, scheduling):
        self.B, self.n = len(chains), n
        phi = ParameterVector("phi", n)
        qc = QuantumCircuit(n)
        for k in range(n):
            qc.ry(phi[k], k)
        for c, t in gates:
            qc.cx(c, t)
        self.g2 = next(g for g in ("cz", "ecr", "cx") if g in backend.target.operation_names)
        self.isa, self.obs, self.phi_idx = [], [], []
        for chain in chains:
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=chain,
                                              scheduling_method=scheduling)
            isa = pm.run(qc)
            if scheduling:
                # the delays on the other qubits of the chip (the other chains, on the device)
                # would stop Aer from dropping those idle qubits; without crosstalk in the noise
                # model they do not change this chain
                chain_qubits = {isa.qubits[q] for q in chain}
                scheduled, isa = isa, isa.copy_empty_like()
                for inst in scheduled.data:
                    if inst.operation.name != "delay" or inst.qubits[0] in chain_qubits:
                        isa.append(inst)
            self.isa.append(isa)
            self.obs.append([SparsePauliOp.from_sparse_list([("Z", [k], 1.0)], num_qubits=n)
                             .apply_layout(isa.layout) for k in range(n)])
            order = list(isa.parameters)
            self.phi_idx.append([order.index(p) for p in phi])
        self.n_qubits_used = self.B * n
        self.depth = max(isa.depth() for isa in self.isa)
        self.n_2q = sum(isa.count_ops().get(self.g2, 0) for isa in self.isa)
        self.n_2q_expected = self.B * len(gates)
        self.precision = precision
        # n noisy qubits per circuit: exact density matrix, then the shots are sampled
        self.estimator = BackendEstimatorV2(backend=AerSimulator.from_backend(backend, method="density_matrix"))

    def run(self, angle_sets, job_ids, verbose=True):
        """angle_sets (T, B, n) -> <Z> (T, B, n). One job (B pubs)."""
        pubs = []
        for b, (isa, obs, idx) in enumerate(zip(self.isa, self.obs, self.phi_idx)):
            vals = np.zeros((len(angle_sets), isa.num_parameters))
            vals[:, idx] = angle_sets[:, b]
            pubs.append((isa, obs, vals[:, None, :]))
        job = self.estimator.run(pubs, precision=self.precision)
        job_ids.append(job.job_id())
        if verbose:
            print(f"  job {job.job_id()} enviado ({len(angle_sets)} conjuntos de parâmetros)", flush=True)
        return np.stack([np.asarray(r.data.evs) for r in job.result()], axis=1)


# ----------------------------------------------------------------------
# FD stencil and the QPU-in-the-loop step
# ----------------------------------------------------------------------
def stencil_inputs(x_f, t_f):
    """5 blocks of n rows (centre, x+hx, x-hx, t+ht, t-ht), same stencil as main.functions.train."""
    x_p, x_m = tc.remainder(x_f + HX, L), tc.remainder(x_f - HX, L)
    return tc.cat([tc.cat([x_f, t_f], 1), tc.cat([x_p, t_f], 1), tc.cat([x_m, t_f], 1),
                   tc.cat([x_f, t_f + HT], 1), tc.cat([x_f, t_f - HT], 1)])


def residual_loss(model, xt_stencil):
    """MSE of u_t + u u_x - nu u_xx on the stencil (the loss of main.functions.train)."""
    u_0, u_xp, u_xm, u_tp, u_tm = model(xt_stencil).chunk(5)
    u_t = (u_tp - u_tm) / (2.0 * HT)
    u_x = (u_xp - u_xm) / (2.0 * HX)
    u_xx = (u_xp - 2.0 * u_0 + u_xm) / HX**2
    return tc.mean((u_t + u_0 * u_x - NU * u_xx) ** 2)


def qpu_step(model, opt, xt, angles, q_qpu):
    """One AdamW step of one seed, given the <Z> measured at the stencil centres `angles`
    (module docstring, step 2). Returns the loss on the QPU values."""
    with tc.no_grad():
        q_poly = model.quantum(angles)
    delta = tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly
    model.quantum.offset = delta.repeat(5, 1)          # same correction on the 5 stencil points
    try:
        loss = residual_loss(model, xt)
    finally:
        model.quantum.offset = None
    opt.zero_grad()
    loss.backward()
    tc.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return float(loss.detach())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-qubits", type=int, default=N_QUBITS)
    p.add_argument("--n-col", type=int, default=N_COLLOCATION, help="pontos de colocação por seed")
    p.add_argument("--tfin", type=float, default=TFIN)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="seeds treinadas juntas")
    p.add_argument("--backend", default=BACKEND, choices=tuple(FAKE_BACKENDS), help="QPU simulado pelo fake")
    p.add_argument("--blocks", type=int, default=None, help="cadeias usadas (padrão: todas que cabem)")
    p.add_argument("--max-chain-score", type=float, default=None,
                   help=f"descarta cadeias com nota acima disso (padrão: {MAX_CHAIN_SCORE_PER_QUBIT} × n)")
    p.add_argument("--chain-outlier", type=float, default=CHAIN_OUTLIER,
                   help="descarta cadeias com desajuste na calibração > isso × a mediana (0 = nunca)")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="épocas com o QPU no loop (1 job cada)")
    p.add_argument("--recal-every", type=int, default=RECAL_EVERY,
                   help="reajusta o polinômio a cada N épocas (0 = nunca)")
    p.add_argument("--val-replicas", type=int, default=None,
                   help="réplicas por ponto na validação (padrão: uma em cada cadeia)")
    args = p.parse_args()
    if args.max_chain_score is None:
        args.max_chain_score = MAX_CHAIN_SCORE_PER_QUBIT * args.n_qubits
    return args


def main():
    args = parse_args()
    n, n_col, Tfin, seeds = args.n_qubits, args.n_col, args.tfin, args.seeds
    S = len(seeds)
    out_dir = OUT_DIR / f"T_{Tfin}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- circuit structure ----------
    gates = cnot_gate_list(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    cones = causal_cones(n, gates)
    if max(map(len, cones)) > MAX_CONE:
        raise SystemExit(f"Cone causal de {max(map(len, cones))} qubits em '{FIXED_LAYER}': "
                         f"3^{max(map(len, cones))} coeficientes por saída é demais para calibrar.")

    backend = FAKE_BACKENDS[args.backend]()
    tag = lambda seed: f"{n_col}_NQ{n}_{seed}_{backend.name}_HW{args.epochs}"
    batch_tag = f"{n_col}_NQ{n}_seeds{seeds[0]}-{seeds[-1]}_{backend.name}_HW{args.epochs}"
    log_path = out_dir / f"run_{batch_tag}.json"

    # ---------- chip packing ----------
    chains, scores, _ = find_chains(backend, n)
    kept = [c for c, s in zip(chains, scores) if s <= args.max_chain_score]   # sorted best first
    print(f"{len(kept)} de {len(chains)} cadeias de {n} qubits com nota ≤ {args.max_chain_score:.2f} "
          f"(descartadas: {', '.join(f'{s:.3f}' for s in scores[len(kept):]) or 'nenhuma'})")
    chains = kept[:args.blocks] if args.blocks else kept
    if not chains:
        raise SystemExit(f"Nenhuma cadeia de {n} qubits conectados em {backend.name}.")
    make_circuit = lambda chains: FakeChainCircuit(backend, chains, n, gates, PRECISION, SCHEDULING)
    hw = make_circuit(chains)
    shots = int(np.ceil(PRECISION ** -2))
    print(f"Backend: {backend.name} (Aer density matrix, agendamento {SCHEDULING or 'nenhum'}) | rede {FIXED_LAYER} × "
          f"{FIXED_LAYER_REPEATS}, {n} qubits, cone causal ≤ {max(map(len, cones))} -> "
          f"≤ {3 ** max(map(len, cones))} coeficientes por saída")
    print(f"{hw.B} cadeias × {n} qubits = {hw.n_qubits_used} de {backend.num_qubits} qubits | "
          f"profundidade ISA {hw.depth} | {hw.n_2q} portas {hw.g2} | {shots} shots por circuito")
    if hw.n_2q != hw.n_2q_expected:
        print(f"AVISO: esperado {hw.n_2q_expected} portas {hw.g2}: o transpilador inseriu SWAPs.")

    log = dict(
        backend=backend.name, n_qubits=n, n_collocation=n_col, Tfin=Tfin, seeds=seeds, nu=NU,
        network=FIXED_LAYER, n_cnot_layers=FIXED_LAYER_REPEATS, scheduling=SCHEDULING,
        chains=chains, blocks=hw.B, shots=shots, cones=cones, job_ids=[],
        training_config=dict(epochs=args.epochs, lr0=LR0, lr_step=LR_STEP, lr_every=LR_EVERY, lr_min=LR_MIN,
                             weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, recal_every=args.recal_every,
                             collocation_distribution=COLLOCATION_DISTRIBUTION,
                             n_evaluation_points=N_EVALUATION_POINTS, L=L, nu=NU, hx=HX, ht=HT),
        recal=[])

    def save_log():
        log_path.write_text(json.dumps(log, indent=2))

    t0 = time.time()

    # ---------- FDM reference (as in the notebook) ----------
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, N_EVALUATION_POINTS).reshape(-1, 1)
    u_ref_eval, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, Tfin)]), dtype=tc.get_default_dtype())

    # ---------- 1. calibration on the fake QPU ----------
    T_cal = math.ceil(CAL_SAMPLES / hw.B)
    print(f"1/3 calibração no QPU: 1 job × {T_cal} conjuntos de parâmetros...")
    rng = np.random.default_rng(seeds[0])
    angle_sets = rng.uniform(-np.pi, np.pi, size=(T_cal, hw.B, n))
    t_job = time.time()
    evs = hw.run(angle_sets, log["job_ids"])
    cal_angles, cal_evs = angle_sets.reshape(-1, n), evs.reshape(-1, n)
    cal_path = out_dir / f"calibration_{batch_tag}.npz"
    np.savez(cal_path, angles=cal_angles, evs=cal_evs, chains=np.array(chains), backend=backend.name)
    log.update(calibration_file=str(cal_path), calibration_time_s=time.time() - t_job)

    # chains that do not follow the chip-average polynomial are left out of everything below
    keep, misfit = screen_chains(cal_angles, cal_evs, hw.B, cones, M, args.chain_outlier)
    log["chain_misfit"] = misfit.tolist()
    if not keep.all():
        print(f"  cadeias descartadas pela calibração (desajuste > {args.chain_outlier} × mediana "
              f"{np.median(misfit):.3f}): " + ", ".join(f"{chains[b]} ({misfit[b]:.3f})"
                                                       for b in np.flatnonzero(~keep)))
        rows = keep[np.arange(len(cal_angles)) % hw.B]
        cal_angles, cal_evs = cal_angles[rows], cal_evs[rows]
        log["dropped_chains"] = [c for c, k in zip(chains, keep) if not k]
        chains = [c for c, k in zip(chains, keep) if k]
        hw = make_circuit(chains)
        log.update(chains=chains, blocks=hw.B)

    coefs, fit_rms, contrast = fit_surrogate(cal_angles, cal_evs, cones, M)
    ideal = np.prod(np.where(M.astype(bool)[None], np.cos(cal_angles)[:, None, :], 1.0), axis=2)
    log.update(fit_rms=fit_rms, contrast=contrast,
               rms_hw_vs_exact=float(np.sqrt(np.mean((cal_evs - ideal) ** 2))))
    save_log()
    print(f"  {log['calibration_time_s']:.0f} s | ajuste: RMS do resíduo {np.mean(fit_rms):.4f} "
          f"(ruído de shot ≈ {PRECISION:.4f}) | contraste do termo ideal {np.min(contrast):.3f}–"
          f"{np.max(contrast):.3f} (ideal = 1) | RMS hardware − fórmula exata {log['rms_hw_vs_exact']:.4f}")

    # ---------- 2. training with the QPU in the loop, all seeds in one job per epoch ----------
    models, opts, samplers = [], [], []
    for seed in seeds:
        set_seed(seed)              # same initialization as the notebook's run of this seed
        model = build_model(n, Tfin)
        model.quantum = HardwareSurrogateLayer(cones, coefs)
        models.append(model)
        opts.append(tc.optim.AdamW(model.parameters(), lr=LR0, weight_decay=WEIGHT_DECAY))
        generator = tc.Generator().manual_seed(seed)   # this seed's stream of random collocation points
        samplers.append(lambda g=generator: (tc.rand((n_col, 1), generator=g) * L,
                                             HT + tc.rand((n_col, 1), generator=g) * (Tfin - 2.0 * HT)))
    x_mon, t_mon = sample_collocation_points(n_col, L, Tfin, HT, distribution="uniform_exact")
    xt_mon = stencil_inputs(x_mon, t_mon)

    def monitor_losses():
        """Residual on a fixed uniform grid, with the measured model (no QPU job)."""
        with tc.no_grad():
            return [float(residual_loss(m, xt_mon)) for m in models]

    hist = dict(lr=[], loss_qpu=[], loss_monitor=[monitor_losses()])
    hw_angles, hw_evs = [], []
    T_epoch = math.ceil(S * n_col / hw.B)
    print(f"2/3 treino com o QPU no loop: {args.epochs} épocas, {S} seeds × {n_col} pontos = {S * n_col} "
          f"pontos por job ({T_epoch} conjuntos × {hw.B} cadeias), lr {LR0} -> {LR_MIN} "
          f"(−{LR_STEP} a cada {LR_EVERY} épocas)...")
    t_hw = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = learning_rate(epoch)
        xts, angles = [], []
        for model, opt, sample in zip(models, opts, samplers):
            for group in opt.param_groups:
                group["lr"] = lr
            model.train()
            x_f, t_f = sample()
            xts.append(stencil_inputs(x_f, t_f))
            with tc.no_grad():
                angles.append(model.angles(xts[-1][:n_col]))
        # one job: the stencil centres of every seed
        q_qpu, meas_angles, meas_evs = hw.evaluate(tc.cat(angles).numpy(), log["job_ids"], shift=epoch,
                                                   verbose=False)
        hw_angles.append(meas_angles)
        hw_evs.append(meas_evs)
        losses = [qpu_step(model, opt, xt, a, q_qpu[i * n_col:(i + 1) * n_col])
                  for i, (model, opt, xt, a) in enumerate(zip(models, opts, xts, angles))]
        hist["lr"].append(lr)
        hist["loss_qpu"].append(losses)
        hist["loss_monitor"].append(monitor_losses())

        if args.recal_every and epoch % args.recal_every == 0:
            coefs, fit_rms, _ = fit_surrogate(np.concatenate([cal_angles, *hw_angles]),
                                              np.concatenate([cal_evs, *hw_evs]), cones, M)
            for model in models:
                model.quantum.set_coefs(coefs)
            log["recal"].append(dict(epoch=epoch, fit_rms=fit_rms,
                                     n_samples=len(cal_angles) + sum(map(len, hw_angles))))
        if epoch % 10 == 0 or epoch == args.epochs:
            rate = (time.time() - t_hw) / epoch
            print(f"  época {epoch}/{args.epochs} | lr {lr:.2f} | loss QPU (mediana das seeds) "
                  f"{np.median(losses):.4e} | loss monitor (modelo) {np.median(hist['loss_monitor'][-1]):.4e} | "
                  f"{rate:.1f} s/época, faltam ~{rate * (args.epochs - epoch) / 60:.0f} min", flush=True)
    log["hw_time_s"] = time.time() - t_hw
    save_log()

    # ---------- 3. validation: the trained networks running on the QPU ----------
    R_val = args.val_replicas or hw.B
    print(f"3/3 validação no QPU: {S} seeds × {N_EVALUATION_POINTS} pontos × {R_val} réplicas, 1 job...")
    with tc.no_grad():
        val_angles = [m.eval().angles(xt_plot) for m in models]
    q_val, _, _ = hw.evaluate(tc.cat(val_angles).numpy(), log["job_ids"], replicas=R_val)

    def errors(u):
        return dict(mse=float(np.mean((u - u_ref_eval) ** 2)), mae=float(np.mean(np.abs(u - u_ref_eval))),
                    relative_l2=relative_l2_error(u, u_ref_eval))

    runs = []
    for i, (seed, model, angles) in enumerate(zip(seeds, models, val_angles)):
        q_qpu = q_val[i * N_EVALUATION_POINTS:(i + 1) * N_EVALUATION_POINTS]
        with tc.no_grad():
            u_qpu = model.from_features(xt_plot, q_qpu).numpy()
            u_sur = model(xt_plot).numpy()
            q_sur = model.quantum(angles).numpy()
            q_exact = cnot_cosine_expectations(angles, model.cnot_matrix)
            u_ideal = model.from_features(xt_plot, q_exact).numpy()
            q_exact = q_exact.numpy()
        run = dict(seed=seed, **errors(u_qpu), surrogate=errors(u_sur), ideal_circuit=errors(u_ideal),
                   rms_q_qpu_vs_surrogate=float(np.sqrt(np.mean((q_qpu - q_sur) ** 2))),
                   rms_q_qpu_vs_exact=float(np.sqrt(np.mean((q_qpu - q_exact) ** 2))),
                   final_loss_qpu=hist["loss_qpu"][-1][i], final_loss_monitor=hist["loss_monitor"][-1][i])
        runs.append(run)
        # per-seed files, with the fields of the notebook's run files
        (out_dir / f"run_{tag(seed)}.json").write_text(json.dumps(
            dict(run, network=FIXED_LAYER, n_cnot_layers=FIXED_LAYER_REPEATS, Tfin=Tfin, n_collocation=n_col,
                 n_qubits=n, backend=backend.name, stopped_at_step=args.epochs, adam_steps=args.epochs,
                 lbfgs_steps=0, model_config=model.config, training_config=log["training_config"],
                 batch_log=str(log_path)), indent=2))
        tc.save(dict(state_dict=model.state_dict(), config=model.config, cones=cones,
                     coefs=[c.tolist() for c in coefs], run=run), out_dir / f"model_{tag(seed)}.pt")
        np.savez(out_dir / f"validation_{tag(seed)}.npz", x=x_plot, angles=angles.numpy(), q_qpu=q_qpu,
                 q_surrogate=q_sur, q_exact=q_exact, u_qpu=u_qpu, u_surrogate=u_sur, u_ideal=u_ideal,
                 u_reference=u_ref_eval)
        plot_solution(x_plot, {f"QPU ({backend.name})": u_qpu, "measured model": u_sur,
                               "same weights, ideal circuit": u_ideal}, ref=u_ref_eval,
                      save_path=out_dir / f"solution_{tag(seed)}.png")
        print(f"  seed {seed}: MSE QPU {run['mse']:.3e} | modelo medido {run['surrogate']['mse']:.3e} | "
              f"circuito ideal {run['ideal_circuit']['mse']:.3e} | L2 relativo QPU {run['relative_l2']:.3e}")

    mse = np.array([r["mse"] for r in runs])
    log.update(runs=runs, mse_mean=float(mse.mean()), mse_min=float(mse.min()), mse_max=float(mse.max()),
               mse_std=float(mse.std()), wall_time_s=time.time() - t0)
    save_log()
    np.savez(out_dir / f"losshist_{batch_tag}.npz", seeds=np.array(seeds), lr=np.array(hist["lr"]),
             loss_qpu=np.array(hist["loss_qpu"]), loss_monitor=np.array(hist["loss_monitor"]))
    plot_loss({f"seed {s} (QPU)": np.array(hist["loss_qpu"])[:, i] for i, s in enumerate(seeds)},
              title=f"Training loss on the QPU -- {backend.name}, T={Tfin}",
              save_path=out_dir / f"loss_qpu_{batch_tag}.png")
    plot_loss({f"seed {s} (monitor)": np.array(hist["loss_monitor"])[:, i] for i, s in enumerate(seeds)},
              title=f"Monitor loss (measured model, fixed grid) -- {backend.name}, T={Tfin}",
              save_path=out_dir / f"loss_monitor_{batch_tag}.png")

    print(f"MSE de u(x,{Tfin}) no QPU sobre {S} seeds: média {log['mse_mean']:.3e} "
          f"(mín {log['mse_min']:.3e}, máx {log['mse_max']:.3e})")
    print(f"tempo total {log['wall_time_s']:.0f} s | {len(log['job_ids'])} jobs | saída em {out_dir}")


if __name__ == "__main__":
    main()
