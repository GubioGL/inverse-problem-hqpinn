"""Hybrid inverse PINN for viscous Burgers (learns nu jointly with u(x,t)) with the
quantum bottleneck running on a real IBM QPU -- single seed.

Same problem, model and loss as Inverser_quantum_exact_pinn_paralelo.py (the loss, FD
stencil, AdamW + L-BFGS loop and seeding are imported from there; the configuration
block below is this script's own). The quantum layer is the response MEASURED on the
QPU (noise included) instead of the exact formula <Z_i> = prod_j cos(phi_j)^M[i,j].

The measured layer (exact, not an approximation of the circuit):
  each phi_j enters a single RY gate (on the hardware: one virtual RZ between fixed
  pulses), so every measured <Z_i>, WITH the device noise, is a trigonometric
  polynomial of degree <= 1 in each angle of the causal cone of qubit i:

      <Z_i>(phi) = sum_c a_{i,c} prod_{j in cone(i)} {1, cos phi_j, sin phi_j}_{c_j}

  For the brickwork CNOT layer the cone has <= 4 qubits -> 3^4 = 81 coefficients.

Pipeline:
  1. calibration (1 QPU job): random angles on every n-qubit chain packed on the chip
     (find_chains: disjoint chains of physically coupled qubits, no SWAPs, best
     calibrated chains first) -> least squares for a_{i,c};
  2. phase 1, simulator (EPOCHS, classical): AdamW (+ optional L-BFGS) with the quantum
     layer replaced by that measured polynomial (HardwareSurrogateLayer), or by the
     ideal formula with SIM_LAYER = "exact";
  3. phase 2, QPU in the loop (HW_EPOCHS, 1 job per epoch, inside a Session): every
     AdamW step measures <Z> on the QPU at the current angles of the collocation points
     and of the data points. The forward pass uses the QPU values; the gradient goes
     through the measured polynomial (straight-through):

         q(phi) = poly(phi) + stopgrad(q_QPU(phi_c) - poly(phi_c)),

     with the same correction on the 5 points of the FD stencil of collocation point c.
     Measuring the stencil neighbours separately would put the shot noise on u_xx times
     HX^-2 ~ 2.6e5; with a common correction the differences come from the polynomial and
     the values (u in u*u_x, data misfit) from the QPU. Every HW_RECAL_EVERY epochs the
     polynomial is refitted with the calibration + all phase-2 measurements, so it
     tracks the device where the network actually sends its angles;
  4. validation (1 QPU job): the trained network runs on the QPU -- angles of the
     validation points on the chains, several replicas per point -> u(x, Tfin) from
     the hardware, compared with the exact Cole-Hopf solution.

Phase 2 writes a checkpoint every HW_CKPT_EVERY epochs; --resume continues from it
(skipping calibration and phase 1) if the Session expires or a job fails.

Uso (ambiente conda `qiskit`):
    python Inverser_quantum_hardware_pinn.py --fake --hw-epochs 20      # FakeBerlin, local
    python Inverser_quantum_hardware_pinn.py                            # QPU real: BACKEND (ibm_berlin)
    python Inverser_quantum_hardware_pinn.py --backend ibm_aachen       # outro QPU
    python Inverser_quantum_hardware_pinn.py --calibration <arquivo.npz> ...  # reusa a calibração
    python Inverser_quantum_hardware_pinn.py ... --resume               # continua a fase 2
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
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch as tc
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeAachen, FakeBerlin

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src" / "quantum_hardware"))

# Only functions/classes whose every setting is passed as an argument are imported: the
# configuration below is the single source of truth for this script.
from Inverser_quantum_exact_pinn_paralelo import (  # noqa: E402  -- same problem / protocol
    ExactQuantumLayer, QuantumExactInversePINN,
    exact_solution, initial_condition, mae_objective, plot_loss, plot_parameter_history,
    plot_solution, relative_l2_error, sample_collocation_points, set_seed, stencil_inputs, train,
)
from main.functions import compose_cnot_gates, repeated_cnot_network_matrix  # noqa: E402
from regression_hardware_nqubits import INSTANCE, find_chains  # noqa: E402

# ---- QPU ----------------------------------------------------------------------
BACKEND = "ibm_berlin"            # IBM Nighthawk r1 (120 qubits, square lattice, CZ)
FAKE_BACKENDS = {"ibm_berlin": FakeBerlin, "ibm_aachen": FakeAachen}   # --fake: local noise model
# find_chains score = readout + sx errors of the chain + its CZ error; on ibm_berlin the median
# 2-qubit chain is ~0.05, and a few chains with a broken qubit score > 1: they would bias the
# chip-average polynomial, so chains above this score are dropped
MAX_CHAIN_SCORE = 0.3
# The backend's error table misses some bad chains (on ibm_aachen one chain with a good score
# had contrast -0.56 on its CZ output and alone pushed the calibration misfit from ~0.03 to
# 0.11), so the calibration job also screens them: a chain whose misfit to the chip-average
# polynomial is above CHAIN_OUTLIER x the median misfit is dropped (0 = keep all)
CHAIN_OUTLIER = 3.0
# ---- error suppression / mitigation (Estimator options) -------------------------
# The SAME options are used in calibration, phase 2 and validation: the polynomial is fitted
# to whatever the mitigated device returns, so it must keep its structure. Every preset below
# keeps <Z_i> a degree-1 trigonometric polynomial in each angle -- TREX is a per-qubit affine
# readout correction, twirling averages channels, DD inserts identities on idle qubits, ZNE
# folds only the CZs and its LINEAR extrapolator is a fixed linear combination of the noise
# factors (the exponential one is nonlinear in the data and would break the polynomial).
# Noise that is stable in time is already absorbed by the measured polynomial; what these
# options buy is less chain-to-chain spread and drift, and <Z> closer to the ideal circuit.
_TREX = dict(measure_mitigation=True)
_SUPPRESS = dict(dynamical_decoupling=dict(enable=True, sequence_type="XpXm"),
                 twirling=dict(enable_gates=True, enable_measure=True))
MITIGATIONS = {
    "none": dict(resilience_level=0),                                   # raw counts
    "trex": dict(resilience_level=0, resilience=_TREX),                 # = Estimator default (level 1)
    "suppress": dict(resilience_level=0, resilience=_TREX, **_SUPPRESS),   # + DD + Pauli twirling
    "zne": dict(resilience_level=0, **_SUPPRESS,                        # + ZNE, ~3x circuits
                resilience=dict(**_TREX, zne_mitigation=True,
                                zne=dict(amplifier="gate_folding", noise_factors=(1, 3, 5),
                                         extrapolator="linear"))),
}
MITIGATION = "suppress"

OUT_DIR = ROOT / "data" / "Inverso" / "quantum_inverse_hardware"
EXACT_DIR = ROOT / "data" / "Inverso" / "quantum_inverse_exact"   # runs of the exact script, for comparison
MAX_CONE = 6                                          # 3^6 = 729 coefficients per output


# ---- physical parameters ---------------------------------------------------
L, NU = 1.0, 0.01
# ---- discretization of the PDE residual (FD stencil applied to the network)
HX, HT = L / 512, 1.0e-3
# ---- data / collocation -----------------------------------------------------
N_DATA = 100      # data points at t = Tfin (exact solution), needed to identify nu
DATA_GRID = 512   # data x positions: N_DATA points of this uniform grid (same as v1)
COLLOCATION_DISTRIBUTION = "random"      # training collocation points
MONITOR_DISTRIBUTION = "uniform_exact"   # fixed grid used to pick the best snapshot
# ---- training ---------------------------------------------------------------
# phase 1, simulator (classical): AdamW on network + nu jointly, then optional L-BFGS
EPOCHS, LR = 500, 1.0e-1
SIM_LAYER = "surrogate"                  # 'surrogate': polynomial measured on the chip; 'exact': ideal formula
LBFGS_EPOCHS, LBFGS_LR = 0, 0.5
LBFGS_ONLY_NU = False                    # True: L-BFGS refines nu alone (network frozen)
# phase 2, QPU in the loop: new AdamW (shot noise -> smaller LR; L-BFGS line search
# does not work with noisy evaluations), 1 QPU job per epoch
HW_EPOCHS, HW_LR = 1000, 1.0e-3
HW_RECAL_EVERY = 50                      # refit the polynomial with the phase-2 data (0 = never)
HW_CKPT_EVERY = 10                       # checkpoint for --resume
SESSION_MAX_TIME = "12m"               # safety limit of the Session (phase 2 + validation)
# ---- architecture -------------------------------------------------------------
# pre -> quantum -> post; the trainable "net" bottleneck of the classical
# script is replaced by a fixed (parameter-free) exact CNOT-cosine feature
# map with N_QUBITS wires -- swept per job below, like INTRA_NEURON in the
# classical script.
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
ANGLE_SCALE = math.pi
FIXED_LAYER = "brickwork_cnot"        # see FIXED_LAYERS below for every option
FIXED_LAYER_REPEATS = 1
# The exact cosine-product formula only holds for Rx/Ry product encodings --
# Rz acting on |0> leaves <Z_i> independent of the angle, so it is rejected
# below rather than silently producing a degenerate feature map.
INPUT_ROTATION_AXIS = "ry"         # 'rx' or 'ry'
# ---- run (overridable from the command line) ----------------------------------
N_QUBITS = 2
N_COLLOCATION = 10                 # training (collocation) points of the PDE residual


def build_model(n_qubits, Tfin, nu_init):
    """Same as the exact script's build_model, but with THIS file's configuration."""
    return QuantumExactInversePINN(
        pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        n_qubits=n_qubits, post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        L=L, Tfin=Tfin, nu_init=nu_init, angle_scale=ANGLE_SCALE,
        fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
        input_rotation_axis=INPUT_ROTATION_AXIS,
    )


def problem_setup(n_col, Tfin, device):
    """Same as the exact script's problem_setup, but with THIS file's configuration:
    (monitor_points, data_points), data = exact solution at t = Tfin."""
    x_mon, t_mon = sample_collocation_points(n_col, L, Tfin, HT, distribution=MONITOR_DISTRIBUTION)
    x_grid = np.linspace(0.0, L, DATA_GRID, endpoint=False)
    x_np = x_grid[np.linspace(0, DATA_GRID - 1, N_DATA).astype(int)]
    ic_err = np.max(np.abs(exact_solution(x_np, 0.0, nu=NU, L=L) - initial_condition(x_np, L)))
    if ic_err > 1.0e-8:
        raise RuntimeError(f"exact_solution does not reproduce u0 (max err {ic_err:.1e}).")
    u_np = exact_solution(x_np, Tfin, nu=NU, L=L)
    to_tensor = lambda a: tc.tensor(a.reshape(-1, 1), dtype=tc.get_default_dtype(), device=device)
    x_data = to_tensor(x_np)
    t_data = tc.full_like(x_data, float(Tfin))
    return (x_mon.to(device), t_mon.to(device)), (x_data, t_data, to_tensor(u_np))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-qubits", type=int, default=N_QUBITS)
    p.add_argument("--n-col", type=int, default=N_COLLOCATION, help="collocation points (N_COLLOCATION)")
    p.add_argument("--tfin", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default=BACKEND, help="nome do QPU ('least_busy': o menos ocupado)")
    p.add_argument("--fake", action="store_true", help="ensaio local com o fake do --backend (FakeBerlin)")
    p.add_argument("--blocks", type=int, default=None, help="cadeias usadas (padrão: todas que cabem)")
    p.add_argument("--max-chain-score", type=float, default=MAX_CHAIN_SCORE,
                   help="descarta cadeias com nota de erro acima disso")
    p.add_argument("--chain-outlier", type=float, default=CHAIN_OUTLIER,
                   help="descarta cadeias com desajuste na calibração > isso × a mediana (0 = nunca)")
    p.add_argument("--mitigation", choices=tuple(MITIGATIONS), default=MITIGATION,
                   help="supressão/mitigação de erros do Estimator (mesma em todos os jobs)")
    p.add_argument("--time-test", type=int, default=0, metavar="K",
                   help="só mede o tempo: K jobs do tamanho de uma época para cada --mitigation, e sai")
    p.add_argument("--cal-samples", type=int, default=1024,
                   help="amostras de calibração por saída (≈ blocos × conjuntos de parâmetros)")
    p.add_argument("--val-points", type=int, default=100, help="pontos x da validação no QPU, em t = Tfin")
    p.add_argument("--val-replicas", type=int, default=None,
                   help="réplicas por ponto na validação (padrão: uma em cada cadeia)")
    p.add_argument("--precision", type=float, default=1 / 32, help="precisão por valor esperado (shots ≈ 1/p²)")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="fase 1: épocas de AdamW no simulador")
    p.add_argument("--lr", type=float, default=LR, help="fase 1: taxa de aprendizado")
    p.add_argument("--sim-layer", choices=("surrogate", "exact"), default=SIM_LAYER,
                   help="fase 1: polinômio medido no chip ou fórmula ideal")
    p.add_argument("--lbfgs-epochs", type=int, default=LBFGS_EPOCHS, help="fase 1: L-BFGS após o AdamW")
    p.add_argument("--hw-epochs", type=int, default=HW_EPOCHS, help="fase 2: épocas com o QPU (1 job cada)")
    p.add_argument("--hw-lr", type=float, default=HW_LR, help="fase 2: taxa de aprendizado")
    p.add_argument("--recal-every", type=int, default=HW_RECAL_EVERY,
                   help="fase 2: reajusta o polinômio a cada N épocas (0 = nunca)")
    p.add_argument("--ckpt-every", type=int, default=HW_CKPT_EVERY, help="fase 2: checkpoint a cada N épocas")
    p.add_argument("--max-time", default=SESSION_MAX_TIME, help="tempo máximo da Session (trava de segurança)")
    p.add_argument("--no-session", action="store_true", help="fase 2 em job mode (cada job entra na fila)")
    p.add_argument("--calibration", default=None, help="reusa um calibration_*.npz (pula o job 1)")
    p.add_argument("--resume", action="store_true", help="continua a fase 2 do checkpoint desta configuração")
    return p.parse_args()


def get_backend(args):
    if args.fake:
        if args.backend not in FAKE_BACKENDS:
            raise SystemExit(f"--fake: sem backend fake para '{args.backend}' ({', '.join(FAKE_BACKENDS)}).")
        return FAKE_BACKENDS[args.backend]()
    service = QiskitRuntimeService(instance=INSTANCE)
    if args.backend == "least_busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(args.backend)


# ----------------------------------------------------------------------
# Fixed CNOT network as a gate list (same networks as main/functions.py)
# ----------------------------------------------------------------------
def cnot_gate_list(n, kind, repeats):
    kind = kind.lower()
    if kind == "inverse_linear_cnot":
        kind = "reverse_linear_cnot"
    forward = [(c, c + 1) for c in range(n - 1)]
    reverse = [(c, c - 1) for c in range(n - 1, 0, -1)]
    networks = {
        "linear_cnot": forward,
        "reverse_linear_cnot": reverse,
        "brickwork_cnot": [(c, c + 1) for c in range(0, n - 1, 2)] + [(c, c + 1) for c in range(1, n - 1, 2)],
        "symmetric_cnot_ladder": forward + reverse,
    }
    if kind not in networks:
        raise SystemExit(f"'{kind}' não é suportado no hardware: só redes em cadeia ({', '.join(networks)}).")
    gates = networks[kind] * repeats
    # must be exactly the circuit whose binary matrix the exact model uses
    assert (compose_cnot_gates(n, gates) == repeated_cnot_network_matrix(n, kind, repeats)).all()
    return gates


def causal_cones(n, gates):
    """Qubits whose gates/noise can reach the measurement of each qubit."""
    cones = []
    for i in range(n):
        cone = {i}
        for a, b in reversed(gates):
            if a in cone or b in cone:
                cone |= {a, b}
        cones.append(sorted(cone))
    return cones


# ----------------------------------------------------------------------
# Measured quantum layer: degree-1 trigonometric polynomial on each cone
# ----------------------------------------------------------------------
def cone_features(angles, cone):
    """angles (N, n) -> (N, 3^len(cone)) products of {1, cos, sin} over the cone."""
    feat = np.ones((len(angles), 1))
    for j in cone:
        basis = np.stack([np.ones(len(angles)), np.cos(angles[:, j]), np.sin(angles[:, j])], axis=1)
        feat = (feat[:, :, None] * basis[:, None, :]).reshape(len(angles), -1)
    return feat


def fit_surrogate(angles, evs, cones, M):
    """Least squares per output. Returns coefficients, fit RMS and the coefficient of
    the ideal term prod_{j in supp M_i} cos(phi_j) (1 on a noiseless device)."""
    coefs, rms, contrast = [], [], []
    for i, cone in enumerate(cones):
        F = cone_features(angles, cone)
        c, *_ = np.linalg.lstsq(F, evs[:, i], rcond=None)
        coefs.append(c)
        rms.append(float(np.sqrt(np.mean((F @ c - evs[:, i]) ** 2))))
        ideal_digits = [1 if M[i, j] else 0 for j in cone]       # cos on the support, 1 elsewhere
        contrast.append(float(c[int("".join(map(str, ideal_digits)), 3)]))
    return coefs, rms, contrast


class HardwareSurrogateLayer(nn.Module):
    """Drop-in replacement for ExactQuantumLayer: the polynomial measured on the QPU.
    Cones are padded to the same size with a dummy angle whose basis is (1, 0, 0).
    `offset` (N, n), when set, is added to the output: the QPU correction of phase 2."""

    def __init__(self, cones, coefs):
        super().__init__()
        n = len(cones)
        self.K = K = max(len(c) for c in cones)
        self.cone_sizes = [len(c) for c in cones]
        padded_cones = [c + [n] * (K - len(c)) for c in cones]
        self.register_buffer("cones", tc.as_tensor(padded_cones, dtype=tc.long))
        self.register_buffer("coef", tc.zeros(n, 3 ** K))
        self.register_buffer("dummy", tc.tensor([[1.0, 0.0, 0.0]]))
        self.offset = None
        self.set_coefs(coefs)

    def set_coefs(self, coefs):
        padded = np.zeros(tuple(self.coef.shape))
        for i, (size, c) in enumerate(zip(self.cone_sizes, coefs)):
            # pads are the least significant digits, always 0 (the "1" of the dummy basis)
            padded[i, np.arange(len(c)) * 3 ** (self.K - size)] = c
        self.coef.copy_(tc.as_tensor(padded, dtype=self.coef.dtype))

    def forward(self, angles):
        N = angles.shape[0]
        basis = tc.stack([tc.ones_like(angles), tc.cos(angles), tc.sin(angles)], dim=-1)  # (N, n, 3)
        basis = tc.cat([basis, self.dummy.to(angles.dtype).expand(N, 1, 3)], dim=1)
        b = basis[:, self.cones, :]                                                     # (N, n, K, 3)
        feat = b[:, :, 0, :]
        for m in range(1, self.K):
            feat = (feat.unsqueeze(-1) * b[:, :, m, None, :]).flatten(-2)
        out = (feat * self.coef.to(angles.dtype)).sum(-1)                               # (N, n)
        return out if self.offset is None else out + self.offset


def network_angles(model, x, t):
    """Rotation angles the pre-net sends to the circuit (same as model.forward)."""
    features = tc.cat([tc.sin(2.0 * math.pi * x / model.L), tc.cos(2.0 * math.pi * x / model.L),
                       t / model.Tfin], dim=1)
    return model.angle_scale * tc.tanh(model.pre(features))


def network_output(model, x, t, q):
    """u(x,t) from given quantum features q (same as model.forward after the circuit)."""
    q = tc.as_tensor(q, dtype=model.post[0].weight.dtype)
    return initial_condition(x, L=model.L) + (t / model.Tfin) * model.post(q)


# ----------------------------------------------------------------------
# Hardware: B chains of n qubits packed on the chip, one RY + CNOT network each
# ----------------------------------------------------------------------
def point_grid(n_points, B, replicas=1, shift=0):
    """(T, B) array: point evaluated at (parameter set, chain); -1 = idle slot.
    Replica r of point p runs on chain (p + r + shift) % B, so with replicas = B every
    point is measured once on every chain -- its average matches the chip-average model
    fitted in the calibration (chain-to-chain differences cancel instead of biasing
    points). Changing `shift` every epoch rotates the points over the chains."""
    chain_points = [[] for _ in range(B)]
    for r in range(replicas):
        for p in range(n_points):
            chain_points[(p + r + shift) % B].append(p)
    grid = np.full((max(map(len, chain_points)), B), -1)
    for c, pts in enumerate(chain_points):
        grid[:len(pts), c] = pts
    return grid


def screen_chains(angles, evs, B, cones, M, factor):
    """Misfit of each chain to the chip-average polynomial (calibration rows are ordered
    (parameter set, chain), so row r ran on chain r % B). Returns (keep mask, misfits)."""
    coefs, _, _ = fit_surrogate(angles, evs, cones, M)
    res = np.stack([cone_features(angles, cone) @ c - evs[:, i]
                    for i, (cone, c) in enumerate(zip(cones, coefs))], axis=1)
    chain = np.arange(len(angles)) % B
    misfit = np.array([np.sqrt(np.mean(res[chain == b] ** 2)) for b in range(B)])
    keep = misfit <= factor * np.median(misfit) if factor > 0 else np.ones(B, bool)
    return keep, misfit


class PackedCircuit:
    def __init__(self, backend, chains, n, gates, precision, fake, options):
        B = len(chains)
        self.B, self.n = B, n
        phi = ParameterVector("phi", B * n)
        qc = QuantumCircuit(B * n)
        for b in range(B):
            for k in range(n):
                qc.ry(phi[b * n + k], b * n + k)
            for c, t in gates:
                qc.cx(b * n + c, b * n + t)
        observables = [[SparsePauliOp.from_sparse_list([("Z", [b * n + k], 1.0)], num_qubits=qc.num_qubits)
                        for k in range(n)] for b in range(B)]
        initial_layout = [q for chain in chains for q in chain]
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=initial_layout)
        self.isa = pm.run(qc)
        self.obs = [[op.apply_layout(self.isa.layout) for op in row] for row in observables]
        order = list(self.isa.parameters)
        self.phi_idx = np.array([order.index(p) for p in phi]).reshape(B, n)
        self.n_params = len(order)
        self.n_qubits_used = qc.num_qubits
        self.depth = self.isa.depth()
        g2 = next(g for g in ("cz", "ecr", "cx") if g in backend.target.operation_names)
        self.n_2q, self.n_2q_expected = self.isa.count_ops().get(g2, 0), B * len(gates)
        self.g2 = g2
        self.precision = precision
        self.fake = fake
        # chains are not entangled with each other: MPS handles 100+ qubits easily
        self.mode = AerSimulator.from_backend(backend, method="matrix_product_state") if fake else backend
        self.set_options(options)     # job mode until set_mode(session)

    def set_options(self, options):
        self.options = options
        # local (Aer) mode has no mitigation: only the hardware gets the options
        self.estimator = Estimator(mode=self.mode, options=None if self.fake else options)

    def set_mode(self, mode):
        self.mode = mode
        self.set_options(self.options)

    def run(self, angle_sets, job_ids, verbose=True):
        """angle_sets (T, B, n) -> <Z> (T, B, n). One job."""
        vals = np.zeros((len(angle_sets), self.n_params))
        vals[:, self.phi_idx] = angle_sets
        job = self.estimator.run([(self.isa, self.obs, vals[:, None, None, :])], precision=self.precision)
        self.last_job = job
        job_ids.append(job.job_id())
        if verbose:
            print(f"  job {job.job_id()} enviado ({len(angle_sets)} conjuntos de parâmetros)", flush=True)
        return np.asarray(job.result()[0].data.evs)

    def evaluate(self, angles, job_ids, replicas=1, shift=0, verbose=True):
        """angles (P, n) -> (<Z> per point averaged over replicas (P, n), angles and <Z> of
        every busy slot, for the calibration fit). One job."""
        grid = point_grid(len(angles), self.B, replicas, shift)
        busy = grid >= 0
        angle_sets = np.where(busy[..., None], angles[np.maximum(grid, 0)], 0.0)    # (T, B, n)
        evs = self.run(angle_sets, job_ids, verbose)
        q = np.stack([evs[grid == p].mean(0) for p in range(len(angles))])
        return q, angle_sets[busy], evs[busy]


def hardware_step(model, opt, params, hw, xt_f, n_f, xt_data, u_data, shift, job_ids):
    """One AdamW step with the QPU in the loop (see module docstring, phase 2).
    Returns (loss on the QPU values, angles and <Z> measured in this step)."""
    centers = tc.cat([xt_f[:n_f], xt_data])          # stencil centres + data points
    with tc.no_grad():
        angles = network_angles(model, centers[:, :1], centers[:, 1:])
        q_poly = model.quantum(angles)
    q_qpu, meas_angles, meas_evs = hw.evaluate(angles.numpy(), job_ids, shift=shift, verbose=False)
    delta = tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly
    # stencil_inputs stacks 5 blocks of n_f rows (centre, x+-hx, t+-ht), then the data rows
    model.quantum.offset = tc.cat([delta[:n_f].repeat(5, 1), delta[n_f:]])
    try:
        loss = mae_objective(model, xt_f, n_f, xt_data, u_data, HX, HT)
    finally:
        model.quantum.offset = None
    opt.zero_grad()
    loss.backward()
    tc.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return float(loss.detach()), meas_angles, meas_evs


def time_test(hw, backend, args, T_hw):
    """--time-test K: K jobs of one phase-2 epoch (T_hw random parameter sets) for every
    mitigation preset, in one Session, timed from submission to result (queue + run +
    latency, as in the training loop). Session usage is billed by wall-clock time."""
    rng = np.random.default_rng(0)
    use_session = not args.fake and not args.no_session
    print(f"Teste de tempo: {args.time_test} jobs × {len(MITIGATIONS)} presets, {T_hw} conjuntos por job")
    with (Session(backend=backend, max_time=args.max_time) if use_session else nullcontext()) as session:
        if use_session:
            hw.set_mode(session)
            print("Session ID:", session.session_id)
        hw.run(rng.uniform(-np.pi, np.pi, (T_hw, hw.B, hw.n)), [], verbose=False)   # warm-up: Session start
        for name, options in MITIGATIONS.items():
            hw.set_options(options)
            walls, usage = [], []
            for _ in range(args.time_test):
                t = time.time()
                hw.run(rng.uniform(-np.pi, np.pi, (T_hw, hw.B, hw.n)), [], verbose=False)
                walls.append(time.time() - t)
                if not args.fake:
                    usage.append(hw.last_job.usage())
            w = float(np.median(walls))
            print(f"  {name:9s} {w:6.1f} s/época | uso de QPU {np.median(usage) if usage else float('nan'):5.1f} s/job"
                  f" | {args.hw_epochs} épocas ≈ {w * args.hw_epochs / 3600:.1f} h de Session", flush=True)


def main():
    args = parse_args()
    n, n_col, Tfin, seed = args.n_qubits, args.n_col, args.tfin, args.seed
    out_dir = OUT_DIR / f"T_{Tfin}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- circuit structure ----------
    gates = cnot_gate_list(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    cones = causal_cones(n, gates)
    if max(map(len, cones)) > MAX_CONE:
        raise SystemExit(f"Cone causal de {max(map(len, cones))} qubits em '{FIXED_LAYER}': "
                         f"3^{max(map(len, cones))} coeficientes por saída é demais para calibrar.")

    backend = get_backend(args)
    options = MITIGATIONS[args.mitigation]
    tag = f"{n_col}_NQ{n}_{seed}_{backend.name}_{args.mitigation}_E{args.epochs}_HW{args.hw_epochs}"
    log_path = out_dir / f"run_{tag}.json"
    ckpt_path = out_dir / f"checkpoint_{tag}.pt"

    # ---------- chip packing: chains of the checkpoint / calibration file, or the best now ----------
    ckpt = cal = None
    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume: checkpoint {ckpt_path} não existe.")
        ckpt = tc.load(ckpt_path, weights_only=False)
        chains = ckpt["log"]["chains"]
        print(f"Retomando de {ckpt_path} (fase 2, época {ckpt['epoch']}/{args.hw_epochs})")
    elif args.calibration:
        cal = np.load(args.calibration)
        cal_backend = str(cal["backend"]) if "backend" in cal else Path(args.calibration).stem.split("_E")[0]
        if not cal_backend.endswith(backend.name):    # old files: the backend is the end of the tag
            raise SystemExit(f"--calibration foi medida em '{cal_backend}', não em {backend.name}.")
        cal_mitigation = str(cal["mitigation"]) if "mitigation" in cal else "trex"   # old files: Estimator default
        if cal_mitigation != args.mitigation:
            raise SystemExit(f"--calibration foi medida com --mitigation {cal_mitigation}, não {args.mitigation}.")
        chains = cal["chains"].tolist()     # the polynomial was measured on these qubits
    else:
        chains, scores, _ = find_chains(backend, n)
        kept = [c for c, s in zip(chains, scores) if s <= args.max_chain_score]   # sorted best first
        print(f"{len(kept)} de {len(chains)} cadeias com nota ≤ {args.max_chain_score} "
              f"(descartadas: {', '.join(f'{s:.3f}' for s in scores[len(kept):]) or 'nenhuma'})")
        chains = kept[:args.blocks] if args.blocks else kept
    if not chains:
        raise SystemExit(f"Nenhuma cadeia de {n} qubits conectados em {backend.name}.")
    hw = PackedCircuit(backend, chains, n, gates, args.precision, args.fake, options)
    shots = int(np.ceil(args.precision ** -2))
    T_cal = math.ceil(args.cal_samples / hw.B)

    def plan():
        """Parameter sets per job of validation and of one phase-2 epoch, for the current chains."""
        R = args.val_replicas or hw.B
        return R, point_grid(args.val_points, hw.B, R).shape[0], point_grid(n_col + N_DATA, hw.B).shape[0]

    R_val, T_val, T_hw = plan()
    print(f"Backend: {backend.name} | rede {FIXED_LAYER} × {FIXED_LAYER_REPEATS}, {n} qubits, "
          f"cone causal ≤ {max(map(len, cones))} -> ≤ {3 ** max(map(len, cones))} coeficientes por saída")
    print(f"{hw.B} cadeias × {n} qubits = {hw.n_qubits_used} de {backend.num_qubits} qubits | "
          f"profundidade ISA {hw.depth} | {hw.n_2q} portas {hw.g2} | mitigação '{args.mitigation}'")
    if hw.n_2q != hw.n_2q_expected:
        print(f"AVISO: esperado {hw.n_2q_expected} portas {hw.g2}: o transpilador inseriu SWAPs.")
    print(f"QPU ({shots} shots por conjunto): calibração 1 job × {T_cal} conjuntos | "
          f"fase 2 {args.hw_epochs} jobs × {T_hw} conjuntos ({n_col} colocação + {N_DATA} dados) | "
          f"validação 1 job × {T_val} conjuntos ({args.val_points} pontos × {R_val} réplicas)")

    if args.time_test:
        time_test(hw, backend, args, T_hw)
        return

    log = ckpt["log"] if ckpt else dict(
        backend=backend.name, n_qubits=n, n_col=n_col, Tfin=Tfin, seed=seed, NU=NU,
        fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS, blocks=hw.B,
        mitigation=args.mitigation, mitigation_options=options,
        chains=chains, shots=shots, cones=cones, job_ids=[], session_id=None,
        epochs=args.epochs, lr=args.lr, sim_layer=args.sim_layer, lbfgs_epochs=args.lbfgs_epochs,
        hw_epochs=args.hw_epochs, hw_lr=args.hw_lr, recal_every=args.recal_every, recal=[],
        val_points=args.val_points, val_replicas=R_val)

    def save_log():
        log_path.write_text(json.dumps(log, indent=2))

    t0 = time.time()
    monitor_points, data_points = problem_setup(n_col, Tfin, "cpu")
    x_mon, t_mon = monitor_points
    xt_mon, n_mon = stencil_inputs(x_mon, t_mon, L, HX, HT), x_mon.shape[0]
    x_data, t_data, u_data = data_points
    xt_data = tc.cat([x_data, t_data], dim=1)
    resample_fn = lambda: sample_collocation_points(n_col, L, Tfin, HT, distribution=COLLOCATION_DISTRIBUTION)
    x_val = np.linspace(0.0, L, args.val_points, endpoint=False).reshape(-1, 1)
    x_t = tc.tensor(x_val, dtype=tc.get_default_dtype())
    t_t = tc.full_like(x_t, float(Tfin))
    u_ref = exact_solution(x_val, Tfin, nu=NU, L=L)

    if ckpt:
        # ---------- resume: calibration + phase 1 + the first epochs of phase 2 are in the checkpoint ----------
        cal_angles, cal_evs, coefs = ckpt["cal_angles"], ckpt["cal_evs"], ckpt["coefs"]
        model = build_model(n, Tfin, ckpt["nu_init"])
        model.quantum = HardwareSurrogateLayer(cones, coefs)
        model.load_state_dict(ckpt["model"])
        hist, x_f = ckpt["hist"], ckpt["x_f"]
        hw_angles, hw_evs, start = ckpt["hw_angles"], ckpt["hw_evs"], ckpt["epoch"]
    else:
        # ---------- 1. calibration on the QPU ----------
        if cal is not None:
            cal_angles, cal_evs = cal["angles"], cal["evs"]
            log["calibration_file"] = str(args.calibration)
            print(f"Calibração reaproveitada de {args.calibration} ({len(cal_angles)} amostras)")
        else:
            print("1/4 calibração no QPU...")
            rng = np.random.default_rng(seed)
            angle_sets = rng.uniform(-np.pi, np.pi, size=(T_cal, hw.B, n))
            evs = hw.run(angle_sets, log["job_ids"])
            cal_angles, cal_evs = angle_sets.reshape(-1, n), evs.reshape(-1, n)
            cal_path = out_dir / f"calibration_{tag}.npz"
            np.savez(cal_path, angles=cal_angles, evs=cal_evs, chains=np.array(chains), backend=backend.name,
                     mitigation=args.mitigation)
            log["calibration_file"] = str(cal_path)
            save_log()  # the QPU data are safe on disk before the (long) training starts

        # chains that do not follow the chip-average polynomial are left out of everything below
        keep, misfit = screen_chains(cal_angles, cal_evs, hw.B, cones, M, args.chain_outlier)
        log["chain_misfit"] = misfit.tolist()
        if not keep.all():
            print(f"  cadeias descartadas pela calibração (desajuste > {args.chain_outlier} × mediana "
                  f"{np.median(misfit):.3f}): " + ", ".join(f"{chains[b]} ({misfit[b]:.3f})"
                                                           for b in np.flatnonzero(~keep)))
            rows = keep[np.arange(len(cal_angles)) % hw.B]
            cal_angles, cal_evs = cal_angles[rows], cal_evs[rows]
            chains = [c for c, k in zip(chains, keep) if k]
            hw = PackedCircuit(backend, chains, n, gates, args.precision, args.fake, options)
            R_val, T_val, T_hw = plan()
            log.update(chains=chains, blocks=hw.B, val_replicas=R_val,
                       dropped_chains=[c for c, k in zip(log["chains"], keep) if not k])

        coefs, fit_rms, contrast = fit_surrogate(cal_angles, cal_evs, cones, M)
        ideal = np.prod(np.where(M.astype(bool)[None], np.cos(cal_angles)[:, None, :], 1.0), axis=2)
        log.update(fit_rms=fit_rms, contrast=contrast,
                   rms_hw_vs_exact=float(np.sqrt(np.mean((cal_evs - ideal) ** 2))))
        save_log()
        print(f"  ajuste: RMS do resíduo {np.mean(fit_rms):.4f} (ruído de shot ≈ {args.precision:.4f}) | "
              f"contraste do termo ideal {np.min(contrast):.3f}–{np.max(contrast):.3f} (ideal = 1) | "
              f"RMS hardware − fórmula exata {log['rms_hw_vs_exact']:.4f}")

        # ---------- 2. phase 1: the original protocol on the simulator ----------
        print(f"2/4 fase 1 (simulador, camada '{args.sim_layer}'): AdamW {args.epochs} épocas "
              f"+ L-BFGS {args.lbfgs_epochs} (seed {seed})...")
        set_seed(seed)                                     # same initialization as run_one(seed)
        nu_init = float(np.random.random())
        model = build_model(n, Tfin, nu_init)
        surrogate = HardwareSurrogateLayer(cones, coefs)
        if args.sim_layer == "surrogate":
            model.quantum = surrogate
        x_f, t_f = sample_collocation_points(n_col, L, Tfin, HT, seed=seed, distribution=COLLOCATION_DISTRIBUTION)
        t_train = time.time()
        model, hist = train(model, x_f, t_f, epochs=args.epochs, lr=args.lr, resample_fn=resample_fn,
                            lbfgs_epochs=args.lbfgs_epochs, lbfgs_lr=LBFGS_LR, monitor_points=monitor_points,
                            data_points=data_points, L=L, hx=HX, ht=HT, only_nu=LBFGS_ONLY_NU)
        model.quantum = surrogate                          # phase 2 and validation use the measured layer
        with tc.no_grad():
            u_sim = model(tc.cat([x_t, t_t], 1)).numpy()
        hist["hw_loss"] = [float("nan")] * len(hist["loss"])
        log.update(nu_init=nu_init, nu_after_sim=float(model.learnable_parameter.detach()),
                   best_loss_sim=hist.pop("best_loss"), rel_l2_after_sim=relative_l2_error(u_sim, u_ref),
                   sim_time_s=time.time() - t_train)
        print(f"  nu = {log['nu_after_sim']:.5f} (alvo {NU}) | melhor loss {log['best_loss_sim']:.4e} | "
              f"L2 de u(x,{Tfin}) no modelo medido {log['rel_l2_after_sim']:.3e} | {log['sim_time_s']:.0f} s")
        hw_angles, hw_evs, start = [], [], 0

    # ---------- 3. phase 2: QPU in the loop ----------
    params = list(model.parameters())
    opt = tc.optim.AdamW(params, lr=args.hw_lr, weight_decay=1.0e-4)
    if ckpt:
        opt.load_state_dict(ckpt["opt"])

    def save_ckpt(epoch):
        tc.save(dict(epoch=epoch, model=model.state_dict(), opt=opt.state_dict(), hist=hist, x_f=x_f,
                     nu_init=log["nu_init"], cal_angles=cal_angles, cal_evs=cal_evs, coefs=coefs,
                     hw_angles=hw_angles, hw_evs=hw_evs, log=log), ckpt_path)
        save_log()

    if not ckpt:
        save_ckpt(0)        # phase 1 is not repeated by --resume

    use_session = not args.fake and not args.no_session and start < args.hw_epochs
    with (Session(backend=backend, max_time=args.max_time) if use_session else nullcontext()) as session:
        if use_session:
            hw.set_mode(session)
            log["session_id"] = session.session_id
            print("Session ID:", session.session_id)  # GUARDE: dá para ver o uso no dashboard da IBM
        if start < args.hw_epochs:
            print(f"3/4 fase 2 (QPU no loop): AdamW épocas {start + 1}–{args.hw_epochs}, lr {args.hw_lr}, "
                  f"1 job por época{', em Session' if use_session else ''}...")
        t_hw = time.time()
        for epoch in range(start + 1, args.hw_epochs + 1):
            x_f, t_f = resample_fn()
            xt_f = stencil_inputs(x_f, t_f, L, HX, HT)
            loss_qpu, a, e = hardware_step(model, opt, params, hw, xt_f, n_col, xt_data, u_data,
                                           shift=epoch, job_ids=log["job_ids"])
            hw_angles.append(a)
            hw_evs.append(e)
            with tc.no_grad():
                loss_mon = float(mae_objective(model, xt_mon, n_mon, xt_data, u_data, HX, HT))
            hist["loss"].append(loss_mon)
            hist["hw_loss"].append(loss_qpu)
            hist["nu"].append(float(model.learnable_parameter.detach()))

            if args.recal_every and epoch % args.recal_every == 0:
                coefs, fit_rms, _ = fit_surrogate(np.concatenate([cal_angles, *hw_angles]),
                                                  np.concatenate([cal_evs, *hw_evs]), cones, M)
                model.quantum.set_coefs(coefs)
                log["recal"].append(dict(epoch=epoch, fit_rms=fit_rms,
                                         n_samples=len(cal_angles) + sum(map(len, hw_angles))))
            if epoch % 10 == 0 or epoch == args.hw_epochs:
                rate = (time.time() - t_hw) / (epoch - start)
                print(f"  época {epoch}/{args.hw_epochs} | loss QPU {loss_qpu:.4e} | loss monitor (modelo) "
                      f"{loss_mon:.4e} | nu {hist['nu'][-1]:.5f} | {rate:.1f} s/época, "
                      f"faltam ~{rate * (args.hw_epochs - epoch) / 60:.0f} min", flush=True)
            if epoch % args.ckpt_every == 0 or epoch == args.hw_epochs:
                save_ckpt(epoch)

        nu_learned = float(model.learnable_parameter.detach())
        log.update(nu_learned=nu_learned, hw_time_s=log.get("hw_time_s", 0.0) + time.time() - t_hw)
        save_log()
        print(f"  nu aprendido = {nu_learned:.5f} (alvo {NU})")

        # ---------- 4. validation: the trained network running on the QPU ----------
        print("4/4 validação no QPU...")
        with tc.no_grad():
            angles = network_angles(model, x_t, t_t).numpy()                     # (P, n)
            u_sur = model(tc.cat([x_t, t_t], 1)).numpy()
            q_sur = model.quantum(tc.as_tensor(angles)).numpy()
            q_exact = ExactQuantumLayer(n, FIXED_LAYER, FIXED_LAYER_REPEATS,
                                        INPUT_ROTATION_AXIS)(tc.as_tensor(angles)).numpy()
            u_ideal = network_output(model, x_t, t_t, q_exact).numpy()
        q_qpu, _, _ = hw.evaluate(angles, log["job_ids"], replicas=R_val)
    with tc.no_grad():
        u_qpu = network_output(model, x_t, t_t, q_qpu).numpy()

    log.update(
        rel_l2_surrogate=relative_l2_error(u_sur, u_ref), rel_l2_qpu=relative_l2_error(u_qpu, u_ref),
        rel_l2_ideal_circuit=relative_l2_error(u_ideal, u_ref),
        rms_q_qpu_vs_surrogate=float(np.sqrt(np.mean((q_qpu - q_sur) ** 2))),
        rms_q_qpu_vs_exact=float(np.sqrt(np.mean((q_qpu - q_exact) ** 2))),
        wall_time_s=log.get("wall_time_s", 0.0) + time.time() - t0,
    )

    # ---------- comparison with the exact-formula run of the same seed, if it exists ----------
    exact_cfg = EXACT_DIR / f"T_{Tfin}" / f"model_{n_col}_NQ{n}_{seed}_config.json"
    if exact_cfg.exists():
        tp = json.loads(exact_cfg.read_text())["training_params"]
        log["exact_run"] = dict(nu_learned=tp["NU_LEARNED"], rel_l2=tp["REL_L2_U_T"])
    save_log()

    # ---------- outputs ----------
    tc.save(dict(state_dict=model.state_dict(), config=model.config, cones=cones,
                 coefs=[c.tolist() for c in coefs], log=log), out_dir / f"model_{tag}.pt")
    np.savez(out_dir / f"losshist_{tag}.npz", loss=np.array(hist["loss"]), hw_loss=np.array(hist["hw_loss"]),
             nu=np.array(hist["nu"]), sim_epochs=len(hist["loss"]) - args.hw_epochs)
    np.savez(out_dir / f"validation_{tag}.npz", x=x_val, angles=angles, q_qpu=q_qpu, q_surrogate=q_sur,
             q_exact=q_exact, u_qpu=u_qpu, u_surrogate=u_sur, u_ideal=u_ideal, u_exact=u_ref)
    plot_loss({"monitor (measured model)": hist["loss"], "training batch on the QPU": hist["hw_loss"]},
              title=f"Hardware PINN (inverse) loss -- {backend.name}, T={Tfin}",
              save_path=out_dir / f"loss_{tag}.png")
    plot_parameter_history(hist["nu"], NU, title=f"Learned nu -- {backend.name}, N_col={n_col}, "
                                                 f"N_qubits={n}, T={Tfin}",
                           save_path=out_dir / f"nu_history_{tag}.png")
    plot_solution(x_val, {f"QPU ({backend.name})": u_qpu, "measured model": u_sur,
                          "same weights, ideal circuit": u_ideal}, ref=u_ref,
                  title=f"Hardware PINN at T={Tfin}: rel. L2 QPU = {log['rel_l2_qpu']:.2e}",
                  save_path=out_dir / f"solution_{tag}.png", collocation_x=x_f.numpy())

    print(f"nu: após simulador {log['nu_after_sim']:.5f} -> após QPU {nu_learned:.5f} "
          f"(alvo {NU}, erro {abs(nu_learned - NU) / NU:.1%})")
    print(f"L2 relativo de u(x,{Tfin}): QPU {log['rel_l2_qpu']:.3e} | modelo medido {log['rel_l2_surrogate']:.3e} "
          f"(após simulador {log['rel_l2_after_sim']:.3e}) | circuito ideal {log['rel_l2_ideal_circuit']:.3e}")
    print(f"RMS <Z>: QPU vs modelo medido {log['rms_q_qpu_vs_surrogate']:.4f} | "
          f"QPU vs fórmula exata {log['rms_q_qpu_vs_exact']:.4f}")
    if "exact_run" in log:
        print(f"Script exato, mesma seed: nu = {log['exact_run']['nu_learned']:.5f}, "
              f"L2 = {log['exact_run']['rel_l2']:.3e}")
    print(f"tempo total {log['wall_time_s']:.0f} s | {len(log['job_ids'])} jobs | saída em {out_dir}")


if __name__ == "__main__":
    main()
