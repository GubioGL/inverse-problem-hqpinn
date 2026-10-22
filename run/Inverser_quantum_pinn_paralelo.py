"""Parallelized version of Inverser_quantum_pinn.py (inverse problem: learn nu
jointly with u(x,t), quantum bottleneck), following the same run_one /
ProcessPoolExecutor sweep structure as Inverser_classical_pinn_paralelo.py.

Quantum circuit simulation cost (and memory) grows exponentially with
N_QUBITS, so unlike the classical script this one splits the sweep in two:

  - "small" jobs (N_QUBITS <= QUBIT_THRESHOLD): cheap enough to run fully
    parallel, one worker per CPU core (like the classical script). This is
    the default -- just run the script with no arguments.
  - "large" jobs (N_QUBITS > QUBIT_THRESHOLD): run with only a handful of
    workers (default 3) so the machine doesn't run out of memory/CPU and
    lock up. Run these later with `python Inverser_quantum_pinn_paralelo.py
    --mode large`.

Both modes share the same resume-skip logic (based on the .pt checkpoint
already on disk), so you can run "small" now and "large" overnight/later
without redoing anything.
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

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch as tc
import torch.nn as nn
from tqdm import tqdm

# ---- physical parameters ---------------------------------------------------
L, NU = 1.0, 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling ---------------------------------------------------
N_DATA = 100  # fixed data points at t=Tfin, needed to identify nu
COLLOCATION_DISTRIBUTION = "random"        # training collocation points
STOPPING_DISTRIBUTION = "uniform_exact"    # fixed grid used for the stopping/monitor loss
# ---- training ---------------------------------------------------------------
EPOCHS, LR = 20000, 1.0e-2
LBFGS_EPOCHS, LBFGS_LR = 1000, .5
# ---- architecture -------------------------------------------------------------
# pre -> quantum -> post; the trainable "net" bottleneck of the classical
# script is replaced by a fixed (parameter-free) quantum feature map with
# N_QUBITS wires -- swept per job below, like INTRA_NEURON in the classical script.
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
ANGLE_SCALE = math.pi
FIXED_LAYER = "linear_cnot"        # see FIXED_LAYERS below for every option
FIXED_LAYER_REPEATS = 1
INPUT_ROTATION_AXIS = "ry"         # 'rx', 'ry' or 'rz'

OUT_DIR = "data/Inverso/quantum_inverse"

# ---- parallelization --------------------------------------------------------
# Statevector simulation is 2**N_QUBITS in size (plus the autograd graph for
# backprop through 5 forward passes per residual point), so jobs above
# QUBIT_THRESHOLD qubits are run with far fewer concurrent workers.
QUBIT_THRESHOLD = 5
N_WORKERS_SMALL = max(1, (os.cpu_count() or 1) - 1)  # N_QUBITS <= QUBIT_THRESHOLD
N_WORKERS_LARGE = 8                                  # N_QUBITS >  QUBIT_THRESHOLD
MIN_WORKERS_LARGE = 4  # never drop below this in 'large' mode, even with --workers


def set_seed(seed=42):
    np.random.seed(seed)
    tc.manual_seed(seed)

def initial_condition(x, L=1.0):
    """u0(x) = sin(2*pi*x/L). Accepts a NumPy array or a Torch tensor."""
    if isinstance(x, tc.Tensor):
        return tc.sin(2.0 * math.pi * x / L)
    return np.sin(2.0 * np.pi * x / L)

# ----------------------------------------------------------------------
# Reference solution (finite differences, used only for comparison/plots)
# ----------------------------------------------------------------------
def solve_burgers_fd(L=1.0, Tfin=0.1, nu=0.01, Nx=512, dt=None, safety=0.5):
    """Explicit upwind finite-difference solution of u_t + u u_x = nu u_xx
    on a periodic domain [0, L).

    dt: time step. If None (default), it is computed automatically from the
    explicit-scheme stability limits -- dt <= dx^2/(2*nu) for the diffusion
    term and dt <= dx/max|u0| for upwind advection -- scaled by `safety`
    (<1, for margin). A fixed dt that ignores nu (e.g. a large nu with a
    dt tuned for a small one) makes the scheme diverge: u grows without
    bound, overflows, and turns into inf/nan (RuntimeWarnings)."""
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = L / Nx
    u = initial_condition(x, L=L).copy()
    if dt is None:
        dt_diffusion = dx**2 / (2.0 * nu) if nu > 0 else np.inf
        dt_advection = dx / max(np.max(np.abs(u)), 1.0e-12)
        dt = safety * min(dt_diffusion, dt_advection)

    times = np.arange(0.0, Tfin + 0.5 * dt, dt)
    U_hist = np.zeros((len(times), Nx))
    U_hist[0] = u

    for n in range(1, len(times)):
        up, um = np.roll(u, -1), np.roll(u, 1)
        ux_back, ux_fwd = (u - um) / dx, (up - u) / dx
        ux = np.where(u >= 0.0, ux_back, ux_fwd)
        uxx = (up - 2.0 * u + um) / dx**2
        u = u + dt * (-u * ux + nu * uxx)
        U_hist[n] = u

    return x, times, U_hist

def reference_at_time(t_eval, t_ref, x_ref, U_ref, x_query, L=1.0):
    """Periodic interpolation of the FDM reference at the closest stored time."""
    idx = int(np.argmin(np.abs(t_ref - t_eval)))
    xp = np.concatenate([x_ref, [L]])
    fp = np.concatenate([U_ref[idx], [U_ref[idx][0]]])
    return np.interp(x_query.ravel(), xp, fp).reshape(-1, 1)

def reference_pde_residual(x_f, t_f, x_ref, t_ref, U_ref, L, nu, hx, ht):
    """MSE of the PDE residual evaluated on the FDM reference itself, at the
    same collocation points/stencil used by `train`'s loss. Used as the
    training's stopping target: once the PINN's loss reaches this value it
    matches the reference's own discretization error, and pushing it lower
    is just chasing noise in the baseline."""
    x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
    t_f = np.asarray(t_f, dtype=np.float64).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)

    dx = x_ref[1] - x_ref[0]
    dt_ref = t_ref[1] - t_ref[0]
    nx, nt = len(x_ref), len(t_ref)

    def u_ref_interp(x, t):
        x = np.mod(x, L)
        t = np.clip(t, t_ref[0], t_ref[-1])

        fx = x / dx
        ix0 = np.floor(fx).astype(int) % nx
        ix1 = (ix0 + 1) % nx
        wx = fx - np.floor(fx)

        ft = (t - t_ref[0]) / dt_ref
        it0 = np.clip(np.floor(ft).astype(int), 0, nt - 2)
        it1 = it0 + 1
        wt = ft - it0

        u0 = U_ref[it0, ix0] * (1.0 - wx) + U_ref[it0, ix1] * wx
        u1 = U_ref[it1, ix0] * (1.0 - wx) + U_ref[it1, ix1] * wx
        return u0 * (1.0 - wt) + u1 * wt

    u_0 = u_ref_interp(x_f, t_f)
    u_xp, u_xm = u_ref_interp(x_f + hx, t_f), u_ref_interp(x_f - hx, t_f)
    u_tp, u_tm = u_ref_interp(x_f, t_f + ht), u_ref_interp(x_f, t_f - ht)

    u_t = (u_tp - u_tm) / (2.0 * ht)
    u_x = (u_xp - u_xm) / (2.0 * hx)
    u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)
    return float(np.mean((u_t + u_0 * u_x - nu * u_xx) ** 2))

# ----------------------------------------------------------------------
# Collocation points
# ----------------------------------------------------------------------
def sample_collocation_points(n_points, L, T, ht, seed=None, distribution="random"):
    """x in [0,L), t in [ht, T-ht] (interior points, needed for centered
    differences).

    "random": n_points drawn uniformly at random; pass `seed` for a
    reproducible draw.
    "uniform_exact": a uniform grid of exactly n_points (n_t =
    round(sqrt(n_points)) values along t, n_x = ceil(n_points/n_t) along x),
    used here as the fixed validation/stopping set.
    """
    if distribution == "random":
        generator = tc.Generator().manual_seed(seed) if seed is not None else None
        x = tc.rand((n_points, 1), generator=generator) * L
        t = ht + tc.rand((n_points, 1), generator=generator) * (T - 2.0 * ht)
    elif distribution == "uniform_exact":
        n_t = max(1, round(math.sqrt(n_points)))
        n_x = max(1, math.ceil(n_points / n_t))
        x_lin = tc.linspace(0.0, L, n_x + 1)[:-1]
        t_lin = tc.linspace(ht, T - ht, n_t)
        x_grid, t_grid = tc.meshgrid(x_lin, t_lin, indexing="ij")
        x = x_grid.reshape(-1, 1)[:n_points]
        t = t_grid.reshape(-1, 1)[:n_points]
    else:
        raise ValueError("distribution must be 'random' or 'uniform_exact'.")
    return x, t

# ----------------------------------------------------------------------
# Fixed (parameter-free) entangling layer, expressed with PennyLane gates.
# Mirrors `_apply_fixed_gate_layer` in main/models_pennylane.py gate-for-gate.
# ----------------------------------------------------------------------
FIXED_LAYERS = (
    "linear_cnot", "reverse_linear_cnot", "ring_cnot", "brickwork_cnot",
    "symmetric_cnot_ladder", "cz_ring", "ghz_fanout", "swap_network",
)

def _apply_fixed_gate_layer(n_qubits, layer_kind):
    if n_qubits < 2:
        return
    if layer_kind == "linear_cnot":
        for c in range(n_qubits - 1):
            qml.CNOT(wires=[c, c + 1])
    elif layer_kind == "reverse_linear_cnot":
        for c in range(n_qubits - 1, 0, -1):
            qml.CNOT(wires=[c, c - 1])
    elif layer_kind == "ring_cnot":
        for c in range(n_qubits - 1):
            qml.CNOT(wires=[c, c + 1])
        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])
    elif layer_kind == "brickwork_cnot":
        for c in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[c, c + 1])
        for c in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[c, c + 1])
    elif layer_kind == "symmetric_cnot_ladder":
        for c in range(n_qubits - 1):
            qml.CNOT(wires=[c, c + 1])
        for c in range(n_qubits - 1, 0, -1):
            qml.CNOT(wires=[c, c - 1])
    elif layer_kind == "cz_ring":
        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])
        if n_qubits > 2:
            qml.CZ(wires=[n_qubits - 1, 0])
    elif layer_kind == "ghz_fanout":
        for target in range(1, n_qubits):
            qml.CNOT(wires=[0, target])
    elif layer_kind == "swap_network":
        for q in range(n_qubits - 1):
            qml.SWAP(wires=[q, q + 1])
    else:
        raise ValueError(f"Unknown fixed layer '{layer_kind}'. Options: {FIXED_LAYERS}.")

class FixedQuantumLayer(nn.Module):
    """Parameter-free quantum feature layer on PennyLane's `default.qubit`
    simulator with `diff_method="backprop"`: the whole statevector simulation
    runs as ordinary Torch tensor ops, so autograd differentiates it natively
    (including the 2nd-order derivatives the PDE residual needs) -- no
    parameter-shift circuit re-executions. The batch is passed to the QNode
    at once via PennyLane's parameter broadcasting."""

    def __init__(self, n_qubits, fixed_layer="linear_cnot", fixed_layer_repeats=1,
                 input_rotation_axis="ry"):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.fixed_layer = fixed_layer
        self.fixed_layer_repeats = int(fixed_layer_repeats)

        rot = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}.get(input_rotation_axis.lower())
        if rot is None:
            raise ValueError("input_rotation_axis must be one of: 'rx', 'ry', 'rz'.")

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(angles):
            for q in range(self.n_qubits):
                rot(angles[:, q], wires=q)
            for _ in range(self.fixed_layer_repeats):
                _apply_fixed_gate_layer(self.n_qubits, self.fixed_layer)
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, angles):
        out = self.circuit(angles)
        return tc.stack(out, dim=1).to(dtype=angles.dtype)

class QuantumPINN(nn.Module):
    """Hybrid HPINN: classical pre-net -> bounded rotation angles -> fixed
    (parameter-free) quantum gate layer -> classical post-net. Same shape as
    the classical script's pre -> net -> post, with the trainable "net"
    bottleneck swapped for a fixed quantum feature map; only pre/post and
    `learnable_parameter` (nu) are trained."""

    def __init__(self, pre_neurons, pre_depth, n_qubits, post_neurons, post_depth,
                 L=1.0, Tfin=0.1, nu_init=0.1, angle_scale=math.pi,
                 fixed_layer="linear_cnot", fixed_layer_repeats=1,
                 input_rotation_axis="ry"):
        super().__init__()
        self.L, self.Tfin = float(L), float(Tfin)
        self.angle_scale = float(angle_scale)
        # Kwargs needed to reconstruct this model (cls(**config) must work);
        # `nu_init` is left out on purpose -- it only seeds the training run,
        # the loaded state_dict overwrites `learnable_parameter` anyway.
        self.config = dict(
            pre_neurons=pre_neurons, pre_depth=pre_depth, n_qubits=n_qubits,
            post_neurons=post_neurons, post_depth=post_depth, L=self.L, Tfin=self.Tfin,
            angle_scale=self.angle_scale, fixed_layer=fixed_layer,
            fixed_layer_repeats=fixed_layer_repeats, input_rotation_axis=input_rotation_axis,
        )

        def mlp(n_in, hidden, depth, n_out):
            layers = [nn.Linear(n_in, hidden), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, n_out)]
            return nn.Sequential(*layers)

        self.pre = mlp(3, pre_neurons, pre_depth, n_qubits)
        self.quantum = FixedQuantumLayer(
            n_qubits=n_qubits, fixed_layer=fixed_layer,
            fixed_layer_repeats=fixed_layer_repeats,
            input_rotation_axis=input_rotation_axis,
        )
        self.post = mlp(n_qubits, post_neurons, post_depth, 1)
        self.learnable_parameter = nn.Parameter(tc.tensor(float(nu_init)))

    def forward(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=1,
        )
        angles = self.angle_scale * tc.tanh(self.pre(features))
        q = self.quantum(angles).to(dtype=self.post[0].weight.dtype)
        out = self.post(q)
        return initial_condition(x, L=self.L) + (t / self.Tfin) * out  # hard-constrains u(x,0)=u0(x)

    def show_circuit(self):
        dummy = tc.zeros(1, self.quantum.n_qubits)
        return qml.draw_mpl(self.quantum.circuit)(dummy)

    def save(self, path, training_params=None):
        """Saves weights + config to `path` (state_dict + reconstruction
        kwargs), plus a human-readable `<path>_config.json` alongside it so
        the run configuration can be inspected without loading torch."""
        path = str(path)
        payload = {"class_name": type(self).__name__, "config": self.config, "state_dict": self.state_dict()}
        if training_params is not None:
            payload["training_params"] = training_params
        tc.save(payload, path)

        config_path = os.path.splitext(path)[0] + "_config.json"
        json_payload = {"class_name": type(self).__name__, "config": self.config}
        if training_params is not None:
            json_payload["training_params"] = training_params
        with open(config_path, "w") as f:
            json.dump(json_payload, f, indent=2)

    @classmethod
    def load(cls, path, map_location="cpu"):
        """Reconstructs the model from a checkpoint saved with `save`, ready
        for further training or inference."""
        checkpoint = tc.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

def stencil(x, t, L, hx, ht):
    """Shifted neighbor points x+-hx, t+-ht for the centered-difference stencil."""
    return tc.remainder(x + hx, L), tc.remainder(x - hx, L), t + ht, t - ht

def pde_residual(model, x, xp, xm, t, tp, tm, hx, ht):
    """u_t + u*u_x - nu*u_xx at (x, t), with nu = model.learnable_parameter."""
    u0 = model(tc.cat([x, t], dim=1))
    uxp = model(tc.cat([xp, t], dim=1))
    uxm = model(tc.cat([xm, t], dim=1))
    utp = model(tc.cat([x, tp], dim=1))
    utm = model(tc.cat([x, tm], dim=1))
    u_t = (utp - utm) / (2.0 * ht)
    u_x = (uxp - uxm) / (2.0 * hx)
    u_xx = (uxp - 2.0 * u0 + uxm) / hx**2
    return u_t + u0 * u_x - model.learnable_parameter * u_xx

def train(model, x_f, t_f, epochs, lr, resample_fn=None,
          lbfgs_epochs=1000, lbfgs_lr=0.5, target_loss=None,
          validation_points=None, data_points=None, L=1.0, hx=1.0e-3, ht=1.0e-3):
    """resample_fn: optional callable returning a fresh (x_f, t_f) pair,
    called every AdamW epoch (used to train on freshly drawn random points).
    validation_points: optional fixed (x, t) used to monitor/stop training
    instead of the (possibly resampled) training points.
    data_points: optional fixed (x_data, t_data, u_data) supervised targets,
    added to the training loss as MSE(model(x_data, t_data), u_data). Kept
    fixed across epochs, unlike x_f, t_f which may change every epoch.
    target_loss: stop as soon as the monitored (PDE-residual-only) loss
    reaches this value.
    Returns (model, hist), hist["nu"] holding the learned nu at every step,
    and the best (lowest monitored loss) snapshot loaded into `model`."""

    x_p, x_m, t_p, t_m = stencil(x_f, t_f, L, hx, ht)
    val_stencil = None
    if validation_points is not None:
        x_val, t_val = validation_points
        val_p, val_m, val_tp, val_tm = stencil(x_val, t_val, L, hx, ht)
        val_stencil = (x_val, val_p, val_m, t_val, val_tp, val_tm)

    def data_loss():
        if data_points is None:
            return 0.0
        x_data, t_data, u_data = data_points
        return tc.mean(abs(model(tc.cat([x_data, t_data], dim=1)) - u_data))

    def monitor(train_stencil):
        model.eval()
        with tc.no_grad():
            loss = float(tc.mean(pde_residual(model, *(val_stencil or train_stencil), hx, ht) ** 2))
        model.train()
        return loss

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = tc.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)

    best_state, best_loss = snapshot(), float("inf")
    hist = {"loss": [], "nu": []}
    target_reached = False

    for _ in range(epochs):
        model.train()
        if resample_fn is not None:
            x_f, t_f = resample_fn()
            x_p, x_m, t_p, t_m = stencil(x_f, t_f, L, hx, ht)

        residual = pde_residual(model, x_f, x_p, x_m, t_f, t_p, t_m, hx, ht)
        loss = tc.mean(abs(residual)) + data_loss()
        opt.zero_grad()
        loss.backward()
        tc.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        loss_value = monitor((x_f, x_p, x_m, t_f, t_p, t_m))
        hist["loss"].append(loss_value)
        hist["nu"].append(float(model.learnable_parameter.detach()))
        if loss_value < best_loss:
            best_loss, best_state = loss_value, snapshot()
        if target_loss is not None and loss_value <= target_loss:
            target_reached = True
            break

    if lbfgs_epochs > 0:
        model.load_state_dict(best_state)  # hand L-BFGS the best basin Adam found

        # Once u(x,t) already matches the reference's own precision, freeze
        # it and let L-BFGS refine only nu, to sharpen its last digits --
        # otherwise (target not yet reached) L-BFGS still refines everything.
        only_nu = target_reached
        if only_nu:
            for p in params:
                p.requires_grad_(p is model.learnable_parameter)
            lbfgs_params = [model.learnable_parameter]
        else:
            lbfgs_params = params

        lbfgs_opt = tc.optim.LBFGS(
            lbfgs_params, lr=lbfgs_lr, max_iter=300, history_size=100, line_search_fn="strong_wolfe")
        patience, tol, stall = 500, 1.0e-9, 0

        def closure():
            lbfgs_opt.zero_grad()
            residual = pde_residual(model, x_f, x_p, x_m, t_f, t_p, t_m, hx, ht)
            loss = tc.mean((residual)**2) + data_loss()
            loss.backward()
            return loss

        for _ in range(lbfgs_epochs):
            model.train()
            lbfgs_opt.step(closure)
            loss_value = monitor((x_f, x_p, x_m, t_f, t_p, t_m))
            hist["loss"].append(loss_value)
            hist["nu"].append(float(model.learnable_parameter.detach()))
            if loss_value < best_loss - tol:
                best_loss, best_state, stall = loss_value, snapshot(), 0
            else:
                stall += 1
            # In nu-only mode, u is already at the target precision, so this
            # would trip on the very first iteration and cut the refinement
            # short -- only apply it while still refining the whole network.
            if not only_nu and target_loss is not None and loss_value <= target_loss:
                target_reached = True
                break
            if stall >= patience:
                break

        for p in params:
            p.requires_grad_(True)  # restore full trainability before returning

    model.load_state_dict(best_state)  # never return worse than the best point found
    hist["target_reached"] = target_reached
    hist["best_loss"] = best_loss
    return model, hist

def predict(model, x, t_val):
    """Evaluates the model on a spatial grid x at a fixed time t_val."""
    x = np.asarray(x).reshape(-1, 1)
    xt = tc.tensor(np.hstack([x, t_val * np.ones_like(x)]), dtype=tc.get_default_dtype())
    model.eval()
    with tc.no_grad():
        return model(xt).numpy().reshape(-1, 1)

def relative_l2_error(u_pred, u_ref):
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_loss(histories, title="Training loss", save_path=None):
    plt.figure(figsize=(9, 4.5))
    for label, values in histories.items():
        plt.plot(np.asarray(values), linewidth=2.0, label=label)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.yscale("log")
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_parameter_history(nu_history, target, title="Learned nu", save_path=None):
    """nu_history: learned nu at every training step. target: the true nu,
    drawn as a horizontal reference line."""
    plt.figure(figsize=(9, 4.5))
    plt.plot(nu_history, linewidth=2.0, label="learned nu")
    plt.axhline(target, color="black", linestyle="--", linewidth=2.0, label=f"target nu = {target:g}")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("nu", fontsize=14)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_solution(x, curves, ref=None, ref_label="FDM reference", title="",
                   save_path=None, collocation_x=None):
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.gca()
    if ref is not None:
        ax.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label)
    for label, u in curves.items():
        ax.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label)
    if collocation_x is not None:
        collocation_x = np.asarray(collocation_x).ravel()
        rug_trans = ax.get_xaxis_transform()
        ax.scatter(
            collocation_x, np.full_like(collocation_x, 0.02), transform=rug_trans,
            marker="|", s=200, linewidth=1.2, color="black", clip_on=False,
            label=f"collocation points (N={collocation_x.size})",
        )
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u(x,t)", fontsize=14)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------------
# One run of the inverse problem (one worker job)
# ----------------------------------------------------------------------
def run_one(n_col, n_qubits, Tfin, run_id):
    tc.set_num_threads(1)

    SEED = run_id
    set_seed(SEED)
    # Drawn *after* set_seed so it is reproducible per (run_id) -- unlike the
    # serial script, where NU_INIT was drawn once at import time.
    nu_init = float(np.random.random())

    # Reference solution, used only for comparison/plots and for the fixed
    # data points at t=Tfin, never for the PDE-residual part of training.
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

    model = QuantumPINN(
        pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        n_qubits=n_qubits, post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        L=L, Tfin=Tfin, nu_init=nu_init, angle_scale=ANGLE_SCALE,
        fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
        input_rotation_axis=INPUT_ROTATION_AXIS,
    )

    x_f, t_f = sample_collocation_points(
        n_col, L, Tfin, HT, seed=SEED, distribution=COLLOCATION_DISTRIBUTION)
    resample_fn = lambda: sample_collocation_points(
        n_col, L, Tfin, HT, distribution=COLLOCATION_DISTRIBUTION)
    x_val, t_val = sample_collocation_points(
        n_col, L, Tfin, HT, distribution=STOPPING_DISTRIBUTION)

    # Inverse-problem data: the reference solution at the final time, on a
    # fixed subset of points -- unlike x_f, t_f, this set never changes
    # across epochs. Together with the hard-imposed initial condition, this
    # is what makes nu identifiable (the PDE residual alone can't pin it down).
    data_idx = np.linspace(0, len(x_ref) - 1, N_DATA).astype(int)
    to_tensor = lambda a: tc.tensor(a, dtype=tc.get_default_dtype())
    x_data = to_tensor(x_ref[data_idx].reshape(-1, 1))
    t_data = tc.full_like(x_data, float(t_ref[-1]))
    u_data = to_tensor(U_ref[-1][data_idx].reshape(-1, 1))

    # Stop once the PINN's residual matches the reference's own discretization
    # error at these points -- pushing it lower would just chase FD noise.
    target_loss = reference_pde_residual(x_val, t_val, x_ref, t_ref, U_ref, L=L, nu=NU, hx=HX, ht=HT)

    model, hist = train(
        model, x_f, t_f, epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        lbfgs_epochs=LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        target_loss=target_loss, validation_points=(x_val, t_val),
        data_points=(x_data, t_data, u_data), L=L, hx=HX, ht=HT,
    )

    nu_learned = float(model.learnable_parameter.detach())

    out_dir = os.path.join(OUT_DIR, f"T_{Tfin}")
    tag = f"{n_col}_NQ{n_qubits}_{run_id}"

    model.save(
        os.path.join(out_dir, f"model_{tag}.pt"),
        training_params=dict(
            L=L, TFIN=Tfin, NU=NU, HX=HX, HT=HT,
            N_COLLOCATION=n_col, N_DATA=N_DATA, SEED=SEED,
            COLLOCATION_DISTRIBUTION=COLLOCATION_DISTRIBUTION,
            STOPPING_DISTRIBUTION=STOPPING_DISTRIBUTION,
            EPOCHS=EPOCHS, LR=LR, LBFGS_EPOCHS=LBFGS_EPOCHS, LBFGS_LR=LBFGS_LR,
            PRE_NEURONS=PRE_NEURONS, PRE_DEPTH=PRE_DEPTH,
            N_QUBITS=n_qubits, POST_NEURONS=POST_NEURONS, POST_DEPTH=POST_DEPTH,
            ANGLE_SCALE=ANGLE_SCALE, FIXED_LAYER=FIXED_LAYER,
            FIXED_LAYER_REPEATS=FIXED_LAYER_REPEATS, INPUT_ROTATION_AXIS=INPUT_ROTATION_AXIS,
            NU_INIT=nu_init, NU_LEARNED=nu_learned,
            TARGET_LOSS=target_loss, TARGET_REACHED=hist["target_reached"],
            STOPPED_AT_STEP=len(hist["loss"]), BEST_LOSS=hist["best_loss"],
        ),
    )

    # Full cost-function/nu history (every step, AdamW + L-BFGS), kept
    # separate from the checkpoint so the human-readable metadata above
    # stays small.
    np.savez(
        os.path.join(out_dir, f"losshist_{tag}.npz"),
        loss=np.array(hist["loss"]),
        nu=np.array(hist["nu"]),
        target_loss=target_loss,
        target_reached=hist["target_reached"],
    )

    plot_loss(
        {"Quantum PINN": hist["loss"]}, title=f"Quantum PINN (inverse) loss -- T={Tfin}",
        save_path=os.path.join(out_dir, f"loss_{tag}.png"),
    )
    plot_parameter_history(
        hist["nu"], NU, title=f"Learned viscosity nu -- N_col={n_col}, N_qubits={n_qubits}, T={Tfin}",
        save_path=os.path.join(out_dir, f"nu_history_{tag}.png"),
    )

    u_pred = predict(model, x_plot, Tfin)
    u_ref = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Quantum PINN": u_pred}, ref=u_ref,
        title=f"Quantum PINN vs FDM reference at T={Tfin}",
        save_path=os.path.join(out_dir, f"solution_{tag}.png"),
        collocation_x=x_f.numpy(),
    )

    return n_col, n_qubits, Tfin, run_id, nu_learned


if __name__ == "__main__":
    ################# PARALELIZAÇAO ######################################
    parser = argparse.ArgumentParser(
        description="Parallel sweep for the quantum inverse PINN. Splits jobs by "
                     "N_QUBITS so the heavy (>%d qubits) cases can be run later, "
                     "separately, with fewer workers." % QUBIT_THRESHOLD)
    parser.add_argument(
        "--mode", choices=["small", "large"], default="small",
        help="'small' (default): runs jobs with N_QUBITS <= %d, fully parallel "
             "(one worker per CPU core). 'large': runs jobs with N_QUBITS > %d, "
             "with a small worker count so the machine doesn't run out of "
             "memory/CPU -- run this separately, later." % (QUBIT_THRESHOLD, QUBIT_THRESHOLD))
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override the worker count (default: %d for 'small', %d for 'large'; "
             "'large' never drops below %d, even with this flag)."
             % (N_WORKERS_SMALL, N_WORKERS_LARGE, MIN_WORKERS_LARGE))
    args = parser.parse_args()

    SEEDS_NEW = range(0, 10)  # 10 seeds

    # Bloco A: varredura de N_COLLOCATION com N_QUBITS em [3, 7, 15]
    block_A = [
        (n_col, n_qubits, Tfin, run_id)
        for n_col in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for n_qubits in [3, 7]
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    # Bloco B: varredura de N_QUBITS (1 a 10) com N_COLLOCATION fixo em 10 e 100
    block_B = [
        (n_col, n_qubits, Tfin, run_id)
        for n_col in [10, 100]
        for n_qubits in range(1, 11)
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    jobs_by_key = {}
    for block in (block_A, block_B):
        for job in block:
            jobs_by_key[job] = job
    all_jobs = list(jobs_by_key.values())

    # Split by qubit count: "small" (<= QUBIT_THRESHOLD) runs fully parallel
    # now; "large" (> QUBIT_THRESHOLD) is meant to be run later, on its own,
    # with a much smaller worker count (see --mode above).
    if args.mode == "small":
        jobs = [job for job in all_jobs if job[1] <= QUBIT_THRESHOLD]
        n_workers = args.workers or N_WORKERS_SMALL
    else:
        jobs = [job for job in all_jobs if job[1] > QUBIT_THRESHOLD]
        n_workers = args.workers or N_WORKERS_LARGE
        n_workers = max(n_workers, MIN_WORKERS_LARGE)  # keep >=2 concurrent even with --workers 1
    n_workers = max(1, min(n_workers, os.cpu_count() or n_workers))

    # Retomada apos interrupcao: pula jobs cujo checkpoint (.pt) ja existe em
    # disco, para nao re-treinar o que ja foi concluido antes do PC desligar.
    def _output_path(job):
        n_col, n_qubits, Tfin, run_id = job
        return os.path.join(OUT_DIR, f"T_{Tfin}", f"model_{n_col}_NQ{n_qubits}_{run_id}.pt")

    jobs_done = [job for job in jobs if os.path.exists(_output_path(job))]
    jobs = [job for job in jobs if not os.path.exists(_output_path(job))]

    print(f"Modo: {args.mode} (N_QUBITS {'<=' if args.mode == 'small' else '>'} {QUBIT_THRESHOLD}), "
          f"workers={n_workers}")
    print(f"Ja concluidos (pulados): {len(jobs_done)} runs")
    print(f"Faltando (serao rodados agora): {len(jobs)} runs")

    for _, _, Tfin, _ in jobs:
        os.makedirs(os.path.join(OUT_DIR, f"T_{Tfin}"), exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, *job): job for job in jobs}
        progress = tqdm(as_completed(futures), total=len(jobs), desc="runs", unit="run")
        n_failed = 0
        for future in progress:
            n_col, n_qubits, Tfin, run_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                n_failed += 1
                progress.write(
                    f"FAILED: N_COLLOCATION={n_col} N_QUBITS={n_qubits} "
                    f"Tfin={Tfin} run_id={run_id} -> {exc!r}"
                )
            progress.set_postfix(failed=n_failed)
        if n_failed:
            print(f"{n_failed} run(s) failed -- see FAILED lines above.")
