"""Parallelized classical inverse PINN for viscous Burgers (learns nu jointly
with u(x,t)): classical pre-net -> trainable bottleneck -> classical post-net.

v2 -- same training protocol as Inverser_quantum_exact_pinn_paralelo.py, so
the two models differ only in the bottleneck:
  * NU = 0.01;
  * data at t = Tfin come from the EXACT Cole-Hopf solution (no finite-
    difference data -> no numerical-viscosity bias in the data);
  * loss = MAE(PDE residual) + MAE(data), in every phase;
  * no target_loss / early stopping: AdamW runs all EPOCHS, keeping the best
    snapshot of the monitored loss (the same MAE objective, evaluated on a
    fixed monitor grid + the data points);
  * final phase: L-BFGS on the whole network + nu (LBFGS_ONLY_NU = True
    refines nu alone, with the network frozen).
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

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from scipy.special import ive
from tqdm import tqdm

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
EPOCHS, LR = 20000, 1.0e-2               # phase 1: AdamW, network + nu jointly
LBFGS_EPOCHS, LBFGS_LR = 1000, 0.5       # phase 2: L-BFGS
LBFGS_ONLY_NU = False                    # True: phase 2 refines nu alone (network frozen)
# ---- architecture (activation: tanh, see `mlp` in ClassicalPINN) ------------
PRE_NEURONS, PRE_DEPTH = 10, 1
INTRA_NEURONS, INTRA_DEPTH = 10, 1        # width/depth around the bottleneck, fixed
POST_NEURONS, POST_DEPTH = 10, 1

SAVE_EXTRAS = True  # also save the model checkpoint + full loss/nu history
OUT_DIR = "data/Inverso/classical_inverse"   # new folder: the resume logic skips existing .pt files

# ---- device / parallelization -----------------------------------------------
DEVICE = tc.device("cpu")
N_WORKERS = 12


def set_seed(seed=42):
    np.random.seed(seed)
    tc.manual_seed(seed)

def initial_condition(x, L=1.0):
    """u0(x) = sin(2*pi*x/L). Accepts a NumPy array or a Torch tensor."""
    if isinstance(x, tc.Tensor):
        return tc.sin(2.0 * math.pi * x / L)
    return np.sin(2.0 * np.pi * x / L)

# ----------------------------------------------------------------------
# Exact solution (Cole-Hopf): source of the training data and of the plots
# ----------------------------------------------------------------------
def exact_solution(x, t, nu=NU, L=1.0):
    """Exact solution of u_t + u u_x = nu u_xx on [0, L) periodic with
    u(x,0) = sin(2 pi x/L), via the Cole-Hopf transform u = -2 nu phi_x/phi,
    phi_t = nu phi_xx, phi(x,0) ∝ exp(a cos(2 pi x/L)), a = L/(4 pi nu):

        phi(x,t) = I_0(a) + 2 sum_n I_n(a) exp(-nu k_n^2 t) cos(k_n x),  k_n = 2 pi n/L.

    Evaluated in float64 with exponentially scaled Bessel functions (the common
    factor e^{-a} cancels in u). Accurate for nu >~ 0.005 (a <~ 16); for
    smaller nu, phi near x = L/2 is ~e^{-2a} times its mean and the series
    loses precision -- use the Gauss-Hermite integral form (Basdevant et al.,
    Computers & Fluids 14, 1986) instead. x, t: broadcastable arrays/scalars."""
    a = L / (4.0 * np.pi * nu)
    if a > 16.0:
        raise ValueError(f"nu={nu} too small for the Bessel series (a={a:.1f} > 16).")
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    shape = np.broadcast(x, t).shape
    xf, tf = np.broadcast_to(x, shape).ravel(), np.broadcast_to(t, shape).ravel()
    n = np.arange(1, int(4 * a) + 64)
    k = 2.0 * np.pi * n / L
    weights = ive(n, a) * np.exp(-nu * np.outer(tf, k ** 2))   # (M, n_modes)
    phase = np.outer(xf, k)
    phi = ive(0, a) + 2.0 * np.sum(weights * np.cos(phase), axis=1)
    phi_x = -2.0 * np.sum(weights * k * np.sin(phase), axis=1)
    return (-2.0 * nu * phi_x / phi).reshape(shape)

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
    used here as the fixed monitor set.
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

class ClassicalPINN(nn.Module):
    def __init__(self, pre_neurons, pre_depth, intra_neuron, intra_neurons,
                 intra_depth, post_neurons, post_depth, L=1.0, Tfin=0.1, nu_init=.1):
        super().__init__()
        self.L, self.Tfin = float(L), float(Tfin)

        def mlp(n_in, hidden, depth, n_out):
            layers = [nn.Linear(n_in, hidden), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, n_out)]
            return nn.Sequential(*layers)

        self.pre = mlp(3, pre_neurons, pre_depth, intra_neuron)
        self.net = mlp(intra_neuron, intra_neurons, intra_depth, intra_neuron)
        self.post = mlp(intra_neuron, post_neurons, post_depth, 1)
        self.learnable_parameter = nn.Parameter(tc.tensor(float(nu_init)))

    def forward(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=1,
        )
        out = self.post(self.net(self.pre(features)))
        return initial_condition(x, L=self.L) + (t / self.Tfin) * out  # hard-constrains u(x,0)=u0(x)

# ======================================================================
# Training -- this block is identical in the classical and quantum scripts
# ======================================================================
def stencil(x, t, L, hx, ht):
    """Shifted neighbor points x+-hx, t+-ht for the centered-difference stencil."""
    return tc.remainder(x + hx, L), tc.remainder(x - hx, L), t + ht, t - ht

def stencil_inputs(x, t, L, hx, ht):
    """The 5 stencil points (x,t), (x+-hx,t), (x,t+-ht) stacked into a single
    (5N, 2) batch, so the network runs one forward pass instead of five.
    Also accepts (S, N, 1) inputs (one point set per run) -> (S, 5N, 2)."""
    xp, xm, tp, tm = stencil(x, t, L, hx, ht)
    return tc.cat([tc.cat(pair, dim=-1) for pair in ((x, t), (xp, t), (xm, t), (x, tp), (x, tm))], dim=-2)

def residual_terms(u, n, hx, ht):
    """Splits a forward pass over `stencil_inputs` (first 5n rows of u, along
    the points axis) into (A, B), with the PDE residual
    u_t + u*u_x - nu*u_xx = A - nu*B."""
    u0, uxp, uxm, utp, utm = u.narrow(-2, 0, 5 * n).split(n, dim=-2)
    u_t = (utp - utm) / (2.0 * ht)
    u_x = (uxp - uxm) / (2.0 * hx)
    u_xx = (uxp - 2.0 * u0 + uxm) / hx**2
    return u_t + u0 * u_x, u_xx

def pde_residual(model, xt_stencil, n, hx, ht):
    """u_t + u*u_x - nu*u_xx at the n points stacked in `xt_stencil`."""
    a, b = residual_terms(model(xt_stencil), n, hx, ht)
    return a - model.learnable_parameter * b

def mae_objective(model, xt_f, n_f, xt_data=None, u_data=None, hx=HX, ht=HT):
    """MAE(PDE residual at the n_f stencil points) + MAE(data), from one
    forward pass over the stencil and data points together. Single model."""
    u = model(xt_f if xt_data is None else tc.cat([xt_f, xt_data], dim=0))
    a, b = residual_terms(u, n_f, hx, ht)
    loss = tc.mean(tc.abs(a - model.learnable_parameter * b))
    if xt_data is not None:
        loss = loss + tc.mean(tc.abs(u[5 * n_f:] - u_data))
    return loss

def mae_objective_per_run(model, xt_f, n_f, xt_data=None, u_data=None, hx=HX, ht=HT):
    """`mae_objective` for a batched model: xt_f (S, 5 n_f, 2), returns (S,)."""
    u = model(xt_f if xt_data is None else tc.cat([xt_f, xt_data], dim=1))
    a, b = residual_terms(u, n_f, hx, ht)
    loss = tc.mean(tc.abs(a - model.learnable_parameter * b), dim=(1, 2))
    if xt_data is not None:
        loss = loss + tc.mean(tc.abs(u[:, 5 * n_f:] - u_data), dim=(1, 2))
    return loss

def lbfgs_refine(model, xt_f, n_f, hist, best_state, best_loss, lbfgs_epochs, lbfgs_lr,
                 xt_mon, n_mon, xt_data=None, u_data=None, hx=HX, ht=HT, only_nu=False):
    """L-BFGS phase, run after AdamW on a single model. Starts from
    `best_state` (the best AdamW snapshot), minimizes the same MAE objective on
    the points AdamW ended on (xt_f), appends to `hist`, and loads the best
    state found (by the monitored loss) into `model`. only_nu=True freezes
    the network and refines nu alone. Returns the best monitored loss.

    Note: MAE is not smooth, and L-BFGS/strong-Wolfe assume smoothness; the
    `patience` rule below ends the phase if the line search stalls."""
    def monitor():
        with tc.no_grad():
            return float(mae_objective(model, xt_mon, n_mon, xt_data, u_data, hx, ht))

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    params = list(model.parameters())
    model.load_state_dict(best_state)
    if only_nu:
        for p in params:
            p.requires_grad_(p is model.learnable_parameter)
        lbfgs_params = [model.learnable_parameter]
    else:
        lbfgs_params = params

    opt = tc.optim.LBFGS(lbfgs_params, lr=lbfgs_lr, max_iter=300, history_size=100,
                         line_search_fn="strong_wolfe")
    patience, tol, stall = 50, 1.0e-9, 0

    def closure():
        opt.zero_grad()
        loss = mae_objective(model, xt_f, n_f, xt_data, u_data, hx, ht)
        loss.backward()
        return loss

    for _ in range(lbfgs_epochs):
        opt.step(closure)
        loss_value = monitor()
        if not math.isfinite(loss_value):
            break  # diverged: keep the best state found so far
        hist["loss"].append(loss_value)
        hist["nu"].append(float(model.learnable_parameter.detach()))
        if loss_value < best_loss - tol:
            best_loss, best_state, stall = loss_value, snapshot(), 0
        else:
            stall += 1
        if stall >= patience:
            break

    for p in params:
        p.requires_grad_(True)
    model.load_state_dict(best_state)  # never return worse than the best point found
    return best_loss

def train(model, x_f, t_f, epochs, lr, resample_fn=None, lbfgs_epochs=1000, lbfgs_lr=0.5,
          monitor_points=None, data_points=None, L=1.0, hx=HX, ht=HT, only_nu=False):
    """Single-run training (used by run_one): AdamW for `epochs` steps (no
    early stopping), keeping the best snapshot of the monitored MAE objective,
    then `lbfgs_refine`. resample_fn: optional callable returning a fresh
    (x_f, t_f) pair every AdamW step. monitor_points: fixed (x, t) where the
    monitored objective is evaluated (default: the training points).
    data_points: fixed (x_data, t_data, u_data). Returns (model, hist)."""
    n_f, xt_f = x_f.shape[0], stencil_inputs(x_f, t_f, L, hx, ht)
    x_mon, t_mon = monitor_points if monitor_points is not None else (x_f, t_f)
    n_mon, xt_mon = x_mon.shape[0], stencil_inputs(x_mon, t_mon, L, hx, ht)
    xt_data = u_data = None
    if data_points is not None:
        x_data, t_data, u_data = data_points
        xt_data = tc.cat([x_data, t_data], dim=1)

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    params = list(model.parameters())
    opt = tc.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)
    best_state, best_loss = snapshot(), float("inf")
    hist = {"loss": [], "nu": []}

    for _ in range(epochs):
        if resample_fn is not None:
            x_f, t_f = resample_fn()
            xt_f = stencil_inputs(x_f, t_f, L, hx, ht)
        loss = mae_objective(model, xt_f, n_f, xt_data, u_data, hx, ht)
        opt.zero_grad()
        loss.backward()
        tc.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        with tc.no_grad():
            loss_value = float(mae_objective(model, xt_mon, n_mon, xt_data, u_data, hx, ht))
        hist["loss"].append(loss_value)
        hist["nu"].append(float(model.learnable_parameter.detach()))
        if loss_value < best_loss:
            best_loss, best_state = loss_value, snapshot()

    if lbfgs_epochs > 0:
        best_loss = lbfgs_refine(model, xt_f, n_f, hist, best_state, best_loss, lbfgs_epochs,
                                 lbfgs_lr, xt_mon, n_mon, xt_data, u_data, hx, ht, only_nu)
    else:
        model.load_state_dict(best_state)
    hist["best_loss"] = best_loss
    return model, hist

def train_batched_adam(model, x_f, t_f, epochs, lr, resample_fn=None, monitor_points=None,
                       data_points=None, L=1.0, hx=HX, ht=HT):
    """AdamW phase of `train` for S independent runs stacked in one batched
    model. Summing the per-run losses gives each run exactly its own gradient,
    and AdamW acts elementwise, so each run follows its own optimization;
    gradient clipping and the best snapshot are also per run. No early
    stopping: every run takes all `epochs` steps. The L-BFGS phase is not
    batched (its line search would couple the runs) -- see `lbfgs_refine`.

    x_f, t_f: (S, N, 1); resample_fn returns the same shape. monitor_points and
    data_points are shared by all runs, (N, 1). Returns (model, runs): `model`
    holds each run's best AdamW state and runs[i] = dict(hist, best_loss,
    stop_x, stop_t), stop_x/stop_t (N, 1) the points run i ended on."""
    nu = model.learnable_parameter
    S, device = nu.shape[0], nu.device
    params = list(model.parameters())
    per_run = lambda p: (-1,) + (1,) * (p.dim() - 1)  # view to broadcast a (S,) tensor over p

    n_f, xt_f = x_f.shape[-2], stencil_inputs(x_f, t_f, L, hx, ht)
    if monitor_points is not None:
        x_mon, t_mon = monitor_points
        n_mon = x_mon.shape[-2]
        xt_mon = stencil_inputs(x_mon, t_mon, L, hx, ht).expand(S, -1, -1)
    else:
        n_mon, xt_mon = None, None
    xt_data = u_data = None
    if data_points is not None:
        x_data, t_data, u_data = data_points
        xt_data = tc.cat([x_data, t_data], dim=-1).expand(S, -1, -1)

    def clip_grad_norm_per_run(max_norm=1.0):
        norms = tc.sqrt(sum(p.grad.pow(2).flatten(1).sum(1) for p in params))
        coef = tc.clamp(max_norm / (norms + 1.0e-6), max=1.0)
        for p in params:
            p.grad.mul_(coef.view(per_run(p)))

    opt = tc.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)
    best_params = [p.detach().clone() for p in params]
    best_loss = tc.full((S,), float("inf"), device=device)
    loss_hist = tc.zeros(epochs, S, device=device)
    nu_hist = tc.zeros(epochs, S, device=device)

    for step in range(epochs):
        if resample_fn is not None:
            x_f, t_f = resample_fn()
            xt_f = stencil_inputs(x_f, t_f, L, hx, ht)

        loss = mae_objective_per_run(model, xt_f, n_f, xt_data, u_data, hx, ht).sum()
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_per_run(1.0)
        opt.step()

        with tc.no_grad():
            if xt_mon is not None:
                loss_value = mae_objective_per_run(model, xt_mon, n_mon, xt_data, u_data, hx, ht)
            else:
                loss_value = mae_objective_per_run(model, xt_f, n_f, xt_data, u_data, hx, ht)
        loss_hist[step], nu_hist[step] = loss_value, nu.detach().view(-1)
        improved = loss_value < best_loss
        best_loss = tc.where(improved, loss_value, best_loss)
        for bp, p in zip(best_params, params):
            bp.copy_(tc.where(improved.view(per_run(p)), p.detach(), bp))

    with tc.no_grad():
        for bp, p in zip(best_params, params):
            p.copy_(bp)  # best AdamW state of every run

    loss_hist, nu_hist = loss_hist.cpu().numpy(), nu_hist.cpu().numpy()
    best_loss = best_loss.tolist()
    runs = [
        dict(
            hist={"loss": loss_hist[:, i].tolist(), "nu": nu_hist[:, i].tolist()},
            best_loss=best_loss[i],
            stop_x=xt_f[i, :n_f, 0:1], stop_t=xt_f[i, :n_f, 1:2],
        )
        for i in range(S)
    ]
    return model, runs
# ======================================================================
# end of the block shared with the quantum script
# ======================================================================

# ----------------------------------------------------------------------
# Batched model: several independent runs evaluated together
# ----------------------------------------------------------------------
class BatchedPINN(nn.Module):
    """S independent ClassicalPINNs (same architecture), evaluated together.
    Each Linear layer's weights are stacked along a leading run axis, so one
    batched matmul per layer serves all S networks. Every run keeps its own
    weights and its own nu -- nothing is shared between runs."""

    def __init__(self, models):
        super().__init__()
        ref = models[0]
        self.L, self.Tfin = ref.L, ref.Tfin
        self.layers = []  # (name in ClassicalPINN, tanh after it?)
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for block in ("pre", "net", "post"):
            seq = getattr(ref, block)
            for i, layer in enumerate(seq):
                if not isinstance(layer, nn.Linear):
                    continue
                name = f"{block}.{i}"
                self.layers.append((name, i + 1 < len(seq) and isinstance(seq[i + 1], nn.Tanh)))
                linears = [m.get_submodule(name) for m in models]
                self.weights.append(nn.Parameter(tc.stack([lin.weight.detach().T for lin in linears])))  # (S, in, out)
                self.biases.append(nn.Parameter(tc.stack([lin.bias.detach()[None, :] for lin in linears])))  # (S, 1, out)
        self.learnable_parameter = nn.Parameter(
            tc.stack([m.learnable_parameter.detach() for m in models]).view(-1, 1, 1))  # (S, 1, 1)

    def forward(self, xt):
        """xt: (S, M, 2) -- M points (x, t) for each of the S runs."""
        x, t = xt[..., 0:1], xt[..., 1:2]
        h = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=-1,
        )
        for (_, tanh_after), w, b in zip(self.layers, self.weights, self.biases):
            h = tc.baddbmm(b, h, w)
            if tanh_after:
                h = tc.tanh(h)
        return initial_condition(x, L=self.L) + (t / self.Tfin) * h  # hard-constrains u(x,0)=u0(x)

    def run_state_dict(self, i):
        """Run i's weights in ClassicalPINN's state_dict format, on the CPU."""
        sd = {"learnable_parameter": self.learnable_parameter[i, 0, 0].detach().cpu().clone()}
        for (name, _), w, b in zip(self.layers, self.weights, self.biases):
            sd[f"{name}.weight"] = w[i].detach().T.contiguous().cpu()
            sd[f"{name}.bias"] = b[i, 0].detach().cpu().clone()
        return sd

def predict(model, x, t_val):
    """Evaluates the model on a spatial grid x at a fixed time t_val."""
    x = np.asarray(x).reshape(-1, 1)
    device = next(model.parameters()).device
    xt = tc.tensor(np.hstack([x, t_val * np.ones_like(x)]), dtype=tc.get_default_dtype(), device=device)
    model.eval()
    with tc.no_grad():
        return model(xt).cpu().numpy().reshape(-1, 1)

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
    plt.ylabel("Loss (MAE)", fontsize=14)
    plt.yscale("log")
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_parameter_history(nu_history, target, title="Learned nu", save_path=None):
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

def plot_solution(x, curves, ref=None, ref_label="Exact (Cole-Hopf)", title="",
                  save_path=None, collocation_x=None):
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.gca()
    if ref is not None:
        ax.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label)
    for label, u in curves.items():
        ax.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label)
    if collocation_x is not None:
        collocation_x = np.asarray(collocation_x).ravel()
        ax.scatter(
            collocation_x, np.full_like(collocation_x, 0.02), transform=ax.get_xaxis_transform(),
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
# Runs of the inverse problem (worker jobs)
# ----------------------------------------------------------------------
def build_model(intra_neuro, Tfin, nu_init):
    return ClassicalPINN(
        pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        intra_neuron=intra_neuro, intra_neurons=INTRA_NEURONS, intra_depth=INTRA_DEPTH,
        post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        L=L, Tfin=Tfin, nu_init=nu_init,
    )

def problem_setup(n_col, Tfin, device=DEVICE):
    """What a run needs that doesn't depend on its seed:
    (monitor_points, data_points).

    data_points: the EXACT solution at t = Tfin on N_DATA fixed x positions
    (the same positions as v1: N_DATA points of a DATA_GRID uniform grid).
    Together with the hard-imposed initial condition, this is what makes nu
    identifiable -- the PDE residual alone can't pin it down."""
    x_mon, t_mon = sample_collocation_points(n_col, L, Tfin, HT, distribution=MONITOR_DISTRIBUTION)

    x_grid = np.linspace(0.0, L, DATA_GRID, endpoint=False)
    x_np = x_grid[np.linspace(0, DATA_GRID - 1, N_DATA).astype(int)]
    ic_err = np.max(np.abs(exact_solution(x_np, 0.0) - initial_condition(x_np, L)))
    if ic_err > 1.0e-8:
        raise RuntimeError(f"exact_solution does not reproduce u0 (max err {ic_err:.1e}).")
    u_np = exact_solution(x_np, Tfin)

    to_tensor = lambda a: tc.tensor(a.reshape(-1, 1), dtype=tc.get_default_dtype(), device=device)
    x_data = to_tensor(x_np)
    t_data = tc.full_like(x_data, float(Tfin))
    u_data = to_tensor(u_np)
    return (x_mon.to(device), t_mon.to(device)), (x_data, t_data, u_data)

def save_run(model, hist, n_col, intra_neuro, Tfin, seed, nu_init, x_f):
    """Checkpoint, loss/nu history and plots of one trained run. x_f: the
    run's first collocation draw (drawn on the solution plot). Returns the
    learned nu."""
    nu_learned = float(model.learnable_parameter.detach())
    out_dir = os.path.join(OUT_DIR, f"T_{Tfin}")
    tag = f"{n_col}_INTRA{intra_neuro}_{seed}"

    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_pred = predict(model, x_plot, Tfin)
    u_ref = exact_solution(x_plot, Tfin)
    rel_l2 = relative_l2_error(u_pred, u_ref)

    if SAVE_EXTRAS:
        tc.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "training_params": dict(
                    L=L, TFIN=Tfin, NU=NU, HX=HX, HT=HT,
                    N_COLLOCATION=n_col, N_DATA=N_DATA, SEED=seed,
                    DATA_SOURCE="exact_cole_hopf", LOSS="MAE(residual)+MAE(data)",
                    COLLOCATION_DISTRIBUTION=COLLOCATION_DISTRIBUTION,
                    MONITOR_DISTRIBUTION=MONITOR_DISTRIBUTION,
                    EPOCHS=EPOCHS, LR=LR, LBFGS_EPOCHS=LBFGS_EPOCHS, LBFGS_LR=LBFGS_LR,
                    LBFGS_ONLY_NU=LBFGS_ONLY_NU,
                    PRE_NEURONS=PRE_NEURONS, PRE_DEPTH=PRE_DEPTH,
                    INTRA_NEURON=intra_neuro, INTRA_NEURONS=INTRA_NEURONS, INTRA_DEPTH=INTRA_DEPTH,
                    POST_NEURONS=POST_NEURONS, POST_DEPTH=POST_DEPTH,
                    NU_INIT=nu_init, NU_LEARNED=nu_learned, REL_L2_U_T=rel_l2,
                    STOPPED_AT_STEP=len(hist["loss"]), BEST_LOSS=hist["best_loss"],
                ),
            },
            os.path.join(out_dir, f"model_{tag}.pt"),
        )
        np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"),
                 loss=np.array(hist["loss"]), nu=np.array(hist["nu"]))

    plot_loss({"Classical PINN": hist["loss"]}, title=f"Classical PINN (inverse) loss -- T={Tfin}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(
        hist["nu"], NU, title=f"Learned viscosity nu -- N_col={n_col}, intra={intra_neuro}, T={Tfin}",
        save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(
        x_plot, {"Classical PINN": u_pred}, ref=u_ref,
        title=f"Classical PINN vs exact solution at T={Tfin} (rel. L2 = {rel_l2:.2e})",
        save_path=os.path.join(out_dir, f"solution_{tag}.png"),
        collocation_x=x_f.cpu().numpy())
    return nu_learned

def run_one(n_col, intra_neuro, Tfin, run_id):
    """A single run, trained on its own with `train`. The sweep below uses
    `run_group` (much faster); this is kept for single-run checks."""
    tc.set_num_threads(1)
    SEED = run_id
    set_seed(SEED)
    nu_init = float(np.random.random())
    model = build_model(intra_neuro, Tfin, nu_init).to(DEVICE)

    to_device = lambda pair: tuple(a.to(DEVICE) for a in pair)
    x_f, t_f = to_device(sample_collocation_points(n_col, L, Tfin, HT, seed=SEED,
                                                   distribution=COLLOCATION_DISTRIBUTION))
    resample_fn = lambda: to_device(sample_collocation_points(n_col, L, Tfin, HT,
                                                              distribution=COLLOCATION_DISTRIBUTION))
    monitor_points, data_points = problem_setup(n_col, Tfin)
    model, hist = train(
        model, x_f, t_f, epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        lbfgs_epochs=LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR, monitor_points=monitor_points,
        data_points=data_points, L=L, hx=HX, ht=HT, only_nu=LBFGS_ONLY_NU,
    )
    nu_learned = save_run(model, hist, n_col, intra_neuro, Tfin, SEED, nu_init, x_f)
    return n_col, intra_neuro, Tfin, run_id, nu_learned

def run_group(n_col, intra_neuro, Tfin, run_ids):
    """Every seed in `run_ids` of one (n_col, intra_neuro, Tfin) configuration
    (one worker job): the AdamW phase trains them together as one BatchedPINN,
    then each run goes through `lbfgs_refine` on its own. Per seed, the
    initial network, nu_init and first collocation draw are the same as
    run_one's. Returns [(run_id, nu_learned), ...]."""
    tc.set_num_threads(1)

    nu_inits, models, x_first, t_first = [], [], [], []
    for seed in run_ids:
        set_seed(seed)
        nu_inits.append(float(np.random.random()))
        models.append(build_model(intra_neuro, Tfin, nu_inits[-1]))
        x, t = sample_collocation_points(n_col, L, Tfin, HT, seed=seed, distribution=COLLOCATION_DISTRIBUTION)
        x_first.append(x)
        t_first.append(t)
    model = BatchedPINN(models).to(DEVICE)
    x_f, t_f = tc.stack(x_first).to(DEVICE), tc.stack(t_first).to(DEVICE)  # (S, n_col, 1)

    generator = tc.Generator(device=DEVICE).manual_seed(run_ids[0])
    shape = (len(run_ids), n_col, 1)
    resample_fn = lambda: (
        tc.rand(shape, generator=generator, device=DEVICE) * L,
        HT + tc.rand(shape, generator=generator, device=DEVICE) * (Tfin - 2.0 * HT),
    )

    monitor_points, data_points = problem_setup(n_col, Tfin)
    model, runs = train_batched_adam(
        model, x_f, t_f, epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        monitor_points=monitor_points, data_points=data_points, L=L, hx=HX, ht=HT,
    )

    x_mon, t_mon = monitor_points
    xt_mon = stencil_inputs(x_mon, t_mon, L, HX, HT)
    x_data, t_data, u_data = data_points
    xt_data = tc.cat([x_data, t_data], dim=1)

    results = []
    for i, (seed, run) in enumerate(zip(run_ids, runs)):
        run_model = build_model(intra_neuro, Tfin, nu_inits[i])
        run_model.load_state_dict(model.run_state_dict(i))
        run_model = run_model.to(DEVICE)
        hist, best_loss = run["hist"], run["best_loss"]
        if LBFGS_EPOCHS > 0:
            best_state = {k: v.detach().clone() for k, v in run_model.state_dict().items()}
            best_loss = lbfgs_refine(
                run_model, stencil_inputs(run["stop_x"], run["stop_t"], L, HX, HT), n_col,
                hist, best_state, best_loss, LBFGS_EPOCHS, LBFGS_LR,
                xt_mon, x_mon.shape[0], xt_data, u_data, HX, HT, only_nu=LBFGS_ONLY_NU,
            )
        hist["best_loss"] = best_loss
        nu_learned = save_run(run_model.to("cpu"), hist, n_col, intra_neuro, Tfin, seed,
                              nu_inits[i], x_first[i])
        results.append((seed, nu_learned))
    return results


if __name__ == "__main__":
    ################# PARALELIZAÇAO ######################################
    print(f"Device: {DEVICE} -- using {N_WORKERS} workers ({os.cpu_count()} CPUs available).")
    SEEDS_NEW = range(0, 50)  # 50 seeds

    # Bloco A: varredura de N_COLLOCATION com intra_neuro em [3, 7, 15]
    block_A = [
        # (n_col, intra_neuro, Tfin, run_id)
        # for n_col in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # for intra_neuro in [3, 7, 15]
        # for Tfin in [0.1, 1]
        # for run_id in SEEDS_NEW
    ]

    # Bloco B: varredura de intra_neuro (1 a 15) com N_COLLOCATION fixo em 10 e 100
    block_B = [
        (n_col, intra_neuro, Tfin, run_id)
        for n_col in [10, 100]
        for intra_neuro in [11, 12, 13, 14,]
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    jobs_by_key = {}
    for block in (block_A, block_B):
        for job in block:
            jobs_by_key[job] = job
    jobs = list(jobs_by_key.values())

    # Retomada apos interrupcao: pula jobs cujo checkpoint (.pt) ja existe em
    # disco, para nao re-treinar o que ja foi concluido antes do PC desligar.
    def _output_path(job):
        n_col, intra_neuro, Tfin, run_id = job
        return os.path.join(OUT_DIR, f"T_{Tfin}", f"model_{n_col}_INTRA{intra_neuro}_{run_id}.pt")

    jobs_done = [job for job in jobs if os.path.exists(_output_path(job))]
    jobs = [job for job in jobs if not os.path.exists(_output_path(job))]
    print(f"Ja concluidos (pulados): {len(jobs_done)} runs")
    print(f"Faltando (serao rodados agora): {len(jobs)} runs")

    for _, _, Tfin, _ in jobs:
        os.makedirs(os.path.join(OUT_DIR, f"T_{Tfin}"), exist_ok=True)

    groups = {}
    for n_col, intra_neuro, Tfin, run_id in jobs:
        groups.setdefault((n_col, intra_neuro, Tfin), []).append(run_id)
    print(f"{len(groups)} grupos (n_col, intra_neuro, Tfin) em {N_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_group, *key, run_ids): (key, run_ids) for key, run_ids in groups.items()}
        progress = tqdm(total=len(jobs), desc="runs", unit="run")
        n_failed = 0
        for future in as_completed(futures):
            (n_col, intra_neuro, Tfin), run_ids = futures[future]
            try:
                future.result()
            except Exception as exc:
                n_failed += len(run_ids)
                progress.write(
                    f"FAILED: N_COLLOCATION={n_col} intra_neuro={intra_neuro} "
                    f"Tfin={Tfin} run_ids={list(run_ids)} -> {exc!r}"
                )
            progress.update(len(run_ids))
            progress.set_postfix(failed=n_failed)
        progress.close()
        if n_failed:
            print(f"{n_failed} run(s) failed -- see FAILED lines above.")
