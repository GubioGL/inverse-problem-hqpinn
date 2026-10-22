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

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from tqdm import tqdm

# ---- physical parameters ---------------------------------------------------
L, TFIN, NU = 1.0, 0.1, 0.1
NU_INIT = np.random.random()   # initial guess for the learned viscosity (inverse problem)
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling ---------------------------------------------------
N_COLLOCATION = 100
N_DATA = 100  # fixed data points at t=Tfin, needed to identify nu
SEED = 42
# ---- training ---------------------------------------------------------------
EPOCHS, LR = 20000, 1.0e-2            # phase 1: AdamW, network + nu trained jointly
REFINE_EPOCHS, REFINE_LR = 5000, 1.0e-3  # phase 2: Adam, network frozen, nu-only refinement
# ---- architecture -------------------------------------------------------------
PRE_NEURONS, PRE_DEPTH = 10, 1
INTRA_NEURON = 3          # bottleneck width
INTRA_NEURONS, INTRA_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1

OUT_DIR = "data/Inverso/classical_simple"


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
    bound, overflows, and turns into inf/nan (the RuntimeWarnings seen when
    stepping the loop below)."""
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
          refine_epochs=5000, refine_lr=1.0e-4, target_loss=None,
          validation_points=None, data_points=None, L=1.0, hx=1.0e-3, ht=1.0e-3):
    """Two-phase training, both phases with Adam(W) -- no L-BFGS.

    Phase 1 (AdamW, lr): all trainable parameters (network weights + nu)
    are updated jointly for up to `epochs` steps, stopping early once the
    monitored loss reaches `target_loss`.

    Phase 2 (Adam, refine_lr): the network is frozen (only
    model.learnable_parameter keeps requires_grad=True) and training
    continues on nu alone, for up to `refine_epochs` steps, with a smaller
    learning rate to refine its last digits now that u(x,t) is already a
    good fit. Uses the same (possibly stale) x_f, t_f phase 1 ended on --
    resample_fn is not called again in this phase. Early-stops after
    `patience` steps without improving `best_loss` by more than `tol`.

    resample_fn: optional callable returning a fresh (x_f, t_f) pair,
    called every phase-1 epoch (used to train on freshly drawn random points).
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
        return tc.mean((model(tc.cat([x_data, t_data], dim=1)) - u_data) ** 2)

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

    # ---- Phase 1: AdamW, network + nu trained jointly ----------------------
    for _ in tqdm(range(epochs), desc="Phase 1 (AdamW, net+nu)"):
        model.train()
        if resample_fn is not None:
            x_f, t_f = resample_fn()
            x_p, x_m, t_p, t_m = stencil(x_f, t_f, L, hx, ht)

        residual = pde_residual(model, x_f, x_p, x_m, t_f, t_p, t_m, hx, ht)
        loss = tc.mean(residual ** 2) + data_loss()
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

    # ---- Phase 2: Adam, network frozen, nu-only refinement ------------------
    if refine_epochs > 0:
        model.load_state_dict(best_state)  # refine from the best basin phase 1 found

        for p in params:
            p.requires_grad_(p is model.learnable_parameter)
        nu_opt = tc.optim.Adam([model.learnable_parameter], lr=refine_lr)
        #patience, tol, stall = 50, 1.0e-9, 0

        for _ in tqdm(range(refine_epochs), desc="Phase 2 (Adam, nu only)"):
            model.train()
            residual = pde_residual(model, x_f, x_p, x_m, t_f, t_p, t_m, hx, ht)
            loss = tc.mean(residual ** 2) + data_loss()
            nu_opt.zero_grad()
            loss.backward()
            nu_opt.step()

            loss_value = monitor((x_f, x_p, x_m, t_f, t_p, t_m))
            hist["loss"].append(loss_value)
            hist["nu"].append(float(model.learnable_parameter.detach()))
            # if loss_value < best_loss - tol:
            #     best_loss, best_state, stall = loss_value, snapshot(), 0
            # else:
            #     stall += 1
            # if target_loss is not None and loss_value <= target_loss:
            #     target_reached = True
            #     break
            # if stall >= patience:
            #     break

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
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Reference solution, used only for comparison/plots, never for training.
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=TFIN, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

    model = ClassicalPINN(
        pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        intra_neuron=INTRA_NEURON, intra_neurons=INTRA_NEURONS, intra_depth=INTRA_DEPTH,
        post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        L=L, Tfin=TFIN, nu_init=NU_INIT,
    )

    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, TFIN, HT, seed=SEED, distribution="random")
    resample_fn = lambda: sample_collocation_points(N_COLLOCATION, L, TFIN, HT, distribution="random")
    x_val, t_val = sample_collocation_points(N_COLLOCATION, L, TFIN, HT, distribution="uniform_exact")

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
        refine_epochs=REFINE_EPOCHS, refine_lr=REFINE_LR,
        target_loss=target_loss, validation_points=(x_val, t_val),
        data_points=(x_data, t_data, u_data), L=L, hx=HX, ht=HT,
    )

    nu_learned = float(model.learnable_parameter.detach())
    nu_tol = 0.01 * NU  # within 5% of the true value counts as "reached"
    print(
        f"Stopped after {len(hist['loss'])} steps "
        f"(target {'reached' if hist['target_reached'] else 'NOT reached'}, "
        f"loss={hist['loss'][-1]:.3e}, target={target_loss:.3e})"
    )
    if abs(nu_learned - NU) <= nu_tol:
        print(f"True nu reached: learned nu={nu_learned:.6f}, true nu={NU:.6f} (tol={nu_tol:.1e}).")
    else:
        print(f"True nu NOT reached: learned nu={nu_learned:.6f}, true nu={NU:.6f} (tol={nu_tol:.1e}).")

    # tc.save({"state_dict": model.state_dict(), "hist": hist}, os.path.join(OUT_DIR, "model.pt"))

    plot_loss(
        {"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
        save_path=os.path.join(OUT_DIR, "loss.png"),
    )
    plot_parameter_history(
        hist["nu"], NU, title="Learned viscosity nu vs training steps",
        save_path=os.path.join(OUT_DIR, "nu_history.png"),
    )

    u_pred = predict(model, x_plot, TFIN)
    u_ref = reference_at_time(TFIN, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Classical PINN": u_pred}, ref=u_ref,
        title=f"Classical PINN vs FDM reference at T={TFIN}",
        save_path=os.path.join(OUT_DIR, "solution.png"),
        collocation_x=x_f.numpy(),
    )

    # print(f"Relative L2 error at T={TFIN}: {relative_l2_error(u_pred, u_ref):.6e}")
