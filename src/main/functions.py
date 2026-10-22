import csv
import glob
import json
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from tqdm import tqdm

def set_seed(seed=42):
    np.random.seed(seed)
    tc.manual_seed(seed)

def initial_condition(x, L=1.0):
    """u0(x) = sin(2*pi*x/L). Accepts a NumPy array or a Torch tensor."""
    if isinstance(x, tc.Tensor):
        return tc.sin(2.0 * math.pi * x / L)
    return np.sin(2.0 * np.pi * x / L)

# ----------------------------------------------------------------------
# Reference solution (pure numerical baseline, no neural network involved)
# ----------------------------------------------------------------------
def solve_burgers_fd(L=1.0, Tfin=0.1, nu=0.01, Nx=512, dt=1.0e-4, scheme="upwind"):
    """
    Explicit finite-difference solution of u_t + u u_x = nu u_xx on a
    periodic domain [0, L). Used as the ground-truth reference for every
    PINN comparison, and as the standalone numerical baseline.
    """
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = L / Nx
    u = initial_condition(x, L=L).copy()

    times = np.arange(0.0, Tfin + 0.5 * dt, dt)
    U_hist = np.zeros((len(times), Nx), dtype=np.float64)
    U_hist[0] = u.copy()

    for n in range(1, len(times)):
        up, um = np.roll(u, -1), np.roll(u, 1)

        if scheme == "central":
            ux = (up - um) / (2.0 * dx)
        elif scheme == "upwind":
            ux_back = (u - um) / dx
            ux_fwd = (up - u) / dx
            ux = np.where(u >= 0.0, ux_back, ux_fwd)
        else:
            raise ValueError("scheme must be 'central' or 'upwind'.")

        uxx = (up - 2.0 * u + um) / dx**2
        u = u + dt * (-u * ux + nu * uxx)
        U_hist[n] = u.copy()

    return x, times, U_hist


def reference_at_time(t_eval, t_ref, x_ref, U_ref, x_query, L=1.0):
    """Periodic interpolation of the FDM reference at the closest stored time."""
    idx = int(np.argmin(np.abs(t_ref - t_eval)))
    xp = np.concatenate([x_ref, [L]])
    fp = np.concatenate([U_ref[idx], [U_ref[idx][0]]])
    u = np.interp(x_query.ravel(), xp, fp).reshape(-1, 1)
    return u, float(t_ref[idx])

def reference_pde_residual(x_f, t_f, x_ref, t_ref, U_ref, L, nu, hx, ht):
    """MSE of the PDE residual (u_t + u*u_x - nu*u_xx) evaluated on the FDM
    reference solution itself, at the same collocation points (x_f, t_f) and
    with the exact same finite-difference stencil (hx, ht) used by `train`'s
    loss. U_ref is bilinearly interpolated (periodic in x, linear in t) onto
    x_f, t_f and their +-hx/+-ht neighbors to build that residual.

    Used as a target/stopping value for the PINN training loss: since U_ref
    is itself only an approximate solution (its own FD scheme has nonzero
    truncation error), this residual is never exactly 0. It is the precision
    ceiling of the reference at this (hx, ht, N_COLLOCATION) -- once the
    PINN's residual loss reaches it, the PINN matches the reference's own
    discretization error at these points, and driving the loss lower is
    chasing noise in the baseline rather than the true PDE solution.
    """
    x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
    t_f = np.asarray(t_f, dtype=np.float64).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)

    dx = x_ref[1] - x_ref[0]
    dt_ref = t_ref[1] - t_ref[0]
    nx = len(x_ref)
    nt = len(t_ref)

    def u_ref_interp(x, t):
        # Periodic bilinear interpolation of U_ref over its (x_ref, t_ref) grid.
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

        u00, u01 = U_ref[it0, ix0], U_ref[it0, ix1]
        u10, u11 = U_ref[it1, ix0], U_ref[it1, ix1]
        u0 = u00 * (1.0 - wx) + u01 * wx
        u1 = u10 * (1.0 - wx) + u11 * wx
        return u0 * (1.0 - wt) + u1 * wt

    u_0 = u_ref_interp(x_f, t_f)
    u_xp = u_ref_interp(x_f + hx, t_f)
    u_xm = u_ref_interp(x_f - hx, t_f)
    u_tp = u_ref_interp(x_f, t_f + ht)
    u_tm = u_ref_interp(x_f, t_f - ht)

    u_t = (u_tp - u_tm) / (2.0 * ht)
    u_x = (u_xp - u_xm) / (2.0 * hx)
    u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)

    residual = u_t + u_0 * u_x - nu * u_xx
    return float(np.mean(residual**2))

# ----------------------------------------------------------------------
# Collocation points and physics-informed residual
# ----------------------------------------------------------------------
def sample_collocation_points(n_points, L, T, ht, seed=None, distribution="uniform"):
    """x in [0,L), t in [ht, T-ht] (interior points, needed for centered differences).

    distribution="uniform" (default): points are laid out on a uniform grid
    (not sampled randomly): n_side = round(sqrt(n_points)) values along each
    axis, combined via meshgrid, for n_side**2 total collocation points.
    NOTE: this grid does NOT come out to exactly n_points (see
    "uniform_exact" below for that) -- kept as-is for backward compatibility
    with existing runs/scripts that rely on this exact point count.

    distribution="uniform_exact": same idea as "uniform" (a uniform grid,
    not randomly sampled), but returns EXACTLY n_points: n_t =
    round(sqrt(n_points)) values along t, n_x = ceil(n_points / n_t) along
    x, combined via meshgrid (n_x*n_t >= n_points) and truncated to the
    first n_points. Use this whenever the caller needs a uniform reference
    set directly comparable (point-for-point) to an n_points-sized "random"
    draw -- e.g. a fixed validation/stopping-condition set that must use the
    same N as training regardless of the training collocation distribution.

    distribution="random": n_points points drawn independently from a
    uniform distribution over the domain, x ~ U[0, L) and t ~ U[ht, T-ht].
    Pass `seed` for a reproducible draw. Leave `seed=None` to draw from the
    current (global) torch RNG state instead of a fresh local generator --
    that state advances on every call, so calling this repeatedly with
    seed=None (e.g. once per epoch via `train`'s `resample_fn`) yields a
    different set of random collocation points each time.
    """
    if T <= 2.0 * ht:
        raise ValueError("T must satisfy T > 2*ht.")

    if distribution == "uniform":
        n_side = max(1, round(math.sqrt(n_points)))
        x_lin = tc.linspace(0.0, L, int(n_side/2) + 1)[:-1]
        t_lin = tc.linspace(ht, T - ht, n_side)

        x_grid, t_grid = tc.meshgrid(x_lin, t_lin, indexing="ij")
        x = x_grid.reshape(-1, 1)
        t = t_grid.reshape(-1, 1)
    elif distribution == "uniform_exact":
        n_t = max(1, round(math.sqrt(n_points)))
        n_x = max(1, math.ceil(n_points / n_t))
        x_lin = tc.linspace(0.0, L, n_x + 1)[:-1]
        t_lin = tc.linspace(ht, T - ht, n_t)

        x_grid, t_grid = tc.meshgrid(x_lin, t_lin, indexing="ij")
        x = x_grid.reshape(-1, 1)[:n_points]
        t = t_grid.reshape(-1, 1)[:n_points]
    elif distribution == "random":
        generator = None
        if seed is not None:
            generator = tc.Generator().manual_seed(seed)
        x = tc.rand((n_points, 1), generator=generator) * L
        t = ht + tc.rand((n_points, 1), generator=generator) * (T - 2.0 * ht)
    else:
        raise ValueError("distribution must be 'uniform', 'uniform_exact' or 'random'.")

    return x, t

# ----------------------------------------------------------------------
# Non-uniform mesh finite differences
#
# Both formulas fall back exactly to the standard symmetric central
# difference when h_L == h_R. They operate elementwise on tensors of
# shape (n_points, 1), so a whole (possibly scattered/non-uniform)
# collocation set is differentiated in one shot -- no Python loop over
# the mesh is needed, uniform or not.
# ----------------------------------------------------------------------
# UNUSED (not called anywhere in the codebase, including main/old/) --
# commented out 2026-09-03.
# def nonuniform_first_derivative(f_left, f_center, f_right, h_L, h_R):
#     """f'(x_i) via the 3-point asymmetric central difference on a non-uniform mesh."""
#     return (
#         (h_L / (h_R * (h_L + h_R))) * f_right
#         + ((h_R - h_L) / (h_L * h_R)) * f_center
#         - (h_R / (h_L * (h_L + h_R))) * f_left
#     )

# UNUSED (not called anywhere in the codebase, including main/old/) --
# commented out 2026-09-03.
# def nonuniform_second_derivative(f_left, f_center, f_right, h_L, h_R):
#     """f''(x_i) via the 3-point central difference on a non-uniform mesh."""
#     return (2.0 / (h_L + h_R)) * ((f_right - f_center) / h_R - (f_center - f_left) / h_L)

# ----------------------------------------------------------------------
# Exact fixed-CNOT feature maps through binary algebra
# ----------------------------------------------------------------------
# These are the five CNOT topologies shared by the simulated QuantumPINN and
# its exact cosine-product counterpart. The alias "inverse_linear_cnot" is
# accepted by `fixed_cnot_layer_matrix` for compatibility with older scripts.
EXACT_CNOT_NETWORKS = (
    "linear_cnot",
    "reverse_linear_cnot",
    "ring_cnot",
    "brickwork_cnot",
    "symmetric_cnot_ladder",
)


def _validate_positive_integer(value, name):
    """Return `value` as int after rejecting booleans and non-positive values."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def gf2_matmul(left, right):
    """Multiply two compatible matrices over the binary field GF(2)."""
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("GF(2) operands must be two-dimensional matrices.")
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes for multiplication: {left.shape} and {right.shape}."
        )

    return ((left @ right) % 2).astype(np.uint8)


def gf2_matrix_power(matrix, exponent):
    """Compute a non-negative integer matrix power over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.int64) % 2

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if isinstance(exponent, bool) or not isinstance(exponent, (int, np.integer)):
        raise TypeError("exponent must be an integer.")
    if exponent < 0:
        raise ValueError("exponent must be non-negative.")

    result = np.eye(matrix.shape[0], dtype=np.uint8)
    base = matrix.astype(np.uint8)
    power = int(exponent)

    while power > 0:
        if power & 1:
            result = gf2_matmul(result, base)
        base = gf2_matmul(base, base)
        power >>= 1

    return result


def cnot_gate_matrix(n_qubits, control, target):
    """Return the binary map of one CNOT(control, target), using zero-based labels."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")

    for value, name in ((control, "control"), (target, "target")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
    control, target = int(control), int(target)

    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("control and target must be valid zero-based qubit indices.")
    if control == target:
        raise ValueError("A CNOT must have different control and target qubits.")

    matrix = np.eye(n_qubits, dtype=np.uint8)
    matrix[target, control] = 1
    return matrix


def compose_cnot_gates(n_qubits, gates):
    """Compose CNOTs in circuit order and return b_out = M b_in over GF(2)."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    total = np.eye(n_qubits, dtype=np.uint8)

    for gate_index, gate in enumerate(gates):
        try:
            control, target = gate
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Gate {gate_index} must be a (control, target) pair."
            ) from error
        # Later circuit gates multiply the accumulated transformation from the left.
        total = gf2_matmul(cnot_gate_matrix(n_qubits, control, target), total)

    return total


def linear_cnot_matrix(n_qubits):
    """Forward nearest-neighbor ladder: 0->1->...->n-1."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    return compose_cnot_gates(
        n_qubits,
        [(control, control + 1) for control in range(n_qubits - 1)],
    )


def reverse_linear_cnot_matrix(n_qubits):
    """Reverse nearest-neighbor ladder: n-1->...->1->0."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    return compose_cnot_gates(
        n_qubits,
        [(control, control - 1) for control in range(n_qubits - 1, 0, -1)],
    )


def ring_cnot_matrix(n_qubits):
    """Forward ladder followed by n-1->0 when at least three qubits exist."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    gates = [(control, control + 1) for control in range(n_qubits - 1)]
    if n_qubits > 2:
        gates.append((n_qubits - 1, 0))
    return compose_cnot_gates(n_qubits, gates)


def brickwork_cnot_matrix(n_qubits):
    """Even-bond CNOT sublayer followed by the shifted odd-bond sublayer."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    first = [(control, control + 1) for control in range(0, n_qubits - 1, 2)]
    second = [(control, control + 1) for control in range(1, n_qubits - 1, 2)]
    return compose_cnot_gates(n_qubits, first + second)


def symmetric_cnot_ladder_matrix(n_qubits):
    """Forward CNOT ladder followed by the reverse CNOT ladder."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    return gf2_matmul(
        reverse_linear_cnot_matrix(n_qubits),
        linear_cnot_matrix(n_qubits),
    )


def fixed_cnot_layer_matrix(n_qubits, network_kind):
    """Build the binary matrix of one supported fixed CNOT layer."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    network_kind = str(network_kind).lower()
    if network_kind == "inverse_linear_cnot":
        network_kind = "reverse_linear_cnot"

    builders = {
        "linear_cnot": linear_cnot_matrix,
        "reverse_linear_cnot": reverse_linear_cnot_matrix,
        "ring_cnot": ring_cnot_matrix,
        "brickwork_cnot": brickwork_cnot_matrix,
        "symmetric_cnot_ladder": symmetric_cnot_ladder_matrix,
    }
    if network_kind not in builders:
        options = ", ".join(EXACT_CNOT_NETWORKS)
        raise ValueError(
            f"Unknown CNOT network '{network_kind}'. Available options: {options}."
        )
    return builders[network_kind](n_qubits)


def repeated_cnot_network_matrix(n_qubits, network_kind, n_layers=1):
    """Return the complete binary map for repeated identical CNOT layers."""
    n_qubits = _validate_positive_integer(n_qubits, "n_qubits")
    if isinstance(n_layers, bool) or not isinstance(n_layers, (int, np.integer)):
        raise TypeError("n_layers must be an integer.")
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative.")

    one_layer = fixed_cnot_layer_matrix(n_qubits, network_kind)
    return gf2_matrix_power(one_layer, int(n_layers))


def cnot_cosine_expectations(angle_values, binary_matrix):
    r"""
    Evaluate every <Z_i> exactly for an Rx/Ry product encoding followed by CNOTs.

    If b_out = M b_in over GF(2), row i of M determines

        <Z_i> = product_j cos(angle_j)**M[i,j].

    `angle_values` must have shape (batch_size, n_qubits). The implementation
    uses only differentiable Torch operations and therefore needs neither
    Qiskit, a statevector simulation, shots, nor parameter-shift evaluations.
    """
    if not isinstance(angle_values, tc.Tensor):
        raise TypeError("angle_values must be a torch.Tensor.")
    if angle_values.ndim != 2:
        raise ValueError("angle_values must have shape (batch_size, n_qubits).")

    raw_matrix = tc.as_tensor(binary_matrix, device=angle_values.device)
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] != raw_matrix.shape[1]:
        raise ValueError("binary_matrix must be square.")
    if raw_matrix.dtype != tc.bool and not bool(
        tc.all((raw_matrix == 0) | (raw_matrix == 1)).item()
    ):
        raise ValueError("binary_matrix must contain only zeros and ones.")
    if raw_matrix.shape[1] != angle_values.shape[1]:
        raise ValueError(
            f"Circuit matrix uses {raw_matrix.shape[1]} qubits, but angle_values "
            f"contains {angle_values.shape[1]} angles per sample."
        )
    matrix = raw_matrix.to(dtype=tc.bool)

    cosines = tc.cos(angle_values)
    selected = tc.where(
        matrix.unsqueeze(0),
        cosines.unsqueeze(1),
        tc.ones((), dtype=angle_values.dtype, device=angle_values.device),
    )
    return tc.prod(selected, dim=2)


def cnot_cosine_product_expressions(binary_matrix, angle_symbol="theta"):
    """Return readable product formulas corresponding to the matrix rows."""
    matrix = np.asarray(binary_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("binary_matrix must be square.")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("binary_matrix must contain only zeros and ones.")

    expressions = []
    for row in matrix.astype(bool):
        indices = np.flatnonzero(row)
        factors = [f"cos({angle_symbol}_{index + 1})" for index in indices]
        expressions.append(" * ".join(factors) if factors else "1")
    return expressions

# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
def train(model, x_f, t_f, loss_kwargs, epochs, lr, grad_clip=1.0,
          weight_decay=1.0e-6, resample_fn=None, lbfgs_epochs=None, lbfgs_lr=1.0,
          lbfgs_resample_fn=None, lbfgs_patience=50, lbfgs_tol=1.0e-9,
          target_loss=None, validation_points=None):
    """Minimizes the Burgers residual loss with AdamW + ReduceLROnPlateau,
    followed by an L-BFGS refinement phase -- the same two-phase schedule as
    `train_autograd` (see its docstring for the rationale). The only
    difference from `train_autograd` is how u_t, u_x, u_xx are obtained:
    here they come from finite differences on shifted collocation points
    instead of autograd.

    resample_fn: optional zero-argument callable returning a fresh (x_f, t_f)
    pair, called at the start of every epoch during the AdamW phase to
    replace the collocation points used that epoch. Pass a closure over
    `sample_collocation_points(..., distribution="random")` (with seed left
    as None) to train on different random collocation points each epoch;
    leave as None to keep training on the fixed x_f, t_f passed in (e.g. a
    uniform grid).

    lbfgs_epochs: number of L-BFGS iterations for the second phase (defaults
    to `epochs` when not given). Set lbfgs_epochs=0 to skip it and recover
    the old AdamW-only behavior.

    lbfgs_resample_fn: optional zero-argument callable, same shape as
    resample_fn, called once per L-BFGS *outer* iteration (before that
    iteration's `lbfgs_opt.step(closure)`) to replace the collocation
    points used from then on. A single step() call still evaluates
    `closure` several times internally (the strong_wolfe bracketing/zoom
    search) and every one of those internal calls sees the same, just
    -resampled point set -- only the set changes *between* outer steps, so
    the line search itself is never fed a moving target. What does shift
    across outer steps is the (s, y) curvature pairs L-BFGS accumulates in
    its history, since they're now measured on slightly different
    objectives; this is a real tradeoff against a fully fixed point set,
    so prefer a lbfgs_resample_fn that draws a reasonably large batch (a
    handful of points relative to the model's parameter count lets L-BFGS
    drive the residual to ~0 by pure interpolation in one step, which is
    what produces a long flat tail on the loss curve rather than genuine
    convergence). Leave as None (default) to keep the old behavior: a
    single fixed point set for the whole L-BFGS phase (whatever x_f, t_f
    ended up being after the last AdamW epoch).

    lbfgs_patience, lbfgs_tol: early-stopping controls for the L-BFGS loop.
    If `best_loss` fails to improve by more than `lbfgs_tol` for
    `lbfgs_patience` consecutive outer iterations, the loop stops instead
    of burning through the rest of `lbfgs_epochs` -- this is what a flat
    loss tail actually means (L-BFGS has nothing left to descend) and there
    is no point paying for more iterations once it happens. Set
    lbfgs_patience=None to disable early stopping and always run the full
    lbfgs_epochs.

    target_loss: optional stopping target for the monitored residual loss
    (e.g. from `reference_pde_residual`, the reference FDM solution's own
    discretization error). Training stops as soon as an accepted iterate's
    monitored loss reaches `<= target_loss`. Whenever the target is never
    reached, `epochs` plus `lbfgs_epochs` remains the fallback limit.

    validation_points: optional fixed tuple `(x_validation, t_validation)`.
    When supplied, every stopping decision, scheduler update, best-state
    comparison, and returned `hist["loss"]` value is evaluated on this same
    fixed set. The optimizer may still receive newly sampled training points
    through `resample_fn`. Monitoring is performed after each accepted
    optimizer step, so the saved snapshot and its reported loss correspond to
    the same parameter state.
    """
    L, nu, hx, ht = loss_kwargs["L"], loss_kwargs["nu"], loss_kwargs["hx"], loss_kwargs["ht"]

    def stencil_points(x_f, t_f):
        x_p = tc.remainder(x_f + hx, L)
        x_m = tc.remainder(x_f - hx, L)
        return x_p, x_m, t_f + ht, t_f - ht

    x_p, x_m, t_p, t_m = stencil_points(x_f, t_f)

    validation_stencil = None
    if validation_points is not None:
        if not isinstance(validation_points, (tuple, list)) or len(validation_points) != 2:
            raise ValueError(
                "validation_points must be a tuple (x_validation, t_validation)."
            )
        x_validation, t_validation = validation_points
        if x_validation.shape != t_validation.shape:
            raise ValueError(
                "x_validation and t_validation must have the same shape."
            )
        if x_validation.ndim != 2 or x_validation.shape[1] != 1:
            raise ValueError(
                "Fixed validation coordinates must have shape (N, 1)."
            )
        x_validation_p, x_validation_m, t_validation_p, t_validation_m = (
            stencil_points(x_validation, t_validation)
        )
        validation_stencil = (
            x_validation,
            x_validation_p,
            x_validation_m,
            t_validation,
            t_validation_p,
            t_validation_m,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = tc.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Decays lr as the loss plateaus: a larger lr early on drives the broad
    # global search, but without decay Adam is still bouncing around at that
    # same high lr on its last iteration, so L-BFGS would build its local
    # quadratic model from a noisy, badly-conditioned point instead of a
    # settled basin.
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(10, epochs // 10), min_lr=1.0e-4
    )

    hist = {
        "loss": [],
        "residual_rms": [],
        "optimizer_loss": [],
        "monitor_source": (
            "fixed_validation" if validation_stencil is not None else "training_batch"
        ),
    }

    def compute_residual(x_f, x_p, x_m, t_f, t_p, t_m):
        u_0  = model(tc.cat([x_f, t_f], dim=1))
        u_xp = model(tc.cat([x_p, t_f], dim=1))
        u_xm = model(tc.cat([x_m, t_f], dim=1))
        u_tp = model(tc.cat([x_f, t_p], dim=1))
        u_tm = model(tc.cat([x_f, t_m], dim=1))

        u_t = (u_tp - u_tm) / (2.0 * ht)
        u_x = (u_xp - u_xm) / (2.0 * hx)
        u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)

        return u_t + u_0 * u_x - nu * u_xx

    def evaluate_accepted_state(current_training_stencil):
        """Evaluate the state currently stored in the model without gradients."""
        monitor_stencil = (
            validation_stencil
            if validation_stencil is not None
            else current_training_stencil
        )
        was_training = model.training
        model.eval()
        with tc.no_grad():
            monitor_residual = compute_residual(*monitor_stencil)
            monitor_loss = float(tc.mean(monitor_residual**2))
            monitor_rms = float(tc.sqrt(tc.mean(monitor_residual**2)))
        if was_training:
            model.train()
        return monitor_loss, monitor_rms

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    best_state = snapshot()
    best_loss = float("inf")
    target_reached = False

    for _ in range(epochs):#tqdm(range(epochs), desc="AdamW"):
        model.train()

        if resample_fn is not None:
            x_f, t_f = resample_fn()
            x_p, x_m, t_p, t_m = stencil_points(x_f, t_f)

        residual = compute_residual(x_f, x_p, x_m, t_f, t_p, t_m)
        loss = tc.mean(residual**2)

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            tc.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        optimizer_loss_value = float(loss.detach())
        current_training_stencil = (x_f, x_p, x_m, t_f, t_p, t_m)
        loss_value, residual_rms = evaluate_accepted_state(current_training_stencil)
        scheduler.step(loss_value)

        if math.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_state = snapshot()

        hist["loss"].append(loss_value)
        hist["residual_rms"].append(residual_rms)
        hist["optimizer_loss"].append(optimizer_loss_value)

        # First stopping condition: the residual loss has already reached the
        # reference's own precision at these points -- no need for L-BFGS to
        # refine further, or to burn through the rest of `epochs`.
        if target_loss is not None and loss_value <= target_loss:
            target_reached = True
            break

    if lbfgs_epochs is None:
        lbfgs_epochs = epochs

    if lbfgs_epochs > 0 and not target_reached:
        # Hand L-BFGS the best basin Adam actually found, not whatever point
        # its last (possibly still-noisy) iterate landed on.
        model.load_state_dict(best_state)

        lbfgs_opt = tc.optim.LBFGS(
            params, lr=lbfgs_lr, max_iter=10, history_size=50,
            line_search_fn="strong_wolfe",
        )

        # Squared (not abs) residual: L-BFGS builds its Hessian approximation
        # from consecutive gradients, and the kink of |residual| at 0 makes
        # that curvature estimate unreliable, which is what drives loss
        # instability. Gradient clipping is skipped here for the same reason
        # -- it would distort the (s, y) pairs L-BFGS relies on. `strong_wolfe`
        # rejects any candidate step that doesn't give sufficient decrease,
        # but its bracketing/zoom search can still fall back to a non-improving
        # step if it can't find one inside its own iteration budget -- the
        # best-snapshot tracking below is what actually guarantees the final
        # model is never worse than the best point found.
        last = {}

        def closure():
            lbfgs_opt.zero_grad()
            residual = compute_residual(x_f, x_p, x_m, t_f, t_p, t_m)
            loss = tc.mean(residual**2)
            loss.backward()

            last["loss"] = float(loss.detach())
            last["residual_rms"] = float(tc.sqrt(tc.mean(residual.detach() ** 2)))
            return loss

        stall_count = 0
        for _ in range(lbfgs_epochs):#tqdm(range(lbfgs_epochs), desc="L-BFGS"):
            model.train()

            if lbfgs_resample_fn is not None:
                x_f, t_f = lbfgs_resample_fn()
                x_p, x_m, t_p, t_m = stencil_points(x_f, t_f)

            lbfgs_opt.step(closure)

            # The final closure evaluation is not guaranteed to be the state
            # accepted by the line search. Re-evaluate the retained parameters.
            current_training_stencil = (x_f, x_p, x_m, t_f, t_p, t_m)
            loss_value, residual_rms = evaluate_accepted_state(current_training_stencil)
            optimizer_loss_value = last.get("loss", float("nan"))

            previous_best_loss = best_loss
            if math.isfinite(loss_value) and loss_value < best_loss:
                best_loss = loss_value
                best_state = snapshot()

            if (
                math.isfinite(loss_value)
                and loss_value < previous_best_loss - lbfgs_tol
            ):
                stall_count = 0
            else:
                stall_count += 1

            hist["loss"].append(loss_value)
            hist["residual_rms"].append(residual_rms)
            hist["optimizer_loss"].append(optimizer_loss_value)

            if target_loss is not None and loss_value <= target_loss:
                target_reached = True
                break

            if lbfgs_patience is not None and stall_count >= lbfgs_patience:
                break

    hist["target_loss"] = target_loss
    hist["target_reached"] = target_reached
    hist["best_loss"] = best_loss

    # Always return the lowest-loss point seen across both phases, never
    # just whichever iterate the last epoch happened to land on.
    model.load_state_dict(best_state)
    return model, hist

# UNUSED (only referenced by main/old/run_quantum.py and main/old/run_quantum_autograd_pennylane.py,
# not by any other file in the repo) -- commented out 2026-09-03.
# def train_autograd(model, x_f, t_f, loss_kwargs, epochs, lr, grad_clip=1.0,
#                     weight_decay=1.0e-6, resample_fn=None, lbfgs_epochs=None, lbfgs_lr=1.0,
#                     lbfgs_resample_fn=None, lbfgs_patience=50, lbfgs_tol=1.0e-9,
#                     target_loss=None):
#     """Same Burgers-residual training as `train`, but u_t, u_x, u_xx come from
#     a single forward pass differentiated with autograd (torch.autograd.grad,
#     create_graph=True) instead of finite differences on shifted collocation
#     points. Only `nu` from `loss_kwargs` is used; hx/ht are not needed since
#     no stencil is built.
# 
#     Training runs in two phases: `epochs` iterations of AdamW doing a broad,
#     global search of parameter space (its lr decays via ReduceLROnPlateau as
#     the loss stalls, so it actually settles instead of still bouncing around
#     at a high lr when L-BFGS takes over), followed by `lbfgs_epochs`
#     iterations of L-BFGS refining that region into a low-loss local minimum
#     (defaults to `epochs` when not given). Set `lbfgs_epochs=0` to skip the
#     second phase.
# 
#     resample_fn: optional zero-argument callable returning a fresh (x_f, t_f)
#     pair, called at the start of every epoch during the AdamW phase to
#     replace the collocation points used that epoch. Pass a closure over
#     `sample_collocation_points(..., distribution="random")` (with seed left
#     as None) to train on different random collocation points each epoch;
#     leave as None to keep training on the fixed x_f, t_f passed in (e.g. a
#     uniform grid).
# 
#     lbfgs_resample_fn: optional zero-argument callable, same shape as
#     resample_fn, called once per L-BFGS *outer* iteration (before that
#     iteration's `lbfgs_opt.step(closure)`) to replace the collocation
#     points used from then on. A single step() call still evaluates
#     `closure` several times internally (the strong_wolfe bracketing/zoom
#     search) and every one of those internal calls sees the same, just
#     -resampled point set -- only the set changes *between* outer steps, so
#     the line search itself is never fed a moving target. What does shift
#     across outer steps is the (s, y) curvature pairs L-BFGS accumulates in
#     its history, since they're now measured on slightly different
#     objectives; this is a real tradeoff against a fully fixed point set,
#     so prefer a lbfgs_resample_fn that draws a reasonably large batch (a
#     handful of points relative to the model's parameter count lets L-BFGS
#     drive the residual to ~0 by pure interpolation in one step, which is
#     what produces a long flat tail on the loss curve rather than genuine
#     convergence). Leave as None (default) to keep the old behavior: a
#     single fixed point set for the whole L-BFGS phase (whatever x_f, t_f
#     ended up being after the last AdamW epoch).
# 
#     lbfgs_patience, lbfgs_tol: early-stopping controls for the L-BFGS loop.
#     If `best_loss` fails to improve by more than `lbfgs_tol` for
#     `lbfgs_patience` consecutive outer iterations, the loop stops instead
#     of burning through the rest of `lbfgs_epochs` -- this is what a flat
#     loss tail actually means (L-BFGS has nothing left to descend) and there
#     is no point paying for more iterations once it happens. Set
#     lbfgs_patience=None to disable early stopping and always run the full
#     lbfgs_epochs.
# 
#     target_loss: optional stopping target for the residual loss (e.g. from
#     `reference_pde_residual`, the reference FDM solution's own discretization
#     error at these collocation points). Training stops as soon as an
#     iterate's loss reaches `<= target_loss` -- in the AdamW phase this also
#     skips the L-BFGS phase entirely, since there is nothing left to refine
#     once the PINN already matches the reference's own precision. This is the
#     *first* stopping condition; whenever it is never reached, `epochs` (plus
#     `lbfgs_epochs`) is what stops training, exactly as when target_loss is
#     left at its default None.
# 
#     The lowest-loss parameter snapshot seen across *both* phases is what
#     gets loaded into `model` before returning -- not simply whichever
#     iterate training happened to end on. This is what makes L-BFGS strictly
#     help: even if a step lands somewhere much worse (a bad quasi-Newton
#     step, or a noisy hand-off point from Adam), the snapshot rollback means
#     the returned model can never be worse than the best point already
#     found."""
#     nu = loss_kwargs["nu"]
# 
#     x_f = x_f.clone().requires_grad_(True)
#     t_f = t_f.clone().requires_grad_(True)
# 
#     params = [p for p in model.parameters() if p.requires_grad]
#     opt = tc.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
#     # Decays lr as the loss plateaus: a larger lr early on drives the broad
#     # global search, but without decay Adam is still bouncing around at that
#     # same high lr on its last iteration, so L-BFGS would build its local
#     # quadratic model from a noisy, badly-conditioned point instead of a
#     # settled basin.
#     scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
#         opt, mode="min", factor=0.5, patience=max(10, epochs // 10), min_lr=1.0e-4
#     )
# 
#     hist = {"loss": [], "residual_rms": []}
# 
#     def compute_residual():
#         u_0 = model(tc.cat([x_f, t_f], dim=1))
# 
#         u_x = tc.autograd.grad(u_0, x_f, grad_outputs=tc.ones_like(u_0), create_graph=True)[0]
#         u_xx = tc.autograd.grad(u_x, x_f, grad_outputs=tc.ones_like(u_x), create_graph=True)[0]
#         u_t = tc.autograd.grad(u_0, t_f, grad_outputs=tc.ones_like(u_0), create_graph=True)[0]
# 
#         return u_t + u_0 * u_x - nu * u_xx
# 
#     def snapshot():
#         return {k: v.detach().clone() for k, v in model.state_dict().items()}
# 
#     best_state = snapshot()
#     best_loss = float("inf")
#     target_reached = False
# 
#     for _ in tqdm(range(epochs), desc="Adam"):
#         model.train()
# 
#         if resample_fn is not None and _ % 10 == 0:  # Resample every 10 iterations
#             x_f, t_f = resample_fn()
#             x_f = x_f.clone().requires_grad_(True)
#             t_f = t_f.clone().requires_grad_(True)
# 
#         residual = compute_residual()
#         loss = tc.mean(residual**2)
# 
#         opt.zero_grad()
#         loss.backward()
#         if grad_clip is not None:
#             tc.nn.utils.clip_grad_norm_(params, grad_clip)
#         opt.step()
# 
#         loss_value = float(loss.detach())
#         scheduler.step(loss_value)
# 
#         if loss_value < best_loss:
#             best_loss = loss_value
#             best_state = snapshot()
# 
#         hist["loss"].append(loss_value)
#         hist["residual_rms"].append(float(tc.sqrt(tc.mean(residual.detach() ** 2))))
# 
#         # First stopping condition: the residual loss has already reached the
#         # reference's own precision at these points -- no need for L-BFGS to
#         # refine further, or to burn through the rest of `epochs`.
#         if target_loss is not None and loss_value <= target_loss:
#             target_reached = True
#             break
# 
#     if lbfgs_epochs is None:
#         lbfgs_epochs = epochs
# 
#     if lbfgs_epochs > 0 and not target_reached:
#         # Hand L-BFGS the best basin Adam actually found, not whatever point
#         # its last (possibly still-noisy) iterate landed on.
#         model.load_state_dict(best_state)
# 
#         lbfgs_opt = tc.optim.LBFGS(
#             params, lr=lbfgs_lr, max_iter=10, history_size=50,
#             line_search_fn="strong_wolfe",
#         )
# 
#         # Squared (not abs) residual: L-BFGS builds its Hessian approximation
#         # from consecutive gradients, and the kink of |residual| at 0 makes
#         # that curvature estimate unreliable, which is what drives loss
#         # instability. Gradient clipping is skipped here for the same reason
#         # -- it would distort the (s, y) pairs L-BFGS relies on. `strong_wolfe`
#         # rejects any candidate step that doesn't give sufficient decrease,
#         # but its bracketing/zoom search can still fall back to a non-improving
#         # step if it can't find one inside its own iteration budget -- the
#         # best-snapshot tracking below is what actually guarantees the final
#         # model is never worse than the best point found.
#         last = {}
# 
#         def closure():
#             lbfgs_opt.zero_grad()
#             residual = compute_residual()
#             loss = tc.mean(residual**2)
#             loss.backward()
# 
#             last["loss"] = float(loss.detach())
#             last["residual_rms"] = float(tc.sqrt(tc.mean(residual.detach() ** 2)))
#             return loss
# 
#         stall_count = 0
#         for _ in tqdm(range(lbfgs_epochs), desc="L-BFGS"):
#             model.train()
# 
#             if lbfgs_resample_fn is not None:
#                 x_f, t_f = lbfgs_resample_fn()
#                 x_f = x_f.clone().requires_grad_(True)
#                 t_f = t_f.clone().requires_grad_(True)
# 
#             lbfgs_opt.step(closure)
#             # Only the point the line search actually accepted this step is
#             # logged -- `closure` is called several times per step to test
#             # candidate step sizes, and logging every trial would make
#             # rejected, never-applied candidates show up as fake spikes.
#             loss_value = last["loss"]
# 
#             if math.isfinite(loss_value) and loss_value < best_loss - lbfgs_tol:
#                 best_loss = loss_value
#                 best_state = snapshot()
#                 stall_count = 0
#             else:
#                 stall_count += 1
# 
#             hist["loss"].append(loss_value)
#             hist["residual_rms"].append(last["residual_rms"])
# 
#             if target_loss is not None and loss_value <= target_loss:
#                 target_reached = True
#                 break
# 
#             if lbfgs_patience is not None and stall_count >= lbfgs_patience:
#                 break
# 
#     hist["target_loss"] = target_loss
#     hist["target_reached"] = target_reached
# 
#     # Always return the lowest-loss point seen across both phases, never
#     # just whichever iterate the last epoch happened to land on.
#     model.load_state_dict(best_state)
#     return model, hist

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
def asymmetric_yerr(mean, vmin, vmax):
    """Builds the (2, N) `yerr` array matplotlib's `errorbar()` expects for
    asymmetric error bars that reach down to `vmin` and up to `vmax` at each
    point, instead of a symmetric +/-std whisker.

    Use this (with `vmin`/`vmax` taken from the actual per-seed samples)
    instead of `yerr=std` whenever the error bars are drawn on a log-scale
    axis: mean +/- std routinely goes negative when std > mean, which is
    common with only a handful of seeds and a skewed, positive-only metric
    like MSE/MAE -- and a log axis cannot render that, so matplotlib clips
    the lower whisker all the way to the bottom of the plot instead of
    stopping below the mean. Because `vmin`/`vmax` come from samples that
    were actually observed (always > 0 for MSE/MAE), the lower whisker can
    never go negative or below the smallest point actually measured.
    """
    mean = np.asarray(mean, dtype=float)
    vmin = np.asarray(vmin, dtype=float)
    vmax = np.asarray(vmax, dtype=float)
    return np.vstack([mean - vmin, vmax - mean])


def std_yerr(mean, std):
    """Builds the (2, N) `yerr` array matplotlib's `errorbar()` expects for
    symmetric +/-std whiskers, clipping the lower whisker so it never reaches
    down to or below zero. Plain mean-std can go <= 0 when std >= mean (a
    handful of seeds and a skewed, positive-only metric like MSE/MAE makes
    this common), which a log-scale y-axis can't render -- matplotlib would
    silently clip that whisker to the bottom of the plot instead of stopping
    below the mean. Clamping the lower half to just under `mean` keeps
    mean-lower a valid positive value while still showing the full std on
    the upper half."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    lower = np.minimum(std, mean * (1 - 1e-9))
    return np.vstack([lower, std])


ERROR_BAR_TYPES = ("minimo_maximo", "desvio_padrao", "variancia")


def error_bar_yerr(error_type, mean, std, var, vmin, vmax):
    """Builds the (2, N) `yerr` array for one of the three error-bar types
    the plote_*.py scripts let the user pick between:

    - "minimo_maximo": asymmetric whiskers reaching the observed min/max
      (via `asymmetric_yerr`).
    - "desvio_padrao": symmetric +/-std whiskers (via `std_yerr`, so the
      lower whisker is clipped instead of going negative on a log axis).
    - "variancia": symmetric +/-variance whiskers (also via `std_yerr`).

    Raises ValueError for any other `error_type`."""
    if error_type == "minimo_maximo":
        return asymmetric_yerr(mean, vmin, vmax)
    if error_type == "desvio_padrao":
        return std_yerr(mean, std)
    if error_type == "variancia":
        return std_yerr(mean, var)
    raise ValueError(
        f"error_type deve ser um de {ERROR_BAR_TYPES} (recebido {error_type!r})."
    )


def plot_loss(histories, title="Training loss", save_path=None, show_cond=False):
    """histories: dict of label -> list of loss values."""
    plt.figure(figsize=(9, 4.5))
    for label, values in histories.items():
        #plt.plot(np.log10(np.asarray(values) + 1.0e-16), linewidth=2.0, label=label)
        plt.plot(np.asarray(values), linewidth=2.0, label=label)
    plt.xlabel("Epochs", fontsize=14)
    #plt.ylabel(r"$\log_{10}(\mathrm{loss})$", fontsize=14)
    plt.ylabel(r"$\mathrm{Loss}$", fontsize=14)
    plt.yscale("log")
    
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()


def plot_solution(x, curves, ref=None, ref_label="FDM reference", title="", save_path=None,
                   show_cond=False, collocation_x=None, colors=None, ref_color=None):
    """
    x: spatial grid, shape (N,1).
    curves: dict of label -> predicted u array, shape (N,1). Used both for a
        single model's solution and for multi-model / multi-time comparisons.
    ref: optional reference array plotted as a thick solid line.
    collocation_x: optional array of the x-coordinates of the collocation
        points used for training; drawn as a rug of ticks along the x-axis
        (independent of the y-scale) so the point density is visible on top
        of the solution curve.
    colors: optional dict of label -> matplotlib color, for curves that need
        a fixed color (e.g. a classical baseline drawn in black) instead of
        the default color cycle.
    ref_color: optional matplotlib color for the reference curve; defaults to
        None (matplotlib's default color cycle) when not given.
    """
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.gca()
    if ref is not None:
        ax.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label, color=ref_color)
    for label, u in curves.items():
        color = None if colors is None else colors.get(label)
        ax.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label, color=color)
    if collocation_x is not None:
        collocation_x = np.asarray(collocation_x).ravel()
        rug_trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
        ax.scatter(
            collocation_x, np.full_like(collocation_x, 0.02), transform=rug_trans,
            marker="|", s=200, linewidth=1.2, color="black", clip_on=False,
            label=f"collocation points (N={collocation_x.size})",
        )
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$u(x,t)$", fontsize=14)
    #ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()

# ----------------------------------------------------------------------
# plote_*.py helpers -- checkpoint discovery, error-vs-width statistics,
# and their companion plots. Every plote_*.py script imports these and
# keeps only its own parameters (paths, labels, colors) plus the call.
# `model_cls` (ClassicalPINN2 / QuantumPINNPennyLane) is passed in by the
# caller rather than imported here, since models.py/models_pennylane.py
# both import from this module -- importing them back would be circular.
# ----------------------------------------------------------------------
def find_classical_configs(t_dir, collocation_distribution=None, n_collocation=None):
    """Returns [(intra_neuron, run_id, pt_path), ...] for every classical
    Cmodel_NEURO* checkpoint in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution` ("uniform" or "random"; leave as
    None to include every distribution found) and whose N_COLLOCATION
    matches `n_collocation` (leave as None to include every value found --
    but note that checkpoints from different N_COLLOCATION sweeps can share
    the same t_dir, so leaving this unfiltered silently averages runs
    trained with different N_COLLOCATION together), reading intra_neuron
    from its companion `_config.json` (the run_id/sample index comes from
    the trailing `_<run_id>` in the filename)."""
    entries = []
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_*_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg["training_params"]
        distribution = training_params.get("COLLOCATION_DISTRIBUTION")
        if collocation_distribution is not None and distribution != collocation_distribution:
            continue
        if n_collocation is not None and training_params.get("N_COLLOCATION") != n_collocation:
            continue
        intra_neuron = cfg["config"]["intra_neuron"]
        run_id = int(os.path.basename(pt_path)[: -len(".pt")].rsplit("_", 1)[1])
        entries.append((intra_neuron, run_id, pt_path))
    return sorted(entries)


# UNUSED (only reachable through main/old/plote_comparativo.py and the two dead
# helpers below it, compute_quantum_error_stats_std / compute_quantum_errors_comparativo)
# -- commented out 2026-09-03.
# def find_quantum_configs(t_dir, input_rotation_axis="ry", collocation_distribution="random"):
#     """Returns [(fixed_layer, n_qubits, seed, pt_path), ...] for every
#     Qmodel_<collocation_distribution>_NQUBITS* checkpoint in `t_dir` whose
#     input_rotation_axis matches, reading fixed_layer/n_qubits/seed from its
#     companion `_config.json`."""
#     entries = []
#     pattern = f"Qmodel_{collocation_distribution}_NQUBITS*_config.json"
#     for config_path in glob.glob(os.path.join(t_dir, pattern)):
#         pt_path = config_path[: -len("_config.json")] + ".pt"
#         if not os.path.exists(pt_path):
#             continue
#         with open(config_path) as f:
#             cfg = json.load(f)["config"]
#         if cfg.get("input_rotation_axis") != input_rotation_axis:
#             continue
#         entries.append((cfg["fixed_layer"], cfg["n_qubits"], cfg["seed"], pt_path))
#     return sorted(entries)


# UNUSED (only referenced by main/old/plote_classico.py, not by any other file) --
# commented out 2026-09-03.
# def compute_classical_mean_errors(t_dir, model_cls, nu):
#     """Loads every classical model in `t_dir`, evaluates each against the FDM
#     reference at t=Tfin, and averages the MSE/MAE across the run_id samples
#     (one per amostragem) for each INTRA_NEURO value. Returns (intra_neurons,
#     mean_mse, mean_mae, n_samples, Tfin), one point per INTRA_NEURO. Used by
#     the legacy `old/plote_classico.py`."""
#     entries = find_classical_configs(t_dir)
#     if not entries:
#         return [], [], [], [], None
# 
#     first_model = model_cls.load(entries[0][2])
#     L, Tfin = first_model.L, first_model.Tfin
# 
#     x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
#     x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
#     u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
# 
#     errors_by_neuron = {}
#     for intra_neuron, _run_id, pt_path in entries:
#         model = model_cls.load(pt_path)
#         u_pred = predict(model, x_plot, Tfin)
#         entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
#         entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
#         entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))
# 
#     intra_neurons = sorted(errors_by_neuron)
#     mse = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
#     mae = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
#     n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
#     return intra_neurons, mse, mae, n_samples, Tfin


# UNUSED (only referenced by main/old/plote_classico.py, not by any other file) --
# commented out 2026-09-03.
# def plot_classical_mean_errors(intra_neurons, mse, mae, Tfin, save_path,
#                                 color_mse="#2a78d6", color_mae="#1baf7a"):
#     """Companion plot for `compute_classical_mean_errors` (legacy, mean-only)."""
#     fig, ax = plt.subplots(figsize=(9, 4.5))
#     ax.plot(intra_neurons, mse, "o-", color=color_mse, linewidth=2.0, markersize=6, label="MSE")
#     ax.plot(intra_neurons, mae, "o-", color=color_mae, linewidth=2.0, markersize=6, label="MAE")
#     ax.set_xlabel("INTRA_NEURO", fontsize=14)
#     ax.set_ylabel("Error", fontsize=14)
#     ax.set_yscale("log")
#     ax.set_xticks(intra_neurons)
#     ax.set_title(f"Classical PINN mean error vs INTRA_NEURO (T = {Tfin})", fontsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)


# UNUSED (not called anywhere in the codebase -- its would-be caller,
# plote_classico_std copy.py, no longer exists in the repo) -- commented out 2026-09-03.
# def compute_classical_mae_stats(t_dir, model_cls, nu):
#     """Loads every classical model in `t_dir`, evaluates each against the FDM
#     reference at t=Tfin, and returns per-INTRA_NEURO mean/std/min/max MAE
#     across the run_id samples. Returns (intra_neurons, mae_mean, mae_std,
#     mae_min, mae_max, n_samples, Tfin), one point per INTRA_NEURO.
#     mae_min/mae_max (the extremes actually observed across run_ids) are what
#     the plot uses for its error bars -- unlike mean +/- std, they can never
#     go negative on the plot's log-scale y-axis. Used by the legacy
#     `plote_classico_std copy.py` (MAE-only, ignores COLLOCATION_DISTRIBUTION)."""
#     entries = find_classical_configs(t_dir)
#     if not entries:
#         return [], [], [], [], [], [], None
# 
#     first_model = model_cls.load(entries[0][2])
#     L, Tfin = first_model.L, first_model.Tfin
# 
#     x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
#     x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
#     u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
# 
#     errors_by_neuron = {}
#     for intra_neuron, _run_id, pt_path in entries:
#         model = model_cls.load(pt_path)
#         u_pred = predict(model, x_plot, Tfin)
#         entry = errors_by_neuron.setdefault(intra_neuron, [])
#         entry.append(float(np.mean(np.abs(u_pred - u_ref))))
# 
#     intra_neurons = sorted(errors_by_neuron)
#     mae_mean = [float(np.mean(errors_by_neuron[n])) for n in intra_neurons]
#     mae_std = [float(np.std(errors_by_neuron[n])) for n in intra_neurons]
#     mae_min = [float(np.min(errors_by_neuron[n])) for n in intra_neurons]
#     mae_max = [float(np.max(errors_by_neuron[n])) for n in intra_neurons]
#     n_samples = [len(errors_by_neuron[n]) for n in intra_neurons]
#     return intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin


# UNUSED (not called anywhere in the codebase -- its would-be caller,
# plote_classico_std copy.py, no longer exists in the repo) -- commented out 2026-09-03.
# def plot_classical_mae_std(intra_neurons, mae_mean, mae_min, mae_max, Tfin, save_path,
#                             color_mae="#1baf7a"):
#     fig, ax = plt.subplots(figsize=(9, 4.5))
#     yerr = asymmetric_yerr(mae_mean, mae_min, mae_max)
#     ax.errorbar(intra_neurons, mae_mean, yerr=yerr, fmt="o-", color=color_mae,
#                 linewidth=2.0, markersize=6, capsize=4, label="MAE (mean, [min,max])")
#     ax.set_xlabel("INTRA_NEURO", fontsize=14)
#     ax.set_ylabel("Error", fontsize=14)
#     ax.set_yscale("log")
#     ax.set_xticks(intra_neurons)
#     ax.set_title(f"Classical PINN MAE vs INTRA_NEURO (T = {Tfin}, legacy data)", fontsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)


def compute_classical_error_stats_std(t_dir, model_cls, nu, collocation_distribution,
                                       n_collocation=None):
    """Loads every classical model in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution` and, when given, whose N_COLLOCATION
    matches `n_collocation`, evaluates each against the FDM reference at
    t=Tfin, and returns per-INTRA_NEURO mean/std/variance/min/max MSE and MAE
    across the run_id samples. Returns (intra_neurons, mse_mean, mse_std,
    mse_var, mse_min, mse_max, mae_mean, mae_std, mae_var, mae_min, mae_max,
    n_samples, Tfin), one point per INTRA_NEURO. The _min/_max values (the
    extremes actually observed across run_ids) are what the plot uses for
    its error bars -- unlike mean +/- std, they can never go negative on the
    plot's log-scale y-axis. Leaving n_collocation as None mixes checkpoints
    from every N_COLLOCATION sweep found in `t_dir` into the same
    INTRA_NEURO point -- pass it explicitly to compare against a quantum
    curve pinned to one N_COLLOCATION value. Used by
    `plote_classico_vs_Intra.py`."""
    entries = find_classical_configs(
        t_dir, collocation_distribution=collocation_distribution, n_collocation=n_collocation)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], [], [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mse_mean = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_std = [float(np.std(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_var = [float(np.var(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_min = [float(np.min(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_max = [float(np.max(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_var = [float(np.var(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_min = [float(np.min(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_max = [float(np.max(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return (intra_neurons, mse_mean, mse_std, mse_var, mse_min, mse_max, mae_mean, mae_std,
            mae_var, mae_min, mae_max, n_samples, Tfin)


def compute_classical_error_and_stopping_per_run(t_dir, model_cls, nu, collocation_distribution,
                                                   n_collocation=None):
    """Like `compute_classical_error_stats_std`, but returns one row per
    individual run_id instead of aggregating across seeds -- every sampled
    checkpoint found in `t_dir`, its MSE/MAE against the FDM reference at
    t=Tfin, and whether its training used its full EPOCHS+LBFGS_EPOCHS budget
    (TARGET_REACHED False, STOPPED_AT_STEP close to the budget) or converged
    early by reaching TARGET_LOSS (TARGET_REACHED True). Returns (rows,
    Tfin), where each row is a dict with keys "intra_neuron", "run_id",
    "mse", "mae", "stopped_at_step", "epoch_budget", "target_reached".
    Companion to `compute_classical_error_stats_std`, read from the same
    checkpoints but without collapsing the run_id samples into a mean/std --
    used by `plote_erro_epocas_classico.py` to plot every sample point
    against its training cost."""
    entries = find_classical_configs(
        t_dir, collocation_distribution=collocation_distribution, n_collocation=n_collocation)
    if not entries:
        return [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    rows = []
    for intra_neuron, run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        mse = float(np.mean((u_pred - u_ref) ** 2))
        mae = float(np.mean(np.abs(u_pred - u_ref)))

        config_path = pt_path[: -len(".pt")] + "_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg.get("training_params", {})
        epochs = training_params.get("EPOCHS")
        lbfgs_epochs = training_params.get("LBFGS_EPOCHS", 0)
        epoch_budget = (epochs + lbfgs_epochs) if epochs is not None else None

        rows.append({
            "intra_neuron": intra_neuron,
            "run_id": run_id,
            "mse": mse,
            "mae": mae,
            "stopped_at_step": training_params.get("STOPPED_AT_STEP"),
            "epoch_budget": epoch_budget,
            "target_reached": bool(training_params.get("TARGET_REACHED")),
        })
    return rows, Tfin


# UNUSED (not called anywhere in the codebase -- plote_classico_vs_Intra.py uses
# plot_classical_vs_quantum_std instead) -- commented out 2026-09-03.
# def plot_classical_errors_std(stats_by_distribution, Tfin, save_path,
#                                color_mse="#2a78d6", dist_styles=None):
#     """`stats_by_distribution` maps COLLOCATION_DISTRIBUTION -> (intra_neurons,
#     mse_mean, mse_min, mse_max) for that distribution; distributions with no
#     models are omitted by the caller. `dist_styles` maps distribution name ->
#     dict(fmt=..., markerfacecolor=...); defaults to solid filled markers for
#     "uniform" and dashed hollow markers for "random", so both variants can
#     share one set of axes."""
#     if dist_styles is None:
#         dist_styles = {
#             "uniform": dict(fmt="o-", markerfacecolor=None),
#             "random": dict(fmt="s--", markerfacecolor="none"),
#         }
#     fig, ax = plt.subplots(figsize=(9, 4.5))
#     all_neurons = set()
#     for distribution, (intra_neurons, mse_mean, mse_min,
#                         mse_max) in stats_by_distribution.items():
#         all_neurons.update(intra_neurons)
#         style = dist_styles.get(distribution, dict(fmt="o-", markerfacecolor=None))
#         mse_yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
#         ax.errorbar(intra_neurons, mse_mean, yerr=mse_yerr, fmt=style["fmt"], color=color_mse,
#                     markerfacecolor=style["markerfacecolor"], linewidth=2.0, markersize=6,
#                     capsize=4, label=f"with {distribution}")
#     ax.set_xlabel("INTRA_NEURO", fontsize=14)
#     ax.set_ylabel("MSE", fontsize=14)
#     ax.set_yscale("log")
#     ax.set_xticks(sorted(all_neurons))
#     ax.set_title(f"T = {Tfin}", fontsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)


def load_quantum_sweep_summary(csv_path):
    """Loads a qubit_sweep_summary.csv (produced by the N_QUBITS sweep script,
    one aggregated row per network/n_cnot_layers/n_qubits/Tfin/n_collocation
    combo, with mse_mean/mse_min/mse_max/mse_std already computed across
    seeds) into a list of dict rows with the numeric columns converted to
    float. Returns [] if `csv_path` doesn't exist. Use the raw
    qubit_sweep_runs.csv (one row per seed) instead only if you need to
    recompute the aggregation yourself."""
    if not os.path.exists(csv_path):
        return []
    numeric_fields = {
        "n_cnot_layers", "n_qubits", "Tfin", "n_collocation", "mse_mean", "mse_min",
        "mse_max", "mse_std", "mse_variance", "steps_mean", "steps_std", "steps_variance",
        "steps_min", "steps_max", "target_reached_fraction", "trainable_parameters", "n_samples",
    }
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in numeric_fields:
            if row.get(field, "") != "":
                row[field] = float(row[field])
    return rows


SWEEP_SUMMARY_FIELDS = (
    "protocol_version", "model_family", "network", "n_cnot_layers", "n_qubits",
    "internal_width", "Tfin", "n_collocation", "mse_mean", "mse_min", "mse_max",
    "mse_std", "mse_variance", "steps_mean", "steps_std", "steps_variance",
    "steps_min", "steps_max", "target_reached_fraction", "early_target_fraction",
    "adam_target_missed_fraction", "adam_epoch_limit", "trainable_parameters", "n_samples",
)


def summarize_raw_runs(raw_rows):
    """Aggregates raw per-seed run rows -- the schema of qubit_sweep_runs.csv,
    or a single-N_QUBITS export like Dados1Qubit.csv: one row per
    model_family/network/n_cnot_layers/n_qubits/Tfin/n_collocation/seed, with
    mse/target_reached/stopped_at_step/adam_steps/trainable_parameters
    columns -- into one summary row per model_family/network/n_cnot_layers/
    n_qubits/Tfin/n_collocation combo, shaped like a qubit_sweep_summary.csv
    row (`SWEEP_SUMMARY_FIELDS`, see `load_quantum_sweep_summary`). A run
    counts toward `early_target_fraction` if it reached target_loss without
    needing the L-BFGS refinement stage (stopped_at_step <= adam_steps);
    `adam_target_missed_fraction` is the complement of that fraction (runs
    that needed L-BFGS, or never reached target, to stop). Returns a list of
    summary dicts sorted by the grouping key."""
    grouped = {}
    for row in raw_rows:
        key = (
            str(row.get("model_family", "quantum_exact")), str(row["network"]),
            int(float(row["n_cnot_layers"])), int(float(row.get("internal_width", row["n_qubits"]))),
            float(row["Tfin"]), int(float(row["n_collocation"])),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (model_family, network, layers, n_qubits, tfin, n_collocation), group in sorted(grouped.items()):
        mse = np.asarray([float(r["mse"]) for r in group], dtype=np.float64)
        steps = np.asarray([float(r["stopped_at_step"]) for r in group], dtype=np.float64)
        adam_steps = np.asarray([float(r["adam_steps"]) for r in group], dtype=np.float64)
        reached = np.asarray([str(r["target_reached"]).strip().lower() == "true" for r in group])
        early = reached & (steps <= adam_steps)
        summary.append({
            "protocol_version": group[0].get("protocol_version", ""),
            "model_family": model_family, "network": network, "n_cnot_layers": layers,
            "n_qubits": n_qubits, "internal_width": n_qubits, "Tfin": tfin,
            "n_collocation": n_collocation,
            "mse_mean": float(np.mean(mse)), "mse_min": float(np.min(mse)),
            "mse_max": float(np.max(mse)), "mse_std": float(np.std(mse)),
            "mse_variance": float(np.var(mse)),
            "steps_mean": float(np.mean(steps)), "steps_std": float(np.std(steps)),
            "steps_variance": float(np.var(steps)), "steps_min": int(np.min(steps)),
            "steps_max": int(np.max(steps)),
            "target_reached_fraction": float(np.mean(reached)),
            "early_target_fraction": float(np.mean(early)),
            "adam_target_missed_fraction": float(1.0 - np.mean(early)),
            "adam_epoch_limit": int(np.max(adam_steps)),
            "trainable_parameters": int(float(group[0]["trainable_parameters"])),
            "n_samples": len(group),
        })
    return summary


def merge_raw_runs_into_sweep_summary(raw_csv_path, summary_csv_path):
    """Aggregates the raw per-seed rows in `raw_csv_path` (see
    `summarize_raw_runs`) and merges them into `summary_csv_path` (a
    qubit_sweep_summary.csv, created if it doesn't exist yet): new
    model_family/network/n_cnot_layers/n_qubits/Tfin/n_collocation combos are
    appended, existing ones are overwritten with the freshly computed
    aggregate. Rows untouched by the merge are written back exactly as read,
    so this doesn't reformat the rest of the file. Used to fold extra raw
    exports (e.g. a single-N_QUBITS Dados1Qubit.csv) into the summary the
    plotting scripts (`plote_classico_vs_Intra.py`) read. Returns the merged
    list of summary rows, sorted by the grouping key."""
    with open(raw_csv_path, newline="") as f:
        new_rows = summarize_raw_runs(list(csv.DictReader(f)))

    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
    else:
        existing_rows = []

    def group_key(row):
        return (
            str(row["model_family"]), str(row["network"]), int(float(row["n_cnot_layers"])),
            int(float(row["n_qubits"])), float(row["Tfin"]), int(float(row["n_collocation"])),
        )

    merged_by_key = {group_key(row): row for row in existing_rows}
    for row in new_rows:
        merged_by_key[group_key(row)] = row
    merged = [merged_by_key[key] for key in sorted(merged_by_key)]

    os.makedirs(os.path.dirname(summary_csv_path) or ".", exist_ok=True)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in merged:
            writer.writerow(row)
    return merged


def quantum_stats_from_sweep_summary(csv_path, network, n_cnot_layers, n_collocation, Tfin,
                                      model_family=None):
    """Filters a qubit_sweep_summary.csv (see `load_quantum_sweep_summary`)
    down to the rows matching `network`, `n_cnot_layers`, `n_collocation`,
    and `Tfin`, and returns (n_qubits, mse_mean, mse_std, mse_var, mse_min,
    mse_max) sorted by n_qubits -- one point per N_QUBITS, in the same shape
    `plot_classical_vs_quantum_std` expects (it picks whichever of
    std/var/min-max the caller asked for via `error_type`). Some sweep
    summaries (e.g. the "fixed_validation" ones) carry both a quantum_exact
    and a classical_mlp row per N_QUBITS in a `model_family` column -- pass
    `model_family` (e.g. "quantum_exact") to keep only one of them; leave as
    None for files with a single row per N_QUBITS (no model_family column,
    or only one family present). Returns ([], [], [], [], [], []) if no rows
    match."""
    rows = load_quantum_sweep_summary(csv_path)
    matches = [
        row for row in rows
        if row.get("network") == network
        and int(row["n_cnot_layers"]) == n_cnot_layers
        and int(row["n_collocation"]) == n_collocation
        and math.isclose(row["Tfin"], Tfin, rel_tol=1e-9, abs_tol=1e-9)
        and (model_family is None or row.get("model_family") == model_family)
    ]
    matches.sort(key=lambda row: row["n_qubits"])
    n_qubits = [int(row["n_qubits"]) for row in matches]
    mse_mean = [row["mse_mean"] for row in matches]
    mse_std = [row["mse_std"] for row in matches]
    mse_var = [row["mse_variance"] for row in matches]
    mse_min = [row["mse_min"] for row in matches]
    mse_max = [row["mse_max"] for row in matches]
    return n_qubits, mse_mean, mse_std, mse_var, mse_min, mse_max


def load_quantum_exact_summary(csv_path):
    """Loads a quantum_exact_summary.csv (produced by run_quantumExact.ipynb,
    one aggregated row per network/n_cnot_layers/Tfin/n_collocation combo at
    a fixed N_QUBITS, with mse_mean/mse_std/mse_variance and
    steps_mean/steps_std/steps_variance already computed across seeds) into a
    list of dict rows with the numeric columns converted to float. Returns
    [] if `csv_path` doesn't exist. Sibling of `load_quantum_sweep_summary`,
    for the fixed-N_QUBITS exact-sweep results instead of the N_QUBITS sweep."""
    if not os.path.exists(csv_path):
        return []
    numeric_fields = {
        "n_cnot_layers", "Tfin", "n_collocation", "mse_mean", "mse_min", "mse_max",
        "mse_std", "mse_variance", "steps_mean", "steps_std", "steps_variance",
        "steps_min", "steps_max", "target_reached_fraction", "early_target_fraction",
        "adam_target_missed_fraction", "adam_epoch_limit", "n_samples",
    }
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in numeric_fields:
            if row.get(field, "") != "":
                row[field] = float(row[field])
    return rows


def quantum_exact_stats_vs_collocation(csv_path, Tfin, network=None, n_cnot_layers=None):
    """Filters a quantum_exact_summary.csv (see `load_quantum_exact_summary`)
    down to the rows matching `Tfin` (and, when given, `network` and
    `n_cnot_layers`), and returns (n_collocation, mse_mean, mse_std, mse_var,
    steps_mean, steps_std, steps_var, target_reached_frac, n_samples) sorted
    by n_collocation -- one point per N_COLLOCATION value, the same shape the
    classical `compute_classical_error_stats_vs_collocation` /
    `compute_classical_stopping_stats_vs_collocation` pair returns, so
    quantum and classical curves can share one plot. Returns an
    all-empty-lists tuple if no rows match."""
    rows = load_quantum_exact_summary(csv_path)
    matches = [
        row for row in rows
        if math.isclose(row["Tfin"], Tfin, rel_tol=1e-9, abs_tol=1e-9)
        and (network is None or row.get("network") == network)
        and (n_cnot_layers is None or int(row["n_cnot_layers"]) == n_cnot_layers)
    ]
    matches.sort(key=lambda row: row["n_collocation"])
    n_collocation = [int(row["n_collocation"]) for row in matches]
    mse_mean = [row["mse_mean"] for row in matches]
    mse_std = [row["mse_std"] for row in matches]
    mse_var = [row["mse_variance"] for row in matches]
    steps_mean = [row["steps_mean"] for row in matches]
    steps_std = [row["steps_std"] for row in matches]
    steps_var = [row["steps_variance"] for row in matches]
    target_reached_frac = [row["target_reached_fraction"] for row in matches]
    n_samples = [int(row["n_samples"]) for row in matches]
    return (n_collocation, mse_mean, mse_std, mse_var, steps_mean, steps_std, steps_var,
            target_reached_frac, n_samples)


def _quantum_exact_rows_by_n_qubits(csv_path_by_n_qubits, Tfin, n_collocation, network=None,
                                     n_cnot_layers=None):
    """Shared lookup for `quantum_exact_stats_vs_n_qubits` and
    `quantum_steps_stats_vs_n_qubits`: for each N_QUBITS in
    `csv_path_by_n_qubits` ({n_qubits: csv_path}, one quantum_exact_summary.csv
    per N_QUBITS -- see `load_quantum_exact_summary`), keeps the single row
    matching `Tfin` and `n_collocation` (and, when given, `network` and
    `n_cnot_layers`). N_QUBITS whose file is missing or has no matching row
    are skipped. Returns [(n_qubits, row), ...] sorted by n_qubits."""
    matches = []
    for n_qubits, csv_path in csv_path_by_n_qubits.items():
        for row in load_quantum_exact_summary(csv_path):
            if (math.isclose(row["Tfin"], Tfin, rel_tol=1e-9, abs_tol=1e-9)
                    and int(row["n_collocation"]) == n_collocation
                    and (network is None or row.get("network") == network)
                    and (n_cnot_layers is None or int(row["n_cnot_layers"]) == n_cnot_layers)):
                matches.append((n_qubits, row))
                break
    matches.sort(key=lambda item: item[0])
    return matches


def quantum_exact_stats_vs_n_qubits(csv_path_by_n_qubits, Tfin, n_collocation, network=None,
                                     n_cnot_layers=None):
    """Companion to `quantum_exact_stats_vs_collocation`, but assembles an
    N_QUBITS sweep at a *fixed* `n_collocation` out of several
    quantum_exact_summary.csv files -- one per N_QUBITS (see
    `csv_path_by_n_qubits`, e.g. QUANTUM_EXACT_CSV_BY_N_QUBITS in
    plote_classico_vs_Intra.py). Returns (n_qubits, mse_mean, mse_std,
    mse_var, mse_min, mse_max) sorted by n_qubits -- the shape
    `plot_classical_vs_quantum_std` expects. Sibling of
    `quantum_steps_stats_vs_n_qubits`; used by `plote_classico_vs_Intra.py`."""
    matches = _quantum_exact_rows_by_n_qubits(csv_path_by_n_qubits, Tfin, n_collocation,
                                               network=network, n_cnot_layers=n_cnot_layers)
    n_qubits = [n for n, _ in matches]
    mse_mean = [row["mse_mean"] for _, row in matches]
    mse_std = [row["mse_std"] for _, row in matches]
    mse_var = [row["mse_variance"] for _, row in matches]
    mse_min = [row["mse_min"] for _, row in matches]
    mse_max = [row["mse_max"] for _, row in matches]
    return n_qubits, mse_mean, mse_std, mse_var, mse_min, mse_max


def quantum_steps_stats_vs_n_qubits(csv_path_by_n_qubits, Tfin, n_collocation, network=None,
                                     n_cnot_layers=None):
    """Steps counterpart of `quantum_exact_stats_vs_n_qubits`: same
    per-N_QUBITS lookup, but returns the training-cost columns instead of
    MSE: (n_qubits, steps_mean, steps_std, steps_var, steps_min, steps_max)
    sorted by n_qubits -- the shape `plot_classical_vs_quantum_steps_std`
    expects."""
    matches = _quantum_exact_rows_by_n_qubits(csv_path_by_n_qubits, Tfin, n_collocation,
                                               network=network, n_cnot_layers=n_cnot_layers)
    n_qubits = [n for n, _ in matches]
    steps_mean = [row["steps_mean"] for _, row in matches]
    steps_std = [row["steps_std"] for _, row in matches]
    steps_var = [row["steps_variance"] for _, row in matches]
    steps_min = [row["steps_min"] for _, row in matches]
    steps_max = [row["steps_max"] for _, row in matches]
    return n_qubits, steps_mean, steps_std, steps_var, steps_min, steps_max


def plot_classical_vs_quantum_std(stats_by_distribution, quantum_stats, Tfin, save_path,
                                   color_mse="b", color_quantum="#000000", dist_styles=None,
                                   error_type="desvio_padrao"):
    """Plots classical MSE vs INTRA_NEURO for each COLLOCATION_DISTRIBUTION in"""
    if dist_styles is None:
        dist_styles = {
            "uniform": dict(fmt="o-", markerfacecolor=None),
            "random": dict(fmt="s--", markerfacecolor="none"),
        }
    error_labels = {
        "minimo_maximo": "min/max",
        "desvio_padrao": "±std",
        "variancia": "±var",
    }
    error_label = error_labels.get(error_type, error_type)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_widths = set()
    for distribution, (intra_neurons, mse_mean, mse_std, mse_var, mse_min,
                        mse_max) in stats_by_distribution.items():
        all_widths.update(intra_neurons)
        style = dist_styles.get(distribution, dict(fmt="o-", markerfacecolor=None))
        mse_yerr = error_bar_yerr(error_type, mse_mean, mse_std, mse_var, mse_min, mse_max)
        ax.errorbar(intra_neurons, mse_mean, yerr=mse_yerr, fmt=style["fmt"],
                    color=style.get("color", color_mse),
                    markerfacecolor=style["markerfacecolor"], linewidth=2.0, markersize=6,
                    capsize=4, label=f"Classical {error_label})")

    if quantum_stats is not None:
        n_qubits, q_mse_mean, q_mse_std, q_mse_var, q_mse_min, q_mse_max = quantum_stats
        if n_qubits:
            all_widths.update(n_qubits)
            q_yerr = error_bar_yerr(error_type, q_mse_mean, q_mse_std, q_mse_var, q_mse_min, q_mse_max)
            ax.errorbar(n_qubits, q_mse_mean, yerr=q_yerr, fmt="^-", color=color_quantum,
                        linewidth=2.0, markersize=6, capsize=4,
                        label=f"Quantum {error_label})")

    ax.set_xlabel("N° of Neurons / N° of Qubits", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_widths))
    ax.set_title(f"T = {Tfin}", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_classical_stopping_stats_std(t_dir, collocation_distribution, n_collocation=None):
    """Reads STOPPED_AT_STEP and TARGET_REACHED from every classical
    checkpoint's training_params in `t_dir` (no model loading needed --
    config json only), grouped by INTRA_NEURO. Returns (intra_neurons,
    steps_mean, steps_std, steps_var, steps_min, steps_max,
    target_reached_frac, n_samples, Tfin), one point per INTRA_NEURO found:
    how many AdamW+L-BFGS iterations training actually ran before hitting
    `target_loss` (or the EPOCHS+LBFGS_EPOCHS fallback), averaged over the
    run_id seeds, plus the fraction of seeds that reached `target_loss`
    rather than exhausting the epoch budget. Companion stat to
    `compute_classical_error_stats_std`, read from the same checkpoints but
    reporting training cost instead of accuracy; mirrors
    `compute_classical_stopping_stats_vs_collocation`, but grouped by
    INTRA_NEURO instead of by N_COLLOCATION. Used by
    `plote_classico_vs_Intra.py`."""
    entries = find_classical_configs(
        t_dir, collocation_distribution=collocation_distribution, n_collocation=n_collocation)
    if not entries:
        return [], [], [], [], [], [], [], [], None

    steps_by_neuron = {}
    Tfin = None
    for intra_neuron, _run_id, pt_path in entries:
        config_path = pt_path[: -len(".pt")] + "_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg.get("training_params", {})
        stopped_at_step = training_params.get("STOPPED_AT_STEP")
        if stopped_at_step is None:
            continue
        Tfin = cfg["config"].get("Tfin", Tfin)
        entry = steps_by_neuron.setdefault(intra_neuron, {"steps": [], "reached": []})
        entry["steps"].append(stopped_at_step)
        entry["reached"].append(bool(training_params.get("TARGET_REACHED")))

    intra_neurons = sorted(steps_by_neuron)
    steps_mean = [float(np.mean(steps_by_neuron[n]["steps"])) for n in intra_neurons]
    steps_std = [float(np.std(steps_by_neuron[n]["steps"])) for n in intra_neurons]
    steps_var = [float(np.var(steps_by_neuron[n]["steps"])) for n in intra_neurons]
    steps_min = [float(np.min(steps_by_neuron[n]["steps"])) for n in intra_neurons]
    steps_max = [float(np.max(steps_by_neuron[n]["steps"])) for n in intra_neurons]
    target_reached_frac = [float(np.mean(steps_by_neuron[n]["reached"])) for n in intra_neurons]
    n_samples = [len(steps_by_neuron[n]["steps"]) for n in intra_neurons]
    return (intra_neurons, steps_mean, steps_std, steps_var, steps_min, steps_max,
            target_reached_frac, n_samples, Tfin)


def quantum_steps_stats_from_sweep_summary(csv_path, network, n_cnot_layers, n_collocation, Tfin,
                                            model_family=None):
    """Like `quantum_stats_from_sweep_summary`, but returns the training-cost
    columns (steps until the stopping criterion was hit) instead of MSE:
    (n_qubits, steps_mean, steps_std, steps_var, steps_min, steps_max),
    sorted by n_qubits -- one point per N_QUBITS, in the shape
    `plot_classical_vs_quantum_steps_std` expects. `model_family` behaves as
    in `quantum_stats_from_sweep_summary` -- pass e.g. "quantum_exact" for a
    sweep summary that also carries a classical_mlp row per N_QUBITS.
    Returns ([], [], [], [], [], []) if no rows match."""
    rows = load_quantum_sweep_summary(csv_path)
    matches = [
        row for row in rows
        if row.get("network") == network
        and int(row["n_cnot_layers"]) == n_cnot_layers
        and int(row["n_collocation"]) == n_collocation
        and math.isclose(row["Tfin"], Tfin, rel_tol=1e-9, abs_tol=1e-9)
        and (model_family is None or row.get("model_family") == model_family)
    ]
    matches.sort(key=lambda row: row["n_qubits"])
    n_qubits = [int(row["n_qubits"]) for row in matches]
    steps_mean = [row["steps_mean"] for row in matches]
    steps_std = [row["steps_std"] for row in matches]
    steps_var = [row["steps_variance"] for row in matches]
    steps_min = [row["steps_min"] for row in matches]
    steps_max = [row["steps_max"] for row in matches]
    return n_qubits, steps_mean, steps_std, steps_var, steps_min, steps_max


def plot_classical_vs_quantum_steps_std(stats_by_distribution, quantum_stats, Tfin, save_path,
                                         color_steps="b", color_quantum="#000000", dist_styles=None,
                                         error_type="desvio_padrao"):
    """Like `plot_classical_vs_quantum_std`, but plots training cost (steps
    until the stopping criterion was hit) instead of MSE against INTRA_NEURO
    / N_QUBITS. `stats_by_distribution` maps distribution -> (intra_neurons,
    steps_mean, steps_std, steps_var, steps_min, steps_max) -- the step
    portion of what `compute_classical_stopping_stats_std` returns. Each
    `dist_styles` entry may also set a "color" key to override `color_steps`
    for that one distribution, same convention as `plot_classical_vs_quantum_std`.
    `quantum_stats` is (n_qubits, steps_mean, steps_std, steps_var,
    steps_min, steps_max), e.g. from `quantum_steps_stats_from_sweep_summary`
    or `quantum_steps_stats_vs_n_qubits`; pass ([], [], [], [], [], []) (or
    None) to skip the quantum curve.
    `error_type` selects what the error bars show, same convention as
    `plot_classical_vs_quantum_std` -- one of `ERROR_BAR_TYPES`. Used by
    `plote_classico_vs_Intra.py`."""
    if dist_styles is None:
        dist_styles = {
            "uniform": dict(fmt="o-", markerfacecolor=None),
            "random": dict(fmt="s--", markerfacecolor="none"),
        }
    error_labels = {
        "minimo_maximo": "min/max",
        "desvio_padrao": "±std",
        "variancia": "±var",
    }
    error_label = error_labels.get(error_type, error_type)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_widths = set()
    for distribution, (intra_neurons, steps_mean, steps_std, steps_var, steps_min,
                        steps_max) in stats_by_distribution.items():
        all_widths.update(intra_neurons)
        style = dist_styles.get(distribution, dict(fmt="o-", markerfacecolor=None))
        steps_yerr = error_bar_yerr(error_type, steps_mean, steps_std, steps_var, steps_min, steps_max)
        ax.errorbar(intra_neurons, steps_mean, yerr=steps_yerr, fmt=style["fmt"],
                    color=style.get("color", color_steps),
                    markerfacecolor=style["markerfacecolor"], linewidth=2.0, markersize=6,
                    capsize=4, label=f"Classical {error_label})")

    if quantum_stats is not None:
        n_qubits, q_steps_mean, q_steps_std, q_steps_var, q_steps_min, q_steps_max = quantum_stats
        if n_qubits:
            all_widths.update(n_qubits)
            q_yerr = error_bar_yerr(error_type, q_steps_mean, q_steps_std, q_steps_var,
                                     q_steps_min, q_steps_max)
            ax.errorbar(n_qubits, q_steps_mean, yerr=q_yerr, fmt="^-", color=color_quantum,
                        linewidth=2.0, markersize=6, capsize=4,
                        label=f"Quantum  {error_label})")

    ax.set_xlabel("N° of Neurons / N° of Qubits", fontsize=14)
    ax.set_ylabel("Steps of train", fontsize=14)
    ax.set_xticks(sorted(all_widths))
    ax.set_title(f"T = {Tfin}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def find_classical_configs_by_n_collocation(t_dir, collocation_distribution=None, intra_neuron=None):
    """Returns [(n_collocation, run_id, pt_path), ...] for every classical
    Cmodel_NEURO* checkpoint in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution` ("uniform" or "random"; leave as None
    to include every distribution found) and whose INTRA_NEURO architecture
    matches `intra_neuron` (leave as None to include every width found),
    reading N_COLLOCATION from each checkpoint's training_params. Mirrors
    `find_classical_configs`, but keyed by N_COLLOCATION (the collocation
    -sampling sweep in `run_classical.py`) instead of by INTRA_NEURO (the
    width sweep)."""
    entries = []
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_*_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg.get("training_params", {})
        distribution = training_params.get("COLLOCATION_DISTRIBUTION")
        if collocation_distribution is not None and distribution != collocation_distribution:
            continue
        if intra_neuron is not None and cfg["config"].get("intra_neuron") != intra_neuron:
            continue
        n_collocation = training_params.get("N_COLLOCATION")
        if n_collocation is None:
            continue
        run_id = int(os.path.basename(pt_path)[: -len(".pt")].rsplit("_", 1)[1])
        entries.append((n_collocation, run_id, pt_path))
    return sorted(entries)


def compute_classical_error_stats_vs_collocation(t_dir, model_cls, nu, collocation_distribution,
                                                   intra_neuron=None):
    """Loads every classical model in `t_dir` matching `collocation_distribution`
    (and, when given, a fixed `intra_neuron` architecture width), evaluates
    each against the FDM reference at t=Tfin, and returns per-N_COLLOCATION
    mean/std/variance MSE and MAE across the run_id seeds. Returns
    (n_collocation, mse_mean, mse_std, mse_var, mae_mean, mae_std, mae_var,
    n_samples, Tfin), one point per N_COLLOCATION value actually found.
    Companion to `plot_classical_error_vs_collocation`; mirrors
    `compute_classical_error_stats_std`, but grouped by N_COLLOCATION
    instead of by INTRA_NEURO."""
    entries = find_classical_configs_by_n_collocation(
        t_dir, collocation_distribution=collocation_distribution, intra_neuron=intra_neuron)
    if not entries:
        return [], [], [], [], [], [], [], [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_n = {}
    for n_collocation, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_n.setdefault(n_collocation, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    n_collocation_list = sorted(errors_by_n)
    mse_mean = [float(np.mean(errors_by_n[n]["mse"])) for n in n_collocation_list]
    mse_std = [float(np.std(errors_by_n[n]["mse"])) for n in n_collocation_list]
    mse_var = [float(np.var(errors_by_n[n]["mse"])) for n in n_collocation_list]
    mae_mean = [float(np.mean(errors_by_n[n]["mae"])) for n in n_collocation_list]
    mae_std = [float(np.std(errors_by_n[n]["mae"])) for n in n_collocation_list]
    mae_var = [float(np.var(errors_by_n[n]["mae"])) for n in n_collocation_list]
    n_samples = [len(errors_by_n[n]["mse"]) for n in n_collocation_list]
    return (n_collocation_list, mse_mean, mse_std, mse_var, mae_mean, mae_std, mae_var,
            n_samples, Tfin)


# UNUSED (not called anywhere in the codebase -- only mentioned in a docstring;
# plote_classico_vs_collocation.py defines its own local plot function instead)
# -- commented out 2026-09-03.
# def plot_classical_error_vs_collocation(n_collocation, mse_mean, mse_var, Tfin, save_path,
#                                          color_mse="#2a78d6"):
#     """MSE (mean +/- variance across seeds) between the classical PINN and
#     the FDM reference, plotted against N_COLLOCATION. Points sit at equally
#     spaced positions on the x-axis regardless of the actual gap between
#     consecutive N_COLLOCATION values (e.g. 100 -> 250 -> 500) -- only the
#     tick label at each position shows the real value -- so the sampling
#     density along the sweep doesn't visually distort the curve. Companion
#     plot for `compute_classical_error_stats_vs_collocation`."""
#     fig, ax = plt.subplots(figsize=(9, 4.5))
#     positions = np.arange(len(n_collocation))
#     ax.errorbar(positions, mse_mean, yerr=mse_var, fmt="o-", color=color_mse,
#                 linewidth=2.0, markersize=6, capsize=4, label="MSE (mean +/- variance)")
#     ax.set_xlabel("N_COLLOCATION", fontsize=14)
#     ax.set_ylabel("MSE", fontsize=14)
#     ax.set_yscale("log")
#     ax.set_xticks(positions)
#     ax.set_xticklabels([str(n) for n in n_collocation])
#     ax.set_title(f"Classical PINN: MSE vs N_COLLOCATION (T = {Tfin})", fontsize=13)
#     ax.grid(True, which="both", linestyle="--", alpha=0.6)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)


def compute_classical_stopping_stats_vs_collocation(t_dir, collocation_distribution, intra_neuron=None):
    """Reads STOPPED_AT_STEP and TARGET_REACHED from every classical
    checkpoint's training_params in `t_dir` (no model loading needed --
    config json only), grouped by N_COLLOCATION. Returns (n_collocation,
    steps_mean, steps_std, steps_var, target_reached_frac, n_samples, Tfin),
    one point per N_COLLOCATION value found: how many AdamW+L-BFGS
    iterations training actually ran before hitting `target_loss` (or the
    EPOCHS+LBFGS_EPOCHS fallback), averaged over the run_id seeds, plus the
    fraction of seeds that reached `target_loss` rather than exhausting the
    epoch budget. Companion stat to `compute_classical_error_stats_vs_collocation`,
    read from the same checkpoints but reporting training cost instead of
    accuracy."""
    steps_by_n = {}
    Tfin = None
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_*_NEURO*_config.json")):
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg.get("training_params", {})
        distribution = training_params.get("COLLOCATION_DISTRIBUTION")
        if collocation_distribution is not None and distribution != collocation_distribution:
            continue
        if intra_neuron is not None and cfg["config"].get("intra_neuron") != intra_neuron:
            continue
        n_collocation = training_params.get("N_COLLOCATION")
        stopped_at_step = training_params.get("STOPPED_AT_STEP")
        if n_collocation is None or stopped_at_step is None:
            continue
        Tfin = cfg["config"].get("Tfin", Tfin)
        entry = steps_by_n.setdefault(n_collocation, {"steps": [], "reached": []})
        entry["steps"].append(stopped_at_step)
        entry["reached"].append(bool(training_params.get("TARGET_REACHED")))

    n_collocation_list = sorted(steps_by_n)
    steps_mean = [float(np.mean(steps_by_n[n]["steps"])) for n in n_collocation_list]
    steps_std = [float(np.std(steps_by_n[n]["steps"])) for n in n_collocation_list]
    steps_var = [float(np.var(steps_by_n[n]["steps"])) for n in n_collocation_list]
    target_reached_frac = [float(np.mean(steps_by_n[n]["reached"])) for n in n_collocation_list]
    n_samples = [len(steps_by_n[n]["steps"]) for n in n_collocation_list]
    return n_collocation_list, steps_mean, steps_std, steps_var, target_reached_frac, n_samples, Tfin


# UNUSED (not called anywhere in the codebase) -- commented out 2026-09-03.
# def plot_classical_stopping_steps_vs_collocation(n_collocation, steps_mean, steps_var, Tfin, save_path,
#                                                    color_mean="#2a78d6", color_var="#e34948"):
#     """Number of training steps (AdamW + L-BFGS) until the stopping
#     criterion was hit, vs N_COLLOCATION -- mean on the left axis, variance
#     across the run_id seeds on the right axis (plotted separately, not as an
#     error bar: mean and variance of a step *count* live on very different
#     scales -- steps vs steps^2 -- so overlaying them as mean +/- variance
#     would put the whisker far outside the visible range). Points sit at
#     equally spaced x positions, same convention as
#     `plot_classical_error_vs_collocation`. Companion plot for
#     `compute_classical_stopping_stats_vs_collocation`."""
#     fig, ax1 = plt.subplots(figsize=(9, 4.5))
#     positions = np.arange(len(n_collocation))
# 
#     line_mean = ax1.plot(positions, steps_mean, "o-", color=color_mean, linewidth=2.0,
#                           markersize=6, label="Média de passos")
#     ax1.set_xlabel("N_COLLOCATION", fontsize=14)
#     ax1.set_ylabel("Passos até a parada (média)", fontsize=14, color=color_mean)
#     ax1.tick_params(axis="y", labelcolor=color_mean)
#     ax1.set_xticks(positions)
#     ax1.set_xticklabels([str(n) for n in n_collocation])
#     ax1.grid(True, linestyle="--", alpha=0.6)
# 
#     ax2 = ax1.twinx()
#     line_var = ax2.plot(positions, steps_var, "s--", color=color_var, linewidth=2.0,
#                          markersize=6, label="Variância de passos")
#     ax2.set_ylabel("Variância dos passos", fontsize=14, color=color_var)
#     ax2.tick_params(axis="y", labelcolor=color_var)
# 
#     lines = line_mean + line_var
#     ax1.legend(lines, [l.get_label() for l in lines], loc="best")
#     #ax1.set_yscale("log")
#     #ax1.set_title(f"Classical PINN: passos até a parada vs N_COLLOCATION (T = {Tfin})", fontsize=13)
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)


# UNUSED (only referenced by main/old/plote_quantico_std.py, not by any other file) --
# commented out 2026-09-03.
# def compute_quantum_error_stats_std(t_dir, model_cls, nu, input_rotation_axis="ry"):
#     """Loads every quantum model in `t_dir`, evaluates each against the FDM
#     reference at t=Tfin, and returns per-(FIXED_LAYER, N_QUBITS) mean/std/
#     min/max MAE across the SEED samples. Returns (stats, Tfin), where stats
#     is {fixed_layer: (n_qubits, mae_mean, mae_std, mae_min, mae_max,
#     n_samples)}, one entry per FIXED_LAYER found. mae_min/mae_max (the
#     extremes actually observed across seeds) are what the plot uses for its
#     error bars -- unlike mean +/- std, they can never go negative on the
#     plot's log-scale y-axis. Used by `plote_quantico_std.py`."""
#     entries = find_quantum_configs(t_dir, input_rotation_axis=input_rotation_axis)
#     if not entries:
#         return {}, None
# 
#     first_model = model_cls.load(entries[0][3])
#     L, Tfin = first_model.L, first_model.Tfin
# 
#     x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
#     x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
#     u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
# 
#     errors_by_layer = {}
#     for fixed_layer, n_qubits, _seed, pt_path in entries:
#         model = model_cls.load(pt_path)
#         u_pred = predict(model, x_plot, Tfin)
#         by_n_qubits = errors_by_layer.setdefault(fixed_layer, {})
#         by_n_qubits.setdefault(n_qubits, []).append(float(np.mean(np.abs(u_pred - u_ref))))
# 
#     stats = {}
#     for fixed_layer, by_n_qubits in errors_by_layer.items():
#         n_qubits_list = sorted(by_n_qubits)
#         mae_mean = [float(np.mean(by_n_qubits[n])) for n in n_qubits_list]
#         mae_std = [float(np.std(by_n_qubits[n])) for n in n_qubits_list]
#         mae_min = [float(np.min(by_n_qubits[n])) for n in n_qubits_list]
#         mae_max = [float(np.max(by_n_qubits[n])) for n in n_qubits_list]
#         n_samples = [len(by_n_qubits[n]) for n in n_qubits_list]
#         stats[fixed_layer] = (n_qubits_list, mae_mean, mae_std, mae_min, mae_max, n_samples)
#     return stats, Tfin


# UNUSED (only referenced by main/old/plote_quantico_std.py, not by any other file) --
# commented out 2026-09-03.
# def plot_quantum_errors_std(stats, Tfin, save_path, colors=None):
#     if colors is None:
#         colors = ["#2a78d6", "#e34948", "#eb6834", "#4a3aa7", "#1baf7a", "#e87ba4"]
#     fig, ax = plt.subplots(figsize=(9, 4.5))
#     all_n_qubits = set()
#     for (fixed_layer, (n_qubits, mae_mean, _mae_std, mae_min, mae_max, _n_samples)), color in zip(
#             sorted(stats.items()), colors):
#         all_n_qubits.update(n_qubits)
#         yerr = asymmetric_yerr(mae_mean, mae_min, mae_max)
#         ax.errorbar(n_qubits, mae_mean, yerr=yerr, fmt="o-", color=color,
#                     linewidth=2.0, markersize=6, capsize=4, label=fixed_layer)
#     ax.set_xlabel("Number of qubits", fontsize=14)
#     ax.set_ylabel("Error", fontsize=14)
#     ax.set_yscale("log")
#     ax.set_xticks(sorted(all_n_qubits))
#     #ax.set_title(f"Quantum PINN MAE vs number of qubits (T = {Tfin})", fontsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150)
#     plt.close(fig)


# UNUSED (only referenced by main/old/plote_comparativo.py, not by any other file) --
# commented out 2026-09-03.
# def compute_quantum_errors_comparativo(t_dir, model_cls, x_plot, u_ref, Tfin,
#                                         input_rotation_axis, collocation_distribution):
#     """Per-(FIXED_LAYER, N_QUBITS) MSE/MAE stats against a caller-supplied
#     `u_ref` grid, used by `plote_comparativo.py` to compare quantum models
#     against the classical baseline on the same x_plot/u_ref/Tfin."""
#     entries = find_quantum_configs(
#         t_dir, input_rotation_axis=input_rotation_axis, collocation_distribution=collocation_distribution
#     )
# 
#     errors_by_layer = {}
#     for fixed_layer, n_qubits, _seed, pt_path in entries:
#         model = model_cls.load(pt_path)
#         u_pred = predict(model, x_plot, Tfin)
#         by_n = errors_by_layer.setdefault(fixed_layer, {})
#         entry = by_n.setdefault(n_qubits, {"mse": [], "mae": []})
#         entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
#         entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))
# 
#     stats = {}
#     for fixed_layer, by_n in errors_by_layer.items():
#         n_list = sorted(by_n)
#         mse_mean = [float(np.mean(by_n[n]["mse"])) for n in n_list]
#         mse_std = [float(np.std(by_n[n]["mse"])) for n in n_list]
#         mse_min = [float(np.min(by_n[n]["mse"])) for n in n_list]
#         mse_max = [float(np.max(by_n[n]["mse"])) for n in n_list]
#         mae_mean = [float(np.mean(by_n[n]["mae"])) for n in n_list]
#         mae_std = [float(np.std(by_n[n]["mae"])) for n in n_list]
#         n_samples = [len(by_n[n]["mse"]) for n in n_list]
#         stats[fixed_layer] = (n_list, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples)
#     return stats


# UNUSED (only referenced by main/old/plote_comparativo.py, not by any other file) --
# commented out 2026-09-03.
# def compute_classical_errors_comparativo(t_dir, model_cls, x_plot, u_ref, Tfin, collocation_distribution):
#     """Per-INTRA_NEURO MSE/MAE stats against a caller-supplied `u_ref` grid,
#     used by `plote_comparativo.py` to compare classical models against the
#     quantum ones on the same x_plot/u_ref/Tfin."""
#     entries = find_classical_configs(t_dir, collocation_distribution=collocation_distribution)
#     if not entries:
#         return None
# 
#     errors_by_neuron = {}
#     for intra_neuron, _run_id, pt_path in entries:
#         model = model_cls.load(pt_path)
#         u_pred = predict(model, x_plot, Tfin)
#         entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
#         entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
#         entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))
# 
#     intra_neurons = sorted(errors_by_neuron)
#     mse_mean = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
#     mse_std = [float(np.std(errors_by_neuron[n]["mse"])) for n in intra_neurons]
#     mse_min = [float(np.min(errors_by_neuron[n]["mse"])) for n in intra_neurons]
#     mse_max = [float(np.max(errors_by_neuron[n]["mse"])) for n in intra_neurons]
#     mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
#     mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
#     n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
#     return intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples


# UNUSED (only referenced by main/old/plote_comparativo.py, not by any other file) --
# commented out 2026-09-03.
# def plot_comparison(quantum_stats, classical_data, Tfin, save_path,
#                      color_classical="black", colors_quantum=None, fallback_colors=None):
#     if colors_quantum is None:
#         colors_quantum = {"cz_ring": "#2a78d6", "ring_cnot": "#1baf7a"}
#     if fallback_colors is None:
#         fallback_colors = ["#eb6834", "#4a3aa7", "#e87ba4"]
# 
#     fig, ax = plt.subplots(figsize=(9, 4.5))
#     all_widths = set()
# 
#     fallback = iter(fallback_colors)
#     for fixed_layer, (n_qubits, mse_mean, _mse_std, mse_min, mse_max, _mae_mean, _mae_std, _n_samples) in sorted(
#         quantum_stats.items()
#     ):
#         color = colors_quantum.get(fixed_layer, next(fallback, None))
#         all_widths.update(n_qubits)
#         yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
#         ax.errorbar(n_qubits, mse_mean, yerr=yerr, fmt="o-", color=color, linewidth=2.0,
#                     markersize=6, capsize=4, label=f"Quantum PINN ({fixed_layer})")
# 
#     if classical_data is not None:
#         intra_neurons, mse_mean, _mse_std, mse_min, mse_max, _mae_mean, _mae_std, _n_samples = classical_data
#         all_widths.update(intra_neurons)
#         yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
#         ax.errorbar(intra_neurons, mse_mean, yerr=yerr, fmt="s--", color=color_classical,
#                     linewidth=2.0, markersize=6, capsize=4, label="Classical PINN")
# 
#     ax.set_xlabel("Width (N_QUBITS / INTRA_NEURO)", fontsize=14)
#     ax.set_ylabel("MSE", fontsize=14)
#     ax.set_yscale("log")
#     ax.set_xticks(sorted(all_widths))
#     #ax.set_title(f"Quantum vs classical PINN mean MSE vs width (T = {Tfin})", fontsize=13)
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=250)


# ----------------------------------------------------------------------
# Inverse-problem (nu estimation) classical PINN sweeps -- checkpoints
# written by Inverser_classical_pinn_paralelo.py as
# data/classical_inverse/T_<Tfin>/model_<n_col>_INTRA<intra_neuron>_<run_id>.pt,
# each holding its own "training_params" dict (NU, NU_LEARNED, N_COLLOCATION,
# INTRA_NEURON, TFIN, ...) -- unlike the forward-problem checkpoints above,
# there is no companion `_config.json`.
# ----------------------------------------------------------------------

def find_classical_inverse_configs(t_dir, n_collocation=None, intra_neuron=None):
    """Returns [(n_collocation, intra_neuron, run_id, training_params), ...]
    for every `model_<n_col>_INTRA<intra_neuron>_<run_id>.pt` checkpoint in
    `t_dir`, optionally filtered to a fixed `n_collocation` and/or
    `intra_neuron`. `training_params` is the dict saved inside the
    checkpoint itself (no model weights are loaded -- only the small
    metadata dict, via `tc.load(..., map_location="cpu")`)."""
    entries = []
    for pt_path in glob.glob(os.path.join(t_dir, "model_*_INTRA*_*.pt")):
        name = os.path.basename(pt_path)[: -len(".pt")]
        body = name[len("model_"):]
        n_col_str, rest = body.split("_INTRA", 1)
        intra_str, run_id_str = rest.rsplit("_", 1)
        n_col, intra, run_id = int(n_col_str), int(intra_str), int(run_id_str)
        if n_collocation is not None and n_col != n_collocation:
            continue
        if intra_neuron is not None and intra != intra_neuron:
            continue
        checkpoint = tc.load(pt_path, map_location="cpu", weights_only=False)
        training_params = checkpoint["training_params"]
        entries.append((n_col, intra, run_id, training_params))
    return sorted(entries, key=lambda e: (e[0], e[1], e[2]))


def _nu_error_stats_by_key(entries, key_index):
    """Shared aggregation for the two functions below: groups `entries`
    (as returned by `find_classical_inverse_configs`) by
    `entries[i][key_index]` and returns per-key mean/std/var/min/max of the
    squared and absolute nu-estimation error (NU_LEARNED vs NU), plus
    n_samples and Tfin."""
    errors_by_key = {}
    Tfin = None
    for n_col, intra, _run_id, training_params in entries:
        key = n_col if key_index == 0 else intra
        nu_true = training_params["NU"]
        nu_learned = training_params["NU_LEARNED"]
        Tfin = training_params.get("TFIN", Tfin)
        entry = errors_by_key.setdefault(key, {"mse": [], "abs": []})
        entry["mse"].append((nu_learned - nu_true) ** 2)
        entry["abs"].append(abs(nu_learned - nu_true))

    keys = sorted(errors_by_key)
    mse_mean = [float(np.mean(errors_by_key[k]["mse"])) for k in keys]
    mse_std = [float(np.std(errors_by_key[k]["mse"])) for k in keys]
    mse_var = [float(np.var(errors_by_key[k]["mse"])) for k in keys]
    mse_min = [float(np.min(errors_by_key[k]["mse"])) for k in keys]
    mse_max = [float(np.max(errors_by_key[k]["mse"])) for k in keys]
    abs_mean = [float(np.mean(errors_by_key[k]["abs"])) for k in keys]
    abs_std = [float(np.std(errors_by_key[k]["abs"])) for k in keys]
    abs_var = [float(np.var(errors_by_key[k]["abs"])) for k in keys]
    abs_min = [float(np.min(errors_by_key[k]["abs"])) for k in keys]
    abs_max = [float(np.max(errors_by_key[k]["abs"])) for k in keys]
    n_samples = [len(errors_by_key[k]["mse"]) for k in keys]
    return (keys, mse_mean, mse_std, mse_var, mse_min, mse_max, abs_mean, abs_std,
            abs_var, abs_min, abs_max, n_samples, Tfin)


def compute_classical_inverse_nu_error_stats_vs_intra(t_dir, n_collocation=None):
    """Groups every classical inverse-problem checkpoint in `t_dir` (whose
    N_COLLOCATION matches `n_collocation`, when given) by INTRA_NEURON and
    returns per-INTRA_NEURON mean/std/variance/min/max of the squared and
    absolute nu-estimation error (NU_LEARNED vs the true NU). Returns
    (intra_neurons, nu_mse_mean, nu_mse_std, nu_mse_var, nu_mse_min,
    nu_mse_max, nu_abs_mean, nu_abs_std, nu_abs_var, nu_abs_min, nu_abs_max,
    n_samples, Tfin), one point per INTRA_NEURON found. Mirrors
    `compute_classical_error_stats_std`, but the metric is nu-estimation
    error instead of solution MSE/MAE against the FDM reference. Used by
    `plote_inverse_classico_vs_Intra.py`."""
    entries = find_classical_inverse_configs(t_dir, n_collocation=n_collocation)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], [], [], None
    return _nu_error_stats_by_key(entries, key_index=1)


def compute_classical_inverse_nu_error_stats_vs_collocation(t_dir, intra_neuron=None):
    """Groups every classical inverse-problem checkpoint in `t_dir` (whose
    INTRA_NEURON matches `intra_neuron`, when given) by N_COLLOCATION and
    returns per-N_COLLOCATION mean/std/variance/min/max of the squared and
    absolute nu-estimation error (NU_LEARNED vs the true NU). Returns
    (n_collocation, nu_mse_mean, nu_mse_std, nu_mse_var, nu_mse_min,
    nu_mse_max, nu_abs_mean, nu_abs_std, nu_abs_var, nu_abs_min, nu_abs_max,
    n_samples, Tfin), one point per N_COLLOCATION value found. Mirrors
    `compute_classical_error_stats_vs_collocation`, but the metric is
    nu-estimation error instead of solution MSE/MAE against the FDM
    reference. Used by `plote_inverse_classico_vs_collocation.py`."""
    entries = find_classical_inverse_configs(t_dir, intra_neuron=intra_neuron)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], [], [], None
    return _nu_error_stats_by_key(entries, key_index=0)


# ----------------------------------------------------------------------
# Inverse-problem (nu estimation) quantum PINN sweeps -- checkpoints
# written by Inverser_quantum_pinn_paralelo.py as
# data/quantum_inverse/T_<Tfin>/model_<n_col>_NQ<n_qubits>_<run_id>.pt.
# Mirrors the classical inverse-problem helpers above exactly, with
# N_QUBITS in place of INTRA_NEURON.
# ----------------------------------------------------------------------

def find_quantum_inverse_configs(t_dir, n_collocation=None, n_qubits=None):
    """Returns [(n_collocation, n_qubits, run_id, training_params), ...]
    for every `model_<n_col>_NQ<n_qubits>_<run_id>.pt` checkpoint in
    `t_dir`, optionally filtered to a fixed `n_collocation` and/or
    `n_qubits`. Mirrors `find_classical_inverse_configs`."""
    entries = []
    for pt_path in glob.glob(os.path.join(t_dir, "model_*_NQ*_*.pt")):
        name = os.path.basename(pt_path)[: -len(".pt")]
        body = name[len("model_"):]
        n_col_str, rest = body.split("_NQ", 1)
        nq_str, run_id_str = rest.rsplit("_", 1)
        n_col, nq, run_id = int(n_col_str), int(nq_str), int(run_id_str)
        if n_collocation is not None and n_col != n_collocation:
            continue
        if n_qubits is not None and nq != n_qubits:
            continue
        checkpoint = tc.load(pt_path, map_location="cpu", weights_only=False)
        training_params = checkpoint["training_params"]
        entries.append((n_col, nq, run_id, training_params))
    return sorted(entries, key=lambda e: (e[0], e[1], e[2]))


def compute_quantum_inverse_nu_error_stats_vs_intra(t_dir, n_collocation=None):
    """Groups every quantum inverse-problem checkpoint in `t_dir` (whose
    N_COLLOCATION matches `n_collocation`, when given) by N_QUBITS and
    returns per-N_QUBITS mean/std/variance/min/max of the squared and
    absolute nu-estimation error (NU_LEARNED vs the true NU). Returns
    (n_qubits, nu_mse_mean, nu_mse_std, nu_mse_var, nu_mse_min, nu_mse_max,
    nu_abs_mean, nu_abs_std, nu_abs_var, nu_abs_min, nu_abs_max, n_samples,
    Tfin), one point per N_QUBITS found. Mirrors
    `compute_classical_inverse_nu_error_stats_vs_intra`. Used by
    `plote_inverse_classico_vs_Intra.py`."""
    entries = find_quantum_inverse_configs(t_dir, n_collocation=n_collocation)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], [], [], None
    return _nu_error_stats_by_key(entries, key_index=1)


def compute_quantum_inverse_nu_error_stats_vs_collocation(t_dir, n_qubits=None):
    """Groups every quantum inverse-problem checkpoint in `t_dir` (whose
    N_QUBITS matches `n_qubits`, when given) by N_COLLOCATION and returns
    per-N_COLLOCATION mean/std/variance/min/max of the squared and absolute
    nu-estimation error (NU_LEARNED vs the true NU). Returns (n_collocation,
    nu_mse_mean, nu_mse_std, nu_mse_var, nu_mse_min, nu_mse_max, nu_abs_mean,
    nu_abs_std, nu_abs_var, nu_abs_min, nu_abs_max, n_samples, Tfin), one
    point per N_COLLOCATION value found. Mirrors
    `compute_classical_inverse_nu_error_stats_vs_collocation`. Used by
    `plote_inverse_classico_vs_collocation.py`."""
    entries = find_quantum_inverse_configs(t_dir, n_qubits=n_qubits)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], [], [], None
    return _nu_error_stats_by_key(entries, key_index=0)
