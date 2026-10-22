import json
import math
import os

import numpy as np
import torch as tc
import torch.nn as nn

from .functions import (
    cnot_cosine_expectations,
    cnot_cosine_product_expressions,
    initial_condition,
    repeated_cnot_network_matrix,
)

try:
    from qiskit.primitives import StatevectorEstimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
except ModuleNotFoundError:
    StatevectorEstimator = None
    EstimatorQNN = None
    TorchConnector = None
    ParamShiftEstimatorGradient = None
    QuantumCircuit = None
    ParameterVector = None
    SparsePauliOp = None

class PINNBase(nn.Module):
    """
    Mixin adding checkpoint save/load to PINN models. Subclasses must set
    `self.config` in `__init__` to the exact kwargs needed to reconstruct
    the model (i.e. `cls(**config)` must work).
    """

    def save(self, path, training_params=None):
        """Saves weights + config to `path` (state_dict + reconstruction
        kwargs), plus a human-readable `<path>_config.json` alongside it so
        the run configuration can be inspected without loading torch.

        `training_params` is optional extra metadata (e.g. NU, HX, HT,
        N_COLLOCATION, SEED, EPOCHS, LR, HIDDEN_NEURONS, HIDDEN_DEPTH) recorded
        for bookkeeping only -- it is not part of `config` and is never passed
        to `cls(**config)` on load."""
        path = str(path)
        payload = {
            "class_name": type(self).__name__,
            "config": self.config,
            "state_dict": self.state_dict(),
        }
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


class ClassicalPINN(PINNBase):
    """Fully classical MLP ansatz for the 1D viscous Burgers equation."""

    def __init__(self, hidden_neurons=20, hidden_depth=2, L=1.0, Tfin=0.1):
        super().__init__()
        self.config = dict(hidden_neurons=hidden_neurons, hidden_depth=hidden_depth, L=L, Tfin=Tfin)
        self.L, self.Tfin = float(L), float(Tfin)

        layers = [nn.Linear(3, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        layers += [nn.Linear(hidden_neurons, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        input, x = self.Input_angular(xt)
        output = self.net(input)
        return self.Hard_constrain(x, input[:, 2:3], output)

    def Input_angular(self, xt):
        """Maps (x, t) to periodic (sin, cos, tau) features so the network sees x through its L-periodic image."""
        x, tt = xt[:, 0:1], xt[:, 1:2]
        input = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tt / self.Tfin],
            dim=1,
        )
        return input, x

    def Hard_constrain(self, x, tau, output_nn):
        """Hard-imposes the initial condition at tau=0 by construction."""
        return initial_condition(x, L=self.L) + tau * output_nn

class SIN(tc.nn.Module):
    def __init__(self): 
        super(SIN, self).__init__() 
    def forward(self, x):
        return tc.sin(x)

class ClassicalPINN2(PINNBase):

    def __init__(self,
                 pre_neurons, pre_depth,
                 intra_neuron, intra_neurons, intra_depth,
                 post_neurons, post_depth, 
                 L=1.0, Tfin=0.1,
                 seno=False):
        super().__init__()
        self.config = dict(pre_neurons=pre_neurons, pre_depth=pre_depth,
                         intra_neuron=intra_neuron, intra_neurons=intra_neurons, intra_depth=intra_depth,
                         post_neurons=post_neurons, post_depth=post_depth,
                           L=L, Tfin=Tfin, seno=seno)
        self.L, self.Tfin = float(L), float(Tfin)


        # Pré layers
        pre_layers = [nn.Linear(3, pre_neurons), nn.Tanh()]
        for _ in range(pre_depth - 1):
            pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
        pre_layers += [nn.Linear(pre_neurons, intra_neuron)]
        self.pre = nn.Sequential(*pre_layers)

        # Intra layers
        if seno==True:
            layers = [nn.Linear(intra_neuron, intra_neurons), SIN()]
            for _ in range(intra_depth - 1):
                layers += [nn.Linear(intra_neurons, intra_neurons), SIN()]
            layers += [nn.Linear(intra_neurons, intra_neuron)]
            self.net = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(intra_neuron, intra_neurons), nn.Tanh()]
            for _ in range(intra_depth - 1):
                layers += [nn.Linear(intra_neurons, intra_neurons), nn.Tanh()]
            layers += [nn.Linear(intra_neurons, intra_neuron)]
            self.net = nn.Sequential(*layers)

        # Pós layers
        post_layers = [nn.Linear(intra_neuron, post_neurons), nn.Tanh()]
        for _ in range(post_depth - 1):
            post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
        post_layers += [nn.Linear(post_neurons, 1)]
        self.post = nn.Sequential(*post_layers)
        
    def forward(self, xt):
        input, x = self.Input_angular(xt)
        output = self.pre(input)
        output = self.net(output)
        output = self.post(output)
        return self.Hard_constrain(x, input[:, 2:3], output)

    def Input_angular(self, xt):
        """Maps (x, t) to periodic (sin, cos, tau) features so the network sees x through its L-periodic image."""
        x, tt = xt[:, 0:1], xt[:, 1:2]
        input = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tt / self.Tfin],
            dim=1,
        )
        return input, x

    def Hard_constrain(self, x, tau, output_nn):
        """Hard-imposes the initial condition at tau=0 by construction."""
        return initial_condition(x, L=self.L) + tau * output_nn


class QuantumExactPINN(PINNBase):
    r"""
    Exact cosine-product HPINN with no Qiskit simulation.

    Its architecture is the direct three-block counterpart of `ClassicalPINN2`:

        periodic inputs -> pre network -> fixed CNOT feature map -> post network.

    The classical internal MLP is replaced by a parameter-free circuit composed
    of an Rx/Ry product encoding and fixed CNOT layers. For the Z expectations,
    binary circuit algebra gives the exact differentiable expression

        <Z_i> = product_j cos(theta_j)**M[i,j],

    where M is the complete repeated-CNOT matrix over GF(2). Consequently this
    model is algebraically identical to the corresponding noiseless quantum
    circuit for these observables, but requires neither statevectors nor shots.

    For an architecture-matched comparison with `ClassicalPINN2`, use
    `intra_neuron=n_qubits` and the same pre/post widths and depths. This matches
    the interfaces and outer networks; it does not match the total parameter
    count because the fixed quantum block has no trainable parameters.
    """

    def __init__(
        self,
        n_qubits=5,
        pre_neurons=10,
        pre_depth=1,
        post_neurons=10,
        post_depth=1,
        fixed_layer="linear_cnot",
        fixed_layer_repeats=1,
        input_rotation_axis="rx",
        angle_scale=None,
        L=1.0,
        Tfin=0.1,
    ):
        super().__init__()

        integer_options = {
            "n_qubits": n_qubits,
            "pre_neurons": pre_neurons,
            "pre_depth": pre_depth,
            "post_neurons": post_neurons,
            "post_depth": post_depth,
        }
        for name, value in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if (
            isinstance(fixed_layer_repeats, bool)
            or not isinstance(fixed_layer_repeats, (int, np.integer))
        ):
            raise TypeError("fixed_layer_repeats must be an integer.")
        if fixed_layer_repeats < 0:
            raise ValueError("fixed_layer_repeats must be non-negative.")
        if angle_scale is not None:
            if isinstance(angle_scale, bool) or not isinstance(
                angle_scale, (int, float, np.integer, np.floating)
            ):
                raise TypeError("angle_scale must be a real number or None.")
            if not math.isfinite(float(angle_scale)) or float(angle_scale) <= 0.0:
                raise ValueError("angle_scale must be finite and positive.")

        input_rotation_axis = str(input_rotation_axis).lower()
        if input_rotation_axis not in ("rx", "ry"):
            raise ValueError(
                "The exact cosine-product model supports only 'rx' and 'ry' "
                "input rotations. Rz acting on |0> does not produce this feature map."
            )

        self.n_qubits = int(n_qubits)
        self.fixed_layer = str(fixed_layer).lower()
        if self.fixed_layer == "inverse_linear_cnot":
            self.fixed_layer = "reverse_linear_cnot"
        self.fixed_layer_repeats = int(fixed_layer_repeats)
        self.input_rotation_axis = input_rotation_axis
        self.angle_scale = None if angle_scale is None else float(angle_scale)
        self.L, self.Tfin = float(L), float(Tfin)
        if self.L <= 0.0 or self.Tfin <= 0.0:
            raise ValueError("L and Tfin must be positive.")

        self.config = dict(
            n_qubits=self.n_qubits,
            pre_neurons=int(pre_neurons),
            pre_depth=int(pre_depth),
            post_neurons=int(post_neurons),
            post_depth=int(post_depth),
            fixed_layer=self.fixed_layer,
            fixed_layer_repeats=self.fixed_layer_repeats,
            input_rotation_axis=self.input_rotation_axis,
            angle_scale=self.angle_scale,
            L=self.L,
            Tfin=self.Tfin,
        )

        # Same pre-network layout as ClassicalPINN2 with intra_neuron=n_qubits.
        pre_layers = [nn.Linear(3, pre_neurons), nn.Tanh()]
        for _ in range(pre_depth - 1):
            pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
        pre_layers += [nn.Linear(pre_neurons, self.n_qubits)]
        self.pre = nn.Sequential(*pre_layers)

        complete_matrix = repeated_cnot_network_matrix(
            n_qubits=self.n_qubits,
            network_kind=self.fixed_layer,
            n_layers=self.fixed_layer_repeats,
        )
        # A persistent buffer follows the model across devices and checkpoints,
        # but is not a trainable parameter.
        self.register_buffer(
            "cnot_matrix",
            tc.as_tensor(complete_matrix, dtype=tc.bool),
            persistent=True,
        )

        # Same post-network layout as ClassicalPINN2 with intra_neuron=n_qubits.
        post_layers = [nn.Linear(self.n_qubits, post_neurons), nn.Tanh()]
        for _ in range(post_depth - 1):
            post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
        post_layers += [nn.Linear(post_neurons, 1)]
        self.post = nn.Sequential(*post_layers)

    def forward(self, xt):
        parameter = self.pre[0].weight
        xt = xt.to(dtype=parameter.dtype, device=parameter.device)
        input_features, x = self.Input_angular(xt)

        raw_angles = self.pre(input_features)
        if self.angle_scale is None:
            # Default used by the original exact cosine-product HPINN.
            angles = raw_angles
        else:
            # This is the same bounded-angle convention used by QuantumPINN.
            angles = self.angle_scale * tc.tanh(raw_angles)

        quantum_features = cnot_cosine_expectations(angles, self.cnot_matrix)
        output = self.post(quantum_features)
        return self.Hard_constrain(x, input_features[:, 2:3], output)

    def Input_angular(self, xt):
        """Map (x,t) to exact L-periodic spatial features and normalized time."""
        x, tt = xt[:, 0:1], xt[:, 1:2]
        input_features = tc.cat(
            [
                tc.sin(2.0 * math.pi * x / self.L),
                tc.cos(2.0 * math.pi * x / self.L),
                tt / self.Tfin,
            ],
            dim=1,
        )
        return input_features, x

    def Hard_constrain(self, x, tau, output_nn):
        """Hard-impose u(x,0)=u0(x), exactly as in the classical controls."""
        return initial_condition(x, L=self.L) + tau * output_nn

    def cosine_expressions(self, angle_symbol="theta"):
        """Return the exact formula associated with every output channel."""
        return cnot_cosine_product_expressions(
            self.cnot_matrix.detach().cpu().numpy().astype(np.uint8),
            angle_symbol=angle_symbol,
        )


# ----------------------------------------------------------------------
# Quantum-specific circuit builders (only used by QuantumPINN, kept local
# to it rather than in functions.py since nothing else needs them).
# ----------------------------------------------------------------------
FIXED_LAYERS = (
    "linear_cnot", "reverse_linear_cnot", "ring_cnot", "brickwork_cnot",
    "symmetric_cnot_ladder", "cz_ring", "ghz_fanout", "swap_network",
)

def _apply_fixed_gate_layer(qc, layer_kind):
    if QuantumCircuit is None:
        raise ImportError("qiskit is required to instantiate QuantumPINN.")

    n = qc.num_qubits
    if n < 2:
        return

    if layer_kind == "linear_cnot":
        for c in range(n - 1):
            qc.cx(c, c + 1)
    elif layer_kind == "reverse_linear_cnot":
        for c in range(n - 1, 0, -1):
            qc.cx(c, c - 1)
    elif layer_kind == "ring_cnot":
        for c in range(n - 1):
            qc.cx(c, c + 1)
        if n > 2:
            qc.cx(n - 1, 0)
    elif layer_kind == "brickwork_cnot":
        for c in range(0, n - 1, 2):
            qc.cx(c, c + 1)
        for c in range(1, n - 1, 2):
            qc.cx(c, c + 1)
    elif layer_kind == "symmetric_cnot_ladder":
        for c in range(n - 1):
            qc.cx(c, c + 1)
        for c in range(n - 1, 0, -1):
            qc.cx(c, c - 1)
    elif layer_kind == "cz_ring":
        for q in range(n - 1):
            qc.cz(q, q + 1)
        if n > 2:
            qc.cz(n - 1, 0)
    elif layer_kind == "ghz_fanout":
        for target in range(1, n):
            qc.cx(0, target)
    elif layer_kind == "swap_network":
        for q in range(n - 1):
            qc.swap(q, q + 1)
    else:
        raise ValueError(f"Unknown fixed layer '{layer_kind}'. Options: {FIXED_LAYERS}.")

def _build_fixed_circuit(n_qubits, layer_kind, repeats, axis):
    if QuantumCircuit is None or ParameterVector is None:
        raise ImportError("qiskit is required to instantiate QuantumPINN.")

    input_params = ParameterVector("inputs", n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"fixed_{layer_kind}")

    rot = {"rx": qc.rx, "ry": qc.ry, "rz": qc.rz}.get(axis.lower())
    if rot is None:
        raise ValueError("axis must be one of: 'rx', 'ry', 'rz'.")
    for q, p in enumerate(input_params):
        rot(p, q)

    for _ in range(repeats):
        _apply_fixed_gate_layer(qc, layer_kind)

    return qc, list(input_params)

def _z_observables(n_qubits):
    if SparsePauliOp is None:
        raise ImportError("qiskit is required to instantiate QuantumPINN.")

    return [
        SparsePauliOp.from_list([("I" * (n_qubits - q - 1) + "Z" + "I" * q, 1.0)])
        for q in range(n_qubits)
    ]

class QuantumPINN(PINNBase):
    """
    Hybrid HPINN: classical pre-net -> bounded rotation angles -> fixed
    (parameter-free) quantum gate layer -> classical post-net. Only the
    classical layers are trainable; the quantum circuit has zero weights.
    """

    def __init__(self, n_qubits=5, pre_neurons=10, pre_depth=1, post_neurons=10,
                 post_depth=1, angle_scale=math.pi, fixed_layer="linear_cnot",
                 fixed_layer_repeats=1, input_rotation_axis="rx", seed=42,
                 L=1.0, Tfin=0.1):
        super().__init__()

        if (
            StatevectorEstimator is None
            or EstimatorQNN is None
            or TorchConnector is None
            or ParamShiftEstimatorGradient is None
            or QuantumCircuit is None
            or ParameterVector is None
            or SparsePauliOp is None
        ):
            raise ImportError("qiskit, qiskit-machine-learning e qiskit-algorithms sao necessarios para QuantumPINN.")

        self.config = dict(
            n_qubits=n_qubits, pre_neurons=pre_neurons, pre_depth=pre_depth,
            post_neurons=post_neurons, post_depth=post_depth, angle_scale=angle_scale,
            fixed_layer=fixed_layer, fixed_layer_repeats=fixed_layer_repeats,
            input_rotation_axis=input_rotation_axis, seed=seed, L=L, Tfin=Tfin,
        )
        self.L, self.Tfin = float(L), float(Tfin)
        self.angle_scale = float(angle_scale)

        pre_layers = [nn.Linear(3, pre_neurons), nn.Tanh()]
        for _ in range(pre_depth - 1):
            pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
        pre_layers += [nn.Linear(pre_neurons, n_qubits)]
        self.pre = nn.Sequential(*pre_layers)

        self.circuit, input_params = _build_fixed_circuit(
            n_qubits, fixed_layer, fixed_layer_repeats, input_rotation_axis
        )
        observables = _z_observables(n_qubits)

        estimator = StatevectorEstimator(default_precision=0.0, seed=seed)
        gradient = ParamShiftEstimatorGradient(estimator=estimator, precision=0.0)
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=estimator,
            observables=observables,
            input_params=input_params,
            weight_params=[],
            gradient=gradient,
            input_gradients=True,
            # FIX: EstimatorQNN has its own default_precision (0.015625) that
            # overrides the estimator's default_precision=0.0 above if left
            # unset, silently making the "exact" StatevectorEstimator run
            # with shot-noise-like approximation instead. Pass it explicitly
            # to get the exact simulation the surrounding code intends.
            #default_precision=0.0,
        )
        # No initial_weights beyond the empty array: the circuit is fixed, zero trainable weights.
        self.quantum = TorchConnector(qnn, initial_weights=np.zeros(qnn.num_weights, dtype=np.float32))

        post_layers = [nn.Linear(n_qubits, post_neurons), nn.Tanh()]
        for _ in range(post_depth - 1):
            post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
        post_layers += [nn.Linear(post_neurons, 1)]
        self.post = nn.Sequential(*post_layers)

    def forward(self, xt):
        input, x = self.Input_angular(xt)
        angles = self.angle_scale * tc.tanh(self.pre(input))
        # TorchConnector/EstimatorQNN returns its own dtype regardless of the input; cast back for self.post.
        q = self.quantum(angles).to(dtype=self.post[0].weight.dtype)
        output = self.post(q)
        return self.Hard_constrain(x, input[:, 2:3], output)

    def Input_angular(self, xt):
        """Maps (x, t) to periodic (sin, cos, tau) features so the network sees x through its L-periodic image."""
        x, tt = xt[:, 0:1], xt[:, 1:2]
        input = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tt / self.Tfin],
            dim=1,
        )
        return input, x

    def Hard_constrain(self, x, tau, output_nn):
        """Hard-imposes the initial condition at tau=0 by construction."""
        return initial_condition(x, L=self.L) + tau * output_nn

    def show_circuit(self):
        return self.circuit.draw("mpl")

# ----------------------------------------------------------------------
# Generic checkpoint loading (use when the caller doesn't already know
# which PINN subclass produced the checkpoint).
# ----------------------------------------------------------------------
_MODEL_REGISTRY = {
    "ClassicalPINN": ClassicalPINN2,
    "ClassicalPINN2": ClassicalPINN2,
    "QuantumExactPINN": QuantumExactPINN,
    "QuantumPINN": QuantumPINN,
}

def load_model(path, map_location="cpu"):
    """Loads a checkpoint saved with `PINNBase.save`, picking the right
    model class automatically. Returns a model ready for retraining or
    inference."""
    checkpoint = tc.load(path, map_location=map_location, weights_only=False)
    cls = _MODEL_REGISTRY[checkpoint["class_name"]]
    model = cls(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model