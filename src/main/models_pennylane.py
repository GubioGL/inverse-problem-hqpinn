import math

import pennylane as qml
import torch as tc
import torch.nn as nn

from .functions import initial_condition
from .models import FIXED_LAYERS, PINNBase

# ----------------------------------------------------------------------
# Fixed (parameter-free) entangling layer, expressed with PennyLane gates.
# Mirrors `_apply_fixed_gate_layer` in models.py (qiskit version) gate-for-gate.
# ----------------------------------------------------------------------
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

class PennyLaneFixedQuantumLayer(nn.Module):
    """
    Parameter-free quantum feature layer built on PennyLane's `default.qubit`
    simulator with `diff_method="backprop"`: the whole statevector simulation
    is executed as ordinary Torch tensor ops, so autograd differentiates it
    natively (fast forward, and cheap 1st/2nd-order backward -- no
    parameter-shift circuit re-executions, unlike the Qiskit `TorchConnector`
    path in `models.py`). The full batch is passed to the QNode at once via
    PennyLane's parameter broadcasting: no per-sample Python loop.
    """

    def __init__(self, n_qubits, fixed_layer="linear_cnot", fixed_layer_repeats=1,
                 input_rotation_axis="rx"):
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

    def draw(self):
        dummy = tc.zeros(1, self.n_qubits)
        return qml.draw_mpl(self.circuit)(dummy)

class QuantumPINNPennyLane(PINNBase):
    """Hybrid HPINN identical in structure to `models.QuantumPINN` (classical
    pre-net -> bounded rotation angles -> fixed quantum layer -> classical
    post-net), but with the quantum layer implemented in PennyLane instead of
    Qiskit so the PDE derivatives can be obtained with plain `torch.autograd`
    end-to-end, cheaply."""

    def __init__(self, n_qubits=5, pre_neurons=10, pre_depth=1, post_neurons=10,
                 post_depth=1, angle_scale=math.pi, fixed_layer="linear_cnot",
                 fixed_layer_repeats=1, input_rotation_axis="rx", seed=42,
                 L=1.0, Tfin=0.1, estrutura=0):
        super().__init__()

        self.config = dict(
            n_qubits=n_qubits, pre_neurons=pre_neurons, pre_depth=pre_depth,
            post_neurons=post_neurons, post_depth=post_depth, angle_scale=angle_scale,
            fixed_layer=fixed_layer, fixed_layer_repeats=fixed_layer_repeats,
            input_rotation_axis=input_rotation_axis, seed=seed, L=L, Tfin=Tfin,
            estrutura=estrutura
        )
        self.estrutura = int(estrutura)
        self.L, self.Tfin = float(L), float(Tfin)
        self.angle_scale = float(angle_scale)
        
        if self.estrutura == 0:
            # quantum -> pre -> post
            # O quantum recebe o input bruto (sin, cos, tau) direto como angulos,
            # entao n_qubits precisa ser 3 para casar com as 3 features de entrada.
            if n_qubits != 3:
                raise ValueError(
                    "estrutura=0 alimenta o quantum diretamente com o input bruto "
                    "(sin, cos, tau), logo n_qubits deve ser 3."
                )

            self.quantum = PennyLaneFixedQuantumLayer(
                n_qubits=n_qubits, fixed_layer=fixed_layer,
                fixed_layer_repeats=fixed_layer_repeats,
                input_rotation_axis=input_rotation_axis,
            )

            pre_layers = [nn.Linear(n_qubits, pre_neurons), nn.Tanh()]
            for _ in range(pre_depth - 1):
                pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
            pre_layers += [nn.Linear(pre_neurons, n_qubits)]
            self.pre = nn.Sequential(*pre_layers)

            post_layers = [nn.Linear(n_qubits, post_neurons), nn.Tanh()]
            for _ in range(post_depth - 1):
                post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
            post_layers += [nn.Linear(post_neurons, 1)]
            self.post = nn.Sequential(*post_layers)

        elif self.estrutura == 1:
            pre_layers = [nn.Linear(3, pre_neurons), nn.Tanh()]
            for _ in range(pre_depth - 1):
                pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
            pre_layers += [nn.Linear(pre_neurons, n_qubits)]
            self.pre = nn.Sequential(*pre_layers)

            self.quantum = PennyLaneFixedQuantumLayer(
                n_qubits=n_qubits, fixed_layer=fixed_layer,
                fixed_layer_repeats=fixed_layer_repeats,
                input_rotation_axis=input_rotation_axis,
            )

            post_layers = [nn.Linear(n_qubits, post_neurons), nn.Tanh()]
            for _ in range(post_depth - 1):
                post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
            post_layers += [nn.Linear(post_neurons, 1)]
            self.post = nn.Sequential(*post_layers)

        elif self.estrutura == 2:
            # pre -> post -> quantum
            # post agora devolve n_qubits valores (em vez de 1), que viram os
            # angulos do quantum; a saida do quantum e reduzida a um escalar
            # pela media dos n_qubits expvals.
            pre_layers = [nn.Linear(3, pre_neurons), nn.Tanh()]
            for _ in range(pre_depth - 1):
                pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
            pre_layers += [nn.Linear(pre_neurons, n_qubits)]
            self.pre = nn.Sequential(*pre_layers)

            post_layers = [nn.Linear(n_qubits, post_neurons), nn.Tanh()]
            for _ in range(post_depth - 1):
                post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
            post_layers += [nn.Linear(post_neurons, n_qubits)]
            self.post = nn.Sequential(*post_layers)

            self.quantum = PennyLaneFixedQuantumLayer(
                n_qubits=n_qubits, fixed_layer=fixed_layer,
                fixed_layer_repeats=fixed_layer_repeats,
                input_rotation_axis=input_rotation_axis,
            )

    def forward(self, xt):
        input, x = self.Input_angular(xt)
        
        if self.estrutura == 0:
            # Quantum
            angles = self.angle_scale * tc.tanh(input)
            q = self.quantum(angles).to(dtype=self.pre[0].weight.dtype)
            # Classical
            pre_out = self.pre(q)
            # Classical
            output = self.post(pre_out)
            return self.Hard_constrain(x, input[:, 2:3], output)
        elif self.estrutura == 1:
            # Classical
            angles = self.angle_scale * tc.tanh(self.pre(input))
            # Quantum
            q = self.quantum(angles).to(dtype=self.post[0].weight.dtype)
            # Classical
            output = self.post(q)
            return self.Hard_constrain(x, input[:, 2:3], output)
        elif self.estrutura == 2:
            # Classical
            pre_out = self.pre(input)
            # Classical
            post_out = self.post(pre_out)
            # Quantum
            angles = self.angle_scale * tc.tanh(post_out)
            q = self.quantum(angles)
            output = q.mean(dim=1, keepdim=True)
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
        return self.quantum.draw()
