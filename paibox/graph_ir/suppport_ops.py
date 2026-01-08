from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from spikingjelly.activation_based import neuron
from torch import nn


class OpLoc(Enum):
    OFFLINE_CORE = auto()
    ONLINE_CORE = auto()
    CPU = auto()


OpType = type[nn.Module] | Callable | str


@dataclass
class OpRegisterRule:
    op: OpType
    op_loc: OpLoc
    op_call: str


@dataclass
class OpModuleRegisterRule(OpRegisterRule):
    op: type[nn.Module]
    op_loc: OpLoc
    op_call: str = "call_module"


@dataclass
class OpFunctionRegisterRule(OpRegisterRule):
    op: Callable
    op_loc: OpLoc
    op_call: str = "call_function"


@dataclass
class OpMethodRegisterRule(OpRegisterRule):
    op: str
    op_loc: OpLoc
    op_call: str = "call_method"


class SupportOpRegistry:
    def __init__(self) -> None:
        self.neuron_ops = []
        self.activation_ops = []
        self.computation_ops = []

    def register_neuron_op(self, rule: OpRegisterRule) -> None:
        self.neuron_ops.append(rule)


SUPPPORT_NEURON_OPS = [neuron.LIFNode, neuron.IFNode]
USE_CROSSBAR_OPS = [nn.Conv1d, nn.Conv2d, nn.Linear]
COMP_OPS = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.Identity]

# https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
ACTIVATION_OPS = [
    nn.PReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.RReLU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.SiLU,
    nn.GLU,
]


SUPPORT_OP_REGISTRY = SupportOpRegistry()
SUPPORT_OP_REGISTRY.register_neuron_op(
    OpModuleRegisterRule(neuron.IFNode, OpLoc.OFFLINE_CORE)
)


def is_module_neuron(m: nn.Module, only_support: bool = True) -> bool:
    if only_support:
        return isinstance(m, tuple(SUPPPORT_NEURON_OPS))
    else:
        return isinstance(m, neuron.BaseNode)


def is_module_activation(m: nn.Module) -> bool:
    return any(isinstance(m, op) for op in ACTIVATION_OPS)


def is_module_computation(m: nn.Module) -> bool:
    return any(isinstance(m, op) for op in COMP_OPS)
