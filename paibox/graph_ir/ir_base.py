from dataclasses import dataclass
from typing import Any

from paicorelib import (
    RM,
    AddPotentialMode,
    CSCAccelerateMode,
    FoldType,
    InputSignMode,
    InputWidthFormat,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OutputSignMode,
    OutputType,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
    WeightSignMode,
    WeightWidth,
    ZeroOutputMode,
)
from torch import fx, nn

from .suppport_ops import OpLoc


@dataclass
class OfflineCoreMetaAttr:
    snn_ann: SNNMode
    max_pooling: PoolingMode
    add_potential: AddPotentialMode
    zero_output: ZeroOutputMode
    input_sign: InputSignMode
    input_width: InputWidthFormat
    output_sign: OutputSignMode
    weight_sign: WeightSignMode
    weight_width: WeightWidth
    csc_accelerate: CSCAccelerateMode
    thread_number: int = 0
    tick_start: int = 1
    tick_duration: int = 9999
    tick_initial: int = 0


@dataclass
class OfflineNeuronMetaAttr:
    output_type: OutputType
    fold_type: FoldType
    neuron_type: NeuronType
    reset_mode: RM
    thres_neg_mode: ThresholdNegMode
    thres_pos_mode: ThresholdPosMode
    thres_neg: int
    thres_pos: int
    lateral_inhibition: LateralInhibitionMode
    leak_multi_comparison_order: LeakMultiComparisonOrder
    leak_multi_input: LeakMultiInputMode
    leak_multi_mode: LeakMultiMode
    leak_add_mode: LeakAddMode
    leak_tau: int
    weight_compress_type: WeightCompressType
    v_init: int


@dataclass
class CoreOpNodeMetaAttr:
    pass


@dataclass
class OfflineCoreOpNodeMetaAttr(CoreOpNodeMetaAttr):
    core: OfflineCoreMetaAttr
    neuron: OfflineNeuronMetaAttr


class PAIIR(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class ViewIR(PAIIR):
    pass


class InputIR(ViewIR):
    """
    Represents a virtual input node for the chip.
    """


class OutputIR(ViewIR):
    """
    Represents a virtual output node for the chip.
    """


class CoreOpNode(PAIIR):
    _op_loc: OpLoc

    def __init__(self, comp_op: fx.Node, act_op: fx.Node) -> None:
        name = comp_op.name + "_" + act_op.name + "_coreop"
        super().__init__(name)
        self._comp_op = comp_op
        self._act_op = act_op
        # We recorded the shape in meta["tensor_meta"]
        self.output_shape = act_op.meta["tensor_meta"].shape

    def forward(self, x):
        return x

    def opnode_eq(self, other: Any) -> bool:
        return self.meta_attr == other.meta_attr

    @property
    def all_input_nodes(self) -> list[fx.Node]:
        return self._comp_op.all_input_nodes


class OfflineCoreOpNode(CoreOpNode):
    _op_loc: OpLoc = OpLoc.OFFLINE_CORE

    def __init__(self, comp_op: fx.Node, act_op: fx.Node) -> None:
        super().__init__(comp_op, act_op)


# TODO 对于Add算子，如何给其构建IR？

# class CPUOpNode(PAIIR):
#     def __init__(self) -> None:
#         super().__init__()
