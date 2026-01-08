from typing import Any

import torch
from spikingjelly.activation_based import neuron
from torch import fx, nn
from torch.fx.experimental.optimization import fuse
from torch.fx.node import Argument, Target
from torch.fx.passes.shape_prop import ShapeProp


class NeuronAsOpTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Treat neuron `BaseNode` as leaf module
        if isinstance(m, neuron.BaseNode):
            return True

        return super().is_leaf_module(m, module_qualified_name)


class DropoutRemover(fx.Transformer):
    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        if isinstance(self.submodules[target], nn.Dropout):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)


def trace_spikingjelly_model(m: nn.Module) -> fx.GraphModule:
    tracer = NeuronAsOpTracer()
    traced_graph = tracer.trace(m)
    traced = fx.GraphModule(m, traced_graph)
    traced.graph.lint()
    return traced


def remove_dropout_and_fuse_conv_bn(m: nn.Module) -> fx.GraphModule:
    gm = trace_spikingjelly_model(m)
    m = DropoutRemover(gm).transform()
    # m.graph.print_tabular()
    m = fuse(m, inplace=True, no_trace=True)
    return m  # type: ignore


def propagate_tensor_shape(gm: fx.GraphModule, input: torch.Tensor) -> None:
    # TODO dtype?
    shape_prop = ShapeProp(gm)
    shape_prop.propagate(input)

    for node in gm.graph.nodes:
        print(node.name, node.meta["tensor_meta"].dtype,
              node.meta["tensor_meta"].shape)
