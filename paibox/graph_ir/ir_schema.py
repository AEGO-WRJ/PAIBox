from typing import Any

import torch
from spikingjelly.activation_based import neuron, layer
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


def strip_seq_to_ann(module: nn.Module):
    """
    Recursively replace layer.SeqToANNContainer with nn.Sequential.
    This is necessary for torch.fx tracing in step_mode='s', as SeqToANNContainer
    contains shape logic that is hard to trace symbolically.
    """
    for name, child in module.named_children():
        if isinstance(child, layer.SeqToANNContainer):
            # Replace SeqToANNContainer with nn.Sequential containing the same sub-modules
            print(
                f">>> Replaced SeqToANNContainer at '{name}' with nn.Sequential")
            new_child = nn.Sequential(*list(child.children()))
            # Update the module in the parent
            if isinstance(module, nn.Sequential):
                # For nn.Sequential, we can use accessing via index/name if setattr works,
                # but simplest is often just setattr for named_children keys which are usually '0', '1'...
                module._modules[name] = new_child
            else:
                setattr(module, name, new_child)
        else:
            strip_seq_to_ann(child)


def trace_spikingjelly_model(m: nn.Module) -> fx.GraphModule:
    strip_seq_to_ann(m)
    tracer = NeuronAsOpTracer()
    traced_graph = tracer.trace(m)
    traced = fx.GraphModule(m, traced_graph)
    traced = DropoutRemover(traced).transform()
    traced.graph.lint()
    return traced


def propagate_tensor_shape(gm: fx.GraphModule, input: torch.Tensor) -> None:
    # TODO dtype?
    shape_prop = ShapeProp(gm)
    shape_prop.propagate(input)

    for node in gm.graph.nodes:
        print(node.name, node.meta["tensor_meta"].dtype,
              node.meta["tensor_meta"].shape)
