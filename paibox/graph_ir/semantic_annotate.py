from enum import Enum, auto, unique

from spikingjelly.activation_based import neuron
from torch import fx, nn
from torch.fx.experimental.optimization import fuse
from torch.fx.node import Argument, Target
from torch.fx.passes.shape_prop import ShapeProp

COMP_OPS = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.Identity]
ACT_OPS = [nn.ReLU, nn.ReLU6, nn.PReLU, nn.LeakyReLU, nn.RReLU, nn.ELU]


@unique
class DataSemanticType(Enum):
    UNKNOWN = auto()
    SPIKE = auto()
    ACTIVATION = auto()
    POTENTIAL = auto()


class DataSemanticAnnotator:
    KEY = "data_semantic"

    @classmethod
    def annotate(cls, gm: fx.GraphModule) -> None:
        modules = dict(gm.named_modules())

        for node in gm.graph.nodes:
            print(f"Processing node: {node}, target: {node.target}, op: {node.op}")
            if node.op == "placeholder":
                semantic_type = DataSemanticType.UNKNOWN
            elif node.op == "call_module":
                assert isinstance(node.target, str)
                m = modules[node.target]

                if is_module_neuron(m):
                    semantic_type = DataSemanticType.SPIKE
                elif is_module_activation(m):
                    semantic_type = DataSemanticType.ACTIVATION
                elif is_module_computation(m):
                    semantic_type = DataSemanticType.POTENTIAL

            elif node.op == "call_function":
                # TODO add cat matmul logic_op
                semantic_type = DataSemanticType.UNKNOWN

            elif node.op == "output":
                input_node = node.args[0]
                assert isinstance(input_node, fx.Node)
                if (semantic_type := input_node.meta.get(cls.KEY)) is None:
                    raise RuntimeError(
                        f"Input node of {node.name} {input_node} has no semantic, cannot propagate"
                    )
            else:  # getattr, call_method
                print(f"  - Unhandled op: {node.op}")

            node.meta[cls.KEY] = semantic_type
            print(f"  - Set semantic for {node.name} to '{semantic_type.name}'")
