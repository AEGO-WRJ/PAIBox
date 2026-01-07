from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.functional import reset_net
import torch
from torch import fx, nn
from torch.fx import replace_pattern
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

from paibox.graph_ir.ir_schema import (
    NeuronAsOpTracer,
    remove_dropout_and_fuse_conv_bn,
    propagate_tensor_shape,
)
from paibox.graph_ir.semantic_annotate import DataSemanticAnnotator


class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = neuron.LIFNode(v_threshold=1.0, tau=2.0)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.if1 = neuron.IFNode(v_threshold=2.0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.lif1(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x1 = self.if1(self.conv2(x))
        x2 = self.if1(self.conv2(x))
        return x1 + x2


class Pattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3)
        self.lif1 = neuron.LIFNode(v_threshold=1.0, tau=2.0)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.lif1(self.conv1(x))
        return x


def pattern(x):
    m = Pattern()
    tracer = NeuronAsOpTracer()
    traced_graph = tracer.trace(m)
    traced = fx.GraphModule(m, traced_graph)
    traced.graph.lint()

    return traced.forward(x)


class Replacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


def test_ir_converter():
    model = SimpleSNN()
    reset_net(model)
    model.eval()

    gm = remove_dropout_and_fuse_conv_bn(model)
    assert isinstance(gm, fx.GraphModule)
    gm.graph.print_tabular()

    # Propagate shape
    example_input = torch.randn((4, 64, 64))
    propagate_tensor_shape(gm, example_input)

    # Replace patterns
    # Not fit our requirements
    # replace_pattern(gm, pattern, Replacement())


def test_gm_data_semantic_annotate():
    model = SimpleSNN()
    reset_net(model)
    model.eval()

    gm = remove_dropout_and_fuse_conv_bn(model)
    gm.graph.print_tabular()

    # DataSemanticAnnotator.annotate(gm)
