from spikingjelly.activation_based import neuron
from torch import fx, nn
import torch

from paibox.graph_ir.ir_schema import trace_spikingjelly_model, propagate_tensor_shape
from paibox.graph_ir.pattern import ConvNeuronPattern, ConvActPattern, PAIIRPatternMatcher


def test_conv_act_fusion_pattern():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.lif = neuron.LIFNode()
            self.prelu = nn.PReLU()
            self.bn = nn.BatchNorm2d(16)  # 添加一个 BN 来区分

        def forward(self, x):
            x1 = self.conv(x)
            x1 = self.lif(x1)  # Conv -> ReLU
            x2 = self.conv(x)
            x2 = self.prelu(x2)  # Conv -> PReLU
            x3 = self.conv(x)
            x3 = self.bn(x3)  # Conv -> BN (不会被匹配)
            return x1 + x2 + x3

    m = M()
    gm = trace_spikingjelly_model(m)

    # Propagate shape is required for fusion
    example_input = torch.randn(1, 3, 32, 32)
    propagate_tensor_shape(gm, example_input)

    print(gm.graph.print_tabular())

    matcher = PAIIRPatternMatcher()
    matcher.add_pattern(ConvNeuronPattern())
    matcher.add_pattern(ConvActPattern())

    gm2 = matcher(gm)

    print(gm2.graph.print_tabular())
    propagate_tensor_shape(gm2, example_input)


if __name__ == "__main__":
    test_conv_act_fusion_pattern()
