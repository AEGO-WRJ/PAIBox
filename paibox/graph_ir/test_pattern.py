from spikingjelly.activation_based import neuron
from torch import fx, nn

from paibox.graph_ir.ir_schema import trace_spikingjelly_model
from paibox.graph_ir.pattern import ConvNeuronPattern, PAIIRPatternMatcher


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
    print(gm.graph.print_tabular())

    matcher = PAIIRPatternMatcher()
    matcher.add_pattern(ConvNeuronPattern())

    gm2 = matcher(gm)

    print(gm2.graph.print_tabular())
