import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Tracer, GraphModule
from spikingjelly.activation_based import neuron, layer
from paibox.graph_ir.ir_schema import trace_spikingjelly_model, propagate_tensor_shape
from paibox.graph_ir.semantic_annotate import DataSemanticAnnotator, DataSemanticType
from paibox.graph_ir.pattern import ConvNeuronPattern, ConvActPattern, PAIIRPatternMatcher


class CSNN(nn.Module):
    def __init__(self, T: int = 4):
        """
        Convolutional Spiking Neural Network for MNIST classification.
        Structure:
            1. Feature Extractor: Conv2d -> BatchNorm -> LIF
            2. Residual Block: Conv2d -> BatchNorm -> (Identity) -> LIF
            3. Classification: MaxPool -> Flatten -> Linear -> PLIF

        Args:
            T (int): Time steps (simulation duration). Default: 4.
        """
        super().__init__()
        self.T = T

        # 1. Feature Extractor
        # Input: [T, N, 1, 28, 28]
        # Output: 16 channels, same spatial size (padding=1)
        self.feature_extractor = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            ),
            neuron.LIFNode(tau=2.0, detach_reset=True,
                           step_mode='m', v_threshold=1.0)
        )

        # 2. Residual Block
        # Keeps 16 channels, same spatial size
        self.res_conv = layer.SeqToANNContainer(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.res_lif = neuron.LIFNode(
            tau=2.0, step_mode='m', v_threshold=1.0)

        # 3. Output Layer
        # MaxPool 2x2 -> 28x28 becomes 14x14
        self.pool = layer.SeqToANNContainer(nn.MaxPool2d(2, 2))
        self.flatten = layer.SeqToANNContainer(nn.Flatten())

        # Fully Connected: 16 channels * 14 * 14
        self.fc = layer.SeqToANNContainer(nn.Linear(16 * 14 * 14, 10))

        # Output Neuron: PLIF (Parametric LIF) with learnable parameters
        self.output_lif = neuron.ParametricLIFNode(step_mode='m')

    def forward(self, x):
        # x shape: [T, N, C, H, W] in 'm' mode

        # Part 1: Feature Extraction
        x = self.feature_extractor(x)

        # Part 2: Residual Connection
        out = self.res_conv(x)
        out = out + x   # Element-wise add
        x = self.res_lif(out)

        # Part 3: Classification
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.output_lif(x)

        return x


def test_semantic_annotate_basic():

    net = CSNN()

    gm = trace_spikingjelly_model(net)

    example_input = torch.randn(1, 1, 28, 28)
    propagate_tensor_shape(gm, example_input)

    print("Graph before annotation:")
    gm.graph.print_tabular()

    DataSemanticAnnotator.annotate(gm)

    print("\nGraph after annotation:")
    for node in gm.graph.nodes:
        semantic = node.meta.get(DataSemanticAnnotator.KEY, "Metadata Missing")
        print(
            f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Semantic: {semantic}")

    # Verification
    nodes = {node.name: node for node in gm.graph.nodes}

    print("===========================================================")
    print("\n进行合并优化")
    matcher = PAIIRPatternMatcher()
    matcher.add_pattern(ConvNeuronPattern())
    matcher.add_pattern(ConvActPattern())

    gm2 = matcher(gm)
    print(gm2.graph.print_tabular())

    print("===========================================================")
    print("\n进行语义标注")

    DataSemanticAnnotator.annotate(gm2)

    print("\nGraph after annotation:")
    for node in gm2.graph.nodes:
        semantic = node.meta.get(DataSemanticAnnotator.KEY, "Metadata Missing")
        print(
            f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Semantic: {semantic}")

    # Verification
    nodes = {node.name: node for node in gm2.graph.nodes}


if __name__ == "__main__":
    test_semantic_annotate_basic()
