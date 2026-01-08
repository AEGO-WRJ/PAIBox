import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from spikingjelly.activation_based import neuron
from torch import fx, nn
from torch.fx.experimental.optimization import (
    matches_module_pattern as matches_module_pattern2,
)
from torch.nn import functional as F

from .ir_base import OfflineCoreOpNode, PAIIR
from .suppport_ops import COMP_OPS, SUPPPORT_NEURON_OPS, ACTIVATION_OPS, USE_CROSSBAR_OPS


def _parent_name(target: str) -> tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_with_ir(node: fx.Node, modules: dict[str, Any], ir: PAIIR) -> None:
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)

    new_name = ir.name
    new_target = f"{parent_name}.{new_name}" if parent_name else new_name

    modules[new_target] = ir
    setattr(modules[parent_name], new_name, ir)
    node.target = new_target
    node.name = new_name


class Pattern(ABC):
    """
    Base class for types of patterns.
    """

    @abstractmethod
    def match(self, node: fx.Node, modules: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def rewrite(self, node: fx.Node, modules: dict[str, Any], graph: fx.Graph) -> None:
        raise NotImplementedError


class ConvActPattern(Pattern):
    patterns = [(c, a) for c in COMP_OPS for a in ACTIVATION_OPS]

    def match(self, node: fx.Node, modules: dict[str, nn.Module]) -> bool:
        for p in self.patterns:
            if matches_module_pattern2(p, node, modules):
                if isinstance(node.args[0], fx.Node) and len(node.args[0].users) > 1:
                    continue
                else:
                    return True

        return False

    def rewrite(
        self, node: fx.Node, modules: dict[str, nn.Module], graph: fx.Graph
    ) -> None:
        assert isinstance(node.args[0], fx.Node)

        # Capture names before modification
        input_node = node.args[0]
        input_name = input_node.name
        node_name = node.name

        # Prepare modules
        input_target = input_node.target
        node_target = node.target
        input_module = modules.get(input_target) if isinstance(
            input_target, str) else None
        node_module = modules.get(node_target) if isinstance(
            node_target, str) else None

        # TODO: Check if OfflineCoreOpNode constructor signature matches
        new_node = OfflineCoreOpNode(
            input_node, node, input_module, node_module)
        new_node_name = new_node.name

        replace_node_with_ir(input_node, modules, new_node)
        node.replace_all_uses_with(input_node)
        graph.erase_node(node)

        print(
            f"Replaced {input_name} + {node_name} with new node: {new_node_name}"
        )


class ConvNeuronPattern(Pattern):
    patterns = [(c, n) for c in COMP_OPS for n in SUPPPORT_NEURON_OPS]

    def match(self, node: fx.Node, modules: dict[str, nn.Module]) -> bool:
        for p in self.patterns:
            if matches_module_pattern2(p, node, modules):
                if isinstance(node.args[0], fx.Node) and len(node.args[0].users) > 1:
                    continue
                else:
                    return True

        return False

    def rewrite(
        self, node: fx.Node, modules: dict[str, nn.Module], graph: fx.Graph
    ) -> None:
        assert isinstance(node.args[0], fx.Node)

        # Check if it is a neuron op to verify pattern correctness?
        # Actually pattern matching guarantees node is Neuron and args[0] is Conv

        # Capture names before graph modification
        input_node = node.args[0]
        input_name = input_node.name
        node_name = node.name

        # Prepare modules
        input_target = input_node.target
        node_target = node.target
        input_module = modules.get(input_target) if isinstance(
            input_target, str) else None
        node_module = modules.get(node_target) if isinstance(
            node_target, str) else None

        # Create the fused node
        new_node = OfflineCoreOpNode(
            input_node, node, input_module, node_module)
        new_node_name = new_node.name

        replace_node_with_ir(input_node, modules, new_node)
        node.replace_all_uses_with(input_node)
        graph.erase_node(node)

        print(
            f"Replaced {input_name} + {node_name} with new node: {new_node_name}"
        )


def this_before_that_pattern_constraint(
    this: Pattern, that: Pattern
) -> Callable[[Pattern, Pattern], bool]:
    def depends_on(a: Pattern, b: Pattern) -> bool:
        return a != that or b != this

    return depends_on


def _validate_pass_schedule_constraint(
    constraint: Callable[[Pattern, Pattern], bool], passes: list[Pattern]
) -> None:
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1:]):
            if constraint(a, b):
                continue
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )


class PAIIRPatternMatcher:
    _validated: bool = False

    def __init__(self) -> None:
        self.patterns: list[Pattern] = []
        self.constraints = []

    def add_pattern(self, _pattern: Pattern) -> None:
        self.patterns.append(_pattern)
        self._validated = False

    def add_constraint(self, constraint: Callable) -> None:
        self.constraints.append(constraint)
        self._validated = False

    def remove_pattern(self, *_patterns: type[Pattern]) -> None:
        if len(_patterns) == 0:
            return

        patterns_left = [
            ps for ps in self.patterns if not isinstance(ps, _patterns)]
        self.patterns = patterns_left
        self._validated = False

    def validate(self) -> None:
        """
        Validates that current pass schedule defined by `self.patterns` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return

        for constr in self.constraints:
            _validate_pass_schedule_constraint(constr, self.patterns)

        self._validated = True

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        self.validate()

        modules = dict(gm.named_modules())
        new_graph = copy.deepcopy(gm.graph)

        for pattern in self.patterns:
            for node in new_graph.nodes:
                if pattern.match(node, modules):
                    pattern.rewrite(node, modules, new_graph)

            # Apply function-based patterns
            # for func_pattern in self.func_patterns:
            #     func_pattern(node, modules, new_graph)

        return fx.GraphModule(gm, new_graph)
