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
from .suppport_ops import SUPPPORT_NEURON_OPS, ACTIVATION_OPS, USE_CROSSBAR_OPS


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


class TwoOpFusionPattern(Pattern):
    pass


class CompActPattern(TwoOpFusionPattern):
    patterns = [(c, a) for c in COMP_OPS for a in ACT_OPS]

    def match(self, node: fx.Node, modules: dict[str, nn.Module]) -> bool:
        for p in self.patterns:
            if matches_module_pattern2(p, node, modules):
                if len(node.args[0].users) > 1:
                    continue
                else:
                    return True

        return False

    def rewrite(
        self, node: fx.Node, modules: dict[str, nn.Module], graph: fx.Graph
    ) -> None:
        assert isinstance(node.args[0], fx.Node)
        new_node = OfflineCoreOpNode(node.args[0], node, dict())

        replace_node_with_ir(node.args[0], modules, new_node)
        node.replace_all_uses_with(node.args[0])
        graph.erase_node(node)

        print(
            f"Replaced {node.args[0].name} + {node.name} with new node: {new_node.name}"
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
        for j, b in enumerate(passes[i + 1 :]):
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

        patterns_left = [ps for ps in self.patterns if not isinstance(ps, _patterns)]
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
