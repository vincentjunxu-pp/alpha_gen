from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, TypeAlias

from .program import (
    BinaryNode,
    ConstNode,
    FieldNode,
    GateNode,
    Program,
    TreeNode,
    UnaryNode,
    node_depth,
    node_output_type,
    node_size,
    validate_program,
)
from .registry import OperatorSpec, list_operators


NodePath: TypeAlias = tuple[str, ...]
MutationKind: TypeAlias = Literal["subtree", "operator", "terminal"]


@dataclass(frozen=True)
class ProgramGeneratorConfig:
    max_depth: int = 5
    max_size: int = 64
    const_probability: float = 0.05
    const_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    terminal_probability: float = 0.20
    gate_probability: float = 0.15
    subtree_mutation_probability: float = 0.55
    operator_mutation_probability: float = 0.30
    terminal_mutation_probability: float = 0.15
    max_attempts: int = 100

    def __post_init__(self) -> None:
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least 2")
        if self.max_size < 1:
            raise ValueError("max_size must be positive")
        if not 0 <= self.const_probability <= 1:
            raise ValueError("const_probability must be in [0, 1]")
        if not 0 <= self.terminal_probability <= 1:
            raise ValueError("terminal_probability must be in [0, 1]")
        if not 0 <= self.gate_probability <= 1:
            raise ValueError("gate_probability must be in [0, 1]")
        if not self.const_values:
            raise ValueError("const_values must not be empty")
        probs = (
            self.subtree_mutation_probability,
            self.operator_mutation_probability,
            self.terminal_mutation_probability,
        )
        if any(prob < 0 for prob in probs) or sum(probs) <= 0:
            raise ValueError("mutation probabilities must be non-negative and have positive sum")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


def _rng(seed: int | random.Random | None) -> random.Random:
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _choice(rng: random.Random, values: Sequence[object]):
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[rng.randrange(len(values))]


def _weighted_choice(rng: random.Random, items: Sequence[tuple[object, float]]):
    valid = [(item, float(weight)) for item, weight in items if weight > 0]
    if not valid:
        raise ValueError("weighted choice needs at least one positive weight")
    total = sum(weight for _item, weight in valid)
    threshold = rng.random() * total
    running = 0.0
    for item, weight in valid:
        running += weight
        if running >= threshold:
            return item
    return valid[-1][0]


def _numeric_unary_ops() -> tuple[OperatorSpec, ...]:
    return tuple(
        spec
        for spec in list_operators(output_type="numeric")
        if spec.arity == 1 and spec.category != "gate"
    )


def _numeric_binary_ops() -> tuple[OperatorSpec, ...]:
    return tuple(
        spec
        for spec in list_operators(output_type="numeric")
        if spec.arity == 2 and spec.category != "gate"
    )


def _mask_unary_ops() -> tuple[OperatorSpec, ...]:
    return tuple(spec for spec in list_operators(output_type="mask") if spec.arity == 1)


def _gate_ops() -> tuple[OperatorSpec, ...]:
    return tuple(spec for spec in list_operators(category="gate"))


def _random_terminal(
    fields: Sequence[str],
    rng: random.Random,
    config: ProgramGeneratorConfig,
) -> TreeNode:
    if not fields:
        raise ValueError("fields must contain at least one field")
    if rng.random() < config.const_probability:
        return ConstNode(float(_choice(rng, config.const_values)))
    return FieldNode(str(_choice(rng, fields)))


def _random_mask_node(
    fields: Sequence[str],
    rng: random.Random,
    config: ProgramGeneratorConfig,
    max_depth: int,
) -> TreeNode:
    if max_depth < 2:
        raise ValueError("mask node requires max_depth >= 2")
    op = _choice(rng, _mask_unary_ops())
    return UnaryNode(op.name, _random_numeric_node(fields, rng, config, max_depth - 1))


def _random_numeric_node(
    fields: Sequence[str],
    rng: random.Random,
    config: ProgramGeneratorConfig,
    max_depth: int,
) -> TreeNode:
    if max_depth <= 1 or rng.random() < config.terminal_probability:
        return _random_terminal(fields, rng, config)

    choices: list[tuple[str, float]] = []
    if max_depth >= 2:
        choices.extend([("unary", 1.0), ("binary", 0.85)])
    if max_depth >= 3:
        choices.append(("gate", config.gate_probability))
    kind = _weighted_choice(rng, choices)

    if kind == "unary":
        op = _choice(rng, _numeric_unary_ops())
        return UnaryNode(op.name, _random_numeric_node(fields, rng, config, max_depth - 1))
    if kind == "binary":
        op = _choice(rng, _numeric_binary_ops())
        return BinaryNode(
            op.name,
            _random_numeric_node(fields, rng, config, max_depth - 1),
            _random_numeric_node(fields, rng, config, max_depth - 1),
        )
    if kind == "gate":
        op = _choice(rng, _gate_ops())
        return GateNode(
            op.name,
            signal=_random_numeric_node(fields, rng, config, max_depth - 1),
            mask=_random_mask_node(fields, rng, config, max_depth - 1),
        )
    raise ValueError(f"unknown generation kind: {kind!r}")


def random_program(
    fields: Sequence[str],
    *,
    config: ProgramGeneratorConfig | None = None,
    random_state: int | random.Random | None = None,
) -> Program:
    config = config or ProgramGeneratorConfig()
    rng = _rng(random_state)
    field_values = tuple(dict.fromkeys(str(field) for field in fields))
    if not field_values:
        raise ValueError("fields must contain at least one field")

    last_errors: list[str] = []
    for _attempt in range(config.max_attempts):
        program = Program(_random_numeric_node(field_values, rng, config, config.max_depth))
        errors = validate_program(
            program,
            available_fields=field_values,
            max_depth=config.max_depth,
            max_size=config.max_size,
        )
        if not errors:
            return program
        last_errors = errors
    raise RuntimeError("failed to generate a valid program: " + "; ".join(last_errors))


def iter_paths(node: TreeNode, path: NodePath = ()) -> Iterable[tuple[NodePath, TreeNode]]:
    yield path, node
    if isinstance(node, UnaryNode):
        yield from iter_paths(node.child, (*path, "child"))
    elif isinstance(node, BinaryNode):
        yield from iter_paths(node.left, (*path, "left"))
        yield from iter_paths(node.right, (*path, "right"))
    elif isinstance(node, GateNode):
        yield from iter_paths(node.signal, (*path, "signal"))
        yield from iter_paths(node.mask, (*path, "mask"))


def get_subtree(node: TreeNode, path: NodePath) -> TreeNode:
    current = node
    for part in path:
        if isinstance(current, UnaryNode) and part == "child":
            current = current.child
        elif isinstance(current, BinaryNode) and part == "left":
            current = current.left
        elif isinstance(current, BinaryNode) and part == "right":
            current = current.right
        elif isinstance(current, GateNode) and part == "signal":
            current = current.signal
        elif isinstance(current, GateNode) and part == "mask":
            current = current.mask
        else:
            raise ValueError(f"invalid path component {part!r} for {type(current).__name__}")
    return current


def replace_subtree(node: TreeNode, path: NodePath, replacement: TreeNode) -> TreeNode:
    if not path:
        return replacement
    head, *tail = path
    child_path = tuple(tail)
    if isinstance(node, UnaryNode) and head == "child":
        return UnaryNode(node.op, replace_subtree(node.child, child_path, replacement))
    if isinstance(node, BinaryNode) and head == "left":
        return BinaryNode(node.op, replace_subtree(node.left, child_path, replacement), node.right)
    if isinstance(node, BinaryNode) and head == "right":
        return BinaryNode(node.op, node.left, replace_subtree(node.right, child_path, replacement))
    if isinstance(node, GateNode) and head == "signal":
        return GateNode(node.op, replace_subtree(node.signal, child_path, replacement), node.mask)
    if isinstance(node, GateNode) and head == "mask":
        return GateNode(node.op, node.signal, replace_subtree(node.mask, child_path, replacement))
    raise ValueError(f"invalid path component {head!r} for {type(node).__name__}")


def _new_subtree(
    target_type: str,
    fields: Sequence[str],
    rng: random.Random,
    config: ProgramGeneratorConfig,
    max_depth: int,
) -> TreeNode:
    if target_type == "numeric":
        return _random_numeric_node(fields, rng, config, max_depth)
    if target_type == "mask":
        return _random_mask_node(fields, rng, config, max_depth)
    raise ValueError(f"unsupported target type: {target_type!r}")


def _mutate_operator(node: TreeNode, rng: random.Random) -> TreeNode:
    if isinstance(node, UnaryNode):
        current_type = node_output_type(node)
        candidates = [
            spec for spec in list_operators(output_type=current_type)
            if spec.arity == 1 and spec.name != node.op
        ]
        if not candidates:
            return node
        return UnaryNode(_choice(rng, candidates).name, node.child)
    if isinstance(node, BinaryNode):
        candidates = [
            spec for spec in _numeric_binary_ops()
            if spec.name != node.op
        ]
        if not candidates:
            return node
        return BinaryNode(_choice(rng, candidates).name, node.left, node.right)
    if isinstance(node, GateNode):
        candidates = [spec for spec in _gate_ops() if spec.name != node.op]
        if not candidates:
            return node
        return GateNode(_choice(rng, candidates).name, node.signal, node.mask)
    return node


def _mutate_terminal(
    node: TreeNode,
    fields: Sequence[str],
    rng: random.Random,
    config: ProgramGeneratorConfig,
) -> TreeNode:
    if isinstance(node, (FieldNode, ConstNode)):
        return _random_terminal(fields, rng, config)
    return node


def _mutation_kind(rng: random.Random, config: ProgramGeneratorConfig) -> MutationKind:
    return _weighted_choice(
        rng,
        [
            ("subtree", config.subtree_mutation_probability),
            ("operator", config.operator_mutation_probability),
            ("terminal", config.terminal_mutation_probability),
        ],
    )


def mutate_program(
    program: Program,
    fields: Sequence[str],
    *,
    config: ProgramGeneratorConfig | None = None,
    random_state: int | random.Random | None = None,
) -> Program:
    config = config or ProgramGeneratorConfig()
    rng = _rng(random_state)
    field_values = tuple(dict.fromkeys(str(field) for field in fields))
    paths = tuple(iter_paths(program.root))

    for _attempt in range(config.max_attempts):
        kind = _mutation_kind(rng, config)
        path, subtree = _choice(rng, paths)

        if kind == "operator":
            replacement = _mutate_operator(subtree, rng)
        elif kind == "terminal":
            terminal_paths = tuple((p, n) for p, n in paths if isinstance(n, (FieldNode, ConstNode)))
            if not terminal_paths:
                continue
            path, subtree = _choice(rng, terminal_paths)
            replacement = _mutate_terminal(subtree, field_values, rng, config)
        else:
            target_type = node_output_type(subtree)
            remaining_depth = max(1, config.max_depth - len(path))
            replacement = _new_subtree(target_type, field_values, rng, config, remaining_depth)

        if replacement == subtree:
            continue
        candidate = Program(replace_subtree(program.root, path, replacement), version=program.version)
        errors = validate_program(
            candidate,
            available_fields=field_values,
            max_depth=config.max_depth,
            max_size=config.max_size,
        )
        if not errors:
            return candidate

    return random_program(field_values, config=config, random_state=rng)


def crossover_programs(
    left: Program,
    right: Program,
    fields: Sequence[str],
    *,
    config: ProgramGeneratorConfig | None = None,
    random_state: int | random.Random | None = None,
) -> Program:
    config = config or ProgramGeneratorConfig()
    rng = _rng(random_state)
    field_values = tuple(dict.fromkeys(str(field) for field in fields))
    left_paths = tuple(iter_paths(left.root))
    right_paths_by_type: dict[str, list[tuple[NodePath, TreeNode]]] = {"numeric": [], "mask": []}
    for path, node in iter_paths(right.root):
        right_paths_by_type[node_output_type(node)].append((path, node))

    for _attempt in range(config.max_attempts):
        path, subtree = _choice(rng, left_paths)
        target_type = node_output_type(subtree)
        candidates = right_paths_by_type.get(target_type, [])
        if not candidates:
            continue
        _right_path, replacement = _choice(rng, candidates)
        candidate = Program(replace_subtree(left.root, path, replacement), version=left.version)
        errors = validate_program(
            candidate,
            available_fields=field_values,
            max_depth=config.max_depth,
            max_size=config.max_size,
        )
        if not errors:
            return candidate

    return mutate_program(left, field_values, config=config, random_state=rng)


def unique_programs(programs: Iterable[Program]) -> list[Program]:
    from .program import program_key

    seen: set[tuple[object, ...]] = set()
    output: list[Program] = []
    for program in programs:
        key = program_key(program)
        if key in seen:
            continue
        seen.add(key)
        output.append(program)
    return output


def program_fits_limits(program: Program, config: ProgramGeneratorConfig) -> bool:
    return node_depth(program.root) <= config.max_depth and node_size(program.root) <= config.max_size


__all__ = [
    "NodePath",
    "MutationKind",
    "ProgramGeneratorConfig",
    "random_program",
    "mutate_program",
    "crossover_programs",
    "iter_paths",
    "get_subtree",
    "replace_subtree",
    "unique_programs",
    "program_fits_limits",
]
