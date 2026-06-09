from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, TypeAlias

from .registry import OPERATOR_REGISTRY, OperatorSpec, get_operator


OutputType: TypeAlias = Literal["numeric", "mask"]


@dataclass(frozen=True)
class FieldNode:
    """Numeric terminal that resolves to one input field tensor."""

    field: str

    def to_dict(self) -> dict[str, object]:
        return {"node": "field", "field": self.field}


@dataclass(frozen=True)
class ConstNode:
    """Scalar numeric constant, broadcast by the evaluator."""

    value: float

    def to_dict(self) -> dict[str, object]:
        return {"node": "const", "value": float(self.value)}


@dataclass(frozen=True)
class UnaryNode:
    """One-argument operator node.

    This covers numeric transforms such as ``ts_mean_20`` and mask generators
    such as ``mask_rank_high_80``. Unary operators always consume numeric input.
    """

    op: str
    child: "TreeNode"

    def to_dict(self) -> dict[str, object]:
        return {"node": "unary", "op": self.op, "child": self.child.to_dict()}


@dataclass(frozen=True)
class BinaryNode:
    """Two-argument numeric operator node."""

    op: str
    left: "TreeNode"
    right: "TreeNode"

    def to_dict(self) -> dict[str, object]:
        return {
            "node": "binary",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True)
class GateNode:
    """Apply a mask to a numeric signal with ``gate_nan`` or ``gate_zero``."""

    op: str
    signal: "TreeNode"
    mask: "TreeNode"

    def to_dict(self) -> dict[str, object]:
        return {
            "node": "gate",
            "op": self.op,
            "signal": self.signal.to_dict(),
            "mask": self.mask.to_dict(),
        }


TreeNode: TypeAlias = FieldNode | ConstNode | UnaryNode | BinaryNode | GateNode


@dataclass(frozen=True)
class Program:
    """One free-form GP factor expression tree."""

    root: TreeNode
    version: int = 1

    @property
    def output_type(self) -> OutputType:
        return node_output_type(self.root)

    @property
    def size(self) -> int:
        return node_size(self.root)

    @property
    def depth(self) -> int:
        return node_depth(self.root)

    @property
    def complexity_cost(self) -> float:
        return node_complexity_cost(self.root)

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "root": self.root.to_dict()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Program":
        return cls(
            root=node_from_dict(_expect_mapping(raw["root"], "root")),
            version=int(raw.get("version", 1)),
        )

    @classmethod
    def from_json(cls, text: str) -> "Program":
        raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise TypeError("program JSON must decode to a mapping")
        return cls.from_dict(raw)


COMMUTATIVE_OPS = frozenset({"add", "mul"})


def _expect_mapping(raw: object, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return raw


def node_from_dict(raw: Mapping[str, object]) -> TreeNode:
    node_type = str(raw["node"])
    if node_type == "field":
        return FieldNode(field=str(raw["field"]))
    if node_type == "const":
        return ConstNode(value=float(raw["value"]))
    if node_type == "unary":
        return UnaryNode(
            op=str(raw["op"]),
            child=node_from_dict(_expect_mapping(raw["child"], "child")),
        )
    if node_type == "binary":
        return BinaryNode(
            op=str(raw["op"]),
            left=node_from_dict(_expect_mapping(raw["left"], "left")),
            right=node_from_dict(_expect_mapping(raw["right"], "right")),
        )
    if node_type == "gate":
        return GateNode(
            op=str(raw["op"]),
            signal=node_from_dict(_expect_mapping(raw["signal"], "signal")),
            mask=node_from_dict(_expect_mapping(raw["mask"], "mask")),
        )
    raise ValueError(f"unknown node type: {node_type!r}")


def _operator(node: UnaryNode | BinaryNode | GateNode) -> OperatorSpec | None:
    return OPERATOR_REGISTRY.get(node.op)


def node_output_type(node: TreeNode) -> OutputType:
    if isinstance(node, (FieldNode, ConstNode)):
        return "numeric"
    if isinstance(node, (UnaryNode, BinaryNode, GateNode)):
        spec = get_operator(node.op)
        if spec.output_type not in {"numeric", "mask"}:
            raise ValueError(f"operator {node.op!r} has unsupported output type {spec.output_type!r}")
        return spec.output_type  # type: ignore[return-value]
    raise TypeError(type(node).__name__)


def node_size(node: TreeNode) -> int:
    if isinstance(node, (FieldNode, ConstNode)):
        return 1
    if isinstance(node, UnaryNode):
        return 1 + node_size(node.child)
    if isinstance(node, BinaryNode):
        return 1 + node_size(node.left) + node_size(node.right)
    if isinstance(node, GateNode):
        return 1 + node_size(node.signal) + node_size(node.mask)
    raise TypeError(type(node).__name__)


def node_depth(node: TreeNode) -> int:
    if isinstance(node, (FieldNode, ConstNode)):
        return 1
    if isinstance(node, UnaryNode):
        return 1 + node_depth(node.child)
    if isinstance(node, BinaryNode):
        return 1 + max(node_depth(node.left), node_depth(node.right))
    if isinstance(node, GateNode):
        return 1 + max(node_depth(node.signal), node_depth(node.mask))
    raise TypeError(type(node).__name__)


def iter_nodes(node: TreeNode) -> Iterable[TreeNode]:
    yield node
    if isinstance(node, UnaryNode):
        yield from iter_nodes(node.child)
    elif isinstance(node, BinaryNode):
        yield from iter_nodes(node.left)
        yield from iter_nodes(node.right)
    elif isinstance(node, GateNode):
        yield from iter_nodes(node.signal)
        yield from iter_nodes(node.mask)


def field_names(node: TreeNode) -> tuple[str, ...]:
    return tuple(sorted({item.field for item in iter_nodes(node) if isinstance(item, FieldNode)}))


def operator_names(node: TreeNode) -> tuple[str, ...]:
    return tuple(
        item.op
        for item in iter_nodes(node)
        if isinstance(item, (UnaryNode, BinaryNode, GateNode))
    )


def node_complexity_cost(node: TreeNode) -> float:
    if isinstance(node, FieldNode):
        return 0.0
    if isinstance(node, ConstNode):
        return 0.05
    if isinstance(node, UnaryNode):
        spec = get_operator(node.op)
        return float(spec.cost) + node_complexity_cost(node.child)
    if isinstance(node, BinaryNode):
        spec = get_operator(node.op)
        return float(spec.cost) + node_complexity_cost(node.left) + node_complexity_cost(node.right)
    if isinstance(node, GateNode):
        spec = get_operator(node.op)
        return float(spec.cost) + node_complexity_cost(node.signal) + node_complexity_cost(node.mask)
    raise TypeError(type(node).__name__)


def node_key(node: TreeNode) -> tuple[object, ...]:
    if isinstance(node, FieldNode):
        return ("field", node.field)
    if isinstance(node, ConstNode):
        return ("const", round(float(node.value), 12))
    if isinstance(node, UnaryNode):
        return ("unary", node.op, node_key(node.child))
    if isinstance(node, BinaryNode):
        children = (node_key(node.left), node_key(node.right))
        if node.op in COMMUTATIVE_OPS:
            children = tuple(sorted(children))
        return ("binary", node.op, children)
    if isinstance(node, GateNode):
        return ("gate", node.op, node_key(node.signal), node_key(node.mask))
    raise TypeError(type(node).__name__)


def program_key(program: Program) -> tuple[object, ...]:
    return ("program", program.version, node_key(program.root))


def node_expression(node: TreeNode) -> str:
    if isinstance(node, FieldNode):
        return node.field
    if isinstance(node, ConstNode):
        return f"{float(node.value):.6g}"
    if isinstance(node, UnaryNode):
        return f"{node.op}({node_expression(node.child)})"
    if isinstance(node, BinaryNode):
        return f"{node.op}({node_expression(node.left)}, {node_expression(node.right)})"
    if isinstance(node, GateNode):
        return f"{node.op}({node_expression(node.signal)}, {node_expression(node.mask)})"
    raise TypeError(type(node).__name__)


def program_expression(program: Program) -> str:
    return node_expression(program.root)


def validate_node(
    node: TreeNode,
    *,
    available_fields: Iterable[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    field_set = set(available_fields) if available_fields is not None else None

    def visit(current: TreeNode, path: str) -> OutputType:
        if isinstance(current, FieldNode):
            if not current.field:
                errors.append(f"{path}: field name must be non-empty")
            if field_set is not None and current.field not in field_set:
                errors.append(f"{path}: unknown field {current.field!r}")
            return "numeric"

        if isinstance(current, ConstNode):
            if not math.isfinite(float(current.value)):
                errors.append(f"{path}: const value must be finite")
            return "numeric"

        if isinstance(current, UnaryNode):
            spec = _operator(current)
            child_type = visit(current.child, f"{path}.child")
            if spec is None:
                errors.append(f"{path}: unknown operator {current.op!r}")
                return "numeric"
            if spec.arity != 1:
                errors.append(f"{path}: operator {current.op!r} expects arity {spec.arity}, not unary")
            if spec.category == "gate":
                errors.append(f"{path}: gate operator {current.op!r} must use GateNode")
            if child_type != "numeric":
                errors.append(f"{path}: unary operator {current.op!r} requires numeric child, got {child_type}")
            return spec.output_type if spec.output_type in {"numeric", "mask"} else "numeric"  # type: ignore[return-value]

        if isinstance(current, BinaryNode):
            spec = _operator(current)
            left_type = visit(current.left, f"{path}.left")
            right_type = visit(current.right, f"{path}.right")
            if spec is None:
                errors.append(f"{path}: unknown operator {current.op!r}")
                return "numeric"
            if spec.arity != 2:
                errors.append(f"{path}: operator {current.op!r} expects arity {spec.arity}, not binary")
            if spec.category == "gate":
                errors.append(f"{path}: gate operator {current.op!r} must use GateNode")
            if left_type != "numeric":
                errors.append(f"{path}: left child of {current.op!r} must be numeric, got {left_type}")
            if right_type != "numeric":
                errors.append(f"{path}: right child of {current.op!r} must be numeric, got {right_type}")
            if spec.output_type != "numeric":
                errors.append(f"{path}: binary operator {current.op!r} must output numeric")
            return "numeric"

        if isinstance(current, GateNode):
            spec = _operator(current)
            signal_type = visit(current.signal, f"{path}.signal")
            mask_type = visit(current.mask, f"{path}.mask")
            if spec is None:
                errors.append(f"{path}: unknown operator {current.op!r}")
                return "numeric"
            if spec.category != "gate" or spec.arity != 2:
                errors.append(f"{path}: operator {current.op!r} is not a binary gate")
            if signal_type != "numeric":
                errors.append(f"{path}: gate signal must be numeric, got {signal_type}")
            if mask_type != "mask":
                errors.append(f"{path}: gate mask must be mask, got {mask_type}")
            return "numeric"

        errors.append(f"{path}: unknown node class {type(current).__name__}")
        return "numeric"

    visit(node, "root")
    return errors


def validate_program(
    program: Program,
    *,
    available_fields: Iterable[str] | None = None,
    max_depth: int | None = None,
    max_size: int | None = None,
) -> list[str]:
    errors = validate_node(program.root, available_fields=available_fields)
    try:
        root_output_type = program.output_type
    except (KeyError, ValueError):
        root_output_type = "numeric"
    if root_output_type != "numeric":
        errors.append(f"program root must be numeric, got {root_output_type}")
    if max_depth is not None and program.depth > max_depth:
        errors.append(f"program depth {program.depth} exceeds max_depth {max_depth}")
    if max_size is not None and program.size > max_size:
        errors.append(f"program size {program.size} exceeds max_size {max_size}")
    return errors


def is_valid_program(program: Program, **kwargs: object) -> bool:
    return not validate_program(program, **kwargs)


__all__ = [
    "OutputType",
    "FieldNode",
    "ConstNode",
    "UnaryNode",
    "BinaryNode",
    "GateNode",
    "TreeNode",
    "Program",
    "node_from_dict",
    "node_output_type",
    "node_size",
    "node_depth",
    "iter_nodes",
    "field_names",
    "operator_names",
    "node_complexity_cost",
    "node_key",
    "program_key",
    "node_expression",
    "program_expression",
    "validate_node",
    "validate_program",
    "is_valid_program",
]
