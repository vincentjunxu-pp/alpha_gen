from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from alpha_gen.behavior_gen.gene import (
    CONDITION_OP_CHOICES,
    DIRECTION_POLICIES,
    MODE_REGISTRY,
    UNARY_OP_CHOICES,
    BehaviorFieldRule,
    ConditionGene,
    ModeSpec,
    SlotSpec,
    condition_fields_for_mode,
    fields_for_slot,
)


ROLE_PRESERVING_UNARY_OPS = (
    "rank",
    "zscore",
    "ind_rank",
    "ind_zscore",
    "ts_zscore_5d",
    "ts_zscore_20d",
    "neg",
    "abs",
    "clip",
)

ROLE_PRESERVING_BINARY_OPS = (
    "mean",
    "diff",
    "interaction",
    "residual",
)

@dataclass(frozen=True)
class FieldNode:
    """Typed terminal: one metadata-approved field in one semantic slot."""

    semantic_type: str
    field: str
    unary_op: str = "rank_pct"

    def to_dict(self) -> dict[str, object]:
        return {
            "node": "field",
            "semantic_type": self.semantic_type,
            "field": self.field,
            "unary_op": self.unary_op,
        }


@dataclass(frozen=True)
class UnaryNode:
    """Role-preserving unary operation over a typed subtree."""

    semantic_type: str
    op: str
    child: "TreeExpr"

    def to_dict(self) -> dict[str, object]:
        return {
            "node": "unary",
            "semantic_type": self.semantic_type,
            "op": self.op,
            "child": self.child.to_dict(),
        }


@dataclass(frozen=True)
class BinaryNode:
    """Role-preserving binary operation over two same-type subtrees."""

    semantic_type: str
    op: str
    left: "TreeExpr"
    right: "TreeExpr"

    def to_dict(self) -> dict[str, object]:
        return {
            "node": "binary",
            "semantic_type": self.semantic_type,
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


TreeExpr = FieldNode | UnaryNode | BinaryNode


@dataclass(frozen=True)
class ConditionNode:
    """A typed gate condition used by a behavior mechanism template."""

    field: str
    unary_op: str = "rank_pct"
    condition_op: str = "top_quantile"
    threshold: float = 0.6

    def to_condition_gene(self) -> ConditionGene:
        return ConditionGene(
            field=self.field,
            unary_op=self.unary_op,
            condition_op=self.condition_op,
            threshold=self.threshold,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ConditionNode":
        return cls(
            field=str(raw["field"]),
            unary_op=str(raw.get("unary_op", "rank_pct")),
            condition_op=str(raw.get("condition_op", "top_quantile")),
            threshold=float(raw.get("threshold", 0.6)),
        )


@dataclass(frozen=True)
class TemplateTree:
    """Strongly typed behavior tree.

    The root is not a free arithmetic expression. It is a behavior mechanism
    template (`mode`) with typed slot subtrees and one allowed root combiner.
    """

    mode: str
    combiner: str
    slots: Mapping[str, TreeExpr]
    conditions: tuple[ConditionNode, ...] = ()
    direction_policy: str = "fixed"
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "combiner": self.combiner,
            "slots": {name: node.to_dict() for name, node in self.slots.items()},
            "conditions": [condition.to_dict() for condition in self.conditions],
            "direction_policy": self.direction_policy,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "TemplateTree":
        raw_slots = raw.get("slots", {})
        if not isinstance(raw_slots, Mapping):
            raise TypeError("TemplateTree slots must be a mapping")
        return cls(
            mode=str(raw["mode"]),
            combiner=str(raw["combiner"]),
            slots={str(name): node_from_dict(node) for name, node in raw_slots.items()},  # type: ignore[arg-type]
            conditions=tuple(ConditionNode.from_dict(item) for item in raw.get("conditions", ())),  # type: ignore[arg-type]
            direction_policy=str(raw.get("direction_policy", "fixed")),
            version=int(raw.get("version", 1)),
        )


def node_from_dict(raw: Mapping[str, object]) -> TreeExpr:
    node_type = str(raw["node"])
    if node_type == "field":
        return FieldNode(
            semantic_type=str(raw["semantic_type"]),
            field=str(raw["field"]),
            unary_op=str(raw.get("unary_op", "rank_pct")),
        )
    if node_type == "unary":
        return UnaryNode(
            semantic_type=str(raw["semantic_type"]),
            op=str(raw["op"]),
            child=node_from_dict(raw["child"]),  # type: ignore[arg-type]
        )
    if node_type == "binary":
        return BinaryNode(
            semantic_type=str(raw["semantic_type"]),
            op=str(raw["op"]),
            left=node_from_dict(raw["left"]),  # type: ignore[arg-type]
            right=node_from_dict(raw["right"]),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown node type: {node_type!r}")


def node_depth(node: TreeExpr) -> int:
    if isinstance(node, FieldNode):
        return 1
    if isinstance(node, UnaryNode):
        return 1 + node_depth(node.child)
    if isinstance(node, BinaryNode):
        return 1 + max(node_depth(node.left), node_depth(node.right))
    raise TypeError(type(node).__name__)


def node_size(node: TreeExpr) -> int:
    if isinstance(node, FieldNode):
        return 1
    if isinstance(node, UnaryNode):
        return 1 + node_size(node.child)
    if isinstance(node, BinaryNode):
        return 1 + node_size(node.left) + node_size(node.right)
    raise TypeError(type(node).__name__)


def tree_depth(tree: TemplateTree) -> int:
    if not tree.slots:
        return 1
    return 1 + max(node_depth(node) for node in tree.slots.values())


def tree_size(tree: TemplateTree) -> int:
    return 1 + sum(node_size(node) for node in tree.slots.values()) + len(tree.conditions)


def iter_slot_nodes(node: TreeExpr) -> Iterable[TreeExpr]:
    yield node
    if isinstance(node, UnaryNode):
        yield from iter_slot_nodes(node.child)
    elif isinstance(node, BinaryNode):
        yield from iter_slot_nodes(node.left)
        yield from iter_slot_nodes(node.right)


def _node_key(node: TreeExpr) -> tuple[object, ...]:
    if isinstance(node, FieldNode):
        return ("field", node.semantic_type, node.field, node.unary_op)
    if isinstance(node, UnaryNode):
        return ("unary", node.semantic_type, node.op, _node_key(node.child))
    if isinstance(node, BinaryNode):
        if node.op in {"mean", "interaction"}:
            children = tuple(sorted((_node_key(node.left), _node_key(node.right))))
        else:
            children = (_node_key(node.left), _node_key(node.right))
        return ("binary", node.semantic_type, node.op, children)
    raise TypeError(type(node).__name__)


def tree_key(tree: TemplateTree) -> tuple[object, ...]:
    conditions = tuple(
        sorted((condition.field, condition.unary_op, condition.condition_op, round(condition.threshold, 4)) for condition in tree.conditions)
    )
    return (
        tree.mode,
        tree.combiner,
        tuple(sorted((name, _node_key(node)) for name, node in tree.slots.items())),
        conditions,
        tree.direction_policy,
    )


def node_expression(node: TreeExpr) -> str:
    if isinstance(node, FieldNode):
        return f"{node.unary_op}({node.field})"
    if isinstance(node, UnaryNode):
        return f"{node.op}({node_expression(node.child)})"
    if isinstance(node, BinaryNode):
        return f"{node.op}({node_expression(node.left)}, {node_expression(node.right)})"
    raise TypeError(type(node).__name__)


def tree_expression(tree: TemplateTree) -> str:
    slot_text = ", ".join(f"{name}={node_expression(node)}" for name, node in sorted(tree.slots.items()))
    condition_text = ""
    if tree.conditions:
        condition_text = "; if " + " & ".join(
            f"{condition.condition_op}({condition.unary_op}({condition.field}), {condition.threshold:.2f})"
            for condition in tree.conditions
        )
    return f"{tree.mode}/{tree.combiner}: {slot_text}{condition_text}"


def _slot_node_is_valid(
    node: TreeExpr,
    slot_name: str,
    slot_spec: SlotSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> list[str]:
    errors: list[str] = []
    if node.semantic_type != slot_name:
        errors.append(f"node semantic_type {node.semantic_type!r} does not match slot {slot_name!r}")

    if isinstance(node, FieldNode):
        candidates = set(fields_for_slot(field_rules, slot_name, slot_spec))
        if node.field not in field_rules:
            errors.append(f"unknown field {node.field!r}")
        elif node.field not in candidates:
            errors.append(f"field {node.field!r} cannot fill slot {slot_name!r}")
        elif node.unary_op not in field_rules[node.field].allowed_unary_ops:
            errors.append(f"field {node.field!r} does not allow unary_op {node.unary_op!r}")
        if node.unary_op not in UNARY_OP_CHOICES:
            errors.append(f"unknown unary_op {node.unary_op!r}")
    elif isinstance(node, UnaryNode):
        if node.op not in ROLE_PRESERVING_UNARY_OPS:
            errors.append(f"unknown unary tree op {node.op!r}")
        errors.extend(_slot_node_is_valid(node.child, slot_name, slot_spec, field_rules))
    elif isinstance(node, BinaryNode):
        if node.op not in ROLE_PRESERVING_BINARY_OPS:
            errors.append(f"unknown binary tree op {node.op!r}")
        errors.extend(_slot_node_is_valid(node.left, slot_name, slot_spec, field_rules))
        errors.extend(_slot_node_is_valid(node.right, slot_name, slot_spec, field_rules))
    else:
        errors.append(f"unknown node class {type(node).__name__}")
    return errors


def validate_slot_node(
    node: TreeExpr,
    slot_name: str,
    slot_spec: SlotSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> list[str]:
    """Return validation errors for one typed slot subtree."""

    return _slot_node_is_valid(node, slot_name, slot_spec, field_rules)


def _condition_errors(
    tree: TemplateTree,
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> list[str]:
    errors: list[str] = []
    candidates = set(condition_fields_for_mode(field_rules, mode_spec))
    if len(tree.conditions) > mode_spec.max_conditions:
        errors.append(f"mode {tree.mode!r} allows at most {mode_spec.max_conditions} conditions")
    for idx, condition in enumerate(tree.conditions):
        if condition.field not in field_rules:
            errors.append(f"condition {idx} uses unknown field {condition.field!r}")
            continue
        if condition.field not in candidates:
            errors.append(f"condition {idx} field {condition.field!r} is not allowed for mode {tree.mode!r}")
        if condition.unary_op not in UNARY_OP_CHOICES:
            errors.append(f"condition {idx} unknown unary_op {condition.unary_op!r}")
        elif condition.unary_op not in field_rules[condition.field].allowed_unary_ops:
            errors.append(f"condition {idx} field {condition.field!r} does not allow unary_op {condition.unary_op!r}")
        if condition.condition_op not in CONDITION_OP_CHOICES:
            errors.append(f"condition {idx} unknown condition_op {condition.condition_op!r}")
        if condition.condition_op in {"top_quantile", "bottom_quantile"} and not 0.0 < condition.threshold < 1.0:
            errors.append(f"condition {idx} quantile threshold must be in (0, 1)")
    return errors


def validate_tree(
    tree: TemplateTree,
    field_rules: Mapping[str, BehaviorFieldRule],
    *,
    max_depth: int | None = None,
    max_nodes: int | None = None,
    max_slot_depth: int | None = None,
) -> list[str]:
    """Return all semantic and complexity validation errors.

    Binary type-rule validation is always enabled — it uses each field's
    ``sub_type`` (or ``sub_family`` fallback) from *field_rules*, not an
    external mapping.  This guarantees that sampling, mutation, crossover,
    repair and evaluation all share the same legality definition.
    """

    errors: list[str] = []
    if tree.mode not in MODE_REGISTRY:
        return [f"unknown mode {tree.mode!r}"]
    mode_spec = MODE_REGISTRY[tree.mode]
    if tree.combiner not in mode_spec.allowed_combiners:
        errors.append(f"combiner {tree.combiner!r} is not allowed for mode {tree.mode!r}")
    elif tree.combiner in {"rank_gap", "residual_gap", "gated_rank_gap", "confirm", "gated_confirm"}:
        n_selected_slots = len([name for name in tree.slots if name in mode_spec.slots])
        if n_selected_slots < 2:
            errors.append(f"combiner {tree.combiner!r} requires at least two selected slots")

    if tree.direction_policy not in DIRECTION_POLICIES:
        errors.append(f"direction_policy must be one of {DIRECTION_POLICIES}, got {tree.direction_policy!r}")
    elif tree.direction_policy != mode_spec.direction_policy and tree.direction_policy == "regime_switch":
        errors.append(f"mode {tree.mode!r} does not allow regime_switch direction policy")

    unknown_slots = sorted(set(tree.slots) - set(mode_spec.slots))
    if unknown_slots:
        errors.append(f"tree has slots not defined by mode {tree.mode!r}: {unknown_slots}")

    for slot_name, slot_spec in mode_spec.slots.items():
        node = tree.slots.get(slot_name)
        if node is None:
            if slot_spec.required:
                errors.append(f"missing required slot {slot_name!r}")
            continue
        errors.extend(_slot_node_is_valid(node, slot_name, slot_spec, field_rules))

    errors.extend(_condition_errors(tree, mode_spec, field_rules))

    # Type-rule validators — always active.
    # validate_binary_op_types reads sub_type directly from field_rules,
    # so it works for both real (sub_type) and mock (sub_family) metadata.
    from .type_rules import (
        validate_binary_op_types,
        validate_leaf_uniqueness,
        validate_within_tree_data_family,
    )

    errors.extend(validate_leaf_uniqueness(tree))
    errors.extend(validate_within_tree_data_family(tree, field_rules))
    for node in tree.slots.values():
        errors.extend(validate_binary_op_types(node, field_rules))

    if max_slot_depth is not None:
        for slot_name, node in tree.slots.items():
            if node_depth(node) > max_slot_depth:
                errors.append(f"slot {slot_name!r} depth {node_depth(node)} exceeds max_slot_depth {max_slot_depth}")
    if max_depth is not None and tree_depth(tree) > max_depth:
        errors.append(f"tree depth {tree_depth(tree)} exceeds max_depth {max_depth}")
    if max_nodes is not None and tree_size(tree) > max_nodes:
        errors.append(f"tree size {tree_size(tree)} exceeds max_nodes {max_nodes}")
    return errors


def is_valid_tree(
    tree: TemplateTree,
    field_rules: Mapping[str, BehaviorFieldRule],
    *,
    max_depth: int | None = None,
    max_nodes: int | None = None,
    max_slot_depth: int | None = None,
) -> bool:
    return not validate_tree(
        tree, field_rules,
        max_depth=max_depth, max_nodes=max_nodes,
        max_slot_depth=max_slot_depth,
    )
