"""Type-rule system for typed tree GP.

Defines the sub_type taxonomy and BINARY_TYPE_RULES that control
which binary tree ops are legal between fields of different sub_types.

Design rules:
  - Same sub_type: all four binary ops (mean, diff, interaction, residual) are allowed.
  - Cross sub_type: only ops in BINARY_TYPE_RULES are allowed.
  - Not in BINARY_TYPE_RULES and sub_types differ → no ops allowed.
  - All leaves in one tree must share the same data_family.
  - No field may appear twice in leaves of the same tree.
"""

from __future__ import annotations

from typing import Mapping

from alpha_gen.behavior_gen.gene import BehaviorFieldRule

from .typed_tree import BinaryNode, FieldNode, TemplateTree, TreeExpr, UnaryNode

# ── sub_type constants ──────────────────────────────────────────

# fundamental
FUND_GROWTH = "fund_growth"
FUND_QUALITY = "fund_quality"
FUND_VALUE = "fund_value"

# price_volume
PV_MOMENTUM = "pv_momentum"
PV_VOLUME = "pv_volume"
PV_VOLATILITY = "pv_volatility"
PV_CROWDING = "pv_crowding"
PV_PANIC = "pv_panic"
PV_GENERAL = "pv_general"

# moneyflow
MF_LARGE = "mf_large"
MF_SMALL = "mf_small"
MF_ACTIVE = "mf_active"
MF_GENERAL = "mf_general"

# orderbook
OB_SPREAD = "ob_spread"
OB_DEPTH = "ob_depth"
OB_PRESSURE = "ob_pressure"

# control
CONTROL = "control"

ALL_SUB_TYPES: tuple[str, ...] = (
    FUND_GROWTH, FUND_QUALITY, FUND_VALUE,
    PV_MOMENTUM, PV_VOLUME, PV_VOLATILITY, PV_CROWDING, PV_PANIC, PV_GENERAL,
    MF_LARGE, MF_SMALL, MF_ACTIVE, MF_GENERAL,
    OB_SPREAD, OB_DEPTH, OB_PRESSURE,
    CONTROL,
)

# sub_type → data_family
SUB_TYPE_DATA_FAMILY: dict[str, str] = {
    FUND_GROWTH: "fundamental", FUND_QUALITY: "fundamental", FUND_VALUE: "fundamental",
    PV_MOMENTUM: "price_volume", PV_VOLUME: "price_volume",
    PV_VOLATILITY: "price_volume", PV_CROWDING: "price_volume",
    PV_PANIC: "price_volume", PV_GENERAL: "price_volume",
    MF_LARGE: "moneyflow", MF_SMALL: "moneyflow",
    MF_ACTIVE: "moneyflow", MF_GENERAL: "moneyflow",
    OB_SPREAD: "orderbook", OB_DEPTH: "orderbook", OB_PRESSURE: "orderbook",
    CONTROL: "control",
}


# ── binary type rules ──────────────────────────────────────────
#
# Same sub_type pairs: all four ops allowed by default (not listed).
# Cross sub_type: only listed ops are allowed.
# Ops: "mean", "diff", "interaction", "residual"

BINARY_TYPE_RULES: dict[tuple[str, str], frozenset[str]] = {
    # ─── fundamental cross-type ───
    # growth + quality: growth vs quality divergence
    (FUND_GROWTH, FUND_QUALITY): frozenset({"mean", "diff", "residual"}),
    (FUND_QUALITY, FUND_GROWTH): frozenset({"mean", "diff", "residual"}),
    # growth + value: growth-adjusted valuation
    (FUND_GROWTH, FUND_VALUE): frozenset({"mean", "diff", "interaction", "residual"}),
    (FUND_VALUE, FUND_GROWTH): frozenset({"mean", "diff", "interaction", "residual"}),
    # quality + value: quality premium
    (FUND_QUALITY, FUND_VALUE): frozenset({"mean", "diff", "residual"}),
    (FUND_VALUE, FUND_QUALITY): frozenset({"mean", "diff", "residual"}),

    # ─── price_volume cross-type ───
    # momentum + volume: volume-confirmed momentum
    (PV_MOMENTUM, PV_VOLUME): frozenset({"mean", "diff"}),
    (PV_VOLUME, PV_MOMENTUM): frozenset({"mean", "diff"}),
    # momentum + volatility: risk-adjusted return
    (PV_MOMENTUM, PV_VOLATILITY): frozenset({"residual"}),
    (PV_VOLATILITY, PV_MOMENTUM): frozenset({"residual"}),
    # momentum + crowding: crowding feedback
    (PV_MOMENTUM, PV_CROWDING): frozenset({"interaction", "residual"}),
    (PV_CROWDING, PV_MOMENTUM): frozenset({"interaction", "residual"}),
    # momentum + panic: oversold reversal
    (PV_MOMENTUM, PV_PANIC): frozenset({"diff", "interaction"}),
    (PV_PANIC, PV_MOMENTUM): frozenset({"diff", "interaction"}),
    # volume + crowding: volume-led crowding
    (PV_VOLUME, PV_CROWDING): frozenset({"mean", "interaction"}),
    (PV_CROWDING, PV_VOLUME): frozenset({"mean", "interaction"}),
    # volume + volatility: liquidity risk
    (PV_VOLUME, PV_VOLATILITY): frozenset({"residual"}),
    (PV_VOLATILITY, PV_VOLUME): frozenset({"residual"}),
    # crowding + panic: extreme event
    (PV_CROWDING, PV_PANIC): frozenset({"mean", "interaction"}),
    (PV_PANIC, PV_CROWDING): frozenset({"mean", "interaction"}),

    # ─── moneyflow cross-type ───
    # large + small: smart/dumb money divergence
    (MF_LARGE, MF_SMALL): frozenset({"mean", "diff"}),
    (MF_SMALL, MF_LARGE): frozenset({"mean", "diff"}),
    # large + active: institutional conviction
    (MF_LARGE, MF_ACTIVE): frozenset({"mean", "interaction"}),
    (MF_ACTIVE, MF_LARGE): frozenset({"mean", "interaction"}),
    # small + active: retail participation
    (MF_SMALL, MF_ACTIVE): frozenset({"mean", "diff"}),
    (MF_ACTIVE, MF_SMALL): frozenset({"mean", "diff"}),

    # ─── orderbook cross-type ───
    # pressure + spread: true demand net of liquidity cost
    (OB_PRESSURE, OB_SPREAD): frozenset({"residual"}),
    (OB_SPREAD, OB_PRESSURE): frozenset({"residual"}),
    # pressure + depth: liquidity-backed pressure
    (OB_PRESSURE, OB_DEPTH): frozenset({"mean", "interaction"}),
    (OB_DEPTH, OB_PRESSURE): frozenset({"mean", "interaction"}),
    # spread + depth: liquidity profile
    (OB_SPREAD, OB_DEPTH): frozenset({"mean", "diff"}),
    (OB_DEPTH, OB_SPREAD): frozenset({"mean", "diff"}),
}


def allowed_binary_ops(left_sub_type: str, right_sub_type: str) -> frozenset[str]:
    """Return the set of binary ops allowed between two sub_types.

    Same sub_type → all four allowed.
    Different sub_types → only ops in BINARY_TYPE_RULES.
    """
    if left_sub_type == right_sub_type:
        return frozenset({"mean", "diff", "interaction", "residual"})
    return BINARY_TYPE_RULES.get(
        (left_sub_type, right_sub_type),
        frozenset(),
    )


# ── leaf validators ────────────────────────────────────────────


def find_leaf_fields(node: TreeExpr) -> set[str]:
    """Return all field names in leaf positions of a tree expression."""
    if isinstance(node, FieldNode):
        return {node.field}
    if isinstance(node, UnaryNode):
        return find_leaf_fields(node.child)
    if isinstance(node, BinaryNode):
        return find_leaf_fields(node.left) | find_leaf_fields(node.right)
    raise TypeError(type(node).__name__)


def find_leaf_sub_types(
    node: TreeExpr,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> set[str]:
    """Return sub_types of all leaf fields in a tree expression."""
    sub_types: set[str] = set()
    leaves = find_leaf_fields(node)
    for field_name in leaves:
        if field_name not in field_rules:
            continue
        rule = field_rules[field_name]
        sub_types.add(getattr(rule, "sub_type", rule.sub_family))
    return sub_types


def find_leaf_data_families(
    node: TreeExpr,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> set[str]:
    """Return data_family values for all leaves in a tree expression."""
    families: set[str] = set()
    leaves = find_leaf_fields(node)
    for field_name in leaves:
        if field_name not in field_rules:
            continue
        families.add(field_rules[field_name].data_family)
    return families


def _find_leaf_fields_list(node: TreeExpr) -> list[str]:
    """Return all field names in leaf positions as a list (preserving duplicates)."""
    if isinstance(node, FieldNode):
        return [node.field]
    if isinstance(node, UnaryNode):
        return _find_leaf_fields_list(node.child)
    if isinstance(node, BinaryNode):
        return _find_leaf_fields_list(node.left) + _find_leaf_fields_list(node.right)
    raise TypeError(type(node).__name__)


def validate_leaf_uniqueness(tree: TemplateTree) -> list[str]:
    """Same field must not appear in any two leaf positions of one tree.

    Checks both:
      - Duplicates within a single slot's subtree (e.g. mean(a, a))
      - Duplicates across different slots
    """
    errors: list[str] = []
    # Check within each slot subtree for internal duplicates
    for slot_name, node in tree.slots.items():
        leaves = _find_leaf_fields_list(node)
        if len(leaves) != len(set(leaves)):
            from collections import Counter
            counts = Counter(leaves)
            dupes = [f for f, c in counts.items() if c > 1]
            errors.append(
                f"slot {slot_name!r} contains duplicate field(s): {dupes}"
            )
    # Check across slots
    seen: dict[str, str] = {}
    for slot_name, node in tree.slots.items():
        for leaf in _find_leaf_fields_list(node):
            if leaf in seen:
                errors.append(
                    f"field {leaf!r} appears in both slot {seen[leaf]!r} and slot {slot_name!r}"
                )
            seen[leaf] = slot_name
    return errors


def validate_within_tree_data_family(
    tree: TemplateTree,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> list[str]:
    """All leaves in one tree must share the same data_family."""
    errors: list[str] = []
    for slot_name, node in tree.slots.items():
        families = find_leaf_data_families(node, field_rules)
        families.discard("control")
        if len(families) > 1:
            errors.append(
                f"slot {slot_name!r} contains leaves from multiple data_families: {sorted(families)}"
            )
    return errors


def _validate_binary_op_types_node(
    node: TreeExpr,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> list[str]:
    """Recursively validate binary ops on this node and its children."""
    errors: list[str] = []
    if isinstance(node, FieldNode):
        return errors
    if isinstance(node, UnaryNode):
        return _validate_binary_op_types_node(node.child, field_rules)
    if isinstance(node, BinaryNode):
        left_st = find_leaf_sub_types(node.left, field_rules)
        right_st = find_leaf_sub_types(node.right, field_rules)
        for l_st in left_st:
            for r_st in right_st:
                allowed = allowed_binary_ops(l_st, r_st)
                if node.op not in allowed:
                    errors.append(
                        f"binary op {node.op!r} between sub_types "
                        f"{l_st!r} and {r_st!r} is not allowed "
                        f"(allowed: {sorted(allowed)})"
                    )
        errors.extend(_validate_binary_op_types_node(node.left, field_rules))
        errors.extend(_validate_binary_op_types_node(node.right, field_rules))
    return errors


def validate_binary_op_types(
    node: TreeExpr,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> list[str]:
    """Validate all BinaryNode ops in a tree against BINARY_TYPE_RULES."""
    return _validate_binary_op_types_node(node, field_rules)
