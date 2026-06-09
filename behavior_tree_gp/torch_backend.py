"""Recursive tensor evaluator for typed behavior trees.

Translates TreeExpr AST nodes into GPU tensors by walking the tree
recursively. Reuses BehaviorTorchContext for data access and core
tensor ops from alpha_gen.core.torch_backend.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import torch

from alpha_gen.behavior_gen.torch_backend import (
    BehaviorTorchContext,
    _condition_mask,
    _feature,
    _rolling_ts_zscore,
)
from alpha_gen.core.metrics import FactorScore
from alpha_gen.core.torch_backend import (
    _apply_mask,
    _nan_like,
    cross_sectional_residual_torch,
    cs_rank_pct_torch,
    cs_zscore_torch,
    evaluate_factor_tensor,
    industry_neutralize_torch,
    industry_rank_pct_torch,
    industry_zscore_torch,
)

from alpha_gen.behavior_gen.gene import BehaviorFieldRule  # re-export for convenience
from .typed_tree import (
    BinaryNode,
    FieldNode,
    TemplateTree,
    TreeExpr,
    UnaryNode,
    node_depth,
    tree_size,
)

# ═══════════════════════════════════════════════════════════════
# Tree-level tensor operators
# ═══════════════════════════════════════════════════════════════


def _tree_rank(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return cs_rank_pct_torch(values, mask=mask) - 0.5


def _tree_zscore(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return cs_zscore_torch(values, mask=mask)


def _tree_ind_rank(
    values: torch.Tensor,
    industry_codes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return industry_rank_pct_torch(values, industry_codes, mask=mask) - 0.5


def _tree_ind_zscore(
    values: torch.Tensor,
    industry_codes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return industry_zscore_torch(values, industry_codes, mask=mask)


def _tree_ts_zscore_5d(values: torch.Tensor) -> torch.Tensor:
    return _rolling_ts_zscore(values, 5)


def _tree_ts_zscore_20d(values: torch.Tensor) -> torch.Tensor:
    return _rolling_ts_zscore(values, 20)


def _tree_neg(values: torch.Tensor) -> torch.Tensor:
    return -values


def _tree_abs(values: torch.Tensor) -> torch.Tensor:
    return values.abs()


def _tree_clip(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(-8.0, 8.0)


def _tree_mean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left + right) * 0.5


def _tree_diff(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left - right


def _tree_interaction(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left * right


def _tree_residual(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return cross_sectional_residual_torch(left, right, mask=mask)


# Dispatch tables
UNARY_TREE_OPS: dict[str, object] = {
    "rank": _tree_rank,
    "zscore": _tree_zscore,
    "ind_rank": _tree_ind_rank,
    "ind_zscore": _tree_ind_zscore,
    "ts_zscore_5d": _tree_ts_zscore_5d,
    "ts_zscore_20d": _tree_ts_zscore_20d,
    "neg": _tree_neg,
    "abs": _tree_abs,
    "clip": _tree_clip,
}

BINARY_TREE_OPS: dict[str, object] = {
    "mean": _tree_mean,
    "diff": _tree_diff,
    "interaction": _tree_interaction,
    "residual": _tree_residual,
}

# Unary ops that need a mask parameter
_UNARY_OPS_WITH_MASK = {"rank", "zscore"}

# Unary ops that need industry_codes
_UNARY_OPS_WITH_INDUSTRY = {"ind_rank", "ind_zscore"}


# ═══════════════════════════════════════════════════════════════
# Recursive tree evaluator
# ═══════════════════════════════════════════════════════════════


def evaluate_slot_tree(
    node: TreeExpr,
    ctx: BehaviorTorchContext,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Recursively evaluate a typed tree expression to a [T, N] tensor.

    FieldNode  → _feature(field, unary_op, ctx, mask)
    UnaryNode  → apply tree-level unary op to child
    BinaryNode → apply tree-level binary op to left and right children
    """
    if isinstance(node, FieldNode):
        return _feature(node.field, node.unary_op, ctx, mask)

    if isinstance(node, UnaryNode):
        child = evaluate_slot_tree(node.child, ctx, mask)
        op_fn = UNARY_TREE_OPS[node.op]
        if node.op in _UNARY_OPS_WITH_INDUSTRY:
            return _call_unary(op_fn, child, ctx.industry_codes(), mask)
        if node.op in _UNARY_OPS_WITH_MASK:
            return _call_unary(op_fn, child, mask)
        return _call_unary(op_fn, child)

    if isinstance(node, BinaryNode):
        left = evaluate_slot_tree(node.left, ctx, mask)
        right = evaluate_slot_tree(node.right, ctx, mask)
        op_fn = BINARY_TREE_OPS[node.op]
        if node.op == "residual":
            return _call_binary(op_fn, left, right, mask)
        return _call_binary(op_fn, left, right)

    raise TypeError(f"unknown tree node type: {type(node).__name__}")


def _call_unary(fn: object, *args: object) -> torch.Tensor:
    result = fn(*args)
    assert isinstance(result, torch.Tensor)
    return result


def _call_binary(fn: object, *args: object) -> torch.Tensor:
    result = fn(*args)
    assert isinstance(result, torch.Tensor)
    return result


# ═══════════════════════════════════════════════════════════════
# Slot → tensor evaluation
# ═══════════════════════════════════════════════════════════════


def _slot_values(
    tree: TemplateTree,
    ctx: BehaviorTorchContext,
    mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Evaluate each slot subtree to a tensor."""
    return {
        name: evaluate_slot_tree(node, ctx, mask)
        for name, node in tree.slots.items()
    }


# ═══════════════════════════════════════════════════════════════
# Combiner logic (adapted from behavior_gen/torch_backend.py)
# ═══════════════════════════════════════════════════════════════


def _zeros_like_context(ctx: BehaviorTorchContext) -> torch.Tensor:
    return torch.zeros_like(ctx.label())


def _sum_existing(tensors: list[torch.Tensor], ctx: BehaviorTorchContext) -> torch.Tensor:
    if not tensors:
        return _zeros_like_context(ctx)
    out = tensors[0]
    for t in tensors[1:]:
        out = out + t
    return out


def _apply_tree_conditions(
    raw: torch.Tensor,
    tree: TemplateTree,
    ctx: BehaviorTorchContext,
    base_mask: torch.Tensor | None,
    *,
    gate_fill: str = "nan",
) -> torch.Tensor:
    """Apply TemplateTree conditions without depending on behavior_gen internals."""
    if not tree.conditions:
        return raw
    if gate_fill not in {"zero", "nan"}:
        raise ValueError("gate_fill must be 'zero' or 'nan'")

    gate = torch.ones_like(raw, dtype=torch.bool)
    for condition in tree.conditions:
        gate = gate & _condition_mask(condition, ctx, base_mask)
    if base_mask is not None:
        gate = gate & base_mask

    fill_value = (
        torch.zeros_like(raw)
        if gate_fill == "zero"
        else torch.full_like(raw, float("nan"))
    )
    return torch.where(gate, raw, fill_value)


def _ordered_slot_values(
    tree: TemplateTree,
    values: Mapping[str, torch.Tensor],
) -> list[torch.Tensor]:
    from alpha_gen.behavior_gen.gene import MODE_REGISTRY

    mode_spec = MODE_REGISTRY[tree.mode]
    return [values[name] for name in mode_spec.slots if name in values]


def _combine_slot_values(
    tree: TemplateTree,
    values: dict[str, torch.Tensor],
    ctx: BehaviorTorchContext,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Combine slot tensor values using the mode's combiner.

    Mirrors _combine_behavior_gene from behavior_gen/torch_backend.py
    but operates on a dict of slot tensor values from evaluate_slot_tree.
    """
    ordered = _ordered_slot_values(tree, values)
    if not ordered:
        raise ValueError(f"tree mode {tree.mode!r} has no slot values")

    combiner = tree.combiner

    if combiner in {"rank_gap", "gated_rank_gap"}:
        if len(ordered) < 2:
            raise ValueError(f"{combiner} requires at least two slot values")
        raw = ordered[0] - ordered[1]

    elif combiner == "residual_gap":
        if len(ordered) < 2:
            raise ValueError("residual_gap requires at least two slot values")
        raw = cross_sectional_residual_torch(ordered[0], ordered[1], mask=mask)

    elif combiner == "quality_gap":
        raw = values["profit_growth"] - values["cashflow_quality"]
        if "price_reaction" in values:
            raw = raw + 0.25 * values["price_reaction"].abs()

    elif combiner == "crowding_interaction":
        raw = values["growth_anchor"] * values["crowding_signal"]
        if "fund_support" in values:
            raw = raw - values["fund_support"]

    elif combiner in {"confirm", "gated_confirm"}:
        if len(ordered) < 2:
            raise ValueError(f"{combiner} requires at least two slot values")
        raw = ordered[0] + ordered[1] + ordered[0] * ordered[1]
        # Identify the two primary slots so they are not double-counted
        # in the secondary loop below (ordered[0] & ordered[1] already
        # contribute 1× each plus the interaction term).
        from alpha_gen.behavior_gen.gene import MODE_REGISTRY as _REG
        primary_slot_names: set[str] = set()
        for slot_name in _REG[tree.mode].slots:
            if slot_name in values:
                primary_slot_names.add(slot_name)
                if len(primary_slot_names) >= 2:
                    break
        for name, value in values.items():
            if name in primary_slot_names:
                continue
            if name in {"fund_anchor", "flow_confirm", "price_anchor", "price_momentum"}:
                continue
            if name.endswith("control"):
                raw = raw - 0.25 * value.abs()
            else:
                raw = raw + 0.5 * value

    elif combiner == "risk_minus_confirm":
        risk_names = {
            "price_momentum", "retail_flow", "close_chase",
            "attention_heat", "crowding_signal", "liquidity_stress",
            "turnover_shock",
        }
        confirm_names = {
            "large_flow", "flow_confirm", "fund_support", "orderbook_filter",
        }
        risk_items = [v for name, v in values.items() if name in risk_names]
        confirm_items = [v for name, v in values.items() if name in confirm_names]
        risk = _sum_existing(risk_items, ctx)
        confirm = _sum_existing(confirm_items, ctx)
        if not risk_items or not torch.isfinite(risk).any():
            risk = _sum_existing(ordered[:2], ctx)
        raw = risk - confirm

    elif combiner == "panic_reversal":
        raw = values["fund_anchor"] * values["drawdown"] - values["sell_pressure"]
        if "orderbook_filter" in values:
            raw = raw + values["orderbook_filter"]

    elif combiner == "attention_risk":
        raw = values["attention_heat"]
        if "price_momentum" in values:
            raw = raw + values["price_momentum"]
        if "fund_support" in values:
            raw = raw - values["fund_support"]

    elif combiner == "orderbook_intent":
        raw = values["orderbook_pressure"]
        if "liquidity_stress" in values:
            raw = raw - values["liquidity_stress"]
        if "price_reaction" in values:
            raw = raw - 0.25 * values["price_reaction"].abs()

    elif combiner == "liquidity_gap":
        raw = values["liquidity_stress"] - values["turnover_shock"]
        if "flow_confirm" in values:
            raw = raw - values["flow_confirm"]

    elif combiner == "anchor_confirm":
        anchor = values.get("price_anchor", values.get("cost_anchor"))
        if anchor is None:
            raise ValueError("anchor_confirm requires price_anchor or cost_anchor")
        raw = anchor + values["price_momentum"] + anchor * values["price_momentum"]
        for extra in ("flow_confirm", "orderbook_filter", "fund_support"):
            if extra in values:
                raw = raw + 0.5 * values[extra]

    else:
        raise ValueError(f"unknown combiner: {combiner!r}")

    if combiner.startswith("gated"):
        raw = _apply_tree_conditions(raw, tree, ctx, mask, gate_fill="nan")

    return raw


# ═══════════════════════════════════════════════════════════════
# Main entry points
# ═══════════════════════════════════════════════════════════════


def calculate_tree_factor_tensor(
    tree: TemplateTree,
    ctx: BehaviorTorchContext,
    *,
    apply_mode_direction: bool = True,
    neutralize_size: bool = True,
    neutralize_industry: bool = False,
    size_field: str = "barra_size",
    tradeable_only: bool = True,
) -> torch.Tensor:
    """Evaluate a TemplateTree to a GPU tensor factor.

    Steps:
      1. Validate the tree against field rules and type constraints.
      2. Evaluate each slot subtree recursively → slot tensor values.
      3. Combine slot values using the mode's combiner.
      4. Apply conditions (for non-gated combiners).
      5. Apply mode direction (fixed policy).
      6. Apply industry and size neutralization.
    """
    # Validate once and reuse the error list
    from .typed_tree import validate_tree

    err_msgs = validate_tree(
        tree, ctx.behavior_field_rules,
    )
    if err_msgs:
        raise ValueError("illegal behavior tree: " + "; ".join(err_msgs))

    tradeable_mask = ctx.tradeable() if tradeable_only else None
    values = _slot_values(tree, ctx, tradeable_mask)
    raw = _combine_slot_values(tree, values, ctx, tradeable_mask)

    if not tree.combiner.startswith("gated"):
        raw = _apply_tree_conditions(raw, tree, ctx, tradeable_mask, gate_fill="nan")

    if apply_mode_direction and tree.direction_policy == "fixed":
        from alpha_gen.behavior_gen.gene import MODE_REGISTRY
        raw = raw * float(MODE_REGISTRY[tree.mode].direction)

    if tradeable_only:
        raw = _apply_mask(raw, tradeable_mask)

    if neutralize_industry:
        raw = industry_neutralize_torch(raw, ctx.industry_codes(), mask=tradeable_mask)

    if neutralize_size:
        size = ctx.get_current(size_field, False)
        size_mask = torch.isfinite(raw) & torch.isfinite(size)
        if tradeable_mask is not None:
            size_mask = size_mask & tradeable_mask
        raw = cross_sectional_residual_torch(raw, size, mask=size_mask)

    return raw


def score_tree_factor_tensor(
    factor: torch.Tensor,
    ctx: BehaviorTorchContext,
    *,
    dates: pd.DatetimeIndex | None = None,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    direction: int | None = None,
) -> FactorScore:
    """Evaluate a tree factor tensor on GPU.

    Delegates to core.torch_backend.evaluate_factor_tensor.
    """
    return evaluate_factor_tensor(
        factor,
        ctx,
        dates=dates,
        ndcg_k=ndcg_k,
        ndcg_top_fraction=ndcg_top_fraction,
        direction=direction,
    )
