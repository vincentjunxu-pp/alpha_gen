from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from alpha_gen.core.preprocess import TransformCache
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
    resolve_device,
)

from .gene import BehaviorFieldRule, BehaviorGene, ConditionGene, MODE_REGISTRY, SlotGene, validate_gene


@dataclass
class BehaviorTorchContext:
    """GPU tensor context for behavior-finance genes."""

    cache: TransformCache
    behavior_field_rules: Mapping[str, BehaviorFieldRule]
    device: torch.device | str = "auto"
    dtype: torch.dtype = torch.float32
    cache_on_device: bool = True
    barra_style_fields: Sequence[str] | str = ()
    barra_corr_threshold: float = 0.30
    barra_max_styles: int = 3
    _tensor_cache: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = resolve_device(str(self.device))
        raw_barra_fields = [self.barra_style_fields] if isinstance(self.barra_style_fields, str) else self.barra_style_fields
        self.barra_style_fields = tuple(
            dict.fromkeys(
                text
                for field_name in raw_barra_fields
                if field_name is not None
                for text in [str(field_name).strip()]
                if text
            )
        )
        self._validate_cache_layout()
        self.date_index = pd.DatetimeIndex(self.cache.label.index)

    @property
    def field_rules(self) -> Mapping[str, BehaviorFieldRule]:
        return self.behavior_field_rules

    def _validate_cache_layout(self) -> None:
        label = self.cache.label
        tradeable = self.cache.tradeable
        if not isinstance(label.index, pd.DatetimeIndex):
            raise TypeError("cache.label must be indexed by Datetime")
        if not label.index.equals(tradeable.index) or not label.columns.equals(tradeable.columns):
            raise ValueError("cache.label and cache.tradeable must share index and columns")
        for key, frame in self.cache.current.items():
            if not frame.index.equals(label.index) or not frame.columns.equals(label.columns):
                raise ValueError(f"cache.current[{key!r}] does not match label layout")

    def _frame_to_tensor(self, key: tuple[object, ...], frame: pd.DataFrame) -> torch.Tensor:
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]
        array = frame.to_numpy(dtype=np.float32, copy=False)
        tensor = torch.as_tensor(array, dtype=self.dtype, device=self.device)
        if self.cache_on_device:
            self._tensor_cache[key] = tensor
        return tensor

    def get_current(self, field_name: str, use_log: bool = False) -> torch.Tensor:
        source_key = (field_name, use_log)
        if source_key not in self.cache.current:
            raise KeyError(f"field {field_name!r} use_log={use_log} is not cached")
        return self._frame_to_tensor(("current", field_name, use_log), self.cache.current[source_key])

    def label(self) -> torch.Tensor:
        return self._frame_to_tensor(("label",), self.cache.label)

    def tradeable(self) -> torch.Tensor:
        tradeable = self._frame_to_tensor(("tradeable",), self.cache.tradeable)
        return torch.isfinite(tradeable) & (tradeable > 0)

    def barra_styles(self) -> torch.Tensor:
        label_shape = self.cache.label.shape
        if not self.barra_style_fields:
            return torch.empty((label_shape[0], label_shape[1], 0), dtype=self.dtype, device=self.device)
        key = ("barra_styles", tuple(self.barra_style_fields))
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]
        tensors = [
            torch.nan_to_num(self.get_current(field_name, False), nan=0.0, posinf=0.0, neginf=0.0)
            for field_name in self.barra_style_fields
        ]
        stacked = torch.stack(tensors, dim=2)
        if self.cache_on_device:
            self._tensor_cache[key] = stacked
        return stacked

    def industry_codes(self) -> torch.Tensor:
        if self.cache.industry is None:
            raise ValueError("industry operations require cache.industry")
        key = ("industry_codes",)
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]
        industry = self.cache.industry.reindex(index=self.cache.label.index, columns=self.cache.label.columns)
        codes, _uniques = pd.factorize(industry.to_numpy(dtype=object).ravel(), sort=True, use_na_sentinel=True)
        tensor = torch.as_tensor(codes.reshape(industry.shape), dtype=torch.long, device=self.device)
        if self.cache_on_device:
            self._tensor_cache[key] = tensor
        return tensor

    def date_positions(self, dates: pd.DatetimeIndex | None) -> torch.Tensor | None:
        if dates is None:
            return None
        positions = self.date_index.get_indexer(pd.DatetimeIndex(dates))
        if (positions < 0).any():
            missing = pd.DatetimeIndex(dates)[positions < 0]
            raise KeyError(f"dates not found in cache: {missing[:3].tolist()}")
        return torch.as_tensor(positions, dtype=torch.long, device=self.device)

    def tensor_to_frame(self, values: torch.Tensor) -> pd.DataFrame:
        frame = pd.DataFrame(
            values.detach().cpu().numpy(),
            index=self.cache.label.index,
            columns=self.cache.label.columns,
        )
        frame.index.name = "Datetime"
        frame.columns.name = "Contract"
        return frame.astype("float32")


def _rolling_ts_zscore(values: torch.Tensor, window: int, min_periods: int | None = None, eps: float = 1e-8) -> torch.Tensor:
    if window <= 1:
        return values
    if min_periods is None:
        min_periods = max(5, window // 3)
    valid = torch.isfinite(values)
    x0 = torch.where(valid, values, torch.zeros_like(values))
    count0 = valid.to(values.dtype)
    sum0 = torch.cat([torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device), x0.cumsum(dim=0)], dim=0)
    cnt0 = torch.cat([torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device), count0.cumsum(dim=0)], dim=0)
    sq0 = torch.cat(
        [torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device), (x0 * x0).cumsum(dim=0)],
        dim=0,
    )
    end = torch.arange(1, values.shape[0] + 1, device=values.device)
    start = torch.clamp(end - window, min=0)
    sums = sum0.index_select(0, end) - sum0.index_select(0, start)
    counts = cnt0.index_select(0, end) - cnt0.index_select(0, start)
    sqs = sq0.index_select(0, end) - sq0.index_select(0, start)
    mean = sums / torch.clamp(counts, min=1.0)
    variance = (sqs - sums * sums / torch.clamp(counts, min=1.0)) / torch.clamp(counts - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    zscore = (values - mean) / (std + eps)
    return torch.where((counts >= min_periods) & valid & (std > 0), zscore, _nan_like(values))


def _center_rank(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return cs_rank_pct_torch(values, mask=mask) - 0.5


def _feature(field_name: str, unary_op: str, ctx: BehaviorTorchContext, mask: torch.Tensor | None) -> torch.Tensor:
    raw = ctx.get_current(field_name, False)
    rule = ctx.behavior_field_rules[field_name]
    if unary_op == "current":
        value = raw
    elif unary_op == "rank_pct":
        value = _center_rank(raw, mask=mask)
    elif unary_op == "zscore":
        value = cs_zscore_torch(raw, mask=mask)
    elif unary_op == "direction_rank":
        value = _center_rank(raw, mask=mask) * float(rule.direction)
    elif unary_op == "direction_zscore":
        value = cs_zscore_torch(raw, mask=mask) * float(rule.direction)
    elif unary_op == "ind_rank_pct":
        value = industry_rank_pct_torch(raw, ctx.industry_codes(), mask=mask) - 0.5
    elif unary_op == "ind_zscore":
        value = industry_zscore_torch(raw, ctx.industry_codes(), mask=mask)
    elif unary_op == "ts_zscore_5d":
        value = _rolling_ts_zscore(raw, 5)
    elif unary_op == "ts_zscore_20d":
        value = _rolling_ts_zscore(raw, 20)
    else:
        raise ValueError(f"unknown unary op: {unary_op!r}")
    return _apply_mask(value, mask) if mask is not None else value


def _slot_value(slot: SlotGene, ctx: BehaviorTorchContext, mask: torch.Tensor | None) -> torch.Tensor:
    return _feature(slot.field, slot.unary_op, ctx, mask)


def _zeros_like_context(ctx: BehaviorTorchContext) -> torch.Tensor:
    return torch.zeros_like(ctx.label())


def _sum_existing(values: list[torch.Tensor], ctx: BehaviorTorchContext) -> torch.Tensor:
    if not values:
        return _zeros_like_context(ctx)
    out = values[0]
    for value in values[1:]:
        out = out + value
    return out


def _slot_values(gene: BehaviorGene, ctx: BehaviorTorchContext, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
    return {name: _slot_value(slot, ctx, mask) for name, slot in gene.slots.items()}


def _ordered_slot_values(gene: BehaviorGene, values: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
    mode_spec = MODE_REGISTRY[gene.mode]
    return [values[name] for name in mode_spec.slots if name in values]


def _condition_mask(condition: ConditionGene, ctx: BehaviorTorchContext, base_mask: torch.Tensor | None) -> torch.Tensor:
    value = _feature(condition.field, condition.unary_op, ctx, base_mask)
    if condition.condition_op == "top_quantile":
        return cs_rank_pct_torch(value, mask=base_mask) >= condition.threshold
    if condition.condition_op == "bottom_quantile":
        return cs_rank_pct_torch(value, mask=base_mask) <= (1.0 - condition.threshold)
    if condition.condition_op == "above_median":
        return cs_rank_pct_torch(value, mask=base_mask) >= 0.5
    if condition.condition_op == "below_median":
        return cs_rank_pct_torch(value, mask=base_mask) < 0.5
    if condition.condition_op == "positive":
        return value > 0
    if condition.condition_op == "negative":
        return value < 0
    raise ValueError(f"unknown condition op: {condition.condition_op!r}")


def _apply_conditions(
    raw: torch.Tensor,
    gene: BehaviorGene,
    ctx: BehaviorTorchContext,
    base_mask: torch.Tensor | None,
    *,
    gate_fill: str = "zero",
) -> torch.Tensor:
    """Apply condition gating to *raw*.

    Parameters
    ----------
    gate_fill : ``"zero"`` or ``"nan"``
        Fill value for cells that do NOT satisfy the conditions.
        ``"zero"`` (default) preserves the original behavior but inflates
        coverage.  ``"nan"`` removes ungated cells from cross-sectional
        ranking and is recommended for factor discovery.
    """
    if not gene.conditions:
        return raw
    gate = torch.ones_like(raw, dtype=torch.bool)
    for condition in gene.conditions:
        gate = gate & _condition_mask(condition, ctx, base_mask)
    if base_mask is not None:
        gate = gate & base_mask
    fill_value = (
        torch.zeros_like(raw)
        if gate_fill == "zero"
        else torch.full_like(raw, float("nan"))
    )
    return torch.where(gate, raw, fill_value)


def _combine_behavior_gene(gene: BehaviorGene, values: Mapping[str, torch.Tensor], ctx: BehaviorTorchContext, mask: torch.Tensor | None) -> torch.Tensor:
    ordered = _ordered_slot_values(gene, values)
    if not ordered:
        raise ValueError(f"gene {gene.mode!r} has no slot values")

    if gene.combiner in {"rank_gap", "gated_rank_gap"}:
        if len(ordered) < 2:
            raise ValueError(f"{gene.combiner} requires at least two slot values")
        raw = ordered[0] - ordered[1]
    elif gene.combiner == "residual_gap":
        if len(ordered) < 2:
            raise ValueError("residual_gap requires at least two slot values")
        raw = cross_sectional_residual_torch(ordered[0], ordered[1], mask=mask)
    elif gene.combiner == "quality_gap":
        raw = values["profit_growth"] - values["cashflow_quality"]
        if "price_reaction" in values:
            raw = raw + 0.25 * values["price_reaction"].abs()
    elif gene.combiner == "crowding_interaction":
        raw = values["growth_anchor"] * values["crowding_signal"]
        if "fund_support" in values:
            raw = raw - 0.5*values["fund_support"]
    elif gene.combiner in {"confirm", "gated_confirm"}:
        if len(ordered) < 2:
            raise ValueError(f"{gene.combiner} requires at least two slot values")
        raw = ordered[0] + ordered[1] + ordered[0] * ordered[1]
        for name, value in values.items():
            if name in {"fund_anchor", "flow_confirm", "price_anchor", "price_momentum"}:
                continue
            if name.endswith("control"):
                raw = raw - 0.25 * value.abs()
            else:
                raw = raw + 0.5 * value
    elif gene.combiner == "risk_minus_confirm":
        risk_names = {"price_momentum", "retail_flow", "close_chase", "attention_heat", "crowding_signal", "liquidity_stress", "turnover_shock"}
        confirm_names = {"large_flow", "flow_confirm", "fund_support", "orderbook_filter"}
        risk_items = [value for name, value in values.items() if name in risk_names]
        confirm_items = [value for name, value in values.items() if name in confirm_names]
        risk = _sum_existing(risk_items, ctx)
        confirm = _sum_existing(confirm_items, ctx)
        if not risk_items or not torch.isfinite(risk).any():
            risk = _sum_existing(ordered[:2], ctx)
        raw = risk - confirm
    elif gene.combiner == "panic_reversal":
        raw = values["fund_anchor"] * values["drawdown"] - values["sell_pressure"]
        if "orderbook_filter" in values:
            raw = raw + values["orderbook_filter"]
    elif gene.combiner == "attention_risk":
        raw = values["attention_heat"]
        if "price_momentum" in values:
            raw = raw + values["price_momentum"]
        if "fund_support" in values:
            raw = raw - values["fund_support"]
    elif gene.combiner == "orderbook_intent":
        raw = values["orderbook_pressure"]
        if "liquidity_stress" in values:
            raw = raw - values["liquidity_stress"]
        if "price_reaction" in values:
            raw = raw - 0.25 * values["price_reaction"].abs()
    elif gene.combiner == "liquidity_gap":
        raw = values["liquidity_stress"] - values["turnover_shock"]
        if "flow_confirm" in values:
            raw = raw - values["flow_confirm"]
    elif gene.combiner == "anchor_confirm":
        anchor = values["price_anchor"] if "price_anchor" in values else values["cost_anchor"]
        raw = anchor + values["price_momentum"] + anchor * values["price_momentum"]
        if "flow_confirm" in values:
            raw = raw + 0.5 * values["flow_confirm"]
        if "orderbook_filter" in values:
            raw = raw + 0.5 * values["orderbook_filter"]
        if "fund_support" in values:
            raw = raw + 0.5 * values["fund_support"]
    else:
        raise ValueError(f"unknown combiner: {gene.combiner!r}")

    if gene.combiner.startswith("gated"):
        raw = _apply_conditions(raw, gene, ctx, mask)
    return raw


def calculate_behavior_factor_tensor(
    gene: BehaviorGene,
    ctx: BehaviorTorchContext,
    *,
    apply_mode_direction: bool = True,
    neutralize_size: bool = True,
    neutralize_industry: bool = False,
    size_field: str = "barra_size",
    tradeable_only: bool = True,
) -> torch.Tensor:
    """Calculate a BehaviorGene into a GPU tensor factor."""

    errors = validate_gene(gene, ctx.behavior_field_rules)
    if errors:
        raise ValueError("illegal behavior gene: " + "; ".join(errors))

    tradeable_mask = ctx.tradeable() if tradeable_only else None
    values = _slot_values(gene, ctx, tradeable_mask)
    raw = _combine_behavior_gene(gene, values, ctx, tradeable_mask)
    raw = _apply_conditions(raw, gene, ctx, tradeable_mask) if not gene.combiner.startswith("gated") else raw

    if apply_mode_direction and gene.direction_policy == "fixed":
        raw = raw * float(MODE_REGISTRY[gene.mode].direction)

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


def score_behavior_factor_tensor(
    factor: torch.Tensor,
    ctx: BehaviorTorchContext,
    *,
    dates: pd.DatetimeIndex | None = None,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    direction: int | None = None,
):
    """Evaluate a behavior factor on GPU using the core tensor evaluator."""

    return evaluate_factor_tensor(
        factor,
        ctx,  # duck-typed to the core TorchEvalContext interface
        dates=dates,
        ndcg_k=ndcg_k,
        ndcg_top_fraction=ndcg_top_fraction,
        direction=direction,
    )
