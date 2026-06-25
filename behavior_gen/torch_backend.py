from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from alpha_gen.core.preprocess import TransformCache
from alpha_gen.core.torch_backend import (
    NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY,
    NEUTRALIZED_METRIC_NONE,
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


NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY = "raw_full_barra_industry"
NEUTRALIZATION_SIZE_THEN_INDUSTRY = "size_then_industry"
NEUTRALIZATION_RAW_NONE = "raw_none"
BEHAVIOR_NEUTRALIZATION_MODES = (
    NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    NEUTRALIZATION_SIZE_THEN_INDUSTRY,
    NEUTRALIZATION_RAW_NONE,
)


def validate_neutralization_requirements(
    neutralization_mode: str,
    *,
    barra_style_fields: tuple[str, ...],
    has_industry: bool,
) -> str | None:
    """Return an error message when the neutralization mode is incompatible
    with the provided data, or ``None`` when the configuration is valid."""
    if neutralization_mode == NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY:
        if len(barra_style_fields) != 10:
            return (
                "raw_full_barra_industry requires exactly 10 Barra style fields, "
                f"got {len(barra_style_fields)}"
            )
        if not has_industry:
            return "raw_full_barra_industry requires industry data"
    return None


@dataclass
class BehaviorTorchContext:
    """GPU tensor context for behavior-finance genes.

    Parameters
    ----------
    max_cache_mb:
        Soft ceiling for the on-device field-cache in MiB.  When the cache
        exceeds this size the oldest **non-pinned** entries are evicted
        (FIFO via dict insertion order).  Permanent tensors — label,
        tradeable, industry codes, barra styles — are pinned and never
        evicted.  Set to 0 to disable the limit.
    """

    cache: TransformCache
    behavior_field_rules: Mapping[str, BehaviorFieldRule]
    device: torch.device | str = "auto"
    dtype: torch.dtype = torch.float32
    cache_on_device: bool = True
    max_cache_mb: float = 4096  # 4 GiB — safe for 12 GiB cards with headroom
    barra_style_fields: Sequence[str] | str = ()
    _tensor_cache: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    _pinned_keys: set[tuple[object, ...]] = field(default_factory=set)

    # -- cached byte-count; updated incrementally so _evict_cache() is O(1)
    _cache_bytes: int = field(default=0, init=False, repr=False)

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
        if self.max_cache_mb < 0:
            raise ValueError("max_cache_mb must be ≥ 0")
        self._validate_cache_layout()
        self.date_index = pd.DatetimeIndex(self.cache.label.index)

    def clear_tensor_cache(self) -> None:
        """Release all cached GPU tensors to free device memory.

        Call this before a long validation loop to avoid OOM when the
        training-phase cache already holds several GB of field tensors.
        Field tensors are re-loaded from CPU on demand afterwards.

        Pinned entries (label, tradeable, …) are preserved.
        """
        for key in list(self._tensor_cache):
            if key in self._pinned_keys:
                continue
            self._cache_bytes -= self._tensor_cache[key].element_size() * self._tensor_cache[key].numel()
            del self._tensor_cache[key]
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _pin(self, key: tuple[object, ...]) -> None:
        """Mark *key* as permanent — it will survive cache eviction."""
        self._pinned_keys.add(key)

    # ------------------------------------------------------------------
    # Cache eviction
    # ------------------------------------------------------------------
    def _evict_cache(self) -> None:
        """Drop oldest non-pinned entries while *total* cached bytes exceed
        *max_cache_mb*.  Dict insertion order in Python ≥ 3.7 gives FIFO
        semantics — the earliest loaded field is evicted first.

        Does **not** call ``torch.cuda.empty_cache()`` — per-gene driver
        coalescing happens once in ``evaluate_behavior_gene_on_train``
        after ``del factor``, avoiding 10–50 redundant driver calls per
        gene while the cache is at its ceiling.
        """
        if self.max_cache_mb <= 0:
            return
        max_bytes = int(self.max_cache_mb * 1024 * 1024)
        if self._cache_bytes <= max_bytes:
            return

        for key in list(self._tensor_cache):
            if key in self._pinned_keys:
                continue
            if self._cache_bytes <= max_bytes:
                break
            tensor = self._tensor_cache.pop(key)
            self._cache_bytes -= tensor.element_size() * tensor.numel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
            # Move hit entry to end so FIFO eviction keeps recently-used fields.
            tensor = self._tensor_cache.pop(key)
            self._tensor_cache[key] = tensor
            return tensor
        array = frame.to_numpy(dtype=np.float32, copy=False)
        tensor = torch.as_tensor(array, dtype=self.dtype, device=self.device)
        if self.cache_on_device:
            self._tensor_cache[key] = tensor
            self._cache_bytes += tensor.element_size() * tensor.numel()
            self._evict_cache()
        return tensor

    def get_current(self, field_name: str, use_log: bool = False) -> torch.Tensor:
        source_key = (field_name, use_log)
        if source_key not in self.cache.current:
            raise KeyError(f"field {field_name!r} use_log={use_log} is not cached")
        return self._frame_to_tensor(("current", field_name, use_log), self.cache.current[source_key])

    def label(self) -> torch.Tensor:
        key = ("label",)
        result = self._frame_to_tensor(key, self.cache.label)
        self._pin(key)
        return result

    def tradeable(self) -> torch.Tensor:
        key = ("tradeable_mask",)
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]
        array = self.cache.tradeable.to_numpy(dtype=np.bool_, copy=False)
        tensor = torch.as_tensor(array, dtype=torch.bool, device=self.device)
        if self.cache_on_device:
            self._tensor_cache[key] = tensor
            self._cache_bytes += tensor.element_size() * tensor.numel()
            self._pin(key)
            self._evict_cache()
        return tensor

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
            self._cache_bytes += stacked.element_size() * stacked.numel()
            self._pin(key)
            self._evict_cache()
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
            self._cache_bytes += tensor.element_size() * tensor.numel()
            self._pin(key)
            self._evict_cache()
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


def _rolling_delta(values: torch.Tensor, window: int) -> torch.Tensor:
    """``values[t] - values[t-window]`` — raw period-over-period change."""
    if window < 1:
        raise ValueError("window must be positive")
    shifted = _nan_like(values)
    if window < values.shape[0]:
        shifted[window:] = values[:-window]
    valid = torch.isfinite(values) & torch.isfinite(shifted)
    result = values - shifted
    # first *window* rows have no valid lag
    valid[:window] = False
    return torch.where(valid, result, _nan_like(values))


def _rolling_vol(values: torch.Tensor, window: int, min_periods: int | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Rolling standard deviation (volatility).  Reuses the same prefix-sum
    machinery as :func:`_rolling_ts_zscore`."""
    if window <= 1:
        return torch.zeros_like(values)
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
    variance = (sqs - sums * sums / torch.clamp(counts, min=1.0)) / torch.clamp(counts - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return torch.where((counts >= min_periods) & valid & (std > 0), std, _nan_like(values))


def _rolling_max_drawdown(values: torch.Tensor, window: int, min_periods: int | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Max-drawdown over *window*: ``1 − current / rolling_max``.
    Larger positive → deeper drawdown."""
    if window < 1:
        raise ValueError("window must be positive")
    if min_periods is None:
        min_periods = max(5, window // 3)
    valid = torch.isfinite(values)
    # Use max_pool1d instead of unfold — O(window × T × C) → O(T × C)
    import torch.nn.functional as F  # noqa: E402 (fine at call-site)
    # max_pool1d expects (N, C, L) → (C, 1, T)
    v = values.T.unsqueeze(1)  # (C, 1, T)
    v_pad = F.pad(v, (window - 1, 0), mode="constant", value=float("-inf"))
    rolling_max = F.max_pool1d(v_pad, kernel_size=window, stride=1).squeeze(1).T  # (T, C)
    dd = 1.0 - values / torch.clamp(rolling_max, min=eps)
    enough = rolling_max > eps
    return torch.where(valid & enough, dd, _nan_like(values))


def _decay_linear(values: torch.Tensor, window: int, min_periods: int | None = None) -> torch.Tensor:
    """Decay-linear weighted average over *window* (recency-weighted)."""
    if window < 1:
        raise ValueError("window must be positive")
    if min_periods is None:
        min_periods = window
    valid = torch.isfinite(values)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    valid0 = valid.to(values.dtype)
    # time-weight vector: 1..T as column
    time_weight = torch.arange(1, values.shape[0] + 1, dtype=values.dtype, device=values.device).unsqueeze(1)
    zeros = torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device)
    end = torch.arange(1, values.shape[0] + 1, device=values.device)
    start = torch.clamp(end - window, min=0)

    def _windowed(source: torch.Tensor) -> torch.Tensor:
        prefix = torch.cat([zeros, source.cumsum(dim=0)], dim=0)
        return prefix.index_select(0, end) - prefix.index_select(0, start)

    raw_sum = _windowed(values0)
    raw_count = _windowed(valid0)
    weighted_sum = _windowed(values0 * time_weight) - start.unsqueeze(1).to(values.dtype) * raw_sum
    weighted_count = _windowed(valid0 * time_weight) - start.unsqueeze(1).to(values.dtype) * raw_count
    decayed = weighted_sum / torch.clamp(weighted_count, min=1.0)
    return torch.where(valid & (raw_count >= min_periods), decayed, _nan_like(values))


def _center_rank(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-sectional rank percentile centred at zero.

    NaN cells are ignored during ranking and later filled with 0
    (neutral — no directional exposure).  This prevents NaN from
    propagating through multiplicative combiners like ``confirm`` or
    ``crowding_interaction``.
    """
    ranked = cs_rank_pct_torch(values, mask=mask)
    centered = ranked - 0.5
    return torch.where(torch.isfinite(centered), centered, torch.zeros_like(centered))


def _feature(field_name: str, unary_op: str, ctx: BehaviorTorchContext, mask: torch.Tensor | None) -> torch.Tensor:
    raw = ctx.get_current(field_name, False)
    rule = ctx.behavior_field_rules[field_name]
    if unary_op == "current":
        value = raw
    elif unary_op == "rank_pct":
        value = _center_rank(raw, mask=mask)
    elif unary_op == "zscore":
        value = cs_zscore_torch(raw, mask=mask)
    elif unary_op == "ind_rank_pct":
        value = industry_rank_pct_torch(raw, ctx.industry_codes(), mask=mask) - 0.5
        value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))  # NaN → 0 neutral
    elif unary_op == "ind_zscore":
        value = industry_zscore_torch(raw, ctx.industry_codes(), mask=mask)
    elif unary_op == "ts_zscore_5d":
        value = _rolling_ts_zscore(raw, 5)
    elif unary_op == "ts_zscore_20d":
        value = _rolling_ts_zscore(raw, 20)
    elif unary_op == "ts_delta_5d":
        value = _rolling_delta(raw, 5)
    elif unary_op == "ts_delta_20d":
        value = _rolling_delta(raw, 20)
    elif unary_op == "ts_vol_20d":
        value = _rolling_vol(raw, 20)
    elif unary_op == "ts_max_dd_20d":
        value = _rolling_max_drawdown(raw, 20)
    elif unary_op == "ts_max_dd_60d":
        value = _rolling_max_drawdown(raw, 60)
    elif unary_op == "decay_linear_20d":
        value = _decay_linear(raw, 20)
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
    if condition.condition_op == "extreme_tail":
        rank = cs_rank_pct_torch(value, mask=base_mask)
        return (rank >= 0.97) | (rank <= 0.03)
    if condition.condition_op == "vol_breakout":
        # |value - rolling_mean(20)| > 2 * rolling_std(20)
        valid = torch.isfinite(value)
        x0 = torch.where(valid, value, torch.zeros_like(value))
        count0 = valid.to(value.dtype)
        sum0 = torch.cat([torch.zeros((1, value.shape[1]), dtype=value.dtype, device=value.device), x0.cumsum(dim=0)], dim=0)
        cnt0 = torch.cat([torch.zeros((1, value.shape[1]), dtype=value.dtype, device=value.device), count0.cumsum(dim=0)], dim=0)
        sq0 = torch.cat(
            [torch.zeros((1, value.shape[1]), dtype=value.dtype, device=value.device), (x0 * x0).cumsum(dim=0)],
            dim=0,
        )
        w = 20
        end = torch.arange(1, value.shape[0] + 1, device=value.device)
        start = torch.clamp(end - w, min=0)
        sums = sum0.index_select(0, end) - sum0.index_select(0, start)
        counts = cnt0.index_select(0, end) - cnt0.index_select(0, start)
        sqs = sq0.index_select(0, end) - sq0.index_select(0, start)
        mu = sums / torch.clamp(counts, min=1.0)
        variance = (sqs - sums * sums / torch.clamp(counts, min=1.0)) / torch.clamp(counts - 1.0, min=1.0)
        sigma = torch.sqrt(torch.clamp(variance, min=0.0))
        deviation = torch.abs(value - mu)
        return (counts >= 5) & valid & (sigma > 1e-8) & (deviation > 2.0 * sigma)
    raise ValueError(f"unknown condition op: {condition.condition_op!r}")


def _apply_conditions(
    raw: torch.Tensor,
    gene: BehaviorGene,
    ctx: BehaviorTorchContext,
    base_mask: torch.Tensor | None,
    *,
    gate_fill: str = "nan",
) -> torch.Tensor:
    """Apply condition gating to *raw*.

    Parameters
    ----------
    gate_fill : ``"nan"`` or ``"zero"``
        Fill value for cells that do NOT satisfy the conditions.
        ``"nan"`` (default) removes ungated cells from cross-sectional
        ranking so coverage reflects the true gated universe.
        ``"zero"`` preserves the original behavior but inflates coverage.
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

    if gene.combiner == "rank_gap":
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
    elif gene.combiner == "confirm":
        if len(ordered) < 2:
            raise ValueError(f"{gene.combiner} requires at least two slot values")
        raw = ordered[0] + ordered[1] + ordered[0] * ordered[1]
        # core slots = first two slots in mode definition order that have values.
        # Dynamic instead of hardcoded so new modes get correct weights.
        mode_spec = MODE_REGISTRY[gene.mode]
        ordered_names = [name for name in mode_spec.slots if name in values]
        core_slots = set(ordered_names[:2])
        for name, value in values.items():
            if name in core_slots:
                continue
            if name.endswith("control"):
                raw = raw - 0.25 * value.abs()
            else:
                raw = raw + 0.5 * value
    elif gene.combiner == "risk_minus_confirm":
        risk_names = {
            "price_momentum", "retail_flow", "close_chase", "attention_heat",
            "crowding_signal", "liquidity_stress", "turnover_shock",
            # ── new slots (microstructure / volatility / sentiment) ──
            "spread_stress", "depth_drain", "imbalance_divergence",
            "volatility_shock", "sentiment_energy",
        }
        confirm_names = {
            "large_flow", "flow_confirm", "fund_support", "orderbook_filter",
            # ── new slots ──
            "volume_confirm", "institution_flow", "earnings_accel",
        }
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
        if "price_anchor" in values:
            anchor = values["price_anchor"]
        elif "cost_anchor" in values:
            anchor = values["cost_anchor"]
        else:
            raise ValueError(
                f"anchor_confirm combiner requires a price_anchor or cost_anchor slot, "
                f"got slots {sorted(values)}"
            )
        raw = anchor + values["price_momentum"] + anchor * values["price_momentum"]
        if "flow_confirm" in values:
            raw = raw + 0.5 * values["flow_confirm"]
        if "orderbook_filter" in values:
            raw = raw + 0.5 * values["orderbook_filter"]
        if "fund_support" in values:
            raw = raw + 0.5 * values["fund_support"]
    else:
        raise ValueError(f"unknown combiner: {gene.combiner!r}")

    return raw


def neutralize_behavior_factor_tensor(
    factor: torch.Tensor,
    ctx: BehaviorTorchContext,
    *,
    neutralization_mode: str,
    size_field: str = "barra_size",
    tradeable_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the factor-level neutralization required by one behavior strategy."""

    if neutralization_mode not in BEHAVIOR_NEUTRALIZATION_MODES:
        raise ValueError(
            f"neutralization_mode must be one of {BEHAVIOR_NEUTRALIZATION_MODES}, "
            f"got {neutralization_mode!r}"
        )
    if neutralization_mode in (NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY, NEUTRALIZATION_RAW_NONE):
        return factor

    size = ctx.get_current(size_field, False)
    size_mask = torch.isfinite(factor) & torch.isfinite(size)
    if tradeable_mask is not None:
        size_mask = size_mask & tradeable_mask
    residual = cross_sectional_residual_torch(factor, size, mask=size_mask)
    return industry_neutralize_torch(
        residual,
        ctx.industry_codes(),
        mask=tradeable_mask,
    )


def calculate_behavior_factor_tensor(
    gene: BehaviorGene,
    ctx: BehaviorTorchContext,
    *,
    apply_mode_direction: bool = True,
    neutralization_mode: str | None = None,
    size_field: str = "barra_size",
    tradeable_only: bool = True,
) -> torch.Tensor:
    """Calculate a BehaviorGene into a GPU tensor factor.

    .. important::
       This function returns the **raw factor** (possibly with the
       configured ``neutralization_mode`` applied to the factor tensor
       itself).  Neutralized **metrics** (RIC/RIR on the Barra+industry
       residual) are computed later by
       :func:`score_behavior_factor_tensor` → :func:`evaluate_factor_tensor`
       using a **temporary copy** — the factor tensor returned here is
       never modified by metric-level neutralization.

    Parameters
    ----------
    neutralization_mode : str or None
        One of ``BEHAVIOR_NEUTRALIZATION_MODES``.  Defaults to
        ``NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY`` when ``None``.
    """

    if neutralization_mode is not None and neutralization_mode not in BEHAVIOR_NEUTRALIZATION_MODES:
        raise ValueError(
            f"neutralization_mode must be one of {BEHAVIOR_NEUTRALIZATION_MODES}, "
            f"got {neutralization_mode!r}"
        )
    if neutralization_mode is None:
        neutralization_mode = NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY

    errors = validate_gene(gene, ctx.behavior_field_rules)
    if errors:
        raise ValueError("illegal behavior gene: " + "; ".join(errors))

    tradeable_mask = ctx.tradeable() if tradeable_only else None
    values = _slot_values(gene, ctx, tradeable_mask)
    raw = _combine_behavior_gene(gene, values, ctx, tradeable_mask)
    raw = _apply_conditions(raw, gene, ctx, tradeable_mask)

    if apply_mode_direction and gene.direction_policy == "fixed":
        raw = raw * float(MODE_REGISTRY[gene.mode].direction)

    if tradeable_only:
        raw = _apply_mask(raw, tradeable_mask)

    raw = neutralize_behavior_factor_tensor(
        raw,
        ctx,
        neutralization_mode=neutralization_mode,
        size_field=size_field,
        tradeable_mask=tradeable_mask,
    )

    return raw


def score_behavior_factor_tensor(
    factor: torch.Tensor,
    ctx: BehaviorTorchContext,
    *,
    dates: pd.DatetimeIndex | None = None,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    direction: int | None = None,
    neutralization_mode: str = NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
):
    """Evaluate a behavior factor on GPU using the core tensor evaluator."""

    if neutralization_mode not in BEHAVIOR_NEUTRALIZATION_MODES:
        raise ValueError(
            f"neutralization_mode must be one of {BEHAVIOR_NEUTRALIZATION_MODES}, "
            f"got {neutralization_mode!r}"
        )
    error = validate_neutralization_requirements(
        neutralization_mode,
        barra_style_fields=ctx.barra_style_fields,
        has_industry=ctx.cache.industry is not None,
    )
    if error is not None:
        raise ValueError(error)
    return evaluate_factor_tensor(
        factor,
        ctx,  # duck-typed to the core TorchEvalContext interface
        dates=dates,
        ndcg_k=ndcg_k,
        ndcg_top_fraction=ndcg_top_fraction,
        direction=direction,
        neutralized_metric_mode=(
            NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY
            if neutralization_mode == NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY
            else NEUTRALIZED_METRIC_NONE
        ),
    )
