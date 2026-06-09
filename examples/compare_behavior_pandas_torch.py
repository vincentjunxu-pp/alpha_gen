from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.behavior_gen.gene import (  # noqa: E402
    BehaviorFieldRule,
    BehaviorGene,
    ConditionGene,
    MODE_REGISTRY,
    SlotGene,
    load_behavior_field_rules,
)
from alpha_gen.behavior_gen.torch_backend import (  # noqa: E402
    BehaviorTorchContext,
    calculate_behavior_factor_tensor,
)
from alpha_gen.core.gene import FieldRule, load_field_rules  # noqa: E402
from alpha_gen.core.preprocess import TransformCache, build_transform_cache, load_panel  # noqa: E402


EPS = 1e-8


def screenshot_gene() -> BehaviorGene:
    """The attention_overreaction/attention_risk gene discussed in the notes."""

    return BehaviorGene(
        mode="attention_overreaction",
        combiner="attention_risk",
        slots={
            "attention_heat": SlotGene(field="AMP5", unary_op="zscore"),
            "fund_support": SlotGene(field="operating_cash_flow_per_share_ttm", unary_op="direction_rank"),
        },
        conditions=(
            ConditionGene(
                field="mf_sm_net_ratio_20d",
                unary_op="direction_zscore",
                condition_op="positive",
                threshold=0.0,
            ),
        ),
        direction_policy="fixed",
    )


def gene_fields(gene: BehaviorGene) -> set[str]:
    fields = {slot.field for slot in gene.slots.values()}
    fields.update(condition.field for condition in gene.conditions)
    return fields


def finite_mask(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).notna()


def row_broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    values = np.broadcast_to(series.to_numpy()[:, None], like.shape)
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype="float64")


def masked_valid(values: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    valid = finite_mask(values)
    if mask is not None:
        valid = valid & mask
    return valid


def cs_zscore(values: pd.DataFrame, mask: pd.DataFrame | None = None, eps: float = EPS) -> pd.DataFrame:
    values = values.astype("float64")
    valid = masked_valid(values, mask)
    count = valid.sum(axis=1).astype("float64")
    mean = values.where(valid, 0.0).sum(axis=1) / count.clip(lower=1.0)
    centered = values.sub(mean, axis=0).where(valid, 0.0)
    variance = (centered * centered).sum(axis=1) / (count - 1.0).clip(lower=1.0)
    std = np.sqrt(variance)
    out = values.sub(mean, axis=0).div(std + eps, axis=0)
    return out.where(valid & row_broadcast(count >= 2.0, values))


def cs_rank_pct(values: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    valid = masked_valid(values.astype("float64"), mask)
    return values.where(valid).rank(axis=1, method="average", pct=True)


def center_rank(values: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    return cs_rank_pct(values, mask=mask) - 0.5


def rolling_ts_zscore(values: pd.DataFrame, window: int, eps: float = EPS) -> pd.DataFrame:
    if window <= 1:
        return values.astype("float64")
    min_periods = max(5, window // 3)
    x = values.astype("float64").replace([np.inf, -np.inf], np.nan)
    count = x.rolling(window=window, min_periods=1).count()
    mean = x.rolling(window=window, min_periods=1).mean()
    std = x.rolling(window=window, min_periods=2).std(ddof=1)
    out = (x - mean) / (std + eps)
    return out.where((count >= min_periods) & x.notna() & (std > 0))


def unique_industries(industry: pd.DataFrame, valid: pd.DataFrame) -> list[object]:
    raw = pd.unique(industry.where(valid).to_numpy().ravel())
    return [value for value in raw if pd.notna(value)]


def industry_zscore(
    values: pd.DataFrame,
    industry: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    eps: float = EPS,
) -> pd.DataFrame:
    values = values.astype("float64")
    valid = masked_valid(values, mask) & industry.notna()
    out = empty_like(values)
    for code in unique_industries(industry, valid):
        group_valid = valid & industry.eq(code)
        count = group_valid.sum(axis=1).astype("float64")
        mean = values.where(group_valid, 0.0).sum(axis=1) / count.clip(lower=1.0)
        centered = values.sub(mean, axis=0).where(group_valid, 0.0)
        variance = (centered * centered).sum(axis=1) / (count - 1.0).clip(lower=1.0)
        std = np.sqrt(variance)
        calc = values.sub(mean, axis=0).div(std + eps, axis=0)
        cond = group_valid & row_broadcast((count >= 2.0) & (std > 0), values)
        out = out.mask(cond, calc)
    return out


def industry_rank_pct(
    values: pd.DataFrame,
    industry: pd.DataFrame,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    valid = masked_valid(values.astype("float64"), mask) & industry.notna()
    out = empty_like(values)
    for code in unique_industries(industry, valid):
        group_valid = valid & industry.eq(code)
        ranks = values.where(group_valid).rank(axis=1, method="average", pct=True)
        out = out.mask(group_valid, ranks)
    return out


def industry_neutralize(
    values: pd.DataFrame,
    industry: pd.DataFrame,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = values.astype("float64")
    valid = masked_valid(values, mask) & industry.notna()
    out = empty_like(values)
    for code in unique_industries(industry, valid):
        group_valid = valid & industry.eq(code)
        count = group_valid.sum(axis=1).astype("float64")
        mean = values.where(group_valid, 0.0).sum(axis=1) / count.clip(lower=1.0)
        calc = values.sub(mean, axis=0)
        cond = group_valid & row_broadcast(count > 0.0, values)
        out = out.mask(cond, calc)
    return out


def cross_sectional_residual(
    y: pd.DataFrame,
    x: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    y = y.astype("float64")
    x = x.astype("float64")
    valid = finite_mask(y) & finite_mask(x)
    if mask is not None:
        valid = valid & mask

    count = valid.sum(axis=1).astype("float64")
    y_mean = y.where(valid, 0.0).sum(axis=1) / count.clip(lower=1.0)
    x_mean = x.where(valid, 0.0).sum(axis=1) / count.clip(lower=1.0)

    y_centered = y.sub(y_mean, axis=0).where(valid, 0.0)
    x_centered = x.sub(x_mean, axis=0).where(valid, 0.0)
    denom = (x_centered * x_centered).sum(axis=1)
    numer = (x_centered * y_centered).sum(axis=1)
    slope = numer.div(denom.where(denom > eps, np.nan)).fillna(0.0)
    intercept = y_mean - slope * x_mean

    residual = y - x.mul(slope, axis=0).sub(-intercept, axis=0)
    demeaned = y.sub(y_mean, axis=0)
    residual = residual.where(row_broadcast(denom > eps, y), demeaned)
    return residual.where(valid & row_broadcast(count >= 3.0, y))


def feature_frame(
    field_name: str,
    unary_op: str,
    cache: TransformCache,
    behavior_rules: Mapping[str, BehaviorFieldRule],
    mask: pd.DataFrame | None,
) -> pd.DataFrame:
    raw = cache.get_current(field_name, False).astype("float64")
    rule = behavior_rules[field_name]
    if unary_op == "current":
        value = raw
    elif unary_op == "rank_pct":
        value = center_rank(raw, mask=mask)
    elif unary_op == "zscore":
        value = cs_zscore(raw, mask=mask)
    elif unary_op == "direction_rank":
        value = center_rank(raw, mask=mask) * float(rule.direction)
    elif unary_op == "direction_zscore":
        value = cs_zscore(raw, mask=mask) * float(rule.direction)
    elif unary_op == "ind_rank_pct":
        if cache.industry is None:
            raise ValueError("industry_rank_pct requires an industry matrix")
        value = industry_rank_pct(raw, cache.industry, mask=mask) - 0.5
    elif unary_op == "ind_zscore":
        if cache.industry is None:
            raise ValueError("industry_zscore requires an industry matrix")
        value = industry_zscore(raw, cache.industry, mask=mask)
    elif unary_op == "ts_zscore_5d":
        value = rolling_ts_zscore(raw, 5)
    elif unary_op == "ts_zscore_20d":
        value = rolling_ts_zscore(raw, 20)
    else:
        raise ValueError(f"unknown unary op: {unary_op!r}")
    return value.where(mask) if mask is not None else value


def condition_mask(
    condition: ConditionGene,
    cache: TransformCache,
    behavior_rules: Mapping[str, BehaviorFieldRule],
    base_mask: pd.DataFrame | None,
) -> pd.DataFrame:
    value = feature_frame(condition.field, condition.unary_op, cache, behavior_rules, base_mask)
    if condition.condition_op == "top_quantile":
        return cs_rank_pct(value, mask=base_mask) >= condition.threshold
    if condition.condition_op == "bottom_quantile":
        return cs_rank_pct(value, mask=base_mask) <= (1.0 - condition.threshold)
    if condition.condition_op == "above_median":
        return cs_rank_pct(value, mask=base_mask) >= 0.5
    if condition.condition_op == "below_median":
        return cs_rank_pct(value, mask=base_mask) < 0.5
    if condition.condition_op == "positive":
        return value > 0
    if condition.condition_op == "negative":
        return value < 0
    raise ValueError(f"unknown condition op: {condition.condition_op!r}")


def apply_conditions(
    raw: pd.DataFrame,
    gene: BehaviorGene,
    cache: TransformCache,
    behavior_rules: Mapping[str, BehaviorFieldRule],
    base_mask: pd.DataFrame | None,
    *,
    gate_fill: str = "zero",
) -> pd.DataFrame:
    if not gene.conditions:
        return raw
    gate = pd.DataFrame(True, index=raw.index, columns=raw.columns)
    for condition in gene.conditions:
        gate = gate & condition_mask(condition, cache, behavior_rules, base_mask)
    if base_mask is not None:
        gate = gate & base_mask
    fill = 0.0 if gate_fill == "zero" else np.nan
    return raw.where(gate, fill)


def sum_existing(values: list[pd.DataFrame], like: pd.DataFrame) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(0.0, index=like.index, columns=like.columns)
    out = values[0]
    for value in values[1:]:
        out = out + value
    return out


def combine_behavior_gene(
    gene: BehaviorGene,
    values: Mapping[str, pd.DataFrame],
    cache: TransformCache,
    behavior_rules: Mapping[str, BehaviorFieldRule],
    mask: pd.DataFrame | None,
) -> pd.DataFrame:
    mode_spec = MODE_REGISTRY[gene.mode]
    ordered = [values[name] for name in mode_spec.slots if name in values]
    if not ordered:
        raise ValueError(f"gene {gene.mode!r} has no slot values")

    if gene.combiner in {"rank_gap", "gated_rank_gap"}:
        raw = ordered[0] - ordered[1]
    elif gene.combiner == "residual_gap":
        raw = cross_sectional_residual(ordered[0], ordered[1], mask=mask)
    elif gene.combiner == "quality_gap":
        raw = values["profit_growth"] - values["cashflow_quality"]
        if "price_reaction" in values:
            raw = raw + 0.25 * values["price_reaction"].abs()
    elif gene.combiner == "crowding_interaction":
        raw = values["growth_anchor"] * values["crowding_signal"]
        if "fund_support" in values:
            raw = raw - values["fund_support"]
    elif gene.combiner in {"confirm", "gated_confirm"}:
        raw = ordered[0] + ordered[1] + ordered[0] * ordered[1]
        for name, value in values.items():
            if name in {"fund_anchor", "flow_confirm", "price_anchor", "price_momentum"}:
                continue
            if name.endswith("control"):
                raw = raw - 0.25 * value.abs()
            else:
                raw = raw + 0.5 * value
    elif gene.combiner == "risk_minus_confirm":
        risk_names = {
            "price_momentum",
            "retail_flow",
            "close_chase",
            "attention_heat",
            "crowding_signal",
            "liquidity_stress",
            "turnover_shock",
        }
        confirm_names = {"large_flow", "flow_confirm", "fund_support", "orderbook_filter"}
        risk_items = [value for name, value in values.items() if name in risk_names]
        confirm_items = [value for name, value in values.items() if name in confirm_names]
        risk = sum_existing(risk_items, cache.label)
        confirm = sum_existing(confirm_items, cache.label)
        if not risk_items or not np.isfinite(risk.to_numpy()).any():
            risk = sum_existing(ordered[:2], cache.label)
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
        raw = apply_conditions(raw, gene, cache, behavior_rules, mask)
    return raw


def pandas_behavior_factor(
    gene: BehaviorGene,
    cache: TransformCache,
    behavior_rules: Mapping[str, BehaviorFieldRule],
    *,
    apply_mode_direction: bool,
    neutralize_size: bool,
    neutralize_industry: bool,
    size_field: str,
    tradeable_only: bool,
) -> pd.DataFrame:
    tradeable_mask = cache.tradeable.astype(bool) if tradeable_only else None
    values = {
        name: feature_frame(slot.field, slot.unary_op, cache, behavior_rules, tradeable_mask)
        for name, slot in gene.slots.items()
    }
    raw = combine_behavior_gene(gene, values, cache, behavior_rules, tradeable_mask)
    if not gene.combiner.startswith("gated"):
        raw = apply_conditions(raw, gene, cache, behavior_rules, tradeable_mask)

    if apply_mode_direction and gene.direction_policy == "fixed":
        raw = raw * float(MODE_REGISTRY[gene.mode].direction)

    if tradeable_only:
        raw = raw.where(tradeable_mask)

    if neutralize_industry:
        if cache.industry is None:
            raise ValueError("neutralize_industry=True requires an industry matrix")
        raw = industry_neutralize(raw, cache.industry, mask=tradeable_mask)

    if neutralize_size:
        size = cache.get_current(size_field, False)
        size_mask = finite_mask(raw) & finite_mask(size)
        if tradeable_mask is not None:
            size_mask = size_mask & tradeable_mask
        raw = cross_sectional_residual(raw, size, mask=size_mask)

    return raw.astype("float32")


def compare_frames(pandas_factor: pd.DataFrame, torch_factor: pd.DataFrame) -> dict[str, float | int]:
    p = pandas_factor.to_numpy(dtype="float64", copy=False).ravel()
    t = torch_factor.to_numpy(dtype="float64", copy=False).ravel()
    p_finite = np.isfinite(p)
    t_finite = np.isfinite(t)
    both = p_finite & t_finite
    nan_agree = np.mean(p_finite == t_finite) if p.size else np.nan

    diff = np.abs(p[both] - t[both]) if both.any() else np.array([], dtype="float64")
    p_both = p[both]
    t_both = t[both]
    pearson = np.corrcoef(p_both, t_both)[0, 1] if both.sum() >= 2 else np.nan
    spearman = pd.Series(p_both).corr(pd.Series(t_both), method="spearman") if both.sum() >= 2 else np.nan
    sign_agree = np.mean(np.sign(p_both) == np.sign(t_both)) if both.any() else np.nan

    daily_spearman: list[float] = []
    for date in pandas_factor.index:
        pair = pd.DataFrame({"pandas": pandas_factor.loc[date], "torch": torch_factor.loc[date]}).replace(
            [np.inf, -np.inf], np.nan
        )
        pair = pair.dropna()
        if len(pair) >= 3:
            corr = pair["pandas"].corr(pair["torch"], method="spearman")
            if pd.notna(corr):
                daily_spearman.append(float(corr))

    return {
        "n_cells": int(p.size),
        "n_both_finite": int(both.sum()),
        "pandas_finite_rate": float(p_finite.mean()) if p.size else np.nan,
        "torch_finite_rate": float(t_finite.mean()) if t.size else np.nan,
        "finite_agreement_rate": float(nan_agree),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "sign_agreement": float(sign_agree),
        "mean_abs_diff": float(diff.mean()) if diff.size else np.nan,
        "median_abs_diff": float(np.quantile(diff, 0.50)) if diff.size else np.nan,
        "p95_abs_diff": float(np.quantile(diff, 0.95)) if diff.size else np.nan,
        "p99_abs_diff": float(np.quantile(diff, 0.99)) if diff.size else np.nan,
        "max_abs_diff": float(diff.max()) if diff.size else np.nan,
        "daily_spearman_mean": float(np.mean(daily_spearman)) if daily_spearman else np.nan,
        "daily_spearman_min": float(np.min(daily_spearman)) if daily_spearman else np.nan,
    }


def print_summary(summary: Mapping[str, float | int]) -> None:
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.10g}")
        else:
            print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare a pandas implementation with the torch behavior factor backend.",
    )
    parser.add_argument("--data", required=True, help="Long-format parquet panel.")
    parser.add_argument(
        "--metadata",
        default=str(
            root / "data" / "metadata" / "production" / "real_behavior_metadata.json"
        ),
        help="Metadata JSON containing field_rules and behavior_field_rules.",
    )
    parser.add_argument("--label-col", default="label_20d")
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--industry-col", default="industry_code")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--apply-mode-direction", action="store_true")
    parser.add_argument("--neutralize-size", action="store_true")
    parser.add_argument("--neutralize-industry", action="store_true")
    parser.add_argument("--include-untradeable", action="store_true")
    parser.add_argument("--cache-on-device", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    meta_path = Path(args.metadata)

    gene = screenshot_gene()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    size_field = str(metadata.get("size_field", "barra_size"))
    needed_fields = gene_fields(gene)
    if args.neutralize_size:
        needed_fields.add(size_field)

    field_rules_all: Mapping[str, FieldRule] = load_field_rules(meta_path)
    missing_rules = sorted(field for field in needed_fields if field not in field_rules_all)
    if missing_rules:
        raise KeyError(f"fields missing from field_rules: {missing_rules}")
    field_rules = {field: field_rules_all[field] for field in sorted(needed_fields)}
    behavior_rules = load_behavior_field_rules(meta_path)

    panel = load_panel(data_path)
    missing_columns = sorted(field for field in needed_fields if field not in panel.columns)
    if missing_columns:
        raise KeyError(f"fields missing from parquet data: {missing_columns}")
    if args.label_col not in panel.columns:
        panel = panel.copy()
        panel[args.label_col] = 0.0
    if args.tradeable_col not in panel.columns:
        panel = panel.copy()
        panel[args.tradeable_col] = 1

    cache = build_transform_cache(
        panel,
        field_rules,
        label_col=args.label_col,
        tradeable_col=args.tradeable_col,
        industry_col=args.industry_col,
        extra_current_fields=[size_field] if args.neutralize_size else None,
    )

    ctx = BehaviorTorchContext(
        cache=cache,
        behavior_field_rules=behavior_rules,
        device=args.device,
        cache_on_device=args.cache_on_device,
    )
    torch_tensor = calculate_behavior_factor_tensor(
        gene,
        ctx,
        apply_mode_direction=args.apply_mode_direction,
        neutralize_size=args.neutralize_size,
        neutralize_industry=args.neutralize_industry,
        size_field=size_field,
        tradeable_only=not args.include_untradeable,
    )
    torch_factor = ctx.tensor_to_frame(torch_tensor)
    pandas_factor = pandas_behavior_factor(
        gene,
        cache,
        behavior_rules,
        apply_mode_direction=args.apply_mode_direction,
        neutralize_size=args.neutralize_size,
        neutralize_industry=args.neutralize_industry,
        size_field=size_field,
        tradeable_only=not args.include_untradeable,
    )

    if args.start_date is not None:
        pandas_factor = pandas_factor.loc[pandas_factor.index >= pd.Timestamp(args.start_date)]
        torch_factor = torch_factor.loc[torch_factor.index >= pd.Timestamp(args.start_date)]
    if args.end_date is not None:
        pandas_factor = pandas_factor.loc[pandas_factor.index <= pd.Timestamp(args.end_date)]
        torch_factor = torch_factor.loc[torch_factor.index <= pd.Timestamp(args.end_date)]

    print("gene:", gene.to_dict())
    print("apply_mode_direction:", args.apply_mode_direction)
    print("neutralize_size:", args.neutralize_size)
    print("neutralize_industry:", args.neutralize_industry)
    print("tradeable_only:", not args.include_untradeable)
    print_summary(compare_frames(pandas_factor, torch_factor))


if __name__ == "__main__":
    main()
