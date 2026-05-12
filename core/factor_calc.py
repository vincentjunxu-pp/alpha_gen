from __future__ import annotations

import numpy as np
import pandas as pd

from .gene import TRANSFORM_WINDOWS, FactorGene, validate_gene
from .preprocess import TransformCache
from .utils import cs_resi, cs_zscore, dot_add, dot_div, dot_sub, validate_pivot_format


# ---------------------------------------------------------------------------
# Factor calculation for the structured expression search.
#
# This file converts a legal FactorGene into one factor matrix. Each field may
# first receive one metadata-approved unary transform. The generated expression
# templates are:
#
#   single:     A
#   ratio:      A / B
#   pair_ratio: (A +/- B) / (C +/- D)
#   resi:       residual(A ~ B)
#   ratio_product: (A / B) * (C / D)
#
# Ne means size neutralization. Raw market cap is log-transformed for the size
# control so extreme large caps do not dominate the cross-sectional regression.
# ---------------------------------------------------------------------------


def _clean_factor(factor: pd.DataFrame, dtype: str = "float32") -> pd.DataFrame:
    """Replace infinite values with NaN and preserve pivot-table naming."""

    cleaned = factor.replace([np.inf, -np.inf], np.nan)
    cleaned.index.name = "Datetime"
    cleaned.columns.name = "Contract"
    return cleaned.astype(dtype)


def _safe_divide(left: pd.DataFrame, right: pd.DataFrame, eps: float = 1e-2) -> pd.DataFrame:
    """Element-wise division used by ratio templates.

    If the denominator is too close to zero, the ratio has no stable economic
    meaning, so the result is set to NaN rather than clipped to a large number.
    """

    left, right = left.align(right, join="inner", axis=0)
    left, right = left.align(right, join="inner", axis=1)
    result = dot_div(left, right.where(right.abs() > eps))
    return _clean_factor(result)


def _cross_sectional_residual(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Residual of y regressed on x for each date.

    Each row is treated as one cross-section and regressed on the matching row
    of `x`. The wrapper keeps alpha_gen's stricter alignment and dtype
    conventions in one visible place.
    """

    y, x = y.align(x, join="inner", axis=0)
    y, x = y.align(x, join="inner", axis=1)
    residual = cs_resi(y, x, parallel=True, use_tqdm=False)
    residual = residual.reindex(index=y.index, columns=y.columns)
    return _clean_factor(residual)


def _apply_transform(raw: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Apply one same-contract historical transform without future data."""

    if transform == "current":
        return _clean_factor(raw)
    if transform == "log":
        return _clean_factor(np.log(raw.where(raw > 0)))
    if transform == "zscore":
        return _clean_factor(cs_zscore(raw))

    if transform.endswith("_2q"):
        window = TRANSFORM_WINDOWS["2q"]
    elif transform.endswith("_1y"):
        window = TRANSFORM_WINDOWS["1y"]
    else:
        raise ValueError(f"unknown transform: {transform!r}")

    if transform.startswith("diff_"):
        return _clean_factor(raw.diff(periods=window))
    if transform.startswith("pct_"):
        return _clean_factor(raw.pct_change(periods=window, fill_method=None))
    if transform.startswith("std_"):
        return _clean_factor(raw.rolling(window=window, min_periods=window).std())
    raise ValueError(f"unknown transform: {transform!r}")


def _feature(field: str, transform: str, cache: TransformCache) -> pd.DataFrame:
    """Fetch one field, apply its unary transform, then mask untradeable names."""

    raw = cache.get_current(field, use_log=False)
    if transform == "zscore":
        masked = apply_tradeable_mask(raw, cache.tradeable)
        return _clean_factor(cs_zscore(masked))
    transformed = _apply_transform(raw, transform)
    return apply_tradeable_mask(transformed, cache.tradeable)


def _combine(left: pd.DataFrame, right: pd.DataFrame, op: str) -> pd.DataFrame:
    """Combine two transformed fields with the gene-selected pair operator."""

    left, right = left.align(right, join="inner", axis=0)
    left, right = left.align(right, join="inner", axis=1)
    if op == "+":
        return _clean_factor(dot_add(left, right))
    if op == "-":
        return _clean_factor(dot_sub(left, right))
    raise ValueError(f"unknown pair operator: {op!r}")


def apply_tradeable_mask(factor: pd.DataFrame, tradeable: pd.DataFrame) -> pd.DataFrame:
    """Mask out untradeable names before evaluation or regression."""

    factor, tradeable = factor.align(tradeable, join="inner", axis=0)
    factor, tradeable = factor.align(tradeable, join="inner", axis=1)
    mask = tradeable.replace([np.inf, -np.inf], np.nan).fillna(0).gt(0)
    return _clean_factor(factor.where(mask))


def _log_size_control(size: pd.DataFrame) -> pd.DataFrame:
    """Use log market cap as the size control for neutralization."""

    return _clean_factor(np.log(size.where(size > 0)))


def calculate_raw_factor(gene: FactorGene, cache: TransformCache) -> pd.DataFrame:
    """Calculate the structured expression before size neutralization.

    The gene must be legal under the cache's field rules. Returning the raw
    factor separately makes it easy to debug whether poor scores come from the
    expression itself or from the neutralization step.
    """

    errors = validate_gene(gene, cache.field_rules)
    if errors:
        raise ValueError("illegal gene: " + "; ".join(errors))

    a = _feature(gene.a, gene.a_transform, cache)
    if gene.mode == "single":
        return a

    b = _feature(gene.b, gene.b_transform, cache)
    if gene.mode == "ratio":
        return _safe_divide(a, b)
    if gene.mode == "resi":
        return _cross_sectional_residual(a, b)
    if gene.mode == "ratio_product":
        c = _feature(gene.c, gene.c_transform, cache)
        d = _feature(gene.d, gene.d_transform, cache)
        return _clean_factor(_safe_divide(a, b).mul(_safe_divide(c, d)))

    left = _combine(a, b, gene.left_op)
    c = _feature(gene.c, gene.c_transform, cache)
    d = _feature(gene.d, gene.d_transform, cache)
    right = _combine(c, d, gene.right_op)

    if gene.mode == "pair_ratio":
        return _safe_divide(left, right)

    # validate_gene should already catch this, but keep a defensive error here.
    raise ValueError(f"unknown mode: {gene.mode!r}")


def size_neutralize(
    factor: pd.DataFrame,
    cache: TransformCache,
    *,
    size_field: str = "market_cap",
    use_log_size: bool = True,
    tradeable_only: bool = True,
) -> pd.DataFrame:
    """Neutralize a factor against size at each date.

    The size field is raw market cap by default, so neutralization uses
    `log(market_cap)` unless `use_log_size=False` is explicitly passed.
    """

    size = cache.get_current(size_field, use_log=False)
    if use_log_size:
        size = _log_size_control(size)
    factor = apply_tradeable_mask(factor, cache.tradeable) if tradeable_only else _clean_factor(factor)
    size = apply_tradeable_mask(size, cache.tradeable) if tradeable_only else _clean_factor(size)

    neutralized = _cross_sectional_residual(factor, size)
    neutralized = apply_tradeable_mask(neutralized, cache.tradeable) if tradeable_only else neutralized
    validate_pivot_format(neutralized, require_time_component=True, strict=True)
    return _clean_factor(neutralized)


def calculate_factor(
    gene: FactorGene,
    cache: TransformCache,
    *,
    neutralize_size: bool = True,
    tradeable_only: bool = True,
) -> pd.DataFrame:
    """Calculate the final structured factor matrix for one gene."""

    raw_factor = calculate_raw_factor(gene, cache)
    if not neutralize_size:
        return apply_tradeable_mask(raw_factor, cache.tradeable) if tradeable_only else raw_factor
    return size_neutralize(raw_factor, cache, tradeable_only=tradeable_only)
