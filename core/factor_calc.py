from __future__ import annotations

import numpy as np
import pandas as pd

from .gene import TRANSFORM_WINDOWS, FactorGene, validate_gene
from .preprocess import TransformCache
from .utils import cs_rank, cs_resi, cs_zscore, dot_add, dot_div, dot_log, dot_sub, validate_pivot_format


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
#   resi_pair:  residual(A ~ B + C)
#   multi_resi: residual(A ~ B + C + D)
#   spread:     A / B - C / D
#   style_composite: combine(rank_score(A), rank_score(B))
#
# Ne means size neutralization. The default control is Barra size exposure.
# Raw market_cap remains a backward-compatible fallback and is signed-log
# transformed only when that legacy field is explicitly selected.
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


def _solve_small_linear_system(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    """Solve a small dense linear system without calling platform LAPACK."""

    a = matrix.astype("float64", copy=True)
    b = vector.astype("float64", copy=True)
    n = len(b)
    for i in range(n):
        a[i, i] += eps

    for col in range(n):
        pivot = col + int(np.argmax(np.abs(a[col:, col])))
        pivot_value = float(a[pivot, col])
        if not np.isfinite(pivot_value) or abs(pivot_value) <= eps:
            return None
        if pivot != col:
            a[[col, pivot]] = a[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]

        for row in range(col + 1, n):
            factor = a[row, col] / a[col, col]
            a[row, col:] -= factor * a[col, col:]
            b[row] -= factor * b[col]

    solution = np.zeros(n, dtype="float64")
    for row in range(n - 1, -1, -1):
        denom = float(a[row, row])
        if not np.isfinite(denom) or abs(denom) <= eps:
            return None
        trailing = 0.0
        for col in range(row + 1, n):
            trailing += float(a[row, col]) * float(solution[col])
        solution[row] = (b[row] - trailing) / denom
    return solution


def _normal_equations(design: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build X'X and X'y with scalar loops for small cross-sectional regressions."""

    n_obs, n_cols = design.shape
    xtx = np.zeros((n_cols, n_cols), dtype="float64")
    xty = np.zeros(n_cols, dtype="float64")
    for obs in range(n_obs):
        y_value = float(y_values[obs])
        for i in range(n_cols):
            xi = float(design[obs, i])
            xty[i] += xi * y_value
            for j in range(n_cols):
                xtx[i, j] += xi * float(design[obs, j])
    return xtx, xty


def _cross_sectional_multi_residual(y: pd.DataFrame, controls: list[pd.DataFrame]) -> pd.DataFrame:
    """Residual of y regressed on multiple controls for each date."""

    common_index = y.index
    common_columns = y.columns
    for control in controls:
        common_index = common_index.intersection(control.index)
        common_columns = common_columns.intersection(control.columns)
    y_aligned = y.reindex(index=common_index, columns=common_columns)
    controls_aligned = [control.reindex(index=common_index, columns=common_columns) for control in controls]

    result = pd.DataFrame(index=common_index, columns=common_columns, dtype="float64")
    min_obs = len(controls) + 2
    for dt in common_index:
        row_data = {"y": y_aligned.loc[dt]}
        for idx, control in enumerate(controls_aligned):
            row_data[f"x{idx}"] = control.loc[dt]
        tmp = pd.DataFrame(row_data).replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < min_obs:
            continue

        y_values = tmp["y"].to_numpy(dtype="float64")
        x_values = tmp[[f"x{idx}" for idx in range(len(controls))]].to_numpy(dtype="float64")
        design = np.column_stack([np.ones(len(tmp), dtype="float64"), x_values])
        xtx, xty = _normal_equations(design, y_values)
        beta = _solve_small_linear_system(xtx, xty)
        if beta is None:
            continue
        fitted = np.zeros(len(tmp), dtype="float64")
        for obs in range(len(tmp)):
            value = 0.0
            for col in range(len(beta)):
                value += float(design[obs, col]) * float(beta[col])
            fitted[obs] = value
        result.loc[dt, tmp.index] = y_values - fitted

    return _clean_factor(result.reindex(index=y.index, columns=y.columns))


def _industry_relative(
    raw: pd.DataFrame,
    industry: pd.DataFrame | None,
    tradeable: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    """Cross-sectional transform within each date's industry groups."""

    if industry is None:
        raise ValueError(f"{method} transform requires cache.industry")

    values = apply_tradeable_mask(raw, tradeable)
    values, industry = values.align(industry, join="inner", axis=0)
    values, industry = values.align(industry, join="inner", axis=1)
    result = pd.DataFrame(index=values.index, columns=values.columns, dtype="float64")

    for dt in values.index:
        tmp = pd.DataFrame({"value": values.loc[dt], "industry": industry.loc[dt]})
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty:
            continue
        for _industry_name, group in tmp.groupby("industry", observed=True):
            group_values = group["value"]
            if method == "ind_rank_pct":
                result.loc[dt, group.index] = group_values.rank(method="average", pct=True)
            elif method == "ind_zscore":
                if len(group_values) < 2:
                    continue
                std = group_values.std()
                if not np.isfinite(std) or std <= 0:
                    continue
                result.loc[dt, group.index] = (group_values - group_values.mean()) / (std + 1e-8)
            else:
                raise ValueError(f"unknown industry relative transform: {method!r}")

    return _clean_factor(result)


def _apply_transform(raw: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Apply one same-contract historical transform without future data."""

    if transform == "current":
        return _clean_factor(raw)
    if transform == "log":
        return _clean_factor(dot_log(raw))
    if transform == "zscore":
        return _clean_factor(cs_zscore(raw))
    if transform == "rank_pct":
        return _clean_factor(cs_rank(raw, pct=True))

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
    raise ValueError(f"unknown transform: {transform!r}")


def _feature(field: str, transform: str, cache: TransformCache) -> pd.DataFrame:
    """Fetch one field, apply its unary transform, then mask untradeable names."""

    raw = cache.get_current(field, use_log=False)
    if transform == "zscore":
        masked = apply_tradeable_mask(raw, cache.tradeable)
        return _clean_factor(cs_zscore(masked))
    if transform == "rank_pct":
        masked = apply_tradeable_mask(raw, cache.tradeable)
        return _clean_factor(cs_rank(masked, pct=True))
    if transform in {"ind_rank_pct", "ind_zscore"}:
        return _industry_relative(raw, cache.industry, cache.tradeable, method=transform)
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


def industry_neutralize(
    factor: pd.DataFrame,
    cache: TransformCache,
    *,
    tradeable_only: bool = True,
) -> pd.DataFrame:
    """Remove industry dummy exposure by demeaning within each date/industry.

    This is used only when the search universe spans all or multiple
    industries. A single-industry universe should skip this step because the
    industry dummy would be constant and would only remove the daily intercept.
    """

    if cache.industry is None:
        raise ValueError("industry neutralization requires an industry matrix in TransformCache")

    factor = apply_tradeable_mask(factor, cache.tradeable) if tradeable_only else _clean_factor(factor)
    factor, industry = factor.align(cache.industry, join="inner", axis=0)
    factor, industry = factor.align(industry, join="inner", axis=1)

    neutralized = pd.DataFrame(index=factor.index, columns=factor.columns, dtype="float64")
    for dt in factor.index:
        tmp = pd.DataFrame({"factor": factor.loc[dt], "industry": industry.loc[dt]})
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty:
            continue
        group_mean = tmp.groupby("industry", observed=True)["factor"].transform("mean")
        neutralized.loc[dt, tmp.index] = tmp["factor"] - group_mean

    neutralized = neutralized.reindex(index=factor.index, columns=factor.columns)
    neutralized = apply_tradeable_mask(neutralized, cache.tradeable) if tradeable_only else neutralized
    validate_pivot_format(neutralized, require_time_component=True, strict=True)
    return _clean_factor(neutralized)


def _log_size_control(size: pd.DataFrame) -> pd.DataFrame:
    """Use signed log1p Barra size as the size control for neutralization."""

    return _clean_factor(dot_log(size))


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
    if gene.mode == "resi_pair":
        c = _feature(gene.c, gene.c_transform, cache)
        return _cross_sectional_residual(a, _combine(b, c, "+"))
    if gene.mode == "multi_resi":
        c = _feature(gene.c, gene.c_transform, cache)
        d = _feature(gene.d, gene.d_transform, cache)
        return _cross_sectional_residual(a, _combine(_combine(b, c, "+"), d, "+"))
    if gene.mode == "spread":
        c = _feature(gene.c, gene.c_transform, cache)
        d = _feature(gene.d, gene.d_transform, cache)
        return _clean_factor(_safe_divide(a, b) - _safe_divide(c, d))
    if gene.mode == "style_composite":
        a_style = (_feature(gene.a, "rank_pct", cache) - 0.5) * cache.field_rules[gene.a].direction
        b_style = (_feature(gene.b, "rank_pct", cache) - 0.5) * cache.field_rules[gene.b].direction
        return _combine(a_style, b_style, "+")

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
    size_field: str = "barra_size",
    use_log_size: bool | None = None,
    tradeable_only: bool = True,
) -> pd.DataFrame:
    """Neutralize a factor against the configured Barra size field at each date.

    Raw market cap is log-scaled by default for backward compatibility. A
    passed Barra size exposure is usually already standardized, so custom
    `size_field` values are used as-is unless `use_log_size=True` is explicit.
    """

    if (size_field, False) not in cache.current:
        raise KeyError(
            f"size neutralization field {size_field!r} is not cached; "
            "include the Barra size field in metadata/field_rules or cache.current"
        )
    size = cache.get_current(size_field, use_log=False)
    if use_log_size is None:
        use_log_size = size_field == "market_cap"
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
    neutralize_industry: bool = False,
    size_field: str = "barra_size",
    use_log_size: bool | None = None,
    tradeable_only: bool = True,
) -> pd.DataFrame:
    """Calculate the final structured factor matrix for one gene."""

    factor = calculate_raw_factor(gene, cache)
    if neutralize_industry:
        factor = industry_neutralize(factor, cache, tradeable_only=tradeable_only)
    if not neutralize_size:
        return apply_tradeable_mask(factor, cache.tradeable) if tradeable_only else factor
    return size_neutralize(
        factor,
        cache,
        size_field=size_field,
        use_log_size=use_log_size,
        tradeable_only=tradeable_only,
    )
