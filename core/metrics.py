from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Evaluation metrics used by the multi-objective genetic algorithm.
#
# The search now optimizes three maximized objectives:
#   1. raw RankICIR        -> raw cross-sectional monotonicity stability
#   2. raw NDCG@k          -> long-leg/top-group performance
#   3. neutralized RankICIR -> robustness after dynamic Barra style stripping
#
# This module does not know about genes or mutation. It only evaluates one
# already-calculated factor matrix against the future-return label matrix.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorScore:
    """Evaluation result for one factor on one date range."""

    mean_rank_ic: float
    abs_rank_ic: float
    rank_ic_ir: float
    ic_win_rate: float
    ndcg_at_k: float
    direction: int
    n_ic_obs: int
    coverage: float
    neutralized_icir: float = 0.0
    neutralized_mean_rank_ic: float = 0.0
    neutralized_abs_rank_ic: float = 0.0
    neutralized_ic_win_rate: float = 0.0
    neutralized_n_ic_obs: int = 0
    barra_max_abs_corr: float = 0.0
    barra_selected_count: int = 0
    barra_selected_styles: tuple[str, ...] = ()

    @property
    def objectives(self) -> tuple[float, float, float]:
        """The three values NSGA-II will maximize."""

        return (self.rank_ic_ir, self.ndcg_at_k, self.neutralized_icir)

    def to_dict(self) -> dict[str, float | int | str]:
        """Plain dict for CSV/JSON logging."""

        return {
            "mean_rank_ic": self.mean_rank_ic,
            "abs_rank_ic": self.abs_rank_ic,
            "rank_ic_ir": self.rank_ic_ir,
            "ic_win_rate": self.ic_win_rate,
            "ndcg_at_k": self.ndcg_at_k,
            "direction": self.direction,
            "n_ic_obs": self.n_ic_obs,
            "coverage": self.coverage,
            "neutralized_icir": self.neutralized_icir,
            "neutralized_mean_rank_ic": self.neutralized_mean_rank_ic,
            "neutralized_abs_rank_ic": self.neutralized_abs_rank_ic,
            "neutralized_ic_win_rate": self.neutralized_ic_win_rate,
            "neutralized_n_ic_obs": self.neutralized_n_ic_obs,
            "barra_max_abs_corr": self.barra_max_abs_corr,
            "barra_selected_count": self.barra_selected_count,
            "barra_selected_styles": ",".join(self.barra_selected_styles),
        }


def _tradeable_to_mask(tradeable: pd.DataFrame) -> pd.DataFrame:
    """Convert a tradeable matrix to a strict boolean mask."""

    return tradeable.replace([np.inf, -np.inf], np.nan).fillna(0).gt(0)


def align_for_evaluation(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    tradeable: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align factor and label matrices using alpha_factory evaluation semantics."""

    factor, label = factor.align(label, join="inner", axis=0)
    factor, label = factor.align(label, join="inner", axis=1)

    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        factor = factor.loc[factor.index.intersection(dates)]
        label = label.loc[label.index.intersection(dates)]

    if tradeable is not None:
        tradeable = tradeable.reindex(index=factor.index, columns=factor.columns)
        mask = _tradeable_to_mask(tradeable)
        label = label.where(mask)

    return factor, label


def _daily_corr_series(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    method: str,
    name: str,
    min_cross_section_size: int = 3,
) -> pd.Series:
    """Daily cross-sectional correlation series.

    Spearman is implemented by ranking each row and then calculating row-wise
    Pearson correlation locally. This avoids scipy edge-case crashes on tiny
    cross-sections while keeping factor and label ranks on the same universe.
    """

    if min_cross_section_size < 2:
        raise ValueError("min_cross_section_size must be at least 2")

    factor_mtx = factor.replace([np.inf, -np.inf], np.nan)
    label_mtx = label.replace([np.inf, -np.inf], np.nan)
    factor_mtx = factor_mtx.align(label_mtx, join="right")[0].dropna(axis=0, how="all")
    label_mtx = label_mtx.reindex(index=factor_mtx.index, columns=factor_mtx.columns)

    common_counts = (factor_mtx.notna() & label_mtx.notna()).sum(axis=1)
    eligible_dates = common_counts[common_counts >= min_cross_section_size].index

    valid_common = factor_mtx.notna() & label_mtx.notna()
    if method == "spearman":
        left = factor_mtx.where(valid_common).rank(axis=1, method="average")
        right = label_mtx.where(valid_common).rank(axis=1, method="average")
    elif method == "pearson":
        left = factor_mtx
        right = label_mtx
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    left = left.loc[eligible_dates]
    right = right.loc[eligible_dates]
    valid = left.notna() & right.notna()
    n = valid.sum(axis=1).astype(float)

    left_common = left.where(valid)
    right_common = right.where(valid)
    left_centered = left_common.sub(left_common.mean(axis=1), axis=0).where(valid, 0.0)
    right_centered = right_common.sub(right_common.mean(axis=1), axis=0).where(valid, 0.0)

    cov = (left_centered * right_centered).sum(axis=1)
    var_left = (left_centered * left_centered).sum(axis=1)
    var_right = (right_centered * right_centered).sum(axis=1)
    denom = np.sqrt(var_left * var_right)
    corr = cov.div(denom.where((denom > 0) & (n >= min_cross_section_size))).dropna()
    corr.name = name
    return corr


def daily_rank_ic(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    min_cross_section_size: int = 3,
) -> pd.Series:
    """Spearman RankIC for every date, computed on common valid names."""

    return _daily_corr_series(
        factor,
        label,
        method="spearman",
        name="rank_ic",
        min_cross_section_size=min_cross_section_size,
    )


def _direction_from_ic(ic_series: pd.Series) -> int:
    """Choose factor direction from mean RankIC."""

    if ic_series.empty:
        return 1
    return 1 if ic_series.mean() >= 0 else -1


def _coverage(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    tradeable: pd.DataFrame | None = None,
) -> float:
    """Average coverage, matching alpha_factory FactorQualityAnalyzer."""

    factor = factor.replace([np.inf, -np.inf], np.nan)
    if tradeable is not None:
        tradeable = tradeable.reindex(index=factor.index, columns=factor.columns)
        mask = _tradeable_to_mask(tradeable)
        denom = mask.sum(axis=1).replace(0, np.nan)
        cov = factor.notna().where(mask).sum(axis=1).div(denom)
    else:
        denom = label.notna().sum(axis=1).replace(0, np.nan)
        cov = factor.notna().sum(axis=1).div(denom)
    return float(cov.mean()) if cov.notna().any() else 0.0


def _row_relevance(label_row: pd.Series, n_groups: int) -> pd.Series:
    """Convert future returns into non-negative relevance scores.

    Higher realized returns receive higher relevance. We use quantile groups
    instead of raw returns because NDCG in the report is group-ranking oriented.
    """

    values = label_row.dropna()
    if len(values) < 2:
        return pd.Series(dtype="float64")

    groups = min(n_groups, len(values))
    ranks = values.rank(method="first")
    try:
        relevance = pd.qcut(ranks, q=groups, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(dtype="float64")

    return pd.Series(relevance.astype(float), index=values.index)


def _ndcg_one_date(factor_row: pd.Series, label_row: pd.Series, k: int, n_groups: int) -> float:
    """NDCG@k for one cross-section."""

    relevance = _row_relevance(label_row, n_groups=n_groups)
    if relevance.empty:
        return np.nan

    scores = factor_row.reindex(relevance.index).dropna()
    relevance = relevance.reindex(scores.index).dropna()
    scores = scores.reindex(relevance.index)

    if len(scores) == 0:
        return np.nan

    k_eff = min(k, len(scores))
    predicted_order = scores.sort_values(ascending=False).index[:k_eff]
    predicted_rel = relevance.loc[predicted_order].to_numpy(dtype=float)

    ideal_rel = np.sort(relevance.to_numpy(dtype=float))[::-1][:k_eff]
    discounts = np.log2(np.arange(2, k_eff + 2, dtype=float))

    dcg = np.sum((np.power(2.0, predicted_rel) - 1.0) / discounts)
    idcg = np.sum((np.power(2.0, ideal_rel) - 1.0) / discounts)
    if idcg <= 0:
        return np.nan
    return float(dcg / idcg)


def ndcg_at_k(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    k: int | None = None,
    top_fraction: float = 0.10,
    n_groups: int = 10,
) -> float:
    """Average NDCG@k across dates.

    If k is not specified, use roughly the top decile of the available universe.
    This mirrors the report's stock-pool setting while still working for small
    local mock universes.
    """

    values: list[float] = []
    for dt in factor.index:
        factor_row = factor.loc[dt]
        label_row = label.loc[dt]
        available = (factor_row.notna() & label_row.notna()).sum()
        if available < 2:
            continue

        k_eff = k if k is not None else max(1, int(np.ceil(available * top_fraction)))
        score = _ndcg_one_date(factor_row, label_row, k=int(k_eff), n_groups=n_groups)
        if np.isfinite(score):
            values.append(score)

    return float(np.mean(values)) if values else 0.0


def evaluate_factor(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    tradeable: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    n_groups: int = 10,
    direction: int | None = None,
    min_cross_section_size: int = 3,
) -> FactorScore:
    """Evaluate one factor on a date range.

    Direction handling is important. A negative-IC factor can still be useful
    after multiplying by -1. We therefore choose the direction from mean RankIC,
    then compute ICIR, IC win rate and NDCG on the direction-adjusted factor.
    """

    if direction is not None and direction not in {-1, 1}:
        raise ValueError("direction must be -1, 1, or None")

    tradeable_eval = None
    if tradeable is not None:
        tradeable_eval = tradeable.reindex(index=factor.index, columns=factor.columns)
        if dates is not None:
            dates_index = pd.DatetimeIndex(dates)
            tradeable_eval = tradeable_eval.loc[tradeable_eval.index.intersection(dates_index)]

    factor_eval, label_eval = align_for_evaluation(
        factor=factor,
        label=label,
        tradeable=tradeable,
        dates=dates,
    )

    ic_series = daily_rank_ic(factor_eval, label_eval, min_cross_section_size=min_cross_section_size)
    if ic_series.empty:
        return FactorScore(
            mean_rank_ic=0.0,
            abs_rank_ic=0.0,
            rank_ic_ir=0.0,
            ic_win_rate=0.0,
            ndcg_at_k=0.0,
            direction=1,
            n_ic_obs=0,
            coverage=_coverage(factor_eval, label_eval, tradeable_eval),
            neutralized_icir=0.0,
            neutralized_mean_rank_ic=0.0,
            neutralized_abs_rank_ic=0.0,
            neutralized_ic_win_rate=0.0,
            neutralized_n_ic_obs=0,
        )

    direction = int(direction) if direction is not None else _direction_from_ic(ic_series)
    oriented_ic = ic_series * direction
    oriented_factor = factor_eval * direction

    mean_ic = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1))

    # ICIR is reported after choosing the usable direction. The raw signed IC is
    # still kept in mean_rank_ic, while rank_ic_ir is suitable for later greedy
    # factor selection and weighting.
    rank_ic_ir = float(oriented_ic.mean() / ic_std) if ic_std > 0 else 0.0

    ic_win_rate = float((oriented_ic > 0).mean())

    # CPU evaluator has no Barra tensor context, so it cannot honestly compute
    # neutralized ICIR. Keep the neutralized objective at 0.0 instead of copying
    # raw ICIR; otherwise NSGA-II would treat an unevaluated robustness metric
    # as if it had passed dynamic Barra stripping.
    neutralized_icir = 0.0

    return FactorScore(
        mean_rank_ic=mean_ic,
        abs_rank_ic=abs(mean_ic),
        rank_ic_ir=rank_ic_ir,
        ic_win_rate=ic_win_rate,
        ndcg_at_k=ndcg_at_k(
            oriented_factor,
            label_eval,
            k=ndcg_k,
            top_fraction=ndcg_top_fraction,
            n_groups=n_groups,
        ),
        direction=direction,
        n_ic_obs=int(len(ic_series)),
        coverage=_coverage(factor_eval, label_eval, tradeable_eval),
        neutralized_icir=neutralized_icir,
        neutralized_mean_rank_ic=0.0,
        neutralized_abs_rank_ic=0.0,
        neutralized_ic_win_rate=0.0,
        neutralized_n_ic_obs=0,
    )



def top_group_excess_return(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    tradeable: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
    direction: int = 1,
    top_fraction: float = 0.10,
    label_horizon: int = 20,
    rebalance_freq: int | None = None,
    annualization_days: int = 244,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    stamp_tax_rate: float = 0.001,
) -> float:
    """Annualized top-group return from the provided label matrix.

    Kept for compatibility with older callers. The calculation follows
    alpha_factory's group PnL semantics, but uses the provided forward-return
    label directly. By default the rebalance frequency equals the label horizon,
    so a 20-day label is evaluated on non-overlapping 20-trading-day samples.
    """

    n_groups = max(2, int(round(1.0 / top_fraction))) if 0 < top_fraction <= 1 else 5
    return top_group_annual_return_from_label(
        factor=factor,
        label=label,
        tradeable=tradeable,
        dates=dates,
        direction=direction,
        n_groups=n_groups,
        label_horizon=label_horizon,
        rebalance_freq=rebalance_freq,
        annualization_days=annualization_days,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        stamp_tax_rate=stamp_tax_rate,
    )


def factor_group_pnl(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    tradeable: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
    direction: int = 1,
    n_groups: int = 5,
    label_horizon: int = 20,
    rebalance_freq: int | None = None,
    annualization_days: int = 244,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    stamp_tax_rate: float = 0.001,
) -> dict[str, object]:
    """Group PnL metrics using alpha_factory's grouping semantics.

    `label` is treated as the already prepared future-return matrix. The PnL
    dates are sampled by `rebalance_freq`, which defaults to `label_horizon`.
    This keeps a 20-day label from being evaluated as overlapping daily PnL.
    """

    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")
    if label_horizon <= 0:
        raise ValueError("label_horizon must be positive")
    if rebalance_freq is None:
        rebalance_freq = label_horizon
    if rebalance_freq <= 0:
        raise ValueError("rebalance_freq must be positive")
    if commission_rate < 0 or slippage_rate < 0 or stamp_tax_rate < 0:
        raise ValueError("transaction cost rates must be non-negative")

    round_trip_cost = 2.0 * (commission_rate + slippage_rate) + stamp_tax_rate

    factor_eval, rtn_sampled = factor.align(label, join="inner", axis=0)
    factor_eval, rtn_sampled = factor_eval.align(rtn_sampled, join="inner", axis=1)
    factor_eval = factor_eval.replace([np.inf, -np.inf], np.nan) * int(direction)
    rtn_sampled = rtn_sampled.replace([np.inf, -np.inf], np.nan)
    if tradeable is not None:
        tradeable = tradeable.reindex(index=rtn_sampled.index, columns=rtn_sampled.columns)
        tradeable_mask = _tradeable_to_mask(tradeable)
        factor_eval = factor_eval.where(tradeable_mask, other=np.nan)
        rtn_sampled = rtn_sampled.where(tradeable_mask, other=np.nan)
    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        sampled_dates = rtn_sampled.index.intersection(dates)
        factor_eval = factor_eval.loc[sampled_dates]
        rtn_sampled = rtn_sampled.loc[sampled_dates]

    if rebalance_freq > 1 and len(rtn_sampled.index) > 0:
        sampled_dates = rtn_sampled.index[::rebalance_freq]
        factor_eval = factor_eval.loc[sampled_dates]
        rtn_sampled = rtn_sampled.loc[sampled_dates]

    factor_sampled = factor_eval.align(rtn_sampled, join="right")[0].dropna(axis=0, how="all")

    if factor_sampled.empty:
        empty = pd.DataFrame()
        return {
            "label_horizon": int(label_horizon),
            "rebalance_freq": int(rebalance_freq),
            "n_rebalance_obs": int(len(rtn_sampled.index)),
            "round_trip_cost": round_trip_cost,
            "grouped_pnl_df": empty,
            "grouped_pnl_cumsum_df": empty,
            "pnl_long": pd.Series(dtype="float64"),
            "pnl_short": pd.Series(dtype="float64"),
            "pnl_longshort": pd.Series(dtype="float64"),
            "pnl_long_ann": 0.0,
            "pnl_short_ann": 0.0,
            "pnl_longshort_ann": 0.0,
            "long_turnover": 0.0,
            "short_turnover": 0.0,
            "longshort_turnover": 0.0,
        }

    group_labels = (
        factor_sampled.rank(axis=1, pct=True, method="first")
        .stack()
        .groupby(level=0)
        .transform(lambda x: pd.qcut(x, n_groups, labels=False, duplicates="drop").values)
        .unstack()
    )
    group_labels = group_labels.align(rtn_sampled, join="right")[0].dropna(axis=0, how="all")

    periods_per_year = annualization_days / rebalance_freq
    result: dict[str, object] = {
        "label_horizon": int(label_horizon),
        "rebalance_freq": int(rebalance_freq),
        "n_rebalance_obs": int(len(rtn_sampled.index)),
        "round_trip_cost": round_trip_cost,
    }
    pnl_dict: dict[str, pd.Series] = {}
    gross_pnl_dict: dict[str, pd.Series] = {}
    group_turnover_dict: dict[str, pd.Series] = {}
    group_cost_dict: dict[str, pd.Series] = {}
    for group in range(n_groups):
        group_mask = group_labels == group
        group_returns = rtn_sampled.where(group_mask, other=np.nan)
        gross_group_return = group_returns.mean(axis=1).dropna()

        group_positions = group_mask.astype(int)
        group_turnover = group_positions.diff().abs().sum(axis=1) / (2 * group_positions.sum(axis=1))
        group_turnover = group_turnover.replace(np.inf, np.nan).reindex(gross_group_return.index).fillna(0.0)
        group_cost = group_turnover * round_trip_cost
        net_group_return = gross_group_return - group_cost

        gross_pnl_dict[f"pnl_group{group}"] = gross_group_return.copy()
        group_turnover_dict[f"turnover_group{group}"] = group_turnover.copy()
        group_cost_dict[f"cost_group{group}"] = group_cost.copy()
        pnl_dict[f"pnl_group{group}"] = net_group_return.copy()
        result[f"pnl_gross_mean_group{group}"] = float(gross_group_return.mean()) if not gross_group_return.empty else 0.0
        result[f"pnl_mean_group{group}"] = float(net_group_return.mean()) if not net_group_return.empty else 0.0
        result[f"turnover_group{group}"] = float(group_turnover.mean()) if not group_turnover.empty else 0.0
        result[f"cost_mean_group{group}"] = float(group_cost.mean()) if not group_cost.empty else 0.0

    grouped_pnl_df = pd.DataFrame(pnl_dict)
    grouped_gross_pnl_df = pd.DataFrame(gross_pnl_dict)
    grouped_turnover_df = pd.DataFrame(group_turnover_dict)
    grouped_cost_df = pd.DataFrame(group_cost_dict)
    result["grouped_pnl_df"] = grouped_pnl_df
    result["grouped_gross_pnl_df"] = grouped_gross_pnl_df
    result["grouped_turnover_df"] = grouped_turnover_df
    result["grouped_cost_df"] = grouped_cost_df
    result["grouped_pnl_cumsum_df"] = grouped_pnl_df.cumsum()
    long_group = f"pnl_group{n_groups - 1}"
    short_group = "pnl_group0"
    result["pnl_long"] = grouped_pnl_df.get(long_group, pd.Series(dtype="float64")).copy()
    result["pnl_short"] = grouped_pnl_df.get(short_group, pd.Series(dtype="float64")).copy()
    result["pnl_long_gross"] = grouped_gross_pnl_df.get(long_group, pd.Series(dtype="float64")).copy()
    result["pnl_short_gross"] = grouped_gross_pnl_df.get(short_group, pd.Series(dtype="float64")).copy()
    result["pnl_longshort_gross"] = result["pnl_long_gross"] - result["pnl_short_gross"]
    long_cost = grouped_cost_df.get(f"cost_group{n_groups - 1}", pd.Series(dtype="float64"))
    short_cost = grouped_cost_df.get("cost_group0", pd.Series(dtype="float64"))
    longshort_cost = long_cost.reindex(result["pnl_longshort_gross"].index).fillna(0.0)
    longshort_cost = longshort_cost + short_cost.reindex(result["pnl_longshort_gross"].index).fillna(0.0)
    result["pnl_longshort"] = result["pnl_longshort_gross"] - longshort_cost
    benchmark_return = rtn_sampled.mean(axis=1).dropna()
    result["benchmark_return"] = benchmark_return.copy()
    result["benchmark_return_mean"] = float(benchmark_return.mean()) if not benchmark_return.empty else 0.0
    result["benchmark_return_ann"] = float(result["benchmark_return_mean"] * periods_per_year)

    for pnl_type in ["pnl_long", "pnl_short", "pnl_longshort"]:
        pnl_series = result[pnl_type]
        assert isinstance(pnl_series, pd.Series)
        pnl_name = pnl_type.split("_")[1]
        pnl_mean = pnl_series.mean()
        pnl_std = pnl_series.std()
        result[f"{pnl_type}_mean"] = float(pnl_mean) if np.isfinite(pnl_mean) else 0.0
        result[f"{pnl_type}_ann"] = float(result[f"{pnl_type}_mean"] * periods_per_year)
        result[f"{pnl_type}_cumsum"] = pnl_series.cumsum()
        raw_sharpe = float(pnl_mean / pnl_std * np.sqrt(periods_per_year)) if pnl_std > 0 else 0.0
        result[f"{pnl_name}_raw_sharpe"] = raw_sharpe

        if pnl_type == "pnl_longshort":
            # Long-short is already market-relative by construction.
            excess_series = pnl_series.dropna()
        else:
            excess_series = pnl_series.sub(benchmark_return.reindex(pnl_series.index), fill_value=np.nan).dropna()
        excess_mean = excess_series.mean()
        excess_std = excess_series.std()
        result[f"{pnl_type}_excess"] = excess_series
        result[f"{pnl_type}_excess_mean"] = float(excess_mean) if np.isfinite(excess_mean) else 0.0
        result[f"{pnl_type}_excess_ann"] = float(result[f"{pnl_type}_excess_mean"] * periods_per_year)
        result[f"{pnl_type}_excess_cumsum"] = excess_series.cumsum()
        result[f"{pnl_name}_excess_sharpe"] = (
            float(excess_mean / excess_std * np.sqrt(periods_per_year))
            if excess_std > 0
            else 0.0
        )
        # 全行业搜索时，组合本身就是目标投资域，long/short Sharpe 使用组合
        # 自身收益序列。相对等权基准的超额收益 Sharpe 只保留为诊断字段。
        result[f"{pnl_name}_sharpe"] = raw_sharpe

        cumulative_pnl = result[f"{pnl_type}_cumsum"]
        assert isinstance(cumulative_pnl, pd.Series)
        if result[f"{pnl_type}_mean"] <= 0:
            cumulative_pnl = -cumulative_pnl
        drawdown = cumulative_pnl - cumulative_pnl.cummax()
        max_drawdown = drawdown.min()
        max_duration = (drawdown < 0).astype(int).groupby((drawdown >= 0).astype(int).cumsum()).cumsum().max()
        result[f"{pnl_type}_max_drawdown"] = float(max_drawdown) if np.isfinite(max_drawdown) else 0.0
        result[f"{pnl_type}_max_drawdown_duration"] = int(max_duration) if np.isfinite(max_duration) else 0

    long_positions = (group_labels == (n_groups - 1)).astype(int)
    short_positions = (group_labels == 0).astype(int)
    long_turnover = long_positions.diff().abs().sum(axis=1) / (2 * long_positions.sum(axis=1))
    short_turnover = short_positions.diff().abs().sum(axis=1) / (2 * short_positions.sum(axis=1))
    long_turnover_mean = long_turnover.replace(np.inf, np.nan).mean()
    short_turnover_mean = short_turnover.replace(np.inf, np.nan).mean()
    result["long_turnover"] = float(long_turnover_mean) if np.isfinite(long_turnover_mean) else 0.0
    result["short_turnover"] = float(short_turnover_mean) if np.isfinite(short_turnover_mean) else 0.0
    result["longshort_turnover"] = (result["long_turnover"] + result["short_turnover"]) / 2
    result["long_cost_mean"] = float(result.get(f"cost_mean_group{n_groups - 1}", 0.0))
    result["short_cost_mean"] = float(result.get("cost_mean_group0", 0.0))
    result["longshort_cost_mean"] = result["long_cost_mean"] + result["short_cost_mean"]

    for pnl_type in ["long", "short", "longshort"]:
        turnover = max(float(result[f"{pnl_type}_turnover"]), 0.125)
        ann = float(result[f"pnl_{pnl_type}_ann"])
        result[f"{pnl_type}_fitness"] = float(result[f"{pnl_type}_sharpe"] * np.sqrt(abs(ann) / turnover))

    return result


def top_group_annual_return_from_label(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    *,
    tradeable: pd.DataFrame | None = None,
    dates: pd.DatetimeIndex | None = None,
    direction: int = 1,
    n_groups: int = 5,
    label_horizon: int = 20,
    rebalance_freq: int | None = None,
    annualization_days: int = 244,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    stamp_tax_rate: float = 0.001,
) -> float:
    """Top-group annualized label return using alpha_factory group PnL semantics."""

    result = factor_group_pnl(
        factor=factor,
        label=label,
        tradeable=tradeable,
        dates=dates,
        direction=direction,
        n_groups=n_groups,
        label_horizon=label_horizon,
        rebalance_freq=rebalance_freq,
        annualization_days=annualization_days,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        stamp_tax_rate=stamp_tax_rate,
    )
    return float(result["pnl_long_ann"])
