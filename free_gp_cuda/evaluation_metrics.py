from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .neutralizer import neutralize_factor_tensor


Tensor = torch.Tensor
EPS = 1e-12


def _nan_like(values: Tensor) -> Tensor:
    return torch.full_like(values, float("nan"))


def _ensure_2d(values: Tensor, name: str) -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D [date, contract], got shape {tuple(values.shape)}")


def _as_bool_mask(mask: Tensor | None, like: Tensor, *, name: str = "mask") -> Tensor | None:
    if mask is None:
        return None
    if tuple(mask.shape) != tuple(like.shape):
        raise ValueError(f"{name} must share shape {tuple(like.shape)}, got {tuple(mask.shape)}")
    if mask.dtype == torch.bool:
        return mask
    return torch.isfinite(mask) & (mask != 0)


def apply_mask(values: Tensor, mask: Tensor | None) -> Tensor:
    _ensure_2d(values, "values")
    bool_mask = _as_bool_mask(mask, values)
    valid = torch.isfinite(values)
    if bool_mask is not None:
        valid = valid & bool_mask
    return torch.where(valid, values, _nan_like(values))


def _row_corr(
    left: Tensor,
    right: Tensor,
    *,
    min_cross_section_size: int = 3,
    eps: float = EPS,
) -> Tensor:
    _ensure_2d(left, "left")
    _ensure_2d(right, "right")
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("left and right must share shape")
    if min_cross_section_size < 2:
        raise ValueError("min_cross_section_size must be at least 2")

    valid = torch.isfinite(left) & torch.isfinite(right)
    n = valid.sum(dim=1, keepdim=True).to(left.dtype)
    enough = n >= min_cross_section_size

    left0 = torch.where(valid, left, torch.zeros_like(left))
    right0 = torch.where(valid, right, torch.zeros_like(right))
    left_mean = left0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    right_mean = right0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)

    left_centered = torch.where(valid, left - left_mean, torch.zeros_like(left))
    right_centered = torch.where(valid, right - right_mean, torch.zeros_like(right))
    cov = (left_centered * right_centered).sum(dim=1)
    var_left = (left_centered * left_centered).sum(dim=1)
    var_right = (right_centered * right_centered).sum(dim=1)
    denom = torch.sqrt(var_left * var_right)
    return torch.where(
        (denom > eps) & enough.squeeze(1),
        cov / denom,
        torch.full_like(cov, float("nan")),
    )


def nan_rank(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Row-wise ascending average ranks, ignoring NaN and mask-out entries."""

    _ensure_2d(values, "values")
    bool_mask = _as_bool_mask(mask, values)
    valid = torch.isfinite(values)
    if bool_mask is not None:
        valid = valid & bool_mask

    filled = torch.where(valid, values, torch.full_like(values, float("inf")))
    order = torch.argsort(filled, dim=1, stable=True)
    sorted_values = values.gather(1, order)
    sorted_valid = valid.gather(1, order)

    n_rows, n_cols = values.shape
    new_group = torch.ones((n_rows, n_cols), dtype=torch.bool, device=values.device)
    if n_cols > 1:
        same_as_prev = (
            sorted_valid[:, 1:]
            & sorted_valid[:, :-1]
            & (sorted_values[:, 1:] == sorted_values[:, :-1])
        )
        new_group[:, 1:] = ~same_as_prev

    group_id = new_group.to(torch.long).cumsum(dim=1) - 1
    positions = torch.arange(1, n_cols + 1, device=values.device, dtype=values.dtype)
    positions = positions.unsqueeze(0).expand_as(values)

    group_sum = torch.zeros((n_rows, n_cols), dtype=values.dtype, device=values.device)
    group_count = torch.zeros((n_rows, n_cols), dtype=values.dtype, device=values.device)
    group_sum.scatter_add_(1, group_id, torch.where(sorted_valid, positions, torch.zeros_like(positions)))
    group_count.scatter_add_(1, group_id, sorted_valid.to(values.dtype))

    group_average = group_sum / torch.clamp(group_count, min=1.0)
    sorted_ranks = group_average.gather(1, group_id)
    sorted_ranks = torch.where(sorted_valid, sorted_ranks, torch.full_like(sorted_ranks, float("nan")))

    ranks = torch.empty_like(values)
    ranks.scatter_(1, order, sorted_ranks)
    return ranks


def daily_ic(
    factor: Tensor,
    label: Tensor,
    *,
    min_cross_section_size: int = 2,
) -> Tensor:
    """Finite daily Pearson IC series."""

    valid = torch.isfinite(factor) & torch.isfinite(label)
    factor_common = torch.where(valid, factor, _nan_like(factor))
    label_common = torch.where(valid, label, _nan_like(label))
    ic = _row_corr(factor_common, label_common, min_cross_section_size=min_cross_section_size)
    return ic[torch.isfinite(ic)]


def daily_rank_ic(
    factor: Tensor,
    label: Tensor,
    *,
    min_cross_section_size: int = 2,
) -> Tensor:
    """Finite daily Spearman RankIC series.

    Matches pandas ``corrwith(method="spearman")`` behaviour:
    factor is ranked on all finite factor values, label on all finite label
    values, then the Pearson correlation is taken on the
    :func:`common valid intersection <_row_corr>`.

    Prior to 2026-06-03 the two series were ranked on the *same* valid
    intersection, which gave a narrower rank domain when the factor universe
    was larger than the label universe (e.g. factor computed on all stocks
    while the label is tradeable-masked).  The change is typically << 1e-3
    for N > 1000.
    """

    factor_valid = torch.isfinite(factor)
    label_valid = torch.isfinite(label)
    factor_rank = nan_rank(factor, factor_valid)
    label_rank = nan_rank(label, label_valid)
    ic = _row_corr(factor_rank, label_rank, min_cross_section_size=min_cross_section_size)
    return ic[torch.isfinite(ic)]


def factor_coverage(factor: Tensor, label: Tensor, tradeable: Tensor | None = None) -> float:
    _ensure_2d(factor, "factor")
    _ensure_2d(label, "label")
    if tuple(factor.shape) != tuple(label.shape):
        raise ValueError("factor and label must share shape")

    if tradeable is not None:
        universe = _as_bool_mask(tradeable, factor, name="tradeable")
    else:
        universe = torch.isfinite(label)
    assert universe is not None

    denom = universe.sum(dim=1).to(factor.dtype)
    numer = (torch.isfinite(factor) & universe).sum(dim=1).to(factor.dtype)
    daily = numer / torch.clamp(denom, min=1.0)
    daily = daily[denom > 0]
    if daily.numel() == 0:
        return 0.0
    value = daily.mean()
    return float(value.detach().cpu().item()) if torch.isfinite(value) else 0.0


def _top_k_mask(
    values: Tensor,
    valid: Tensor,
    *,
    k: int | None = None,
    top_fraction: float = 0.10,
) -> Tensor:
    _ensure_2d(values, "values")
    if tuple(valid.shape) != tuple(values.shape):
        raise ValueError("valid must share values shape")
    if k is not None and k < 1:
        raise ValueError("k must be positive or None")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    count = valid.sum(dim=1)
    if k is None:
        k_eff = torch.ceil(count.to(values.dtype) * float(top_fraction)).to(torch.long)
    else:
        k_eff = torch.full_like(count, int(k))
    k_eff = torch.minimum(k_eff, count).clamp(min=0)

    scores = torch.where(valid, values, torch.full_like(values, -float("inf")))
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0).expand_as(values)
    selected_sorted = positions < k_eff.unsqueeze(1)
    selected = torch.zeros_like(valid)
    selected.scatter_(1, order, selected_sorted)
    return selected & valid


def ndcg_at_k(
    factor: Tensor,
    label: Tensor,
    *,
    k: int | None = None,
    top_fraction: float = 0.10,
    n_groups: int = 10,
) -> float:
    """Average GPU NDCG@k using rank-quantile label relevance."""

    _ensure_2d(factor, "factor")
    _ensure_2d(label, "label")
    if tuple(factor.shape) != tuple(label.shape):
        raise ValueError("factor and label must share shape")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")

    valid = torch.isfinite(factor) & torch.isfinite(label)
    count = valid.sum(dim=1)
    label_rank = nan_rank(label, valid)
    count_float = torch.clamp(count.to(label.dtype), min=1.0).unsqueeze(1)
    relevance = torch.floor((label_rank - 1.0) * float(n_groups) / count_float)
    relevance = torch.clamp(relevance, min=0.0, max=float(n_groups - 1))
    relevance = torch.where(valid, relevance, torch.zeros_like(relevance))
    gain = torch.pow(torch.full_like(relevance, 2.0), relevance) - 1.0

    top_mask = _top_k_mask(factor, valid, k=k, top_fraction=top_fraction)
    k_eff = top_mask.sum(dim=1)
    eligible = (count >= 2) & (k_eff > 0)

    scores = torch.where(valid, factor, torch.full_like(factor, -float("inf")))
    predicted_order = torch.argsort(scores, dim=1, descending=True, stable=True)
    predicted_gain = gain.gather(1, predicted_order)

    ideal_scores = torch.where(valid, gain, torch.full_like(gain, -float("inf")))
    ideal_order = torch.argsort(ideal_scores, dim=1, descending=True, stable=True)
    ideal_gain = gain.gather(1, ideal_order)

    positions = torch.arange(factor.shape[1], device=factor.device).unsqueeze(0).expand_as(factor)
    top_positions = positions < k_eff.unsqueeze(1)
    discounts = torch.log2(
        torch.arange(2, factor.shape[1] + 2, dtype=factor.dtype, device=factor.device)
    ).unsqueeze(0)

    dcg = torch.where(top_positions, predicted_gain / discounts, torch.zeros_like(predicted_gain)).sum(dim=1)
    idcg = torch.where(top_positions, ideal_gain / discounts, torch.zeros_like(ideal_gain)).sum(dim=1)
    values = torch.where((idcg > EPS) & eligible, dcg / idcg, torch.full_like(dcg, float("nan")))
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0
    return float(values.mean().detach().cpu().item())


def top_turnover(
    factor: Tensor,
    *,
    tradeable: Tensor | None = None,
    k: int | None = None,
    top_fraction: float = 0.10,
) -> float:
    _ensure_2d(factor, "factor")
    valid = torch.isfinite(factor)
    bool_tradeable = _as_bool_mask(tradeable, factor, name="tradeable")
    if bool_tradeable is not None:
        valid = valid & bool_tradeable

    selected = _top_k_mask(factor, valid, k=k, top_fraction=top_fraction)
    counts = selected.sum(dim=1, keepdim=True).to(factor.dtype)
    weights = torch.where(selected, 1.0 / torch.clamp(counts, min=1.0), torch.zeros_like(factor))
    if weights.shape[0] < 2:
        return 0.0
    valid_pairs = (counts[1:, 0] > 0) & (counts[:-1, 0] > 0)
    turnover = (weights[1:] - weights[:-1]).abs().sum(dim=1) / 2.0
    turnover = turnover[valid_pairs]
    if turnover.numel() == 0:
        return 0.0
    return float(turnover.mean().detach().cpu().item())


def factor_stability(
    factor: Tensor,
    *,
    tradeable: Tensor | None = None,
    min_cross_section_size: int = 3,
) -> float:
    _ensure_2d(factor, "factor")
    if factor.shape[0] < 2:
        return 0.0
    left = factor[:-1]
    right = factor[1:]
    if tradeable is not None:
        bool_tradeable = _as_bool_mask(tradeable, factor, name="tradeable")
        assert bool_tradeable is not None
        left = apply_mask(left, bool_tradeable[:-1])
        right = apply_mask(right, bool_tradeable[1:])
    corr = _row_corr(left, right, min_cross_section_size=min_cross_section_size)
    corr = corr[torch.isfinite(corr)]
    if corr.numel() == 0:
        return 0.0
    return float(corr.mean().detach().cpu().item())


def _row_zscore_2d(values: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    valid = torch.isfinite(values)
    bool_mask = _as_bool_mask(mask, values)
    if bool_mask is not None:
        valid = valid & bool_mask
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    variance = (centered * centered).sum(dim=1, keepdim=True) / torch.clamp(n - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    zscore = (values - mean) / (std + eps)
    return torch.where(valid & (n >= 2) & (std > eps), zscore, _nan_like(values))


def _row_zscore_3d(values: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    if values.ndim != 3:
        raise ValueError(f"values must be 3D [date, contract, style], got shape {tuple(values.shape)}")
    valid = torch.isfinite(values)
    if mask is not None:
        if tuple(mask.shape) != tuple(values.shape[:2]):
            raise ValueError("mask must share the first two style tensor dimensions")
        valid = valid & mask.to(torch.bool).unsqueeze(2)
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    variance = (centered * centered).sum(dim=1, keepdim=True) / torch.clamp(n - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    zscore = (values - mean) / (std + eps)
    return torch.where(valid & (n >= 2) & (std > eps), zscore, torch.full_like(values, float("nan")))


@dataclass(frozen=True)
class StyleNeutralizationResult:
    residual_factor: Tensor
    abs_mean_corr: Tensor
    selected_mask: Tensor
    selected_fields: tuple[str, ...] = ()


def dynamic_style_neutralize(
    factor: Tensor,
    style_tensors: Tensor,
    *,
    mask: Tensor | None = None,
    style_fields: Sequence[str] = (),
    corr_threshold: float = 0.30,
    max_styles: int = 2,
    ridge: float = 1e-6,
) -> StyleNeutralizationResult:
    """Select persistent style exposures, then residualize factor cross-sectionally."""

    _ensure_2d(factor, "factor")
    if style_tensors.ndim != 3:
        raise ValueError("style_tensors must be 3D [date, contract, style]")
    if tuple(style_tensors.shape[:2]) != tuple(factor.shape):
        raise ValueError("factor and style_tensors must share date/contract dimensions")
    if corr_threshold < 0:
        raise ValueError("corr_threshold must be non-negative")
    if max_styles <= 0:
        raise ValueError("max_styles must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    n_styles = style_tensors.shape[2]
    empty_corr = torch.empty((0,), dtype=factor.dtype, device=factor.device)
    if n_styles == 0:
        return StyleNeutralizationResult(
            residual_factor=factor,
            abs_mean_corr=empty_corr,
            selected_mask=torch.empty((0,), dtype=torch.bool, device=factor.device),
            selected_fields=(),
        )

    bool_mask = _as_bool_mask(mask, factor)
    valid_factor = torch.isfinite(factor)
    if bool_mask is not None:
        valid_factor = valid_factor & bool_mask

    factor_z = _row_zscore_2d(factor, valid_factor)
    style_z = _row_zscore_3d(style_tensors, valid_factor)
    common = torch.isfinite(factor_z).unsqueeze(2) & torch.isfinite(style_z)
    n = common.sum(dim=1).to(factor.dtype)
    product = torch.where(common, factor_z.unsqueeze(2) * style_z, torch.zeros_like(style_z))
    daily_corr = product.sum(dim=1) / torch.clamp(n - 1.0, min=1.0)
    daily_corr = torch.where(n >= 2, daily_corr, torch.full_like(daily_corr, float("nan")))

    finite_corr = torch.isfinite(daily_corr)
    corr_obs = finite_corr.sum(dim=0)
    mean_corr = torch.where(
        corr_obs > 0,
        torch.where(finite_corr, daily_corr, torch.zeros_like(daily_corr)).sum(dim=0)
        / torch.clamp(corr_obs.to(factor.dtype), min=1.0),
        torch.zeros((n_styles,), dtype=factor.dtype, device=factor.device),
    )
    abs_mean_corr = mean_corr.abs()

    k_eff = min(int(max_styles), n_styles)
    eligible_scores = torch.where(
        abs_mean_corr > corr_threshold,
        abs_mean_corr,
        torch.full_like(abs_mean_corr, -float("inf")),
    )
    top_scores, top_idx = torch.topk(eligible_scores, k=k_eff, largest=True, sorted=True)
    top_valid = torch.isfinite(top_scores)
    selected_mask = torch.zeros((n_styles,), dtype=torch.bool, device=factor.device)
    selected_mask.scatter_(0, top_idx, top_valid)
    selected_count = int(top_valid.sum().detach().cpu().item())
    names = tuple(str(item) for item in style_fields)
    selected_fields = tuple(
        names[idx] if idx < len(names) else f"style_{idx}"
        for idx, flag in enumerate(selected_mask.detach().cpu().tolist())
        if bool(flag)
    )

    if selected_count == 0:
        return StyleNeutralizationResult(
            residual_factor=factor,
            abs_mean_corr=abs_mean_corr,
            selected_mask=selected_mask,
            selected_fields=(),
        )

    raw_selected = style_z.index_select(2, top_idx)
    active = top_valid.view(1, 1, k_eff)
    selected_finite = torch.where(active, torch.isfinite(raw_selected), torch.ones_like(raw_selected, dtype=torch.bool))
    valid = valid_factor & selected_finite.all(dim=2)
    selected_styles = torch.where(
        active & torch.isfinite(raw_selected),
        raw_selected,
        torch.zeros_like(raw_selected),
    )

    intercept = torch.ones((*factor.shape, 1), dtype=factor.dtype, device=factor.device)
    design = torch.cat([intercept, selected_styles], dim=2)
    design0 = torch.where(valid.unsqueeze(2), design, torch.zeros_like(design))
    y0 = torch.where(valid, factor, torch.zeros_like(factor)).unsqueeze(2)
    xt = design0.transpose(1, 2)
    xtx = torch.matmul(xt, design0)
    xty = torch.matmul(xt, y0)
    eye = torch.eye(k_eff + 1, dtype=factor.dtype, device=factor.device).unsqueeze(0)
    beta = torch.linalg.solve(xtx + eye * ridge, xty)
    fitted = torch.matmul(design, beta).squeeze(2)
    residual = factor - fitted
    enough = valid.sum(dim=1, keepdim=True) >= (selected_count + 2)
    residual = torch.where(enough & valid, residual, _nan_like(factor))
    return StyleNeutralizationResult(
        residual_factor=residual,
        abs_mean_corr=abs_mean_corr,
        selected_mask=selected_mask,
        selected_fields=selected_fields,
    )


@dataclass(frozen=True)
class FactorScore:
    mean_ic: float
    abs_ic: float
    ic_ir: float
    pearson_ic_win_rate: float
    n_pearson_ic_obs: int
    mean_rank_ic: float
    abs_rank_ic: float
    rank_ic_ir: float
    ic_win_rate: float
    ndcg_at_k: float
    direction: int
    n_ic_obs: int
    coverage: float
    turnover: float
    stability: float
    neutralized_icir: float = 0.0
    neutralized_mean_rank_ic: float = 0.0
    neutralized_abs_rank_ic: float = 0.0
    neutralized_ic_win_rate: float = 0.0
    neutralized_n_ic_obs: int = 0
    style_max_abs_corr: float = 0.0
    style_selected_count: int = 0
    style_selected_fields: tuple[str, ...] = ()

    @property
    def objectives(self) -> tuple[float, float, float]:
        return (self.rank_ic_ir, self.ndcg_at_k, self.neutralized_icir)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "mean_ic": self.mean_ic,
            "abs_ic": self.abs_ic,
            "ic_ir": self.ic_ir,
            "pearson_ic_win_rate": self.pearson_ic_win_rate,
            "n_pearson_ic_obs": self.n_pearson_ic_obs,
            "mean_rank_ic": self.mean_rank_ic,
            "abs_rank_ic": self.abs_rank_ic,
            "rank_ic_ir": self.rank_ic_ir,
            "ic_win_rate": self.ic_win_rate,
            "ndcg_at_k": self.ndcg_at_k,
            "direction": self.direction,
            "n_ic_obs": self.n_ic_obs,
            "coverage": self.coverage,
            "turnover": self.turnover,
            "stability": self.stability,
            "neutralized_icir": self.neutralized_icir,
            "neutralized_mean_rank_ic": self.neutralized_mean_rank_ic,
            "neutralized_abs_rank_ic": self.neutralized_abs_rank_ic,
            "neutralized_ic_win_rate": self.neutralized_ic_win_rate,
            "neutralized_n_ic_obs": self.neutralized_n_ic_obs,
            "style_max_abs_corr": self.style_max_abs_corr,
            "style_selected_count": self.style_selected_count,
            "style_selected_fields": ",".join(self.style_selected_fields),
        }


def empty_factor_score(*, coverage: float = 0.0) -> FactorScore:
    return FactorScore(
        mean_ic=0.0,
        abs_ic=0.0,
        ic_ir=0.0,
        pearson_ic_win_rate=0.0,
        n_pearson_ic_obs=0,
        mean_rank_ic=0.0,
        abs_rank_ic=0.0,
        rank_ic_ir=0.0,
        ic_win_rate=0.0,
        ndcg_at_k=0.0,
        direction=1,
        n_ic_obs=0,
        coverage=float(coverage),
        turnover=0.0,
        stability=0.0,
    )


def _ic_ir(ic_series: Tensor, direction: int) -> float:
    if ic_series.numel() <= 1:
        return 0.0
    oriented = ic_series * int(direction)
    std = ic_series.std(unbiased=True)
    if not torch.isfinite(std) or std <= 0:
        return 0.0
    value = oriented.mean() / std
    return float(value.detach().cpu().item()) if torch.isfinite(value) else 0.0


def _win_rate(ic_series: Tensor, direction: int) -> float:
    if ic_series.numel() == 0:
        return 0.0
    return float(((ic_series * int(direction)) > 0).to(torch.float32).mean().detach().cpu().item())


def evaluate_factor_tensor(
    factor: Tensor,
    label: Tensor,
    *,
    tradeable: Tensor | None = None,
    style_tensors: Tensor | None = None,
    industry_codes: Tensor | None = None,
    style_fields: Sequence[str] = (),
    neutralize_industry: bool = False,
    neutralize_styles: bool = True,
    standardize_styles: bool = True,
    mask_factor_by_tradeable: bool = True,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    n_groups: int = 10,
    direction: int | None = None,
    min_cross_section_size: int = 2,
    style_corr_threshold: float = 0.30,
    style_max_fields: int = 2,
) -> FactorScore:
    """Evaluate one factor tensor against one forward-return label tensor.

    When ``mask_factor_by_tradeable=False``, the factor is used as-is
    (all stocks participate in the rank domain), matching the behaviour
    of ``alpha_factory.factor.evaluation.ic.ICAnalyzer`` where the
    pre-computed factor is fed through ``corrwith(method="spearman")``
    without any extra tradeable trimming.
    """

    _ensure_2d(factor, "factor")
    _ensure_2d(label, "label")
    if tuple(factor.shape) != tuple(label.shape):
        raise ValueError("factor and label must share shape")
    if direction is not None and direction not in {-1, 1}:
        raise ValueError("direction must be -1, 1, or None")

    tradeable_mask = _as_bool_mask(tradeable, factor, name="tradeable")
    if mask_factor_by_tradeable:
        factor_eval = apply_mask(factor, tradeable_mask)
    else:
        # alpha_factory alignment: factor keeps values for all stocks;
        # only the label is restricted to the tradeable universe.
        factor_eval = factor
    label_eval = apply_mask(label, tradeable_mask)
    coverage = factor_coverage(factor_eval, label_eval, tradeable_mask)

    rank_ic = daily_rank_ic(
        factor_eval,
        label_eval,
        min_cross_section_size=min_cross_section_size,
    )
    pearson_ic = daily_ic(
        factor_eval,
        label_eval,
        min_cross_section_size=min_cross_section_size,
    )
    if rank_ic.numel() == 0:
        return empty_factor_score(coverage=coverage)

    mean_rank_ic_tensor = rank_ic.mean()
    direction_value = int(direction) if direction is not None else (1 if mean_rank_ic_tensor.item() >= 0 else -1)
    oriented_factor = factor_eval * direction_value

    mean_rank_ic = float(mean_rank_ic_tensor.detach().cpu().item())
    mean_ic = float(pearson_ic.mean().detach().cpu().item()) if pearson_ic.numel() > 0 else 0.0
    rank_ic_ir = _ic_ir(rank_ic, direction_value)
    ic_ir = _ic_ir(pearson_ic, direction_value)
    ic_win_rate = _win_rate(rank_ic, direction_value)
    pearson_ic_win_rate = _win_rate(pearson_ic, direction_value)

    ndcg = ndcg_at_k(
        oriented_factor,
        label_eval,
        k=ndcg_k,
        top_fraction=ndcg_top_fraction,
        n_groups=n_groups,
    )
    turnover = top_turnover(
        oriented_factor,
        tradeable=tradeable_mask,
        k=ndcg_k,
        top_fraction=ndcg_top_fraction,
    )
    stability = factor_stability(
        factor_eval,
        tradeable=tradeable_mask,
        min_cross_section_size=min_cross_section_size,
    )

    neutralized_icir = 0.0
    neutralized_mean_rank_ic = 0.0
    neutralized_abs_rank_ic = 0.0
    neutralized_ic_win_rate = 0.0
    neutralized_n_ic_obs = 0
    style_max_abs_corr = 0.0
    style_selected_count = 0
    style_selected_fields: tuple[str, ...] = ()

    if neutralize_industry or (neutralize_styles and style_tensors is not None):
        neutralization = neutralize_factor_tensor(
            factor_eval,
            industry_codes=industry_codes,
            style_tensors=style_tensors,
            mask=tradeable_mask,
            style_fields=style_fields,
            neutralize_industry=neutralize_industry,
            neutralize_styles=neutralize_styles,
            standardize_styles=standardize_styles,
        )
        if neutralization.abs_mean_corr.numel() > 0:
            style_max_abs_corr = float(neutralization.abs_mean_corr.max().detach().cpu().item())
        style_selected_count = int(neutralization.selected_mask.sum().detach().cpu().item())
        style_selected_fields = neutralization.selected_fields

        residual_ic = daily_rank_ic(
            neutralization.residual_factor,
            label_eval,
            min_cross_section_size=min_cross_section_size,
        )
        neutralized_n_ic_obs = int(residual_ic.numel())
        if residual_ic.numel() > 0:
            neutralized_mean_rank_ic = float(residual_ic.mean().detach().cpu().item())
            neutralized_abs_rank_ic = abs(neutralized_mean_rank_ic)
            neutralized_icir = _ic_ir(residual_ic, direction_value)
            neutralized_ic_win_rate = _win_rate(residual_ic, direction_value)

    return FactorScore(
        mean_ic=mean_ic,
        abs_ic=abs(mean_ic),
        ic_ir=ic_ir,
        pearson_ic_win_rate=pearson_ic_win_rate,
        n_pearson_ic_obs=int(pearson_ic.numel()),
        mean_rank_ic=mean_rank_ic,
        abs_rank_ic=abs(mean_rank_ic),
        rank_ic_ir=rank_ic_ir,
        ic_win_rate=ic_win_rate,
        ndcg_at_k=ndcg,
        direction=direction_value,
        n_ic_obs=int(rank_ic.numel()),
        coverage=coverage,
        turnover=turnover,
        stability=stability,
        neutralized_icir=neutralized_icir,
        neutralized_mean_rank_ic=neutralized_mean_rank_ic,
        neutralized_abs_rank_ic=neutralized_abs_rank_ic,
        neutralized_ic_win_rate=neutralized_ic_win_rate,
        neutralized_n_ic_obs=neutralized_n_ic_obs,
        style_max_abs_corr=style_max_abs_corr,
        style_selected_count=style_selected_count,
        style_selected_fields=style_selected_fields,
    )


__all__ = [
    "FactorScore",
    "StyleNeutralizationResult",
    "apply_mask",
    "daily_ic",
    "daily_rank_ic",
    "dynamic_style_neutralize",
    "empty_factor_score",
    "evaluate_factor_tensor",
    "factor_coverage",
    "factor_stability",
    "nan_rank",
    "ndcg_at_k",
    "top_turnover",
]
