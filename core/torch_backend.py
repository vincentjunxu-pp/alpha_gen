from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .gene import TRANSFORM_WINDOWS, FactorGene, FieldRule, validate_gene
from .metrics import FactorScore
from .preprocess import TransformCache


NEUTRALIZED_METRIC_DYNAMIC_BARRA = "dynamic_barra"
NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY = "full_barra_then_industry"
NEUTRALIZED_METRIC_NONE = "none"
NEUTRALIZED_METRIC_MODES = (
    NEUTRALIZED_METRIC_DYNAMIC_BARRA,
    NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY,
    NEUTRALIZED_METRIC_NONE,
)


# ---------------------------------------------------------------------------
# Torch/CUDA backend.
#
# The CPU pipeline keeps every cached transform as a pandas DataFrame. This
# backend keeps the source cache in host memory, then moves only the matrices
# requested by the current gene onto the GPU. For local smoke runs,
# `cache_on_device=True` also keeps transferred tensors on the GPU to avoid
# repeated PCIe copies.
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a user device string to a torch.device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    return requested


@dataclass
class TorchEvalContext:
    """Bridge between TransformCache DataFrames and GPU tensors."""

    cache: TransformCache
    device: torch.device | str = "auto"
    dtype: torch.dtype = torch.float32
    cache_on_device: bool = True
    barra_style_fields: Sequence[str] | str = ()
    barra_corr_threshold: float = 0.30
    barra_max_styles: int = 2
    _tensor_cache: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = resolve_device(str(self.device))
        raw_barra_fields = [self.barra_style_fields] if isinstance(self.barra_style_fields, str) else self.barra_style_fields
        self.barra_style_fields = tuple(
            dict.fromkeys(
                text
                for field in raw_barra_fields
                if field is not None
                for text in [str(field).strip()]
                if text
            )
        )
        if self.barra_corr_threshold < 0:
            raise ValueError("barra_corr_threshold must be non-negative")
        if self.barra_max_styles <= 0:
            raise ValueError("barra_max_styles must be positive")
        self._validate_cache_layout()
        self.date_index = pd.DatetimeIndex(self.cache.label.index)

    @property
    def field_rules(self) -> Mapping[str, FieldRule]:
        return self.cache.field_rules

    def _validate_cache_layout(self) -> None:
        """Ensure every cached matrix is a Datetime x Contract wide table."""

        label = self.cache.label
        tradeable = self.cache.tradeable
        if not isinstance(label.index, pd.DatetimeIndex):
            raise TypeError("cache.label must be a wide DataFrame indexed by Datetime")
        if not label.index.equals(tradeable.index) or not label.columns.equals(tradeable.columns):
            raise ValueError("cache.label and cache.tradeable must have identical Datetime index and Contract columns")

        for key, frame in self.cache.current.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"cache.current[{key!r}] must be a pandas DataFrame")
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError(f"cache.current[{key!r}] must be indexed by Datetime")
            if not frame.index.equals(label.index):
                raise ValueError(f"cache.current[{key!r}] index does not match cache.label index")
            if not frame.columns.equals(label.columns):
                raise ValueError(f"cache.current[{key!r}] columns do not match cache.label columns")

    def _frame_to_tensor(self, key: tuple[object, ...], frame: pd.DataFrame) -> torch.Tensor:
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]

        array = frame.to_numpy(dtype=np.float32, copy=False)
        tensor = torch.as_tensor(array, dtype=self.dtype, device=self.device)

        if self.cache_on_device:
            self._tensor_cache[key] = tensor
        return tensor

    def get_current(self, field: str, use_log: bool) -> torch.Tensor:
        key = ("current", field, use_log)
        source_key = (field, use_log)
        if source_key not in self.cache.current:
            raise KeyError(
                f"field {field!r} with use_log={use_log} is not cached; "
                "include it in metadata/field_rules or cache.current before building TorchEvalContext"
            )
        return self._frame_to_tensor(key, self.cache.current[source_key])

    def label(self) -> torch.Tensor:
        return self._frame_to_tensor(("label",), self.cache.label)

    def tradeable(self) -> torch.Tensor:
        key = ("tradeable_mask",)
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]
        array = self.cache.tradeable.to_numpy(dtype=np.bool_, copy=False)
        tensor = torch.as_tensor(array, dtype=torch.bool, device=self.device)
        if self.cache_on_device:
            self._tensor_cache[key] = tensor
        return tensor

    def barra_styles(self) -> torch.Tensor:
        """Return cached Barra style tensors as [date, contract, style].

        The expected input is already cross-sectionally z-scored and NaNs have
        been filled with 0.0. We still sanitize infinities here so the batched
        correlation and regression kernels never receive non-finite controls.
        """

        label_shape = self.cache.label.shape
        if not self.barra_style_fields:
            return torch.empty(
                (label_shape[0], label_shape[1], 0),
                dtype=self.dtype,
                device=self.device,
            )

        key = ("barra_styles", tuple(self.barra_style_fields))
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]

        tensors = [
            torch.nan_to_num(self.get_current(field, False), nan=0.0, posinf=0.0, neginf=0.0)
            for field in self.barra_style_fields
        ]
        stacked = torch.stack(tensors, dim=2)
        if self.cache_on_device:
            self._tensor_cache[key] = stacked
        return stacked

    def industry_codes(self) -> torch.Tensor:
        if self.cache.industry is None:
            raise ValueError("industry-relative transforms require cache.industry")
        key = ("industry_codes",)
        if self.cache_on_device and key in self._tensor_cache:
            return self._tensor_cache[key]

        industry = self.cache.industry.reindex(index=self.cache.label.index, columns=self.cache.label.columns)
        codes, _uniques = pd.factorize(industry.to_numpy(dtype=object).ravel(), sort=True, use_na_sentinel=True)
        codes_array = codes.reshape(industry.shape)
        tensor = torch.as_tensor(codes_array, dtype=torch.long, device=self.device)
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


def _nan_like(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, float("nan"))


def _safe_divide(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Element-wise division with tiny denominators set to NaN."""

    return torch.where(right.abs() > eps, left / right, _nan_like(left))


def _signed_log1p_torch(values: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of sign(x) * log(1 + abs(x))."""

    return torch.sign(values) * torch.log1p(values.abs())


def _apply_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, _nan_like(values))


def _shift_tensor(values: torch.Tensor, periods: int) -> torch.Tensor:
    shifted = _nan_like(values)
    if periods < values.shape[0]:
        shifted[periods:] = values[:-periods]
    return shifted


def apply_transform_torch(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply one same-contract historical transform without future data."""

    if transform == "current":
        return values
    if transform == "log":
        return _signed_log1p_torch(values)
    if transform == "zscore":
        return cs_zscore_torch(values)
    if transform == "rank_pct":
        return cs_rank_pct_torch(values)

    if transform.endswith("_2q"):
        window = TRANSFORM_WINDOWS["2q"]
    elif transform.endswith("_1y"):
        window = TRANSFORM_WINDOWS["1y"]
    else:
        raise ValueError(f"unknown transform: {transform!r}")

    if transform.startswith("diff_"):
        return values - _shift_tensor(values, window)
    if transform.startswith("pct_"):
        shifted = _shift_tensor(values, window)
        return torch.where(shifted != 0, values / shifted - 1.0, _nan_like(values))
    raise ValueError(f"unknown transform: {transform!r}")


def cs_zscore_torch(values: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise cross-sectional z-score, matching the local pandas path."""

    valid = torch.isfinite(values)
    if mask is not None:
        valid = valid & mask

    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    enough = n >= 2
    values0 = torch.where(valid, values, torch.zeros_like(values))
    mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    # Pandas std uses ddof=1 by default through DataFrame.std(axis=1).
    variance = (centered * centered).sum(dim=1, keepdim=True) / torch.clamp(n - 1.0, min=1.0)
    std = torch.sqrt(variance)
    zscore = (values - mean) / (std + eps)
    return torch.where(enough & valid, zscore, _nan_like(values))


def cs_rank_pct_torch(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Row-wise percentile ranks, matching pandas rank(pct=True)."""

    valid = torch.isfinite(values)
    if mask is not None:
        valid = valid & mask
    ranked_values = torch.where(valid, values, _nan_like(values))
    ranks = nan_rank_torch(ranked_values)
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    pct = ranks / torch.clamp(n, min=1.0)
    return torch.where(valid & (n >= 1), pct, _nan_like(values))


def industry_zscore_torch(
    values: torch.Tensor,
    industry_codes: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Row-wise z-score inside each industry group."""

    valid = torch.isfinite(values) & (industry_codes >= 0)
    if mask is not None:
        valid = valid & mask

    out = _nan_like(values)
    unique_codes = torch.unique(industry_codes[valid])
    for code in unique_codes:
        group_valid = valid & (industry_codes == code)
        n = group_valid.sum(dim=1, keepdim=True).to(values.dtype)
        enough = n >= 2
        values0 = torch.where(group_valid, values, torch.zeros_like(values))
        mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
        centered = torch.where(group_valid, values - mean, torch.zeros_like(values))
        variance = (centered * centered).sum(dim=1, keepdim=True) / torch.clamp(n - 1.0, min=1.0)
        std = torch.sqrt(variance)
        zscore = (values - mean) / (std + eps)
        out = torch.where(enough & group_valid & (std > 0), zscore, out)
    return out


def industry_rank_pct_torch(
    values: torch.Tensor,
    industry_codes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-wise percentile rank inside each industry group."""

    valid = torch.isfinite(values) & (industry_codes >= 0)
    if mask is not None:
        valid = valid & mask

    out = _nan_like(values)
    unique_codes = torch.unique(industry_codes[valid])
    for code in unique_codes:
        group_valid = valid & (industry_codes == code)
        group_values = torch.where(group_valid, values, _nan_like(values))
        ranks = nan_rank_torch(group_values)
        n = group_valid.sum(dim=1, keepdim=True).to(values.dtype)
        pct = ranks / torch.clamp(n, min=1.0)
        out = torch.where(group_valid & (n >= 1), pct, out)
    return out


def industry_neutralize_torch(
    values: torch.Tensor,
    industry_codes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove industry dummy exposure by demeaning within date/industry groups."""

    valid = torch.isfinite(values) & (industry_codes >= 0)
    if mask is not None:
        valid = valid & mask

    out = _nan_like(values)
    unique_codes = torch.unique(industry_codes[valid])
    for code in unique_codes:
        group_valid = valid & (industry_codes == code)
        n = group_valid.sum(dim=1, keepdim=True).to(values.dtype)
        values0 = torch.where(group_valid, values, torch.zeros_like(values))
        mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
        out = torch.where(group_valid & (n > 0), values - mean, out)
    return out


def _industry_to_dummies_torch(
    codes: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    drop_first: bool = True,
) -> torch.Tensor:
    """Convert integer industry codes to one-hot dummy variables.

    Args:
        codes: ``[dates, assets]`` integer industry codes.  ``-1`` marks missing.
        mask: ``[dates, assets]`` boolean mask for valid observations.
        drop_first: If True (default), drop the first dummy to avoid
            perfect collinearity with the intercept term in
            ``cross_sectional_multi_residual_torch``.

    Returns:
        ``[dates, assets, n_industries - (1 if drop_first else 0)]``
        one-hot dummy tensor (float32).
    """
    valid = codes >= 0
    if mask is not None:
        valid = valid & mask

    flat = codes[valid]
    if flat.numel() == 0:
        return torch.zeros(*codes.shape, 0, dtype=torch.float32, device=codes.device)

    unique_codes = torch.unique(flat)
    n_dummies = unique_codes.numel()
    if drop_first and n_dummies <= 1:
        # Cannot drop the only dummy — return empty (intercept-only baseline).
        return torch.zeros(*codes.shape, 0, dtype=torch.float32, device=codes.device)

    # Broadcast comparison: [dates, assets] vs [n_industries] → [dates, assets, n_industries]
    dummies = (codes.unsqueeze(-1) == unique_codes.unsqueeze(0).unsqueeze(0)).to(torch.float32)
    dummies = dummies * valid.unsqueeze(-1).to(torch.float32)
    if drop_first:
        dummies = dummies[:, :, 1:]  # drop first dummy, keep intercept + ridge doing the work
    return dummies


def cross_sectional_residual_torch(
    y: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Vectorized cross-sectional residual y ~ x for every date.

    Shapes are [date, contract]. Missing values are represented by NaN. If a
    row has fewer than three valid names, the residual is left as NaN.
    """

    valid = torch.isfinite(y) & torch.isfinite(x)
    if mask is not None:
        valid = valid & mask

    n = valid.sum(dim=1, keepdim=True).to(y.dtype)
    enough = n >= 3

    y0 = torch.where(valid, y, torch.zeros_like(y))
    x0 = torch.where(valid, x, torch.zeros_like(x))

    y_mean = y0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    x_mean = x0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)

    y_centered = torch.where(valid, y - y_mean, torch.zeros_like(y))
    x_centered = torch.where(valid, x - x_mean, torch.zeros_like(x))

    denom = (x_centered * x_centered).sum(dim=1, keepdim=True)
    numer = (x_centered * y_centered).sum(dim=1, keepdim=True)
    slope = torch.where(denom > eps, numer / denom, torch.zeros_like(denom))
    intercept = y_mean - slope * x_mean

    residual = y - (intercept + slope * x)

    # If x has no cross-sectional variation, return y demeaned. That matches
    # the CPU fallback and still removes the intercept component.
    demeaned = y - y_mean
    residual = torch.where(denom > eps, residual, demeaned)
    residual = torch.where(enough & valid, residual, _nan_like(y))
    return residual


def cross_sectional_multi_residual_torch(
    y: torch.Tensor,
    controls: list[torch.Tensor],
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Vectorized cross-sectional residual y ~ controls for every date."""

    if not controls:
        return y - torch.nanmean(y, dim=1, keepdim=True)

    valid = torch.isfinite(y)
    for control in controls:
        valid = valid & torch.isfinite(control)
    if mask is not None:
        valid = valid & mask

    k = len(controls)
    n = valid.sum(dim=1, keepdim=True).to(y.dtype)
    enough = n >= (k + 2)

    columns = [torch.ones_like(y), *controls]
    design = torch.stack(columns, dim=2)
    design0 = torch.where(valid.unsqueeze(2), design, torch.zeros_like(design))
    y0 = torch.where(valid, y, torch.zeros_like(y)).unsqueeze(2)

    xt = design0.transpose(1, 2)
    xtx = torch.matmul(xt, design0)
    xty = torch.matmul(xt, y0)
    eye = torch.eye(k + 1, dtype=y.dtype, device=y.device).unsqueeze(0)
    beta = torch.linalg.solve(xtx + eye * eps, xty)
    fitted = torch.matmul(design, beta).squeeze(2)
    residual = y - fitted
    return torch.where(enough & valid, residual, _nan_like(y))


@dataclass(frozen=True)
class BarraNeutralizationResult:
    """Diagnostics from dynamic Barra style neutralization."""

    residual_factor: torch.Tensor
    abs_mean_corr: torch.Tensor
    selected_mask: torch.Tensor


def dynamic_barra_neutralize_torch(
    factor: torch.Tensor,
    barra_styles: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    corr_threshold: float = 0.30,
    max_styles: int = 2,
    ridge: float = 1e-6,
) -> BarraNeutralizationResult:
    """动态寻找最大 Barra 暴露并做截面残差化。

    参数形状：
    - factor: [T, N]，待评估的原因子矩阵。
    - barra_styles: [T, N, K]，已按截面 z-score 且 NaN 已填 0.0 的 Barra 风格矩阵。
    - mask: [T, N]，可交易或评估股票池掩码。

    实现要点：
    - 相关性计算使用一次 einsum 完成所有日期和所有风格的点乘。
    - 风格筛选是全张量 top-k；回归使用批量正规方程，没有逐日期循环。
    - 若没有任何风格超过阈值，残差因子直接返回原因子，避免引入无意义扰动。
    """

    if factor.ndim != 2:
        raise ValueError(f"factor must be 2D [date, contract], got shape {tuple(factor.shape)}")
    if barra_styles.ndim != 3:
        raise ValueError(f"barra_styles must be 3D [date, contract, style], got shape {tuple(barra_styles.shape)}")
    if barra_styles.shape[:2] != factor.shape:
        raise ValueError("factor and barra_styles must share the same date/contract dimensions")
    if corr_threshold < 0:
        raise ValueError("corr_threshold must be non-negative")
    if max_styles <= 0:
        raise ValueError("max_styles must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    n_styles = barra_styles.shape[2]
    empty_corr = torch.empty((0,), dtype=factor.dtype, device=factor.device)
    if n_styles == 0:
        return BarraNeutralizationResult(
            residual_factor=factor,
            abs_mean_corr=empty_corr,
            selected_mask=torch.empty((0,), dtype=torch.bool, device=factor.device),
        )

    valid = torch.isfinite(factor)
    if mask is not None:
        if mask.shape != factor.shape:
            raise ValueError("mask must share the same shape as factor")
        mask = mask.to(torch.bool)
        valid = valid & mask

    # 1. 将原因子按截面 z-score 后填 0，和预处理好的 Barra 风格做批量点乘。
    factor_z = torch.nan_to_num(cs_zscore_torch(factor, mask=valid), nan=0.0, posinf=0.0, neginf=0.0)
    barra_z = torch.nan_to_num(barra_styles, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        barra_z = torch.where(mask.unsqueeze(2), barra_z, torch.zeros_like(barra_z))

    count = valid.sum(dim=1).to(factor.dtype)
    enough_for_corr = count >= 2
    denom = torch.clamp(count - 1.0, min=1.0).unsqueeze(1)
    daily_corr = torch.einsum("tn,tnk->tk", factor_z, barra_z) / denom
    daily_corr = torch.where(enough_for_corr.unsqueeze(1), daily_corr, torch.full_like(daily_corr, float("nan")))

    corr_finite = torch.isfinite(daily_corr)
    corr_obs = corr_finite.sum(dim=0)
    mean_corr = torch.where(
        corr_obs > 0,
        torch.where(corr_finite, daily_corr, torch.zeros_like(daily_corr)).sum(dim=0)
        / torch.clamp(corr_obs.to(factor.dtype), min=1.0),
        torch.zeros((n_styles,), dtype=factor.dtype, device=factor.device),
    )
    abs_mean_corr = mean_corr.abs()

    # 2. 只在超过阈值的风格中取全局 top-k，最多剥离 max_styles 个 Barra 暴露。
    k_eff = min(int(max_styles), n_styles)
    neg_inf = torch.full_like(abs_mean_corr, -float("inf"))
    eligible_scores = torch.where(abs_mean_corr > corr_threshold, abs_mean_corr, neg_inf)
    top_scores, top_idx = torch.topk(eligible_scores, k=k_eff, largest=True, sorted=True)
    top_valid = torch.isfinite(top_scores)
    selected_mask = torch.zeros((n_styles,), dtype=torch.bool, device=factor.device)
    selected_mask.scatter_(0, top_idx, top_valid)
    selected_count = int(top_valid.sum().detach().cpu().item())

    if selected_count == 0:
        return BarraNeutralizationResult(
            residual_factor=factor,
            abs_mean_corr=abs_mean_corr,
            selected_mask=selected_mask,
        )

    # 3. 批量截面 OLS：factor ~ intercept + selected_barra_styles。
    #    top_valid 会把未入选的占位列置零；有效自变量个数用 selected_count 控制。
    selected_barra = barra_z.index_select(2, top_idx) * top_valid.to(factor.dtype).view(1, 1, k_eff)
    intercept = torch.ones((*factor.shape, 1), dtype=factor.dtype, device=factor.device)
    design = torch.cat([intercept, selected_barra], dim=2)

    design0 = torch.where(valid.unsqueeze(2), design, torch.zeros_like(design))
    y0 = torch.where(valid, factor, torch.zeros_like(factor)).unsqueeze(2)
    xt = design0.transpose(1, 2)
    xtx = torch.matmul(xt, design0)
    xty = torch.matmul(xt, y0)

    eye = torch.eye(k_eff + 1, dtype=factor.dtype, device=factor.device).unsqueeze(0)
    beta = torch.linalg.solve(xtx + eye * ridge, xty)
    fitted = torch.matmul(design, beta).squeeze(2)
    residual = factor - fitted

    min_obs = selected_count + 2
    enough_for_reg = valid.sum(dim=1, keepdim=True) >= min_obs
    residual = torch.where(enough_for_reg & valid, residual, _nan_like(factor))
    return BarraNeutralizationResult(
        residual_factor=residual,
        abs_mean_corr=abs_mean_corr,
        selected_mask=selected_mask,
    )


def calculate_factor_tensor(
    gene: FactorGene,
    ctx: TorchEvalContext,
    *,
    neutralize_size: bool = True,
    neutralize_industry: bool = False,
    size_field: str = "barra_size",
    use_log_size: bool | None = None,
    tradeable_only: bool = True,
) -> torch.Tensor:
    """Calculate one structured-expression factor directly on torch tensors."""

    errors = validate_gene(gene, ctx.field_rules)
    if errors:
        raise ValueError("illegal gene: " + "; ".join(errors))

    tradeable_mask = ctx.tradeable() if tradeable_only else None

    def feature(field: str, transform: str) -> torch.Tensor:
        if transform == "zscore":
            return cs_zscore_torch(ctx.get_current(field, False), mask=tradeable_mask)
        if transform == "rank_pct":
            return cs_rank_pct_torch(ctx.get_current(field, False), mask=tradeable_mask)
        if transform == "ind_zscore":
            return industry_zscore_torch(ctx.get_current(field, False), ctx.industry_codes(), mask=tradeable_mask)
        if transform == "ind_rank_pct":
            return industry_rank_pct_torch(ctx.get_current(field, False), ctx.industry_codes(), mask=tradeable_mask)
        transformed = apply_transform_torch(ctx.get_current(field, False), transform)
        return _apply_mask(transformed, tradeable_mask) if tradeable_mask is not None else transformed

    def combine(left: torch.Tensor, right: torch.Tensor, op: str) -> torch.Tensor:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        raise ValueError(f"unknown pair operator: {op!r}")

    a = feature(gene.a, gene.a_transform)
    if gene.mode == "single":
        raw = a
    else:
        b = feature(gene.b, gene.b_transform)
        if gene.mode == "ratio":
            raw = _safe_divide(a, b)
        elif gene.mode == "resi":
            raw = cross_sectional_residual_torch(a, b, mask=tradeable_mask)
        elif gene.mode == "resi_pair":
            c = feature(gene.c, gene.c_transform)
            raw = cross_sectional_residual_torch(a, b + c, mask=tradeable_mask)
        elif gene.mode == "multi_resi":
            c = feature(gene.c, gene.c_transform)
            d = feature(gene.d, gene.d_transform)
            raw = cross_sectional_residual_torch(a, b + c + d, mask=tradeable_mask)
        elif gene.mode == "spread":
            c = feature(gene.c, gene.c_transform)
            d = feature(gene.d, gene.d_transform)
            raw = _safe_divide(a, b) - _safe_divide(c, d)
        elif gene.mode == "style_composite":
            a_style = (feature(gene.a, "rank_pct") - 0.5) * float(ctx.field_rules[gene.a].direction)
            b_style = (feature(gene.b, "rank_pct") - 0.5) * float(ctx.field_rules[gene.b].direction)
            raw = combine(a_style, b_style, "+")
        else:
            left = combine(a, b, gene.left_op)
            c = feature(gene.c, gene.c_transform)
            d = feature(gene.d, gene.d_transform)
            right = combine(c, d, gene.right_op)
            if gene.mode == "pair_ratio":
                raw = _safe_divide(left, right)
            else:
                raise ValueError(f"unknown mode: {gene.mode!r}")

    if tradeable_only:
        raw = _apply_mask(raw, ctx.tradeable())

    if neutralize_industry:
        raw = industry_neutralize_torch(raw, ctx.industry_codes(), mask=ctx.tradeable() if tradeable_only else None)

    if not neutralize_size:
        return raw

    size_raw = ctx.get_current(size_field, False)
    if use_log_size is None:
        use_log_size = size_field == "market_cap"
    size = _signed_log1p_torch(size_raw) if use_log_size else size_raw
    mask = torch.isfinite(raw) & torch.isfinite(size)
    if tradeable_only:
        mask = mask & ctx.tradeable()
    return cross_sectional_residual_torch(raw, size, mask=mask)


def _take_dates(values: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
    if positions is None:
        return values
    return values.index_select(0, positions)


def _row_corr(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    min_cross_section_size: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Row-wise Pearson correlation with NaN masking."""

    if min_cross_section_size < 2:
        raise ValueError("min_cross_section_size must be at least 2")

    valid = torch.isfinite(a) & torch.isfinite(b)
    n = valid.sum(dim=1, keepdim=True).to(a.dtype)
    enough = n >= min_cross_section_size

    a0 = torch.where(valid, a, torch.zeros_like(a))
    b0 = torch.where(valid, b, torch.zeros_like(b))

    a_mean = a0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    b_mean = b0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)

    ac = torch.where(valid, a - a_mean, torch.zeros_like(a))
    bc = torch.where(valid, b - b_mean, torch.zeros_like(b))
    cov = (ac * bc).sum(dim=1)
    var_a = (ac * ac).sum(dim=1)
    var_b = (bc * bc).sum(dim=1)
    denom = torch.sqrt(var_a * var_b)

    corr = torch.where((denom > eps) & enough.squeeze(1), cov / denom, torch.full_like(cov, float("nan")))
    return corr


def nan_rank_torch(values: torch.Tensor) -> torch.Tensor:
    """Row-wise ascending average ranks, ignoring NaNs.

    This matches pandas `DataFrame.rank(axis=1, method="average")` closely
    enough for CPU/GPU RankIC comparisons, including tied values from
    forward-filled financial fields.
    """

    valid = torch.isfinite(values)
    filled = torch.where(valid, values, torch.full_like(values, float("inf")))
    order = torch.argsort(filled, dim=1, stable=True)
    sorted_values = values.gather(1, order)
    sorted_valid = valid.gather(1, order)

    n_rows, n_cols = values.shape
    new_group = torch.ones((n_rows, n_cols), dtype=torch.bool, device=values.device)
    if n_cols > 1:
        same_as_prev = sorted_valid[:, 1:] & sorted_valid[:, :-1] & (sorted_values[:, 1:] == sorted_values[:, :-1])
        new_group[:, 1:] = ~same_as_prev

    group_id = new_group.to(torch.long).cumsum(dim=1) - 1
    positions = torch.arange(1, n_cols + 1, device=values.device, dtype=values.dtype).unsqueeze(0).expand_as(values)

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


def daily_ic_torch(
    factor: torch.Tensor,
    label: torch.Tensor,
    *,
    min_cross_section_size: int = 3,
) -> torch.Tensor:
    """Daily Pearson IC series on GPU."""
    valid = torch.isfinite(factor) & torch.isfinite(label)
    f = torch.where(valid, factor, torch.full_like(factor, float("nan")))
    l = torch.where(valid, label, torch.full_like(label, float("nan")))
    ic = _row_corr(f, l, min_cross_section_size=min_cross_section_size)
    return ic[torch.isfinite(ic)]


def daily_rank_ic_torch(
    factor: torch.Tensor,
    label: torch.Tensor,
    *,
    min_cross_section_size: int = 3,
) -> torch.Tensor:
    """Daily RankIC series on GPU, using the common factor/label universe."""

    valid = torch.isfinite(factor) & torch.isfinite(label)
    factor_common = torch.where(valid, factor, torch.full_like(factor, float("nan")))
    label_common = torch.where(valid, label, torch.full_like(label, float("nan")))

    factor_rank = nan_rank_torch(factor_common)
    label_rank = nan_rank_torch(label_common)
    ic = _row_corr(factor_rank, label_rank, min_cross_section_size=min_cross_section_size)
    return ic[torch.isfinite(ic)]


def ndcg_at_k_torch(
    factor: torch.Tensor,
    label: torch.Tensor,
    *,
    k: int | None = None,
    top_fraction: float = 0.10,
    n_groups: int = 10,
) -> float:
    """Average NDCG@k, delegated to the pandas implementation for exact qcut semantics."""

    from .metrics import ndcg_at_k

    factor_frame = pd.DataFrame(factor.detach().cpu().numpy())
    label_frame = pd.DataFrame(label.detach().cpu().numpy())
    return ndcg_at_k(
        factor_frame,
        label_frame,
        k=k,
        top_fraction=top_fraction,
        n_groups=n_groups,
    )


def evaluate_factor_tensor(
    factor: torch.Tensor,
    ctx: TorchEvalContext,
    *,
    dates: pd.DatetimeIndex | None = None,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    n_groups: int = 10,
    direction: int | None = None,
    min_cross_section_size: int = 3,
    neutralized_metric_mode: str = NEUTRALIZED_METRIC_DYNAMIC_BARRA,
) -> FactorScore:
    """Evaluate one factor tensor on the requested dates."""

    if direction is not None and direction not in {-1, 1}:
        raise ValueError("direction must be -1, 1, or None")
    if neutralized_metric_mode not in NEUTRALIZED_METRIC_MODES:
        raise ValueError(
            f"neutralized_metric_mode must be one of {NEUTRALIZED_METRIC_MODES}, "
            f"got {neutralized_metric_mode!r}"
        )

    positions = ctx.date_positions(dates)
    factor_eval = _take_dates(factor, positions)
    label_eval = _take_dates(ctx.label(), positions)
    tradeable_eval = _take_dates(ctx.tradeable(), positions)

    factor_eval = _apply_mask(factor_eval, tradeable_eval)
    label_eval = _apply_mask(label_eval, tradeable_eval)

    ic_series = daily_rank_ic_torch(
        factor_eval,
        label_eval,
        min_cross_section_size=min_cross_section_size,
    )
    tradeable_count = tradeable_eval.sum(dim=1)
    coverage_series = torch.isfinite(factor_eval).sum(dim=1) / torch.clamp(tradeable_count, min=1)
    coverage_series = coverage_series[tradeable_count > 0]
    coverage = coverage_series.mean() if coverage_series.numel() > 0 else torch.tensor(0.0, device=factor.device)
    coverage_value = float(coverage.detach().cpu().item()) if torch.isfinite(coverage) else 0.0

    if ic_series.numel() == 0:
        return FactorScore(
            mean_rank_ic=0.0,
            rank_ic_ir=0.0,
            ic_win_rate=0.0,
            ndcg_at_k=0.0,
            direction=1,
            n_ic_obs=0,
            coverage=coverage_value,
            ic=0.0,
            ir=0.0,
            long_rank_ic=0.0,
            long_rank_ic_ir=0.0,
            long_ic=0.0,
            long_ir=0.0,
            neutralized_icir=0.0,
            neutralized_mean_rank_ic=0.0,
            neutralized_ic_win_rate=0.0,
            neutralized_n_ic_obs=0,
        )

    mean_ic_tensor = ic_series.mean()
    direction_value = int(direction) if direction is not None else (1 if mean_ic_tensor.item() >= 0 else -1)
    oriented_ic = ic_series * direction_value
    oriented_factor = factor_eval * direction_value

    ic_std = ic_series.std(unbiased=True) if ic_series.numel() > 1 else torch.tensor(0.0, device=factor.device)
    rank_ic_ir = float((oriented_ic.mean() / ic_std).detach().cpu().item()) if ic_std > 0 else 0.0

    # ---- Pearson IC ------------------------------------------------
    _ic, _ir = 0.0, 0.0
    try:
        pearson_ic = daily_ic_torch(oriented_factor, label_eval, min_cross_section_size=min_cross_section_size)
        if pearson_ic.numel() > 1:
            _ic = float(pearson_ic.mean().detach().cpu().item())
            p_std = pearson_ic.std(unbiased=True)
            _ir = float((pearson_ic.mean() / p_std).detach().cpu().item()) if p_std > 0 else 0.0
    except Exception:
        pass

    # ---- long-side Rank IC & Pearson IC (top-half) ----------------
    long_rank_ic = 0.0
    long_rank_ic_ir = 0.0
    _long_ic = 0.0
    _long_ir = 0.0
    try:
        rank_pct_long = cs_rank_pct_torch(oriented_factor, mask=tradeable_eval)
        long_mask = tradeable_eval & (rank_pct_long >= 0.5)
        long_factor = torch.where(long_mask, oriented_factor, _nan_like(oriented_factor))
        long_label = torch.where(long_mask, label_eval, _nan_like(label_eval))
        long_ic = daily_rank_ic_torch(long_factor, long_label, min_cross_section_size=min_cross_section_size)
        if long_ic.numel() > 1:
            long_ic_mean = long_ic.mean()
            long_ic_std = long_ic.std(unbiased=True)
            long_rank_ic = float(long_ic_mean.detach().cpu().item())
            long_rank_ic_ir = float((long_ic_mean / long_ic_std).detach().cpu().item()) if long_ic_std > 0 else 0.0
            # ---- long-side Pearson IC (reuse long_mask) ------------
            try:
                l_pearson = daily_ic_torch(long_factor, long_label, min_cross_section_size=min_cross_section_size)
                if l_pearson.numel() > 1:
                    _long_ic = float(l_pearson.mean().detach().cpu().item())
                    l_std = l_pearson.std(unbiased=True)
                    _long_ir = float((l_pearson.mean() / l_std).detach().cpu().item()) if l_std > 0 else 0.0
            except Exception:
                pass
    except Exception:
        pass

    mean_ic = float(mean_ic_tensor.detach().cpu().item())
    ic_win_rate = float((oriented_ic > 0).to(factor.dtype).mean().detach().cpu().item())
    ndcg = ndcg_at_k_torch(
        oriented_factor,
        label_eval,
        k=ndcg_k,
        top_fraction=ndcg_top_fraction,
        n_groups=n_groups,
    )

    # These fields are deliberately zero until a Barra style context is
    # actually available. Do not copy raw ICIR into neutralized ICIR; that would
    # turn an unevaluated robustness objective into a false positive.
    neutralized_icir = 0.0
    neutralized_mean_rank_ic = 0.0
    neutralized_ic_win_rate = 0.0
    neutralized_n_ic_obs = 0

    if neutralized_metric_mode != NEUTRALIZED_METRIC_NONE and ctx.barra_style_fields:
        barra_eval = _take_dates(ctx.barra_styles(), positions)
        if neutralized_metric_mode == NEUTRALIZED_METRIC_DYNAMIC_BARRA:
            residual_factor = dynamic_barra_neutralize_torch(
                factor_eval, barra_eval,
                mask=tradeable_eval,
                corr_threshold=ctx.barra_corr_threshold,
                max_styles=ctx.barra_max_styles,
            ).residual_factor
        else:
            # Simultaneous regression on 10 Barra styles + industry dummies.
            # Previously this was a two-step process (Barra residual → industry
            # demean), which double-counted shared Barra–industry covariance.
            barra_list = list(barra_eval.unbind(dim=2))
            industry_codes_eval = _take_dates(ctx.industry_codes(), positions)
            industry_dummies = _industry_to_dummies_torch(industry_codes_eval, tradeable_eval)
            industry_list = list(industry_dummies.unbind(dim=2))
            all_controls = barra_list + industry_list
            residual_factor = cross_sectional_multi_residual_torch(
                factor_eval, all_controls, mask=tradeable_eval,
            )

        residual_ic = daily_rank_ic_torch(
            residual_factor, label_eval,
            min_cross_section_size=min_cross_section_size,
        )
        neutralized_n_ic_obs = int(residual_ic.numel())
        if residual_ic.numel() > 0:
            residual_mean_tensor = residual_ic.mean()
            residual_oriented_ic = residual_ic * direction_value
            residual_std = (
                residual_ic.std(unbiased=True)
                if residual_ic.numel() > 1
                else torch.tensor(0.0, device=factor.device)
            )
            neutralized_icir = (
                float((residual_oriented_ic.mean() / residual_std).detach().cpu().item())
                if residual_std > 0
                else 0.0
            )
            neutralized_mean_rank_ic = float(residual_mean_tensor.detach().cpu().item())
            neutralized_ic_win_rate = float(
                (residual_oriented_ic > 0).to(factor.dtype).mean().detach().cpu().item()
            )

    return FactorScore(
        mean_rank_ic=mean_ic,
        rank_ic_ir=rank_ic_ir,
        ic_win_rate=ic_win_rate,
        ndcg_at_k=ndcg,
        direction=direction_value,
        n_ic_obs=int(ic_series.numel()),
        coverage=coverage_value,
        ic=_ic,
        ir=_ir,
        long_rank_ic=long_rank_ic,
        long_rank_ic_ir=long_rank_ic_ir,
        long_ic=_long_ic,
        long_ir=_long_ir,
        neutralized_icir=neutralized_icir,
        neutralized_mean_rank_ic=neutralized_mean_rank_ic,
        neutralized_ic_win_rate=neutralized_ic_win_rate,
        neutralized_n_ic_obs=neutralized_n_ic_obs,
    )


def top_group_excess_return_tensor(
    factor: torch.Tensor,
    ctx: TorchEvalContext,
    *,
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
    """Annualized top-group label return on tensors.

    This mirrors alpha_factory's group-PnL semantics using the provided label
    matrix as the forward-return input. By default the rebalance frequency
    equals the label horizon, so overlapping 20-day labels are not evaluated as
    daily PnL. Turnover-based transaction costs are deducted from the top-group
    return.
    """

    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if label_horizon <= 0:
        raise ValueError("label_horizon must be positive")
    if rebalance_freq is None:
        rebalance_freq = label_horizon
    if rebalance_freq <= 0:
        raise ValueError("rebalance_freq must be positive")
    if commission_rate < 0 or slippage_rate < 0 or stamp_tax_rate < 0:
        raise ValueError("transaction cost rates must be non-negative")

    positions = ctx.date_positions(dates)
    factor_eval = _take_dates(factor, positions) * int(direction)
    label_eval = _take_dates(ctx.label(), positions)
    tradeable_eval = _take_dates(ctx.tradeable(), positions)
    n_groups = max(2, int(round(1.0 / top_fraction))) if 0 < top_fraction <= 1 else 5
    round_trip_cost = 2.0 * (commission_rate + slippage_rate) + stamp_tax_rate

    values: list[torch.Tensor] = []
    previous_position: torch.Tensor | None = None
    valid = torch.isfinite(factor_eval) & torch.isfinite(label_eval) & tradeable_eval
    for row_id in range(0, factor_eval.shape[0], rebalance_freq):
        row_valid = valid[row_id]
        available = int(row_valid.sum().item())
        if available < 2:
            continue
        k_eff = max(1, int(np.ceil(available / n_groups)))

        factor_row = torch.where(row_valid, factor_eval[row_id], torch.full_like(factor_eval[row_id], -float("inf")))
        top_idx = torch.topk(factor_row, k=k_eff).indices
        current_position = torch.zeros_like(row_valid, dtype=factor.dtype)
        current_position[top_idx] = 1.0
        if previous_position is None:
            turnover = torch.tensor(0.0, device=factor.device, dtype=factor.dtype)
        else:
            denom = 2.0 * current_position.sum()
            turnover = (current_position - previous_position).abs().sum() / denom if denom > 0 else torch.tensor(0.0, device=factor.device, dtype=factor.dtype)
        previous_position = current_position
        values.append(label_eval[row_id, top_idx].mean() - turnover * round_trip_cost)

    if not values:
        return 0.0
    mean_return = torch.stack(values).mean()
    return float((mean_return * annualization_days / rebalance_freq).detach().cpu().item())


def cuda_memory_summary() -> dict[str, float | str]:
    """Small diagnostic helper for scripts."""

    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 3),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 3),
    }
