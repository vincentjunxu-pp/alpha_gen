from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .gene import TRANSFORM_WINDOWS, FactorGene, FieldRule, validate_gene
from .metrics import FactorScore
from .preprocess import TransformCache


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
    _tensor_cache: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = resolve_device(str(self.device))
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
        return self._frame_to_tensor(key, self.cache.current[(field, use_log)])

    def label(self) -> torch.Tensor:
        return self._frame_to_tensor(("label",), self.cache.label)

    def tradeable(self) -> torch.Tensor:
        tradeable = self._frame_to_tensor(("tradeable",), self.cache.tradeable)
        return torch.isfinite(tradeable) & (tradeable > 0)

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


def _apply_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, _nan_like(values))


def _shift_tensor(values: torch.Tensor, periods: int) -> torch.Tensor:
    shifted = _nan_like(values)
    if periods < values.shape[0]:
        shifted[periods:] = values[:-periods]
    return shifted


def _rolling_std_torch(values: torch.Tensor, window: int) -> torch.Tensor:
    out = _nan_like(values)
    if window <= 1 or values.shape[0] < window:
        return out

    windows = values.unfold(0, window, 1)
    valid = torch.isfinite(windows)
    count = valid.sum(dim=2).to(values.dtype)
    enough = count >= window
    filled = torch.where(valid, windows, torch.zeros_like(windows))
    mean = filled.sum(dim=2) / torch.clamp(count, min=1.0)
    centered = torch.where(valid, windows - mean.unsqueeze(2), torch.zeros_like(windows))
    variance = (centered * centered).sum(dim=2) / torch.clamp(count - 1.0, min=1.0)
    std = torch.sqrt(variance)
    out[window - 1 :] = torch.where(enough, std, torch.full_like(std, float("nan")))
    return out


def apply_transform_torch(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply one same-contract historical transform without future data."""

    if transform == "current":
        return values
    if transform == "log":
        return torch.where(values > 0, torch.log(values), _nan_like(values))
    if transform == "zscore":
        return cs_zscore_torch(values)

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
    if transform.startswith("std_"):
        return _rolling_std_torch(values, window)
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


def calculate_factor_tensor(
    gene: FactorGene,
    ctx: TorchEvalContext,
    *,
    neutralize_size: bool = True,
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
        elif gene.mode == "ratio_product":
            c = feature(gene.c, gene.c_transform)
            d = feature(gene.d, gene.d_transform)
            raw = _safe_divide(a, b) * _safe_divide(c, d)
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

    if not neutralize_size:
        return raw

    size_raw = ctx.get_current("market_cap", False)
    size = torch.where(size_raw > 0, torch.log(size_raw), _nan_like(size_raw))
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
) -> FactorScore:
    """Evaluate one factor tensor on the requested dates."""

    if direction is not None and direction not in {-1, 1}:
        raise ValueError("direction must be -1, 1, or None")

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
        return FactorScore(0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, coverage_value)

    mean_ic_tensor = ic_series.mean()
    direction_value = int(direction) if direction is not None else (1 if mean_ic_tensor.item() >= 0 else -1)
    oriented_ic = ic_series * direction_value
    oriented_factor = factor_eval * direction_value

    ic_std = ic_series.std(unbiased=True) if ic_series.numel() > 1 else torch.tensor(0.0, device=factor.device)
    rank_ic_ir = float((oriented_ic.mean() / ic_std).detach().cpu().item()) if ic_std > 0 else 0.0

    mean_ic = float(mean_ic_tensor.detach().cpu().item())
    ic_win_rate = float((oriented_ic > 0).to(factor.dtype).mean().detach().cpu().item())
    ndcg = ndcg_at_k_torch(
        oriented_factor,
        label_eval,
        k=ndcg_k,
        top_fraction=ndcg_top_fraction,
        n_groups=n_groups,
    )

    return FactorScore(
        mean_rank_ic=mean_ic,
        abs_rank_ic=abs(mean_ic),
        rank_ic_ir=rank_ic_ir,
        ic_win_rate=ic_win_rate,
        ndcg_at_k=ndcg,
        direction=direction_value,
        n_ic_obs=int(ic_series.numel()),
        coverage=coverage_value,
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
