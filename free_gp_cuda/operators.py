from __future__ import annotations

import torch


Tensor = torch.Tensor


def _nan_like(values: Tensor) -> Tensor:
    return torch.full_like(values, float("nan"))


def _ensure_2d(values: Tensor, name: str = "values") -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D [date, contract], got shape {tuple(values.shape)}")


def _clean(values: Tensor) -> Tensor:
    _ensure_2d(values)
    return torch.where(torch.isfinite(values), values, _nan_like(values))


def _valid(values: Tensor, mask: Tensor | None = None) -> Tensor:
    _ensure_2d(values)
    valid = torch.isfinite(values)
    if mask is not None:
        if mask.shape != values.shape:
            raise ValueError("mask must share the input tensor shape")
        valid = valid & mask.to(torch.bool)
    return valid


def neg(values: Tensor) -> Tensor:
    values = _clean(values)
    return -values


def abs(values: Tensor) -> Tensor:  # noqa: A001 - exported operator name
    values = _clean(values)
    return values.abs()


def sign(values: Tensor) -> Tensor:
    values = _clean(values)
    return torch.sign(values)


def slog(values: Tensor) -> Tensor:
    values = _clean(values)
    return torch.sign(values) * torch.log1p(values.abs())


def sqrt_abs(values: Tensor) -> Tensor:
    values = _clean(values)
    return torch.sqrt(values.abs())


def add(left: Tensor, right: Tensor) -> Tensor:
    valid = torch.isfinite(left) & torch.isfinite(right)
    return torch.where(valid, left + right, _nan_like(left))


def sub(left: Tensor, right: Tensor) -> Tensor:
    valid = torch.isfinite(left) & torch.isfinite(right)
    return torch.where(valid, left - right, _nan_like(left))


def mul(left: Tensor, right: Tensor) -> Tensor:
    valid = torch.isfinite(left) & torch.isfinite(right)
    return torch.where(valid, left * right, _nan_like(left))


def qdiv(left: Tensor, right: Tensor, eps: float = 1e-6) -> Tensor:
    valid = torch.isfinite(left) & torch.isfinite(right) & (right.abs() > eps)
    return torch.where(valid, left / right, _nan_like(left))


def _nan_rank(values: Tensor, mask: Tensor | None = None) -> Tensor:
    _ensure_2d(values)
    valid = _valid(values, mask)
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


def cs_rank(values: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = _valid(values, mask)
    ranks = _nan_rank(values, mask)
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    pct = ranks / torch.clamp(n, min=1.0)
    return torch.where(valid & (n >= 1), pct, _nan_like(values))


def cs_demean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = _valid(values, mask)
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    demeaned = values - mean
    return torch.where(valid & (n > 0), demeaned, _nan_like(values))


def cs_zscore(values: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    valid = _valid(values, mask)
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    variance = (centered * centered).sum(dim=1, keepdim=True) / torch.clamp(n - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    zscore = (values - mean) / (std + eps)
    return torch.where(valid & (n >= 2) & (std > eps), zscore, _nan_like(values))


def _row_quantile(values: Tensor, q: float, mask: Tensor | None = None, min_count: int = 1) -> Tensor:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    valid = _valid(values, mask)
    filled = torch.where(valid, values, torch.full_like(values, float("inf")))
    sorted_values, _ = torch.sort(filled, dim=1, stable=True)
    n_cols = values.shape[1]
    count = valid.sum(dim=1, keepdim=True)
    pos = (count.to(values.dtype) - 1.0) * float(q)
    lower_idx = torch.floor(torch.clamp(pos, min=0.0)).to(torch.long).clamp(max=n_cols - 1)
    upper_idx = torch.ceil(torch.clamp(pos, min=0.0)).to(torch.long).clamp(max=n_cols - 1)
    lower = sorted_values.gather(1, lower_idx)
    upper = sorted_values.gather(1, upper_idx)
    weight = pos - lower_idx.to(values.dtype)
    quantile = lower + (upper - lower) * weight
    return torch.where(count >= min_count, quantile, torch.full_like(quantile, float("nan")))


def cs_winsorize_5pct(values: Tensor, mask: Tensor | None = None, min_count: int = 3) -> Tensor:
    valid = _valid(values, mask)
    lower = _row_quantile(values, 0.05, mask=mask, min_count=min_count)
    upper = _row_quantile(values, 0.95, mask=mask, min_count=min_count)
    clipped = torch.minimum(torch.maximum(values, lower), upper)
    enough = torch.isfinite(lower) & torch.isfinite(upper)
    return torch.where(valid & enough, clipped, _nan_like(values))


def _cs_regression_parts(y: Tensor, x: Tensor, mask: Tensor | None = None, eps: float = 1e-12) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    valid = torch.isfinite(y) & torch.isfinite(x)
    if mask is not None:
        if mask.shape != y.shape:
            raise ValueError("mask must share the input tensor shape")
        valid = valid & mask.to(torch.bool)
    n = valid.sum(dim=1, keepdim=True).to(y.dtype)
    y0 = torch.where(valid, y, torch.zeros_like(y))
    x0 = torch.where(valid, x, torch.zeros_like(x))
    y_mean = y0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    x_mean = x0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    yc = torch.where(valid, y - y_mean, torch.zeros_like(y))
    xc = torch.where(valid, x - x_mean, torch.zeros_like(x))
    denom = (xc * xc).sum(dim=1, keepdim=True)
    numer = (xc * yc).sum(dim=1, keepdim=True)
    slope = numer / torch.clamp(denom, min=eps)
    intercept = y_mean - slope * x_mean
    enough = (n >= 3) & (denom > eps)
    return valid, enough, slope, intercept


def cs_resid(y: Tensor, x: Tensor, mask: Tensor | None = None, eps: float = 1e-12) -> Tensor:
    valid, enough, slope, intercept = _cs_regression_parts(y, x, mask=mask, eps=eps)
    residual = y - (intercept + slope * x)
    return torch.where(valid & enough, residual, _nan_like(y))


def delay(values: Tensor, window: int) -> Tensor:
    if window < 1:
        raise ValueError("window must be positive")
    values = _clean(values)
    shifted = _nan_like(values)
    if window < values.shape[0]:
        shifted[window:] = values[:-window]
    return shifted


def ts_delta(values: Tensor, window: int) -> Tensor:
    return sub(values, delay(values, window))


def ts_return(values: Tensor, window: int, eps: float = 1e-6) -> Tensor:
    shifted = delay(values, window)
    return sub(qdiv(values, shifted, eps=eps), torch.ones_like(values))


def _rolling_sums(values: Tensor, window: int) -> tuple[Tensor, Tensor, Tensor]:
    if window < 1:
        raise ValueError("window must be positive")
    _ensure_2d(values)
    valid = torch.isfinite(values)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    count0 = valid.to(values.dtype)
    zeros = torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device)
    sums = torch.cat([zeros, values0.cumsum(dim=0)], dim=0)
    counts = torch.cat([zeros, count0.cumsum(dim=0)], dim=0)
    squares = torch.cat([zeros, (values0 * values0).cumsum(dim=0)], dim=0)
    end = torch.arange(1, values.shape[0] + 1, device=values.device)
    start = torch.clamp(end - window, min=0)
    window_sum = sums.index_select(0, end) - sums.index_select(0, start)
    window_count = counts.index_select(0, end) - counts.index_select(0, start)
    window_square = squares.index_select(0, end) - squares.index_select(0, start)
    return window_sum, window_count, window_square


def ts_mean(values: Tensor, window: int, min_periods: int | None = None) -> Tensor:
    if min_periods is None:
        min_periods = window
    sums, counts, _squares = _rolling_sums(values, window)
    mean = sums / torch.clamp(counts, min=1.0)
    current_valid = torch.isfinite(values)
    return torch.where(current_valid & (counts >= min_periods), mean, _nan_like(values))


def ts_std(values: Tensor, window: int, min_periods: int | None = None, eps: float = 0.0) -> Tensor:
    if min_periods is None:
        min_periods = window
    sums, counts, squares = _rolling_sums(values, window)
    variance = (squares - sums * sums / torch.clamp(counts, min=1.0)) / torch.clamp(counts - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    current_valid = torch.isfinite(values)
    return torch.where(current_valid & (counts >= max(min_periods, 2)) & (std >= eps), std, _nan_like(values))


def _rolling_windows(values: Tensor, window: int) -> Tensor:
    if window < 1:
        raise ValueError("window must be positive")
    _ensure_2d(values)
    if window > values.shape[0]:
        return torch.empty((0, values.shape[1], window), dtype=values.dtype, device=values.device)
    return values.unfold(0, window, 1)


def ts_median(values: Tensor, window: int, min_periods: int | None = None) -> Tensor:
    if min_periods is None:
        min_periods = window
    windows = _rolling_windows(values, window)
    out = _nan_like(values)
    if windows.shape[0] == 0:
        return out
    valid = torch.isfinite(windows)
    count = valid.sum(dim=2)
    sorted_values, _ = torch.sort(torch.where(valid, windows, torch.full_like(windows, float("inf"))), dim=2, stable=True)
    lower_idx = torch.floor((count.to(values.dtype) - 1.0) * 0.5).to(torch.long).clamp(min=0, max=window - 1)
    upper_idx = torch.ceil((count.to(values.dtype) - 1.0) * 0.5).to(torch.long).clamp(min=0, max=window - 1)
    lower = sorted_values.gather(2, lower_idx.unsqueeze(2)).squeeze(2)
    upper = sorted_values.gather(2, upper_idx.unsqueeze(2)).squeeze(2)
    median = (lower + upper) * 0.5
    enough = count >= min_periods
    current_valid = torch.isfinite(values[window - 1 :])
    out[window - 1 :] = torch.where(enough & current_valid, median, _nan_like(median))
    return out


def ts_zscore(values: Tensor, window: int, min_periods: int | None = None, eps: float = 1e-8) -> Tensor:
    mean = ts_mean(values, window, min_periods=min_periods)
    std = ts_std(values, window, min_periods=min_periods)
    zscore = (values - mean) / (std + eps)
    return torch.where(torch.isfinite(values) & torch.isfinite(mean) & torch.isfinite(std) & (std > eps), zscore, _nan_like(values))


def ts_max_to_min(values: Tensor, window: int, min_periods: int | None = None) -> Tensor:
    if min_periods is None:
        min_periods = window
    windows = _rolling_windows(values, window)
    out = _nan_like(values)
    if windows.shape[0] == 0:
        return out
    valid = torch.isfinite(windows)
    count = valid.sum(dim=2)
    max_values = torch.where(valid, windows, torch.full_like(windows, -float("inf"))).max(dim=2).values
    min_values = torch.where(valid, windows, torch.full_like(windows, float("inf"))).min(dim=2).values
    diff = max_values - min_values
    current_valid = torch.isfinite(values[window - 1 :])
    out[window - 1 :] = torch.where((count >= min_periods) & current_valid, diff, _nan_like(diff))
    return out


def ts_meanrank(values: Tensor, window: int, min_periods: int | None = None) -> Tensor:
    return ts_mean(cs_rank(values), window=window, min_periods=min_periods)


def diff_sign(values: Tensor, window: int, min_periods: int | None = None) -> Tensor:
    mean = ts_mean(values, window=window, min_periods=min_periods)
    diff = values - mean
    return torch.where(torch.isfinite(diff), torch.sign(diff), _nan_like(values))


def _rolling_pair_sums(left: Tensor, right: Tensor, window: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if window < 1:
        raise ValueError("window must be positive")
    valid = torch.isfinite(left) & torch.isfinite(right)
    left0 = torch.where(valid, left, torch.zeros_like(left))
    right0 = torch.where(valid, right, torch.zeros_like(right))
    count0 = valid.to(left.dtype)
    zeros = torch.zeros((1, left.shape[1]), dtype=left.dtype, device=left.device)
    end = torch.arange(1, left.shape[0] + 1, device=left.device)
    start = torch.clamp(end - window, min=0)

    def windowed(source: Tensor) -> Tensor:
        prefix = torch.cat([zeros, source.cumsum(dim=0)], dim=0)
        return prefix.index_select(0, end) - prefix.index_select(0, start)

    count = windowed(count0)
    sum_left = windowed(left0)
    sum_right = windowed(right0)
    sum_left_square = windowed(left0 * left0)
    sum_right_square = windowed(right0 * right0)
    sum_product = windowed(left0 * right0)
    return count, sum_left, sum_right, sum_left_square, sum_right_square, sum_product


def ts_corr(left: Tensor, right: Tensor, window: int, min_periods: int | None = None, eps: float = 1e-8) -> Tensor:
    if min_periods is None:
        min_periods = window
    count, sum_left, sum_right, sum_left_square, sum_right_square, sum_product = _rolling_pair_sums(left, right, window)
    cov_num = sum_product - sum_left * sum_right / torch.clamp(count, min=1.0)
    var_left = sum_left_square - sum_left * sum_left / torch.clamp(count, min=1.0)
    var_right = sum_right_square - sum_right * sum_right / torch.clamp(count, min=1.0)
    denom = torch.sqrt(torch.clamp(var_left * var_right, min=0.0))
    corr = cov_num / torch.clamp(denom, min=eps)
    current_valid = torch.isfinite(left) & torch.isfinite(right)
    return torch.where(current_valid & (count >= max(min_periods, 2)) & (denom > eps), corr, _nan_like(left))


def _rolling_selmean(
    x: Tensor,
    y: Tensor,
    window: int,
    n: int,
    *,
    largest: bool,
    min_periods: int | None = None,
) -> Tensor:
    if n < 1:
        raise ValueError("n must be positive")
    if min_periods is None:
        min_periods = window
    x_windows = _rolling_windows(x, window)
    y_windows = _rolling_windows(y, window)
    out = _nan_like(x)
    if x_windows.shape[0] == 0:
        return out
    if n > window:
        return out
    valid = torch.isfinite(x_windows) & torch.isfinite(y_windows)
    count = valid.sum(dim=2)
    rank_source = torch.where(
        valid,
        y_windows,
        torch.full_like(y_windows, -float("inf") if largest else float("inf")),
    )
    index = torch.topk(rank_source, k=n, dim=2, largest=largest, sorted=False).indices
    selected_x = x_windows.gather(2, index)
    selected_valid = valid.gather(2, index)
    selected_sum = torch.where(selected_valid, selected_x, torch.zeros_like(selected_x)).sum(dim=2)
    selected_count = selected_valid.sum(dim=2).to(x.dtype)
    selected_mean = selected_sum / torch.clamp(selected_count, min=1.0)
    enough = (count >= max(min_periods, n)) & (selected_valid.sum(dim=2) == n)
    current_valid = torch.isfinite(x[window - 1 :]) & torch.isfinite(y[window - 1 :])
    out[window - 1 :] = torch.where(enough & current_valid, selected_mean, _nan_like(selected_mean))
    return out


def rolling_selmean_diff(x: Tensor, y: Tensor, window: int, n: int, min_periods: int | None = None) -> Tensor:
    top = _rolling_selmean(x, y, window, n, largest=True, min_periods=min_periods)
    btm = _rolling_selmean(x, y, window, n, largest=False, min_periods=min_periods)
    return sub(top, btm)


def decay_linear(values: Tensor, window: int, min_periods: int | None = None) -> Tensor:
    if window < 1:
        raise ValueError("window must be positive")
    if min_periods is None:
        min_periods = window
    valid = torch.isfinite(values)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    valid0 = valid.to(values.dtype)
    time_weight = torch.arange(1, values.shape[0] + 1, dtype=values.dtype, device=values.device).unsqueeze(1)
    zeros = torch.zeros((1, values.shape[1]), dtype=values.dtype, device=values.device)
    end = torch.arange(1, values.shape[0] + 1, device=values.device)
    start = torch.clamp(end - window, min=0)

    def windowed(source: Tensor) -> Tensor:
        prefix = torch.cat([zeros, source.cumsum(dim=0)], dim=0)
        return prefix.index_select(0, end) - prefix.index_select(0, start)

    raw_sum = windowed(values0)
    raw_count = windowed(valid0)
    weighted_sum = windowed(values0 * time_weight) - start.unsqueeze(1).to(values.dtype) * raw_sum
    weighted_count = windowed(valid0 * time_weight) - start.unsqueeze(1).to(values.dtype) * raw_count
    decayed = weighted_sum / torch.clamp(weighted_count, min=1.0)
    return torch.where(valid & (raw_count >= min_periods), decayed, _nan_like(values))


def _float_mask(condition: Tensor, like: Tensor) -> Tensor:
    return torch.where(condition, torch.ones_like(like), torch.zeros_like(like))


def mask_rank_high_50(values: Tensor, mask: Tensor | None = None) -> Tensor:
    ranked = cs_rank(values, mask=mask)
    valid = torch.isfinite(ranked)
    return _float_mask(valid & (ranked >= 0.5), values)


def mask_rank_high_80(values: Tensor, mask: Tensor | None = None) -> Tensor:
    ranked = cs_rank(values, mask=mask)
    valid = torch.isfinite(ranked)
    return _float_mask(valid & (ranked >= 0.8), values)


def mask_rank_low_20(values: Tensor, mask: Tensor | None = None) -> Tensor:
    ranked = cs_rank(values, mask=mask)
    valid = torch.isfinite(ranked)
    return _float_mask(valid & (ranked <= 0.2), values)


def mask_sign_pos(values: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = _valid(values, mask)
    return _float_mask(valid & (values > 0), values)


def mask_sign_neg(values: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = _valid(values, mask)
    return _float_mask(valid & (values < 0), values)


def gate_nan(values: Tensor, condition: Tensor, mask: Tensor | None = None) -> Tensor:
    gate = condition.to(torch.bool)
    if mask is not None:
        gate = gate & mask.to(torch.bool)
    return torch.where(gate & torch.isfinite(values), values, _nan_like(values))


def gate_zero(values: Tensor, condition: Tensor, mask: Tensor | None = None) -> Tensor:
    gate = condition.to(torch.bool)
    if mask is not None:
        mask_bool = mask.to(torch.bool)
        selected = gate & mask_bool & torch.isfinite(values)
        blocked = (~gate) & mask_bool
        return torch.where(selected, values, torch.where(blocked, torch.zeros_like(values), _nan_like(values)))
    selected = gate & torch.isfinite(values)
    blocked = ~gate
    return torch.where(selected, values, torch.where(blocked, torch.zeros_like(values), _nan_like(values)))


__all__ = [
    "neg",
    "abs",
    "sign",
    "slog",
    "sqrt_abs",
    "add",
    "sub",
    "mul",
    "qdiv",
    "cs_rank",
    "cs_zscore",
    "cs_demean",
    "cs_winsorize_5pct",
    "cs_resid",
    "delay",
    "ts_delta",
    "ts_return",
    "ts_mean",
    "ts_median",
    "ts_std",
    "ts_zscore",
    "ts_max_to_min",
    "ts_meanrank",
    "diff_sign",
    "ts_corr",
    "rolling_selmean_diff",
    "decay_linear",
    "mask_rank_high_50",
    "mask_rank_high_80",
    "mask_rank_low_20",
    "mask_sign_pos",
    "mask_sign_neg",
    "gate_nan",
    "gate_zero",
]
