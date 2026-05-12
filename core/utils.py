from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


DATETIME_LEVEL = "Datetime"
CONTRACT_LEVEL = "Contract"
MULTI_INDEX = [DATETIME_LEVEL, CONTRACT_LEVEL]


# ---------------------------------------------------------------------------
# Local utility layer copied and trimmed from the alpha_factory APIs used by
# alpha_gen. Keep shared data-format helpers, basic factor operators and
# rolling-window logic here so the rest of the project has no runtime
# dependency on alpha_factory.
# ---------------------------------------------------------------------------


def _as_datetime(value: Any) -> pd.Timestamp:
    """Parse integer/string/timestamp date inputs used by rolling helpers."""

    if isinstance(value, pd.Timestamp):
        return value
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d")
    return pd.to_datetime(value)


def _has_time_component(index: pd.DatetimeIndex) -> bool:
    return bool(((index.hour != 0) | (index.minute != 0) | (index.second != 0)).any())


def _check_datetime_index(index: pd.Index, *, require_time_component: bool, strict: bool) -> bool:
    if not isinstance(index, pd.DatetimeIndex):
        if strict:
            raise TypeError("Datetime index must be a pandas DatetimeIndex")
        return False
    if require_time_component and not _has_time_component(index):
        if strict:
            raise ValueError("Datetime index must include an intraday time component")
        return False
    return True


def _check_contract_index(index: pd.Index, *, strict: bool) -> bool:
    if len(index) == 0:
        if strict:
            raise ValueError("Contract index must not be empty")
        return False
    if index.hasnans:
        if strict:
            raise ValueError("Contract index contains NaN values")
        return False
    return True


def validate_long_format(
    long_df: pd.DataFrame,
    require_time_component: bool = True,
    strict: bool = False,
) -> bool:
    """Validate long panel format: MultiIndex[Datetime, Contract]."""

    if not isinstance(long_df, pd.DataFrame):
        if strict:
            raise TypeError("input must be a pandas DataFrame")
        return False
    if not isinstance(long_df.index, pd.MultiIndex):
        if strict:
            raise TypeError("long format requires a MultiIndex")
        return False
    if list(long_df.index.names[:2]) != MULTI_INDEX:
        if strict:
            raise ValueError(f"long format index names must be {MULTI_INDEX}")
        return False

    datetime_values = long_df.index.get_level_values(DATETIME_LEVEL)
    contract_values = long_df.index.get_level_values(CONTRACT_LEVEL)
    if not _check_datetime_index(
        pd.DatetimeIndex(datetime_values),
        require_time_component=require_time_component,
        strict=strict,
    ):
        return False
    if not _check_contract_index(pd.Index(contract_values), strict=strict):
        return False
    return True


def validate_pivot_format(
    pivot_df: pd.DataFrame,
    require_time_component: bool = True,
    strict: bool = False,
) -> bool:
    """Validate pivot factor format: Datetime index x Contract columns."""

    if not isinstance(pivot_df, pd.DataFrame):
        if strict:
            raise TypeError("input must be a pandas DataFrame")
        return False
    if not _check_datetime_index(
        pivot_df.index,
        require_time_component=require_time_component,
        strict=strict,
    ):
        return False
    if pivot_df.columns.hasnans:
        if strict:
            raise ValueError("pivot columns contain NaN contract values")
        return False
    return True


def _ensure_long_sorted(long_df: pd.DataFrame) -> pd.DataFrame:
    if not long_df.index.is_monotonic_increasing:
        return long_df.sort_index()
    return long_df


def _ensure_pivot_sorted(pivot_df: pd.DataFrame) -> pd.DataFrame:
    if not pivot_df.index.is_monotonic_increasing:
        return pivot_df.sort_index()
    return pivot_df


def _long_to_pivot_no_validate(long_df: pd.DataFrame, factor_name: str | None = None) -> pd.DataFrame:
    frame = _ensure_long_sorted(long_df)
    if factor_name is None:
        if frame.shape[1] != 1:
            raise ValueError("factor_name is required when long_df has multiple columns")
        factor_name = str(frame.columns[0])
    pivot = frame[factor_name].unstack(CONTRACT_LEVEL)
    pivot.index.name = DATETIME_LEVEL
    return _ensure_pivot_sorted(pivot)


def long_to_pivot(long_df: pd.DataFrame, factor_name: str | None = None) -> pd.DataFrame:
    """Convert one long-format factor column to a pivot matrix."""

    validate_long_format(long_df, require_time_component=True, strict=True)
    return _long_to_pivot_no_validate(long_df, factor_name=factor_name)


def _pivot_to_long_no_validate(pivot_df: pd.DataFrame, factor_name: str = "factor") -> pd.DataFrame:
    frame = _ensure_pivot_sorted(pivot_df)
    long_df = frame.stack(dropna=False).to_frame(factor_name)
    long_df.index = long_df.index.set_names(MULTI_INDEX)
    return _ensure_long_sorted(long_df)


def pivot_to_long(pivot_df: pd.DataFrame, factor_name: str = "factor") -> pd.DataFrame:
    """Convert one pivot matrix to long panel format."""

    validate_pivot_format(pivot_df, require_time_component=True, strict=True)
    return _pivot_to_long_no_validate(pivot_df, factor_name=factor_name)


def dot_log(df: pd.DataFrame, eps: float = 0) -> pd.DataFrame:
    """Element-wise log; callers should mask non-positive values when needed."""

    return np.log(df + eps)


def dot_div(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Element-wise division with pandas alignment."""

    return df1.div(df2)


def dot_add(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Element-wise addition with pandas alignment."""

    return df1.add(df2)


def dot_sub(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Element-wise subtraction with pandas alignment."""

    return df1.sub(df2)


def cs_rank(df: pd.DataFrame, method: str = "average", pct: bool = True) -> pd.DataFrame:
    """Cross-sectional rank by date."""

    return df.rank(axis=1, method=method, pct=pct)


def cs_zscore(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Cross-sectional z-score by date."""

    mean = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, skipna=True)
    return df.sub(mean, axis=0).div(std + eps, axis=0)


def _residualize_row(y: pd.Series, x: pd.Series, eps: float) -> pd.Series:
    """Return cross-sectional residuals for one date."""

    out = pd.Series(np.nan, index=y.index, dtype="float64")
    tmp = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3:
        return out

    x_values = tmp["x"].to_numpy(dtype="float64")
    y_values = tmp["y"].to_numpy(dtype="float64")
    x_centered = x_values - x_values.mean()
    y_centered = y_values - y_values.mean()
    denom = float(np.dot(x_centered, x_centered))

    if denom <= eps:
        residual = y_centered
    else:
        beta = float(np.dot(x_centered, y_centered) / denom)
        alpha = float(y_values.mean() - beta * x_values.mean())
        residual = y_values - (alpha + beta * x_values)

    out.loc[tmp.index] = residual
    return out


def cs_resi(
    df: pd.DataFrame,
    by: pd.DataFrame,
    parallel: bool = True,
    max_workers: int | None = None,
    use_tqdm: bool = False,
) -> pd.DataFrame:
    """Cross-sectional residualization: residual(df ~ by) for each date.

    The optional parallel/progress arguments are kept for compatibility with
    older call signatures. The local implementation is serial to avoid
    introducing extra runtime dependencies in alpha_gen.
    """

    del parallel, max_workers, use_tqdm
    aligned_y, aligned_x = df.align(by, join="inner", axis=None)
    result = pd.DataFrame(index=aligned_y.index, columns=aligned_y.columns, dtype="float64")
    for dt in aligned_y.index:
        result.loc[dt] = _residualize_row(aligned_y.loc[dt], aligned_x.loc[dt], eps=1e-12)
    return result.reindex(index=df.index, columns=df.columns)


def get_rolling_windows(
    all_dates: Iterable[Any],
    train_start_date: Any,
    test_start_date: Any,
    stride: int = 120,
    rolling_type: str = "sliding",
    horizon: int = 20,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Build chronological rolling train/test windows on the real date index.

    For a test window starting at index i, the train window ends at
    i - horizon - 1, which leaves a gap of `horizon` trading dates between train
    and test. Returned windows preserve the exact timestamps in `all_dates`,
    for example `YYYY-MM-DD 15:00:00`. The default matches label_20d.
    """

    dates = pd.DatetimeIndex(pd.to_datetime(list(all_dates))).sort_values().unique()
    if len(dates) == 0:
        return []
    if stride <= 0:
        raise ValueError("stride must be positive")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if rolling_type not in {"sliding", "expanding"}:
        raise ValueError("rolling_type must be 'sliding' or 'expanding'")

    base_train_start = int(dates.searchsorted(_as_datetime(train_start_date), side="left"))
    base_test_start = int(dates.searchsorted(_as_datetime(test_start_date), side="left"))
    if base_train_start >= len(dates) or base_test_start >= len(dates):
        return []
    if base_train_start >= base_test_start:
        raise ValueError("train_start_date must be before test_start_date")

    train_span = base_test_start - base_train_start - horizon
    if train_span <= 0:
        raise ValueError("horizon leaves no train dates")

    windows: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    test_start_idx = base_test_start
    while test_start_idx < len(dates):
        test_end_idx = min(test_start_idx + stride, len(dates)) - 1
        if test_end_idx < test_start_idx:
            break

        if rolling_type == "sliding":
            train_start_idx = max(0, test_start_idx - horizon - train_span)
        else:
            train_start_idx = base_train_start
        train_end_idx = test_start_idx - horizon - 1
        if train_end_idx < train_start_idx:
            break

        train_window = pd.DatetimeIndex(dates[train_start_idx : train_end_idx + 1])
        test_window = pd.DatetimeIndex(dates[test_start_idx : test_end_idx + 1])
        windows.append((train_window, test_window))
        test_start_idx += stride

    return windows
