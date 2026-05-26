from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .gene import FieldRule
from .utils import (
    dot_log,
    long_to_pivot,
    validate_long_format,
    validate_pivot_format,
)


# ---------------------------------------------------------------------------
# This module prepares the panel data before genetic search. Current alpha_gen
# expects YoY/QoQ/log/TTM/ratio style features to be prepared upstream, so the
# cache stores only current field matrices by default. Optional log matrices are
# kept for the standalone gplearn baseline.
# ---------------------------------------------------------------------------


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = ""):
    """Wrap an iterable with tqdm when progress display is requested."""

    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


@dataclass
class TransformCache:
    """In-memory matrices prepared for factor calculation.

    current:
        Keyed by (field, use_log). Main GA uses use_log=False because fields
        should be prepared upstream. use_log=True is optional compatibility for
        the gplearn baseline.
    label:
        Future return matrix used as the supervised target.
    tradeable:
        Tradeable mask. For mock data this is a 0/1 matrix.
    industry:
        Optional industry matrix for inspection or future filtering.
    """

    current: dict[tuple[str, bool], pd.DataFrame]
    label: pd.DataFrame
    tradeable: pd.DataFrame
    industry: pd.DataFrame | None
    field_rules: Mapping[str, FieldRule]

    def get_current(self, field: str, use_log: bool = False) -> pd.DataFrame:
        """Fetch current raw/log values for one field."""

        return self.current[(field, use_log)]


def load_panel(parquet_path: str | Path) -> pd.DataFrame:
    """Read a long-format parquet panel and validate the index contract."""

    data = pd.read_parquet(parquet_path)
    validate_long_format(data, require_time_component=True, strict=True)
    return data.sort_index()


def field_to_pivot(long_df: pd.DataFrame, field: str, dtype: str = "float32") -> pd.DataFrame:
    """Convert one long-format field to a Datetime x Contract pivot matrix."""

    if field not in long_df.columns:
        raise KeyError(f"field {field!r} is not in input data")

    pivot = long_to_pivot(long_df[[field]], factor_name=field)
    validate_pivot_format(pivot, require_time_component=True, strict=True)

    # Numeric factor matrices are cast to float32 to keep the local cache small.
    # Category/string matrices such as industry_code are handled separately.
    if pd.api.types.is_numeric_dtype(pivot.dtypes.iloc[0]):
        pivot = pivot.astype(dtype)
    return pivot


def _safe_log(pivot: pd.DataFrame, dtype: str = "float32") -> pd.DataFrame:
    """Signed log1p transform that keeps negative fundamentals finite."""

    logged = dot_log(pivot)
    logged.index.name = pivot.index.name
    logged.columns.name = pivot.columns.name
    return logged.astype(dtype)


def build_transform_cache(
    long_df: pd.DataFrame,
    field_rules: Mapping[str, FieldRule],
    *,
    label_col: str = "label_20d",
    tradeable_col: str = "is_tradeable",
    industry_col: str = "industry_code",
    period_to_days: Mapping[str, int] | None = None,
    dtype: str = "float32",
    show_progress: bool = False,
    build_log_cache: bool = False,
    extra_current_fields: Iterable[str] | None = None,
) -> TransformCache:
    """Build current field matrices used by later factor evaluation.

    Only fields present in `field_rules` are cached as candidate inputs. Columns
    such as label, tradeable mask and industry are handled separately.
    `extra_current_fields` is for neutralization controls, such as a Barra size
    exposure, that should be cached but not searched as candidate factors.
    """

    validate_long_format(long_df, require_time_component=True, strict=True)
    del period_to_days

    current: dict[tuple[str, bool], pd.DataFrame] = {}

    rule_items = _progress_iter(
        field_rules.items(),
        enabled=show_progress,
        total=len(field_rules),
        desc="build transform cache",
    )
    for field, rule in rule_items:
        if field not in long_df.columns:
            raise KeyError(f"field rule exists for {field!r}, but data column is missing")

        raw = field_to_pivot(long_df, field, dtype=dtype)
        current[(field, False)] = raw

        if build_log_cache and rule.allow_log:
            current[(field, True)] = _safe_log(raw, dtype=dtype)

    for field in extra_current_fields or []:
        if field in field_rules or (field, False) in current:
            continue
        if field not in long_df.columns:
            raise KeyError(f"extra current field {field!r} is missing from input data")
        current[(field, False)] = field_to_pivot(long_df, field, dtype=dtype)

    label = field_to_pivot(long_df, label_col, dtype=dtype)
    tradeable = field_to_pivot(long_df, tradeable_col, dtype=dtype)
    tradeable = tradeable.replace([np.inf, -np.inf], np.nan).fillna(0)
    tradeable = (tradeable > 0).astype("int8")

    industry = None
    if industry_col in long_df.columns:
        industry = long_to_pivot(long_df[[industry_col]], factor_name=industry_col)
        validate_pivot_format(industry, require_time_component=True, strict=True)

    return TransformCache(
        current=current,
        label=label,
        tradeable=tradeable,
        industry=industry,
        field_rules=field_rules,
    )

def cache_memory_usage_mb(cache: TransformCache) -> float:
    """Approximate memory used by cached matrices."""

    frames = list(cache.current.values()) + [cache.label, cache.tradeable]
    if cache.industry is not None:
        frames.append(cache.industry)
    bytes_used = sum(frame.memory_usage(index=True, deep=True).sum() for frame in frames)
    return float(bytes_used / 1024 / 1024)


def cache_summary(cache: TransformCache) -> dict[str, object]:
    """Small diagnostic summary for notebooks and smoke tests."""

    dates = cache.label.index
    contracts = cache.label.columns
    return {
        "n_current": len(cache.current),
        "n_dates": int(len(dates)),
        "n_contracts": int(len(contracts)),
        "start": str(dates.min()),
        "end": str(dates.max()),
        "memory_mb": round(cache_memory_usage_mb(cache), 3),
    }
