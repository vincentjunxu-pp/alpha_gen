from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


DATETIME_LEVEL = "Datetime"
CONTRACT_LEVEL = "Contract"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        requested = device
    elif str(device) == "auto":
        requested = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    return requested


def validate_long_panel(
    panel: pd.DataFrame,
    *,
    datetime_level: str = DATETIME_LEVEL,
    contract_level: str = CONTRACT_LEVEL,
) -> None:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel must use a MultiIndex")
    missing = {datetime_level, contract_level} - set(panel.index.names)
    if missing:
        raise ValueError(f"panel index is missing required levels: {sorted(missing)}")
    if panel.index.has_duplicates:
        raise ValueError("panel index must not contain duplicate Datetime/Contract rows")
    datetimes = pd.to_datetime(panel.index.get_level_values(datetime_level), errors="coerce")
    if pd.isna(datetimes).any():
        raise ValueError("Datetime index level contains values that cannot be converted to datetime")
    contracts = panel.index.get_level_values(contract_level)
    if pd.isna(contracts).any():
        raise ValueError("Contract index level contains missing values")


def _as_datetime_index(values: Iterable[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values)))


@dataclass
class CudaFactorContext:
    """Long-panel to CUDA tensor bridge for free GP factor evaluation.

    The context keeps the source long table in host memory. Individual fields
    are pivoted to [date, contract] tensors only when requested, then optionally
    cached on the target device with an LRU limit.
    """

    panel: pd.DataFrame
    label_col: str = "label_20d"
    tradeable_col: str = "is_tradeable"
    industry_col: str | None = "industry_code"
    candidate_fields: Sequence[str] | None = None
    exclude_fields: Sequence[str] = ()
    exclude_prefixes: Sequence[str] = ()
    device: str | torch.device = "auto"
    dtype: torch.dtype = torch.float32
    cache_on_device: bool = True
    max_cached_tensors: int | None = 256
    datetime_level: str = DATETIME_LEVEL
    contract_level: str = CONTRACT_LEVEL
    _tensor_cache: OrderedDict[tuple[object, ...], torch.Tensor] = field(default_factory=OrderedDict, init=False)
    _industry_labels: tuple[object, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        validate_long_panel(
            self.panel,
            datetime_level=self.datetime_level,
            contract_level=self.contract_level,
        )
        if self.label_col not in self.panel.columns:
            raise KeyError(f"label column {self.label_col!r} is missing")
        if self.tradeable_col not in self.panel.columns:
            raise KeyError(f"tradeable column {self.tradeable_col!r} is missing")
        if self.max_cached_tensors is not None and self.max_cached_tensors < 0:
            raise ValueError("max_cached_tensors must be non-negative or None")

        self.device = resolve_device(self.device)
        self.panel = self.panel.sort_index()
        self.dates = pd.DatetimeIndex(
            pd.to_datetime(self.panel.index.get_level_values(self.datetime_level).unique()),
            name=self.datetime_level,
        )
        self.contracts = pd.Index(
            self.panel.index.get_level_values(self.contract_level).unique(),
            name=self.contract_level,
        )
        self.shape = (len(self.dates), len(self.contracts))
        self._candidate_fields = self._resolve_candidate_fields(self.candidate_fields)

    @property
    def available_columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.panel.columns)

    @property
    def searchable_fields(self) -> tuple[str, ...]:
        return self._candidate_fields

    @property
    def industry_labels(self) -> tuple[object, ...]:
        return self._industry_labels

    def _resolve_candidate_fields(self, candidate_fields: Sequence[str] | None) -> tuple[str, ...]:
        excluded = {self.label_col, self.tradeable_col, *self.exclude_fields}
        if self.industry_col:
            excluded.add(self.industry_col)
        exclude_prefixes = tuple(str(p) for p in self.exclude_prefixes if p)

        def _is_excluded(field_name: str) -> bool:
            if field_name in excluded:
                return True
            return any(field_name.startswith(prefix) for prefix in exclude_prefixes)

        if candidate_fields is not None:
            # Deduplicate while preserving order, then filter through the same exclusion rules
            fields = tuple(dict.fromkeys(str(field) for field in candidate_fields))
            fields = tuple(f for f in fields if not _is_excluded(f))
            missing = [field for field in fields if field not in self.panel.columns]
            if missing:
                raise KeyError(f"candidate fields are missing from panel: {missing[:5]}")
            non_numeric = [field for field in fields if not pd.api.types.is_numeric_dtype(self.panel[field])]
            if non_numeric:
                raise TypeError(f"candidate fields must be numeric: {non_numeric[:5]}")
            return fields

        output: list[str] = []
        for column in self.panel.columns:
            field_name = str(column)
            if _is_excluded(field_name):
                continue
            if pd.api.types.is_numeric_dtype(self.panel[column]):
                output.append(field_name)
        return tuple(output)

    def _column_to_frame(self, column: str, *, numeric: bool) -> pd.DataFrame:
        if column not in self.panel.columns:
            raise KeyError(f"column {column!r} is missing from panel")
        series = self.panel[column]
        if numeric:
            series = pd.to_numeric(series, errors="coerce")
        frame = series.unstack(self.contract_level)
        frame = frame.reindex(index=self.dates, columns=self.contracts)
        frame.index.name = self.datetime_level
        frame.columns.name = self.contract_level
        return frame

    def _cache_get(self, key: tuple[object, ...]) -> torch.Tensor | None:
        if not self.cache_on_device:
            return None
        tensor = self._tensor_cache.get(key)
        if tensor is not None:
            self._tensor_cache.move_to_end(key)
        return tensor

    def _cache_put(self, key: tuple[object, ...], tensor: torch.Tensor) -> torch.Tensor:
        if not self.cache_on_device or self.max_cached_tensors == 0:
            return tensor
        self._tensor_cache[key] = tensor
        self._tensor_cache.move_to_end(key)
        if self.max_cached_tensors is not None:
            while len(self._tensor_cache) > self.max_cached_tensors:
                self._tensor_cache.popitem(last=False)
        return tensor

    def _frame_to_tensor(self, key: tuple[object, ...], frame: pd.DataFrame) -> torch.Tensor:
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        array = frame.to_numpy(dtype=np.float32, copy=False)
        tensor = torch.as_tensor(array, dtype=self.dtype, device=self.device)
        return self._cache_put(key, tensor)

    def get_field(self, field_name: str) -> torch.Tensor:
        """Return a numeric field tensor with shape [date, contract]."""

        if field_name not in self.panel.columns:
            raise KeyError(f"field {field_name!r} is missing from panel")
        if not pd.api.types.is_numeric_dtype(self.panel[field_name]):
            raise TypeError(f"field {field_name!r} must be numeric")
        frame = self._column_to_frame(field_name, numeric=True)
        return self._frame_to_tensor(("field", field_name), frame)

    def label(self) -> torch.Tensor:
        return self.get_field(self.label_col)

    def tradeable(self) -> torch.Tensor:
        cached = self._cache_get(("tradeable_bool", self.tradeable_col))
        if cached is not None:
            return cached.to(torch.bool)
        raw = self.get_field(self.tradeable_col)
        tradeable = torch.isfinite(raw) & (raw > 0)
        if self.cache_on_device and self.max_cached_tensors != 0:
            self._cache_put(("tradeable_bool", self.tradeable_col), tradeable)
        return tradeable

    def industry_codes(self) -> torch.Tensor:
        if not self.industry_col:
            raise ValueError("industry_col is not configured")
        if self.industry_col not in self.panel.columns:
            raise KeyError(f"industry column {self.industry_col!r} is missing")
        cached = self._cache_get(("industry_codes", self.industry_col))
        if cached is not None:
            return cached.to(torch.long)

        frame = self._column_to_frame(self.industry_col, numeric=False)
        codes, uniques = pd.factorize(frame.to_numpy(dtype=object).ravel(), sort=True, use_na_sentinel=True)
        self._industry_labels = tuple(uniques.tolist())
        tensor = torch.as_tensor(codes.reshape(self.shape), dtype=torch.long, device=self.device)
        return self._cache_put(("industry_codes", self.industry_col), tensor)

    def date_positions(self, dates: Iterable[object] | pd.DatetimeIndex | None) -> torch.Tensor | None:
        if dates is None:
            return None
        date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        positions = self.dates.get_indexer(date_index)
        if (positions < 0).any():
            missing = date_index[positions < 0]
            raise KeyError(f"dates not found in context: {missing[:3].tolist()}")
        return torch.as_tensor(positions, dtype=torch.long, device=self.device)

    def take_dates(self, values: torch.Tensor, dates: Iterable[object] | pd.DatetimeIndex | None) -> torch.Tensor:
        positions = self.date_positions(dates)
        if positions is None:
            return values
        return values.index_select(0, positions)

    def const_like(self, value: float) -> torch.Tensor:
        return torch.full(self.shape, float(value), dtype=self.dtype, device=self.device)

    def tensor_to_frame(self, values: torch.Tensor) -> pd.DataFrame:
        if tuple(values.shape) != self.shape:
            raise ValueError(f"values shape {tuple(values.shape)} does not match context shape {self.shape}")
        frame = pd.DataFrame(
            values.detach().cpu().numpy(),
            index=self.dates,
            columns=self.contracts,
        )
        frame.index.name = self.datetime_level
        frame.columns.name = self.contract_level
        return frame.astype("float32")

    def clear_cache(self) -> None:
        self._tensor_cache.clear()

    def cache_keys(self) -> tuple[tuple[object, ...], ...]:
        return tuple(self._tensor_cache.keys())

    def cache_info(self) -> dict[str, object]:
        return {
            "enabled": bool(self.cache_on_device and self.max_cached_tensors != 0),
            "size": len(self._tensor_cache),
            "max_cached_tensors": self.max_cached_tensors,
            "keys": self.cache_keys(),
        }

    def summary(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "n_dates": len(self.dates),
            "n_contracts": len(self.contracts),
            "n_columns": len(self.panel.columns),
            "n_searchable_fields": len(self.searchable_fields),
            "label_col": self.label_col,
            "tradeable_col": self.tradeable_col,
            "industry_col": self.industry_col,
            "exclude_fields": tuple(self.exclude_fields),
            "exclude_prefixes": tuple(self.exclude_prefixes),
            "cache": self.cache_info(),
        }


__all__ = [
    "DATETIME_LEVEL",
    "CONTRACT_LEVEL",
    "resolve_device",
    "validate_long_panel",
    "CudaFactorContext",
]
