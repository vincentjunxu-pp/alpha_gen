from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Build the real TMT panel expected by alpha_gen.
#
# Target output:
#   index   = MultiIndex(["Datetime", "Contract"])
#   columns = market_cap, industry_code, is_tradeable, label_20d, optional close,
#             and selected numeric candidate fields.
#
# This script intentionally stays explicit instead of hiding the whole process
# behind a large framework. The real data source may be parquet/csv/pickle and
# may be either a daily long panel or a wide Datetime x Contract matrix, so the
# code below standardizes those layouts step by step.
# ---------------------------------------------------------------------------


DATETIME = "Datetime"
CONTRACT = "Contract"

RESERVED_COLUMNS = {
    "industry_code",
    "is_tradeable",
    "label_20d",
}

DEFAULT_TMT_VALUES = ("电子", "计算机", "通信", "传媒")
DEFAULT_FINANCIAL_TABLES = {"rqf_income", "rqf_balancesheet", "rqf_cashflow"}
DEFAULT_DERIVATIVE_TABLES = {
    "rqf_der_valuation",
    "rqf_der_operating",
    "rqf_der_cashflow",
    "rqf_der_financial",
    "rqf_der_growth",
}

LOG_DENY_KEYWORDS = (
    "ratio",
    "rate",
    "pct",
    "percent",
    "score",
    "rank",
    "beta",
    "alpha",
    "turnover",
    "margin",
    "roe",
    "roa",
    "roic",
)

PCT_DENY_KEYWORDS = (
    "ratio",
    "rate",
    "pct",
    "percent",
    "score",
    "rank",
    "roe",
    "roa",
    "roic",
)


@dataclass(frozen=True)
class SourceInfo:
    """Metadata carried with each candidate field."""

    table: str
    kind: str


def _read_table(path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    """Read one local file.

    Keep the supported formats small and common. Database extraction should be
    done upstream into one of these files so this script remains reproducible.
    """

    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()

    if fmt in {"parquet", "pq"}:
        return pd.read_parquet(path)
    if fmt in {"pkl", "pickle"}:
        return pd.read_pickle(path)
    if fmt in {"csv"}:
        return pd.read_csv(path)
    if fmt in {"feather", "ipc"}:
        return pd.read_feather(path)
    raise ValueError(f"unsupported file format for {path}: {fmt!r}")


def _normalize_datetime_index(index: pd.Index, close_time: str = "15:00:00") -> pd.DatetimeIndex:
    """Convert an index to DatetimeIndex and add close time to pure dates."""

    dt = pd.to_datetime(index)
    dt = pd.DatetimeIndex(dt)

    # alpha_gen validates that Datetime has an intraday component. If the source
    # is daily dates at 00:00:00, shift them to the close timestamp.
    if (dt == dt.normalize()).all():
        dt = dt + pd.Timedelta(close_time)

    dt.name = DATETIME
    return dt


def _standardize_panel_index(df: pd.DataFrame, close_time: str = "15:00:00") -> pd.DataFrame:
    """Return a long panel indexed by Datetime/Contract."""

    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        if DATETIME not in names or CONTRACT not in names:
            if len(names) < 2:
                raise ValueError("MultiIndex panel needs Datetime and Contract levels")
            names[0] = DATETIME
            names[1] = CONTRACT
            out.index = out.index.set_names(names)
        out = out.reorder_levels([DATETIME, CONTRACT]).sort_index()
    elif {DATETIME, CONTRACT}.issubset(out.columns):
        out[DATETIME] = pd.to_datetime(out[DATETIME])
        out = out.set_index([DATETIME, CONTRACT]).sort_index()
    else:
        raise ValueError("panel data must have a Datetime/Contract MultiIndex or columns")

    dt = _normalize_datetime_index(out.index.get_level_values(DATETIME), close_time=close_time)
    contracts = out.index.get_level_values(CONTRACT).astype(str)
    out.index = pd.MultiIndex.from_arrays([dt, contracts], names=[DATETIME, CONTRACT])
    return out.sort_index()


def _wide_to_long_col(wide: pd.DataFrame, name: str, close_time: str = "15:00:00") -> pd.DataFrame:
    """Convert a Datetime x Contract wide matrix to one long panel column."""

    out = wide.copy()

    # CSV files often keep Datetime as the first ordinary column.
    if not isinstance(out.index, pd.DatetimeIndex):
        date_col = DATETIME if DATETIME in out.columns else None
        if date_col is None:
            lowered = {str(col).lower(): col for col in out.columns}
            date_col = lowered.get("datetime") or lowered.get("date")
        if date_col is not None:
            out[date_col] = pd.to_datetime(out[date_col])
            out = out.set_index(date_col)

    out.index = _normalize_datetime_index(out.index, close_time=close_time)
    out.columns = out.columns.astype(str)
    out.columns.name = CONTRACT
    stacked = _stack_wide(out, name=name)
    stacked.index = stacked.index.set_names([DATETIME, CONTRACT])
    return stacked.to_frame()


def _stack_wide(wide: pd.DataFrame, name: str) -> pd.Series:
    """Stack a wide matrix while keeping NaNs and supporting pandas versions.

    pandas 2.1 introduced the future stack implementation and warns on the old
    `stack(dropna=False)` path. The fallback keeps compatibility with older
    versions that do not know `future_stack`.
    """

    try:
        stacked = wide.stack(future_stack=True)
    except TypeError:
        stacked = wide.stack(dropna=False)
    return stacked.rename(name)


def _load_value_source(spec: dict[str, Any], default_name: str, close_time: str) -> pd.DataFrame:
    """Load market cap, industry, close, label or tradeable source."""

    raw = _read_table(spec["path"], spec.get("format"))
    layout = str(spec.get("layout", "auto")).lower()
    name = str(spec.get("name") or default_name)
    column = spec.get("column")

    if layout == "wide" or (layout == "auto" and not isinstance(raw.index, pd.MultiIndex) and column is None):
        return _wide_to_long_col(raw, name=name, close_time=close_time)

    panel = _standardize_panel_index(raw, close_time=close_time)
    if column is None:
        if name in panel.columns:
            column = name
        elif len(panel.columns) == 1:
            column = panel.columns[0]
        else:
            raise ValueError(f"{default_name} source must set column when panel has multiple columns")

    return panel[[column]].rename(columns={column: name})


def _load_factor_table(spec: dict[str, Any], close_time: str) -> tuple[pd.DataFrame, dict[str, SourceInfo]]:
    """Load one factor table as a Datetime/Contract panel.

    The usual real tables should already be long panels with many indicator
    columns. A single wide matrix is also supported by setting value_name.
    """

    table_name = str(spec.get("name") or Path(spec["path"]).stem)
    kind = str(spec.get("kind") or _infer_table_kind(table_name))
    raw = _read_table(spec["path"], spec.get("format"))
    layout = str(spec.get("layout", "auto")).lower()
    value_name = spec.get("value_name")

    if layout == "wide" or value_name is not None:
        field_name = str(value_name or table_name)
        panel = _wide_to_long_col(raw, name=field_name, close_time=close_time)
    else:
        panel = _standardize_panel_index(raw, close_time=close_time)

    include = spec.get("include_columns")
    exclude = set(spec.get("exclude_columns", [])) | RESERVED_COLUMNS

    if include:
        keep = [col for col in include if col in panel.columns]
    else:
        keep = [col for col in panel.columns if col not in exclude]

    # Only numeric columns can become GA inputs. Strings and categories should
    # stay in special columns such as industry_code, not in field_rules.
    numeric_keep = [col for col in keep if pd.api.types.is_numeric_dtype(panel[col])]
    panel = panel[numeric_keep].copy()

    rename_prefix = spec.get("rename_prefix")
    if rename_prefix:
        panel = panel.rename(columns={col: f"{rename_prefix}{col}" for col in panel.columns})

    source = {str(col): SourceInfo(table=table_name, kind=kind) for col in panel.columns}
    return panel, source


def _infer_table_kind(table_name: str) -> str:
    """Infer whether a table is derived daily data or raw financial statement data."""

    lower = table_name.lower()
    if lower in DEFAULT_FINANCIAL_TABLES:
        return "financial_statement"
    if lower in DEFAULT_DERIVATIVE_TABLES or lower.startswith("rqf_der_"):
        return "derivative"
    return "other"


def _filter_index(df: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    """Restrict a panel to the target TMT index."""

    return df.reindex(index)


def _filter_contracts(df: pd.DataFrame, contracts: pd.Index) -> pd.DataFrame:
    """Keep only target contracts while preserving source dates."""

    mask = df.index.get_level_values(CONTRACT).isin(contracts.astype(str))
    return df.loc[mask].sort_index()


def _ffill_by_contract(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Forward-fill low-frequency fields inside each stock only."""

    return df.groupby(level=CONTRACT, group_keys=False).ffill(limit=limit)


def _align_factor_table(
    df: pd.DataFrame,
    target_index: pd.MultiIndex,
    *,
    ffill: bool = False,
    ffill_limit: int | None = None,
) -> pd.DataFrame:
    """Align a factor table to the target daily TMT index.

    For low-frequency financial statement tables, include both source report
    dates and target trading dates before ffill. Otherwise disclosure rows that
    are not exactly present in the daily industry index would be discarded.
    """

    if not ffill:
        return df.reindex(target_index)

    contracts = pd.Index(target_index.get_level_values(CONTRACT).unique())
    filtered = _filter_contracts(df, contracts)
    combined_index = filtered.index.union(target_index).sort_values()
    filled = _ffill_by_contract(filtered.reindex(combined_index), limit=ffill_limit)
    return filled.reindex(target_index)


def _make_forward_return(close_long: pd.DataFrame, horizon: int, label_name: str) -> pd.DataFrame:
    """Compute future horizon return from an adjusted close column."""

    close_col = close_long.columns[0]
    close_wide = close_long[close_col].unstack(CONTRACT).sort_index()
    label_wide = close_wide.shift(-horizon).div(close_wide) - 1.0
    label_wide.index.name = DATETIME
    label_wide.columns.name = CONTRACT
    label = _stack_wide(label_wide, name=label_name).to_frame()
    label.index = label.index.set_names([DATETIME, CONTRACT])
    return label


def _coverage_stats(panel: pd.DataFrame, source: dict[str, SourceInfo]) -> pd.DataFrame:
    """Calculate field diagnostics used for automatic metadata drafting."""

    rows: list[dict[str, Any]] = []
    n_rows = max(len(panel), 1)

    for col in panel.columns:
        s = pd.to_numeric(panel[col], errors="coerce")
        non_na = s.notna()
        finite = np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))
        valid_values = s[non_na]
        src = source.get(str(col), SourceInfo(table="", kind="other"))
        rows.append(
            {
                "field": str(col),
                "table": src.table,
                "kind": src.kind,
                "coverage": float(non_na.sum() / n_rows),
                "finite_ratio": float(finite.sum() / n_rows),
                "positive_ratio": float((valid_values > 0).sum() / max(len(valid_values), 1)),
                "zero_ratio": float((valid_values == 0).sum() / max(len(valid_values), 1)),
                "unique_count": int(valid_values.nunique(dropna=True)),
                "dtype": str(panel[col].dtype),
            }
        )

    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats = stats.sort_values(["coverage", "field"], ascending=[False, True]).reset_index(drop=True)
    return stats


def _keyword_hit(name: str, keywords: Iterable[str]) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in keywords)


def _coverage_threshold(kind: str, config: dict[str, Any]) -> float:
    thresholds = config.get("coverage_thresholds", {})
    if kind in thresholds:
        return float(thresholds[kind])
    return float(thresholds.get("default", 0.70))


def _draft_field_rule(field: str, row: pd.Series) -> dict[str, bool]:
    """Draft a field rule from coverage/name statistics.

    This is intentionally conservative. The generated JSON is a starting point:
    users should still review economically important fields before production.
    """

    kind = str(row.get("kind", "other"))
    positive_ratio = float(row.get("positive_ratio", 0.0))
    allow_log = positive_ratio >= 0.95 and not _keyword_hit(field, LOG_DENY_KEYWORDS)
    allow_pct = not _keyword_hit(field, PCT_DENY_KEYWORDS)
    allow_std = kind != "financial_statement"

    return {
        "can_y": True,
        "can_x": True,
        "allow_log": bool(allow_log),
        "allow_current": True,
        "allow_lag": True,
        "allow_diff": True,
        "allow_pct": bool(allow_pct),
        "allow_std": bool(allow_std),
    }


def _build_metadata(
    panel: pd.DataFrame,
    stats: pd.DataFrame,
    selected_fields: list[str],
    config: dict[str, Any],
    output_name: str,
) -> dict[str, Any]:
    """Create the JSON metadata consumed by alpha_gen.core.gene."""

    field_rules: dict[str, dict[str, bool]] = {}

    # close is not required by the GA if label_20d already exists. When present,
    # keep it as an optional x-side market variable and never as y.
    if "close" in panel.columns:
        field_rules["close"] = {
            "can_y": False,
            "can_x": True,
            "allow_log": True,
            "allow_current": False,
            "allow_lag": False,
            "allow_diff": False,
            "allow_pct": True,
            "allow_std": True,
        }

    # market_cap is required for size neutralization. alpha_gen treats it as a
    # prepared size proxy, so log-transform it upstream if log-size is desired.
    field_rules["market_cap"] = {
        "can_y": False,
        "can_x": True,
        "allow_log": False,
        "allow_current": True,
        "allow_lag": False,
        "allow_diff": True,
        "allow_pct": True,
        "allow_std": True,
    }

    stats_by_field = stats.set_index("field") if not stats.empty else pd.DataFrame()
    for field in selected_fields:
        if field in field_rules:
            continue
        row = stats_by_field.loc[field]
        field_rules[field] = _draft_field_rule(field, row)

    dates = panel.index.get_level_values(DATETIME)
    contracts = panel.index.get_level_values(CONTRACT)
    numeric_fields = [
        str(col)
        for col in panel.columns
        if pd.api.types.is_numeric_dtype(panel[col])
    ]

    return {
        "dataset": output_name,
        "format": "parquet",
        "compression": "zstd",
        "index": [DATETIME, CONTRACT],
        "shape": [int(panel.shape[0]), int(panel.shape[1])],
        "n_dates": int(dates.nunique()),
        "n_contracts": int(contracts.nunique()),
        "start": str(dates.min()),
        "end": str(dates.max()),
        "notes": [
            "Generated by alpha_gen/data/build_real_tmt_panel.py.",
            "Field rules are automatically drafted from coverage/name heuristics and should be reviewed.",
            "Financial statement tables are forward-filled by stock before coverage testing when configured.",
            "close is optional unless label_20d must be generated from prices.",
        ],
        "tmt_values": list(config.get("tmt_values", DEFAULT_TMT_VALUES)),
        "coverage_thresholds": config.get("coverage_thresholds", {}),
        "selected_fields": selected_fields,
        "numeric_fields": numeric_fields,
        "categorical_fields": [
            str(col)
            for col in panel.columns
            if not pd.api.types.is_numeric_dtype(panel[col])
        ],
        "field_rules": field_rules,
    }


def _select_fields(stats: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Choose candidate fields that pass coverage and uniqueness checks."""

    if stats.empty:
        return []

    min_unique = int(config.get("min_unique_count", 20))
    max_fields = config.get("max_fields")
    manual_include = set(config.get("manual_include_fields", []))
    manual_exclude = set(config.get("manual_exclude_fields", []))

    selected: list[str] = []
    for _, row in stats.iterrows():
        field = str(row["field"])
        if field in manual_exclude or field in RESERVED_COLUMNS or field in {"market_cap", "close"}:
            continue

        threshold = _coverage_threshold(str(row["kind"]), config)
        pass_auto = float(row["coverage"]) >= threshold and int(row["unique_count"]) >= min_unique
        if pass_auto or field in manual_include:
            selected.append(field)

    if max_fields is not None:
        selected = selected[: int(max_fields)]
    return selected


def build_real_tmt_panel(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build panel, coverage stats and metadata from a config dictionary."""

    close_time = str(config.get("close_time", "15:00:00"))
    label_name = str(config.get("label_name", "label_20d"))
    industry_name = str(config.get("industry_name", "industry_code"))
    market_cap_name = str(config.get("market_cap_name", "market_cap"))

    if "industry" not in config:
        raise ValueError("config must contain an industry source")
    if "market_cap" not in config:
        raise ValueError("config must contain a market_cap source")

    industry = _load_value_source(config["industry"], default_name=industry_name, close_time=close_time)
    industry[industry_name] = industry[industry_name].astype("category")

    tmt_values = set(config.get("tmt_values", DEFAULT_TMT_VALUES))
    tmt_mask = industry[industry_name].astype(str).isin(tmt_values)
    tmt_index = industry.index[tmt_mask]
    if len(tmt_index) == 0:
        raise ValueError("TMT mask is empty; check industry values and config.tmt_values")

    industry = _filter_index(industry, tmt_index)
    market_cap = _filter_index(
        _load_value_source(config["market_cap"], default_name=market_cap_name, close_time=close_time),
        tmt_index,
    )
    market_cap = market_cap.rename(columns={market_cap.columns[0]: "market_cap"})

    parts: list[pd.DataFrame] = [
        market_cap,
        industry.rename(columns={industry_name: "industry_code"}),
    ]

    close_long = None
    if config.get("close"):
        close_long = _filter_index(_load_value_source(config["close"], default_name="close", close_time=close_time), tmt_index)
        close_long = close_long.rename(columns={close_long.columns[0]: "close"})
        parts.append(close_long)

    if config.get("label"):
        label = _filter_index(_load_value_source(config["label"], default_name=label_name, close_time=close_time), tmt_index)
        label = label.rename(columns={label.columns[0]: label_name})
    elif close_long is not None:
        horizon = int(config.get("label_horizon", 20))
        label = _filter_index(_make_forward_return(close_long, horizon=horizon, label_name=label_name), tmt_index)
    else:
        raise ValueError("provide config.label or config.close so label_20d can be available")

    parts.append(label)

    source: dict[str, SourceInfo] = {}
    ffill_limit = config.get("financial_ffill_limit", 300)

    for table_spec in config.get("factor_tables", []):
        table, table_source = _load_factor_table(table_spec, close_time=close_time)
        kind = str(next(iter(table_source.values())).kind) if table_source else _infer_table_kind(str(table_spec.get("name", "")))
        do_ffill = bool(table_spec.get("ffill", kind == "financial_statement"))
        table = _align_factor_table(
            table,
            tmt_index,
            ffill=do_ffill,
            ffill_limit=None if ffill_limit is None else int(ffill_limit),
        )

        parts.append(table)
        source.update(table_source)

    panel = pd.concat(parts, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated(keep="first")]
    panel = panel.sort_index()

    # Build a simple tradeable mask when no dedicated table is supplied. If the
    # user has a suspension/ST/limit-up mask, pass it as config.tradeable.
    if config.get("tradeable"):
        tradeable = _filter_index(_load_value_source(config["tradeable"], default_name="is_tradeable", close_time=close_time), tmt_index)
        tradeable = tradeable.rename(columns={tradeable.columns[0]: "is_tradeable"})
        panel = panel.join(tradeable, how="left")
        panel["is_tradeable"] = panel["is_tradeable"].fillna(0).astype("int8")
    else:
        panel["is_tradeable"] = panel["market_cap"].notna().astype("int8")

    # Cast numeric columns to float32 where possible to keep parquet and RAM
    # usage low. Keep is_tradeable and industry_code in compact non-float types.
    for col in panel.columns:
        if col == "is_tradeable":
            panel[col] = panel[col].fillna(0).astype("int8")
        elif col == "industry_code":
            panel[col] = panel[col].astype("category")
        elif pd.api.types.is_numeric_dtype(panel[col]):
            panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).astype("float32")

    candidate_panel = panel[[col for col in source if col in panel.columns]]
    stats = _coverage_stats(candidate_panel, source)
    selected_fields = _select_fields(stats, config)

    keep_cols = ["market_cap", "industry_code", "is_tradeable", label_name]
    if "close" in panel.columns and bool(config.get("keep_close", True)):
        keep_cols.append("close")
    keep_cols.extend(selected_fields)
    keep_cols = [col for col in keep_cols if col in panel.columns]
    panel = panel[keep_cols].copy()

    metadata = _build_metadata(
        panel=panel,
        stats=stats,
        selected_fields=selected_fields,
        config=config,
        output_name=Path(config.get("output_panel", "real_tmt_daily.parquet")).name,
    )

    return panel, stats, metadata


def write_template(path: str | Path) -> None:
    """Write a config template that can be edited for the real data location."""

    template = {
        "output_panel": "alpha_gen/data/panels/real_tmt_daily.parquet",
        "output_metadata": "alpha_gen/data/metadata/production/real_metadata.json",
        "output_coverage": "alpha_gen/artifacts/data_quality/real_field_coverage.csv",
        "close_time": "15:00:00",
        "tmt_values": list(DEFAULT_TMT_VALUES),
        "label_horizon": 20,
        "financial_ffill_limit": 300,
        "coverage_thresholds": {
            "derivative": 0.70,
            "financial_statement": 0.50,
            "other": 0.70,
            "default": 0.70,
        },
        "min_unique_count": 20,
        "max_fields": 150,
        "manual_include_fields": [],
        "manual_exclude_fields": [],
        "industry": {
            "path": "PATH/industry_wide.parquet",
            "layout": "wide",
            "name": "industry_code",
        },
        "market_cap": {
            "path": "PATH/market_cap_wide.parquet",
            "layout": "wide",
            "name": "market_cap",
        },
        "close": {
            "path": "PATH/adjusted_close_wide.parquet",
            "layout": "wide",
            "name": "close",
        },
        "factor_tables": [
            {
                "name": "rqf_der_valuation",
                "path": "PATH/rqf_der_valuation.parquet",
                "layout": "panel",
                "kind": "derivative",
            },
            {
                "name": "rqf_income",
                "path": "PATH/rqf_income.parquet",
                "layout": "panel",
                "kind": "financial_statement",
                "ffill": True,
            },
        ],
    }
    Path(path).write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alpha_gen real TMT panel and metadata.")
    parser.add_argument("--config", type=str, help="Path to a JSON config file.")
    parser.add_argument("--write-template", type=str, help="Write a config template to this path and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_template:
        write_template(args.write_template)
        print(f"template written: {args.write_template}")
        return

    if not args.config:
        raise SystemExit("please pass --config or --write-template")

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    panel, stats, metadata = build_real_tmt_panel(config)

    output_panel = Path(config.get("output_panel", "alpha_gen/data/panels/real_tmt_daily.parquet"))
    output_metadata = Path(
        config.get("output_metadata", "alpha_gen/data/metadata/production/real_metadata.json")
    )
    output_coverage = Path(
        config.get("output_coverage", "alpha_gen/artifacts/data_quality/real_field_coverage.csv")
    )
    output_panel.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_coverage.parent.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(output_panel, compression="zstd")
    stats.to_csv(output_coverage, index=False, encoding="utf-8-sig")
    output_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"panel: {output_panel} shape={panel.shape}")
    print(f"metadata: {output_metadata} field_rules={len(metadata['field_rules'])}")
    print(f"coverage: {output_coverage} fields={len(stats)}")


if __name__ == "__main__":
    main()
