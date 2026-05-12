from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SPECIAL_COLUMNS = {
    "industry_code",
    "is_tradeable",
    "tradeable",
    "label",
    "label_20d",
}

SIZE_FIELDS = {
    "market_cap",
    "market_cap_2",
    "market_cap_3",
    "a_share_market_val",
    "a_share_market_val_in_circulation",
}

PRICE_RELATED_FIELDS = {
    "pe_ratio_lyr",
    "pe_ratio_ttm",
    "ep_ratio_lyr",
    "ep_ratio_ttm",
    "pcf_ratio_total_lyr",
    "pcf_ratio_total_ttm",
    "pcf_ratio_lyr",
    "pcf_ratio_ttm",
    "cfp_ratio_lyr",
    "cfp_ratio_ttm",
    "pb_ratio_lyr",
    "pb_ratio_ttm",
    "pb_ratio_lf",
    "book_to_market_ratio_lyr",
    "book_to_market_ratio_ttm",
    "book_to_market_ratio_lf",
    "ps_ratio_lyr",
    "ps_ratio_ttm",
    "sp_ratio_lyr",
    "sp_ratio_ttm",
    "peg_ratio_lyr",
    "peg_ratio_ttm",
    "dividend_yield_ttm",
    "market_leverage_lyr",
    "market_leverage_ttm",
    "market_leverage_lf",
    "ev_lyr",
    "ev_ttm",
    "ev_lf",
    "ev_no_cash_lyr",
    "ev_no_cash_ttm",
    "ev_no_cash_lf",
    "ev_to_ebitda_lyr",
    "ev_to_ebitda_ttm",
    "ev_no_cash_to_ebit_lyr",
    "ev_no_cash_to_ebit_ttm",
}

IDENTITY_FIELDS = {
    "industry_code",
}

TRADEABLE_FIELDS = {
    "is_tradeable",
    "tradeable",
}


def _base_rule(
    *,
    can_y: bool,
    can_x: bool,
    allow_log: bool,
    allow_current: bool = True,
    allow_lag: bool = False,
    allow_diff: bool = True,
    allow_pct: bool = True,
    allow_std: bool = True,
) -> dict[str, bool]:
    return {
        "can_y": can_y,
        "can_x": can_x,
        "allow_log": allow_log,
        "allow_current": allow_current,
        "allow_lag": allow_lag,
        "allow_diff": allow_diff,
        "allow_pct": allow_pct,
        "allow_std": allow_std,
    }


def _is_ratio_like(field: str) -> bool:
    tokens = (
        "_ratio",
        "ratio_",
        "_rate",
        "rate_",
        "_margin",
        "margin_",
        "_multiplier",
        "multiplier",
        "_multiple",
        "multiple",
        "_yield",
        "yield",
        "_turnover",
        "turnover_",
        "_cycle",
        "cycle_",
        "_days",
        "days_",
        "_to_",
        "pe_ratio",
        "pb_ratio",
        "pcf_ratio",
        "ps_ratio",
        "sp_ratio",
        "ep_ratio",
        "cfp_ratio",
        "peg_ratio",
    )
    return any(token in field for token in tokens)


def _is_price_related(field: str) -> bool:
    prefixes = (
        "pe_ratio",
        "ep_ratio",
        "pcf_ratio",
        "cfp_ratio",
        "pb_ratio",
        "book_to_market_ratio",
        "ps_ratio",
        "sp_ratio",
        "peg_ratio",
        "ev_",
        "ev_to_",
        "ev_no_cash",
        "market_leverage",
    )
    return field in PRICE_RELATED_FIELDS or field in SIZE_FIELDS or field.startswith(prefixes)


def _is_growth_like(field: str) -> bool:
    return "growth" in field or field.startswith("inc_")


def _is_per_share(field: str) -> bool:
    return "per_share" in field or field.endswith("_ps")


def _is_cashflow_amount(field: str) -> bool:
    return field.startswith("cash_") or "cash_flow" in field or "cashflow" in field or field in {"fcff", "fcfe", "ocf"}


def _is_income_statement_amount(field: str) -> bool:
    tokens = (
        "revenue",
        "income",
        "profit",
        "expense",
        "cost",
        "tax",
        "ebit",
        "ebitda",
        "sales",
        "r_n_d",
        "depreciation",
        "amortization",
    )
    return any(token in field for token in tokens)


def _is_balance_sheet_amount(field: str) -> bool:
    tokens = (
        "asset",
        "liabilit",
        "equity",
        "receivable",
        "payable",
        "inventory",
        "deposit",
        "loan",
        "debt",
        "capital",
        "reserve",
        "goodwill",
        "cash_equivalent",
        "working_capital",
        "treasury_stock",
        "prepayment",
        "deferred",
    )
    return any(token in field for token in tokens)


def infer_field_rule(field: str) -> dict[str, bool] | None:
    """Infer one current alpha_gen field rule from a column name.

    The current gene supports field-level `current`, `log`, `zscore`, `diff_2q`,
    `diff_1y`, `pct_2q`, `pct_1y`, `std_2q`, and `std_1y`. `zscore` is enabled
    whenever `allow_current=True`, so it does not need a separate metadata flag.
    """

    field = str(field)
    if field in SPECIAL_COLUMNS or field.startswith("label_"):
        return None

    if _is_price_related(field):
        return _base_rule(
            can_y=False,
            can_x=True,
            allow_log=False,
            allow_diff=False,
            allow_pct=False,
            allow_std=False,
        )

    if _is_growth_like(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=False,
            allow_diff=True,
            allow_pct=False,
            allow_std=True,
        )

    if _is_ratio_like(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=False,
            allow_diff=True,
            allow_pct=False,
            allow_std=True,
        )

    if _is_per_share(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=False,
            allow_diff=True,
            allow_pct=True,
            allow_std=True,
        )

    if _is_cashflow_amount(field) or _is_income_statement_amount(field) or _is_balance_sheet_amount(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=True,
            allow_diff=True,
            allow_pct=True,
            allow_std=True,
        )

    return _base_rule(
        can_y=True,
        can_x=True,
        allow_log=False,
        allow_diff=True,
        allow_pct=False,
        allow_std=True,
    )


def build_metadata(
    columns: Iterable[str],
    *,
    dataset: str = "",
    shape: tuple[int, int] | None = None,
    n_dates: int | None = None,
    n_contracts: int | None = None,
    start: object | None = None,
    end: object | None = None,
    index_names: list[str] | None = None,
    label_col: str = "label_20d",
    tradeable_col: str = "is_tradeable",
    industry_col: str = "industry_code",
) -> dict[str, object]:
    """Build a metadata dict compatible with alpha_gen.core.gene.FieldRule."""

    columns = [str(col) for col in columns]
    field_rules = {
        field: rule
        for field in columns
        if (rule := infer_field_rule(field)) is not None
    }

    numeric_fields = [
        field
        for field in columns
        if field not in IDENTITY_FIELDS and field not in TRADEABLE_FIELDS
    ]
    categorical_fields = [field for field in columns if field == industry_col]

    metadata: dict[str, object] = {
        "dataset": dataset,
        "format": Path(dataset).suffix.lstrip(".") if dataset else "",
        "index": index_names or ["Datetime", "Contract"],
        "shape": list(shape) if shape is not None else None,
        "n_dates": n_dates,
        "n_contracts": n_contracts,
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "label_col": label_col,
        "tradeable_col": tradeable_col,
        "industry_col": industry_col,
        "notes": [
            "Generated from panel columns for current alpha_gen gene parameters.",
            "Transforms are controlled by field_rules: log, zscore(current), diff_2q/1y, pct_2q/1y, std_2q/1y.",
            "Growth and ratio-like fields disable log and pct by default to avoid unstable second-order ratios.",
            "Price-related valuation fields such as PE/PB/PS/PCF/EV are x-only and allow current/zscore only.",
        ],
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "field_rules": field_rules,
    }
    return metadata


def metadata_from_panel(
    panel: pd.DataFrame,
    *,
    dataset: str = "",
    label_col: str = "label_20d",
    tradeable_col: str = "is_tradeable",
    industry_col: str = "industry_code",
) -> dict[str, object]:
    """Build metadata from a loaded long-format panel."""

    n_dates = None
    n_contracts = None
    start = None
    end = None
    index_names = list(panel.index.names) if isinstance(panel.index, pd.MultiIndex) else None
    if isinstance(panel.index, pd.MultiIndex) and {"Datetime", "Contract"}.issubset(set(panel.index.names)):
        dates = panel.index.get_level_values("Datetime")
        contracts = panel.index.get_level_values("Contract")
        n_dates = int(pd.Index(dates).nunique())
        n_contracts = int(pd.Index(contracts).nunique())
        start = dates.min()
        end = dates.max()

    return build_metadata(
        panel.columns,
        dataset=dataset,
        shape=panel.shape,
        n_dates=n_dates,
        n_contracts=n_contracts,
        start=start,
        end=end,
        index_names=index_names,
        label_col=label_col,
        tradeable_col=tradeable_col,
        industry_col=industry_col,
    )


def write_metadata(metadata: dict[str, object], output_path: str | Path) -> Path:
    """Write metadata JSON with stable ordering and UTF-8 encoding."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def read_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported panel file type: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, help="Long panel parquet/csv path.")
    parser.add_argument("--columns-json", type=Path, help="JSON file containing final_tmt_panel.columns.tolist().")
    parser.add_argument("--output", type=Path, required=True, help="Output metadata JSON path.")
    parser.add_argument("--dataset", default="", help="Dataset name stored in metadata.")
    parser.add_argument("--label-col", default="label_20d")
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--industry-col", default="industry_code")
    args = parser.parse_args()

    if args.panel is None and args.columns_json is None:
        raise ValueError("one of --panel or --columns-json is required")

    if args.panel is not None:
        panel = read_panel(args.panel)
        metadata = metadata_from_panel(
            panel,
            dataset=args.dataset or args.panel.name,
            label_col=args.label_col,
            tradeable_col=args.tradeable_col,
            industry_col=args.industry_col,
        )
    else:
        columns = json.loads(args.columns_json.read_text(encoding="utf-8"))
        metadata = build_metadata(
            columns,
            dataset=args.dataset or args.columns_json.name,
            label_col=args.label_col,
            tradeable_col=args.tradeable_col,
            industry_col=args.industry_col,
        )

    output = write_metadata(metadata, args.output)
    print(f"saved: {output}")
    print(f"field_rules: {len(metadata['field_rules'])}")


if __name__ == "__main__":
    main()
