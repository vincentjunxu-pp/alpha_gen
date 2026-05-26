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
    "label_5d",
    "label_1d",
}

INVALID_COLUMNS = {
    # Accidental diagnostic column; real all-industry panel no longer contains it.
    "barra_factor_rtn",
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

BARRA_STYLE_FIELDS = [
    "barra_size",
    "barra_beta",
    "barra_momentum",
    "barra_residual_volatility",
    "barra_non_linear_size",
    "barra_book_to_price",
    "barra_liquidity",
    "barra_earnings_yield",
    "barra_growth",
    "barra_leverage",
]


def _base_rule(
    *,
    can_y: bool,
    can_x: bool,
    allow_log: bool,
    allow_current: bool = True,
    allow_lag: bool = False,
    allow_diff: bool = True,
    allow_pct: bool = True,
    allow_std: bool = False,
    family: str = "other",
    unit_type: str = "unknown",
    statement: str = "other",
    period_type: str = "unknown",
    direction: int = 1,
    add_group: str = "unknown",
    allow_industry_relative: bool = True,
) -> dict[str, object]:
    return {
        "can_y": can_y,
        "can_x": can_x,
        "allow_log": allow_log,
        "allow_current": allow_current,
        "allow_lag": allow_lag,
        "allow_diff": allow_diff,
        "allow_pct": allow_pct,
        "allow_std": allow_std,
        "family": family,
        "unit_type": unit_type,
        "statement": statement,
        "period_type": period_type,
        "direction": direction,
        "add_group": add_group,
        "allow_industry_relative": allow_industry_relative,
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


def _is_raw_price_field(field: str) -> bool:
    return field in {"open", "high", "low", "close", "vwap"}


def _is_market_value_field(field: str) -> bool:
    return field in SIZE_FIELDS or field in {
        "ev_lyr",
        "ev_ttm",
        "ev_lf",
        "ev_no_cash_lyr",
        "ev_no_cash_ttm",
        "ev_no_cash_lf",
    }


def _is_growth_like(field: str) -> bool:
    return "growth" in field or field.startswith("inc_")


def _is_per_share(field: str) -> bool:
    return "per_share" in field or field.endswith("_ps")


def _period_type(field: str) -> str:
    suffixes = ("_mrq", "_ttm", "_lyr", "_lf", "_ry")
    for suffix in suffixes:
        if field.endswith(suffix):
            return suffix[1:]
    return "unknown"


def _valuation_direction(field: str) -> int:
    cheap_when_high = (
        "ep_ratio",
        "cfp_ratio",
        "pcf_ratio",
        "book_to_market_ratio",
        "sp_ratio",
        "dividend_yield",
    )
    expensive_when_high = (
        "pe_ratio",
        "pb_ratio",
        "ps_ratio",
        "peg_ratio",
        "ev_to_",
        "ev_no_cash_to_",
    )
    if any(token in field for token in cheap_when_high):
        return 1
    if any(token in field for token in expensive_when_high) or field.startswith("ev_"):
        return -1
    return 1


def _ratio_family(field: str) -> str:
    if _is_price_related(field):
        return "valuation"
    if "growth" in field or field.startswith("inc_"):
        return "growth"
    if "margin" in field or field in {"roe", "roa", "roic"} or any(token in field for token in ("return_on", "profit_rate")):
        return "profitability"
    if "turnover" in field or "cycle" in field or "days" in field:
        return "efficiency"
    if any(token in field for token in ("debt", "leverage", "liabilit")):
        return "leverage"
    if any(token in field for token in ("cash_ratio", "quick_ratio", "current_ratio")):
        return "liquidity"
    return "quality"


def _ratio_unit_type(field: str) -> str:
    if "growth" in field or field.startswith("inc_"):
        return "growth"
    if "turnover" in field:
        return "turnover"
    if "yield" in field:
        return "yield"
    if "days" in field or "cycle" in field:
        return "days"
    if "rate" in field or "margin" in field:
        return "rate"
    return "ratio"


def _amount_semantics(field: str) -> tuple[str, str, str]:
    if any(token in field for token in ("market_cap", "market_val", "enterprise_value", "ev_")):
        return ("size", "market", "market_value")
    if any(token in field for token in ("revenue", "sales")):
        return ("growth" if field.startswith("forecast_") else "profitability", "income", "revenue")
    if any(token in field for token in ("profit", "income", "ebit", "ebitda")):
        return ("growth" if field.startswith("forecast_") else "profitability", "income", "profit")
    if any(token in field for token in ("expense", "cost", "tax", "r_n_d", "depreciation", "amortization", "capex")):
        return ("quality", "income", "expense")
    if _is_cashflow_amount(field):
        return ("cashflow", "cashflow", "cashflow")
    if any(token in field for token in ("debt", "loan", "liabilit")):
        return ("leverage", "balance_sheet", "debt")
    if any(token in field for token in ("cash", "deposit")):
        return ("liquidity", "balance_sheet", "cash")
    if "working_capital" in field:
        return ("working_capital", "balance_sheet", "working_capital")
    if "equity" in field or "book" in field:
        return ("size", "balance_sheet", "equity")
    if "asset" in field:
        return ("size", "balance_sheet", "asset")
    if any(token in field for token in ("receivable", "payable", "inventory")):
        return ("working_capital", "balance_sheet", "working_capital")
    return ("quality", "other", "amount")


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


def infer_field_rule(field: str) -> dict[str, object] | None:
    """Infer one current alpha_gen field rule from a column name.

    The current gene supports field-level `current`, signed `log`, `rank_pct`,
    `zscore`, `ind_rank_pct`, `ind_zscore`, `diff_2q`, `diff_1y`, `pct_2q`,
    and `pct_1y`. Rolling std transforms are intentionally not searchable.
    """

    field_name = str(field)
    field = field_name.lower()
    if field in INVALID_COLUMNS or field in BARRA_STYLE_FIELDS:
        return None
    if field in SPECIAL_COLUMNS or field.startswith("label_"):
        return None

    if _is_raw_price_field(field):
        return _base_rule(
            can_y=False,
            can_x=True,
            allow_log=True,
            allow_current=False,
            allow_diff=False,
            allow_pct=True,
            family="price",
            unit_type="price",
            statement="market",
            period_type="daily",
            direction=1,
            add_group="price",
        )

    if _is_price_related(field):
        if _is_market_value_field(field):
            family, statement, add_group = _amount_semantics(field)
        else:
            family, statement, add_group = "valuation", "market", "valuation"
        return _base_rule(
            can_y=False,
            can_x=True,
            allow_log=False,
            allow_diff=False,
            allow_pct=False,
            allow_std=False,
            family=family,
            unit_type="currency" if add_group == "market_value" else _ratio_unit_type(field),
            statement=statement,
            period_type=_period_type(field),
            direction=-1 if family == "size" else _valuation_direction(field),
            add_group=add_group,
        )

    if _is_growth_like(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=False,
            allow_diff=True,
            allow_pct=False,
            family="growth",
            unit_type="growth",
            statement="derived",
            period_type=_period_type(field),
            direction=1,
            add_group="growth",
        )

    if _is_ratio_like(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=False,
            allow_diff=True,
            allow_pct=False,
            family=_ratio_family(field),
            unit_type=_ratio_unit_type(field),
            statement="derived",
            period_type=_period_type(field),
            direction=_valuation_direction(field) if _ratio_family(field) == "valuation" else 1,
            add_group="metric",
        )

    if _is_per_share(field):
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=False,
            allow_diff=True,
            allow_pct=True,
            family="quality",
            unit_type="currency",
            statement="derived",
            period_type=_period_type(field),
            direction=1,
            add_group="per_share",
        )

    if _is_cashflow_amount(field) or _is_income_statement_amount(field) or _is_balance_sheet_amount(field):
        family, statement, add_group = _amount_semantics(field)
        return _base_rule(
            can_y=True,
            can_x=True,
            allow_log=True,
            allow_diff=True,
            allow_pct=True,
            family=family,
            unit_type="currency",
            statement=statement,
            period_type=_period_type(field),
            direction=-1 if add_group in {"expense", "debt"} else 1,
            add_group=add_group,
        )

    return _base_rule(
        can_y=True,
        can_x=True,
        allow_log=False,
        allow_diff=True,
        allow_pct=False,
        family="analyst" if "score" in field or "rating" in field else "other",
        unit_type="score" if "score" in field or "rating" in field else "unknown",
        statement="analyst" if "score" in field or "rating" in field else "other",
        period_type=_period_type(field),
        direction=1,
        add_group="score" if "score" in field or "rating" in field else "unknown",
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

    columns = [str(col) for col in columns if str(col).lower() not in INVALID_COLUMNS]
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
    seen_numeric = set(numeric_fields)
    for field in BARRA_STYLE_FIELDS:
        if field not in seen_numeric:
            numeric_fields.append(field)
            seen_numeric.add(field)
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
            "Generated from panel columns for the current alpha_gen structured-expression framework.",
            "Free unary transforms are controlled by field_rules: current, signed log1p, rank_pct, zscore, ind_rank_pct, ind_zscore, diff_2q/1y and pct_2q/1y.",
            "Rolling std transforms are intentionally disabled and kept out of the searchable unary transform pool.",
            "resi accepts unrestricted transform-legal A/B fields; resi_pair and multi_resi require additive controls with matching unit_type, add_group and accounting transform.",
            "Barra style fields are listed as neutralization controls, not searchable gene fields; pass them through extra_current_fields/barra_style_fields.",
            "Barra style fields are expected to be cross-sectionally z-scored by date with NaN values filled as 0.0 before entering TorchEvalContext.",
            "Use size_field=barra_size for size neutralization when the real panel contains the Barra size exposure; market_cap remains available only as a fallback.",
        ],
        "size_field": "barra_size",
        "barra_style_fields": BARRA_STYLE_FIELDS,
        "extra_current_fields": BARRA_STYLE_FIELDS,
        "neutralization": {
            "industry_field": industry_col,
            "size_field": "barra_size",
            "barra_style_fields": BARRA_STYLE_FIELDS,
            "barra_corr_threshold": 0.30,
            "barra_max_styles": 2,
        },
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
