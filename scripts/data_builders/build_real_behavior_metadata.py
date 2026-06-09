from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


FIELD_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|(.+)\|\s*$")
CODE_FIELD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
SECTION_RE = re.compile(r"^###\s+(7(?:\.\d+)?)\s+(.+)$")

BASE_UNARY = ["current", "rank_pct", "zscore", "ind_rank_pct", "ind_zscore"]
DIRECTION_UNARY = BASE_UNARY + ["direction_rank", "direction_zscore"]
DAILY_UNARY = BASE_UNARY + ["ts_zscore_5d", "ts_zscore_20d"]
FLOW_UNARY = DAILY_UNARY + ["direction_rank", "direction_zscore"]
EVENT_UNARY = ["current", "rank_pct", "zscore"]

CORE_RULE_KEYS = (
    "can_y",
    "can_x",
    "allow_log",
    "allow_current",
    "allow_diff",
    "allow_pct",
    "family",
    "unit_type",
    "direction",
    "add_group",
    "allow_industry_relative",
)

BEHAVIOR_RULE_KEYS = (
    "data_family",
    "behavior_roles",
    "direction",
    "allowed_slots",
    "allowed_unary_ops",
)

EXTRA_BASE_FIELDS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Vwap",
    "Turnover",
    "Amount",
    "Volume",
}

# These names came from an older field document but are not produced by the
# formulas in basefactor.py. The exact formula-side names are added below.
OBSOLETE_FORMULA_FIELDS = {
    "DII",
    "operating_profit_growth_accel",
    "ocf_growth_accel",
    "operating_profit_vs_revenue_growth_gap",
    "net_profit_vs_revenue_growth_gap",
    "net_profit_vs_gross_profit_growth_gap",
    "operating_vs_net_profit_growth_gap",
    "ocf_vs_operating_profit_growth_gap",
    "net_cashflow_vs_net_profit_growth_gap",
    "revenue_asset_growth_efficiency",
    "profit_asset_growth_efficiency",
    "ocf_asset_growth_efficiency",
    "roe_vs_net_asset_growth_gap",
    "growth_rank_dispersion",
    "ocf_growth_last_revision",
    "ocf_growth_update_flag",
    "short_debt_pressure",
    "liquidity_buffer",
    "asset_liquidity_structure",
    "net_debt_to_book_value",
    "retained_earnings_book_support",
    "obv_price_divergence",
    "drawdown_volume_panic",
    "mf_mid_large_net_ratio",
}

FRAMEWORK_BEHAVIOR_OVERRIDES = {
    "Vwap": {
        "behavior_roles": ["price_anchor", "cost_anchor", "anchor"],
        "allowed_slots": ["price_anchor", "cost_anchor", "state_signal"],
    },
}

VALUATION_DERIVED_FIELDS = {
    "ep_revaluation",
    "cfp_revaluation",
    "bm_revaluation",
    "sp_revaluation",
    "ev_ebitda_revaluation",
    "cashflow_vs_earnings_yield_gap",
    "book_vs_earnings_yield_gap",
    "sales_vs_earnings_yield_gap",
    "dividend_vs_earnings_yield_gap",
    "ev_no_cash_discount_ttm",
    "ev_no_cash_discount_lf",
    "ev_cash_adjustment_gap",
    "value_composite_rank",
    "deep_value_score",
    "valuation_dispersion",
    "valuation_consensus",
}

GROWTH_DERIVED_FIELDS = {
    "rev_growth_accel",
    "gross_profit_growth_accel",
    "oper_profit_growth_accel",
    "net_profit_growth_accel",
    "parent_profit_growth_accel",
    "oper_cash_growth_accel",
    "gross_vs_revenue_growth_gap",
    "oper_vs_revenue_growth_gap",
    "net_vs_oper_growth_gap",
    "parent_vs_gross_growth_gap",
    "oper_vs_net_profit_growth_gap",
    "parent_vs_net_profit_growth_gap",
    "ocf_vs_net_profit_growth_gap",
    "ocf_vs_oper_profit_growth_gap",
    "ocf_vs_revenue_growth_gap",
    "asset_rev_growth_efficiency",
    "asset_profit_growth_efficiency",
    "asset_operate_cash_efficiency",
    "financing_vs_ocf_growth_gap",
    "financing_vs_profit_growth_gap",
    "asset_vs_net_asset_growth_gap",
    "balanced_growth_mean",
    "balanced_growth_min",
    "growth_dispersion",
    "growth_positive_count",
    "balanced_growth_rank_mean",
    "balanced_growth_rank_min",
    "balanced_growth_rank_dispersion",
    "operating_profit_growth_last_revision",
    "net_profit_growth_last_revision",
    "parent_profit_last_revision",
    "operate_cash_last_revision",
    "operating_profit_growth_update_flag",
    "net_profit_growth_update_flag",
    "oper_cash_growth_update_flag",
}

FUNDAMENTAL_QUALITY_DERIVED_FIELDS = {
    "leverage_change",
    "equity_multiplier_change",
    "interest_debt_pressure",
    "short_debt_bufferr",
    "liquidity_bufferr",
    "asset_liability_structure",
    "intangible_leverage_pressure",
    "net_debt_to_book",
    "cash_eq_to_share_support",
    "retained_earnings_support",
    "roe_leverage_adjusted",
    "adjusted_roe_gap",
    "roic_vs_roa_gap",
    "margin_quality_spread",
    "operating_margin_quality",
    "expense_pressure",
    "non_operating_profit_pressure",
    "core_profit_quality",
    "turnover_efficiency_score",
    "working_capital_cycle_pressure",
    "ocf_per_share_vs_eps",
    "fcff_vs_ocf_gap",
    "fcfe_vs_ocf_gap",
    "cashflow_debt_coverage",
    "cashflow_short_debt_buffer",
    "free_cashflow_quality",
    "cashflow_interest_safety",
}

PRICE_MOMENTUM_DERIVED_FIELDS = {
    "ma_5_20_spread",
    "ma_20_60_spread",
    "ma_60_120_spread",
    "ema_5_20_spread",
    "ema_20_60_spread",
    "trend_consensus",
    "trend_extreme_count",
    "trend_dispersion",
    "trend_vol_adjusted_20",
    "momentum_drawdown_adjusted",
    "momentum_wave_adjusted",
    "macd_vol_adjusted",
    "trend_strength_confirmed",
    "aroon_direction_gap",
    "aroon_trend_confirmed",
    "ret_1d",
    "ret_5d",
    "ret_20d",
}

PRICE_CROWDING_DERIVED_FIELDS = {
    "volume_crowding_5_20",
    "volume_crowding_5_60",
    "volume_crowding_20_120",
    "amount_crowding_5_20",
    "amount_crowding_20_60",
    "overbought_consensus",
    "overbought_extreme_count",
    "reversal_signal_dispersion",
    "crowded_momentum_risk",
    "crowded_overbought_risk",
    "amount_shock_5_20",
    "volume_shock_5_20",
    "turnover_shock_5_20",
    "short_term_overheat",
}

PRICE_DIVERGENCE_DERIVED_FIELDS = {
    "price_volume_divergence_5_20",
    "price_volume_divergence_20_60",
    "obv_price_divergence_20_60",
    "mfi_rsi_divergence",
}

PRICE_PANIC_DERIVED_FIELDS = {
    "oversold_extreme_count",
    "panic_oversold_score",
    "downside_volume_panic",
}

PRICE_VOLATILITY_DERIVED_FIELDS = {
    "intraday_range",
    "range_std_20d",
}

CROSS_MODULE_FIELDS = {
    "flow_price_divergence_5d",
    "institution_accumulation",
    "retail_chase_risk",
    "large_order_confirmed_momentum",
    "moneyflow_crowding_risk",
    "orderbook_panic_reversal",
    "orderbook_chase_risk",
    "liquidity_neglect",
}


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def extract_field_rows(markdown_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_section = "7.0"
    current_section_title = "字段清单速查"
    in_field_list = False
    in_code_block = False

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        clean_line = line.lstrip("\ufeff")
        if clean_line.startswith("## 7."):
            in_field_list = True
            continue
        if not in_field_list:
            continue
        if clean_line.startswith("## 8."):
            break

        section_match = SECTION_RE.match(clean_line)
        if section_match:
            current_section = section_match.group(1)
            current_section_title = section_match.group(2).strip()
            continue

        stripped = clean_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            for field in CODE_FIELD_RE.findall(clean_line):
                rows.append(
                    {
                        "field": field,
                        "group": current_section_title,
                        "meaning": current_section_title,
                        "section": current_section,
                        "section_title": current_section_title,
                    }
                )
            continue

        match = FIELD_ROW_RE.match(clean_line)
        if not match:
            continue
        cells = _split_row(clean_line)
        if len(cells) < 3:
            continue
        first_cell = cells[0]
        field_match = re.fullmatch(r"`([^`]+)`", first_cell)
        if not field_match:
            continue
        field = field_match.group(1).strip()
        if not field or "*" in field or "/" in field:
            continue
        rows.append(
            {
                "field": field,
                "group": cells[1],
                "meaning": cells[2],
                "section": current_section,
                "section_title": current_section_title,
            }
        )

    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        field = row["field"]
        if field in seen:
            continue
        seen.add(field)
        unique.append(row)
    return unique


def _window(field: str) -> str:
    suffixes = ("lyr", "ttm", "lf", "lfr")
    for suffix in suffixes:
        if field.endswith(f"_{suffix}"):
            return suffix
    match = re.search(r"_(\d+d)$", field)
    if match:
        return match.group(1)
    match = re.search(r"_(\d+_\d+)$", field)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)$", field)
    if match:
        return match.group(1)
    return "daily"


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _data_family(row: dict[str, str]) -> str:
    field = row["field"]
    section = row["section"]
    group = row["group"]

    if field.startswith("barra_") or field in {
        "market_cap",
        "market_cap_2",
        "market_cap_3",
        "a_share_market_val",
        "a_share_market_val_in_circulation",
    }:
        return "control"
    if section in {"7.1", "7.4", "7.10"}:
        return "price_volume"
    if field.startswith("ob_") or field.startswith("orderbook_") or field == "liquidity_neglect":
        return "orderbook"
    if field.startswith("mf_") or field in {
        "flow_price_divergence_5d",
        "institution_accumulation",
        "retail_chase_risk",
        "large_order_confirmed_momentum",
        "moneyflow_crowding_risk",
    }:
        return "moneyflow"
    if section == "7.6":
        return "fundamental"
    if section == "7.7":
        return "moneyflow"
    if group in {"市值", "市值扩展口径", "A 股市值", "A 股流通市值"}:
        return "control"
    return "fundamental"


def _sub_family(row: dict[str, str], data_family: str) -> str:
    field = row["field"]
    text = f"{field} {row['group']} {row['section_title']}"

    if data_family == "control":
        return "size"
    if data_family == "orderbook":
        if _contains_any(text, ("价差", "流动性", "spread", "liquidity_stress")):
            return "orderbook_liquidity"
        if _contains_any(text, ("深度", "总挂单", "depth")):
            return "orderbook_depth"
        if _contains_any(text, ("尾盘", "close_chase", "close30")):
            return "orderbook_close_pressure"
        if _contains_any(text, ("开盘", "open", "open30", "pmopen30")):
            return "orderbook_open_intent"
        return "orderbook_pressure"
    if data_family == "moneyflow":
        if _contains_any(text, ("小单", "散户", "retail", "_sm_")):
            return "retail_flow"
        if _contains_any(text, ("大单", "超大单", "机构", "large", "lg", "elg")):
            return "large_flow"
        if _contains_any(text, ("主动", "imbalance")):
            return "active_flow"
        if _contains_any(text, ("拥挤", "crowding")):
            return "moneyflow_crowding"
        return "moneyflow"
    if data_family == "price_volume":
        if field in {"Open", "High", "Low", "Close", "Vwap"}:
            return "price_anchor"
        if field in {"Turnover", "Amount", "Volume"}:
            return "volume_crowding"
        if field.startswith("ret_"):
            return "momentum"
        if _contains_any(text, ("成交拥挤", "成交量", "成交金额", "crowding", "VOL", "AMV", "QTYR", "VMA")):
            return "volume_crowding"
        if _contains_any(text, ("回撤", "恐慌", "MDD", "drawdown", "panic")):
            return "panic_drawdown"
        if _contains_any(text, ("超买", "超卖", "RSI", "KDJ", "WR", "CCI", "BIAS")):
            return "overreaction"
        if _contains_any(text, ("波动", "ATR", "BOLL", "AMP", "MASS")):
            return "volatility"
        if _contains_any(text, ("均线", "锚", "成本", "MA", "EMA", "WMA", "MCST")):
            return "price_anchor"
        if _contains_any(text, ("趋势", "动量", "MACD", "ROC", "MTM", "trend")):
            return "momentum"
        if _contains_any(text, ("量价背离", "背离", "OBV", "MFI", "VR")):
            return "price_volume_divergence"
        return "price_volume"

    if _contains_any(text, ("估值", "收益率", "value", "valuation", "ep_", "cfp_", "book_to_market", "dividend", "peg", "sp_", "ev_")):
        return "valuation"
    if _contains_any(text, ("收入增长", "主营收入", "revenue")):
        return "revenue_growth"
    if _contains_any(text, ("盈利增长", "利润增长", "归母", "毛利", "profit", "earnings", "eps")):
        return "profit_growth"
    if _contains_any(text, ("现金流", "ocf", "fcf", "cashflow", "cash_flow", "cash_flow_per_share")):
        return "cashflow_quality"
    if _contains_any(text, ("资产扩张", "资产效率", "asset_growth")):
        return "asset_growth"
    if _contains_any(text, ("杠杆", "债务", "负债", "debt", "leverage")):
        return "leverage_risk"
    if _contains_any(text, ("ROE", "回报率", "利润率", "周转效率", "质量", "margin", "turnover")):
        return "quality"
    if _contains_any(text, ("信息修正", "信息更新", "revision", "update_flag")):
        return "revision"
    return "fundamental"


def _unit_type(row: dict[str, str], data_family: str, sub_family: str) -> str:
    field = row["field"]
    text = f"{field} {row['group']} {row['meaning']}"

    if data_family == "control":
        return "currency"
    if field.endswith("_flag") or "update_flag" in field:
        return "event"
    if data_family in {"price_volume", "orderbook", "moneyflow"}:
        if field in {"Open", "High", "Low", "Close", "Vwap"}:
            return "price"
        if _contains_any(text, ("amount", "金额", "AMV", "成交额")):
            return "currency"
        if _contains_any(text, ("depth", "VOL", "VMA", "挂单量", "成交量")):
            return "volume"
        if _contains_any(text, ("ratio", "占比", "比率", "rate", "收益率", "乖离率")):
            return "ratio"
        return "score"
    if _contains_any(text, ("growth", "增长", "增速", "改善", "accel")):
        return "growth"
    if _contains_any(text, ("ratio", "rate", "率", "收益率", "利润率", "ROE", "ROA", "ROIC", "margin")):
        return "ratio"
    if _contains_any(text, ("per_share", "每股", "ev", "market_cap", "debt", "现金", "资本", "资产", "负债")):
        return "currency"
    if sub_family == "valuation":
        return "ratio"
    return "score"


def _direction(row: dict[str, str], data_family: str, sub_family: str) -> int:
    field = row["field"]
    text = f"{field} {row['group']} {row['meaning']} {row['section_title']}"
    negative_patterns = (
        "pressure",
        "risk",
        "stress",
        "leverage",
        "debt",
        "liabilities",
        "expense",
        "cost_to_sales",
        "non_operating",
        "investment_profit_to_profit",
        "current_debt",
        "interest_bearing_debt",
        "net_debt",
        "cash_conversion_cycle",
        "operating_cycle",
        "turnover_days",
        "MDD",
        "overbought",
        "crowded",
        "spread_full_max",
        "spread_full_std",
        "liquidity_stress",
        "active_sell",
        "融资依赖",
        "杠杆",
        "债务",
        "负债",
        "压力",
        "风险",
        "成本费用",
        "营业外",
    )
    positive_overrides = (
        "book_to_market",
        "ep_ratio",
        "cfp_ratio",
        "sp_ratio",
        "dividend_yield",
        "liquidity_buffer",
        "cashflow_debt_coverage",
        "cashflow_short_debt_buffer",
        "cashflow_interest_safety",
        "core_profit_quality",
        "turnover_efficiency_score",
    )
    negative_overrides = (
        "peg_ratio",
        "ev_to_ebitda",
        "ev_no_cash_to_ebit",
        "ev_ebitda_revaluation",
    )

    if _contains_any(field, positive_overrides):
        return 1
    if _contains_any(field, negative_overrides):
        return -1
    if _contains_any(text, negative_patterns):
        return -1
    if data_family == "orderbook" and _contains_any(field, ("ob_spread",)):
        return -1
    return 1


def _fundamental_roles_slots(row: dict[str, str], sub_family: str, direction: int) -> tuple[list[str], list[str]]:
    field = row["field"]
    text = f"{field} {row['group']} {row['meaning']} {row['section_title']}"

    if field.endswith("_flag") or "last_revision" in field:
        return ["anchor", "growth", "underreaction", "attention"], ["fund_anchor", "state_signal"]
    if sub_family == "valuation":
        return ["valuation", "support", "anchor"], ["fund_anchor", "fund_support"]
    if sub_family == "cashflow_quality":
        roles = ["anchor", "quality", "cashflow_quality", "support"]
        if "growth" in field or "增长" in text:
            roles.append("growth")
        return roles, ["fund_anchor", "cashflow_quality", "fund_support"]
    if sub_family in {"revenue_growth", "profit_growth", "asset_growth"} or _contains_any(text, ("增长", "growth", "accel", "均衡成长")):
        return ["anchor", "growth", "support"], ["fund_anchor", "growth_anchor", "profit_growth", "fund_support"]
    if sub_family == "leverage_risk" or direction < 0:
        return ["risk", "quality", "support"], ["fund_support", "state_signal", "control"]
    if sub_family == "quality":
        return ["anchor", "quality", "support"], ["fund_anchor", "fund_support", "cashflow_quality"]
    return ["anchor", "quality", "support"], ["fund_anchor", "fund_support"]


def _price_roles_slots(row: dict[str, str], sub_family: str) -> tuple[list[str], list[str]]:
    field = row["field"]
    text = f"{field} {row['group']} {row['meaning']}"

    if field in {"Open", "High", "Low", "Close"}:
        return ["price_anchor", "anchor"], ["state_signal"]
    if field == "Vwap":
        return ["price_anchor", "cost_anchor", "anchor"], ["price_anchor", "cost_anchor"]
    if sub_family in {"volume_crowding"} or _contains_any(text, ("crowding", "拥挤", "成交", "放量", "活跃")):
        return ["crowding", "attention", "liquidity", "risk"], ["crowding_signal", "turnover_shock", "attention_heat", "state_signal"]
    if sub_family in {"panic_drawdown"} or _contains_any(text, ("回撤", "恐慌", "oversold")):
        return ["panic", "drawdown", "oversold", "attention"], ["drawdown", "attention_heat", "state_signal"]
    if sub_family in {"overreaction", "volatility"}:
        return ["attention", "overreaction", "volatility", "lottery"], ["attention_heat", "price_reaction", "state_signal"]
    if sub_family == "price_anchor" or field in {"MCST", "ma_5_20_spread", "ma_20_60_spread", "ma_60_120_spread", "ema_5_20_spread", "ema_20_60_spread"}:
        if field.startswith(("MA", "EMA", "WMA", "BOLL", "BBI")) and field not in {"ma_5_20_spread", "ma_20_60_spread", "ma_60_120_spread", "ema_5_20_spread", "ema_20_60_spread"}:
            return ["anchor", "price_anchor"], ["state_signal"]
        return ["price_anchor", "cost_anchor", "anchor", "momentum"], ["price_anchor", "cost_anchor", "price_momentum"]
    if sub_family == "price_volume_divergence":
        return ["reaction", "momentum", "attention"], ["price_reaction", "price_momentum", "price_control", "state_signal"]
    return ["momentum", "reaction", "anchor_momentum"], ["price_reaction", "price_momentum", "price_control"]


def _orderbook_roles_slots(row: dict[str, str], sub_family: str) -> tuple[list[str], list[str]]:
    field = row["field"]
    text = f"{field} {row['group']} {row['meaning']}"

    if sub_family == "orderbook_liquidity":
        return ["liquidity", "stress", "spread"], ["liquidity_stress", "state_signal"]
    if sub_family == "orderbook_depth":
        return ["liquidity", "support"], ["state_signal"]
    if "close" in field or "尾盘" in text:
        return ["close_chase", "crowding", "buy_pressure", "orderbook_pressure"], ["close_chase", "orderbook_pressure", "orderbook_filter"]
    if "open" in field or "开盘" in text:
        return ["open_intent", "buy_pressure", "confirmation", "orderbook_pressure"], ["orderbook_pressure", "orderbook_filter"]
    if "panic" in field:
        return ["open_intent", "buy_pressure", "confirmation"], ["orderbook_pressure", "orderbook_filter"]
    if "chase" in field:
        return ["close_chase", "crowding", "risk"], ["close_chase", "sell_pressure", "orderbook_pressure"]
    return ["orderbook_pressure", "buy_pressure", "open_intent", "confirmation"], ["orderbook_pressure", "orderbook_filter"]


def _moneyflow_roles_slots(row: dict[str, str], sub_family: str) -> tuple[list[str], list[str]]:
    field = row["field"]
    text = f"{field} {row['group']} {row['meaning']}"

    if _contains_any(text, ("小单", "散户", "retail", "_sm_")):
        return ["retail", "small_flow", "chase", "crowding"], ["retail_flow", "sell_pressure", "crowding_signal"]
    if _contains_any(text, ("主动卖", "sell")):
        return ["stress", "orderbook_pressure"], ["sell_pressure", "state_signal"]
    if _contains_any(text, ("拥挤", "crowding", "participation")):
        return ["crowding", "attention", "risk"], ["crowding_signal", "state_signal"]
    if _contains_any(text, ("大单", "超大单", "机构", "large", "lg", "elg", "institution")):
        return ["large_flow", "confirmation", "underreaction"], ["flow_confirm", "large_flow"]
    if _contains_any(text, ("背离", "divergence", "净流入", "主动买")):
        return ["large_flow", "confirmation", "underreaction"], ["flow_confirm", "large_flow", "state_signal"]
    return ["large_flow", "confirmation"], ["flow_confirm"]


def _allowed_unary(data_family: str, unit_type: str, sub_family: str, slots: list[str]) -> list[str]:
    if unit_type == "event":
        return EVENT_UNARY
    if data_family == "fundamental":
        if "state_signal" in slots and "fund_anchor" not in slots:
            return BASE_UNARY
        return DIRECTION_UNARY
    if data_family == "price_volume":
        return DAILY_UNARY
    if data_family == "orderbook":
        if sub_family == "orderbook_liquidity":
            return DAILY_UNARY
        return FLOW_UNARY
    if data_family == "moneyflow":
        return FLOW_UNARY
    if data_family == "control":
        return BASE_UNARY
    return BASE_UNARY


def _core_rule(data_family: str, unit_type: str, direction: int, window: str) -> dict[str, object]:
    del window
    return {
        "can_y": data_family != "control",
        "can_x": True,
        "allow_log": unit_type in {"currency", "price", "volume"},
        "allow_current": True,
        "allow_diff": unit_type != "event",
        "allow_pct": unit_type in {"price", "currency", "ratio", "rate", "growth", "volume"},
        "family": data_family,
        "unit_type": unit_type,
        "direction": direction,
        "add_group": data_family,
        "allow_industry_relative": True,
    }


def _compact_rule(rule: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    return {key: rule[key] for key in keys if key in rule}


def _compact_behavior_rule(field: str, rule: dict[str, object]) -> dict[str, object]:
    del field
    return _compact_rule(rule, BEHAVIOR_RULE_KEYS)


def extract_basefactor_output_fields(basefactor_path: Path) -> dict[str, str]:
    """Return final formula output fields keyed by their source dataframe.

    Minute-level order-book intermediates stored in ``x`` are intentionally
    excluded. The selected daily order-book output is taken from
    ``ORDERBOOK_KEEP_COLS``. Dynamic money-flow loop outputs are expanded
    explicitly because their names are represented as f-strings in the source.
    """

    source = basefactor_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(basefactor_path))
    final_frames = {"panel_raw", "alpha", "moneyflow", "out", "df"}
    fields: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                continue
            frame = target.value.id
            if frame not in final_frames:
                continue
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                fields[target.slice.value] = frame

    for node in tree.body:
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "ORDERBOOK_KEEP_COLS":
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    fields[item.value] = "out"

    for prefix in ("sm", "md", "lg", "elg"):
        for suffix in ("s1", "s2", "s3"):
            fields[f"mf_{prefix}_{suffix}"] = "moneyflow"

    for window in (5, 20):
        for template in (
            "mf_net_amount_ratio_{window}d",
            "mf_large_net_ratio_{window}d",
            "mf_sm_net_ratio_{window}d",
            "mf_large_vs_small_net_{window}d",
            "mf_large_positive_days_{window}d",
            "mf_active_imbalance_std_{window}d",
        ):
            fields[template.format(window=window)] = "moneyflow"

    return fields


def _formula_rule(
    *,
    data_family: str,
    roles: list[str],
    sub_family: str,
    unit_type: str,
    direction: int,
    slots: list[str],
    field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    window = _window(field)
    unique_roles = list(dict.fromkeys(roles))
    unique_slots = list(dict.fromkeys(slots))
    behavior_rule = {
        "data_family": data_family,
        "behavior_roles": unique_roles,
        "direction": direction,
        "allowed_slots": unique_slots,
        "allowed_unary_ops": _allowed_unary(data_family, unit_type, sub_family, unique_slots),
    }
    return _core_rule(data_family, unit_type, direction, window), behavior_rule


def classify_basefactor_field(field: str, source_frame: str) -> tuple[dict[str, object], dict[str, object]]:
    """Classify one field from its actual formula and output module."""

    if source_frame == "panel_raw":
        if field in VALUATION_DERIVED_FIELDS:
            negative = {
                "ev_ebitda_revaluation",
                "ev_no_cash_discount_ttm",
                "ev_no_cash_discount_lf",
                "ev_cash_adjustment_gap",
                "valuation_dispersion",
            }
            return _formula_rule(
                data_family="fundamental",
                roles=["valuation", "support", "anchor"],
                sub_family="valuation",
                unit_type="ratio",
                direction=-1 if field in negative else 1,
                slots=["fund_anchor", "fund_support"],
                field=field,
            )

        if field in GROWTH_DERIVED_FIELDS:
            is_event = field.endswith("_update_flag")
            cashflow_quality = field.startswith(("ocf_", "oper_cash", "operate_cash"))
            negative = (
                "dispersion" in field
                or field.startswith("financing_")
                or field == "asset_vs_net_asset_growth_gap"
                or field == "net_vs_oper_growth_gap"
            )
            roles = ["anchor", "growth", "support"]
            slots = ["fund_anchor", "growth_anchor", "profit_growth", "fund_support"]
            if cashflow_quality:
                roles.extend(["quality", "cashflow_quality"])
                slots.append("cashflow_quality")
            if is_event:
                roles.append("attention")
                slots.append("state_signal")
            return _formula_rule(
                data_family="fundamental",
                roles=roles,
                sub_family="revision" if "revision" in field or is_event else "profit_growth",
                unit_type="event" if is_event else "growth",
                direction=-1 if negative else 1,
                slots=slots,
                field=field,
            )

        if field in FUNDAMENTAL_QUALITY_DERIVED_FIELDS:
            negative = any(
                token in field
                for token in (
                    "leverage",
                    "debt_pressure",
                    "short_debt",
                    "intangible",
                    "net_debt",
                    "expense_pressure",
                    "non_operating_profit_pressure",
                    "cycle_pressure",
                )
            )
            cashflow = any(
                token in field
                for token in ("cashflow", "ocf_", "fcff_", "fcfe_", "cash_eq", "free_cashflow")
            )
            roles = ["anchor", "quality", "support"]
            slots = ["fund_anchor", "fund_support", "cashflow_quality"]
            if negative:
                roles.append("risk")
            if cashflow:
                roles.append("cashflow_quality")
            return _formula_rule(
                data_family="fundamental",
                roles=roles,
                sub_family="cashflow_quality" if cashflow else "quality",
                unit_type="ratio",
                direction=-1 if negative else 1,
                slots=slots,
                field=field,
            )

    if source_frame == "alpha" or field in PRICE_MOMENTUM_DERIVED_FIELDS:
        if field in PRICE_CROWDING_DERIVED_FIELDS:
            return _formula_rule(
                data_family="price_volume",
                roles=["crowding", "attention", "liquidity", "risk", "overreaction"],
                sub_family="volume_crowding",
                unit_type="ratio",
                direction=-1,
                slots=["crowding_signal", "turnover_shock", "attention_heat", "state_signal"],
                field=field,
            )
        if field in PRICE_DIVERGENCE_DERIVED_FIELDS:
            return _formula_rule(
                data_family="price_volume",
                roles=["reaction", "momentum", "attention"],
                sub_family="price_volume_divergence",
                unit_type="score",
                direction=1,
                slots=["price_reaction", "price_momentum", "price_control", "state_signal"],
                field=field,
            )
        if field in PRICE_PANIC_DERIVED_FIELDS:
            return _formula_rule(
                data_family="price_volume",
                roles=["panic", "drawdown", "oversold", "attention"],
                sub_family="panic_drawdown",
                unit_type="score",
                direction=1,
                slots=["drawdown", "attention_heat", "state_signal"],
                field=field,
            )
        if field in PRICE_VOLATILITY_DERIVED_FIELDS:
            return _formula_rule(
                data_family="price_volume",
                roles=["attention", "overreaction", "volatility", "lottery"],
                sub_family="volatility",
                unit_type="ratio",
                direction=-1,
                slots=["attention_heat", "state_signal"],
                field=field,
            )
        return _formula_rule(
            data_family="price_volume",
            roles=["momentum", "reaction", "anchor_momentum", "price_anchor"],
            sub_family="momentum",
            unit_type="ratio",
            direction=1,
            slots=["price_reaction", "price_momentum", "price_control", "price_anchor"],
            field=field,
        )

    if source_frame == "moneyflow":
        row = {
            "field": field,
            "group": "资金流公式字段",
            "meaning": "资金流公式字段",
            "section_title": "资金流公式字段",
        }
        sub_family = _sub_family(row, "moneyflow")
        roles, slots = _moneyflow_roles_slots(row, sub_family)
        if "amount" in field and "ratio" not in field:
            unit_type = "currency"
        elif "_vol" in field and "ratio" not in field:
            unit_type = "volume"
        else:
            unit_type = "ratio"
        direction = -1 if any(token in field for token in ("sell", "sm_", "small", "_std_")) else 1
        return _formula_rule(
            data_family="moneyflow",
            roles=roles,
            sub_family=sub_family,
            unit_type=unit_type,
            direction=direction,
            slots=slots,
            field=field,
        )

    if source_frame == "out":
        row = {
            "field": field,
            "group": "盘口日频公式字段",
            "meaning": "盘口日频公式字段",
            "section_title": "盘口日频公式字段",
        }
        sub_family = _sub_family(row, "orderbook")
        roles, slots = _orderbook_roles_slots(row, sub_family)
        direction = -1 if "spread" in field or "liquidity_stress" in field else 1
        return _formula_rule(
            data_family="orderbook",
            roles=roles,
            sub_family=sub_family,
            unit_type="score",
            direction=direction,
            slots=slots,
            field=field,
        )

    if source_frame == "df":
        if field in CROSS_MODULE_FIELDS:
            if field in {"flow_price_divergence_5d", "institution_accumulation", "large_order_confirmed_momentum"}:
                return _formula_rule(
                    data_family="moneyflow",
                    roles=["large_flow", "confirmation", "underreaction"],
                    sub_family="large_flow",
                    unit_type="score",
                    direction=1,
                    slots=["flow_confirm", "large_flow", "state_signal"],
                    field=field,
                )
            if field in {"retail_chase_risk", "moneyflow_crowding_risk"}:
                return _formula_rule(
                    data_family="moneyflow",
                    roles=["retail", "small_flow", "chase", "crowding", "risk"],
                    sub_family="moneyflow_crowding",
                    unit_type="score",
                    direction=-1,
                    slots=["retail_flow", "crowding_signal", "sell_pressure", "state_signal"],
                    field=field,
                )
            if field == "liquidity_neglect":
                return _formula_rule(
                    data_family="orderbook",
                    roles=["liquidity", "stress", "spread"],
                    sub_family="orderbook_liquidity",
                    unit_type="score",
                    direction=-1,
                    slots=["liquidity_stress", "state_signal"],
                    field=field,
                )
            if field == "orderbook_panic_reversal":
                return _formula_rule(
                    data_family="orderbook",
                    roles=["open_intent", "buy_pressure", "confirmation", "orderbook_pressure"],
                    sub_family="orderbook_pressure",
                    unit_type="score",
                    direction=1,
                    slots=["orderbook_pressure", "orderbook_filter", "state_signal"],
                    field=field,
                )
            return _formula_rule(
                data_family="orderbook",
                roles=["close_chase", "crowding", "risk", "orderbook_pressure"],
                sub_family="orderbook_close_pressure",
                unit_type="score",
                direction=-1,
                slots=["close_chase", "sell_pressure", "orderbook_pressure", "state_signal"],
                field=field,
            )

        if field in PRICE_CROWDING_DERIVED_FIELDS:
            return classify_basefactor_field(field, "alpha")
        if field in PRICE_VOLATILITY_DERIVED_FIELDS:
            return classify_basefactor_field(field, "alpha")
        return classify_basefactor_field(field, "alpha")

    raise KeyError(f"no basefactor classification for field={field!r}, source={source_frame!r}")


def classify_field(row: dict[str, str]) -> tuple[dict[str, object], dict[str, object]]:
    data_family = _data_family(row)
    sub_family = _sub_family(row, data_family)
    unit_type = _unit_type(row, data_family, sub_family)
    direction = _direction(row, data_family, sub_family)
    window = _window(row["field"])

    if data_family == "control":
        roles, slots = ["control"], ["control"]
    elif data_family == "fundamental":
        roles, slots = _fundamental_roles_slots(row, sub_family, direction)
    elif data_family == "price_volume":
        roles, slots = _price_roles_slots(row, sub_family)
    elif data_family == "orderbook":
        roles, slots = _orderbook_roles_slots(row, sub_family)
    elif data_family == "moneyflow":
        roles, slots = _moneyflow_roles_slots(row, sub_family)
    else:
        roles, slots = ["anchor"], []

    behavior_rule = {
        "data_family": data_family,
        "behavior_roles": list(dict.fromkeys(roles)),
        "direction": direction,
        "allowed_slots": list(dict.fromkeys(slots)),
        "allowed_unary_ops": _allowed_unary(data_family, unit_type, sub_family, slots),
    }
    return _core_rule(data_family, unit_type, direction, window), behavior_rule


def sync_metadata_with_basefactor(
    metadata: dict[str, object],
    basefactor_path: Path,
) -> dict[str, object]:
    """Synchronize formula-derived rules without rewriting unrelated fields."""

    field_rules = dict(metadata.get("field_rules", {}))
    behavior_rules = dict(metadata.get("behavior_field_rules", {}))

    for field in OBSOLETE_FORMULA_FIELDS:
        field_rules.pop(field, None)
        behavior_rules.pop(field, None)

    formula_sources = extract_basefactor_output_fields(basefactor_path)
    for field, source_frame in formula_sources.items():
        core_rule, behavior_rule = classify_basefactor_field(field, source_frame)
        field_rules[field] = core_rule
        behavior_rules[field] = behavior_rule

    for field, override in FRAMEWORK_BEHAVIOR_OVERRIDES.items():
        if field in behavior_rules:
            behavior_rules[field] = {**behavior_rules[field], **override}

    return {
        "size_field": str(metadata.get("size_field", "barra_size")),
        "barra_style_fields": list(metadata.get("barra_style_fields", ())),
        "field_rules": {
            field: _compact_rule(rule, CORE_RULE_KEYS)
            for field, rule in field_rules.items()
        },
        "behavior_field_rules": {
            field: _compact_behavior_rule(field, rule)
            for field, rule in behavior_rules.items()
        },
    }


def build_metadata(source_md: Path, basefactor_path: Path | None = None) -> dict[str, object]:
    rows = extract_field_rows(source_md)
    if not rows:
        raise RuntimeError(f"no field rows parsed from {source_md}")

    rows_by_field = {
        row["field"]: row
        for row in rows
        if row["field"] not in OBSOLETE_FORMULA_FIELDS
    }
    for field in sorted(EXTRA_BASE_FIELDS):
        rows_by_field.setdefault(
            field,
            {
                "field": field,
                "group": "基础量价字段",
                "meaning": "基础量价字段",
                "section": "7.4",
                "section_title": "基础量价字段",
            },
        )

    formula_sources: dict[str, str] = {}
    if basefactor_path is not None:
        formula_sources = extract_basefactor_output_fields(basefactor_path)
        for field, source_frame in formula_sources.items():
            rows_by_field[field] = {
                "field": field,
                "group": f"basefactor.py {source_frame} 公式字段",
                "meaning": f"basefactor.py {source_frame} 公式字段",
                "section": "formula",
                "section_title": f"basefactor.py {source_frame} 公式字段",
            }

    rows = list(rows_by_field.values())
    raw_fields = {row["field"] for row in rows}
    label_fields = [field for field in ["label_1d", "label_5d", "label_20d"] if field in raw_fields]
    if not label_fields:
        label_fields = ["label_1d", "label_5d", "label_20d"]
    tradeable_field = "is_tradeable"
    industry_field = "industry_code"
    excluded_fields = set(label_fields) | {tradeable_field, industry_field}
    barra_style_fields = sorted(field for field in raw_fields if field.startswith("barra_"))
    size_field = "barra_size" if "barra_size" in raw_fields else "market_cap"

    field_rules: dict[str, dict[str, object]] = {}
    behavior_rules: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["field"] in excluded_fields:
            continue
        field = row["field"]
        if field in formula_sources:
            core_rule, behavior_rule = classify_basefactor_field(field, formula_sources[field])
        else:
            core_rule, behavior_rule = classify_field(row)
        field_rules[field] = _compact_rule(core_rule, CORE_RULE_KEYS)
        behavior_rules[field] = _compact_behavior_rule(field, behavior_rule)

    return {
        "size_field": size_field,
        "barra_style_fields": barra_style_fields,
        "field_rules": field_rules,
        "behavior_field_rules": behavior_rules,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build real behavior-finance metadata from the field markdown.")
    parser.add_argument(
        "--source-md",
        type=Path,
        default=root / "behavior_fin" / "基础字段业务解释_基本面增长类.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "metadata" / "production" / "real_behavior_metadata.json",
    )
    parser.add_argument(
        "--basefactor-path",
        type=Path,
        default=root / "scripts" / "data_builders" / "basefactor.py",
    )
    parser.add_argument(
        "--rebuild-from-source",
        action="store_true",
        help="Rebuild all base-field rules from the markdown instead of preserving the existing output metadata.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.rebuild_from_source:
        metadata = json.loads(args.output.read_text(encoding="utf-8"))
        metadata = sync_metadata_with_basefactor(metadata, args.basefactor_path)
    else:
        metadata = build_metadata(args.source_md, args.basefactor_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    behavior_rules = metadata["behavior_field_rules"]
    data_family_counts = Counter(rule["data_family"] for rule in behavior_rules.values())
    print(f"field_count={len(behavior_rules)}")
    print(f"data_family_counts={dict(sorted(data_family_counts.items()))}")


if __name__ == "__main__":
    main()
