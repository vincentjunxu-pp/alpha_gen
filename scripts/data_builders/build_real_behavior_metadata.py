from __future__ import annotations

import argparse
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


def _session(field: str) -> str:
    for session in ("open30", "amclose30", "pmopen30", "close30", "full"):
        if session in field:
            return session
    return "full"


def _investor_type(field: str) -> str:
    if "large_vs_small" in field:
        return "mixed"
    if "_elg_" in field or "elg" in field:
        return "extra_large"
    if "_lg_" in field or "large" in field or "institution" in field:
        return "large"
    if "_md_" in field or "mid_large" in field:
        return "mid"
    if "_sm_" in field or "small" in field or "retail" in field:
        return "retail"
    return "none"


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
    return {
        "can_y": data_family != "control",
        "can_x": True,
        "allow_log": unit_type in {"currency", "price", "volume"},
        "allow_current": True,
        "allow_lag": False,
        "allow_diff": unit_type != "event",
        "allow_pct": unit_type in {"price", "currency", "ratio", "rate", "growth", "volume"},
        "allow_std": False,
        "family": data_family,
        "unit_type": unit_type,
        "statement": "behavior",
        "period_type": window,
        "direction": direction,
        "add_group": data_family,
        "allow_industry_relative": True,
    }


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
        "sub_family": sub_family,
        "unit_type": unit_type,
        "window": window,
        "session": _session(row["field"]) if data_family == "orderbook" else "full",
        "investor_type": _investor_type(row["field"]) if data_family == "moneyflow" else "none",
        "direction": direction,
        "allowed_slots": list(dict.fromkeys(slots)),
        "allowed_unary_ops": _allowed_unary(data_family, unit_type, sub_family, slots),
    }
    return _core_rule(data_family, unit_type, direction, window), behavior_rule


def build_metadata(source_md: Path) -> dict[str, object]:
    rows = extract_field_rows(source_md)
    if not rows:
        raise RuntimeError(f"no field rows parsed from {source_md}")

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
        core_rule, behavior_rule = classify_field(row)
        field = row["field"]
        field_rules[field] = core_rule
        behavior_rules[field] = behavior_rule

    data_family_counts = Counter(rule["data_family"] for rule in behavior_rules.values())
    return {
        "dataset": "real_behavior_daily.parquet",
        "format": "parquet",
        "index": ["Datetime", "Contract"],
        "source_document": str(source_md),
        "notes": [
            "Generated from the real behavior-finance field explanation markdown.",
            "Only executable metadata constraints are included; descriptive extension fields are intentionally omitted.",
            "Label, tradeable, and industry columns are excluded from searchable field rules and kept as top-level metadata.",
        ],
        "label_fields": label_fields,
        "tradeable_field": tradeable_field,
        "industry_field": industry_field,
        "size_field": size_field,
        "barra_style_fields": barra_style_fields,
        "field_count": len(behavior_rules),
        "data_family_counts": dict(sorted(data_family_counts.items())),
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
        default=root / "data" / "metadata" / "real_behavior_metadata.json",
    )
    args = parser.parse_args()

    metadata = build_metadata(args.source_md)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"field_count={metadata['field_count']}")
    print(f"data_family_counts={metadata['data_family_counts']}")


if __name__ == "__main__":
    main()
