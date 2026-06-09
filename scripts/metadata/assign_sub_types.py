"""Assign sub_type to every field in real_behavior_metadata.json.

One-shot script. Reads the metadata, classifies each field's sub_type based on
sub_family and field name patterns, then writes back.

sub_type taxonomy (~15 types):
  fundamental:
    fund_growth  — growth rates
    fund_quality — cashflow quality, margin spread, debt coverage
    fund_value   — valuation ratios, EV, yield

  price_volume:
    pv_momentum   — returns, momentum indicators
    pv_volume     — volume, turnover, amount
    pv_volatility — vol, ATR, amplitude
    pv_crowding   — crowding, overbought, volume shock
    pv_panic      — drawdown, panic, oversold
    pv_general    — catch-all for price_volume not in above

  moneyflow:
    mf_large  — large/elg order flow
    mf_small  — small/retail order flow
    mf_active — active buy/sell imbalance
    mf_general — catch-all moneyflow

  orderbook:
    ob_spread   — bid-ask spread
    ob_depth    — book depth
    ob_pressure — imbalance, net bid change, buy/sell pressure
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "metadata" / "real_behavior_metadata.json"


def classify_sub_type(field: str, rule: dict) -> str:
    """Classify a field's sub_type from its name, sub_family, and behavior_roles."""
    fam = rule.get("data_family", "")
    sub = rule.get("sub_family", "")
    roles = set(rule.get("behavior_roles", []))

    # ── fundamental ──────────────────────────────────────────
    if fam == "fundamental":
        # Growth
        if any(kw in field for kw in [
            "_growth_", "growth_accel", "inc_revenue", "inc_return",
            "inc_book_per_share", "balanced_growth_", "growth_dispersion",
            "growth_positive_count", "growth_rank_",
        ]):
            return "fund_growth"
        # Quality / cashflow
        if any(kw in field for kw in [
            "cashflow_", "ocf_", "fcff", "fcfe", "free_cashflow",
            "core_profit_quality", "margin_quality", "operating_margin",
            "cash_to_liabilities", "cash_equivalent",
            "surplus_cash", "cashflow", "ocf_per_share_vs",
            "roe_leverage_adjusted", "adjusted_roe_gap", "roic_vs_roa_gap",
            "leverage_change", "equity_multiplier_change",
            "interest_debt_pressure", "short_debt_pressure",
            "liquidity_buffer", "asset_liquidity_structure",
            "intangible_leverage_pressure", "net_debt_to_book_value",
            "retained_earnings_book_support",
        ]):
            return "fund_quality"
        if any(kw in field for kw in [
            "_vs_", "_gap",
        ]) and "growth" not in field:
            return "fund_quality"
        # Valuation
        if any(kw in field for kw in [
            "ep_ratio", "cfp_ratio", "book_to_market", "sp_ratio",
            "dividend_yield", "peg_ratio",
            "ev_", "ev_to_ebitda", "ev_no_cash", "ev_ebitda_revaluation",
            "ep_revaluation", "cfp_revaluation", "bm_revaluation", "sp_revaluation",
            "value_composite", "deep_value", "valuation_dispersion", "valuation_consensus",
            "ev_cash_adjustment", "ev_no_cash_discount",
        ]):
            return "fund_value"
        # Quality gap types
        if "quality" in roles or "cashflow_quality" in roles:
            return "fund_quality"
        if "valuation" in roles:
            return "fund_value"
        if "growth" in roles:
            return "fund_growth"
        # Debt / leverage → quality
        if any(kw in field for kw in [
            "debt_", "leverage", "interest_bearing", "net_debt",
            "current_ratio", "quick_ratio", "super_quick_ratio", "cash_ratio",
            "liabilities_per_share", "capital_to_equity", "equity_multiplier",
            "working_capital", "ebitda_to_debt",
        ]):
            return "fund_quality"
        # Turnover / cycle → quality
        if any(kw in field for kw in [
            "turnover_", "operating_cycle", "cash_conversion_cycle",
            "average_payment_period", "account_payable", "account_receivable",
            "inventory_turnover", "asset_turnover", "equity_turnover",
        ]):
            return "fund_quality"
        # Cost / expense / profit pressure
        if any(kw in field for kw in [
            "cost_to_sales", "expense_to_revenue", "expense_pressure",
            "non_operating_profit", "investment_profit_to",
            "income_tax_to", "adjusted_profit_to",
        ]):
            return "fund_quality"
        # Profitability
        if any(kw in field for kw in [
            "return_on_", "net_profit_margin", "gross_profit_margin",
            "profit_from_operation", "ebit_to_revenue", "ebit_per_share",
            "du_", "income_from_main",
        ]):
            return "fund_value"
        # Per-share metrics
        if any(kw in field for kw in [
            "book_value_per_share", "diluted_earnings", "adjusted_earnings",
            "operating_revenue_per_share", "operating_total_revenue_per_share",
            "tangible_asset_per_share",
        ]):
            return "fund_value"
        # Cashflow per share
        if any(kw in field for kw in [
            "cash_flow_per_share", "operating_cash_flow_per_share",
            "free_cash_flow", "depreciation_and_amortization",
        ]):
            return "fund_quality"
        # Asset structure
        if any(kw in field for kw in [
            "current_asset_to_total", "non_current_asset_to_total",
            "fixed_asset_ratio", "intangible_asset_ratio", "equity_fixed_asset_ratio",
        ]):
            return "fund_quality"
        # Growth revision flags
        if "growth_last_revision" in field or "growth_update_flag" in field:
            return "fund_growth"
        # Time interest earned
        if "time_interest_earned" in field:
            return "fund_quality"
        # Review: revenue/asset efficiency → quality
        if "efficiency" in field or "roe_vs_net_asset" in field:
            return "fund_quality"
        # Fallback for fundamental
        if "profit_growth" in sub or "growth" in sub:
            return "fund_growth"
        return "fund_value"

    # ── price_volume ──────────────────────────────────────────
    if fam == "price_volume":
        # Momentum
        if any(kw in field for kw in [
            "ret_", "momentum", "MACD", "MATRIX", "TRIX", "RSI",
            "KDJ", "SKD", "WR", "LWR", "CCI", "ROC", "OSC", "MTM", "MAMTM",
            "BIAS", "MABIAS",
        ]):
            return "pv_momentum"
        # Panic / drawdown
        if any(kw in field for kw in [
            "drawdown", "MDD", "panic", "oversold",
            "drawdown_volume_panic", "orderbook_panic_reversal",
        ]):
            return "pv_panic"
        # Crowding / overbought
        if any(kw in field for kw in [
            "crowding", "crowded", "overbought", "shock_",
            "attention_heat", "short_term_overheat",
        ]):
            return "pv_crowding"
        # Volatility
        if any(kw in field for kw in [
            "volatility", "VOLT", "ATR", "TR", "AMP", "range_",
            "BOLL", "BBIBOLL", "MASS",
        ]):
            return "pv_volatility"
        # Volume / turnover / amount
        if any(kw in field for kw in [
            "volume", "turnover", "amount", "VOL", "VMA", "AMV",
            "DAVOL", "QTYR", "TAPI", "MATAPI", "OBV", "MFI", "VR", "MAVR",
            "CYF", "CYR", "MACYR",
        ]):
            return "pv_volume"
        # MA / EMA / WMA / spread
        if any(kw in field for kw in [
            "MA5", "MA10", "MA20", "MA60", "MA120", "MA250",
            "EMA5", "EMA10", "EMA20", "EMA60", "EMA120", "EMA250",
            "WMA5", "WMA10", "WMA20", "WMA60", "WMA120", "WMA250",
            "ma_", "ema_",
        ]) and "MACD" not in field and "MARSI" not in field and "MAASS" not in field:
            return "pv_momentum"
        # Trending indicators
        if any(kw in field for kw in [
            "DPO", "MADPO", "DKX", "MADKX", "BBI",
            "DII", "DI1", "DI2", "ADX", "ADXR",
            "AROON", "AR", "BR", "CR", "MACR",
            "ADTM", "MAADTM", "ASI", "ASIT", "ACCER",
            "PCNT", "SY", "SWL", "SWS",
            "MCST", "UDL", "MAUDL",
        ]):
            return "pv_momentum"
        # Divergence / dispersion
        if any(kw in field for kw in [
            "divergence", "dispersion", "consensus",
        ]):
            return "pv_crowding"
        # Flow/price composite
        if any(kw in field for kw in [
            "flow_price", "institution_accumulation",
            "retail_chase_risk", "orderbook_chase_risk",
            "large_order_confirmed", "moneyflow_crowding",
            "liquidity_neglect",
        ]):
            return "pv_crowding"
        # Trend composite
        if any(kw in field for kw in [
            "trend_", "reversal_signal", "momentum_drawdown", "macd_vol",
            "aroon_direction", "aroon_trend",
            "trend_strength",
        ]):
            return "pv_momentum"
        # Fallback
        return "pv_general"

    # ── moneyflow ──────────────────────────────────────────
    if fam == "moneyflow":
        # Large / institutional
        if any(kw in field for kw in [
            "lg", "_lg_", "large", "elg", "_elg_",
            "institution_accumulation", "large_order_confirmed",
        ]):
            return "mf_large"
        # Small / retail
        if any(kw in field for kw in [
            "sm", "_sm_", "small", "retail",
        ]):
            return "mf_small"
        # Active buy/sell
        if any(kw in field for kw in [
            "active_buy", "active_sell", "active_imbalance",
            "net_amount_ratio", "net_mf",
            "mf_buy_amount_total", "mf_sell_amount_total",
        ]):
            return "mf_active"
        # Net buy/sell volume ratios
        if any(kw in field for kw in [
            "mf_net_sm_vol", "mf_net_md_vol", "mf_net_lg_vol", "mf_net_elg_vol",
            "mf_trade_amount_total",
        ]):
            return "mf_active"
        # md (mid) → larger pool
        if "md" in field or "mid" in field:
            return "mf_active"
        # s1/s2/s3 → unclassified
        if any(field.endswith(s) for s in ["_s1", "_s2", "_s3"]):
            return "mf_general"
        # Large net ratio
        if "mf_large_vs_small" in field or "mf_mid_large" in field:
            return "mf_active"
        # Large positive days
        if "large_positive_days" in field:
            return "mf_large"
        # Participation
        if "participation" in field:
            return "mf_active"
        # Buy/sell volumes and amounts (raw)
        if any(kw in field for kw in [
            "buy_sm_", "sell_sm_", "buy_md_", "sell_md_",
            "buy_lg_", "sell_lg_", "buy_elg_", "sell_elg_",
        ]):
            if "sm" in field:
                return "mf_small"
            if "lg" in field or "elg" in field:
                return "mf_large"
            if "md" in field:
                return "mf_active"
            return "mf_general"
        # Moneyflow crowding
        if "moneyflow_crowding" in field:
            return "mf_active"
        # Fallback
        return "mf_general"

    # ── orderbook ──────────────────────────────────────────
    if fam == "orderbook":
        # Spread
        if "spread" in field:
            return "ob_spread"
        # Depth
        if "depth" in field:
            return "ob_depth"
        # Pressure / imbalance / net bid
        if any(kw in field for kw in [
            "imbalance", "net_bid", "buy_intent", "chase_pressure",
            "micro_price", "book_shape", "close_buying",
        ]):
            return "ob_pressure"
        # Liquidity stress
        if "liquidity_stress" in field:
            return "ob_spread"
        # Fallback
        return "ob_pressure"

    # ── control ──────────────────────────────────────────
    if fam == "control":
        return "control"

    return "unknown"


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rules = data["behavior_field_rules"]

    from collections import Counter
    stats = Counter()
    for field, rule in rules.items():
        st = classify_sub_type(field, rule)
        rule["sub_type"] = st
        stats[st] += 1

    data["field_count"] = len(rules)
    data["data_family_counts"] = dict(Counter(r["data_family"] for r in rules.values()))

    print(f"Total fields: {len(rules)}")
    print(f"sub_type distribution ({len(stats)} types):")
    for st, count in stats.most_common():
        print(f"  {st}: {count}")

    json.dump(data, open(SRC, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nWritten to {SRC}")


if __name__ == "__main__":
    main()
