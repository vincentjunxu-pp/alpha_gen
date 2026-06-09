# ======================基础字段汇总======================
# 1.增长基础字段
growth_base_cols = [
    "inc_revenue_lyr",
    "inc_revenue_ttm",
    "inc_return_on_equity_lyr",
    "inc_return_on_equity_ttm",
    "inc_book_per_share_lyr",
    "inc_book_per_share_ttm",
    "inc_book_per_share_lf",
    "operating_profit_growth_ratio_lyr",
    "operating_profit_growth_ratio_ttm",
    "net_profit_growth_ratio_lyr",
    "net_profit_growth_ratio_ttm",
    "profit_growth_ratio_lyr",
    "profit_growth_ratio_ttm",
    "gross_profit_growth_ratio_lyr",
    "gross_profit_growth_ratio_ttm",
    "operating_revenue_growth_ratio_lyr",
    "operating_revenue_growth_ratio_ttm",
    "net_asset_growth_lyr",
    "net_asset_growth_ratio_ttm",
    "net_asset_growth_ratio_lf",
    "total_asset_growth_ratio_lyr",
    "total_asset_growth_ratio_ttm",
    "total_asset_growth_ratio_lf",
    "net_profit_parent_company_growth_ratio_lyr",
    "net_profit_parent_company_growth_ratio_ttm",
    "net_cash_flow_growth_lyr",
    "net_cash_flow_growth_ratio_ttm",
    "net_operate_cash_flow_growth_ratio_lyr",
    "net_operate_cash_flow_growth_ratio_ttm",
    "net_investing_cash_flow_growth_ratio_lyr",
    "net_investing_cash_flow_growth_ratio_ttm",
    "net_financing_cash_flow_growth_ratio_lyr",
    "net_financing_cash_flow_growth_ratio_ttm",
]

# 2.估值基础字段
# 已去掉 PE/PB/PS 等反向重复字段，尽量保留"越大越便宜"的方向
valuation_base_cols = [
    "ep_ratio_lyr",
    "ep_ratio_ttm",
    "cfp_ratio_lyr",
    "cfp_ratio_ttm",
    "book_to_market_ratio_lyr",
    "book_to_market_ratio_ttm",
    "book_to_market_ratio_lf",
    "dividend_yield_ttm",
    "dividend_yield_lyr",
    "peg_ratio_lyr",
    "peg_ratio_ttm",
    "sp_ratio_lyr",
    "sp_ratio_ttm",

    # size / market value controls
    "market_cap",
    # "market_cap_2",
    # "market_cap_3",
    "a_share_market_val",
    "a_share_market_val_in_circulation",

    # enterprise value related
    "ev_lyr",
    "ev_lf",
    "ev_ttm",
    "ev_no_cash_lyr",
    "ev_no_cash_ttm",
    "ev_no_cash_lf",
    "ev_to_ebitda_lyr",
    "ev_to_ebitda_ttm",
    "ev_no_cash_to_ebit_lyr",
    "ev_no_cash_to_ebit_ttm",
]

# 3.财务基础字段
financial_structure_cols = [
    # leverage / capital structure
    "debt_to_asset_ratio_ttm",
    "debt_to_asset_ratio_lyr",
    "debt_to_asset_ratio_lf",
    "equity_multiplier_ttm",
    "equity_multiplier_lyr",
    "equity_multiplier_lf",
    "capital_to_equity_ratio_ttm",
    "capital_to_equity_ratio_lf",
    "debt_to_equity_ratio_ttm",
    "debt_to_equity_ratio_lf",
    "book_leverage_ttm",
    "market_leverage_ttm",

    # interest-bearing debt
    "interest_bearing_debt_ttm",
    "interest_bearing_debt_lf",
    "interest_bearing_debt_to_capital_ttm",
    "interest_bearing_debt_to_capital_lf",
    "net_debt_ttm",
    "net_debt_lf",

    # debt maturity structure
    "current_debt_to_total_debt_ttm",
    "current_debt_to_total_debt_lf",
    "non_current_debt_to_total_debt_ttm",
    "non_current_debt_to_total_debt_lf",

    # liquidity / solvency
    "current_ratio_ttm",
    "current_ratio_lf",
    "quick_ratio_ttm",
    "quick_ratio_lf",
    "super_quick_ratio_ttm",
    "super_quick_ratio_lf",
    "cash_ratio_ttm",
    "cash_ratio_lf",

    # asset structure
    "current_asset_to_total_asset_lf",
    "current_asset_to_total_asset_ttm",
    "non_current_asset_to_total_asset_ttm",
    "non_current_asset_to_total_asset_lf",
    "fixed_asset_ratio_ttm",
    "fixed_asset_ratio_lf",
    "intangible_asset_ratio_ttm",
    "intangible_asset_ratio_lf",
    "equity_fixed_asset_ratio_ttm",
    "equity_fixed_asset_ratio_lf",

    # working capital
    "working_capital_ttm",
    "working_capital_lf",
    "net_working_capital_ttm",
    "net_working_capital_lf",
    "long_term_debt_to_working_capital_ttm",
    "long_term_debt_to_working_capital_lf",

    # per-share balance-sheet quality
    "book_value_per_share_ttm",
    "book_value_per_share_lf",
    "retained_earnings_per_share_ttm",
    "retained_earnings_per_share_lf",
    "undistributed_profit_per_share_ttm",
    "undistributed_profit_per_share_lf",
    "tangible_asset_per_share_ttm",
    "tangible_asset_per_share_lf",
    "cash_equivalent_per_share_ttm",
    "cash_equivalent_per_share_lf",
    "liabilities_per_share_ttm",
    "liabilities_per_share_lf",
]

# 4.经营基础字段
operating_quality_cols = [
    # EPS / per-share operating result
    "diluted_earnings_per_share_ttm",
    "adjusted_earnings_per_share_ttm",
    "adjusted_fully_diluted_earnings_per_share_ttm",
    "operating_total_revenue_per_share_ttm",
    "operating_revenue_per_share_ttm",
    "ebit_per_share_ttm",

    # profitability / return
    "return_on_equity_ttm",
    "return_on_equity_diluted_ttm",
    "adjusted_return_on_equity_ttm",
    "adjusted_return_on_equity_diluted_ttm",
    "return_on_asset_ttm",
    "return_on_asset_net_profit_ttm",
    "return_on_invested_capital_ttm",

    # margin structure
    "net_profit_margin_ttm",
    "gross_profit_margin_ttm",
    "cost_to_sales_ttm",
    "net_profit_to_revenue_ttm",
    "profit_from_operation_to_revenue_ttm",
    "ebit_to_revenue_ttm",
    "expense_to_revenue_ttm",

    # profit composition
    "operating_profit_to_profit_before_tax_ttm",
    "investment_profit_to_profit_before_tax_ttm",
    "non_operating_profit_to_profit_before_tax_ttm",
    "income_tax_to_profit_before_tax_ttm",
    "adjusted_profit_to_total_profit_ttm",

    # debt service from operations
    "ebitda_to_debt_ttm",
    "time_interest_earned_ratio_ttm",

    # turnover / efficiency
    "account_payable_turnover_rate_ttm",
    "account_payable_turnover_days_ttm",
    "account_receivable_turnover_rate_ttm",
    "account_receivable_turnover_days_ttm",
    "inventory_turnover_ttm",
    "current_asset_turnover_ttm",
    "fixed_asset_turnover_ttm",
    "total_asset_turnover_ttm",
    "equity_turnover_ratio_ttm",
    "operating_cycle_ttm",
    "average_payment_period_ttm",
    "cash_conversion_cycle_ttm",

    # DuPont style
    "du_profit_margin_ttm",
    "du_return_on_equity_ttm",
    "du_return_on_sales_ttm",
    "income_from_main_operations_ttm",
]

# 5.现金流基础字段
cashflow_quality_cols = [
    # per-share cash flow
    "cash_flow_per_share_ttm",
    "operating_cash_flow_per_share_ttm",
    "free_cash_flow_company_per_share_ttm",
    "free_cash_flow_equity_per_share_ttm",

    # free cash flow
    "fcff_ttm",
    "fcfe_ttm",

    # cash flow coverage
    "ocf_to_debt_ttm",
    "surplus_cash_protection_multiples_ttm",
    "ocf_to_interest_bearing_debt_ttm",
    "ocf_to_current_ratio_ttm",
    "ocf_to_net_debt_ttm",

    # non-cash expense
    "depreciation_and_amortization_lyr",
]

base_cols = growth_base_cols + valuation_base_cols + financial_structure_cols + operating_quality_cols + cashflow_quality_cols

print("\n--- 开始合并全市场面板 ---")
panel_raw = pd.concat(
    table_dfs.values(),
    axis=1,
    join="outer",
).sort_index()


# ======================衍生指标计算======================
## 一、估值衍生指标
# 1. 估值重估: ttm - lyr
panel_raw = panel_raw.sort_index()
panel_raw["ep_revaluation"] = panel_raw["ep_ratio_ttm"] - panel_raw["ep_ratio_lyr"]
panel_raw["cfp_revaluation"] = panel_raw["cfp_ratio_ttm"] - panel_raw["cfp_ratio_lyr"]
panel_raw["bm_revaluation"] = panel_raw["book_to_market_ratio_ttm"] - panel_raw["book_to_market_ratio_lyr"]
panel_raw["sp_revaluation"] = panel_raw["sp_ratio_ttm"] - panel_raw["sp_ratio_lyr"]
panel_raw["ev_ebitda_revaluation"] = panel_raw["ev_to_ebitda_ttm"] - panel_raw["ev_to_ebitda_lyr"]

# 2. 不同估值锚之间的差异
panel_raw["cashflow_vs_earnings_yield_gap"] = panel_raw["cfp_ratio_ttm"] - panel_raw["ep_ratio_ttm"]
panel_raw["book_vs_earnings_yield_gap"] = panel_raw["book_to_market_ratio_ttm"] - panel_raw["ep_ratio_ttm"]
panel_raw["sales_vs_earnings_yield_gap"] = panel_raw["sp_ratio_ttm"] - panel_raw["ep_ratio_ttm"]
panel_raw["dividend_vs_earnings_yield_gap"] = panel_raw["dividend_yield_ttm"] - panel_raw["ep_ratio_ttm"]

# 3. EV 现金调整
panel_raw["ev_no_cash_discount_ttm"] = panel_raw["ev_no_cash_ttm"] - panel_raw["ev_ttm"]
panel_raw["ev_no_cash_discount_lf"] = panel_raw["ev_no_cash_lf"] - panel_raw["ev_lf"]
panel_raw["ev_cash_adjustment_gap"] = panel_raw["ev_no_cash_to_ebit_ttm"] - panel_raw["ev_to_ebitda_ttm"]

# 4. 横截面 rank 后的综合价值
value_cols = [
    "ep_ratio_ttm",
    "cfp_ratio_ttm",
    "book_to_market_ratio_ttm",
    "sp_ratio_ttm",
    "dividend_yield_ttm",
]
value_rank = panel_raw[value_cols].groupby(level="Datetime").rank(pct=True)
panel_raw["value_composite_rank"] = value_rank.mean(axis=1)
panel_raw["deep_value_score"] = value_rank.min(axis=1)
panel_raw["valuation_dispersion"] = value_rank.std(axis=1)
panel_raw["valuation_consensus"] = value_rank.gt(0.7).sum(axis=1)


## 二、增长衍生指标
# 1.增长幅度: ttm - lyr
panel_raw["rev_growth_accel"] = panel_raw["operating_revenue_growth_ratio_ttm"] - panel_raw["operating_revenue_growth_ratio_lyr"]
panel_raw["gross_profit_growth_accel"] = panel_raw["gross_profit_growth_ratio_ttm"] - panel_raw["gross_profit_growth_ratio_lyr"]
panel_raw["oper_profit_growth_accel"] = panel_raw["operating_profit_growth_ratio_ttm"] - panel_raw["operating_profit_growth_ratio_lyr"]
panel_raw["net_profit_growth_accel"] = panel_raw["net_profit_growth_ratio_ttm"] - panel_raw["net_profit_growth_ratio_lyr"]
panel_raw["parent_profit_growth_accel"] = panel_raw["net_profit_parent_company_growth_ratio_ttm"] - panel_raw["net_profit_parent_company_growth_ratio_lyr"]
panel_raw["oper_cash_growth_accel"] = panel_raw["net_operate_cash_flow_growth_ratio_ttm"] - panel_raw["net_operate_cash_flow_growth_ratio_lyr"]

# 2.增长传导错配
panel_raw["gross_vs_revenue_growth_gap"] = panel_raw["gross_profit_growth_ratio_ttm"] - panel_raw["operating_revenue_growth_ratio_ttm"]
panel_raw["oper_vs_revenue_growth_gap"] = panel_raw["operating_profit_growth_ratio_ttm"] - panel_raw["operating_revenue_growth_ratio_ttm"]
panel_raw["net_vs_oper_growth_gap"] = panel_raw["net_profit_growth_ratio_ttm"] - panel_raw["operating_profit_growth_ratio_ttm"]
panel_raw["parent_vs_gross_growth_gap"] = panel_raw["net_profit_parent_company_growth_ratio_ttm"] - panel_raw["gross_profit_growth_ratio_ttm"]

# 3.主业利润质量
panel_raw["oper_vs_net_profit_growth_gap"] = panel_raw["operating_profit_growth_ratio_ttm"] - panel_raw["net_profit_growth_ratio_ttm"]
panel_raw["parent_vs_net_profit_growth_gap"] = panel_raw["net_profit_parent_company_growth_ratio_ttm"] - panel_raw["net_profit_growth_ratio_ttm"]

# 4.现金流支撑
panel_raw["ocf_vs_net_profit_growth_gap"] = panel_raw["net_operate_cash_flow_growth_ratio_ttm"] - panel_raw["net_profit_growth_ratio_ttm"]
panel_raw["ocf_vs_oper_profit_growth_gap"] = panel_raw["net_operate_cash_flow_growth_ratio_ttm"] - panel_raw["operating_profit_growth_ratio_ttm"]
panel_raw["ocf_vs_revenue_growth_gap"] = panel_raw["net_cash_flow_growth_ratio_ttm"] - panel_raw["operating_revenue_growth_ratio_ttm"]

# 5.资产扩张效率
panel_raw["asset_rev_growth_efficiency"] = panel_raw["operating_revenue_growth_ratio_ttm"] - panel_raw["total_asset_growth_ratio_ttm"]
panel_raw["asset_profit_growth_efficiency"] = panel_raw["net_profit_growth_ratio_ttm"] - panel_raw["total_asset_growth_ratio_ttm"]
panel_raw["asset_operate_cash_efficiency"] = panel_raw["net_operate_cash_flow_growth_ratio_ttm"] - panel_raw["total_asset_growth_ratio_ttm"]

# 6.融资依赖与扩张压力
panel_raw["financing_vs_ocf_growth_gap"] = panel_raw["net_financing_cash_flow_growth_ratio_ttm"] - panel_raw["net_operate_cash_flow_growth_ratio_ttm"]
panel_raw["financing_vs_profit_growth_gap"] = panel_raw["net_financing_cash_flow_growth_ratio_ttm"] - panel_raw["net_profit_growth_ratio_ttm"]
panel_raw["asset_vs_net_asset_growth_gap"] = panel_raw["total_asset_growth_ratio_ttm"] - panel_raw["net_asset_growth_ratio_ttm"]

# 7.均衡增长
quality_growth_cols = [
    "operating_revenue_growth_ratio_ttm",
    "gross_profit_growth_ratio_ttm",
    "operating_profit_growth_ratio_ttm",
    "net_profit_growth_ratio_ttm",
    "net_operate_cash_flow_growth_ratio_ttm",
    "inc_return_on_equity_ttm",
]
panel_raw["balanced_growth_mean"] = panel_raw[quality_growth_cols].mean(axis=1)
panel_raw["balanced_growth_min"] = panel_raw[quality_growth_cols].min(axis=1)
panel_raw["growth_dispersion"] = panel_raw[quality_growth_cols].std(axis=1)
panel_raw["growth_positive_count"] = panel_raw[quality_growth_cols].gt(0).sum(axis=1)

ranked_growth = panel_raw[quality_growth_cols].groupby(level="Datetime").rank(pct=True)
panel_raw["balanced_growth_rank_mean"] = ranked_growth.mean(axis=1)
panel_raw["balanced_growth_rank_min"] = ranked_growth.min(axis=1)
panel_raw["balanced_growth_rank_dispersion"] = ranked_growth.std(axis=1)

# 8.最近一次非零修正
def last_nonzero_revision(s:pd.Series):
    delta = s.groupby("Contract").diff()
    delta = delta.where(delta.abs()>0, np.nan)
    return delta.groupby("Contract").ffill()

def update_flag(s, eps: float = 0.0):
    delta = s.groupby(level="Contract").diff()
    return (delta.abs() > eps).astype(float)

panel_raw["operating_profit_growth_last_revision"] = last_nonzero_revision(panel_raw["operating_profit_growth_ratio_lyr"])
panel_raw["net_profit_growth_last_revision"] = last_nonzero_revision(panel_raw["net_profit_growth_ratio_lyr"])
panel_raw["parent_profit_last_revision"] = last_nonzero_revision(panel_raw["net_profit_parent_company_growth_ratio_lyr"])
panel_raw["operate_cash_last_revision"] = last_nonzero_revision(panel_raw["net_operate_cash_flow_growth_ratio_lyr"])

panel_raw["operating_profit_growth_update_flag"] = update_flag(panel_raw["operating_profit_growth_ratio_lyr"])
panel_raw["net_profit_growth_update_flag"] = update_flag(panel_raw["net_profit_growth_ratio_lyr"])
panel_raw["oper_cash_growth_update_flag"] = update_flag(panel_raw["net_operate_cash_flow_growth_ratio_lyr"])

# 杠杆与偿债衍生
panel_raw["leverage_change"] = panel_raw["debt_to_asset_ratio_ttm"] - panel_raw["debt_to_asset_ratio_lyr"]
panel_raw["equity_multiplier_change"] = panel_raw["equity_multiplier_ttm"] - panel_raw["equity_multiplier_lyr"]
panel_raw["interest_debt_pressure"] = panel_raw["interest_bearing_debt_to_capital_ttm"] + panel_raw["interest_bearing_debt_to_asset_ratio_ttm"]
panel_raw["short_debt_bufferr"] = panel_raw["current_debt_to_total_debt_ttm"] - panel_raw["quick_ratio_ttm"]

panel_raw["liquidity_bufferr"] = panel_raw["quick_ratio_ttm"] + panel_raw["super_quick_ratio_ttm"] - panel_raw["cash_ratio_ttm"] - panel_raw["current_debt_to_total_debt_ttm"]
panel_raw["asset_liability_structure"] = panel_raw["current_asset_to_total_asset_ttm"] / panel_raw["fixed_asset_ratio_ttm"]
panel_raw["intangible_leverage_pressure"] = panel_raw["intangible_asset_ratio_ttm"] / panel_raw["total_asset_turnover_ttm"]

panel_raw["net_debt_to_book"] = safe_div(panel_raw["net_debt_ttm"], panel_raw["book_value_per_share_ttm"])
panel_raw["cash_eq_to_share_support"] = safe_div(panel_raw["cash_equivalent_per_share_ttm"], panel_raw["liabilities_per_share_ttm"])
panel_raw["retained_earnings_support"] = safe_div(panel_raw["retained_earnings_per_share_ttm"], panel_raw["book_value_per_share_ttm"])


## 三、经营质量衍生
panel_raw["roe_leverage_adjusted"] = panel_raw["return_on_equity_ttm"] - panel_raw["equity_multiplier_ttm"]
panel_raw["adjusted_roe_gap"] = panel_raw["adjusted_return_on_equity_ttm"] - panel_raw["return_on_equity_ttm"]
panel_raw["roic_vs_roa_gap"] = panel_raw["return_on_invested_capital_ttm"] - panel_raw["return_on_asset_ttm"]

panel_raw["margin_quality_spread"] = panel_raw["gross_profit_margin_ttm"] - panel_raw["net_profit_margin_ttm"]
panel_raw["operating_margin_quality"] = panel_raw["profit_from_operation_to_revenue_ttm"] - panel_raw["net_profit_to_revenue_ttm"]

panel_raw["expense_pressure"] = panel_raw["expense_to_revenue_ttm"] + panel_raw["cost_to_sales_ttm"]
panel_raw["non_operating_profit_pressure"] = panel_raw["non_operating_profit_to_profit_before_tax_ttm"] + panel_raw["investment_profit_to_profit_before_tax_ttm"]
panel_raw["core_profit_quality"] = panel_raw["operating_profit_to_profit_before_tax_ttm"] - panel_raw["non_operating_profit_to_profit_before_tax_ttm"]

panel_raw["turnover_efficiency_score"] = panel_raw["account_receivable_turnover_rate_ttm"] + panel_raw["inventory_turnover_ttm"] + panel_raw["total_asset_turnover_ttm"]
# 注意: inventory_turnover_ttm 周转率越高越好, 建议替换为 inventory_turnover_days
panel_raw["working_capital_cycle_pressure"] = panel_raw["account_receivable_turnover_days_ttm"] + panel_raw["inventory_turnover_ttm"].rank(pct=True) + panel_raw["cash_conversion_cycle_ttm"]

# 现金流质量衍生
panel_raw["ocf_per_share_vs_eps"] = panel_raw["operating_cash_flow_per_share_ttm"] - panel_raw["diluted_earnings_per_share_ttm"]
panel_raw["fcff_vs_ocf_gap"] = panel_raw["fcff_ttm"] - panel_raw["operating_cash_flow_per_share_ttm"]
panel_raw["fcfe_vs_ocf_gap"] = panel_raw["fcfe_ttm"] - panel_raw["operating_cash_flow_per_share_ttm"]

panel_raw["cashflow_debt_coverage"] = panel_raw["ocf_to_debt_ttm"] + panel_raw["ocf_to_interest_bearing_debt_ttm"] + panel_raw["ocf_to_net_debt_ttm"]
panel_raw["cashflow_short_debt_buffer"] = panel_raw["ocf_to_current_ratio_ttm"] + panel_raw["quick_ratio_ttm"] - panel_raw["current_debt_to_total_debt_ttm"]

panel_raw["free_cashflow_quality"] = panel_raw["free_cash_flow_company_per_share_ttm"] + panel_raw["free_cash_flow_equity_per_share_ttm"] - panel_raw["cash_flow_per_share_ttm"]
panel_raw["cashflow_interest_safety"] = panel_raw["ocf_to_interest_bearing_debt_ttm"] + panel_raw["time_interest_earned_ratio_ttm"]



import pandas as pd
import numpy as np

# ======================一、量价基础字段与量价因子衍生======================
# 1.基础量价指标列表
price_volume_base_cols = [
    # MACD / trend
    "MACD_DIFF", "MACD_DEA", "MACD_HIST",
    "TRIX", "MATRIX", "DPO", "MADPO", "DKX", "MADKX",

    # BOLL / price deviation / channel
    "BOLL", "BOLL_UP", "BOLL_DOWN",
    "BBI", "BBIBOLL_UP", "BBIBOLL_DOWN",
    "BIAS5", "BIAS10", "BIAS20", "BIAS36", "BIAS612", "MABIAS",

    # moving average representatives
    "MA5", "MA10", "MA20", "MA60", "MA120", "MA250",
    "EMA5", "EMA10", "EMA20", "EMA60", "EMA120", "EMA250",
    "WMA5", "WMA10", "WMA20", "WMA60", "WMA120", "WMA250",

    # reversal / oscillator
    "RSI6", "RSI10", "MARSI6", "MARSI10",
    "KDJ_K", "KDJ_D", "KDJ_J",
    "SKD_K", "SKD_D",
    "WR", "LWR1", "LWR2",
    "CCI", "ROC", "OSC", "MTM", "MAMTM",

    # volatility / range / drawdown
    "TR", "ATR",
    "MDD20", "MDD60",
    "AMP1", "AMP3", "AMP5", "AMP10", "AMP20", "AMP60",
    "MASS", "MAMASS",

    # volume / attention / crowding
    "VOL5", "VOL10", "VOL20", "VOL60", "VOL120", "VOL250",
    "DAVOL5", "DAVOL10", "DAVOL20",
    "VMA5", "VMA10", "VMA20", "VMA60", "VMA120", "VMA250",
    "AMV5", "AMV10", "AMV20", "AMV60", "AMV120", "AMV250",
    "TAPI", "MATAPI", "OBV", "MFI", "VR", "MAVR",

    # trend strength / directional movement
    "DI1", "DI2",
    "ADX", "ADXR",
    "AROON_UP", "AROON_DOWN",

    # sentiment / pressure / energy
    "AR", "BR", "CR",
    "MACR1", "MACR2", "MACR3", "MACR4",
    "CYF", "CYR", "MACYR",
    "PCNT", "SY", "SWL", "SWS", "ADTM", "MAADTM",
    "ASI", "ASIT", "ACCER", "MCST",

    # other quantity-price state
    "UDL", "MAUDL",
    "VOLT20", "VOLT60",
    "QTYR_5_20",
]

alpha = alpha[price_volume_base_cols]
alpha = alpha.sort_index()

# 通用函数
def cs_rank(s):
    return s.groupby(level="Datetime").rank(pct=True)

def safe_div(a, b, eps=1e-12):
    return a / b.where(b.abs() > eps)

# ------------------1.均线偏离: 不要直接用 MA 水平------------------
alpha["ma_5_20_spread"] = safe_div(alpha["MA5"], alpha["MA20"]) - 1
alpha["ma_20_60_spread"] = safe_div(alpha["MA20"], alpha["MA60"]) - 1
alpha["ma_60_120_spread"] = safe_div(alpha["MA60"], alpha["MA120"]) - 1
alpha["ema_5_20_spread"] = safe_div(alpha["EMA5"], alpha["EMA20"]) - 1
alpha["ema_20_60_spread"] = safe_div(alpha["EMA20"], alpha["EMA60"]) - 1

# ------------------2.成交量拥挤 / 注意力------------------
alpha["volume_crowding_5_20"] = safe_div(alpha["VOL5"], alpha["VOL20"]) - 1
alpha["volume_crowding_5_60"] = safe_div(alpha["VOL5"], alpha["VOL60"]) - 1
alpha["volume_crowding_20_120"] = safe_div(alpha["VOL20"], alpha["VOL120"]) - 1
alpha["amount_crowding_5_20"] = safe_div(alpha["AMV5"], alpha["AMV20"]) - 1
alpha["amount_crowding_20_60"] = safe_div(alpha["AMV20"], alpha["AMV60"]) - 1

# ------------------3.量价背离------------------
alpha["price_volume_divergence_5_20"] = cs_rank(alpha["ma_5_20_spread"]) - cs_rank(alpha["volume_crowding_5_20"])
alpha["price_volume_divergence_20_60"] = cs_rank(alpha["ma_20_60_spread"]) - cs_rank(alpha["volume_crowding_20_120"])
alpha["obv_price_divergence_20_60"] = cs_rank(alpha["OBV"]) - cs_rank(alpha["ma_20_60_spread"])
alpha["mfi_rsi_divergence"] = cs_rank(alpha["MFI"]) - cs_rank(alpha["RSI10"])

# ------------------4.超买超卖共振------------------
reversal_cols = ["RSI6", "RSI10", "KDJ_K", "KDJ_D", "KDJ_J", "WR", "CCI", "BIAS5", "BIAS20"]
reversal_cols = [c for c in reversal_cols if c in alpha.columns]
reversal_rank = alpha[reversal_cols].groupby(level="Datetime").rank(pct=True)

alpha["overbought_consensus"] = reversal_rank.mean(axis=1)
alpha["overbought_extreme_count"] = reversal_rank.gt(0.8).sum(axis=1)
alpha["oversold_extreme_count"] = reversal_rank.lt(0.2).sum(axis=1)
alpha["reversal_signal_dispersion"] = reversal_rank.std(axis=1)

# ------------------5.趋势确认------------------
trend_cols = ["MACD_DIFF", "MACD_HIST", "TRIX", "ROC", "ma_5_20_spread", "ma_20_60_spread"]
trend_cols = [c for c in trend_cols if c in alpha.columns]
trend_rank = alpha[trend_cols].groupby(level="Datetime").rank(pct=True)

alpha["trend_consensus"] = trend_rank.mean(axis=1)
alpha["trend_extreme_count"] = trend_rank.gt(0.8).sum(axis=1)
alpha["trend_dispersion"] = trend_rank.std(axis=1)

# ------------------6.波动调整后的趋势------------------
alpha["trend_vol_adjusted_20"] = safe_div(alpha["ma_5_20_spread"], alpha["ATR"].abs())
alpha["momentum_drawdown_adjusted"] = safe_div(alpha["ROC"], alpha["MDD20"].abs())
alpha["momentum_wave_adjusted"] = safe_div(alpha["ROC"], alpha["MDD20"].abs())
alpha["macd_vol_adjusted"] = safe_div(alpha["MACD_HIST"], alpha["ATR"].abs())

# ------------------7.拥挤后的反转风险------------------
alpha["crowded_momentum_risk"] = cs_rank(alpha["ma_5_20_spread"]) * cs_rank(alpha["volume_crowding_5_20"]) * cs_rank(alpha["RSI6"])
alpha["crowded_overbought_risk"] = alpha["overbought_consensus"] * cs_rank(alpha["volume_crowding_5_20"])

# ------------------8.恐慌/错杀修复------------------
alpha["panic_oversold_score"] = cs_rank(alpha["MDD20"].abs()) * cs_rank(alpha["ATR"].abs()) * (1 - alpha["overbought_consensus"])
alpha["downside_volume_panic"] = cs_rank(alpha["MDD20"].abs()) * cs_rank(alpha["volume_crowding_5_20"])

# ------------------9.趋势强度与方向结合------------------
alpha["trend_strength_confirmed"] = cs_rank(alpha["ADX"]) * cs_rank(alpha["ma_20_60_spread"])
alpha["aroon_direction_gap"] = alpha["AROON_UP"] - alpha["AROON_DOWN"]
alpha["aroon_trend_confirmed"] = cs_rank(alpha["aroon_direction_gap"]) * cs_rank(alpha["ADX"])

# alpha.shape


# ======================二、资金流(moneyflow)因子计算======================
def div(a, b):
    return a / b.replace(0, np.nan)

# 1.净额计算
moneyflow["mf_net_sm_amount"] = moneyflow["buy_sm_amount"] - moneyflow["sell_sm_amount"]
moneyflow["mf_net_md_amount"] = moneyflow["buy_md_amount"] - moneyflow["sell_md_amount"]
moneyflow["mf_net_lg_amount"] = moneyflow["buy_lg_amount"] - moneyflow["sell_lg_amount"]
moneyflow["mf_net_elg_amount"] = moneyflow["buy_elg_amount"] - moneyflow["sell_elg_amount"]

moneyflow["mf_net_sm_vol"] = moneyflow["buy_sm_vol"] - moneyflow["sell_sm_vol"]
moneyflow["mf_net_md_vol"] = moneyflow["buy_md_vol"] - moneyflow["sell_md_vol"]
moneyflow["mf_net_lg_vol"] = moneyflow["buy_lg_vol"] - moneyflow["sell_lg_vol"]
moneyflow["mf_net_elg_vol"] = moneyflow["buy_elg_vol"] - moneyflow["sell_elg_vol"]

# 2.总额汇总
moneyflow["mf_buy_amount_total"] = (
    moneyflow["buy_sm_amount"] + moneyflow["buy_md_amount"] + moneyflow["buy_lg_amount"] + moneyflow["buy_elg_amount"]
)
moneyflow["mf_sell_amount_total"] = (
    moneyflow["sell_sm_amount"] + moneyflow["sell_md_amount"] + moneyflow["sell_lg_amount"] + moneyflow["sell_elg_amount"]
)
moneyflow["mf_trade_amount_total"] = moneyflow["mf_buy_amount_total"] + moneyflow["mf_sell_amount_total"]

# 3.分母选择
denom_amount = moneyflow["mf_trade_amount_total"]

# 4.比例指标
moneyflow["mf_net_amount_ratio"] = div(moneyflow["mf_net_amount"], denom_amount)
moneyflow["mf_sm_net_ratio"] = div(moneyflow["mf_net_sm_amount"], denom_amount)
moneyflow["mf_md_net_ratio"] = div(moneyflow["mf_net_md_amount"], denom_amount)
moneyflow["mf_lg_net_ratio"] = div(moneyflow["mf_net_lg_amount"], denom_amount)
moneyflow["mf_elg_net_ratio"] = div(moneyflow["mf_net_elg_amount"], denom_amount)

moneyflow["mf_large_net_ratio"] = div(
    moneyflow["mf_net_lg_amount"] + moneyflow["mf_net_elg_amount"],
    denom_amount
)
moneyflow["mf_md_large_net_ratio"] = div(
    moneyflow["mf_net_md_amount"] + moneyflow["mf_net_lg_amount"] + moneyflow["mf_net_elg_amount"],
    denom_amount
)

moneyflow["mf_large_vs_small_net"] = moneyflow["mf_large_net_ratio"] - moneyflow["mf_sm_net_ratio"]

moneyflow["mf_active_buy_ratio"] = div(moneyflow["mf_buy_amount_total"], moneyflow["mf_trade_amount_total"])
moneyflow["mf_active_sell_ratio"] = div(moneyflow["mf_sell_amount_total"], moneyflow["mf_trade_amount_total"])
moneyflow["mf_active_imbalance"] = div(
    moneyflow["mf_buy_amount_total"],
    moneyflow["mf_sell_amount_total"],
)

moneyflow["mf_large_participation"] = div(
    moneyflow["buy_lg_amount"] + moneyflow["sell_lg_amount"] + moneyflow["buy_elg_amount"] + moneyflow["sell_elg_amount"],
    moneyflow["mf_trade_amount_total"],
)
moneyflow["mf_small_participation"] = div(
    moneyflow["buy_sm_amount"] + moneyflow["sell_sm_amount"],
    moneyflow["mf_trade_amount_total"],
)

# S1/S2/S3标准化
for prefix, net_col, buy_col, sell_col in [
    ("sm", "mf_net_sm_amount", "buy_sm_amount", "sell_sm_amount"),
    ("md", "mf_net_md_amount", "buy_md_amount", "sell_md_amount"),
    ("lg", "mf_net_lg_amount", "buy_lg_amount", "sell_lg_amount"),
    ("elg", "mf_net_elg_amount", "buy_elg_amount", "sell_elg_amount"),
]:
    moneyflow[f"mf_{prefix}_s1"] = div(moneyflow[net_col], denom_amount)
    moneyflow[f"mf_{prefix}_s2"] = div(moneyflow[net_col], moneyflow[buy_col])
    moneyflow[f"mf_{prefix}_s3"] = div(moneyflow[net_col], moneyflow[net_col].abs())

# 滚动窗口指标(5日、20日)
for w in [5, 20]:
    moneyflow[f"mf_net_amount_ratio_{w}d"] = div(
        moneyflow["mf_net_amount"].rolling(w).sum(),
        denom_amount.rolling(w).sum(),
    )
    moneyflow[f"mf_large_net_ratio_{w}d"] = div(
        (moneyflow["mf_net_lg_amount"] + moneyflow["mf_net_elg_amount"]).rolling(w).sum(),
        denom_amount.rolling(w).sum(),
    )
    moneyflow[f"mf_sm_net_ratio_{w}d"] = div(
        moneyflow["mf_net_sm_amount"].rolling(w).sum(),
        denom_amount.rolling(w).sum(),
    )
    moneyflow[f"mf_large_vs_small_net_{w}d"] = (
        moneyflow[f"mf_large_net_ratio_{w}d"] - moneyflow[f"mf_sm_net_ratio_{w}d"]
    )
    moneyflow[f"mf_large_positive_days_{w}d"] = (moneyflow["mf_large_net_ratio"] > 0).astype(float).rolling(w).mean()
    moneyflow[f"mf_active_imbalance_std_{w}d"] = moneyflow["mf_active_imbalance"].rolling(w).std()

moneyflow = moneyflow.replace([np.inf, -np.inf], np.nan)

# moneyflow.columns


import gc
from pathlib import Path
import pandas as pd
import numpy as np

EPS = 1e-12

# ======================盘口基础常量与工具函数======================
ORDERBOOK_FIELDS_MIN = [
    "BP1", "SP1",
    "BV1", "BV2", "BV3", "BV4", "BV5",
    "SV1", "SV2", "SV3", "SV4", "SV5",
]

def div(a, b):
    return a / b.replace(0, np.nan)

def month_ranges(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        nxt = cur + pd.offsets.MonthBegin(1)
        yield max(cur, start), min(nxt - pd.Timedelta(days=1), end)

def continuous_auction(df):
    t = df.index.get_level_values("Datetime").time
    morning = (
        (t >= pd.Timestamp("09:35").time())
        & (t <= pd.Timestamp("11:30").time())
    )
    afternoon = (
        (t >= pd.Timestamp("13:05").time())
        & (t <= pd.Timestamp("15:00").time())
    )
    return df[morning | afternoon]

def segment(df, start, end):
    t = df.index.get_level_values("Datetime").time
    return df[
        (t >= pd.Timestamp(start).time())
        & (t <= pd.Timestamp(end).time())
    ]

def optimize_memory(df):
    for c in df.columns:
        df[c] = df[c].astype("float32", copy=False)
    return df

# ======================单日期盘口日度因子加工======================
def build_orderbook_daily_nas(df5):
    x = continuous_auction(df5)
    x = optimize_memory(x.copy())

    bv = ["BV1", "BV2", "BV3", "BV4", "BV5"]
    sv = ["SV1", "SV2", "SV3", "SV4", "SV5"]

    x["ob_mid"] = ((x["BP1"] + x["SP1"]) / 2).astype("float32")
    x["ob_spread"] = div(x["SP1"] - x["BP1"], x["ob_mid"]).astype("float32")

    x["ob_bid_depth_1"] = x["BV1"]
    x["ob_ask_depth_1"] = x["SV1"]
    x["ob_bid_depth_3"] = x[bv[:3]].sum(axis=1).astype("float32")
    x["ob_ask_depth_3"] = x[sv[:3]].sum(axis=1).astype("float32")
    x["ob_bid_depth_5"] = x[bv].sum(axis=1).astype("float32")
    x["ob_ask_depth_5"] = x[sv].sum(axis=1).astype("float32")
    x["ob_total_depth_5"] = (x["ob_bid_depth_5"] + x["ob_ask_depth_5"]).astype("float32")

    x["ob_imbalance_1"] = div(
        x["ob_bid_depth_1"] - x["ob_ask_depth_1"],
        x["ob_bid_depth_1"] + x["ob_ask_depth_1"],
    ).astype("float32")

    x["ob_imbalance_3"] = div(
        x["ob_bid_depth_3"] - x["ob_ask_depth_3"],
        x["ob_bid_depth_3"] + x["ob_ask_depth_3"],
    ).astype("float32")

    x["ob_imbalance_5"] = div(
        x["ob_bid_depth_5"] - x["ob_ask_depth_5"],
        x["ob_total_depth_5"],
    ).astype("float32")

    x["ob_micro_price"] = div(
        x["SP1"] * x["BV1"] + x["BP1"] * x["SV1"],
        x["BV1"] + x["SV1"],
    ).astype("float32")

    x["ob_micro_price_dev"] = div(
        x["ob_micro_price"] - x["ob_mid"],
        x["ob_mid"],
    ).astype("float32")

    x["ob_bid_near_share"] = div(
        x["BV1"] + x["BV2"],
        x["ob_bid_depth_5"],
    ).astype("float32")

    x["ob_ask_near_share"] = div(
        x["SV1"] + x["SV2"],
        x["ob_ask_depth_5"],
    ).astype("float32")

    x["ob_book_shape_gap"] = (
        x["ob_bid_near_share"] - x["ob_ask_near_share"]
    ).astype("float32")

    date = x.index.get_level_values("Datetime").normalize()
    contract = x.index.get_level_values("Contract")
    x["_date"] = date
    x["_contract"] = contract

    g = x.groupby(["_date", "_contract"], sort=False)
    x["ob_net_bid_change_1"] = (
        g["ob_bid_depth_1"].diff() - g["ob_ask_depth_1"].diff()
    ).astype("float32")
    x["ob_net_bid_change_5"] = (
        g["ob_bid_depth_5"].diff() - g["ob_ask_depth_5"].diff()
    ).astype("float32")
    x["ob_net_bid_change_5_ratio"] = div(
        x["ob_net_bid_change_5"],
        x["ob_total_depth_5"],
    ).astype("float32")

    agg_cols = [
        "ob_spread",
        "ob_bid_depth_5",
        "ob_ask_depth_5",
        "ob_total_depth_5",
        "ob_imbalance_1",
        "ob_imbalance_3",
        "ob_imbalance_5",
        "ob_micro_price_dev",
        "ob_book_shape_gap",
        "ob_net_bid_change_1",
        "ob_net_bid_change_5",
        "ob_net_bid_change_5_ratio",
    ]

    segments = {
        "full": x,
        "open30": segment(x, "09:35", "10:00"),
        "amclose30": segment(x, "11:05", "11:30"),
        "pmopen30": segment(x, "13:05", "13:30"),
        "close30": segment(x, "14:35", "15:00"),
    }

    pieces = []
    for name, data in segments.items():
        gg = data.groupby(["_date", "_contract"], sort=False)[agg_cols]
        part = pd.concat([
            gg.mean().add_suffix(f"_{name}_mean"),
            gg.std().add_suffix(f"_{name}_std"),
            gg.skew().add_suffix(f"_{name}_skew"),
            gg.last().add_suffix(f"_{name}_last"),
            gg.max().add_suffix(f"_{name}_max"),
            gg.min().add_suffix(f"_{name}_min"),
        ], axis=1)
        pieces.append(part)

    out = pd.concat(pieces, axis=1)

    out["ob_open_buy_intent"] = (
        out["ob_net_bid_change_5_open30_mean"]
        + out["ob_imbalance_5_open30_mean"]
        - out["ob_spread_open30_mean"]
    )

    out["ob_close_chase_pressure"] = (
        out["ob_imbalance_5_close30_mean"]
        + out["ob_micro_price_dev_close30_mean"]
        + out["ob_net_bid_change_5_close30_mean"]
    )

    out["ob_liquidity_stress"] = (
        out["ob_spread_full_max"]
        + out["ob_spread_full_std"]
        + out["ob_spread_full_std"]
    )

    out["ob_close_buying_vs_day"] = (
        out["ob_imbalance_5_close30_mean"]
        - out["ob_imbalance_5_full_mean"]
    )

    out.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(out.index.get_level_values(0)) + pd.Timedelta("15:00:00"),
        out.index.get_level_values(1),
    ], names=["Datetime", "Contract"])

    out = out.replace([np.inf, -np.inf], np.nan).astype("float32")
    del x, pieces
    gc.collect()
    return out

def ymd(d):
    return pd.Timestamp(d).strftime("%Y%m%d")

# ======================按月批量从nas读取盘口生成日因子======================
def build_orderbook_daily_from_nas(
    nas_storage,
    start_date,
    end_date,
    out_dir,
    table="price",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for start, end in month_ranges(start_date, end_date):
        out_path = out_dir / f"orderbook_daily_{start:%Y_%m}.parquet"
        if out_path.exists():
            print(f"skip {out_path.name}")
            continue
        print(f"load {start.date()} -> {end.date()}")

        df5 = nas_storage.load(
            table,
            ORDERBOOK_FIELDS_MIN,
            start_date=ymd(start),
            end_date=ymd(end),
            freq="5min",
        )
        if len(df5) == 0:
            print("empty")
            continue
        df5 = optimize_memory(df5)
        daily = build_orderbook_daily_nas(df5)
        daily.to_parquet(out_path)
        print(f"saved {out_path}")
        del df5, daily
        gc.collect()

# 任务执行入口
# build_orderbook_daily_from_nas(
#     nas_storage=nas_storage,
#     start_date="2016-01-01",
#     end_date="2024-01-01",
#     out_dir=r"/home/mw/work/gplearn/raw_data/orderbook_daily_features",
# )

# ======================盘口结果合并精选字段配置&加载函数======================
## 筛选最终保留字段
ORDERBOOK_KEEP_COLS = [
    "ob_spread_full_mean",
    "ob_spread_full_std",
    "ob_spread_full_max",
    "ob_total_depth_5_full_mean",
    "ob_total_depth_5_full_std",

    "ob_imbalance_1_full_mean",
    "ob_imbalance_5_full_mean",
    "ob_imbalance_5_full_std",
    "ob_micro_price_dev_full_mean",
    "ob_book_shape_gap_full_mean",

    "ob_spread_open30_mean",
    "ob_imbalance_5_open30_mean",
    "ob_micro_price_dev_open30_mean",
    "ob_net_bid_change_5_open30_mean",
    "ob_net_bid_change_5_open30_std",
    "ob_net_bid_change_5_open30_skew",

    "ob_imbalance_5_pmopen30_mean",
    "ob_net_bid_change_5_pmopen30_mean",
    "ob_net_bid_change_5_pmopen30_std",

    "ob_spread_close30_mean",
    "ob_imbalance_5_close30_mean",
    "ob_imbalance_5_close30_last",
    "ob_micro_price_dev_close30_mean",
    "ob_net_bid_change_5_close30_mean",
    "ob_net_bid_change_5_close30_std",

    "ob_net_bid_change_5_full_mean",
    "ob_net_bid_change_5_full_std",
    "ob_net_bid_change_5_ratio_full_mean",

    "ob_open_buy_intent",
    "ob_close_chase_pressure",
    "ob_liquidity_stress",
    "ob_close_buying_vs_day",
]

def load_orderbook(
    folder,
    pattern="orderbook_daily_*.parquet",
    output_path=None,
):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    dfs = []
    for file in files:
        print(f"load {file.name}")
        df = pd.read_parquet(file)
        use_cols = [c for c in ORDERBOOK_KEEP_COLS if c in df.columns]
        df = df[use_cols].astype("float32")
        dfs.append(df)
        del df
        gc.collect()
    out = pd.concat(dfs, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if output_path is not None:
        out.to_parquet(output_path)
    return out

# orderbook = load_orderbook(
#     folder=r"/home/mw/work/gplearn/raw_data/orderbook_daily_features",
#     output_path=r"/home/mw/work/gplearn/raw_data/orderbook_daily_35.parquet",
# )

# ======================全市场因子二次合成工具函数&Barra风格复合因子======================
# 通用工具
def div(a, b):
    return a / b.replace(0, np.nan)

def cs_rank(s):
    return s.groupby(level="Datetime").rank(pct=True) - 0.5

def cs_zscore(s):
    g = s.groupby(level="Datetime")
    return (s - g.transform("mean")) / (g.transform("std") + EPS)

def roll_mean(s, window):
    return (
        s.groupby(level="Contract")
        .rolling(window, min_periods=max(2, window // 3))
        .mean()
        .droplevel(0)
    )

def roll_std(s, window):
    return (
        s.groupby(level="Contract")
        .rolling(window, min_periods=max(2, window // 3))
        .std()
        .droplevel(0)
    )

# 注:pct_change_by_contract为自定义按合约涨跌幅函数,代码环境内置
# 1.价格行为指标
close = df["Close"]
amount = df["Amount"]
volume = df["Volume"]
turnover = df["Turnover"]

df["ret_1d"] = pct_change_by_contract(close, 1)
df["ret_5d"] = pct_change_by_contract(close, 5)
df["ret_20d"] = pct_change_by_contract(close, 20)

df["amount_shock_5_20"] = div(roll_mean(amount, 5), roll_mean(amount, 20)) - 1
df["volume_shock_5_20"] = div(roll_mean(volume, 5), roll_mean(volume, 20)) - 1
df["turnover_shock_5_20"] = div(roll_mean(turnover, 5), roll_mean(turnover, 20)) - 1

df["intraday_range"] = div(df["High"] - df["Low"], close)
df["range_std_20d"] = roll_std(df["intraday_range"], 20)

# 短期超买过热
df["short_term_overheat"] = (
    cs_zscore(df["amount_shock_5_20"])
    + cs_zscore(df["volume_shock_5_20"])
    + cs_zscore(df["ret_1d"].abs())
    + cs_zscore(df["turnover_shock_5_20"])
)

# 2.跨模块复合因子(资金流+量价+盘口)
## 资金流-价格背离
if "mf_net_amount_ratio_5d" in df.columns and "ret_5d" in df.columns:
    df["flow_price_divergence_5d"] = cs_rank(df["mf_net_amount_ratio_5d"]) - cs_rank(df["ret_5d"])

## 机构累积
if "mf_large_net_ratio_5d" in df.columns and "ret_5d" in df.columns:
    df["institution_accumulation"] = cs_rank(df["mf_large_net_ratio_5d"]) - cs_rank(df["ret_5d"])

## 散户追高风险
if all(c in df.columns for c in ["ret_5d", "mf_sm_net_ratio_5d", "mf_large_net_ratio_5d"]):
    df["retail_chase_risk"] = (
        cs_rank(df["ret_5d"])
        + cs_rank(df["mf_sm_net_ratio_5d"])
        - cs_rank(df["mf_large_net_ratio_5d"])
    )

## 大单确认动量
if all(c in df.columns for c in ["ret_20d", "mf_large_net_ratio_20d"]):
    df["large_order_confirmed_momentum"] = cs_rank(df["ret_20d"]) * cs_rank(df["mf_large_net_ratio_20d"])

## 资金拥挤风险
if all(c in df.columns for c in ["ret_20d", "turnover_shock_5_20", "mf_large_net_ratio_20d"]):
    df["moneyflow_crowding_risk"] = (
        cs_rank(df["ret_20d"])
        + cs_rank(df["turnover_shock_5_20"])
        + cs_rank(df["mf_large_net_ratio_20d"])
    )

## 盘口恐慌反转
if all(c in df.columns for c in ["ret_5d", "ob_imbalance_5_close30_mean", "ob_net_bid_change_5_close30_mean"]):
    df["orderbook_panic_reversal"] = (
        cs_rank(-df["ret_5d"])
        + cs_rank(df["ob_imbalance_5_close30_mean"])
        + cs_rank(df["ob_net_bid_change_5_close30_mean"])
    )

## 盘口追高风险
if all(c in df.columns for c in ["ret_5d", "ob_close_chase_pressure", "mf_sm_net_ratio_5d"]):
    df["orderbook_chase_risk"] = (
        cs_rank(df["ret_5d"])
        + cs_rank(df["ob_close_chase_pressure"])
        + cs_rank(df["mf_sm_net_ratio_5d"])
    )

## 流动性忽视
if all(c in df.columns for c in ["ob_liquidity_stress", "turnover_shock_5_20"]):
    df["liquidity_neglect"] = cs_rank(df["ob_liquidity_stress"]) - cs_rank(df["turnover_shock_5_20"])

df = df.replace([np.inf, -np.inf], np.nan)
