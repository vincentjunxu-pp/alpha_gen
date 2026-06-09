from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_PARQUET = ROOT / "data" / "panels" / "mock_behavior_daily.parquet"
OUT_META = ROOT / "data" / "metadata" / "fixtures" / "mock_behavior_metadata.json"

N_CONTRACTS = 120
N_DAYS = 620
START_DATE = "2023-01-03"
RANDOM_SEED = 20260529

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


def _cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    result = frame.sub(mean, axis=0).div(std, axis=0)
    result.index.name = "Datetime"
    result.columns.name = "Contract"
    return result.astype("float32")


def _cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rank(axis=1, pct=True, method="average").sub(0.5)
    result.index.name = "Datetime"
    result.columns.name = "Contract"
    return result.astype("float32")


def _ts_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(5, window // 3)).mean()
    std = frame.rolling(window, min_periods=max(5, window // 3)).std().replace(0.0, np.nan)
    result = frame.sub(mean).div(std)
    result.index.name = "Datetime"
    result.columns.name = "Contract"
    return result.astype("float32")


def _safe_div(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    result = left.div(right.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    result.index.name = "Datetime"
    result.columns.name = "Contract"
    return result.astype("float32")


def _rolling_sum_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame, window: int) -> pd.DataFrame:
    return _safe_div(
        numerator.rolling(window, min_periods=max(3, window // 3)).sum(),
        denominator.rolling(window, min_periods=max(3, window // 3)).sum(),
    )


def _set_names(frame: pd.DataFrame) -> pd.DataFrame:
    frame.index.name = "Datetime"
    frame.columns.name = "Contract"
    return frame


def _contract_frame(values: np.ndarray, dates: pd.DatetimeIndex, contracts: list[str] | pd.Index) -> pd.DataFrame:
    row = np.asarray(values, dtype=float).reshape(1, -1)
    matrix = np.broadcast_to(row, (len(dates), len(contracts)))
    return _set_names(pd.DataFrame(matrix, index=dates, columns=contracts))


def _make_contracts() -> tuple[list[str], pd.Series]:
    industries = ["electronics", "computer", "communication", "media", "pharma", "machinery"]
    contracts = [f"BHF{i:04d}.SZ" if i % 2 else f"BHF{i:04d}.SH" for i in range(1, N_CONTRACTS + 1)]
    industry = pd.Series(
        [industries[i % len(industries)] for i in range(N_CONTRACTS)],
        index=pd.Index(contracts, name="Contract"),
        name="industry_code",
        dtype="category",
    )
    return contracts, industry


def _make_price_panel(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    contracts: list[str],
    industry: pd.Series,
    latent_quality: np.ndarray,
    latent_growth: np.ndarray,
    latent_sentiment: np.ndarray,
) -> dict[str, pd.DataFrame]:
    n_dates = len(dates)
    n_contracts = len(contracts)
    industry_names = np.array(sorted(industry.astype(str).unique()))
    industry_to_id = {name: i for i, name in enumerate(industry_names)}
    industry_id = np.array([industry_to_id[name] for name in industry.astype(str).to_numpy()])

    market_ret = rng.normal(0.0001, 0.009, n_dates)
    industry_ret = rng.normal(0.0, 0.006, (n_dates, len(industry_names)))
    idio_ret = rng.normal(0.0, 0.018, (n_dates, n_contracts))
    beta = rng.uniform(0.75, 1.35, n_contracts)
    style_drift = 0.00010 * latent_quality + 0.00008 * latent_growth - 0.00005 * latent_sentiment
    ret = beta[None, :] * market_ret[:, None] + industry_ret[:, industry_id] + idio_ret + style_drift[None, :]

    close = _set_names(pd.DataFrame(rng.uniform(8.0, 70.0, n_contracts)[None, :] * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=contracts))
    prev_close = close.shift(1).fillna(close.iloc[0])
    gap = pd.DataFrame(rng.normal(0.0, 0.006, close.shape), index=dates, columns=contracts)
    open_ = _set_names(prev_close * (1.0 + gap))
    intraday_amp = pd.DataFrame(np.abs(rng.normal(0.014, 0.006, close.shape)), index=dates, columns=contracts).clip(0.003, 0.08)
    high = _set_names(pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=dates, columns=contracts) * (1.0 + intraday_amp))
    low = _set_names(pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=dates, columns=contracts) * (1.0 - intraday_amp))

    float_shares = rng.lognormal(mean=1.20, sigma=0.45, size=n_contracts)
    turnover_base = rng.lognormal(mean=-4.25, sigma=0.45, size=(n_dates, n_contracts))
    attention_shock = np.abs(ret) * 12.0 + np.maximum(ret, 0.0) * latent_sentiment[None, :] * 3.0
    turnover = _set_names(pd.DataFrame(turnover_base * (1.0 + attention_shock), index=dates, columns=contracts).clip(0.001, 0.25))
    volume = _set_names(turnover.mul(float_shares * 1e8, axis=1))
    amount = _set_names(volume * close)
    market_cap_raw = _set_names(close.mul(float_shares, axis=1))
    market_cap = _set_names(np.log(market_cap_raw.clip(lower=1e-4)))

    return {
        "open": open_.astype("float32"),
        "high": high.astype("float32"),
        "low": low.astype("float32"),
        "close": close.astype("float32"),
        "volume": volume.astype("float32"),
        "amount": amount.astype("float32"),
        "turnover": turnover.astype("float32"),
        "market_cap": market_cap.astype("float32"),
        "market_cap_raw": market_cap_raw.astype("float32"),
        "beta": pd.Series(beta, index=contracts),
        "market_ret": pd.Series(market_ret, index=dates),
    }


def _make_fundamentals(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    contracts: list[str],
    latent_quality: np.ndarray,
    latent_growth: np.ndarray,
    latent_value: np.ndarray,
) -> dict[str, pd.DataFrame]:
    n_dates = len(dates)
    n_contracts = len(contracts)
    slow_cycle = pd.Series(np.sin(np.arange(n_dates) / 38.0), index=dates).rolling(20, min_periods=1).mean().to_numpy()
    noise = lambda scale: pd.DataFrame(rng.normal(0.0, scale, (n_dates, n_contracts)), index=dates, columns=contracts).rolling(20, min_periods=1).mean()

    revenue_growth = _set_names(pd.DataFrame(0.06 + 0.12 * latent_growth[None, :] + 0.02 * slow_cycle[:, None], index=dates, columns=contracts) + noise(0.05))
    gross_growth = _set_names(revenue_growth + _contract_frame(0.03 * latent_quality, dates, contracts) + noise(0.04))
    op_growth = _set_names(gross_growth + noise(0.05))
    profit_growth = _set_names(op_growth + _contract_frame(0.02 * latent_quality, dates, contracts) + noise(0.07))
    ocf_growth = _set_names(
        profit_growth + _contract_frame(0.09 * latent_quality - 0.05 * latent_growth, dates, contracts) + noise(0.08)
    )
    asset_growth = _set_names(_contract_frame(0.05 + 0.08 * latent_growth, dates, contracts) + noise(0.04))

    rev_accel = _set_names(revenue_growth - revenue_growth.rolling(120, min_periods=20).mean())
    profit_accel = _set_names(profit_growth - profit_growth.rolling(120, min_periods=20).mean())
    ocf_accel = _set_names(ocf_growth - ocf_growth.rolling(120, min_periods=20).mean())

    growth_rank = _set_names((_cs_rank(revenue_growth) + _cs_rank(profit_growth) + _cs_rank(ocf_growth)) / 3.0)
    quality_gap = _set_names(ocf_growth - profit_growth)
    gross_gap = _set_names(gross_growth - revenue_growth)
    operating_gap = _set_names(op_growth - revenue_growth)
    leverage_adjusted_roe = _set_names(_contract_frame(0.09 + 0.05 * latent_quality - 0.03 * latent_growth, dates, contracts) + noise(0.025))
    cashflow_debt_coverage = _set_names(_contract_frame(0.55 + 0.20 * latent_quality - 0.10 * latent_growth, dates, contracts) + noise(0.08))
    core_profit_quality = _set_names(_contract_frame(0.60 + 0.22 * latent_quality, dates, contracts) + noise(0.08))

    ep_ratio = _set_names(_contract_frame(0.04 + 0.025 * latent_value + 0.015 * latent_quality, dates, contracts) + noise(0.012))
    bm_ratio = _set_names(_contract_frame(0.45 + 0.16 * latent_value - 0.05 * latent_growth, dates, contracts) + noise(0.08))
    value_composite = _set_names((_cs_rank(ep_ratio) + _cs_rank(bm_ratio)) / 2.0)

    return {
        "operating_revenue_growth_ratio_ttm": revenue_growth.astype("float32"),
        "gross_profit_growth_ratio_ttm": gross_growth.astype("float32"),
        "operating_profit_growth_ratio_ttm": op_growth.astype("float32"),
        "net_profit_growth_ratio_ttm": profit_growth.astype("float32"),
        "net_operate_cash_flow_growth_ratio_ttm": ocf_growth.astype("float32"),
        "total_asset_growth_ratio_ttm": asset_growth.astype("float32"),
        "rev_growth_accel": rev_accel.astype("float32"),
        "net_profit_growth_accel": profit_accel.astype("float32"),
        "ocf_growth_accel": ocf_accel.astype("float32"),
        "balanced_growth_rank_mean": growth_rank.astype("float32"),
        "ocf_vs_net_profit_growth_gap": quality_gap.astype("float32"),
        "gross_vs_revenue_growth_gap": gross_gap.astype("float32"),
        "operating_profit_vs_revenue_growth_gap": operating_gap.astype("float32"),
        "roe_leverage_adjusted": leverage_adjusted_roe.astype("float32"),
        "cashflow_debt_coverage": cashflow_debt_coverage.astype("float32"),
        "core_profit_quality": core_profit_quality.astype("float32"),
        "ep_ratio_ttm": ep_ratio.astype("float32"),
        "book_to_market_ratio_ttm": bm_ratio.astype("float32"),
        "value_composite_rank": value_composite.astype("float32"),
    }


def _make_price_volume_features(price: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = price["close"]
    high = price["high"]
    low = price["low"]
    volume = price["volume"]
    amount = price["amount"]
    turnover = price["turnover"]
    daily_ret = close.pct_change(fill_method=None)

    ret_5d = close.pct_change(5, fill_method=None)
    ret_20d = close.pct_change(20, fill_method=None)
    ret_60d = close.pct_change(60, fill_method=None)
    ret_120d = close.pct_change(120, fill_method=None)
    skip_momentum = close.shift(20).div(close.shift(80)).sub(1.0)
    vwap_60 = _safe_div(amount.rolling(60, min_periods=20).sum(), volume.rolling(60, min_periods=20).sum())

    rolling_high = high.rolling(252, min_periods=60).max()
    drawdown_20 = close.div(close.rolling(20, min_periods=5).max()).sub(1.0)
    vol_20 = daily_ret.rolling(20, min_periods=5).std()
    vol_60 = daily_ret.rolling(60, min_periods=20).std()
    max_ret_20 = daily_ret.rolling(20, min_periods=5).max()

    volume_crowding_5_20 = _safe_div(volume.rolling(5, min_periods=3).mean(), volume.rolling(20, min_periods=5).mean()).sub(1.0)
    amount_crowding_20_60 = _safe_div(amount.rolling(20, min_periods=5).mean(), amount.rolling(60, min_periods=20).mean()).sub(1.0)
    turnover_shock_20 = _ts_zscore(turnover, 20)
    price_volume_div = _set_names(_cs_rank(ret_20d) - _cs_rank(volume_crowding_5_20))
    overbought_consensus = _set_names((_cs_rank(ret_20d) + _cs_rank(max_ret_20) + _cs_rank(volume_crowding_5_20)) / 3.0)
    panic_oversold = _set_names(-_cs_rank(drawdown_20) + _cs_rank(vol_20))
    crowded_momentum = _set_names(_cs_rank(ret_20d) * _cs_rank(volume_crowding_5_20) * _cs_rank(overbought_consensus))

    return {
        "ret_5d": ret_5d.astype("float32"),
        "ret_20d": ret_20d.astype("float32"),
        "ret_60d": ret_60d.astype("float32"),
        "ret_120d": ret_120d.astype("float32"),
        "skip_momentum_60_20": skip_momentum.astype("float32"),
        "volatility_20d": vol_20.astype("float32"),
        "volatility_60d": vol_60.astype("float32"),
        "drawdown_20d": drawdown_20.astype("float32"),
        "max_ret_20d": max_ret_20.astype("float32"),
        "close_to_high_252d": _safe_div(close, rolling_high).sub(1.0).astype("float32"),
        "close_to_vwap_60d": _safe_div(close, vwap_60).sub(1.0).astype("float32"),
        "volume_crowding_5_20": volume_crowding_5_20.astype("float32"),
        "amount_crowding_20_60": amount_crowding_20_60.astype("float32"),
        "turnover_shock_20d": turnover_shock_20.astype("float32"),
        "price_volume_divergence_20_60": price_volume_div.astype("float32"),
        "overbought_consensus": overbought_consensus.astype("float32"),
        "panic_oversold_score": panic_oversold.astype("float32"),
        "crowded_momentum_risk": crowded_momentum.astype("float32"),
    }


def _make_orderbook_features(
    rng: np.random.Generator,
    price_volume: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    price: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    dates = price["close"].index
    contracts = price["close"].columns
    shape = price["close"].shape

    ret_rank = _cs_rank(price_volume["ret_5d"]).fillna(0.0)
    quality_rank = _cs_rank(fundamentals["ocf_vs_net_profit_growth_gap"]).fillna(0.0)
    crowding_rank = _cs_rank(price_volume["volume_crowding_5_20"]).fillna(0.0)
    liquidity_base = pd.DataFrame(rng.lognormal(mean=-4.8, sigma=0.35, size=shape), index=dates, columns=contracts)

    spread = _set_names(liquidity_base * (1.0 + 0.9 * price_volume["volatility_20d"].fillna(0.0).abs()))
    spread_std = _set_names(spread.rolling(20, min_periods=5).std())
    total_depth = _set_names(pd.DataFrame(rng.lognormal(mean=11.0, sigma=0.55, size=shape), index=dates, columns=contracts) * (1.0 + 0.5 * crowding_rank))
    bid_depth = _set_names(total_depth * (0.50 + 0.06 * quality_rank + 0.04 * ret_rank + rng.normal(0.0, 0.035, shape)))
    ask_depth = _set_names(total_depth - bid_depth)

    imbalance_full = _set_names(_safe_div(bid_depth - ask_depth, bid_depth + ask_depth).clip(-1.0, 1.0))
    imbalance_open = _set_names((imbalance_full + 0.20 * quality_rank + rng.normal(0.0, 0.05, shape)).clip(-1.0, 1.0))
    imbalance_close = _set_names((imbalance_full + 0.25 * ret_rank + 0.15 * crowding_rank + rng.normal(0.0, 0.06, shape)).clip(-1.0, 1.0))
    micro_close = _set_names(imbalance_close * spread * 2.0 + rng.normal(0.0, 0.0005, shape))
    net_bid_open = _set_names(imbalance_open.diff().fillna(0.0) + rng.normal(0.0, 0.035, shape))
    net_bid_close = _set_names(imbalance_close.diff().fillna(0.0) + rng.normal(0.0, 0.04, shape))
    open_intent = _set_names(_cs_rank(imbalance_open) + _cs_rank(net_bid_open) - _cs_rank(spread))
    close_chase = _set_names(_cs_rank(imbalance_close) + _cs_rank(micro_close) + _cs_rank(net_bid_close))
    liquidity_stress = _set_names(_cs_rank(spread) + _cs_rank(spread_std) - _cs_rank(total_depth))
    close_vs_day = _set_names(imbalance_close - imbalance_full)

    return {
        "ob_spread_full_mean": spread.astype("float32"),
        "ob_spread_full_std": spread_std.astype("float32"),
        "ob_bid_depth_5_full_mean": bid_depth.astype("float32"),
        "ob_ask_depth_5_full_mean": ask_depth.astype("float32"),
        "ob_total_depth_5_full_mean": total_depth.astype("float32"),
        "ob_imbalance_5_full_mean": imbalance_full.astype("float32"),
        "ob_imbalance_5_open30_mean": imbalance_open.astype("float32"),
        "ob_imbalance_5_close30_mean": imbalance_close.astype("float32"),
        "ob_micro_price_dev_close30_mean": micro_close.astype("float32"),
        "ob_net_bid_change_5_open30_mean": net_bid_open.astype("float32"),
        "ob_net_bid_change_5_close30_mean": net_bid_close.astype("float32"),
        "ob_open_buy_intent": open_intent.astype("float32"),
        "ob_close_chase_pressure": close_chase.astype("float32"),
        "ob_liquidity_stress": liquidity_stress.astype("float32"),
        "ob_close_buying_vs_day": close_vs_day.astype("float32"),
    }


def _make_moneyflow_features(
    rng: np.random.Generator,
    price: dict[str, pd.DataFrame],
    price_volume: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    orderbook: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    dates = price["close"].index
    contracts = price["close"].columns
    shape = price["close"].shape
    amount = price["amount"]

    ret_rank = _cs_rank(price_volume["ret_20d"]).fillna(0.0)
    growth_rank = _cs_rank(fundamentals["balanced_growth_rank_mean"]).fillna(0.0)
    quality_rank = _cs_rank(fundamentals["ocf_vs_net_profit_growth_gap"]).fillna(0.0)
    chase_rank = _cs_rank(orderbook["ob_close_chase_pressure"]).fillna(0.0)

    large_ratio = _set_names((0.10 * growth_rank + 0.08 * quality_rank + 0.04 * ret_rank + rng.normal(0.0, 0.08, shape)).clip(-0.35, 0.35))
    small_ratio = _set_names((0.12 * ret_rank + 0.10 * chase_rank - 0.06 * quality_rank + rng.normal(0.0, 0.09, shape)).clip(-0.40, 0.40))
    mid_ratio = _set_names((0.5 * large_ratio + rng.normal(0.0, 0.06, shape)).clip(-0.30, 0.30))
    net_ratio = _set_names((0.45 * large_ratio + 0.30 * mid_ratio + 0.25 * small_ratio).clip(-0.35, 0.35))

    large_net_amount = _set_names(large_ratio * amount)
    small_net_amount = _set_names(small_ratio * amount)
    mid_net_amount = _set_names(mid_ratio * amount)
    active_imbalance = _set_names((net_ratio + rng.normal(0.0, 0.06, shape)).clip(-0.6, 0.6))
    large_participation = _set_names((0.22 + 0.10 * _cs_rank(amount).fillna(0.0) + rng.normal(0.0, 0.04, shape)).clip(0.05, 0.65))
    small_participation = _set_names((0.40 - 0.10 * _cs_rank(amount).fillna(0.0) + rng.normal(0.0, 0.05, shape)).clip(0.08, 0.75))

    large_ratio_20d = _rolling_sum_ratio(large_net_amount, amount, 20)
    small_ratio_20d = _rolling_sum_ratio(small_net_amount, amount, 20)
    net_ratio_5d = _rolling_sum_ratio(net_ratio * amount, amount, 5)
    large_vs_small_20d = _set_names(large_ratio_20d - small_ratio_20d)
    large_positive_days_20d = _set_names(large_ratio.gt(0).rolling(20, min_periods=5).mean())
    flow_price_divergence = _set_names(_cs_rank(net_ratio_5d) - _cs_rank(price_volume["ret_5d"]))
    inst_accumulation = _set_names(_cs_rank(large_ratio_20d) - _cs_rank(price_volume["ret_20d"]))
    retail_chase = _set_names(_cs_rank(price_volume["ret_20d"]) + _cs_rank(small_ratio_20d) - _cs_rank(large_ratio_20d))
    large_confirmed_mom = _set_names(_cs_rank(price_volume["ret_60d"]) * _cs_rank(large_ratio_20d))
    flow_crowding = _set_names(_cs_rank(price_volume["ret_20d"]) * _cs_rank(price_volume["turnover_shock_20d"]) * _cs_rank(large_ratio_20d))

    return {
        "mf_net_sm_amount": small_net_amount.astype("float32"),
        "mf_net_md_amount": mid_net_amount.astype("float32"),
        "mf_net_lg_amount": large_net_amount.astype("float32"),
        "mf_net_amount_ratio": net_ratio.astype("float32"),
        "mf_sm_net_ratio": small_ratio.astype("float32"),
        "mf_md_net_ratio": mid_ratio.astype("float32"),
        "mf_large_net_ratio": large_ratio.astype("float32"),
        "mf_mid_large_net_ratio": _set_names((mid_ratio + large_ratio) / 2.0).astype("float32"),
        "mf_large_vs_small_net": _set_names(large_ratio - small_ratio).astype("float32"),
        "mf_active_imbalance": active_imbalance.astype("float32"),
        "mf_large_participation": large_participation.astype("float32"),
        "mf_small_participation": small_participation.astype("float32"),
        "mf_lg_s3": _safe_div(large_net_amount, large_net_amount.abs().rolling(20, min_periods=5).sum()).astype("float32"),
        "mf_sm_s3": _safe_div(small_net_amount, small_net_amount.abs().rolling(20, min_periods=5).sum()).astype("float32"),
        "mf_net_amount_ratio_5d": net_ratio_5d.astype("float32"),
        "mf_large_net_ratio_20d": large_ratio_20d.astype("float32"),
        "mf_sm_net_ratio_20d": small_ratio_20d.astype("float32"),
        "mf_large_vs_small_net_20d": large_vs_small_20d.astype("float32"),
        "mf_large_positive_days_20d": large_positive_days_20d.astype("float32"),
        "flow_price_divergence_5d": flow_price_divergence.astype("float32"),
        "institution_accumulation": inst_accumulation.astype("float32"),
        "retail_chase_risk": retail_chase.astype("float32"),
        "large_order_confirmed_momentum": large_confirmed_mom.astype("float32"),
        "moneyflow_crowding_risk": flow_crowding.astype("float32"),
    }


def _make_barra_fields(
    price: dict[str, pd.DataFrame],
    price_volume: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    close = price["close"]
    market_cap = price["market_cap"]
    daily_ret = close.pct_change(fill_method=None).fillna(0.0)
    beta_frame = _contract_frame(price["beta"].to_numpy(), close.index, close.columns)
    beta_frame = beta_frame + daily_ret.rolling(60, min_periods=20).corr(price["market_ret"]).fillna(0.0) * 0.10

    barra_size = _cs_zscore(market_cap).fillna(0.0)
    barra_beta = _cs_zscore(beta_frame).fillna(0.0)
    barra_momentum = _cs_zscore(price_volume["ret_120d"].sub(price_volume["ret_20d"], fill_value=0.0)).fillna(0.0)
    barra_residual_volatility = _cs_zscore(price_volume["volatility_60d"]).fillna(0.0)
    barra_non_linear_size = _cs_zscore(barra_size.pow(3)).fillna(0.0)
    barra_book_to_price = _cs_zscore(fundamentals["book_to_market_ratio_ttm"]).fillna(0.0)
    barra_liquidity = _cs_zscore(price_volume["turnover_shock_20d"]).fillna(0.0)
    barra_earnings_yield = _cs_zscore(fundamentals["ep_ratio_ttm"]).fillna(0.0)
    barra_growth = _cs_zscore(fundamentals["balanced_growth_rank_mean"]).fillna(0.0)
    barra_leverage = _cs_zscore(-fundamentals["cashflow_debt_coverage"]).fillna(0.0)

    return {
        "barra_size": barra_size.astype("float32"),
        "barra_beta": barra_beta.astype("float32"),
        "barra_momentum": barra_momentum.astype("float32"),
        "barra_residual_volatility": barra_residual_volatility.astype("float32"),
        "barra_non_linear_size": barra_non_linear_size.astype("float32"),
        "barra_book_to_price": barra_book_to_price.astype("float32"),
        "barra_liquidity": barra_liquidity.astype("float32"),
        "barra_earnings_yield": barra_earnings_yield.astype("float32"),
        "barra_growth": barra_growth.astype("float32"),
        "barra_leverage": barra_leverage.astype("float32"),
    }


def _make_labels(
    rng: np.random.Generator,
    fundamentals: dict[str, pd.DataFrame],
    price_volume: dict[str, pd.DataFrame],
    orderbook: dict[str, pd.DataFrame],
    moneyflow: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    index = fundamentals["balanced_growth_rank_mean"].index
    columns = fundamentals["balanced_growth_rank_mean"].columns
    shape = fundamentals["balanced_growth_rank_mean"].shape
    underreaction = _cs_rank(fundamentals["balanced_growth_rank_mean"]) - _cs_rank(price_volume["ret_20d"])
    quality = _cs_rank(fundamentals["ocf_vs_net_profit_growth_gap"])
    inst_confirm = _cs_rank(moneyflow["mf_large_net_ratio_20d"]) + _cs_rank(orderbook["ob_open_buy_intent"])
    retail_risk = _cs_rank(moneyflow["retail_chase_risk"]) + _cs_rank(orderbook["ob_close_chase_pressure"])
    panic_rebound = _cs_rank(fundamentals["core_profit_quality"]) * _cs_rank(price_volume["panic_oversold_score"])
    crowding_risk = _cs_rank(price_volume["crowded_momentum_risk"]) + _cs_rank(moneyflow["moneyflow_crowding_risk"])
    alpha = _set_names(
        0.020 * underreaction
        + 0.018 * quality
        + 0.015 * inst_confirm
        + 0.014 * panic_rebound
        - 0.018 * retail_risk
        - 0.014 * crowding_risk
    ).fillna(0.0)
    noise_1d = pd.DataFrame(rng.normal(0.0, 0.025, shape), index=index, columns=columns)
    noise_5d = pd.DataFrame(rng.normal(0.0, 0.045, shape), index=index, columns=columns)
    noise_20d = pd.DataFrame(rng.normal(0.0, 0.085, shape), index=index, columns=columns)
    label_1d = _set_names(alpha * 0.25 + noise_1d)
    label_5d = _set_names(alpha * 0.70 + noise_5d)
    label_20d = _set_names(alpha * 1.40 + noise_20d)
    return {
        "label_1d": label_1d.shift(-1).astype("float32"),
        "label_5d": label_5d.shift(-5).astype("float32"),
        "label_20d": label_20d.shift(-20).astype("float32"),
        "mock_behavior_alpha": alpha.astype("float32"),
    }


def _core_rule(data_family: str, unit_type: str, direction: int) -> dict[str, object]:
    return {
        "can_y": data_family not in {"control"},
        "can_x": True,
        "allow_log": unit_type in {"currency", "price"},
        "allow_current": True,
        "allow_lag": False,
        "allow_diff": True,
        "allow_pct": unit_type in {"price", "currency", "ratio", "rate", "growth"},
        "allow_std": False,
        "family": data_family,
        "unit_type": unit_type,
        "statement": "behavior",
        "period_type": "daily",
        "direction": direction,
        "add_group": data_family,
        "allow_industry_relative": True,
    }


def _behavior_rule(
    *,
    data_family: str,
    roles: list[str],
    sub_family: str,
    unit_type: str = "score",
    window: str = "daily",
    session: str = "full",
    investor_type: str = "none",
    direction: int = 1,
    allowed_slots: list[str] | None = None,
    allowed_unary_ops: list[str] | None = None,
) -> dict[str, object]:
    return {
        "data_family": data_family,
        "behavior_roles": roles,
        "sub_family": sub_family,
        "unit_type": unit_type,
        "window": window,
        "session": session,
        "investor_type": investor_type,
        "direction": direction,
        "allowed_slots": allowed_slots or [],
        "allowed_unary_ops": allowed_unary_ops
        or ["current", "rank_pct", "zscore", "direction_rank", "direction_zscore", "ind_rank_pct", "ind_zscore"],
    }


def _build_behavior_metadata() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    behavior: dict[str, dict[str, object]] = {}

    def add(field: str, **kwargs: object) -> None:
        behavior[field] = _behavior_rule(**kwargs)  # type: ignore[arg-type]

    for field in [
        "operating_revenue_growth_ratio_ttm",
        "gross_profit_growth_ratio_ttm",
        "operating_profit_growth_ratio_ttm",
        "net_profit_growth_ratio_ttm",
        "total_asset_growth_ratio_ttm",
        "rev_growth_accel",
        "net_profit_growth_accel",
        "balanced_growth_rank_mean",
    ]:
        add(
            field,
            data_family="fundamental",
            roles=["anchor", "growth", "support"],
            sub_family="growth",
            unit_type="growth",
            window="ttm",
            direction=1,
            allowed_slots=["fund_anchor", "fund_support", "growth_anchor", "profit_growth"],
        )

    for field in [
        "net_operate_cash_flow_growth_ratio_ttm",
        "ocf_growth_accel",
        "ocf_vs_net_profit_growth_gap",
        "gross_vs_revenue_growth_gap",
        "operating_profit_vs_revenue_growth_gap",
        "roe_leverage_adjusted",
        "cashflow_debt_coverage",
        "core_profit_quality",
    ]:
        add(
            field,
            data_family="fundamental",
            roles=["anchor", "quality", "cashflow_quality", "support"],
            sub_family="quality",
            unit_type="score",
            window="ttm",
            direction=1,
            allowed_slots=["fund_anchor", "fund_support", "cashflow_quality", "quality_anchor"],
        )

    for field in ["ep_ratio_ttm", "book_to_market_ratio_ttm", "value_composite_rank"]:
        add(
            field,
            data_family="fundamental",
            roles=["valuation", "support", "anchor"],
            sub_family="valuation",
            unit_type="ratio",
            window="ttm",
            direction=1,
            allowed_slots=["fund_anchor", "fund_support", "valuation_support"],
        )

    price_role_map = {
        "ret_5d": ("reaction", "momentum", "5d"),
        "ret_20d": ("reaction", "momentum", "20d"),
        "ret_60d": ("reaction", "momentum", "60d"),
        "ret_120d": ("reaction", "momentum", "120d"),
        "skip_momentum_60_20": ("momentum", "anchor_momentum", "60d"),
        "drawdown_20d": ("panic", "drawdown", "20d"),
        "max_ret_20d": ("attention", "lottery", "20d"),
        "close_to_high_252d": ("anchor", "price_anchor", "252d"),
        "close_to_vwap_60d": ("anchor", "cost_anchor", "60d"),
        "volatility_20d": ("volatility", "uncertainty", "20d"),
        "volatility_60d": ("volatility", "uncertainty", "60d"),
        "volume_crowding_5_20": ("attention", "crowding", "20d"),
        "amount_crowding_20_60": ("attention", "crowding", "60d"),
        "turnover_shock_20d": ("crowding", "liquidity", "20d"),
        "price_volume_divergence_20_60": ("divergence", "reaction", "60d"),
        "overbought_consensus": ("attention", "overreaction", "20d"),
        "panic_oversold_score": ("panic", "oversold", "20d"),
        "crowded_momentum_risk": ("crowding", "overreaction", "20d"),
    }
    for field, (role_a, role_b, window) in price_role_map.items():
        allowed = ["price_reaction", "price_momentum", "attention_heat", "crowding_signal", "state_signal"]
        if role_a == "panic" or role_b == "drawdown":
            allowed.append("drawdown")
        if role_b in {"price_anchor", "cost_anchor"}:
            allowed.extend(["price_anchor", "cost_anchor"])
        if field == "turnover_shock_20d":
            allowed.append("turnover_shock")
        add(
            field,
            data_family="price_volume",
            roles=[role_a, role_b],
            sub_family=role_b,
            unit_type="return" if "ret" in field or "drawdown" in field else "score",
            window=window,
            direction=-1 if "risk" in field else 1,
            allowed_slots=allowed,
        )

    orderbook_specs = {
        "ob_spread_full_mean": (["liquidity", "spread", "stress"], "liquidity", "full", -1, ["liquidity_stress", "orderbook_filter"]),
        "ob_spread_full_std": (["liquidity", "instability", "stress"], "liquidity", "full", -1, ["liquidity_stress", "state_signal"]),
        "ob_imbalance_5_full_mean": (["orderbook_pressure", "buy_pressure"], "pressure", "full", 1, ["orderbook_pressure", "orderbook_filter"]),
        "ob_imbalance_5_open30_mean": (["orderbook_pressure", "open_intent"], "pressure", "open30", 1, ["orderbook_pressure", "orderbook_filter"]),
        "ob_imbalance_5_close30_mean": (["orderbook_pressure", "close_chase"], "pressure", "close30", -1, ["close_chase", "orderbook_pressure"]),
        "ob_net_bid_change_5_open30_mean": (["orderbook_pressure", "open_intent"], "net_bid_change", "open30", 1, ["orderbook_pressure", "orderbook_filter"]),
        "ob_net_bid_change_5_close30_mean": (["orderbook_pressure", "close_chase"], "net_bid_change", "close30", -1, ["close_chase", "orderbook_pressure"]),
        "ob_open_buy_intent": (["orderbook_pressure", "open_intent", "confirmation"], "composite", "open30", 1, ["orderbook_filter", "orderbook_pressure"]),
        "ob_close_chase_pressure": (["orderbook_pressure", "close_chase", "crowding"], "composite", "close30", -1, ["close_chase", "crowding_signal"]),
        "ob_liquidity_stress": (["liquidity", "stress"], "composite", "full", -1, ["liquidity_stress", "state_signal"]),
        "ob_close_buying_vs_day": (["orderbook_pressure", "close_chase"], "pressure", "close30", -1, ["close_chase", "orderbook_pressure"]),
    }
    for field, (roles, sub_family, session, direction, slots) in orderbook_specs.items():
        add(
            field,
            data_family="orderbook",
            roles=roles,
            sub_family=sub_family,
            unit_type="score",
            window="daily",
            session=session,
            direction=direction,
            allowed_slots=slots,
        )

    flow_specs = {
        "mf_net_amount_ratio": (["flow", "net_flow"], "all", "none", 1, ["flow_confirm", "state_signal"]),
        "mf_sm_net_ratio": (["flow", "retail", "small_flow"], "small", "small", -1, ["retail_flow", "sell_pressure"]),
        "mf_large_net_ratio": (["flow", "large_flow", "confirmation"], "large", "large", 1, ["flow_confirm", "large_flow"]),
        "mf_mid_large_net_ratio": (["flow", "large_flow", "confirmation"], "mid_large", "large", 1, ["flow_confirm", "large_flow"]),
        "mf_large_vs_small_net": (["flow", "divergence", "institution_vs_retail"], "divergence", "none", 1, ["flow_confirm", "large_flow"]),
        "mf_active_imbalance": (["flow", "active_imbalance"], "active", "none", 1, ["flow_confirm", "sell_pressure"]),
        "mf_lg_s3": (["flow", "large_flow", "stability"], "large", "large", 1, ["flow_confirm", "large_flow"]),
        "mf_sm_s3": (["flow", "retail", "stability"], "small", "small", -1, ["retail_flow", "sell_pressure"]),
        "mf_net_amount_ratio_5d": (["flow", "net_flow"], "all", "none", 1, ["flow_confirm", "state_signal"]),
        "mf_large_net_ratio_20d": (["flow", "large_flow", "confirmation"], "large", "large", 1, ["flow_confirm", "large_flow"]),
        "mf_sm_net_ratio_20d": (["flow", "retail", "small_flow"], "small", "small", -1, ["retail_flow", "sell_pressure"]),
        "mf_large_vs_small_net_20d": (["flow", "divergence", "institution_vs_retail"], "divergence", "none", 1, ["flow_confirm", "large_flow"]),
        "mf_large_positive_days_20d": (["flow", "large_flow", "persistence"], "large", "large", 1, ["flow_confirm", "large_flow"]),
        "flow_price_divergence_5d": (["flow", "divergence", "underreaction"], "divergence", "none", 1, ["flow_confirm", "state_signal"]),
        "institution_accumulation": (["flow", "large_flow", "underreaction"], "large", "large", 1, ["flow_confirm", "large_flow"]),
        "retail_chase_risk": (["flow", "retail", "chase", "risk"], "small", "small", -1, ["retail_flow", "crowding_signal"]),
        "large_order_confirmed_momentum": (["flow", "large_flow", "momentum_confirmation"], "large", "large", 1, ["flow_confirm", "large_flow"]),
        "moneyflow_crowding_risk": (["flow", "crowding", "risk"], "crowding", "none", -1, ["crowding_signal", "state_signal"]),
    }
    for field, (roles, sub_family, investor_type, direction, slots) in flow_specs.items():
        add(
            field,
            data_family="moneyflow",
            roles=roles,
            sub_family=sub_family,
            unit_type="ratio",
            window="20d" if field.endswith("20d") else "daily",
            investor_type=investor_type,
            direction=direction,
            allowed_slots=slots,
        )

    for field in BARRA_STYLE_FIELDS + ["market_cap"]:
        add(
            field,
            data_family="control",
            roles=["control"],
            sub_family="barra" if field.startswith("barra_") else "size",
            unit_type="score",
            direction=0,
            allowed_slots=["control"],
            allowed_unary_ops=["current", "zscore", "rank_pct"],
        )

    core = {
        field: _core_rule(str(rule["data_family"]), str(rule["unit_type"]), int(rule["direction"]))
        for field, rule in behavior.items()
        if str(rule["data_family"]) != "control" or field == "market_cap"
    }
    return core, behavior


def make_mock_behavior_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    contracts, industry = _make_contracts()
    dates = pd.bdate_range(START_DATE, periods=N_DAYS) + pd.Timedelta(hours=15)
    dates.name = "Datetime"

    latent_quality = rng.normal(0.0, 1.0, N_CONTRACTS)
    latent_growth = rng.normal(0.0, 1.0, N_CONTRACTS)
    latent_value = rng.normal(0.0, 1.0, N_CONTRACTS)
    latent_sentiment = rng.normal(0.0, 1.0, N_CONTRACTS)

    price = _make_price_panel(rng, dates, contracts, industry, latent_quality, latent_growth, latent_sentiment)
    fundamentals = _make_fundamentals(rng, dates, contracts, latent_quality, latent_growth, latent_value)
    price_volume = _make_price_volume_features(price)
    orderbook = _make_orderbook_features(rng, price_volume, fundamentals, price)
    moneyflow = _make_moneyflow_features(rng, price, price_volume, fundamentals, orderbook)
    barra = _make_barra_fields(price, price_volume, fundamentals)
    labels = _make_labels(rng, fundamentals, price_volume, orderbook, moneyflow)

    tradeable = pd.DataFrame((rng.random((len(dates), len(contracts))) > 0.025).astype("int8"), index=dates, columns=contracts)
    tradeable.index.name = "Datetime"
    tradeable.columns.name = "Contract"

    frames: dict[str, pd.DataFrame] = {
        "open": price["open"],
        "high": price["high"],
        "low": price["low"],
        "close": price["close"],
        "volume": price["volume"],
        "amount": price["amount"],
        "turnover": price["turnover"],
        "market_cap": price["market_cap"],
        **fundamentals,
        **price_volume,
        **orderbook,
        **moneyflow,
        **barra,
        "is_tradeable": tradeable,
        **labels,
    }

    long_parts = []
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame):
            part = frame.stack(future_stack=True).rename(name)
            part.index = part.index.set_names(["Datetime", "Contract"])
            long_parts.append(part)
    data = pd.concat(long_parts, axis=1).sort_index()
    industry_long = pd.Series(
        np.tile(industry.reindex(contracts).astype(str).to_numpy(), len(dates)),
        index=data.index,
        name="industry_code",
        dtype="category",
    )
    data["industry_code"] = industry_long
    float_cols = data.select_dtypes(include=["float64"]).columns
    data[float_cols] = data[float_cols].astype("float32")
    data["is_tradeable"] = data["is_tradeable"].fillna(0).astype("int8")
    return data


def save_outputs(data: pd.DataFrame) -> None:
    core_rules, behavior_rules = _build_behavior_metadata()
    try:
        data.to_parquet(OUT_PARQUET, engine="pyarrow", compression="zstd", index=True)
        compression = "zstd"
    except Exception:
        data.to_parquet(OUT_PARQUET, engine="pyarrow", compression="snappy", index=True)
        compression = "snappy"

    metadata = {
        "dataset": OUT_PARQUET.name,
        "format": "parquet",
        "compression": compression,
        "index": ["Datetime", "Contract"],
        "shape": list(data.shape),
        "n_dates": int(data.index.get_level_values("Datetime").nunique()),
        "n_contracts": int(data.index.get_level_values("Contract").nunique()),
        "start": str(data.index.get_level_values("Datetime").min()),
        "end": str(data.index.get_level_values("Datetime").max()),
        "notes": [
            "Mock daily behavior-finance panel for behavior_gen debugging.",
            "Datetime uses 15:00:00 and Contract as a long-table MultiIndex.",
            "Orderbook and moneyflow fields are already aggregated to daily frequency.",
            "label_1d, label_5d, and label_20d are synthetic forward-return labels with planted behavior alpha.",
            "field_rules are compatible with alpha_gen.core.load_field_rules; behavior_field_rules drive behavior_gen.",
        ],
        "label_fields": ["label_1d", "label_5d", "label_20d"],
        "tradeable_field": "is_tradeable",
        "industry_field": "industry_code",
        "size_field": "barra_size",
        "barra_style_fields": BARRA_STYLE_FIELDS,
        "field_rules": core_rules,
        "behavior_field_rules": behavior_rules,
    }
    OUT_META.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    df = make_mock_behavior_data()
    save_outputs(df)
    size_mb = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"saved: {OUT_PARQUET} ({size_mb:.2f} MB)")
    print(f"saved: {OUT_META}")
    print(df.head(8))
