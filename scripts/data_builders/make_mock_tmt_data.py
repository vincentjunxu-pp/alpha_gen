from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


ROOT = Path(__file__).resolve().parents[2]
OUT_PARQUET = ROOT / "data" / "panels" / "mock_tmt_daily.parquet"
OUT_META = ROOT / "data" / "metadata" / "fixtures" / "mock_tmt_metadata.json"

N_CONTRACTS = 80
N_DAYS = 600
START_DATE = "2023-01-03"
RANDOM_SEED = 20260428

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


FIELD_RULES = {
    # close is only used as x in the report, and only after time-series transforms.
    "close": {
        "can_y": False,
        "can_x": True,
        "allow_log": True,
        "allow_current": False,
        "allow_lag": False,
        "allow_diff": False,
        "allow_pct": True,
        "allow_std": False,
    },
    "market_cap": {
        "can_y": False,
        "can_x": True,
        # Stored as log market cap to match alpha_gen's prepared-size contract.
        "allow_log": False,
        "allow_current": True,
        "allow_lag": False,
        "allow_diff": True,
        "allow_pct": True,
        "allow_std": False,
    },
    "enterprise_value": {
        "can_y": False,
        "can_x": True,
        "allow_log": True,
        "allow_current": True,
        "allow_lag": False,
        "allow_diff": False,
        "allow_pct": False,
        "allow_std": False,
    },
    "rating_score_30d": {
        "can_y": True,
        "can_x": True,
        "allow_log": False,
        "allow_current": True,
        "allow_lag": True,
        "allow_diff": True,
        "allow_pct": False,
        "allow_std": False,
    },
    "rating_score_90d": {
        "can_y": True,
        "can_x": True,
        "allow_log": False,
        "allow_current": True,
        "allow_lag": True,
        "allow_diff": True,
        "allow_pct": False,
        "allow_std": False,
    },
    "rating_score_180d": {
        "can_y": True,
        "can_x": True,
        "allow_log": False,
        "allow_current": True,
        "allow_lag": True,
        "allow_diff": True,
        "allow_pct": False,
        "allow_std": False,
    },
}


FUNDAMENTAL_FIELDS = [
    "book_equity",
    "total_assets",
    "revenue_mrq",
    "revenue_ttm",
    "operating_profit_mrq",
    "operating_profit_ttm",
    "net_profit_mrq",
    "net_profit_ttm",
    "rd_expense_ttm",
    "capex_ttm",
    "free_cash_flow_ttm",
    "forecast_revenue_ry",
    "forecast_net_profit_ry",
]

for field in FUNDAMENTAL_FIELDS:
    FIELD_RULES[field] = {
        "can_y": True,
        "can_x": True,
        "allow_log": True,
        "allow_current": True,
        "allow_lag": True,
        "allow_diff": True,
        "allow_pct": True,
        "allow_std": False,
    }

FIELD_SEMANTICS = {
    "close": {
        "family": "price",
        "unit_type": "price",
        "statement": "market",
        "period_type": "daily",
        "direction": 1,
        "add_group": "price",
    },
    "market_cap": {
        "family": "size",
        "unit_type": "currency",
        "statement": "market",
        "period_type": "daily",
        "direction": -1,
        "add_group": "market_value",
    },
    "enterprise_value": {
        "family": "size",
        "unit_type": "currency",
        "statement": "market",
        "period_type": "daily",
        "direction": -1,
        "add_group": "market_value",
    },
    "rating_score_30d": {
        "family": "analyst",
        "unit_type": "score",
        "statement": "analyst",
        "period_type": "30d",
        "direction": 1,
        "add_group": "score",
    },
    "rating_score_90d": {
        "family": "analyst",
        "unit_type": "score",
        "statement": "analyst",
        "period_type": "90d",
        "direction": 1,
        "add_group": "score",
    },
    "rating_score_180d": {
        "family": "analyst",
        "unit_type": "score",
        "statement": "analyst",
        "period_type": "180d",
        "direction": 1,
        "add_group": "score",
    },
    "book_equity": {
        "family": "size",
        "unit_type": "currency",
        "statement": "balance_sheet",
        "period_type": "unknown",
        "direction": 1,
        "add_group": "equity",
    },
    "total_assets": {
        "family": "size",
        "unit_type": "currency",
        "statement": "balance_sheet",
        "period_type": "unknown",
        "direction": 1,
        "add_group": "asset",
    },
    "revenue_mrq": {
        "family": "profitability",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "mrq",
        "direction": 1,
        "add_group": "revenue",
    },
    "revenue_ttm": {
        "family": "profitability",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "ttm",
        "direction": 1,
        "add_group": "revenue",
    },
    "operating_profit_mrq": {
        "family": "profitability",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "mrq",
        "direction": 1,
        "add_group": "profit",
    },
    "operating_profit_ttm": {
        "family": "profitability",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "ttm",
        "direction": 1,
        "add_group": "profit",
    },
    "net_profit_mrq": {
        "family": "profitability",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "mrq",
        "direction": 1,
        "add_group": "profit",
    },
    "net_profit_ttm": {
        "family": "profitability",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "ttm",
        "direction": 1,
        "add_group": "profit",
    },
    "rd_expense_ttm": {
        "family": "quality",
        "unit_type": "currency",
        "statement": "income",
        "period_type": "ttm",
        "direction": -1,
        "add_group": "expense",
    },
    "capex_ttm": {
        "family": "quality",
        "unit_type": "currency",
        "statement": "cashflow",
        "period_type": "ttm",
        "direction": -1,
        "add_group": "expense",
    },
    "free_cash_flow_ttm": {
        "family": "cashflow",
        "unit_type": "currency",
        "statement": "cashflow",
        "period_type": "ttm",
        "direction": 1,
        "add_group": "cashflow",
    },
    "forecast_revenue_ry": {
        "family": "growth",
        "unit_type": "currency",
        "statement": "analyst",
        "period_type": "ry",
        "direction": 1,
        "add_group": "revenue",
    },
    "forecast_net_profit_ry": {
        "family": "growth",
        "unit_type": "currency",
        "statement": "analyst",
        "period_type": "ry",
        "direction": 1,
        "add_group": "profit",
    },
}

for field, semantics in FIELD_SEMANTICS.items():
    FIELD_RULES[field].update(semantics)


def _make_contracts() -> tuple[list[str], pd.Series]:
    industries = ["electronics", "computer", "communication", "media"]
    contracts = [f"TMT{i:04d}.SZ" if i % 2 else f"TMT{i:04d}.SH" for i in range(1, N_CONTRACTS + 1)]
    industry = pd.Series(
        [industries[i % len(industries)] for i in range(N_CONTRACTS)],
        index=pd.Index(contracts, name="Contract"),
        name="industry_code",
        dtype="category",
    )
    return contracts, industry


def _quarterly_to_daily(values: np.ndarray, report_dates: pd.DatetimeIndex, dates: pd.DatetimeIndex, contracts: list[str]) -> pd.DataFrame:
    q_df = pd.DataFrame(values, index=report_dates, columns=contracts)
    q_df.index.name = "Datetime"
    q_df.columns.name = "Contract"

    full_index = dates.union(report_dates).sort_values()
    daily = q_df.reindex(full_index).ffill().reindex(dates)
    daily.index.name = "Datetime"
    daily.columns.name = "Contract"
    return daily


def _rolling_ttm(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rolling(4, min_periods=1).sum().to_numpy()


def _winsor_positive(values: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    return np.maximum(values, floor)


def _cs_zscore_fill0(values: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score for mock Barra exposures, with NaNs filled 0."""

    centered = values.sub(values.mean(axis=1), axis=0)
    scaled = centered.div(values.std(axis=1).replace(0.0, np.nan), axis=0)
    out = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    out.index.name = "Datetime"
    out.columns.name = "Contract"
    return out


def make_mock_tmt_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    contracts, industry = _make_contracts()
    dates = pd.bdate_range(START_DATE, periods=N_DAYS) + pd.Timedelta(hours=15)
    dates.name = "Datetime"
    n_dates = len(dates)

    industry_names = np.array(["electronics", "computer", "communication", "media"])
    industry_to_id = {name: i for i, name in enumerate(industry_names)}
    industry_id = np.array([industry_to_id[name] for name in industry.astype(str).to_numpy()])

    quality = rng.normal(0.0, 1.0, N_CONTRACTS)
    growth = rng.normal(0.025, 0.035, N_CONTRACTS)
    beta = rng.uniform(0.75, 1.25, N_CONTRACTS)

    market_ret = rng.normal(0.0001, 0.009, n_dates)
    industry_ret = rng.normal(0.00005, 0.006, (n_dates, len(industry_names)))
    idio_ret = rng.normal(0.0, 0.018, (n_dates, N_CONTRACTS))
    style_drift = 0.00010 * quality + 0.00005 * (growth - growth.mean()) / growth.std()

    ret = beta[None, :] * market_ret[:, None] + industry_ret[:, industry_id] + idio_ret + style_drift[None, :]
    close = rng.uniform(12.0, 85.0, N_CONTRACTS)[None, :] * np.exp(np.cumsum(ret, axis=0))
    close = pd.DataFrame(close, index=dates, columns=contracts)
    close.index.name = "Datetime"
    close.columns.name = "Contract"

    prev_close = close.shift(1).fillna(close.iloc[0])
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.004, close.shape))
    intraday_spread = np.abs(rng.normal(0.010, 0.004, close.shape))
    high = pd.DataFrame(np.maximum(open_, close) * (1.0 + intraday_spread), index=dates, columns=contracts)
    low = pd.DataFrame(np.minimum(open_, close) * (1.0 - intraday_spread), index=dates, columns=contracts)
    open_ = pd.DataFrame(open_, index=dates, columns=contracts)
    for frame in [open_, high, low]:
        frame.index.name = "Datetime"
        frame.columns.name = "Contract"

    float_shares = rng.lognormal(mean=1.25, sigma=0.35, size=N_CONTRACTS)  # 100m shares.
    turnover = rng.lognormal(mean=-4.4, sigma=0.45, size=close.shape)
    turnover = pd.DataFrame(turnover, index=dates, columns=contracts).clip(0.001, 0.18)
    turnover.index.name = "Datetime"
    turnover.columns.name = "Contract"

    volume = turnover.mul(float_shares * 1e8, axis=1)
    amount = volume * close
    market_cap_raw = close.mul(float_shares, axis=1)  # 100m CNY.
    market_cap = np.log(market_cap_raw)
    market_cap.index.name = "Datetime"
    market_cap.columns.name = "Contract"

    q_start = (dates[0] - pd.DateOffset(years=2)).normalize()
    q_end = dates[-1].normalize()
    quarter_ends = pd.date_range(q_start, q_end, freq="QE-DEC")
    report_dates = pd.DatetimeIndex([q + BDay(20) + pd.Timedelta(hours=15) for q in quarter_ends])
    n_q = len(report_dates)
    q_idx = np.arange(n_q)[:, None]
    q_season = 1.0 + 0.10 * np.sin(2 * np.pi * (q_idx % 4) / 4)

    revenue_base = rng.lognormal(mean=2.55, sigma=0.55, size=N_CONTRACTS)
    revenue_mrq = revenue_base[None, :] * np.exp(growth[None, :] * q_idx / 4.0) * q_season
    revenue_mrq *= rng.lognormal(mean=0.0, sigma=0.10, size=(n_q, N_CONTRACTS))

    net_margin = 0.055 + 0.025 * quality[None, :] + rng.normal(0.0, 0.025, (n_q, N_CONTRACTS))
    net_margin = np.clip(net_margin, -0.08, 0.22)
    op_margin = net_margin + rng.normal(0.035, 0.015, (n_q, N_CONTRACTS))
    op_margin = np.clip(op_margin, -0.04, 0.30)

    net_profit_mrq = revenue_mrq * net_margin
    operating_profit_mrq = revenue_mrq * op_margin
    rd_expense_mrq = revenue_mrq * np.clip(0.055 + 0.025 * quality[None, :] + rng.normal(0.0, 0.012, (n_q, N_CONTRACTS)), 0.01, 0.18)
    capex_mrq = revenue_mrq * np.clip(0.045 + rng.normal(0.0, 0.015, (n_q, N_CONTRACTS)), 0.005, 0.12)
    free_cash_flow_mrq = net_profit_mrq + rng.normal(0.02, 0.08, (n_q, N_CONTRACTS)) * revenue_mrq - capex_mrq

    book_equity_q = _winsor_positive(revenue_base[None, :] * 2.8 + np.cumsum(net_profit_mrq * 0.45, axis=0))
    total_assets_q = _winsor_positive(book_equity_q * rng.uniform(1.25, 2.5, N_CONTRACTS)[None, :])
    liabilities_q = _winsor_positive(total_assets_q - book_equity_q)

    quarterly_daily = {
        "book_equity": _quarterly_to_daily(book_equity_q, report_dates, dates, contracts),
        "total_assets": _quarterly_to_daily(total_assets_q, report_dates, dates, contracts),
        "revenue_mrq": _quarterly_to_daily(revenue_mrq, report_dates, dates, contracts),
        "revenue_ttm": _quarterly_to_daily(_rolling_ttm(revenue_mrq), report_dates, dates, contracts),
        "operating_profit_mrq": _quarterly_to_daily(operating_profit_mrq, report_dates, dates, contracts),
        "operating_profit_ttm": _quarterly_to_daily(_rolling_ttm(operating_profit_mrq), report_dates, dates, contracts),
        "net_profit_mrq": _quarterly_to_daily(net_profit_mrq, report_dates, dates, contracts),
        "net_profit_ttm": _quarterly_to_daily(_rolling_ttm(net_profit_mrq), report_dates, dates, contracts),
        "rd_expense_ttm": _quarterly_to_daily(_rolling_ttm(rd_expense_mrq), report_dates, dates, contracts),
        "capex_ttm": _quarterly_to_daily(_rolling_ttm(capex_mrq), report_dates, dates, contracts),
        "free_cash_flow_ttm": _quarterly_to_daily(_rolling_ttm(free_cash_flow_mrq), report_dates, dates, contracts),
        "liabilities": _quarterly_to_daily(liabilities_q, report_dates, dates, contracts),
    }

    enterprise_value = market_cap_raw + quarterly_daily["liabilities"] - quarterly_daily["book_equity"] * 0.18
    enterprise_value = enterprise_value.clip(lower=1e-4)

    forecast_noise = pd.DataFrame(
        rng.normal(0.0, 0.04, (n_dates, N_CONTRACTS)),
        index=dates,
        columns=contracts,
    ).rolling(20, min_periods=1).mean()
    forecast_revenue_ry = quarterly_daily["revenue_ttm"] * (1.04 + growth[None, :] + forecast_noise)
    forecast_net_profit_ry = quarterly_daily["net_profit_ttm"] * (1.06 + growth[None, :] + forecast_noise)

    momentum_20 = close.pct_change(20).fillna(0.0)
    rating_base = pd.DataFrame(
        0.55 + 0.10 * quality[None, :] + 0.35 * momentum_20.to_numpy() + rng.normal(0.0, 0.08, (n_dates, N_CONTRACTS)),
        index=dates,
        columns=contracts,
    ).clip(0.0, 1.0)
    rating_30d = rating_base.rolling(20, min_periods=1).mean()
    rating_90d = rating_base.rolling(60, min_periods=1).mean()
    rating_180d = rating_base.rolling(120, min_periods=1).mean()
    for frame in [rating_30d, rating_90d, rating_180d, forecast_revenue_ry, forecast_net_profit_ry]:
        frame.index.name = "Datetime"
        frame.columns.name = "Contract"

    # Mock Barra style exposures. They are intended as neutralization controls,
    # not gene fields, so metadata lists them separately from FIELD_RULES.
    daily_ret = close.pct_change().fillna(0.0)
    barra_size = _cs_zscore_fill0(market_cap)
    beta_frame = pd.DataFrame(
        beta[None, :] + daily_ret.rolling(60, min_periods=10).corr(pd.Series(market_ret, index=dates)).fillna(0.0).to_numpy() * 0.10,
        index=dates,
        columns=contracts,
    )
    barra_beta = _cs_zscore_fill0(beta_frame)
    barra_momentum = _cs_zscore_fill0(close.pct_change(120).sub(close.pct_change(20), fill_value=0.0))
    barra_residual_volatility = _cs_zscore_fill0(daily_ret.rolling(60, min_periods=20).std())
    barra_non_linear_size = _cs_zscore_fill0(barra_size.pow(3))
    barra_book_to_price = _cs_zscore_fill0(np.log(quarterly_daily["book_equity"].clip(lower=1e-4)) - market_cap)
    barra_liquidity = _cs_zscore_fill0(np.log1p(turnover.rolling(60, min_periods=10).mean() * amount.rolling(60, min_periods=10).mean()))
    barra_earnings_yield = _cs_zscore_fill0(quarterly_daily["net_profit_ttm"].div(market_cap_raw.replace(0.0, np.nan)))
    barra_growth = _cs_zscore_fill0(forecast_revenue_ry.div(quarterly_daily["revenue_ttm"].replace(0.0, np.nan)).sub(1.0))
    barra_leverage = _cs_zscore_fill0(quarterly_daily["liabilities"].div(quarterly_daily["total_assets"].replace(0.0, np.nan)))

    tradeable = pd.DataFrame((rng.random((n_dates, N_CONTRACTS)) > 0.025).astype("int8"), index=dates, columns=contracts)
    tradeable.index.name = "Datetime"
    tradeable.columns.name = "Contract"
    label_1d = close.shift(-1).div(close).sub(1.0)
    label_5d = close.shift(-5).div(close).sub(1.0)
    label_20d = close.shift(-20).div(close).sub(1.0)

    frames = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
        "turnover": turnover,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "book_equity": quarterly_daily["book_equity"],
        "total_assets": quarterly_daily["total_assets"],
        "revenue_mrq": quarterly_daily["revenue_mrq"],
        "revenue_ttm": quarterly_daily["revenue_ttm"],
        "operating_profit_mrq": quarterly_daily["operating_profit_mrq"],
        "operating_profit_ttm": quarterly_daily["operating_profit_ttm"],
        "net_profit_mrq": quarterly_daily["net_profit_mrq"],
        "net_profit_ttm": quarterly_daily["net_profit_ttm"],
        "rd_expense_ttm": quarterly_daily["rd_expense_ttm"],
        "capex_ttm": quarterly_daily["capex_ttm"],
        "free_cash_flow_ttm": quarterly_daily["free_cash_flow_ttm"],
        "forecast_revenue_ry": forecast_revenue_ry,
        "forecast_net_profit_ry": forecast_net_profit_ry,
        "rating_score_30d": rating_30d,
        "rating_score_90d": rating_90d,
        "rating_score_180d": rating_180d,
        "barra_size": barra_size,
        "barra_beta": barra_beta,
        "barra_momentum": barra_momentum,
        "barra_residual_volatility": barra_residual_volatility,
        "barra_non_linear_size": barra_non_linear_size,
        "barra_book_to_price": barra_book_to_price,
        "barra_liquidity": barra_liquidity,
        "barra_earnings_yield": barra_earnings_yield,
        "barra_growth": barra_growth,
        "barra_leverage": barra_leverage,
        "is_tradeable": tradeable,
        "label_1d": label_1d,
        "label_5d": label_5d,
        "label_20d": label_20d,
    }

    long_parts = []
    for name, frame in frames.items():
        part = frame.stack(future_stack=True).rename(name)
        part.index = part.index.set_names(["Datetime", "Contract"])
        long_parts.append(part)
    data = pd.concat(long_parts, axis=1).sort_index()

    industry_long = pd.Series(
        np.tile(industry.reindex(contracts).astype(str).to_numpy(), n_dates),
        index=data.index,
        name="industry_code",
        dtype="category",
    )
    data["industry_code"] = industry_long

    float_cols = data.select_dtypes(include=["float64"]).columns
    data[float_cols] = data[float_cols].astype("float32")
    data["is_tradeable"] = data["is_tradeable"].astype("int8")
    return data


def save_outputs(data: pd.DataFrame) -> None:
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
            "Mock daily TMT stock panel for local debugging only.",
            "Datetime uses 15:00:00 to satisfy alpha_gen pivot validation.",
            "market_cap is stored as log market cap and should not be logged again inside alpha_gen.",
            "barra_size is the default size neutralization field; market_cap remains a legacy fallback.",
            "Barra style fields are cross-sectionally z-scored controls and are not searchable gene fields.",
            "Fundamental fields are quarterly values with a 20-business-day disclosure lag, then forward-filled to daily.",
            "label_1d, label_5d, and label_20d are future returns; pick one label column explicitly when building the cache.",
            "resi_pair and multi_resi require additive controls with matching unit_type/add_group/accounting transform semantics.",
        ],
        "numeric_fields": [c for c in data.columns if c != "industry_code"],
        "categorical_fields": ["industry_code"],
        "field_rules": FIELD_RULES,
        "size_field": "barra_size",
        "barra_style_fields": BARRA_STYLE_FIELDS,
        "extra_current_fields": BARRA_STYLE_FIELDS,
    }
    OUT_META.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    df = make_mock_tmt_data()
    save_outputs(df)
    size_mb = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"saved: {OUT_PARQUET} ({size_mb:.2f} MB)")
    print(f"saved: {OUT_META}")
    print(df.head(8))
