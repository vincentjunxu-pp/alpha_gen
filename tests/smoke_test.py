from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.core.factor_calc import apply_tradeable_mask, calculate_factor
from alpha_gen.core.ga import GAConfig, run_ga_search, should_neutralize_industry
from alpha_gen.core.gene import FieldRule, FactorGene, describe_gene, load_field_rules, random_gene, validate_gene
from alpha_gen.core.metrics import evaluate_factor, factor_group_pnl, top_group_excess_return
from alpha_gen.core.nsga2 import fast_non_dominated_sort, nsga2_select
from alpha_gen.core.preprocess import build_transform_cache, cache_summary, load_panel
from alpha_gen.core.utils import dot_log, get_rolling_windows, long_to_pivot, validate_long_format, validate_pivot_format


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "panels" / "mock_tmt_daily.parquet"
META_PATH = ROOT / "data" / "metadata" / "fixtures" / "mock_tmt_metadata.json"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_raises(exc_type: type[Exception], fn, message: str) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(message)


def mock_train_valid_dates(index):
    """Single mock window using the public rolling-window helper directly."""

    usable_dates = index[:-20]
    windows = get_rolling_windows(
        usable_dates,
        train_start_date=usable_dates[-(400 + 20 + 120)],
        test_start_date=usable_dates[-120],
        stride=120,
        horizon=20,
    )
    assert_true(len(windows) == 1, "mock single-window setup should return exactly one window")
    return windows[0]


def run_cpu_smoke() -> None:
    """Verify the complete CPU path on a small date window."""

    rules = load_field_rules(META_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    size_field = str(metadata.get("size_field", "barra_size"))
    barra_style_fields = tuple(metadata.get("barra_style_fields", ()))
    panel = load_panel(DATA_PATH)
    assert_true(validate_long_format(panel[["close"]], require_time_component=True), "panel should be long format")

    close = long_to_pivot(panel[["close"]], "close")
    assert_true(validate_pivot_format(close, require_time_component=True), "close should be pivot format")

    cache = build_transform_cache(panel, rules, extra_current_fields=[size_field, *barra_style_fields])
    train_window, valid_window = mock_train_valid_dates(cache.label.index)
    assert_true(train_window[0].hour == 15, "train window should preserve 15:00 timestamps")
    assert_true(valid_window[0].hour == 15, "validation window should preserve 15:00 timestamps")
    train_dates = train_window[-80:]

    manual_windows = get_rolling_windows(
        cache.label.index[:-20],
        train_start_date=cache.label.index[10],
        test_start_date=cache.label.index[120],
        stride=10,
        horizon=2,
    )
    manual_train, manual_valid = manual_windows[0]
    assert_true(len(manual_train) > 5, "manual rolling window should be controlled by start dates")
    assert_true(len(manual_valid) == 10, "manual rolling window should use stride as validation length")
    assert_true(manual_train[0].hour == 15, "manual rolling window should preserve 15:00 timestamps")

    gene = FactorGene(
        a="book_equity",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="ratio",
    )
    errors = validate_gene(gene, rules)
    assert_true(not errors, f"fixed gene should be valid, got {errors}")
    assert_true(not should_neutralize_industry("electronics"), "single industry should skip industry neutralization")
    assert_true(should_neutralize_industry("all"), "all-industry universe should require industry neutralization")
    assert_true(
        should_neutralize_industry(["electronics", "media"]),
        "multi-industry universe should require industry neutralization",
    )

    invalid_single = FactorGene(
        a="forecast_net_profit_ry",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="single",
    )
    assert_true(validate_gene(invalid_single, rules), "single should reject raw currency forecast fields")

    valid_pair_ratio = FactorGene(
        a="revenue_mrq",
        b="revenue_ttm",
        c="market_cap",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="pair_ratio",
    )
    assert_true(not validate_gene(valid_pair_ratio, rules), "pair_ratio should allow same-unit same-group accounting sums")

    invalid_pair_ratio = FactorGene(
        a="revenue_mrq",
        b="rating_score_30d",
        c="market_cap",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="pair_ratio",
    )
    assert_true(validate_gene(invalid_pair_ratio, rules), "pair_ratio should reject cross-unit additions")

    signed_input = pd.DataFrame([[-3.0, 0.0, 4.0]], columns=["A", "B", "C"])
    signed_expected = np.sign(signed_input) * np.log1p(np.abs(signed_input))
    signed_logged = dot_log(signed_input)
    assert_true(
        np.allclose(signed_logged.to_numpy(), signed_expected.to_numpy()),
        "log transform should use sign(x) * log(1 + abs(x))",
    )
    assert_true(np.isfinite(signed_logged.to_numpy()).all(), "signed log should keep negative values finite")

    resi_rules = dict(rules)
    resi_rules["unrestricted_a"] = FieldRule(
        can_y=False,
        can_x=False,
        allow_log=False,
        allow_current=True,
        allow_lag=False,
        allow_diff=False,
        allow_pct=False,
        allow_std=True,
        family="price",
        unit_type="price",
        statement="market",
        period_type="daily",
        direction=1,
        add_group="price",
    )
    resi_rules["unrestricted_b"] = FieldRule(
        can_y=False,
        can_x=False,
        allow_log=True,
        allow_current=True,
        allow_lag=False,
        allow_diff=False,
        allow_pct=False,
        allow_std=False,
        family="price",
        unit_type="ratio",
        statement="market",
        period_type="daily",
        direction=1,
        add_group="price",
    )
    valid_resi_any_ab = FactorGene(
        a="unrestricted_a",
        b="unrestricted_b",
        c="market_cap",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="resi",
        a_transform="rank_pct",
        b_transform="log",
    )
    assert_true(not validate_gene(valid_resi_any_ab, resi_rules), "resi a/b should accept any transform-legal fields")
    invalid_rolling_transform = FactorGene(
        a="unrestricted_a",
        b="unrestricted_b",
        c="market_cap",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="resi",
        a_transform="std_2q",
        b_transform="log",
    )
    assert_true(validate_gene(invalid_rolling_transform, resi_rules), "rolling std transforms should not be searchable")

    resi_only_rules = {
        "target_signal": FieldRule(
            can_y=False,
            can_x=False,
            allow_log=False,
            allow_current=True,
            allow_lag=False,
            allow_diff=False,
            allow_pct=False,
            allow_std=False,
            family="analyst",
            unit_type="score",
            statement="analyst",
            period_type="30d",
            direction=1,
            add_group="score",
        ),
        "unrestricted_b": FieldRule(
            can_y=False,
            can_x=False,
            allow_log=False,
            allow_current=True,
            allow_lag=False,
            allow_diff=False,
            allow_pct=False,
            allow_std=False,
            family="price",
            unit_type="price",
            statement="market",
            period_type="daily",
            direction=1,
            add_group="price",
        ),
    }
    sampled_resi = random_gene(resi_only_rules, np.random.default_rng(11), mode_probabilities={"resi": 1.0})
    assert_true(sampled_resi.mode == "resi", "resi mode should be viable without any can_x fields")
    assert_true(
        not validate_gene(sampled_resi, resi_only_rules),
        "resi sampler should draw a/b from unrestricted transform-legal fields",
    )

    multi_resi_gene = FactorGene(
        a="rating_score_30d",
        b="operating_profit_mrq",
        c="net_profit_mrq",
        d="forecast_net_profit_ry",
        left_op="+",
        right_op="+",
        mode="multi_resi",
    )
    assert_true(not validate_gene(multi_resi_gene, rules), "multi_resi should allow constrained additive controls")

    invalid_multi_resi_gene = FactorGene(
        a="rating_score_30d",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="multi_resi",
    )
    assert_true(validate_gene(invalid_multi_resi_gene, rules), "multi_resi should reject unrelated additive controls")

    resi_pair_gene = FactorGene(
        a="rating_score_30d",
        b="revenue_mrq",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="resi_pair",
    )
    assert_true(not validate_gene(resi_pair_gene, rules), "resi_pair should allow residual(A ~ B + C)")

    spread_gene = FactorGene(
        a="book_equity",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="-",
        right_op="+",
        mode="spread",
    )
    assert_true(not validate_gene(spread_gene, rules), "spread should allow two valid accounting ratios")

    industry_gene = FactorGene(
        a="rating_score_30d",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="single",
        a_transform="ind_rank_pct",
    )
    assert_true(not validate_gene(industry_gene, rules), "industry rank should be a legal unary transform")

    style_rules = dict(rules)
    style_rules["sp_ratio_ttm"] = FieldRule(
        can_y=False,
        can_x=True,
        allow_log=False,
        allow_current=True,
        allow_lag=False,
        allow_diff=False,
        allow_pct=False,
        allow_std=False,
        family="valuation",
        unit_type="ratio",
        statement="market",
        period_type="ttm",
        direction=1,
        add_group="valuation",
    )
    valid_style = FactorGene(
        a="rating_score_30d",
        b="sp_ratio_ttm",
        c="market_cap",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="style_composite",
    )
    assert_true(not validate_gene(valid_style, style_rules), "style_composite should allow whitelisted style pairs")
    invalid_style_sub = FactorGene(
        a="rating_score_30d",
        b="sp_ratio_ttm",
        c="market_cap",
        d="enterprise_value",
        left_op="-",
        right_op="+",
        mode="style_composite",
    )
    assert_true(validate_gene(invalid_style_sub, style_rules), "style_composite should only allow additive style combination")

    cache.current[("sp_ratio_ttm", False)] = cache.get_current("market_cap", use_log=False).mul(-1.0)
    cache.field_rules = style_rules
    style_factor = calculate_factor(valid_style, cache)
    assert_true(style_factor.notna().any().any(), "style_composite should produce a non-empty factor")

    multi_resi_factor = calculate_factor(multi_resi_gene, cache)
    assert_true(multi_resi_factor.notna().any().any(), "multi_resi should produce a non-empty factor")
    resi_pair_factor = calculate_factor(resi_pair_gene, cache)
    assert_true(resi_pair_factor.notna().any().any(), "resi_pair should produce a non-empty factor")
    spread_factor = calculate_factor(spread_gene, cache)
    assert_true(spread_factor.notna().any().any(), "spread should produce a non-empty factor")
    industry_factor = calculate_factor(industry_gene, cache)
    assert_true(industry_factor.notna().any().any(), "industry unary transform should produce a non-empty factor")

    industry_neutral_raw = calculate_factor(
        gene,
        cache,
        neutralize_size=False,
        neutralize_industry=True,
    )
    industry_check = industry_neutral_raw.stack().rename("factor").to_frame()
    industry_check["industry"] = cache.industry.stack().reindex(industry_check.index)
    industry_check = industry_check.dropna()
    max_abs_industry_mean = float(industry_check.groupby(["Datetime", "industry"], observed=True)["factor"].mean().abs().max())
    assert_true(max_abs_industry_mean < 1e-5, "industry-neutralized factor should have near-zero industry means")

    barra_size_factor = calculate_factor(gene, cache, size_field="barra_size")
    assert_true(barra_size_factor.notna().any().any(), "custom Barra size field should be usable for neutralization")
    assert_raises(
        KeyError,
        lambda: calculate_factor(gene, cache, size_field="missing_barra_size"),
        "missing custom Barra size field should raise KeyError",
    )

    factor = calculate_factor(gene, cache)
    score = evaluate_factor(factor, cache.label, tradeable=cache.tradeable, dates=train_dates)
    assert_true(score.n_ic_obs > 0, "CPU factor score should have IC observations")
    assert_true(score.coverage > 0.30, "CPU factor score coverage is unexpectedly low")
    assert_true(len(score.objectives) == 3, "FactorScore should expose three NSGA-II objectives")
    assert_true(
        score.objectives == (score.rank_ic_ir, score.ndcg_at_k, score.neutralized_icir),
        "NSGA-II objectives should be rank_ic_ir, ndcg_at_k, neutralized_icir",
    )
    assert_true(
        score.neutralized_icir == 0.0 and score.neutralized_n_ic_obs == 0,
        "CPU evaluator should not copy raw ICIR into an unavailable Barra-neutralized metric",
    )

    toy_dates = pd.to_datetime(["2026-01-01 15:00:00"])
    toy_cols = ["A", "B", "C"]
    toy_factor = pd.DataFrame([[1.0, 2.0, np.nan]], index=toy_dates, columns=toy_cols)
    toy_label = pd.DataFrame([[1.0, 2.0, np.nan]], index=toy_dates, columns=toy_cols)
    toy_score = evaluate_factor(toy_factor, toy_label)
    assert_true(toy_score.n_ic_obs == 0, "two-name RankIC should be skipped by default")

    toy_tradeable = pd.DataFrame([[1.0, np.nan, 0.0]], index=toy_dates, columns=toy_cols)
    masked = apply_tradeable_mask(pd.DataFrame([[1.0, 2.0, 3.0]], index=toy_dates, columns=toy_cols), toy_tradeable)
    assert_true(np.isnan(masked.loc[toy_dates[0], "B"]), "NaN tradeable values should be masked out")

    assert_raises(
        ValueError,
        lambda: evaluate_factor(toy_factor, toy_label, direction=0),
        "invalid direction should raise ValueError",
    )

    top_group_ann = top_group_excess_return(
        factor,
        cache.label,
        tradeable=cache.tradeable,
        dates=train_dates,
        direction=score.direction,
    )
    pnl_result = factor_group_pnl(
        factor=factor,
        label=cache.label,
        tradeable=cache.tradeable,
        dates=train_dates,
        direction=score.direction,
        n_groups=10,
    )
    gross_pnl_result = factor_group_pnl(
        factor=factor,
        label=cache.label,
        tradeable=cache.tradeable,
        dates=train_dates,
        direction=score.direction,
        n_groups=10,
        commission_rate=0.0,
        slippage_rate=0.0,
        stamp_tax_rate=0.0,
    )
    assert_true(np.isfinite(top_group_ann), "top-group annualized label return should be finite")
    assert_true("pnl_longshort_ann" in pnl_result, "group PnL result should include long-short annualized return")
    assert_true(
        pnl_result["pnl_long_ann"] <= gross_pnl_result["pnl_long_ann"],
        "transaction costs should reduce long-group annualized return",
    )

    pnl_dates = pd.to_datetime(["2026-01-01 15:00:00", "2026-01-02 15:00:00", "2026-01-03 15:00:00"])
    pnl_cols = ["A", "B", "C", "D"]
    toy_factor_pnl = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=pnl_dates, columns=pnl_cols)
    toy_label_pnl = pd.DataFrame(
        [
            [0.00, 0.00, 0.04, 0.06],
            [0.03, 0.01, 0.02, 0.04],
            [-0.01, -0.01, 0.01, 0.03],
        ],
        index=pnl_dates,
        columns=pnl_cols,
    )
    toy_pnl = factor_group_pnl(
        factor=toy_factor_pnl,
        label=toy_label_pnl,
        n_groups=2,
        label_horizon=1,
        rebalance_freq=1,
        commission_rate=0.0,
        slippage_rate=0.0,
        stamp_tax_rate=0.0,
    )
    expected_raw_long_sharpe = toy_pnl["pnl_long"].mean() / toy_pnl["pnl_long"].std() * np.sqrt(244)
    assert_true(
        np.isclose(toy_pnl["long_sharpe"], expected_raw_long_sharpe),
        "long_sharpe should be calculated from raw long portfolio return",
    )
    expected_excess = toy_pnl["pnl_long"] - toy_pnl["benchmark_return"].reindex(toy_pnl["pnl_long"].index)
    expected_excess_sharpe = expected_excess.mean() / expected_excess.std() * np.sqrt(244)
    assert_true(
        np.isclose(toy_pnl["long_excess_sharpe"], expected_excess_sharpe),
        "long_excess_sharpe should keep the benchmark-relative diagnostic",
    )
    expected_longshort_sharpe = toy_pnl["pnl_longshort"].mean() / toy_pnl["pnl_longshort"].std() * np.sqrt(244)
    assert_true(
        np.isclose(toy_pnl["longshort_sharpe"], expected_longshort_sharpe),
        "longshort_sharpe should be calculated directly from long-short return",
    )
    assert_true("long_raw_sharpe" in toy_pnl, "raw Sharpe should be kept for diagnostics")

    toy_pnl_2d = factor_group_pnl(
        factor=toy_factor_pnl,
        label=toy_label_pnl,
        n_groups=2,
        label_horizon=2,
        commission_rate=0.0,
        slippage_rate=0.0,
        stamp_tax_rate=0.0,
    )
    assert_true(toy_pnl_2d["rebalance_freq"] == 2, "rebalance frequency should default to label horizon")
    assert_true(toy_pnl_2d["n_rebalance_obs"] == 2, "20-day style PnL should sample non-overlapping rebalance dates")

    toy_objectives = [(1.0, 1.0), (0.9, 1.2), (0.5, 0.5), (1.1, 0.8)]
    fronts = fast_non_dominated_sort(toy_objectives)
    selected = nsga2_select(toy_objectives, 2)
    assert_true(fronts[0] == [0, 1, 3], f"unexpected NSGA-II front: {fronts[0]}")
    assert_true(len(selected) == 2, "NSGA-II selection size mismatch")

    # Tiny GA smoke. This checks the orchestration path without pretending to be
    # an economically meaningful run.
    ga_result = run_ga_search(
        cache=cache,
        field_rules=rules,
        train_dates=train_dates,
        config=GAConfig(
            population_size=4,
            generations=0,
            random_seed=7,
            size_field="barra_size",
            industry_scope="all",
        ),
    )
    assert_true(len(ga_result.final_population) == 4, "GA final population size mismatch")

    rng = np.random.default_rng(123)
    sampled = random_gene(rules, rng)
    assert_true(not validate_gene(sampled, rules), "random gene should be valid")
    single_sampled = random_gene(rules, rng, mode_probabilities={"single": 1.0})
    assert_true(single_sampled.mode == "single", "mode_probabilities should control random mode sampling")
    multi_resi_sampled = random_gene(rules, rng, mode_probabilities={"multi_resi": 1.0})
    assert_true(multi_resi_sampled.mode == "multi_resi", "multi_resi mode should be sampleable")
    assert_true(not validate_gene(multi_resi_sampled, rules), "sampled multi_resi should satisfy additive controls")
    resi_pair_sampled = random_gene(rules, rng, mode_probabilities={"resi_pair": 1.0})
    assert_true(resi_pair_sampled.mode == "resi_pair", "resi_pair mode should be sampleable")
    spread_sampled = random_gene(rules, rng, mode_probabilities={"spread": 1.0})
    assert_true(spread_sampled.mode == "spread", "spread mode should be sampleable")

    print("CPU smoke ok")
    print("cache", cache_summary(cache))
    print("fixed_gene", describe_gene(gene))
    print("fixed_score", score.to_dict())


def run_gpu_smoke() -> None:
    """Verify Torch/CUDA path if torch is available."""

    from alpha_gen.core.torch_backend import (
        TorchEvalContext,
        apply_transform_torch,
        calculate_factor_tensor,
        cuda_memory_summary,
        dynamic_barra_neutralize_torch,
        evaluate_factor_tensor,
    )
    import torch

    rules = load_field_rules(META_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    size_field = str(metadata.get("size_field", "barra_size"))
    barra_style_fields = tuple(metadata.get("barra_style_fields", ()))
    panel = load_panel(DATA_PATH)
    cache = build_transform_cache(panel, rules, extra_current_fields=[size_field, *barra_style_fields])
    train_window, _ = mock_train_valid_dates(cache.label.index)
    train_dates = train_window[-80:]

    ctx = TorchEvalContext(
        cache=cache,
        device="auto",
        cache_on_device=True,
        barra_style_fields=barra_style_fields,
    )
    torch_signed_input = torch.tensor([[-3.0, 0.0, 4.0]], device=ctx.device)
    torch_signed_expected = torch.sign(torch_signed_input) * torch.log1p(torch_signed_input.abs())
    torch_signed_logged = apply_transform_torch(torch_signed_input, "log")
    assert_true(
        torch.allclose(torch_signed_logged, torch_signed_expected),
        "Torch log transform should use sign(x) * log(1 + abs(x))",
    )

    gene = FactorGene(
        a="book_equity",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="ratio",
    )
    factor = calculate_factor_tensor(gene, ctx)
    assert_true(tuple(factor.shape) == cache.label.shape, "GPU factor tensor must be [date, contract]")

    cpu_factor = calculate_factor(gene, cache).to_numpy(dtype=np.float32)
    gpu_factor = factor.detach().cpu().numpy()
    mean_abs_diff = np.nanmean(np.abs(cpu_factor - gpu_factor))
    assert_true(mean_abs_diff < 1e-4, f"CPU/GPU factor mismatch is too large: {mean_abs_diff}")

    spread_gene = FactorGene(
        a="book_equity",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="-",
        right_op="+",
        mode="spread",
    )
    spread_tensor = calculate_factor_tensor(spread_gene, ctx)
    assert_true(torch.isfinite(spread_tensor).any().item(), "GPU spread factor should be non-empty")

    multi_resi_gene = FactorGene(
        a="rating_score_30d",
        b="operating_profit_mrq",
        c="net_profit_mrq",
        d="forecast_net_profit_ry",
        left_op="+",
        right_op="+",
        mode="multi_resi",
    )
    multi_resi_tensor = calculate_factor_tensor(multi_resi_gene, ctx)
    assert_true(torch.isfinite(multi_resi_tensor).any().item(), "GPU multi_resi factor should be non-empty")

    resi_pair_gene = FactorGene(
        a="rating_score_30d",
        b="revenue_mrq",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="resi_pair",
    )
    resi_pair_tensor = calculate_factor_tensor(resi_pair_gene, ctx)
    assert_true(torch.isfinite(resi_pair_tensor).any().item(), "GPU resi_pair factor should be non-empty")

    industry_gene = FactorGene(
        a="rating_score_30d",
        b="market_cap",
        c="revenue_ttm",
        d="enterprise_value",
        left_op="+",
        right_op="+",
        mode="single",
        a_transform="ind_rank_pct",
    )
    industry_tensor = calculate_factor_tensor(industry_gene, ctx)
    assert_true(torch.isfinite(industry_tensor).any().item(), "GPU industry transform factor should be non-empty")

    custom_neutral_tensor = calculate_factor_tensor(
        gene,
        ctx,
        neutralize_industry=True,
        size_field="barra_size",
    )
    assert_true(torch.isfinite(custom_neutral_tensor).any().item(), "GPU custom neutralization path should be non-empty")

    barra_styles = ctx.barra_styles()
    direct_barra_factor = barra_styles[..., 0] + 0.05 * barra_styles[..., 1]
    neutralization = dynamic_barra_neutralize_torch(
        direct_barra_factor,
        barra_styles,
        mask=ctx.tradeable(),
        corr_threshold=0.30,
        max_styles=2,
    )
    assert_true(neutralization.selected_mask.any().item(), "dynamic Barra neutralization should select high exposure")
    assert_true(torch.isfinite(neutralization.residual_factor).any().item(), "Barra residual factor should be non-empty")

    score = evaluate_factor_tensor(factor, ctx, dates=train_dates)

    assert_true(score.n_ic_obs > 0, "GPU factor score should have IC observations")
    assert_true(score.coverage > 0.30, "GPU factor score coverage is unexpectedly low")
    assert_true(score.barra_max_abs_corr >= 0.0, "GPU score should report Barra exposure diagnostics")
    assert_true(score.neutralized_n_ic_obs > 0, "GPU score should compute residual-factor IC observations")
    assert_true(
        score.barra_selected_count == len(score.barra_selected_styles),
        "GPU score should record the names of selected Barra neutralization styles",
    )
    if score.barra_selected_count > 0:
        assert_true(
            bool(score.to_dict()["barra_selected_styles"]),
            "GPU score dict should expose selected Barra styles for CSV logging",
        )
    assert_true(len(score.objectives) == 3, "GPU FactorScore should expose three NSGA-II objectives")
    assert_true(
        score.objectives == (score.rank_ic_ir, score.ndcg_at_k, score.neutralized_icir),
        "GPU NSGA-II objectives should be rank_ic_ir, ndcg_at_k, neutralized_icir",
    )
    gpu_ga_result = run_ga_search(
        cache=cache,
        field_rules=rules,
        train_dates=train_dates,
        config=GAConfig(
            population_size=4,
            generations=0,
            random_seed=17,
            size_field="barra_size",
            industry_scope="all",
            barra_style_fields=barra_style_fields,
            use_gpu=True,
            device=str(ctx.device),
        ),
        eval_context=ctx,
    )
    assert_true(
        any(item.train_score.neutralized_n_ic_obs > 0 for item in gpu_ga_result.final_population),
        "GPU GA search should feed a real neutralized_icir objective into NSGA-II",
    )

    print("GPU smoke ok")
    print("cuda", cuda_memory_summary())
    print("fixed_score", score.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="also run Torch/CUDA smoke checks")
    args = parser.parse_args()

    if args.gpu:
        # On this Windows/conda setup, importing torch after the pandas/scipy
        # CPU path can fail while loading fbgemm.dll. Preloading torch matches
        # the notebook GPU workflow, where torch_backend is imported up front.
        import torch  # noqa: F401

    run_cpu_smoke()
    if args.gpu:
        run_gpu_smoke()


if __name__ == "__main__":
    main()
