from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.core.factor_calc import apply_tradeable_mask, calculate_factor
from alpha_gen.core.ga import GAConfig, run_ga_search
from alpha_gen.core.gene import FactorGene, describe_gene, load_field_rules, random_gene, validate_gene
from alpha_gen.core.metrics import evaluate_factor, factor_group_pnl, top_group_excess_return
from alpha_gen.core.nsga2 import fast_non_dominated_sort, nsga2_select
from alpha_gen.core.preprocess import build_transform_cache, cache_summary, load_panel
from alpha_gen.core.utils import get_rolling_windows, long_to_pivot, validate_long_format, validate_pivot_format


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mock_tmt_daily.parquet"
META_PATH = ROOT / "data" / "mock_tmt_metadata.json"


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
    panel = load_panel(DATA_PATH)
    assert_true(validate_long_format(panel[["close"]], require_time_component=True), "panel should be long format")

    close = long_to_pivot(panel[["close"]], "close")
    assert_true(validate_pivot_format(close, require_time_component=True), "close should be pivot format")

    cache = build_transform_cache(panel, rules)
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

    factor = calculate_factor(gene, cache)
    score = evaluate_factor(factor, cache.label, tradeable=cache.tradeable, dates=train_dates)
    assert_true(score.n_ic_obs > 0, "CPU factor score should have IC observations")
    assert_true(score.coverage > 0.30, "CPU factor score coverage is unexpectedly low")

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
    expected_excess = toy_pnl["pnl_long"] - toy_pnl["benchmark_return"].reindex(toy_pnl["pnl_long"].index)
    expected_sharpe = expected_excess.mean() / expected_excess.std() * np.sqrt(244)
    assert_true(
        np.isclose(toy_pnl["long_sharpe"], expected_sharpe),
        "long_sharpe should be calculated from long return minus equal-weight benchmark return",
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
        config=GAConfig(population_size=4, generations=0, random_seed=7),
    )
    assert_true(len(ga_result.final_population) == 4, "GA final population size mismatch")

    rng = np.random.default_rng(123)
    sampled = random_gene(rules, rng)
    assert_true(not validate_gene(sampled, rules), "random gene should be valid")

    print("CPU smoke ok")
    print("cache", cache_summary(cache))
    print("fixed_gene", describe_gene(gene))
    print("fixed_score", score.to_dict())


def run_gpu_smoke() -> None:
    """Verify Torch/CUDA path if torch is available."""

    from alpha_gen.core.torch_backend import (
        TorchEvalContext,
        calculate_factor_tensor,
        cuda_memory_summary,
        evaluate_factor_tensor,
    )

    rules = load_field_rules(META_PATH)
    panel = load_panel(DATA_PATH)
    cache = build_transform_cache(panel, rules)
    train_window, _ = mock_train_valid_dates(cache.label.index)
    train_dates = train_window[-80:]

    ctx = TorchEvalContext(cache=cache, device="auto", cache_on_device=True)
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

    score = evaluate_factor_tensor(factor, ctx, dates=train_dates)

    assert_true(score.n_ic_obs > 0, "GPU factor score should have IC observations")
    assert_true(score.coverage > 0.30, "GPU factor score coverage is unexpectedly low")

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
