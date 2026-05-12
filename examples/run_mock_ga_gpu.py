from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.core.ga import (
    GAConfig,
    ValidationCriteria,
    evaluated_to_frame,
    export_search_result,
    run_ga_search,
    validate_population,
)
from alpha_gen.core.gene import load_field_rules
from alpha_gen.core.preprocess import build_transform_cache, cache_summary, load_panel
from alpha_gen.core.torch_backend import TorchEvalContext, cuda_memory_summary


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mock_tmt_daily.parquet"
META_PATH = ROOT / "data" / "mock_tmt_metadata.json"
RESULT_DIR = ROOT / "results"


def _resolve_date(index: pd.DatetimeIndex, value: str | None, *, default_pos: int, side: str = "left") -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(index[default_pos])
    pos = int(index.searchsorted(pd.Timestamp(value), side=side))
    if pos >= len(index):
        raise ValueError(f"date {value!r} is after available data end {index[-1]}")
    return pd.Timestamp(index[pos])


def make_train_valid_dates(
    dates: pd.DatetimeIndex,
    *,
    label_horizon: int,
    train_days: int,
    valid_days: int,
    valid_start: str | None,
    valid_end: str | None,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    usable_dates = pd.DatetimeIndex(dates[:-label_horizon])
    if len(usable_dates) <= label_horizon + 5:
        raise ValueError("not enough dates to build train/validation windows")

    if valid_start is None and valid_end is None:
        valid_len = min(valid_days, max(1, len(usable_dates) // 5))
        valid_start_ts = pd.Timestamp(usable_dates[-valid_len])
        valid_end_ts = pd.Timestamp(usable_dates[-1])
        valid_dates = usable_dates[(usable_dates >= valid_start_ts) & (usable_dates <= valid_end_ts)]
    else:
        default_start_pos = max(0, len(usable_dates) - valid_days)
        valid_start_ts = _resolve_date(usable_dates, valid_start, default_pos=default_start_pos, side="left")
        if valid_end is None:
            valid_end_pos = min(len(usable_dates), int(usable_dates.get_indexer([valid_start_ts])[0]) + valid_days)
            valid_dates = usable_dates[(usable_dates >= valid_start_ts)][: valid_end_pos - int(usable_dates.get_indexer([valid_start_ts])[0])]
        else:
            valid_end_ts = _resolve_date(usable_dates, valid_end, default_pos=-1, side="left")
            valid_dates = usable_dates[(usable_dates >= valid_start_ts) & (usable_dates < valid_end_ts)]

    if valid_dates.empty:
        raise ValueError("validation window is empty")

    valid_start_idx = int(dates.get_indexer([valid_dates[0]])[0])
    train_end_idx = valid_start_idx - label_horizon - 1
    if train_end_idx < 0:
        raise ValueError("label_horizon leaves no train dates before validation")

    train_start_idx = 0 if train_days <= 0 else max(0, train_end_idx - train_days + 1)
    train_dates = pd.DatetimeIndex(dates[train_start_idx : train_end_idx + 1])
    if train_dates.empty:
        raise ValueError("training window is empty")

    return train_dates, pd.DatetimeIndex(valid_dates)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU GA run template for alpha_gen.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--meta-path", type=Path, default=META_PATH)
    parser.add_argument("--label-col", default="label_20d")
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--industry-col", default="industry_code")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--prefix", default="mock_ga_gpu")

    parser.add_argument("--valid-start", default=None, help="Inclusive validation start date, e.g. 2021-01-01.")
    parser.add_argument("--valid-end", default=None, help="Exclusive validation end date, e.g. 2024-01-01.")
    parser.add_argument("--train-days", type=int, default=756, help="Training trading days before the horizon gap. Use <=0 for all available history.")
    parser.add_argument("--valid-days", type=int, default=120, help="Fallback validation length when valid-start/end are omitted.")
    parser.add_argument("--label-horizon", type=int, default=20)
    parser.add_argument("--rebalance-days", type=int, default=None, help="Defaults to label-horizon.")

    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--crossover-prob", type=float, default=0.85)
    parser.add_argument("--mutation-prob", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=20260428)
    parser.add_argument("--ndcg-top-fraction", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.50)

    parser.add_argument("--min-valid-abs-rank-ic", type=float, default=0.015)
    parser.add_argument("--min-valid-ic-win-rate", type=float, default=0.53)
    parser.add_argument("--min-valid-top-excess-ann", type=float, default=0.00)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-on-device", action="store_true", default=True)
    parser.add_argument("--no-cache-on-device", dest="cache_on_device", action="store_false")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    field_rules = load_field_rules(args.meta_path)
    panel = load_panel(args.data_path)
    cache = build_transform_cache(
        panel,
        field_rules,
        label_col=args.label_col,
        tradeable_col=args.tradeable_col,
        industry_col=args.industry_col,
    )
    train_dates, valid_dates = make_train_valid_dates(
        cache.label.index,
        label_horizon=args.label_horizon,
        train_days=args.train_days,
        valid_days=args.valid_days,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )

    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        random_seed=args.random_seed,
        ndcg_k=None,
        ndcg_top_fraction=args.ndcg_top_fraction,
        min_coverage=args.min_coverage,
        use_gpu=True,
        device=args.device,
        cache_on_device=args.cache_on_device,
        show_progress=args.show_progress,
    )

    torch_context = TorchEvalContext(
        cache=cache,
        device=config.device,
        cache_on_device=config.cache_on_device,
    )

    result = run_ga_search(
        cache=cache,
        field_rules=field_rules,
        train_dates=train_dates,
        config=config,
        eval_context=torch_context,
    )

    rebalance_days = args.rebalance_days or args.label_horizon
    validate_population(
        evaluated_population=result.final_population,
        cache=cache,
        valid_dates=valid_dates,
        criteria=ValidationCriteria(
            min_abs_rank_ic=args.min_valid_abs_rank_ic,
            min_ic_win_rate=args.min_valid_ic_win_rate,
            min_top_excess_ann=args.min_valid_top_excess_ann,
            min_coverage=args.min_coverage,
        ),
        ndcg_k=config.ndcg_k,
        ndcg_top_fraction=config.ndcg_top_fraction,
        label_horizon=args.label_horizon,
        rebalance_freq=rebalance_days,
        eval_context=torch_context,
        show_progress=args.show_progress,
    )

    paths = export_search_result(result, args.result_dir, prefix=args.prefix)
    final_df = evaluated_to_frame(result.final_population)

    print("data:", args.data_path)
    print("metadata:", args.meta_path)
    print("cache:", cache_summary(cache))
    print("cuda:", cuda_memory_summary())
    print("train_dates:", train_dates[0], "->", train_dates[-1], len(train_dates))
    print("valid_dates:", valid_dates[0], "->", valid_dates[-1], len(valid_dates))
    print("label_horizon:", args.label_horizon)
    print("rebalance_days:", rebalance_days)
    print("exports:", {key: str(path) for key, path in paths.items()})

    cols = [
        "expression",
        "train_abs_rank_ic",
        "train_ic_win_rate",
        "train_ndcg_at_k",
        "train_coverage",
        "valid_abs_rank_ic",
        "valid_ic_win_rate",
        "valid_top_excess_ann",
        "valid_pnl_rebalance_freq",
        "valid_pnl_n_rebalance_obs",
        "valid_pnl_pnl_long_ann",
        "valid_pnl_pnl_longshort_ann",
        "valid_pnl_longshort_sharpe",
        "passed_validation",
        "error",
    ]
    existing_cols = [col for col in cols if col in final_df.columns]
    print(final_df.head(12)[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
