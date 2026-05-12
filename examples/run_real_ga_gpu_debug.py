from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
from alpha_gen.core.utils import get_rolling_windows
from alpha_gen.data.make_metadata_from_columns import metadata_from_panel, write_metadata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_DATA = ROOT / "data" / "real_tmt_daily.parquet"
DEFAULT_REAL_META = ROOT / "data" / "real_metadata.json"
DEFAULT_MOCK_DATA = ROOT / "data" / "mock_tmt_daily.parquet"
DEFAULT_MOCK_META = ROOT / "data" / "mock_tmt_metadata.json"
RESULT_DIR = ROOT / "results"


def resolve_default_paths(use_mock: bool) -> tuple[Path, Path]:
    if use_mock or not DEFAULT_REAL_DATA.exists():
        return DEFAULT_MOCK_DATA, DEFAULT_MOCK_META
    return DEFAULT_REAL_DATA, DEFAULT_REAL_META


def ensure_metadata(
    panel_path: Path,
    meta_path: Path,
    *,
    auto_metadata: bool,
    label_col: str,
    tradeable_col: str,
    industry_col: str,
) -> Path:
    if meta_path.exists() or not auto_metadata:
        return meta_path

    panel = load_panel(panel_path)
    metadata = metadata_from_panel(
        panel,
        dataset=panel_path.name,
        label_col=label_col,
        tradeable_col=tradeable_col,
        industry_col=industry_col,
    )
    return write_metadata(metadata, meta_path)


def make_debug_windows(
    dates,
    *,
    label_horizon: int,
    train_days: int,
    valid_days: int,
):
    usable_dates = dates[:-label_horizon] if len(dates) > label_horizon else dates
    if len(usable_dates) <= label_horizon + 5:
        raise ValueError("not enough dates to build train/validation windows")

    valid_len = min(valid_days, max(1, len(usable_dates) // 5))
    test_start_idx = max(label_horizon + 1, len(usable_dates) - valid_len)
    train_start_idx = max(0, test_start_idx - label_horizon - train_days)

    windows = get_rolling_windows(
        usable_dates,
        train_start_date=usable_dates[train_start_idx],
        test_start_date=usable_dates[test_start_idx],
        stride=valid_len,
        horizon=label_horizon,
    )
    if not windows:
        raise ValueError("rolling-window helper returned no windows")
    return windows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mock", action="store_true", help="Force mock data for local debugging.")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--meta-path", type=Path)
    parser.add_argument("--auto-metadata", action="store_true", help="Generate metadata from panel columns if meta-path is missing.")
    parser.add_argument("--label-col", default="label_20d")
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--industry-col", default="industry_code")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--prefix", default="real_ga_gpu_debug")
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument("--train-days", type=int, default=80)
    parser.add_argument("--valid-days", type=int, default=40)
    parser.add_argument("--label-horizon", type=int, default=20)
    parser.add_argument("--rebalance-days", type=int, default=None, help="Validation rebalance interval. Defaults to label-horizon.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    default_data, default_meta = resolve_default_paths(args.use_mock)
    data_path = args.data_path or default_data
    meta_path = args.meta_path or default_meta
    meta_path = ensure_metadata(
        data_path,
        meta_path,
        auto_metadata=args.auto_metadata,
        label_col=args.label_col,
        tradeable_col=args.tradeable_col,
        industry_col=args.industry_col,
    )

    field_rules = load_field_rules(meta_path)
    panel = load_panel(data_path)
    cache = build_transform_cache(
        panel,
        field_rules,
        label_col=args.label_col,
        tradeable_col=args.tradeable_col,
        industry_col=args.industry_col,
    )
    train_dates, valid_dates = make_debug_windows(
        cache.label.index,
        label_horizon=args.label_horizon,
        train_days=args.train_days,
        valid_days=args.valid_days,
    )

    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_prob=0.85,
        mutation_prob=0.25,
        random_seed=42,
        ndcg_k=None,
        ndcg_top_fraction=0.20,
        min_coverage=0.50,
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

    validate_population(
        evaluated_population=result.final_population,
        cache=cache,
        valid_dates=valid_dates,
        criteria=ValidationCriteria(
            min_abs_rank_ic=0.015,
            min_ic_win_rate=0.53,
            min_top_excess_ann=0.00,
            min_coverage=0.50,
        ),
        ndcg_k=config.ndcg_k,
        ndcg_top_fraction=config.ndcg_top_fraction,
        label_horizon=args.label_horizon,
        rebalance_freq=args.rebalance_days or args.label_horizon,
        eval_context=torch_context,
        show_progress=args.show_progress,
    )

    paths = export_search_result(result, args.result_dir, prefix=args.prefix)
    final_df = evaluated_to_frame(result.final_population)

    print("data:", data_path)
    print("metadata:", meta_path)
    print("cache:", cache_summary(cache))
    print("cuda:", cuda_memory_summary())
    print("train_dates:", train_dates[0], "->", train_dates[-1], len(train_dates))
    print("valid_dates:", valid_dates[0], "->", valid_dates[-1], len(valid_dates))
    print("rebalance_days:", args.rebalance_days or args.label_horizon)
    print("exports:", {key: str(path) for key, path in paths.items()})

    cols = [
        "expression",
        "train_abs_rank_ic",
        "train_ic_win_rate",
        "train_ndcg_at_k",
        "valid_abs_rank_ic",
        "valid_ic_win_rate",
        "valid_top_excess_ann",
        "passed_validation",
        "error",
    ]
    existing_cols = [col for col in cols if col in final_df.columns]
    print(final_df.head(8)[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
