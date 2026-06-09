"""Runner for typed tree GP behavior-finance factor search.

Usage (from E:/实习):
  D:/Anaconda/envs/pytorch/python.exe -m alpha_gen.behavior_tree_gp.run_typed_tree_gp \
    --data-path alpha_gen/data/panels/mock_behavior_daily.parquet \
    --metadata-path alpha_gen/data/metadata/fixtures/mock_behavior_metadata.json \
    --population-size 8 --generations 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.behavior_gen.gene import load_behavior_field_rules
from alpha_gen.behavior_gen.torch_backend import BehaviorTorchContext
from alpha_gen.core.gene import load_field_rules
from alpha_gen.core.preprocess import build_transform_cache, cache_summary, load_panel

from .ga import (
    TreeGAConfig,
    TreeValidationCriteria,
    evaluated_tree_to_frame,
    export_tree_search_result,
    run_tree_ga_search,
    validate_tree_population,
)
from .typed_sampler import TypedTreeSamplerConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "panels" / "mock_behavior_daily.parquet"
DEFAULT_META = ROOT / "data" / "metadata" / "fixtures" / "mock_behavior_metadata.json"
DEFAULT_RESULT = ROOT / "artifacts" / "results" / "typed_tree_gp"


def make_train_valid_dates(
    dates: pd.DatetimeIndex,
    *,
    label_horizon: int,
    train_days: int,
    valid_days: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    usable_dates = pd.DatetimeIndex(dates[:-label_horizon])
    if len(usable_dates) <= label_horizon + valid_days + 5:
        raise ValueError("not enough dates to build train/validation windows")
    valid_len = min(valid_days, max(1, len(usable_dates) // 5))
    valid_dates = pd.DatetimeIndex(usable_dates[-valid_len:])
    valid_start_idx = int(dates.get_indexer([valid_dates[0]])[0])
    train_end_idx = valid_start_idx - label_horizon - 1
    if train_end_idx < 0:
        raise ValueError("label_horizon leaves no train dates before validation")
    train_start_idx = 0 if train_days <= 0 else max(0, train_end_idx - train_days + 1)
    train_dates = pd.DatetimeIndex(dates[train_start_idx : train_end_idx + 1])
    if train_dates.empty:
        raise ValueError("training window is empty")
    return train_dates, valid_dates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Typed tree GP for behavior-finance factor mining.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_META)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--prefix", default="typed_tree_gp")
    parser.add_argument("--label-col", default="label_20d")
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--industry-col", default="industry_code")
    parser.add_argument("--label-horizon", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--valid-days", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260529)
    parser.add_argument("--ndcg-top-fraction", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--max-slot-depth", type=int, default=3)
    parser.add_argument("--max-total-depth", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--parsimony-coefficient", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true", help="allow running on CPU (disables the CUDA requirement)")
    parser.add_argument("--cache-on-device", action="store_true", default=False)
    parser.add_argument("--no-cache-on-device", dest="cache_on_device", action="store_false")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    size_field = str(metadata.get("size_field", "barra_size"))
    barra_style_fields = tuple(metadata.get("barra_style_fields", ()))

    core_rules = load_field_rules(args.metadata_path)
    behavior_rules = load_behavior_field_rules(args.metadata_path)
    panel = load_panel(args.data_path)
    cache = build_transform_cache(
        panel, core_rules,
        label_col=args.label_col,
        tradeable_col=args.tradeable_col,
        industry_col=args.industry_col,
        extra_current_fields=[size_field, *barra_style_fields],
        show_progress=args.show_progress,
    )
    ctx = BehaviorTorchContext(
        cache=cache,
        behavior_field_rules=behavior_rules,
        device=args.device,
        cache_on_device=args.cache_on_device,
        barra_style_fields=barra_style_fields,
    )

    train_dates, valid_dates = make_train_valid_dates(
        cache.label.index,
        label_horizon=args.label_horizon,
        train_days=args.train_days,
        valid_days=args.valid_days,
    )

    sampler_config = TypedTreeSamplerConfig(
        max_slot_depth=args.max_slot_depth,
        max_total_depth=args.max_total_depth,
        max_nodes=args.max_nodes,
    )
    config = TreeGAConfig(
        population_size=args.population_size,
        generations=args.generations,
        random_seed=args.random_seed,
        ndcg_top_fraction=args.ndcg_top_fraction,
        min_coverage=args.min_coverage,
        sampler_config=sampler_config,
        size_field=size_field,
        require_cuda=(not args.allow_cpu and args.device == "cuda"),
        show_progress=args.show_progress,
        parsimony_coefficient=args.parsimony_coefficient,
    )

    print(f"device: {ctx.device}")
    print(f"cache: {cache_summary(cache)}")
    print(f"train: {len(train_dates)} days, valid: {len(valid_dates)} days")
    print(f"constraints: max_slot_depth={args.max_slot_depth}, max_total_depth={args.max_total_depth}, max_nodes={args.max_nodes}")

    result = run_tree_ga_search(ctx, train_dates, config)

    validate_tree_population(
        result.final_population, ctx, valid_dates,
        criteria=TreeValidationCriteria(
            min_abs_rank_ic=0.01,
            min_ic_win_rate=0.52,
            min_top_excess_ann=0.00,
            min_coverage=args.min_coverage,
        ),
        ndcg_k=config.ndcg_k,
        ndcg_top_fraction=config.ndcg_top_fraction,
        label_horizon=args.label_horizon,
        rebalance_freq=args.label_horizon,
        neutralize_size=config.neutralize_size,
        neutralize_industry=config.neutralize_industry,
        size_field=config.size_field,
        show_progress=args.show_progress,
    )

    paths = export_tree_search_result(result, args.result_dir, prefix=args.prefix)

    final_df = evaluated_tree_to_frame(result.final_population)
    print(f"train: {train_dates[0]} -> {train_dates[-1]} ({len(train_dates)} days)")
    print(f"valid: {valid_dates[0]} -> {valid_dates[-1]} ({len(valid_dates)} days)")
    print(f"exports: {', '.join(str(p) for p in paths.values())}")

    display_cols = [
        "tree_mode", "tree_combiner", "tree_depth", "tree_size",
        "train_rank_ic_ir", "train_ndcg_at_k", "train_neutralized_icir",
        "valid_rank_ic_ir", "valid_top_excess_ann", "passed_validation",
    ]
    display_cols = [c for c in display_cols if c in final_df.columns]
    if not final_df.empty:
        print(final_df.head(8)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
