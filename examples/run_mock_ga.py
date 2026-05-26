from __future__ import annotations

import json
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
from alpha_gen.core.utils import get_rolling_windows


# ---------------------------------------------------------------------------
# Small local run for debugging the full pipeline.
#
# This is not the report-scale setting. It uses a tiny population so we can
# verify data loading, caching, factor calculation, training evaluation,
# NSGA-II selection, validation filtering and CSV export on a laptop.
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mock_tmt_daily.parquet"
META_PATH = ROOT / "data" / "mock_tmt_metadata.json"
RESULT_DIR = ROOT / "results"


def main() -> None:
    field_rules = load_field_rules(META_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    size_field = str(metadata.get("size_field", "barra_size"))
    barra_style_fields = tuple(metadata.get("barra_style_fields", ()))
    panel = load_panel(DATA_PATH)
    cache = build_transform_cache(panel, field_rules, extra_current_fields=[size_field, *barra_style_fields])
    usable_dates = cache.label.index[:-20]
    windows = get_rolling_windows(
        usable_dates,
        train_start_date=usable_dates[-(400 + 20 + 120)],
        test_start_date=usable_dates[-120],
        stride=120,
        horizon=20,
    )
    train_dates, valid_dates = windows[0]

    # Laptop smoke-test parameters. Increase these after the logic is stable:
    # e.g. population_size=200, generations=5 for a stronger local dry run.
    config = GAConfig(
        population_size=24,
        generations=1,
        crossover_prob=0.85,
        mutation_prob=0.25,
        random_seed=20260428,
        size_field=size_field,
        industry_scope="all",
    )

    result = run_ga_search(
        cache=cache,
        field_rules=field_rules,
        train_dates=train_dates,
        config=config,
    )
    validate_population(
        evaluated_population=result.final_population,
        cache=cache,
        valid_dates=valid_dates,
        criteria=ValidationCriteria(
            # Mock data is not expected to reproduce report-level economics.
            # Keep thresholds mild so the script demonstrates the filter table.
            min_abs_rank_ic=0.01,
            min_ic_win_rate=0.52,
            min_top_excess_ann=0.00,
            min_coverage=0.30,
        ),
        ndcg_k=config.ndcg_k,
        ndcg_top_fraction=config.ndcg_top_fraction,
        label_horizon=20,
        rebalance_freq=20,
        size_field=config.size_field,
        industry_scope=config.industry_scope,
    )

    paths = export_search_result(result, RESULT_DIR, prefix="mock_ga")
    final_df = evaluated_to_frame(result.final_population)

    print("cache:", cache_summary(cache))
    print("train_dates:", train_dates[0], "->", train_dates[-1], len(train_dates))
    print("valid_dates:", valid_dates[0], "->", valid_dates[-1], len(valid_dates))
    print("exports:", {key: str(path) for key, path in paths.items()})
    print(final_df.head(8)[
        [
            "expression",
            "train_abs_rank_ic",
            "train_ic_win_rate",
            "train_ndcg_at_k",
            "valid_abs_rank_ic",
            "valid_ic_win_rate",
            "valid_top_excess_ann",
            "passed_validation",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
