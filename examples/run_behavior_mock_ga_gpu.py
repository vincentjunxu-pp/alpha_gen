from __future__ import annotations

# %%
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.behavior_gen.ga import (
    BehaviorGAConfig,
    BehaviorValidationCriteria,
    NSGA_MODE_RIR_LONG_RIR_NDCG,
    evaluated_behavior_to_frame,
    export_behavior_search_result,
    run_behavior_ga_search,
    select_validation_population,
    validate_behavior_population,
)
from alpha_gen.behavior_gen.gene import load_behavior_field_rules
from alpha_gen.behavior_gen.torch_backend import (
    NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    BehaviorTorchContext,
)
from alpha_gen.core.gene import load_field_rules
from alpha_gen.core.preprocess import build_transform_cache, cache_summary, load_panel
from alpha_gen.core.torch_backend import cuda_memory_summary
from alpha_gen.core.utils import get_rolling_windows


# %%
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "panels" / "mock_behavior_daily.parquet"
META_PATH = ROOT / "data" / "metadata" / "fixtures" / "mock_behavior_metadata.json"
RESULT_DIR = ROOT / "artifacts" / "results"

LABEL_COL = "label_20d"
TRADEABLE_COL = "is_tradeable"
INDUSTRY_COL = "industry_code"

LABEL_HORIZON = 20
REBALANCE_FREQ = 20

TRAIN_START_DATE = "20230718"
VALID_START_DATE = "20241231"
VALID_END_DATE = "20250422"
WINDOW_STRIDE = 120000

POPULATION_SIZE = 32
GENERATIONS = 2
RANDOM_SEED = 20260529
NDCG_TOP_FRACTION = 0.20
MIN_COVERAGE = 0.50

DEVICE = "cuda"
CACHE_ON_DEVICE = True
SHOW_PROGRESS = True
EXPORT_PREFIX = "mock_behavior_ga_gpu"


# %%
field_rules = load_field_rules(META_PATH)
behavior_rules = load_behavior_field_rules(META_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

size_field = str(metadata.get("size_field", "barra_size"))
barra_style_fields = tuple(metadata.get("barra_style_fields", ()))

panel = load_panel(DATA_PATH)
cache = build_transform_cache(
    panel,
    field_rules,
    label_col=LABEL_COL,
    tradeable_col=TRADEABLE_COL,
    industry_col=INDUSTRY_COL,
    extra_current_fields=[size_field, *barra_style_fields],
    show_progress=SHOW_PROGRESS,
)

usable_dates = cache.label.index[:-LABEL_HORIZON]
windows = get_rolling_windows(
    usable_dates,
    train_start_date=TRAIN_START_DATE,
    test_start_date=VALID_START_DATE,
    stride=WINDOW_STRIDE,
    horizon=LABEL_HORIZON,
)
if not windows:
    raise RuntimeError("no train/validation window could be built")

train_dates, valid_dates = windows[0]
if VALID_END_DATE is not None:
    valid_dates = valid_dates[valid_dates < VALID_END_DATE]
if valid_dates.empty:
    raise RuntimeError("validation window is empty")

print("size_field:", size_field)
print("barra_style_fields:", barra_style_fields)
print("train:", train_dates[0], "->", train_dates[-1], len(train_dates))
print("valid:", valid_dates[0], "->", valid_dates[-1], len(valid_dates))
print("cache:", cache_summary(cache))


# %%
config = BehaviorGAConfig(
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    crossover_prob=0.85,
    mutation_prob=0.25,
    random_seed=RANDOM_SEED,
    ndcg_k=None,
    ndcg_top_fraction=NDCG_TOP_FRACTION,
    nsga_objective_mode=NSGA_MODE_RIR_LONG_RIR_NDCG,
    min_coverage=MIN_COVERAGE,
    size_field=size_field,
    neutralization_mode=NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    require_cuda=True,
    show_progress=SHOW_PROGRESS,
)

torch_context = BehaviorTorchContext(
    cache=cache,
    behavior_field_rules=behavior_rules,
    device=DEVICE,
    cache_on_device=CACHE_ON_DEVICE,
    barra_style_fields=barra_style_fields,
)

result = run_behavior_ga_search(
    ctx=torch_context,
    train_dates=train_dates,
    config=config,
)

# Step 1: Evaluate ALL history genes on the validation set.
# Returns the full pool — every gene now carries valid_metrics.
validated_all = validate_behavior_population(
    evaluated_population=result.history,
    ctx=torch_context,
    valid_dates=valid_dates,
    criteria=BehaviorValidationCriteria(
        min_abs_rank_ic=0.01,
        min_ic_win_rate=0.52,
        min_coverage=MIN_COVERAGE,
    ),
    ndcg_k=config.ndcg_k,
    ndcg_top_fraction=config.ndcg_top_fraction,
    label_horizon=LABEL_HORIZON,
    rebalance_freq=REBALANCE_FREQ,
    neutralization_mode=config.neutralization_mode,
    size_field=config.size_field,
    show_progress=SHOW_PROGRESS,
    max_validation_genes=config.population_size * 3,
    cuda_clear_every=1,
)

# Step 2: NSGA‑II on validation metrics to pick the final population.
final_population = select_validation_population(
    validated_all,
    population_size=config.population_size,
    nsga_objective_mode=config.nsga_objective_mode,
)
result.final_population = final_population

paths = export_behavior_search_result(
    result,
    RESULT_DIR,
    prefix=EXPORT_PREFIX,
)

# Full result table: ALL validated genes with train + valid metrics
full_df = evaluated_behavior_to_frame(
    validated_all,
    objective_mode=config.nsga_objective_mode,
)
# Selected subset
final_df = evaluated_behavior_to_frame(
    result.final_population,
    objective_mode=config.nsga_objective_mode,
)

print("cuda:", cuda_memory_summary())
print("exports:", {key: str(path) for key, path in paths.items()})

cols = [
    "gene_mode",
    "gene_combiner",
    "gene_direction_policy",
    "train_rir",
    "train_long_rir",
    "train_ndcg_k",
    "valid_rir",
    "valid_long_rir",
    "valid_ndcg_k",
    "passed_validation",
]
print(final_df.head(10)[cols].to_string(index=False))
