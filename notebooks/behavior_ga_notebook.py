# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: pytorch
# ---

# %% [markdown]
# # 行为金融学遗传算法因子挖掘 — Notebook 版
#
# ## 流程概览
#
# ```
# 数据加载 → 缓存构建 → 日期划分 → GA训练(NSGA-II on train metrics, 同时顺手算 valid)
#                                                  ↓
#                                         全部历史基因(父代+子代, 每行已有 train+valid)
#                                                  ↓
#                               Step 1: 去重(gene_key) → 二次去重(expression)
#                                                  ↓
#                               Step 2: NSGA-II on validation metrics → 精选子集
#                                                  ↓
#                                         导出: 全量表 + 精选表 + NSGA rank + 配置JSON
# ```
#
# ## 关键配置决策
#
# | 配置项 | 选项 | 说明 |
# |--------|------|------|
# | 中性化 | `raw_full_barra_industry` | 不改变因子值；中性化指标单独计算（10 Barra + 行业同时回归取残差） |
# |        | `size_then_industry` | 因子先对市值回归，再行业内去均值 |
# | NSGA目标 | `rir_long_rir_neutralized_rir` **(默认)** | 三目标：Rank IC IR / Long-side Rank IC IR / Neutralized Rank IC IR |
# |          | `rir_long_rir_ndcg` | 三目标：Rank IC IR / Long-side Rank IC IR / NDCG@k |
# |          | `rir_long_rir` | 两目标：Rank IC IR / Long-side Rank IC IR |

# %% [markdown]
# ## Cell 1: 环境设置与导入

# %%
import gc
import json
import os
import sys
from pathlib import Path

# 必须在任何 torch import 之前设置，降低 CUDA caching allocator
# 的碎片化程度。Windows 上 expandable_segments 不可用（需 Linux
# 原生 cudaMallocAsync），改用 max_split_size_mb + GC 阈值组合。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.7",
)

import numpy as np
import pandas as pd
import torch

_this_file = Path(__file__).resolve()
WORKSPACE_ROOT = _this_file.parents[1]  # alpha_gen/ 目录
if str(WORKSPACE_ROOT.parent) not in sys.path:  # E:/实习/ 目录
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

from alpha_gen.behavior_gen.ga import (
    BehaviorGAConfig,
    BehaviorValidationCriteria,
    NSGA_MODE_RIR_LONG_RIR,
    NSGA_MODE_RIR_LONG_RIR_NDCG,
    NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR,
    NSGA_OBJECTIVE_MODES,
    EvaluatedBehaviorGene,
    evaluated_behavior_to_frame,
    export_behavior_search_result,
    run_behavior_ga_search,
    select_validation_population,
    selected_behavior_rank_table,
    validate_behavior_population,
)
from alpha_gen.behavior_gen.gene import (
    BehaviorGene,
    SlotGene,
    ConditionGene,
    MODE_REGISTRY,
    describe_gene,
    describe_gene_formula,
    gene_key,
    load_behavior_field_rules,
    validate_gene,
)
from alpha_gen.behavior_gen.sampler import (
    BehaviorSamplerConfig,
    random_gene,
)
from alpha_gen.behavior_gen.torch_backend import (
    BEHAVIOR_NEUTRALIZATION_MODES,
    NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    NEUTRALIZATION_SIZE_THEN_INDUSTRY,
    BehaviorTorchContext,
    calculate_behavior_factor_tensor,
    validate_neutralization_requirements,
)
from alpha_gen.core.gene import load_field_rules
from alpha_gen.core.preprocess import (
    build_transform_cache,
    cache_summary,
    load_panel,
)
from alpha_gen.core.torch_backend import cuda_memory_summary
from alpha_gen.core.utils import get_rolling_windows

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## Cell 2: 数据加载与缓存构建
#
# 替换为你自己的数据路径即可切换到真实数据。

# %%
ROOT = WORKSPACE_ROOT  # 或写死: Path("E:/实习/alpha_gen")
DATA_PATH = ROOT / "data" / "panels" / "mock_behavior_daily.parquet"
META_PATH = ROOT / "data" / "metadata" / "fixtures" / "mock_behavior_metadata.json"
RESULT_DIR = ROOT / "artifacts" / "results"

LABEL_COL = "label_20d"
TRADEABLE_COL = "is_tradeable"
INDUSTRY_COL = "industry_code"

print("Loading metadata ...")
field_rules = load_field_rules(META_PATH)
behavior_rules = load_behavior_field_rules(META_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

size_field = str(metadata.get("size_field", "barra_size"))
barra_style_fields = tuple(metadata.get("barra_style_fields", ()))
print(f"  size_field: {size_field}")
print(f"  barra_style_fields ({len(barra_style_fields)}): {barra_style_fields[:5]}...")
print(f"  behavior field rules: {len(behavior_rules)} 字段")
print(f"  behavior modes: {len(MODE_REGISTRY)} 个")

print("\nBuilding transform cache ...")
panel = load_panel(DATA_PATH)
cache = build_transform_cache(
    panel, field_rules,
    label_col=LABEL_COL,
    tradeable_col=TRADEABLE_COL,
    industry_col=INDUSTRY_COL,
    extra_current_fields=[size_field, *barra_style_fields],
    show_progress=True,
)
print(cache_summary(cache))

# %% [markdown]
# ## Cell 3: 训练/验证日期划分

# %%
LABEL_HORIZON = 20

TRAIN_START_DATE = "20230718"
VALID_START_DATE = "20241231"
VALID_END_DATE = "20250422"
WINDOW_STRIDE = 120000

usable_dates = cache.label.index[:-LABEL_HORIZON]
windows = get_rolling_windows(
    usable_dates,
    train_start_date=TRAIN_START_DATE,
    test_start_date=VALID_START_DATE,
    stride=WINDOW_STRIDE,
    horizon=LABEL_HORIZON,
)

if not windows:
    raise RuntimeError("无法构建训练/验证窗口，请检查日期范围")

train_dates, valid_dates = windows[0]
if VALID_END_DATE is not None:
    valid_dates = valid_dates[valid_dates < VALID_END_DATE]

print(f"训练集: {train_dates[0].date()} → {train_dates[-1].date()}  ({len(train_dates)} 天)")
print(f"验证集: {valid_dates[0].date()} → {valid_dates[-1].date()}  ({len(valid_dates)} 天)")
print(f"交易日总数: {len(cache.label.index)}")

# %% [markdown]
# ## Cell 4: GA 配置与 Torch 上下文
#
# | 参数 | 建议值 | 说明 |
# |------|--------|------|
# | `population_size` | 500-5000 | 种群大小 |
# | `generations` | 3-10 | 迭代代数 |
# | `neutralization_mode` | `raw_full_barra_industry` | 中性化策略 |
# | `nsga_objective_mode` | `rir_long_rir_neutralized_rir` | NSGA 优化目标 |

# %%
POPULATION_SIZE = 16       # 调试用小种群，真实数据改成 2000-5000
GENERATIONS = 2            # 调试用 2 代，真实数据改成 5-10
RANDOM_SEED = 20260529
NDCG_TOP_FRACTION = 0.20
MIN_COVERAGE = 0.30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_ON_DEVICE = True
SHOW_PROGRESS = True
EXPORT_PREFIX = "behavior_ga_notebook"

config = BehaviorGAConfig(
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    crossover_prob=0.85,
    mutation_prob=0.25,
    random_seed=RANDOM_SEED,
    ndcg_k=None,
    ndcg_top_fraction=NDCG_TOP_FRACTION,
    nsga_objective_mode=NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR,
    min_coverage=MIN_COVERAGE,
    size_field=size_field,
    neutralization_mode=NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    require_cuda=(DEVICE == "cuda"),
    show_progress=SHOW_PROGRESS,
)

torch_context = BehaviorTorchContext(
    cache=cache,
    behavior_field_rules=behavior_rules,
    device=DEVICE,
    cache_on_device=CACHE_ON_DEVICE,
    barra_style_fields=barra_style_fields,
)

err = validate_neutralization_requirements(
    config.neutralization_mode,
    barra_style_fields=torch_context.barra_style_fields,
    has_industry=cache.industry is not None,
)
if err:
    raise ValueError(f"中性化配置错误: {err}")

# Field tensor 缓存由 BehaviorTorchContext.max_cache_mb 限制（默认 4 GiB），
# LRU 淘汰按需填充，无需手动管理。


print(f"Device: {torch_context.device}")
print(f"Neutralization: {config.neutralization_mode}")
print(f"NSGA objectives: {NSGA_OBJECTIVE_MODES[config.nsga_objective_mode]}")
print(f"Population: {config.population_size} × {config.generations} generations")

# %% [markdown]
# ## Cell 5: 运行 GA 训练

# %%
print("开始 GA 训练 ...")
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

result = run_behavior_ga_search(
    ctx=torch_context,
    train_dates=train_dates,
    config=config,
    valid_dates=valid_dates,   # ← 训练时顺便算好验证指标，省掉第二遍 GPU 计算
)

print(f"\n训练完成:")
print(f"  历史评估总数: {len(result.history)}")
print(f"  训练 NSGA 选出: {len(result.final_population)} 个基因")
print(f"  去重后唯一基因: {len({gene_key(item.gene) for item in result.history})}")
n_with_valid = sum(1 for item in result.history if item.valid_score is not None)
print(f"  已附带验证指标: {n_with_valid}/{len(result.history)}")

# %% [markdown]
# ## Cell 6: NSGA 精选 + 导出
#
# 训练阶段 NSGA-II 使用 train 指标（generation 内部 selection），验证指标已在
# ``evaluate_behavior_gene_on_train`` 中一并计算好。这里用 **valid 指标**对全部
# 历史基因重新做 NSGA-II，选出在验证集上 Pareto 最优的精选子集。

# %%
# ----- 去重：同一基因跨代只保留一条 -----
seen = set()
validated_all = []
for item in result.history:
    key = gene_key(item.gene)
    if key not in seen and item.valid_score is not None and not item.error:
        seen.add(key)
        validated_all.append(item)
print(f"去重验证池 (gene_key): {len(validated_all)} 个基因")

# ----- 二次去重：按 expression 去语义重复，保留 train_rir 最高的那条 -----
from alpha_gen.behavior_gen.gene import describe_gene
expr_best: dict[str, tuple[float, object]] = {}
for item in validated_all:
    expr = describe_gene(item.gene)
    existing = expr_best.get(expr)
    if existing is None or item.train_score.rank_ic_ir > existing[0]:
        expr_best[expr] = (item.train_score.rank_ic_ir, item)
validated_all = [item for _, item in expr_best.values()]
print(f"二次去重 (expression): {len(validated_all)} 个基因")

# ----- NSGA-II 在验证指标上精选 -----
final_population = select_validation_population(
    validated_all,
    population_size=config.population_size,
    nsga_objective_mode=config.nsga_objective_mode,
    barra_exposure_lambda=config.barra_exposure_lambda,
)
result.final_population = final_population
print(f"NSGA 精选: {len(final_population)} 个基因")

# ----- 导出全量表（含 train + valid 指标）-----
full_df = evaluated_behavior_to_frame(
    validated_all,
    objective_mode=config.nsga_objective_mode,
)
FULL_TABLE_PATH = RESULT_DIR / f"{EXPORT_PREFIX}_all_validated.csv"
full_df.to_csv(FULL_TABLE_PATH, index=False, encoding="utf-8-sig")
print(f"\n全量验证表: {FULL_TABLE_PATH}")
print(f"  {len(full_df)} 行 × {len(full_df.columns)} 列")

# ----- 导出精选表 + config + NSGA rank -----
paths = export_behavior_search_result(result, RESULT_DIR, prefix=EXPORT_PREFIX)
print(f"\n精选导出:")
for key, path in paths.items():
    print(f"  {key}: {path}")

# ----- 精选表 DataFrame -----
final_df = evaluated_behavior_to_frame(
    result.final_population,
    objective_mode=config.nsga_objective_mode,
)

# ----- 查看精选表关键列 -----
key_cols = [
    "expression",
    "formula",
    "gene_mode",
    "gene_combiner",
    "gene_direction_policy",
    "train_rir",
    "train_long_rir",
    "train_neutralized_rir",
    "train_ndcg_k",
    "train_ric",
    "train_win_rate",
    "train_coverage",
    "valid_rir",
    "valid_long_rir",
    "valid_neutralized_rir",
    "valid_ndcg_k",
    "valid_ric",
    "valid_win_rate",
    "valid_coverage",
]
existing_cols = [c for c in key_cols if c in final_df.columns]
final_df[existing_cols].head(10)

# %% [markdown]
# ## Cell 8: 因子值还原（需求5）

# %%
row = final_df.iloc[0]
print(f"Mode: {row['gene_mode']}  Combiner: {row['gene_combiner']}")
print(f"Expression: {row['expression'][:120]}...")
print(f"\nFormula:\n{row['formula']}")

# 从序列化数据重建 BehaviorGene
gene_dict = {
    "mode": row["gene_mode"],
    "combiner": row["gene_combiner"],
    "slots": json.loads(row["gene_slots"]),
    "conditions": json.loads(row.get("gene_conditions", "[]")),
    "direction_policy": row.get("gene_direction_policy", "fixed"),
    "version": int(row.get("gene_version", 1)),
}
reconstructed_gene = BehaviorGene.from_dict(gene_dict)
print(f"\nGene key: {gene_key(reconstructed_gene)}")

# 还原因子值
factor_tensor = calculate_behavior_factor_tensor(
    reconstructed_gene, torch_context,
    neutralization_mode=config.neutralization_mode,
    size_field=config.size_field,
)
factor_df = torch_context.tensor_to_frame(factor_tensor)
valid_factor = factor_df.loc[factor_df.index.isin(valid_dates)]
print(f"\n验证集因子值: shape={valid_factor.shape}")
print(f"  coverage: {(valid_factor.notna().sum().sum() / valid_factor.size):.2%}")
print(f"  mean: {valid_factor.mean().mean():.4f}")
print(f"  std:  {valid_factor.std().mean():.4f}")
valid_factor.iloc[:5, :5]

# %% [markdown]
# ## Cell 9: NSGA-II 诊断表

# %%
rank_df = selected_behavior_rank_table(
    result.final_population,
    objective_mode=config.nsga_objective_mode,
)
print(f"Pareto 前沿分布:")
print(rank_df[["front", "crowding_distance"]].describe())
print(f"\n各 front 基因数:")
print(rank_df["front"].value_counts().sort_index())

frontier = rank_df[rank_df["front"] == 0]
print(f"\nPareto 前沿 ({len(frontier)} 个基因):")
frontier_cols = ["expression", "train_rir", "train_long_rir", "valid_rir", "valid_long_rir"]
fc = [c for c in frontier_cols if c in frontier.columns]
frontier[fc]

# %% [markdown]
# ## Cell 10: GPU 内存监控

# %%
if DEVICE == "cuda":
    print(cuda_memory_summary())
    print(f"\nAllocated:  {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved:   {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## Cell 11: 最终结果表路径汇总

# %%
print("=" * 60)
print("最终结果表路径汇总")
print("=" * 60)
print(f"\n📊 全量验证表（全部基因 train + valid 指标）:")
print(f"   {FULL_TABLE_PATH}")
print(f"\n📊 精选表（NSGA 选中）:")
print(f"   {paths['final_population']}")
print(f"\n📊 NSGA 排名诊断表:")
print(f"   {paths['rank_table']}")
print(f"\n📊 GA 配置:")
print(f"   {paths['config']}")
print(f"\n📊 历史全部评估记录:")
print(f"   {paths['history']}")

# %% [markdown]
# ## 附录 A: 随机基因公式查看

# %%
rng = np.random.default_rng(42)
sample_gene = random_gene(behavior_rules, rng, config=BehaviorSamplerConfig())
print("描述:", describe_gene(sample_gene))
print("\n公式:", describe_gene_formula(sample_gene))
errors = validate_gene(sample_gene, behavior_rules)
print(f"\n{'✅ 合法' if not errors else f'⚠ {errors}'}")

# %% [markdown]
# ## 附录 B: 生产级配置 (pop=2000, gen=5)

# %%
production_config = BehaviorGAConfig(
    population_size=2000,
    generations=5,
    crossover_prob=0.85, mutation_prob=0.25,
    random_seed=20260601,
    ndcg_k=None, ndcg_top_fraction=0.10,
    nsga_objective_mode=NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR,
    min_coverage=0.30,
    size_field=size_field,
    neutralization_mode=NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    require_cuda=True, show_progress=True,
)
print(f"生产配置: pop={production_config.population_size} × {production_config.generations} 代")
print(f"NSGA 目标: {NSGA_OBJECTIVE_MODES[production_config.nsga_objective_mode]}")
print(f"建议 validate_behavior_population 参数:")
print(f"  max_validation_genes = {production_config.population_size * 3}")

# %% [markdown]
# ## 附录 C: 切换中性化/NSGA 模式

# %%
print("可用中性化模式:", BEHAVIOR_NEUTRALIZATION_MODES)
print("可用 NSGA 目标:", {k: v for k, v in NSGA_OBJECTIVE_MODES.items()})

# 示例 1: 切换到 size_then_industry + 两目标 NSGA
config_alt = BehaviorGAConfig(
    population_size=8, generations=1,
    neutralization_mode=NEUTRALIZATION_SIZE_THEN_INDUSTRY,
    nsga_objective_mode=NSGA_MODE_RIR_LONG_RIR,
    min_coverage=0.30, size_field=size_field,
    require_cuda=False, show_progress=False,
)
err = validate_neutralization_requirements(
    config_alt.neutralization_mode,
    barra_style_fields=torch_context.barra_style_fields,
    has_industry=cache.industry is not None,
)
print(f"备选1: neutralization={config_alt.neutralization_mode}, "
      f"objectives={NSGA_OBJECTIVE_MODES[config_alt.nsga_objective_mode]} → "
      f"{'✅ 合法' if not err else f'⚠ {err}'}")

# 示例 2: raw_full_barra_industry + NDCG 目标（不优化 neutralized_rir）
config_alt2 = BehaviorGAConfig(
    population_size=8, generations=1,
    neutralization_mode=NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    nsga_objective_mode=NSGA_MODE_RIR_LONG_RIR_NDCG,
    min_coverage=0.30, size_field=size_field,
    require_cuda=False, show_progress=False,
)
print(f"备选2: neutralization={config_alt2.neutralization_mode}, "
      f"objectives={NSGA_OBJECTIVE_MODES[config_alt2.nsga_objective_mode]}")
