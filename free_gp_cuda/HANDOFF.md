# free_gp_cuda 交接文档

## 当前状态

`alpha_gen/free_gp_cuda/` 已经形成一个隔离的 CUDA 自由 GP 因子挖掘闭环：

1. `CudaFactorContext` 将 `MultiIndex(["Datetime", "Contract"])` 长表字段懒加载为 `torch.Tensor[T, N]`。
2. `operators.py` 提供自包含 CUDA 算子库。
3. `registry.py` 登记可搜索算子元信息。
4. `program.py` 定义类型安全的表达式树。
5. `evaluator.py` 将表达式树执行成因子矩阵。
6. `neutralizer.py` 先用全部 Barra 字段做多元截面回归取残差，再对残差做行业中性化。
7. `evaluation_metrics.py` 在 GPU 上计算 IC、RankIC、NDCG、覆盖率、换手、稳定性和中性化 ICIR。
8. `scorer.py` 封装 program/context/metrics 的评分入口，并支持 raw/neutralized 因子取数。
9. `generator.py` 实现随机树生成、子树变异、同类型子树交叉和去重。
10. `nsga2.py` 实现所有目标最大化的 NSGA-II 精英选择。
11. `search.py` 实现初始化种群、评分缓存、子代生成、父子合并、NSGA-II 选择、验证集复评和结果导出。

生产代码没有导入旧的 `alpha_gen.core.torch_backend`、`behavior_gen`、`core.preprocess`、`TransformCache` 或 `build_transform_cache`。

## 数据约定

输入数据是长表 `pandas.DataFrame`：

- index 必须是 `MultiIndex(["Datetime", "Contract"])`
- 默认标签列：`label_20d`
- 默认可交易列：`is_tradeable`
- 默认行业列：`industry_code`
- 搜索字段默认取所有数值列，并排除 label/tradeable/industry

**字段排除机制（2026-06-02 更新）：**

`CudaFactorContext` 提供两个层次的字段排除：

```python
ctx = CudaFactorContext(
    panel,
    exclude_fields=("label_5d", "label_10d"),  # 精确排除
    exclude_prefixes=("barra_", "label_"),      # 前缀排除
)
```

- `exclude_fields` 和 `exclude_prefixes` 在 **自动发现** 和 **手动指定 `candidate_fields`** 两种模式下均生效
- 例如设置 `exclude_prefixes=("barra_", "label_")` 后，`barra_size`、`barra_beta`、`label_5d` 等均不会出现在 `searchable_fields` 中
- 但通过 `ScorerConfig.style_fields` 指定的 Barra 字段仍然可以参与中性化计算（不依赖 searchable_fields）

核心张量形状统一是：

```text
[T, N] = [Datetime, Contract]
```

## 主要入口

最小搜索示例：

```python
from alpha_gen.free_gp_cuda import (
    CudaFactorContext,
    FreeGPSearchConfig,
    ProgramGeneratorConfig,
    ScorerConfig,
    run_free_gp_search,
    evaluated_to_frame,
)

ctx = CudaFactorContext(panel, device="cuda")

config = FreeGPSearchConfig(
    population_size=80,
    generations=3,
    random_seed=1,
    min_coverage=0.30,
    generator_config=ProgramGeneratorConfig(max_depth=5, max_size=64),
    scorer_config=ScorerConfig(
        ndcg_top_fraction=0.10,
        min_cross_section_size=3,
        style_fields=("barra_size", "barra_beta", "barra_momentum"),
        neutralize_industry=True,
        neutralize_styles=True,
    ),
)

result = run_free_gp_search(
    ctx,
    train_dates=ctx.dates[:500],
    candidate_fields=ctx.searchable_fields,
    config=config,
)

df = evaluated_to_frame(result.final_population, include_program_json=True)
```

单个 program 评分：

```python
from alpha_gen.free_gp_cuda import ProgramScorer, Program, FieldNode

scorer = ProgramScorer(ctx)
program = Program(FieldNode("feature_a"))
scored = scorer.score_program(program, dates=ctx.dates[:200], raise_errors=True)
print(scored.score.to_dict())
```

根据表达式取因子值：

```python
raw_factor = scorer.factor_values(program, view="raw")
neutralized_factor = scorer.factor_values(program, view="neutralized")
```

`view="neutralized"` 不改变表达式树，只是在表达式算出的 raw factor 后做 post-process 中性化。

## IC/IR 指标与 alpha_factory 的对齐 (2026-06-03)

free_gp_cuda 的 IC 计算已对齐 `alpha_factory.factor.evaluation.ic.ICAnalyzer`，默认
行为向后兼容，新增 `mask_factor_by_tradeable=False` 可精确复现 `corrwith(method="spearman")`。

### 差异根源

| 差异 | alpha_factory (`ic.py`) | free_gp_cuda (默认) | free_gp_cuda (对齐) |
|------|------------------------|--------------------|--------------------|
| RankIC rank 域 | factor 全截面 rank（含不可交易标的） | 仅可交易标的 rank | 全截面 rank（`daily_rank_ic` 改为独立 rank） |
| Label tradeable | 只 mask label | factor + label 均 mask | 只 mask label（`mask_factor_by_tradeable=False`） |
| min_cross_section_size | 无下限（pandas 默认 ≥2） | ≥3 | ≥2（对齐 pandas） |

对齐模式使用：

```python
scorer = ProgramScorer(
    ctx,
    ScorerConfig(
        mask_inputs_by_tradeable=False,   # 不对输入字段做 tradeable 遮罩
        mask_factor_by_tradeable=False,   # 不在评估时遮罩 factor
        tradeable_only=False,             # 求值器也不过滤输出
        min_cross_section_size=2,         # pandas 对齐
    ),
)
```

对齐模式下 IC 与 pandas `corrwith(method="spearman")` 的数值差异 << 0.001。

## 当前算子范围

点式数值：

- `neg`, `abs`, `sign`, `slog`, `sqrt_abs`
- `add`, `sub`, `mul`, `qdiv`

横截面：

- `cs_rank`, `cs_zscore`, `cs_demean`, `cs_winsorize_5pct`, `cs_resid`

时序：

- `delay`, `ts_delta`, `ts_return`
- `ts_mean`, `ts_median`, `ts_std`, `ts_zscore`
- `ts_max_to_min`, `ts_meanrank`, `diff_sign`
- `ts_corr`, `rolling_selmean_diff`, `decay_linear`

条件逻辑已经收敛为 mask 生成 + gate 应用：

- mask：`mask_rank_high_50`, `mask_rank_high_80`, `mask_rank_low_20`, `mask_sign_pos`, `mask_sign_neg`
- gate：`gate_nan`, `gate_zero`

固定窗口只注册：

```text
5, 20, 60, 120
```

## 评分目标

`FactorScore.objectives` 当前返回：

```python
(rank_ic_ir, ndcg_at_k, neutralized_icir)
```

NSGA-II 默认最大化这三个目标。

`neutralized_icir` 使用 post-process 中性化后的因子计算，不覆盖 raw factor。

当前默认口径：

1. 确定共同有效样本（factor、Barra 风格、行业均 finite + tradeable）。
2. 对 Barra 风格字段做每日全截面 z-score（可配置关闭，见下文）。
3. 用标准化后的全部 Barra 字段对 raw factor 做每日多元线性回归（带 ridge 正则化）。
4. 取 Barra 回归残差。
5. 对 Barra residual 做每日行业内 demean。
6. 用最终 residual 重新计算 RankICIR。

**为什么 factor 不需要在回归前做 cs_zscore：**

RankIC（Spearman 秩相关）对单调变换不变。factor → a × factor + b（带截距的线性变换）后，
OLS residual 仅缩放 a 倍（截距被设计矩阵的 intercept 列吸收），rank 不变。
因此 z-scoring factor 再回归 vs 直接用 raw factor 回归，对 `neutralized_icir` 无影响。

**`standardize_styles` 开关：**

如果 Barra 风格因子已经提前做好 cs_zscore，可以在 `ScorerConfig` 中关闭以节省计算：

```python
ScorerConfig(
    style_fields=("barra_size", "barra_beta", "barra_momentum"),
    standardize_styles=False,  # 跳过一次冗余 z-score
)
```

默认 `standardize_styles=True`（在共同有效样本上重新标准化，保证统计性质，即使 Barra 已预 zscore 也无害）。

如果 `style_fields=()` 且 `neutralize_industry=True`，`neutralized_icir` 是行业中性化后的 ICIR。

如果 `neutralize_industry=False` 且没有 `style_fields`，中性化因子退化为 raw factor 的有效样本清洗视图。

## 测试命令

CUDA 环境：

```powershell
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_operators
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_program
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_context
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_evaluator
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_evaluation_metrics
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_neutralizer
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_scorer
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_generator
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_nsga2
D:\Anaconda\envs\pytorch\python.exe -m alpha_gen.free_gp_cuda.tests.test_search
```

本轮调试结果：

```text
test_operators:           8 passed
test_program:             8 passed
test_context:             6 passed
test_evaluator:           7 passed
test_evaluation_metrics:  5 passed
test_neutralizer:         4 passed
test_scorer:              6 passed
test_generator:           3 passed
test_nsga2:               3 passed
test_search:              2 passed
```

合计 52 个测试通过。

隔离检查：

```powershell
rg -n "alpha_gen\.core|torch_backend|behavior_gen|core\.preprocess|TransformCache|build_transform_cache" --glob "*.py" free_gp_cuda
```

无输出表示通过。

## 设计取舍

- 搜索树是类型感知的：numeric 子树不能接 mask 输出，gate 必须是 numeric signal + mask。
- 条件算子没有保留大量比较函数，而是通过少量 mask 生成器表达行为状态，再通过 `gate_nan/gate_zero` 应用。
- rolling 统计仍按算子库约定使用严格窗口，早期不足窗口输出 `NaN`。
- `ProgramEvaluator` 默认在输入和输出层应用 `tradeable` mask。
- `ProgramScorer.factor_values(program, view="raw")` 返回表达式原始因子；`view="neutralized"` 返回行业 + 全 Barra 中性化后的因子。
- `ScorerConfig.neutralize_industry=True` 是默认值；如果数据没有行业列，需要显式设为 `False`。
- 全 Barra 中性化使用同一有效样本：Barra 字段只做全截面中心化/标准化，不做行业中性化；raw factor 先回归去除 Barra，残差再做行业中性化。
- 某个 Barra 字段在单日无截面方差时会作为零暴露列处理，不会让整天 residual 变成 `NaN`。
- 搜索时如果因子覆盖率低于 `FreeGPSearchConfig.min_coverage`，该 program 会被置为空分数，但保留错误说明。
- `run_free_gp_search` 使用 `program_key` 做评分缓存和去重；同一结构不会重复评分。
- 搜索每代会在去重后补足随机合法个体，保持父代规模稳定。
- **NSGA-II 复杂度感知选择（2026-06-02 更新）**：同一 Pareto front 内按 `complexity_cost` 升序排列（低复杂度优先），拥挤距离仅作为同复杂度的 tiebreaker。第一层 Pareto front 上的不可替代复杂因子不会被剔除—只有当 front 部分入选时简洁因子才获得优先权。这与传统 NSGA-II 相比，在不改变支配结构的前提下向简洁性施加稳定的选择压力。

## 已知边界

- 当前没有直接接入原生 `gplearn` 的 `SymbolicTransformer` 类；当前实现是自定义树结构，便于支持时序、截面和 gate 类型约束。
- 当前没有并行多进程评分。由于张量计算在 CUDA 上，第一版保持单进程，避免多进程争抢 GPU。
- 当前没有生成算子偏好统计报表，但 `program.operator_names`、`iter_nodes` 和 `evaluated_to_frame` 已经能支撑后续统计。
- 当前没有落盘恢复搜索现场，只提供 `export_search_result` 导出 final/history/config。
- 当前没有真实大样本显存压测。长表字段是懒加载和 LRU cache，但大规模搜索仍需要根据 530 字段实际数据观察显存峰值。

## 下一步建议

1. 增加 operator preference 统计：
   - 对 `result.history` 统计全部子代 program 的 operator 频率。
   - 再按 `rank_ic_ir/coverage/n_ic_obs` 过滤后统计高质量 program 的 operator 频率。
   - 两者对比可以区分“随机生成偏好”和“有效因子偏好”。

2. 增加批量评估日志：
   - 每代输出覆盖率、有效 IC 观察数、error 分布、Pareto front 大小。
   - 方便判断搜索失败是来自算子过强 NaN、字段质量差，还是目标过严。

3. 做真实数据 smoke run：
   - 先用 30-50 个字段、100-200 个交易日、population 50、generation 2。
   - 观察 CUDA 显存、单代耗时、NaN 覆盖率和表达式复杂度分布。

4. 再决定是否写 gplearn adapter：
   - 如果必须兼容 gplearn API，可以把 `Program` 封装成 gplearn 风格 estimator。
   - 不建议把当前 typed tree 强行塞回 gplearn 原始 primitive 模型，否则 mask/gate 和时序窗口约束会变得很脆弱。
