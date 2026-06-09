# alpha_gen 项目结构交接

本文档记录当前 `alpha_gen` 的代码结构、主流程、数据约定、训练/验证触发时机和指标口径。后续继续开发时，先读本文档，再进入源码。

## 1. 项目定位

`alpha_gen` 是一个面向基本面因子挖掘的参数化遗传算法项目。当前主线围绕原始字段和字段级算子搜索结构化表达式：

```text
single      A
ratio       A / B
pair_ratio  (A +/- B) / (C +/- D)
resi        residual(A ~ B)
resi_pair   residual(A ~ B + C)
multi_resi  residual(A ~ B + C + D)
spread      A / B - C / D
style_composite rank_score(A) + rank_score(B)
```

其中 A/B/C/D 都是 `字段 + unary transform`，当前支持 `current`、`log`、`rank_pct`、`zscore`、`ind_rank_pct`、`ind_zscore`、`diff_2q`、`diff_1y`、`pct_2q`、`pct_1y`。rolling 类 `std_2q/std_1y` 已退出自由搜索池。`ind_*` 算子依赖 `TransformCache.industry`。表达式合法性现在不仅看 y/x 角色，还会检查 metadata 中的 `family`、`unit_type`、`add_group`、`direction` 等语义标签。所有表达式最后统一做中性化：当 `industry_scope` 表示全行业或多行业时先按 `industry_code` 做行业中性化，再对 `size_field` 指定的 Barra size 做截面残差中性化。`size_field` 默认为 `market_cap` 以兼容 mock 数据，正式 Barra size 字段可通过配置传入。

训练阶段的 NSGA-II 目标是：

```text
raw RankICIR, raw NDCG@k, dynamic-Barra-neutralized RankICIR
```

`abs_rank_ic` 和 `ic_win_rate` 仍会导出用于观察和验证过滤，但不再作为 NSGA-II 的独立优化目标。Torch 路径可以传入 10 个已截面标准化的 Barra style 字段；评估时会自动选择平均绝对相关超过 0.3 的 top 2 风格做截面残差化，并把残差因子的 RankICIR 作为第三目标。

验证阶段额外计算分组 PnL 指标，并展开到最终结果表。

## 2. 目录结构

```text
alpha_gen/
  HANDOFF.md                         当前交接文档
  API_USAGE.md                       alpha_factory API 参考文档
  __init__.py                        包初始化

  core/
    utils.py                         本地工具层；格式、算子、滚动窗口逻辑
    gene.py                          字段规则、基因结构、合法性检查、随机生成、交叉变异
    preprocess.py                    长表读取、字段 pivot、TransformCache 构建
    factor_calc.py                   Pandas/CPU 因子表达式计算与市值中性化
    torch_backend.py                 Torch/GPU 因子计算、评价与兼容指标
    metrics.py                       Pandas/CPU 指标、分组 PnL、交易成本处理
    nsga2.py                         NSGA-II 非支配排序、拥挤距离、精英选择
    ga.py                            GA 主循环、验证集评估、结果导出

  data/
    panels/
      mock_tmt_daily.parquet         mock 面板数据
    metadata/
      production/                    正式运行 metadata
      fixtures/                      mock 与测试 metadata
      configs/                       数据构建配置模板
      generated/                     可重新生成的候选 metadata
      archive/                       历史版本，仅供追溯

  examples/
    run_mock_ga.py                   CPU mock GA 入口
    run_mock_ga_gpu.py               GPU mock GA 入口
    run_gplearn_baseline.py          gplearn 对照实验入口

  tests/
    smoke_test.py                    CPU/GPU smoke test

  artifacts/results/
    *.csv / *.json / *.png           示例运行结果和分析图

  docs/reports/
    *.md / *.html / *.pdf / *.png    分析报告和报告图片

  docs/papers/
    *.pdf                            参考研报
```

## 3. 数据格式约定

主面板必须是长表：

```text
index   = MultiIndex["Datetime", "Contract"]
columns = 因子字段 + label + tradeable + 可选行业字段
```

关键字段：

```text
label       默认列名 label_20d；如果你的列名是 label，需要 build_transform_cache(..., label_col="label")
tradeable   默认列名 is_tradeable；如果你的列名是 tradeable，需要 build_transform_cache(..., tradeable_col="tradeable")
market_cap  兼容默认 size_field；正式运行推荐传入 Barra size 字段
industry_code 行业中性化字段；当 industry_scope 为 all/多行业时使用
Barra style fields 可选；传给 TorchEvalContext/GAConfig.barra_style_fields，要求已截面 z-score 且 NaN 填 0.0
```

注意：

- `Datetime` 需要是 `pandas.DatetimeIndex`，推荐带 `15:00:00`。
- 数据已经提供 label 字段时，回测/验证收益直接使用该 label 矩阵，不再从 close/trade_price 重新 shift 生成收益。
- 不允许用未来数据回填历史，只允许按可得时间前向填充。
- `log`、`rank_pct`、`zscore`、`ind_rank_pct`、`ind_zscore`、2Q/1Y 的 `diff`、`pct` 已作为 gene 算子参数，不需要预先生成 `log_*` 字段。`log` 统一使用 `sign(x) * log(1 + abs(x))`，不会因负数直接置空。rolling `std_2q/std_1y` 暂不进入自由 unary transform 池。
- `build_transform_cache(..., extra_current_fields=[size_field, *barra_style_fields])` 可缓存不进入搜索池的 Barra size 和 Barra style 控制字段。若 `size_field` 或传入的 style 字段不在缓存中，训练/验证会明确报错。
- 其他已在外部计算完成的衍生字段仍可作为普通字段进入 GA。
- 当前代码不依赖运行时导入 `alpha_factory`；但指标口径尽量对齐 `E:\实习\alpha_factory\factor\evaluation`。
- mock 数据中 `market_cap` 已经是 log size，`metadata/fixtures/mock_tmt_metadata.json` 中 `market_cap.allow_log=false`。

## 4. 主流程

典型入口流程：

```text
load_field_rules(metadata)
  -> load_panel(parquet)
  -> build_transform_cache(panel, field_rules)
  -> get_rolling_windows(...)
  -> run_ga_search(...)
  -> validate_population(...)
  -> export_search_result(...)
```

GPU 入口会额外创建：

```text
TorchEvalContext(cache, barra_style_fields=("size", "beta", ...))
```

### 训练和验证触发时机

`run_ga_search()` 只做训练集搜索：

- 每一代 parent/offspring 都只在 `train_dates` 上评估。
- 评估结果进入 `history`。
- NSGA-II selection 只使用训练目标：`rank_ic_ir`、`ndcg_at_k`、`neutralized_icir`。
- 每一轮子代不会自动触发验证集评估。
- `GAConfig.mode_probabilities` 控制 mode 生成概率；默认 `None` 表示对当前 `MODE_CHOICES` 等权采样。传入字典时只影响随机初始化、补齐种群和 mode 变异，不改变 NSGA-II 选择口径。

`validate_population()` 才会触发验证集评估：

- 示例入口中只对 `result.final_population` 调用。
- 因此默认只有最后保留下来的种群有 `valid_*` 和 `valid_pnl_*` 指标。
- `history.csv` 是训练搜索轨迹，默认不包含每个候选的验证 PnL 指标。

这种设计是为了避免验证集被每一代反复使用，从而污染模型选择。如果后续确实要记录每代验证表现，必须保证这些验证指标不参与 NSGA-II selection。

## 5. 当前表达式搜索逻辑

`gene.py` 中的 `FactorGene` 核心字段：

```text
a, b, c, d, left_op, right_op, mode, a_transform, b_transform, c_transform, d_transform
```

各模式使用字段：

```text
single          a
ratio           a, b
pair_ratio      a, b, c, d, left_op, right_op
resi            a, b
resi_pair       a, b, c
multi_resi      a, b, c, d
spread          a, b, c, d
style_composite a, b, left_op
```

当前 mode 语义约束：

- `single` 只允许 ratio/rate/growth/score/turnover/yield 等本身已经无量纲或可直接解释的指标。
- `ratio` 要求 A/B 的 `unit_type` 和 `add_group` 模板同时命中白名单，且只能使用保持会计量纲的 transform。
- `pair_ratio` 要求 A/B 同 `unit_type`、同 `add_group`、同 transform，C/D 同理，最后分子/分母的 `unit_type` 也必须命中比值白名单。
- `resi` 的 A/B 都只要求是 metadata 中存在且 transform 合法的任意字段，不再限制 A 的 signal family / `can_y=true`，也不再限制 B 的 control family / `can_x=true`。
- `resi_pair` 为 `residual(A ~ B + C)`，A 不限制角色；B/C 必须是可解释的同组可加控制项：同 `unit_type`、同 `add_group`、同 accounting transform，不能和目标项重复。
- `multi_resi` 为 `residual(A ~ B + C + D)`，A 不限制角色；B/C/D 必须是同 `unit_type`、同 `add_group`、同 accounting transform 的可加控制项，不能随机抽三个不相关字段相加。
- `spread` 为两个合法会计比值之差：`A / B - C / D`，左右两侧分别复用 `ratio` 的 unit/add_group/transform 约束。
- `style_composite` 在内部计算 `direction * (rank_pct - 0.5)` 后相加，只允许白名单风格组合，例如 quality/value、growth/value、analyst/value；当前不允许减法。

表达式计算统一流程：

```text
取当前字段矩阵
  -> 按 gene 参数做 log/rank/zscore/行业内 rank 或 zscore/diff/pct 等字段级变换
  -> tradeable mask
  -> 按 mode 组合
  -> 如果 industry_scope 是全行业或多行业，先按 industry_code 去行业均值
  -> 对 size_field 指定的 Barra size 做截面残差中性化
```

Pandas 版本在 `factor_calc.py`，Torch 版本在 `torch_backend.py`。正式运行优先使用 Torch 路径。

中性化配置：

- `GAConfig.industry_scope=None` 或传入具体单行业名时，不做行业中性化。
- `GAConfig.industry_scope="all"`、`"multi"`、`"全行业"`、`"多行业"`，或传入多个行业名时，强制做行业中性化。
- `GAConfig.size_field` 控制市值/规模中性化字段。若是 `market_cap`，默认使用 signed log1p 后回归；若是传入的 Barra size 字段，默认直接使用原始 Barra exposure，不再 log。需要覆盖时可显式设置 `use_log_size`。
- `GAConfig.barra_style_fields` 控制动态 Barra 剥离用的风格字段。字段应在外部完成截面 z-score 和 NaN->0；本地只做张量化相关性筛选和残差化。`barra_corr_threshold` 默认 0.30，`barra_max_styles` 默认 2。

## 6. 指标和边界口径

指标相关代码集中在 `core/metrics.py` 和 `core/torch_backend.py`。

### IC / RankIC

口径对齐 `alpha_factory.factor.evaluation.ic.ICAnalyzer.calc_daily_ic`：

```text
if tradeable is not None:
    label = label.where(tradeable == 1)
factor = factor.align(label, join="right")[0].dropna(axis=0, how="all")
daily corr / rank corr by row
```

当前实现保留两个必要边界保护：

- `tradeable` 缺失、inf、非正值都按不可交易处理。
- RankIC 默认 `min_cross_section_size=3`，2 个有效样本的截面会跳过，避免虚高和 scipy 在 pytorch 环境下的边界崩溃。

### Coverage

口径对齐 `alpha_factory.factor.evaluation.quality.FactorQualityAnalyzer.calc_coverage`：

```text
coverage = factor.notna().where(tradeable == 1).sum(axis=1) / tradeable.sum(axis=1)
coverage_mean = coverage.dropna().mean()
```

也就是说 coverage 衡量的是 tradeable universe 中有因子值的比例，而不是“factor 和 label 同时可用”的比例。

### NDCG

NDCG 是第二个 NSGA-II 目标，用于保留头部多头表现好的因子。CPU/GPU 口径需要继续保持一致，后续修改要同步检查。

### 动态 Barra 风格剥离

动态 Barra 剥离只在 Torch 评估路径中生效：

```text
raw factor
  -> 因子截面 z-score，NaN 填 0
  -> 与 K 个 Barra style 张量点乘，得到每期 Pearson corr
  -> abs(mean(corr_t)) > 0.3 的风格进入候选
  -> 取相关性最大的至多 2 个风格
  -> 每天截面批量 OLS：raw factor ~ intercept + selected Barra styles
  -> residual factor 的 RankICIR 作为 neutralized_icir
```

方向统一由原因子的 `direction` 决定：如果残差因子的自然 IC 方向与原因子相反，也不会重新翻方向，而是使用原因子的方向计算 `neutralized_icir`。CPU 评估路径没有 Barra 张量上下文，因此 `neutralized_icir` 默认回填为 raw `rank_ic_ir`，保持接口兼容。

### 分组 PnL / 回测指标

`factor_group_pnl()` 使用现成 `label` 矩阵作为收益输入，但分组和统计口径对齐 `alpha_factory.factor.evaluation.pnl.PNLAnalyzer.calc_group_pnl`：

```text
factor.rank(pct=True, method="first")
  -> 每日 qcut 成 n_groups 组
  -> 每组平均 label return
  -> long = top group
  -> short = bottom group
  -> longshort = long - short
  -> annualized return, excess sharpe, max drawdown, drawdown duration, turnover, fitness
```

Sharpe 使用超额收益口径，不再直接对组合原始收益计算：

```text
benchmark_return = tradeable universe label return 的每日等权平均
long_sharpe = sharpe(pnl_long - benchmark_return)
short_sharpe = sharpe(pnl_short - benchmark_return)
longshort_sharpe = sharpe(pnl_longshort - benchmark_return)
```

其中 `pnl_long` 和 `pnl_longshort` 已经扣交易成本。为方便核对，原始组合收益 Sharpe 仍会作为 `long_raw_sharpe`、`short_raw_sharpe`、`longshort_raw_sharpe` 导出。

### 交易成本

当前分组 PnL 一定考虑费用成本：

```text
round_trip_cost = 2 * (commission_rate + slippage_rate) + stamp_tax_rate
默认 = 2 * (0.0003 + 0.0002) + 0.001 = 0.002
```

成本按换手扣减：

```text
net_group_return = gross_group_return - group_turnover * round_trip_cost
```

最终导出表中会同时包含 gross、net、turnover、cost 相关标量。

## 7. 结果导出

`export_search_result()` 输出：

```text
*_history.csv
*_final_population.csv
*_config.json
```

`history.csv`：

- 来自 `result.history`。
- 记录训练搜索过程中评估过的候选。
- 默认只有 `train_*` 指标。
- 除非显式对 history 中的候选逐个调用验证，否则不会有 `valid_pnl_*`。

`final_population.csv`：

- 来自 `result.final_population`。
- 如果先调用了 `validate_population(result.final_population, ...)`，会包含：

```text
valid_mean_rank_ic
valid_abs_rank_ic
valid_ic_win_rate
valid_ndcg_at_k
valid_top_excess_ann
passed_validation
valid_pnl_*
```

`valid_pnl_*` 目前会展开所有标量 PnL 指标，例如：

```text
valid_pnl_round_trip_cost
valid_pnl_pnl_gross_mean_group0
valid_pnl_pnl_mean_group0
valid_pnl_turnover_group0
valid_pnl_cost_mean_group0
valid_pnl_pnl_long_ann
valid_pnl_pnl_short_ann
valid_pnl_pnl_longshort_ann
valid_pnl_benchmark_return_mean
valid_pnl_benchmark_return_ann
valid_pnl_pnl_long_excess_ann
valid_pnl_long_sharpe
valid_pnl_long_raw_sharpe
valid_pnl_long_turnover
valid_pnl_short_turnover
valid_pnl_longshort_turnover
valid_pnl_long_cost_mean
valid_pnl_long_fitness
...
```

`valid_top_excess_ann` 当前为兼容旧列名保留，实际值取扣费后的 `valid_pnl_pnl_long_ann`。

## 8. 运行命令

CPU smoke test：

```powershell
python .\alpha_gen\tests\smoke_test.py
```

GPU smoke test：

```powershell
D:\Anaconda\envs\pytorch\python.exe .\alpha_gen\tests\smoke_test.py --gpu
```

GPU mock GA：

```powershell
D:\Anaconda\envs\pytorch\python.exe .\alpha_gen\examples\run_mock_ga_gpu.py
```

当前机器上 `conda env list` 可能被 conda CUDA virtual-package 插件触发 Windows 权限错误；直接调用 `D:\Anaconda\envs\pytorch\python.exe` 更稳定。

## 9. 当前开发原则

- 指标口径优先对齐 `E:\实习\alpha_factory\factor\evaluation`，不要随意发明另一套定义。
- `alpha_gen` 内部模块只依赖 `alpha_gen.core.utils` 和本项目模块，不要把 `alpha_factory` import 分散到各文件。
- 真实正式跑时优先使用 `TorchEvalContext`，CPU 路径主要用于可读性、对照和调试。
- 新增字段变换时先考虑是否应在数据预处理阶段完成；只有确实需要被 GA 搜索的操作才应进入 gene/mode。
- 新增表达式模式时必须同时修改 `gene.py`、`factor_calc.py`、`torch_backend.py`、`ga.py` 和 smoke test。
- 任何指标口径修改都要检查 CPU/GPU 是否一致。
- 验证集指标默认只在 `validate_population()` 中计算，不应参与训练阶段 NSGA-II 选择。

## 10. 已知关注点

- `history.csv` 默认没有验证 PnL 指标，这是流程设计，不是导出漏列。
- 当前 `cs_resi` 本地实现是串行 Pandas 版本；正式搜索走 Torch 路径，影响不大。
- CPU/GPU 的 NDCG 口径需要保持持续关注，避免后续修改造成选择结果偏差。
- `artifacts/results/`、`__pycache__/` 属于运行产物，不应作为研究结论依据。

## 11. 行为金融 GA 挖掘系统（behavior_gen/）

### 11.1 核心思想

传统 GA（第 5 节描述的 `core/ga.py`）将因子表达式建模为字段 A/B/C/D + 变换 + 组合模式（ratio/residual/spread 等），但字段选择和组合缺乏**经济学语义约束**——它可以把 ROE 和换手率做 `ratio`，也能把成交量和大单净流入做 `residual`，不考虑这些组合是否有行为金融学含义。

**行为金融 GA** 的核心改进是：**用行为金融学理论预定义"叙事模板"（mode），基因只能在模板的语义槽位内选择字段**。例如 `fund_price_underreaction`（基本面反应不足）模式要求基因必须有一个 `fund_anchor`（基本面锚）槽位和一个 `price_reaction`（价格反应）槽位——不能把价量字段塞进锚定槽位，也不能把基本面字段塞进反应槽位。

### 11.2 基因结构

```
BehaviorGene
├── mode: str              ← 行为金融模式名（11 种之一）
├── combiner: str          ← 槽位间合成算子（rank_gap / confirm / panic_reversal 等）
├── slots: dict[str, SlotGene]  ← 每个语义槽位选一个字段 + 变换
│   └── SlotGene:
│       ├── field: str     ← 元数据注册的字段名
│       └── unary_op: str  ← 一元变换（rank_pct / zscore / ts_zscore_20d 等）
├── conditions: tuple[ConditionGene]  ← 可选门控条件
│   └── ConditionGene:
│       ├── field, unary_op
│       ├── condition_op: str  ← top_quantile / above_median / positive 等
│       └── threshold: float
└── direction_policy: str  ← "fixed"（理论方向）或 "train_ic"（数据驱动方向）
```

**关键约束系统**：

```
BehaviorFieldRule（每个字段的"身份证"）
├── data_family:    fundamental | price_volume | moneyflow | orderbook
├── behavior_roles: anchor | growth | momentum | crowding | panic | quality | ...
├── allowed_slots:  该字段允许被放入哪些槽位（空白名单 = 全部允许）
└── allowed_unary_ops: 该字段允许的变换算子

ModeSpec（每个模式的"模板"）
├── slots: dict[str, SlotSpec]  ← 定义需要哪些语义槽位
│   └── SlotSpec:
│       ├── data_families:  该槽位允许的数据族
│       ├── behavior_roles: 该槽位要求的字段行为角色
│       └── required: bool  是否必填（False = 可选槽位）
├── allowed_combiners: 该模式允许的合成算子
├── direction:  +1（做多）或 -1（做空）
└── max_conditions:  最多允许几个条件
```

### 11.3 11 种行为金融模式

| 模式 | 方向 | 核心叙事 | 关键槽位 |
|---|---|---|---|
| `fund_price_underreaction` | +1 | 基本面改善但价格反应不足 | fund_anchor + price_reaction |
| `quality_neglect` | −1 | 利润增长但现金流质量差，市场忽视 | profit_growth + cashflow_quality |
| `growth_crowding_risk` | −1 | 成长叙事叠加成交拥挤→回撤风险 | growth_anchor + crowding_signal |
| `fund_flow_confirmation` | +1 | 基本面改善+大资金买入确认 | fund_anchor + flow_confirm |
| `retail_chase_risk` | −1 | 小单追涨+大资金不确认 | price_momentum + retail_flow + close_chase |
| `panic_reversal` | +1 | 基本面尚可但恐慌造成错杀 | fund_anchor + drawdown + sell_pressure |
| `attention_overreaction` | −1 | 极端热度造成过度反应 | attention_heat |
| `orderbook_intent` | +1 | 盘口买卖压力刻画未成交意图 | orderbook_pressure |
| `liquidity_neglect` | −1 | 成交活跃但盘口流动性压力高 | liquidity_stress + turnover_shock |
| `anchor_momentum` | +1 | 52周高点/成本锚定下的动量延续 | price_anchor + price_momentum |
| `disposition_anchor` | +1 | 成本锚定位置→处置效应 | cost_anchor + price_momentum |

### 11.4 基因生成流程（random_gene）

```
random_gene()
│
├── Step 1: 选择模式
│   └── 从可行模式中按 mode_probabilities 加权采样（默认均匀）
│
├── Step 2: 选择 combiner
│   └── 从模式的 allowed_combiners 中均匀随机选
│
├── Step 3: 填充槽位（_sample_slots）
│   ├── 必填槽位: 100% 填充
│   │   ├── 从 fields_for_slot() 候选集中均匀随机选 field
│   │   └── 从 field.allowed_unary_ops 中均匀随机选 unary_op
│   └── 可选槽位: 45% 概率填充（optional_slot_probability=0.45）
│       └── 同上
│
├── Step 4: 生成条件（_sample_conditions）
│   ├── 65% 概率: 无条件
│   └── 35% 概率: 生成 1~max_conditions 个条件
│       ├── 条件字段: 从 condition_fields_for_mode() 候选不放回随机选
│       ├── condition_op: 6 种均匀随机
│       ├── unary_op: 从字段允许列表中均匀随机
│       └── threshold: 从 (0.55, 0.60, 0.70, 0.80) 均匀随机
│
├── Step 5: 选择方向策略
│   └── 从 ("fixed", "train_ic") 均匀随机选
│
└── Step 6: repair_gene() 修复 + is_valid_gene() 校验
```

### 11.5 遗传操作

**变异（mutate_one_parameter）**：从 6 个维度中均匀随机选 1 个（各 1/6），只改这一个维度：

| 变异维度 | 操作 |
|---|---|
| mode | 重新随机采样新模式 |
| combiner | 从新模式允许的 combiner 中均匀重选 |
| slot_field | 随机选一个已填充槽位，重新采样其字段 |
| slot_unary | 随机选一个已填充槽位，重新采样其一元算子 |
| condition | 50% 删除一个条件 / 50% 重新采样 |
| direction_policy | 从 ("fixed", "train_ic") 均匀重选 |

**交叉（crossover_genes）**：两个父代产生两个子代，每个子代有主父代（75% mode）和副父代（25% mode）。每个槽位独立从两父代的同槽位等位基因中随机选一个。条件、direction_policy 各 50% 继承。

### 11.6 combiner 的数学定义

combiner 是将多个槽位值合成为最终因子信号的公式。所有槽位值先通过 `_feature()` 做一元变换（中心化 rank_pct 在 [-0.5, 0.5]，zscore 无界），然后按 combiner 公式合成：

| combiner | 公式 | 使用模式 |
|---|---|---|
| `rank_gap` | `slot₀ − slot₁` | fund_price_underreaction |
| `residual_gap` | `residual(slot₀, slot₁)` — 剥离共线性 | fund_price_underreaction |
| `quality_gap` | `profit − cashflow + 0.25·|price|` | quality_neglect |
| `crowding_interaction` | `growth × crowding − fund_support` | growth_crowding_risk |
| `confirm` | `slot₀ + slot₁ + slot₀×slot₁ + 附加` | fund_flow_confirmation, anchor_momentum |
| `risk_minus_confirm` | `sum(risk) − sum(confirm)` | retail_chase_risk |
| `panic_reversal` | `anchor × drawdown − pressure` | panic_reversal |
| `attention_risk` | `heat + momentum − support` | attention_overreaction |
| `orderbook_intent` | `pressure − stress − 0.25·|price|` | orderbook_intent |
| `liquidity_gap` | `stress − turnover − flow` | liquidity_neglect |
| `anchor_confirm` | `anchor + momentum + anchor×momentum` | anchor_momentum, disposition_anchor |

### 11.7 因子计算评估流程

```
calculate_behavior_factor_tensor()
├── 1. validate_gene()           ← 校验基因合法性
├── 2. _slot_values()            ← 每个槽位: field → _feature() → unary_op 变换
├── 3. _combine_behavior_gene()  ← 按 combiner 语义合成槽位值
├── 4. _apply_conditions()       ← 门控条件掩码（非 gated combiner）
├── 5. × mode.direction          ← fixed 策略: 乘以模式方向
├── 6. _apply_mask(tradeable)    ← 不可交易 cell 置 0
├── 7. industry_neutralize()     ← 行业中性化（可选）
└── 8. residual(size)            ← 市值中性化

score_behavior_factor_tensor()
└── evaluate_factor_tensor()     ← 计算 RankIC IR, NDCG@K, 中性化 ICIR
```

### 11.8 NSGA-II 多目标搜索

```
目标 1: rank_ic_ir     = mean(定向 RankIC) / std(定向 RankIC)
目标 2: ndcg_at_k       = 头部排序质量（默认 top 10%）
目标 3: neutralized_icir = 经 Barra 风格剥离后的残差因子 RankIC IR

每代流程:
  父代 → crossover (85%) + mutation (25%) → 子代
  → 评估子代 → 合并父代+子代 → 去重
  → nsga2_select(objectives, population_size) → 新一代种群
```

---

## 12. 类型化树 GP 挖掘系统（behavior_tree_gp/）

### 12.1 为什么需要 GP？—— GA 的局限

GA 系统的核心局限在于：**每个语义槽位只能选一个字段**。

```
GA 的 fund_anchor:  rank_pct(ROE)              ← 只能是一个字段
```

但行为金融学中的"基本面锚"可能需要多个字段综合衡量：

```
理想的 fund_anchor:  mean( rank_pct(ROE), rank_pct(毛利率) )
                     再剥离掉估值的影响:
                     residual( 上述综合, zscore(PE) )
```

**GP 的核心改进**：把每个槽位的值从"单个字段 + 变换"升级为**一棵类型安全的表达式树**，允许在槽位内部做有限但有意义的算术组合，同时保持类型系统的严格约束。

### 12.2 树表达式结构

```python
TreeExpr = FieldNode | UnaryNode | BinaryNode
```

**三层节点**：

```
FieldNode（叶子）                    UnaryNode（一元内部节点）         BinaryNode（二元内部节点）
┌─────────────────────┐             ┌─────────────────────┐         ┌──────────────────────────┐
│ semantic_type: str  │             │ semantic_type: str  │         │ semantic_type: str       │
│ field: str          │             │ op: str             │         │ op: str                  │
│ unary_op: str       │             │ child: TreeExpr     │         │ left: TreeExpr           │
└─────────────────────┘             └─────────────────────┘         │ right: TreeExpr          │
 例: FieldNode(                      例: UnaryNode(                 └──────────────────────────┘
   type="fund_anchor",                  type="fund_anchor",          例: BinaryNode(
   field="roe_ttm",                     op="neg",                      type="fund_anchor",
   unary_op="rank_pct")                 child=FieldNode(...))          op="mean",
                                                                      left=FieldNode(...),
                                                                      right=FieldNode(...))
```

**关键特性——语义类型保持（Role-Preserving）**：
- 每个节点的 `semantic_type` 固定等于所在槽位名
- UnaryNode 和 BinaryNode 不改变语义类型
- 整个子树的值域始终在槽位的语义空间内
- 这保证了无论树多复杂，根节点的语义类型始终正确

### 12.3 两套算子体系

**叶子层算子**（FieldNode.unary_op，继承自 GA 的 UNARY_OP_CHOICES）：
```
current, rank_pct, zscore, direction_rank, direction_zscore,
ind_rank_pct, ind_zscore, ts_zscore_5d, ts_zscore_20d
```
将原始字段值变换为标准化的可比较形式。

**树内一元算子**（UnaryNode.op，新增）：
```
rank, zscore, ind_rank, ind_zscore,     ← 在树中间层重新标准化
ts_zscore_5d, ts_zscore_20d,            ← 时序标准化
neg,                                    ← 取反（值域翻转）
abs,                                    ← 绝对值
clip                                    ← 截断 [-8, 8]
```

**树内二元算子**（BinaryNode.op，新增）：
```
mean         ← (a + b) / 2       → 多信号取共识
diff         ← a − b             → 两个信号的背离
interaction  ← a × b             → 两个信号的协同放大
residual     ← residual(a, b)    → 剥离 b 对 a 的线性影响
```

### 12.4 类型规则系统（type_rules.py）

GP 系统最关键的约束是**二元操作的类型规则**。不是任意两个字段都能做二元操作——必须符合金融经济学含义。

**子类型体系**（17 种 sub_type）：

| 数据族 | 子类型 |
|---|---|
| fundamental | fund_growth, fund_quality, fund_value |
| price_volume | pv_momentum, pv_volume, pv_volatility, pv_crowding, pv_panic, pv_general |
| moneyflow | mf_large, mf_small, mf_active, mf_general |
| orderbook | ob_spread, ob_depth, ob_pressure |
| control | control |

**BINARY_TYPE_RULES 规则表**（24 条规则）：

规则逻辑：
- **同子类型** → 全部 4 种二元操作都允许
- **跨子类型** → 仅表中列出的操作允许
- **不在表中且子类型不同** → 禁止所有二元操作

示例规则：
```
(momentum, panic):       {diff, interaction}     ← 动量与恐慌可做差或交互，不能取均值
(large_flow, small_flow): {mean, diff}           ← 大单与小单可做差或取均值，不能交互
(pressure, spread):       {residual}             ← 盘口压力只能剥离买卖价差
(growth, quality):        {mean, diff, residual} ← 成长与质量可均值/差/残差，不能交互
```

**附加类型约束**：
- **叶子唯一性**：同一字段不能在同一棵树的任何两个叶子位置出现（`validate_leaf_uniqueness`）
- **数据族一致性**：一棵子树的所有叶子必须属于同一 `data_family`（`validate_within_tree_data_family`）

### 12.5 树生成流程（random_typed_tree）

与 GA 的关键区别：槽位值不再是简单的 field + unary_op 选择，而是**递归生长**一棵树。

```
random_typed_tree()
│
├── Step 1-2: 选择 mode + combiner（同 GA）
│
├── Step 3: 填充槽位（_sample_slots → sample_slot_tree）
│   │
│   └── sample_slot_tree()  ← 递归生长！
│       │
│       ├── 终止条件（以 probability terminal_probability=0.58 触发生成叶子）:
│       │   └── _sample_field_node()
│       │       ├── 从 fields_for_slot() 候选均匀随机选 field
│       │       └── 从 field.allowed_unary_ops 均匀随机选 unary_op
│       │
│       ├── 以 probability unary_probability=0.20 生成 UnaryNode:
│       │   ├── 从 9 种树内一元算子中均匀随机选 op
│       │   └── 递归生长 child 子树（remaining_depth - 1）
│       │
│       └── 以 probability 0.22 生成 BinaryNode:
│           ├── 从 4 种树内二元算子中均匀随机选 op
│           ├── 递归生长 left 子树（remaining_depth - 1）
│           └── 递归生长 right 子树（remaining_depth - 1）
│
├── Step 4: 生成条件（同 GA）
│
└── Step 5-6: 选择方向策略 + validate_tree() 全面校验
    ├── 模式/combiner 合法性
    ├── 每个槽位的树深度、节点数限制（max_slot_depth=3, max_nodes=32）
    ├── semantic_type 匹配
    ├── BINARY_TYPE_RULES 类型规则
    ├── 叶子唯一性和数据族一致性
    └── 失败 → 最多重试 200 次
```

**树的期望形状**（由概率控制）：

```
terminal_probability = 0.58  → 约 58% 概率在当前深度直接生成叶子
unary_probability     = 0.20  → 约 20% 概率生成一元节点
binary (隐式)         = 0.22  → 约 22% 概率生成二元节点

期望深度 ≈ 1 ~ 2.5 层（因为 terminal_probability 较高，树倾向于浅层）
期望节点数 ≈ 2 ~ 5 个（多数树是 1-2 个叶子加少量内部节点）
```

### 12.6 遗传操作

**变异（mutate_typed_tree）**：5 个维度均匀随机选 1 个（各 1/5）：

| 变异维度 | 操作 |
|---|---|
| mode | 重新随机采样新模式 + 重建默认 combiner，清空 slots + conditions |
| combiner | 从模式允许的 combiner 中均匀重选 |
| slot_tree | 随机选一个槽位，**整棵子树重新随机生长** |
| condition | 50% 删除条件 / 50% 重新采样 |
| direction_policy | 从配置中均匀重选 |

**交叉（crossover_typed_trees）**：逐槽位交换整棵子树。
- 子代的每个槽位：如果两父代都有该槽位 → 均匀随机选一个父代的子树
- 如果只有一个父代有 → 继承该父代的子树
- 必填槽位两父代都没有 → 重新生长一棵树
- 可选槽位两父代都没有 → 以 optional_slot_probability 决定是否生长

### 12.7 树计算评估流程

```
calculate_tree_factor_tensor()
├── 1. validate_tree()                    ← 全面类型+语义+复杂度校验
├── 2. evaluate_slot_tree() × N          ← 递归求值每棵槽位子树
│       ├── FieldNode  → _feature(field, unary_op)  ← 叶子: 同 GA
│       ├── UnaryNode  → 应用树内一元算子到 child
│       └── BinaryNode → 递归求值 left, right, 应用二元算子
├── 3. _combine_slot_values()            ← 同 GA 的 combiner 公式
├── 4-8. 条件 / 方向 / 中性化             ← 同 GA
```

**UnaryNode 的递归求值**（`evaluate_slot_tree`）：
```
neg(child)      → -child
abs(child)      → |child|
clip(child)     → clamp(child, -8, 8)
rank(child)     → cs_rank_pct(child) - 0.5    ← 横截面重新排名
zscore(child)   → cs_zscore(child)            ← 横截面重新标准化
ind_rank(child) → industry_rank_pct(child) - 0.5
```

**BinaryNode 的递归求值**：
```
mean(left, right)       → (left + right) * 0.5
diff(left, right)       → left - right
interaction(left, right) → left * right
residual(left, right)   → cross_sectional_residual(left, right)
```

### 12.8 GP vs GA 的关键改进

| 维度 | GA | GP |
|---|---|---|
| **槽位表达力** | 1 字段 + 1 变换 | 任意深度的表达式树 |
| **字段组合位置** | 仅在模式级 combiner | 槽位内部 + 模式级双层组合 |
| **叶子约束** | field ∈ candidates | field ∈ candidates + leaf_uniqueness + data_family 一致性 |
| **操作约束** | unary_op ∈ allowed | unary_op ∈ allowed + BINARY_TYPE_RULES |
| **树结构** | 退化树（深度=1） | 递归树（深度≤3，节点≤32） |
| **复杂度控制** | 隐式（槽位数量 + 条件数量） | 显式 parsimony_coefficient 惩罚节点数 |
| **失败基因处理** | 简单修复 | 200 次重试 + 逐层降级修复 |
| **gate_fill 默认值** | "zero"（可能虚增 coverage） | "nan"（诚实反映覆盖） |
| **无效评分** | 全零分 | 全 -inf（确保 NSGA-II 淘汰） |

### 12.9 简洁性压力（Parsimony Pressure）

GP 独有的机制。在 NSGA-II 的三个目标上都施加正比于树节点数的惩罚：

```python
selection_objectives = (
    rank_ic_ir      - parsimony_coefficient * tree_size,
    ndcg_at_k       - parsimony_coefficient * tree_size,
    neutralized_icir - parsimony_coefficient * tree_size,
)
```

默认 `parsimony_coefficient = 0.001`。这意味着：
- 一个 tree_size=5 的基因，每个目标被扣 0.005
- 一个 tree_size=20 的基因，每个目标被扣 0.020
- NSGA-II 在多目标比较时，自然会偏好"效果好且结构简单"的基因

这防止了 GP 常见的 **bloat（代码膨胀）** 问题——树无限制生长但没有实质性能提升。

### 12.10 GA→GP 的桥接关系

```
GA 基因可以无损表示为 GP 树:
  BehaviorGene(mode, combiner, slots={name: SlotGene(field, unary_op)})
  ↓ 等价于
  TemplateTree(mode, combiner, slots={name: FieldNode(type=name, field=field, unary_op=unary_op)})
  ↑ 所有槽位都是深度为 1 的叶子，树内无 UnaryNode/BinaryNode

GP 树的退化形式就是 GA 基因。
因此 GP 是 GA 的严格超集。
```

### 12.11 文件索引

```
GA 系统:
  behavior_gen/gene.py           基因定义、模式注册表、字段规则、校验
  behavior_gen/sampler.py        随机生成、修复、变异、交叉
  behavior_gen/torch_backend.py  GPU 因子张量计算、combiner、评分
  behavior_gen/ga.py             NSGA-II 搜索主循环、验证、导出

GP 系统:
  behavior_tree_gp/typed_tree.py     类型化树表达式定义、校验、去重键
  behavior_tree_gp/type_rules.py     子类型体系、二元操作规则表、叶子校验
  behavior_tree_gp/typed_sampler.py  树采样、修复、变异、交叉
  behavior_tree_gp/torch_backend.py  树递归求值、combiner、因子计算
  behavior_tree_gp/ga.py             NSGA-II 搜索（带 parsimony）
  behavior_tree_gp/run_typed_tree_gp.py  运行入口

共享基础设施:
  behavior_gen/gene.py  ← 两套系统共享 MODE_REGISTRY, BehaviorFieldRule, UNARY_OP_CHOICES
  core/torch_backend.py ← GPU 张量算子（cs_rank_pct, cs_zscore, residual, NDCG 等）
  core/metrics.py       ← FactorScore, factor_group_pnl
  core/nsga2.py         ← NSGA-II 选择、排名表
```
