# alpha_gen 项目结构交接

本文档记录当前 `alpha_gen` 的代码结构、主流程、数据约定、训练/验证触发时机和指标口径。后续继续开发时，先读本文档，再进入源码。

## 1. 项目定位

`alpha_gen` 是一个面向基本面因子挖掘的参数化遗传算法项目。当前主线围绕原始字段和字段级算子搜索结构化表达式：

```text
single      A
ratio       A / B
pair_ratio  (A +/- B) / (C +/- D)
resi        residual(A ~ B)
ratio_product (A / B) * (C / D)
```

其中 A/B/C/D 都是 `字段 + unary transform`，当前支持 `current`、`log`、`zscore`、`diff_2q`、`diff_1y`、`pct_2q`、`pct_1y`、`std_2q`、`std_1y`。所有表达式最后统一做市值中性化。市值字段默认使用 `market_cap`，mock 数据中已经是 log size。

训练阶段的 NSGA-II 目标仍是：

```text
|RankIC|, IC win rate, NDCG@k
```

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
    make_mock_tmt_data.py            生成 mock TMT 面板和 metadata
    build_real_tmt_panel.py          真实长表/宽表合并脚本
    real_panel_config_template.json  真实数据合并配置模板
    mock_tmt_daily.parquet           mock 面板数据
    mock_tmt_metadata.json           mock 字段规则

  examples/
    run_mock_ga.py                   CPU mock GA 入口
    run_mock_ga_gpu.py               GPU mock GA 入口
    run_gplearn_baseline.py          gplearn 对照实验入口

  tests/
    smoke_test.py                    CPU/GPU smoke test

  results/
    *.csv / *.json                   示例运行结果

  report/
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
market_cap  默认市值中性化字段；mock 数据里已经是 log size
```

注意：

- `Datetime` 需要是 `pandas.DatetimeIndex`，推荐带 `15:00:00`。
- 数据已经提供 label 字段时，回测/验证收益直接使用该 label 矩阵，不再从 close/trade_price 重新 shift 生成收益。
- 不允许用未来数据回填历史，只允许按可得时间前向填充。
- `log`、`zscore`、2Q/1Y 的 `diff`、`pct`、`std` 已作为 gene 算子参数，不需要预先生成 `log_*` 字段。
- 其他已在外部计算完成的衍生字段仍可作为普通字段进入 GA。
- 当前代码不依赖运行时导入 `alpha_factory`；但指标口径尽量对齐 `E:\实习\alpha_factory\factor\evaluation`。
- mock 数据中 `market_cap` 已经是 log size，`mock_tmt_metadata.json` 中 `market_cap.allow_log=false`。

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
TorchEvalContext(cache)
```

### 训练和验证触发时机

`run_ga_search()` 只做训练集搜索：

- 每一代 parent/offspring 都只在 `train_dates` 上评估。
- 评估结果进入 `history`。
- NSGA-II selection 只使用训练目标。
- 每一轮子代不会自动触发验证集评估。

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
single      a
ratio       a, b
pair_ratio  a, b, c, d, left_op, right_op
resi        a, b
ratio_product a, b, c, d
```

表达式计算统一流程：

```text
取当前字段矩阵
  -> 按 gene 参数做 log/diff/pct/std 等字段级变换
  -> tradeable mask
  -> 按 mode 组合
  -> 对 market_cap 做截面残差中性化
```

Pandas 版本在 `factor_calc.py`，Torch 版本在 `torch_backend.py`。正式运行优先使用 Torch 路径。

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

NDCG 仍是 `alpha_gen` 本地目标之一，用于 GA 多目标训练。CPU/GPU 口径需要继续保持一致，后续修改要同步检查。

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
- `results/`、`__pycache__/` 属于运行产物，不应作为研究结论依据。
