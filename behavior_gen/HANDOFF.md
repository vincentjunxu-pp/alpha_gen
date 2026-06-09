# 行为金融学遗传挖掘框架交接

## 1. 当前边界

- `alpha_gen/core/` 保留为原基本面因子挖掘区，不在本轮改动中迁移。
- `alpha_gen/behavior_gen/` 是新的行为金融学因子挖掘区，基因结构已从原来的 A/B/C/D 字段表达式改成 `mode + slot + combiner + condition + control`。
- 行为框架目前是 GPU-first：搜索、组合和评分都围绕 `BehaviorTorchContext`，默认要求 CUDA。

## 2. 新增文件

- `behavior_gen/gene.py`
  - 定义行为字段规则、槽位、模式、条件和 `BehaviorGene`。
  - `MODE_REGISTRY` 当前包含 11 类行为机制。
- `behavior_gen/sampler.py`
  - 模式感知的随机采样、修复、变异和交叉。
  - 会根据元数据里的 `behavior_field_rules` 限制字段和算子组合。
- `behavior_gen/torch_backend.py`
  - 行为基因到 Datetime x Contract 因子张量的 CUDA 计算后端。
  - 复用 `core.torch_backend` 的 RankIC、NDCG、行业中性化、Barra 动态中性化等张量函数。
- `behavior_gen/ga.py`
  - 行为因子 GA 主循环、训练评分、NSGA-II 选择、验证和导出。
- `examples/run_behavior_mock_ga_gpu.py`
  - Mock 数据上的 CUDA 端到端运行入口。
- `scripts/data_builders/make_mock_behavior_data.py`
  - 生成调试用行为金融学模拟数据和元数据。
- `data/panels/mock_behavior_daily.parquet`
  - 模拟长表数据，MultiIndex 为 `Datetime, Contract`。
- `data/metadata/fixtures/mock_behavior_metadata.json`
  - 同时包含旧 `field_rules` 和新 `behavior_field_rules`。

## 3. 基因结构

`BehaviorGene` 的核心字段：

- `mode`：行为机制模板，例如基本面-价格反应不足、散户追涨风险、恐慌反转、盘口意图等。
- `slots`：语义槽位，不再是无语义的 A/B 字段。每个 mode 定义自己需要的槽位，例如 `fund_anchor`、`price_reaction`、`flow_confirm`、`liquidity_stress`。
- `unary_op`：字段的一元变换，例如 `rank_pct`、`zscore`、`direction_rank`、`ind_zscore`、`ts_zscore_20d`。
- `combiner`：槽位之间的组合方式，例如 `rank_gap`、`residual_gap`、`confirm`、`risk_minus_confirm`、`panic_reversal`。
- `conditions`：状态门控，例如高关注度、高拥挤、低流动性、盘口开盘意图。
- `controls`：横截面残差化控制变量，当前主要接 Barra 风格暴露。
- `direction_policy`：`fixed` 使用行为机制的经济方向，`train_ic` 允许训练集学习方向。

这个结构的好处是字段选择仍有随机性，但每个随机表达式必须先满足行为机制约束，避免把基本面、量价、盘口、资金流字段做无语义拼接。

## 4. 数据接口

模拟数据沿用原系统长表格式：

- 索引：`Datetime, Contract`
- 必备列：label、可交易标记、行业代码、Barra 风格。
- 候选字段：基本面、量价、盘口日频、资金流入。
- 元数据：
  - `field_rules`：给旧 `core.preprocess.build_transform_cache` 建缓存用。
  - `behavior_field_rules`：给行为基因采样和验证用。

真实数据接入时，只要保持同样长表结构，并补齐 `behavior_field_rules` 的 `data_family`、`behavior_roles`、`direction`、`allowed_slots`、`allowed_unary_ops` 即可。`sub_family`、`sub_type`、`unit_type`、`window`、`session`、`investor_type` 不参与当前 `behavior_gen` 的采样、校验或计算，因此实际 meta 不再保存。

## 5. 已完成调试

在 `pytorch` conda 环境下完成：

```powershell
conda run -n pytorch python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

结果确认：

- PyTorch：`2.6.0+cu124`
- CUDA：可用
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`

源码编译检查：

```powershell
conda run -n pytorch python -c "from pathlib import Path; files=[Path('behavior_gen/__init__.py'),Path('behavior_gen/gene.py'),Path('behavior_gen/sampler.py'),Path('behavior_gen/torch_backend.py'),Path('behavior_gen/ga.py'),Path('examples/run_behavior_mock_ga_gpu.py')]; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in files]; print('compiled behavior pipeline files', len(files))"
```

采样器检查：

- 11 个 mode 全部可采样。
- 80 个随机基因验证错误数为 0。
- 变异和交叉后子代都能修复成合法基因。

CUDA 单因子检查：

- 随机行为基因成功生成 `cuda:0` 张量。
- 成功计算 RankICIR、NDCG、Barra 中性化 ICIR。

端到端 GA smoke：

```powershell
conda run -n pytorch python examples\run_behavior_mock_ga_gpu.py --population-size 8 --generations 1 --train-days 180 --valid-days 40 --prefix debug_behavior_smoke
```

输出文件：

- `artifacts/results/debug_behavior_smoke_history.csv`
- `artifacts/results/debug_behavior_smoke_final_population.csv`
- `artifacts/results/debug_behavior_smoke_nsga2_rank.csv`
- `artifacts/results/debug_behavior_smoke_config.json`

注意：当前 `conda run` 会额外打印 OpenCL vendor 文件权限噪声，但命令退出码为 0，CUDA 计算已正常执行。

## 6. 常用运行命令

重新生成模拟数据：

```powershell
conda run -n pytorch python alpha_gen\scripts\data_builders\make_mock_behavior_data.py
```

运行较小调试：

```powershell
conda run -n pytorch python examples\run_behavior_mock_ga_gpu.py --population-size 16 --generations 1 --train-days 240 --valid-days 60 --prefix behavior_debug
```

运行稍大的本地搜索：

```powershell
conda run -n pytorch python examples\run_behavior_mock_ga_gpu.py --population-size 64 --generations 3 --train-days 360 --valid-days 80 --prefix behavior_local
```

## 7. 已知BUG修复记录 (2026-05-29)

### 7.1 `attention_risk` combiner + `retail_chase_risk` mode 运行时崩溃

`retail_chase_risk` mode 的 `allowed_combiners` 包含 `"attention_risk"`，但该 combiner 硬编码了 `values["attention_heat"]`（`torch_backend.py:298`），而 `retail_chase_risk` 的 slot 配置中没有 `attention_heat`。修复方式：从 `retail_chase_risk` 的 `allowed_combiners` 中移除 `"attention_risk"`。

### 7.2 `direction=0` 被 Python falsy 语义静默转为 1

`torch_backend.py:183,185` 中 `float(rule.direction or 1)` 当 direction=0 时，Python 将 0 判为 falsy，`0 or 1` 返回 1。但 `BehaviorFieldRule.from_dict` 明确允许 direction=0 表示"无方向/中性字段"。修复方式：改为 `float(rule.direction)`，direction 已被 `from_dict` 严格限制为 -1/0/1。

### 7.3 `confirm`/`gated_confirm` combiner 跨 mode 语义不兼容

`confirm`/`gated_confirm` combiner（`torch_backend.py:274-284`）内部有硬编码的 slot name 白名单 `{"fund_anchor", "flow_confirm", "price_anchor", "price_momentum"}`，只对 `fund_flow_confirmation` 和 `anchor_momentum` 语义正确。修复方式：
- `panic_reversal` 移除 `"gated_confirm"`
- `orderbook_intent` 移除 `"confirm"` 和 `"gated_confirm"`

## 8. 后续建议

1. 接真实数据前先只替换 parquet 和 metadata，不改代码，验证字段规则是否完整。
2. 每新增一类字段，先在 `behavior_field_rules` 中标注 `behavior_roles` 和 `allowed_slots`，再决定是否新增 mode。
3. 如果某类行为机制在研报中证据更强，优先新增 mode 或 combiner，而不是放宽字段随机拼接。
4. 大规模搜索前建议先用 `population_size=32, generations=1` 做 smoke，再扩大到生产参数。
5. 当前 mock 结果只证明工程链路可跑，不代表因子有效；真实有效性仍需用正式样本做滚动训练、验证和样本外组合检验。
