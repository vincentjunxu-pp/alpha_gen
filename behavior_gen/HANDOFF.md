# 行为金融学遗传挖掘框架交接

> 最后更新：2026-06-24

---

## 1. 项目边界

- `alpha_gen/core/` — 原基本面因子挖掘区，不在本轮改动中迁移。
- `alpha_gen/behavior_gen/` — **行为金融学因子挖掘区**，基因结构为 `mode + slot + combiner + condition + control`。
- `alpha_gen/behavior_tree_gp/` — 行为因子树形 GP 区（typed tree 版本）。
- `alpha_gen/free_gp_cuda/` — 自由形式 GPU GP 区，算子库来源（`operators.py`）。
- 行为框架 GPU-first：搜索、组合和评分围绕 `BehaviorTorchContext`，默认要求 CUDA。

---

## 2. 数据流全景

```
┌─────────────────────────────────────────────────────────────────┐
│  原始数据源                                                        │
│  ├─ 基本面表 (growth/valuation/financial/operating/cashflow)       │
│  ├─ 量价表 (OHLCV → 技术指标 + 衍生因子)                            │
│  ├─ 资金流表 (大单/中单/小单/特大单 买卖金额/量)                     │
│  └─ 盘口5分钟快照 (BP1/SP1/BV1-5/SV1-5)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  scripts/data_builders/basefactor.py                             │
│  ├─ 基本面：增长加速度/估值重估/传导错配/主业利润质量/现金流支撑       │
│  ├─ 量价：均线偏离/拥挤/背离/超买超卖共振/趋势确认/波动调整/反转风险   │
│  ├─ 资金流：净额/比例/大单vs散户/滚动窗口/机构累积/散户追涨           │
│  ├─ 盘口：价差/深度/不平衡/微观价格/净委买变化 (full/open30/close30) │
│  └─ 跨模块复合：flow_price_divergence, institution_accumulation,    │
│                 retail_chase_risk, orderbook_chase_risk, ...       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/metadata/production/real_behavior_metadata.json (542 字段)  │
│  ├─ field_rules: 旧系统兼容字段规则                                 │
│  └─ behavior_field_rules: 行为基因采样/验证用                       │
│      ├─ data_family: fundamental/price_volume/moneyflow/orderbook  │
│      ├─ behavior_roles: 96种行为角色                                │
│      ├─ direction: -1/0/1                                          │
│      ├─ allowed_slots: 字段可填入的语义槽位                          │
│      └─ allowed_unary_ops: 允许的一元变换 (全部15个)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  behavior_gen 遗传搜索                                             │
│  ├─ gene.py      → 基因定义、Mode/Combiner注册表、校验              │
│  ├─ sampler.py   → 模式感知随机采样、修复、变异、交叉                │
│  ├─ torch_backend.py → GPU张量计算、因子合成、评分                  │
│  └─ ga.py        → NSGA-II 主循环、训练/验证、导出                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 基因结构

```python
BehaviorGene:
  mode: str              # 行为机制模板 (15种)
  slots: dict[str, SlotGene]  # 语义槽位 → (field, unary_op)
  combiner: str          # 槽位组合方式 (13种)
  conditions: tuple[ConditionGene]  # 状态门控 (可选, 45%概率)
  direction_policy: str  # "fixed" | "train_ic"
  version: int
```

**与旧 A/B/C/D 字段表达式的核心区别**：每个随机表达式必须先满足行为机制约束（data_family + behavior_roles + allowed_slots），避免无语义拼接。

---

## 4. 算子全表

### 4.1 Unary Ops（15个）

| 算子 | 语义 | 公式 |
|------|------|------|
| `current` | 原始值 | `X` |
| `rank_pct` | 截面分位数 | `CS_RANK_PCT(X) - 0.5` |
| `zscore` | 截面标准化 | `CS_ZSCORE(X)` |
| `direction_rank` | 方向感知分位 | `DIR(X) * (CS_RANK_PCT(X) - 0.5)` |
| `direction_zscore` | 方向感知标准分 | `DIR(X) * CS_ZSCORE(X)` |
| `ind_rank_pct` | 行业内分位 | `INDUSTRY_RANK_PCT(X) - 0.5` |
| `ind_zscore` | 行业内标准分 | `INDUSTRY_ZSCORE(X)` |
| `ts_zscore_5d` | 5日时序标准分 | `TS_ZSCORE(X, 5)` |
| `ts_zscore_20d` | 20日时序标准分 | `TS_ZSCORE(X, 20)` |
| `ts_delta_5d` | 5日原始变化 | `X - DELAY(X, 5)` |
| `ts_delta_20d` | 20日原始变化 | `X - DELAY(X, 20)` |
| `ts_vol_20d` | 20日滚动波动率 | `TS_STD(X, 20)` |
| `ts_max_dd_20d` | 20日最大回撤 | `1 - X / ROLLING_MAX(X, 20)` |
| `ts_max_dd_60d` | 60日最大回撤 | `1 - X / ROLLING_MAX(X, 60)` |
| `decay_linear_20d` | 20日衰减线性加权 | `DECAY_LINEAR(X, 20)` |

> 截面系列(7)来自原系统，时序系列(8)从 `free_gp_cuda/operators.py` 迁移并扩展。
> 529 个非 control 字段**全部放开**所有 15 个算子，由 GA 自行筛选有效变换。

### 4.2 Combiners（13个）

| Combiner | 语义 | 硬编码依赖 | 适用 Mode |
|----------|------|-----------|----------|
| `rank_gap` | 通用排序差 | 无（前两槽相减） | 多数 mode |
| `residual_gap` | 通用残差 | 无（截面回归残差） | 多数 mode |
| `gated_rank_gap` | 带门控的排序差 | 无 | 多数 mode |
| `quality_gap` | 利润-现金流质量差 | `profit_growth`, `cashflow_quality` | quality_neglect |
| `crowding_interaction` | 成长×拥挤交互 | `growth_anchor`, `crowding_signal` | growth_crowding_risk |
| `confirm` | 确认型：和+乘积+额外 | 动态（前两槽为core） | fund_flow_confirmation, anchor_momentum, sentiment_momentum, earnings_surprise_drift |
| `gated_confirm` | 带门控的确认型 | 同上 | fund_flow_confirmation, anchor_momentum |
| `risk_minus_confirm` | 风险减确认 | 动态分类（风险槽集合 vs 确认槽集合） | growth_crowding_risk, retail_chase_risk, attention_overreaction, liquidity_neglect, microstructure_deterioration |
| `panic_reversal` | 恐慌反转 | `fund_anchor`, `drawdown`, `sell_pressure` | panic_reversal |
| `attention_risk` | 关注度风险 | `attention_heat` | attention_overreaction |
| `orderbook_intent` | 盘口意图 | `orderbook_pressure` | orderbook_intent |
| `liquidity_gap` | 流动性缺口 | `liquidity_stress`, `turnover_shock` | liquidity_neglect |
| `anchor_confirm` | 锚点确认 | `price_anchor` 或 `cost_anchor`, `price_momentum` | anchor_momentum, disposition_anchor |

> **重要**：`confirm`/`gated_confirm` 的核心槽位识别已改为**动态**（取 mode 定义的前两个 slot），新增 mode 不会出现双重计数 bug。
> `risk_minus_confirm` 的风险/确认分类集合需在新 mode 添加时同步更新。

### 4.3 Condition Ops（8个）

| 算子 | 语义 | 阈值 |
|------|------|------|
| `top_quantile` | 截面排名 ≥ 阈值 | 可配 0.55-0.80 |
| `bottom_quantile` | 截面排名 ≤ 1-阈值 | 可配 |
| `above_median` | 截面排名 ≥ 0.5 | 无 |
| `below_median` | 截面排名 < 0.5 | 无 |
| `positive` | 原始值 > 0 | 无 |
| `negative` | 原始值 < 0 | 无 |
| `extreme_tail` | 极端尾部 (top/bottom 3%) | 无 |
| `vol_breakout` | 波动突破 (偏离均值 > 2σ) | 无 |

> 每个基因有 45% 概率携带 conditions（`BehaviorSamplerConfig.condition_probability=0.45`）。
> 条件门控作用于 factor 层面：非门控单元设为 NaN（默认）或 0。

---

## 5. Behavior Mode 全表（15个）

### 原有 11 个

| Mode | 方向 | 描述 | 核心槽位 |
|------|------|------|---------|
| `fund_price_underreaction` | +1 | 基本面改善后价格反应不足 | fund_anchor, price_reaction |
| `quality_neglect` | -1 | 利润增长与现金流质量背离 | profit_growth, cashflow_quality, price_reaction(opt) |
| `growth_crowding_risk` | -1 | 成长叙事叠加成交拥挤的回撤风险 | growth_anchor, crowding_signal, fund_support(opt) |
| `fund_flow_confirmation` | +1 | 基本面改善获大额资金确认 | fund_anchor, flow_confirm, orderbook_filter(opt), price_control(opt) |
| `retail_chase_risk` | -1 | 散户追涨但大资金不确认 | price_momentum, retail_flow, close_chase, large_flow(opt) |
| `panic_reversal` | +1 | 基本面尚可但恐慌/流动性冲击错杀 | fund_anchor, drawdown, sell_pressure, orderbook_filter(opt) |
| `attention_overreaction` | -1 | 极端收益/成交异常造成过度反应 | attention_heat, price_momentum(opt), fund_support(opt) |
| `orderbook_intent` | +1 | 盘口买卖压力刻画未成交交易意愿 | orderbook_pressure, liquidity_stress(opt), price_reaction(opt) |
| `liquidity_neglect` | -1 | 表面成交活跃但盘口流动性压力高 | liquidity_stress, turnover_shock, flow_confirm(opt) |
| `anchor_momentum` | +1 | 52周高点/成本锚定下的动量延续 | price_anchor, price_momentum, flow_confirm(opt), orderbook_filter(opt) |
| `disposition_anchor` | +1 | 价格相对成本锚剥离动量后的处置效应 | cost_anchor, price_momentum, fund_support(opt) |

### 新增 4 个（2026-06-24）

| Mode | 方向 | 描述 | 核心槽位 | 数据基础 |
|------|------|------|---------|---------|
| `sentiment_momentum` | +1 | 市场情绪能量与价格动量的共振 | sentiment_energy, price_momentum, volume_confirm(opt) | 20个 sentiment_pressure 字段 (AR/BR/CR/PCNT 等) |
| `microstructure_deterioration` | -1 | 盘口微观结构恶化：价差扩大、深度衰减 | spread_stress, depth_drain, imbalance_divergence(opt) | 35个 orderbook 字段 |
| `earnings_surprise_drift` | +1 | 盈余加速度后价格反应不足的滞后补涨 | earnings_accel, price_reaction, institution_flow(opt) | growth_accel + revision 字段 |
| `volatility_cascade` | -1 | 高波动→风控减仓→流动性挤压的级联反馈 | volatility_shock, liquidity_stress, drawdown(opt) | 19个 volatility 字段 |

---

## 6. 中性化策略

`BehaviorGAConfig.neutralization_mode`：

| 模式 | 含义 |
|------|------|
| `raw_full_barra_industry` | 原因子不做强制中性化；中性化 RIC 使用 10 Barra 风格 + 行业 |
| `size_then_industry` | 先对 size 截面回归，再行业内去均值；不计算 Barra 中性化指标 |
| `raw_none` | 不做任何因子层面中性化，也不计算中性化指标 |

默认使用 `raw_full_barra_industry`。训练和验证必须使用同一个 `neutralization_mode`。

---

## 7. NSGA-II 目标

| 模式 | 最大化目标 |
|------|----------|
| `rir_long_rir_ndcg` | RIR, Long RIR, NDCG@k |
| `rir_long_rir` | RIR, Long RIR |
| `rir_long_rir_neutralized_rir` | RIR, Long RIR, 中性化 RIR |

默认三目标版本。

---

## 8. 文件索引

```
behavior_gen/
├── __init__.py           # 公开 API
├── gene.py               # 基因定义、15 Mode、15 Unary Op、13 Combiner、8 Condition Op
├── sampler.py            # 随机采样、修复、变异、交叉
├── torch_backend.py      # GPU 张量计算后端
├── ga.py                 # NSGA-II 主循环、训练/验证/导出
├── result_frequency_analysis.py  # 结果频率分析
└── HANDOFF.md            # 本文档

data/metadata/production/
└── real_behavior_metadata.json  # 生产元数据 (542 字段)

data/metadata/fixtures/
└── mock_behavior_metadata.json  # 调试用模拟数据元数据 (77 字段)

scripts/data_builders/
└── basefactor.py         # 全量因子构造逻辑（基本面/量价/资金流/盘口）

examples/
└── run_behavior_mock_ga_gpu.py  # GPU 端到端运行入口
```

---

## 9. 常用运行命令

```powershell
# 重新生成模拟数据
conda run -n pytorch python alpha_gen\scripts\data_builders\make_mock_behavior_data.py

# 小规模调试
conda run -n pytorch python examples\run_behavior_mock_ga_gpu.py --population-size 16 --generations 1 --train-days 240 --valid-days 60 --prefix behavior_debug

# 中等规模本地搜索
conda run -n pytorch python examples\run_behavior_mock_ga_gpu.py --population-size 64 --generations 3 --train-days 360 --valid-days 80 --prefix behavior_local

# GPU 可用性检查
conda run -n pytorch python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 10. 已知 BUG 修复记录

### 2026-06-24

#### confirm/gated_confirm 双重计数
`confirm` combiner 的 skip set 硬编码为 `{fund_anchor, flow_confirm, price_anchor, price_momentum}`，新增 mode（`sentiment_momentum`、`earnings_surprise_drift`）的 slot 名不在此集合中，导致核心槽位被重复加权 (1.0x + 0.5x = 1.5x)。
**修复**：改为动态取 mode 前两个 slot 名作为 core_slots。

#### risk_minus_confirm 语义降级
`risk_names`/`confirm_names` 硬编码集合不含新 slot 名（`spread_stress`、`depth_drain` 等），导致新 mode 使用此 combiner 时退化为简单求和。
**修复**：扩展集合覆盖新增 slot。

#### microstructure_deterioration + liquidity_gap 冲突
`liquidity_gap` 硬编码 `turnover_shock` slot，但 `microstructure_deterioration` 没有此 slot。
**修复**：从该 mode 的 `allowed_combiners` 中移除 `liquidity_gap`。

#### volatility_cascade + crowding_interaction / liquidity_gap 冲突
`crowding_interaction` 硬编码 `growth_anchor`+`crowding_signal`，`liquidity_gap` 硬编码 `turnover_shock`，均不存在于 `volatility_cascade`。
**修复**：该 mode 仅保留 `rank_gap`。

#### downside_volume_panic direction 错误
`downside_volume_panic` 构造为 `cs_rank(MDD20.abs()) * cs_rank(volume_crowding_5_20)`，值越大恐慌越高，但 direction 误标为 1。
**修复**：direction 1 → -1。

#### ob_total_depth 缺 liquidity_stress slot
两个深度字段有 `liquidity` 角色但缺少 `liquidity_stress` slot。
**修复**：补上 slot。

#### liquidity_neglect 字段 family 错误
该字段为 `cs_rank(ob_liquidity_stress) - cs_rank(turnover_shock_5_20)`（跨 orderbook+price_volume 复合），但误标为 orderbook。
**修复**：family orderbook → price_volume。

#### BBI/DKX/DPO/MATRIX 缺锚点 slot
4个锚点类技术指标有 `anchor`/`price_anchor` 角色但未映射到 `price_anchor`/`cost_anchor` slot。
**修复**：补上 slot。

### 2026-05-29

#### attention_risk combiner + retail_chase_risk mode 崩溃
`retail_chase_risk` mode 的 `allowed_combiners` 包含 `attention_risk`，该 combiner 硬编码 `values["attention_heat"]`，但 mode 的 slot 配置中没有此槽位。
**修复**：从 `retail_chase_risk` 的 `allowed_combiners` 中移除 `attention_risk`。

#### direction=0 被 Python falsy 静默转为 1
`torch_backend.py` 中 `float(rule.direction or 1)` 在 direction=0 时错误返回 1。
**修复**：改为 `float(rule.direction)`。

#### confirm/gated_confirm 跨 mode 语义不兼容
`confirm` combiner 白名单 `{fund_anchor, flow_confirm, price_anchor, price_momentum}` 只对 `fund_flow_confirmation` 和 `anchor_momentum` 正确。
**修复**：`panic_reversal` 移除 `gated_confirm`；`orderbook_intent` 移除 `confirm` 和 `gated_confirm`。

---

## 11. 后续建议

1. **接真实数据**：先替换 parquet 和 metadata，不改代码，验证字段规则完整。
2. **新增字段**：先在 `behavior_field_rules` 中标注 `behavior_roles` 和 `allowed_slots`。
3. **新增 mode**：在 `gene.py:MODE_REGISTRY` 中定义，同步检查 combiner 的硬编码依赖。
4. **新增 combiner**：需要同步改 `gene.py`（CHOICES + formula + MODE_REGISTRY）和 `torch_backend.py`（_combine_behavior_gene 实现）。
5. **新增 unary op**：改 `gene.py`（CHOICES + _feature_formula）和 `torch_backend.py`（_feature 实现）。
6. **大规模搜索前**：`population_size=32, generations=1` 做 smoke → 扩大到生产参数。
7. **Mock 结果仅表示工程链路可跑**，真实有效性需正式样本滚动训练+样本外检验。
