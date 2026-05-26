# 基于遗传算法的全行业基本面因子挖掘项目汇报

## 0. 摘要

本项目在全行业股票池上使用结构化遗传算法挖掘基本面因子。因子表达式由预设的可解释模板生成，训练阶段使用 NSGA-II 多目标优化，并在评估中加入 Barra 风格暴露约束，目标是在保留原始预测能力的同时，降低因子对常见风格因子的依赖。

本轮全行业实验的核心结论如下：

- 挖掘出的有效表达式主要集中在 `resi ` 等截面残差类结构。相比直接比值、复合打分等结构，残差类表达式在 Barra 中性化后保留的 IC 更稳定。
- 候选因子最常暴露的 Barra 风格集中在 `barra_earnings_yield`、`barra_book_to_price` 和 `barra_leverage`，说明基本面因子大量信号与盈利收益率、账面价值和杠杆结构存在重叠。
- 最终入选 9 个因子，筛选流程为：训练集 IC/IR 初筛、因子相关性聚类、每个相关性簇内选取表现最好的代表因子。
- 从累计 IC 和分层回测结果看，入选因子整体具备较强单调性和稳定性，多头组表现明显优于低分组，说明因子排序信息有效。

## 1. 研究框架

### 1.1 数据范围

本轮实验使用全行业股票池，输入字段主要为财务报表、财务衍生指标、估值指标。标签使用未来收益，主要评估周期为 20 个交易日。

为了避免行业和规模暴露污染结果，因子生成后统一执行：

```text
行业中性化 -> barra_size 市值中性化 -> 动态 Barra 风格暴露评估
```


### 1.2 表达式搜索空间

当前遗传算法主要搜索以下结构：

| 模式 | 表达式结构 | 含义 |
|---|---|---|
| `single` | `A` | 单字段信号 |
| `ratio` | `A / B` | 相对估值或相对强度 |
| `pair_ratio` | `(A +/- B) / (C +/- D)` | 复合比值 |
| `resi` | `residual(A ~ B)` | 剔除控制变量后的截面残差 |
| `resi_pair` | `residual(A ~ B + C)` | 双控制变量残差 |
| `multi_resi` | `residual(A ~ B + C + D)` | 多控制变量残差 |
| `style_composite` | `rank_score(A) + rank_score(B)` | 两个可解释风格信号的排序打分组合 |

支持的字段变换包括 `current`、`log`、`rank_pct`、`zscore`、`ind_rank_pct`、`ind_zscore`、`diff_2q / diff_1y`、`pct_2q / pct_1y`。其中 `log` 使用 `sign(x) * log(1 + abs(x))`，避免负值财务字段导致无效。

## 2. NSGA-II 目标与 Barra 约束

本轮优化不只追求原始 IC，而是同时考虑因子在 Barra 风格剥离后的保真度。训练目标包括：

| 目标 | 含义 |
|---|---|
| `rank_ic_ir` | 原始因子的 RankICIR |
| `ndcg_at_k` | 原始因子对头部收益股票的排序质量 |
| `neutralized_icir` | 动态 Barra 风格剥离后的 RankICIR |

动态 Barra 风格剥离逻辑为：

1. 对新生成因子计算其与 9 个 Barra 风格因子的截面相关性（统一对`size`风格做了中性化）。
2. 取平均`person`相关性超过0.3的风格。
3. 选择相关性最高的最多 2 个 Barra 风格。
4. 对原因子做每日截面回归，取残差因子。
5. 用残差因子的 RankICIR 作为第三个优化目标。



## 3. 搜索结果分析

### 3.1 Gene Mode 表现

从 raw IC 的箱线图看，`style_composite` 的原始 IC 水平较高，说明简单风格打分本身能捕捉一部分收益排序信息。但从 Barra 中性化后的 IC 分布看，`style_composite` 的优势明显下降，而 `resi` 的中性化后表现更稳定。
![alt text](image-6.png)

![alt text](image-7.png)
这说明：

- 简单打分和比值类表达式更容易带有明显 Barra 暴露。
- 残差类表达式天然具备“先剥离一个财务解释变量，再观察剩余部分”的结构，更容易提取非标准风格之外的 Alpha。
- `ratio / pair_ratio` 依赖字段量纲匹配，后续应继续保持严格约束，搜索空间被大幅压缩。

因此，本轮全行业结果支持后续提高残差类表达式权重，同时保留少量 `style_composite` 作为可解释风格补充。

### 3.2 Barra 暴露结构
![alt text](image-8.png)
候选因子的 Barra 暴露频率主要集中在：

| Barra 风格 | 观察结论 |
|---|---|
| `barra_earnings_yield` | 暴露频率最高，说明盈利收益率是基本面因子中最常见的隐含来源 |
| `barra_book_to_price` | 暴露频率第二，说明估值类字段容易与账面价值风格重叠 |
| `barra_leverage` | 暴露频率较低但仍然出现，主要来自债务结构、权益比例和费用率相关表达式 |

这一结果符合基本面因子的特征：财务指标中大量字段本身就和盈利、估值、杠杆高度相关。如果不做 Barra 暴露识别，训练集高 IC 很容易来自已有风格因子的重复表达。

## 4. 最终因子筛选流程


1. 训练集 IC/IR 初筛，保留预测稳定性较强的候选因子。
2. 对候选因子做相关性聚类，控制同类表达式重复入选。
3. 在每个相关性簇内选择综合表现最好的代表因子。

`最终每个因子之间的相关性控制在0.5以下`



## 5. 因子的经济解释
![alt text](image-19.png)





## 6. 回测结论
![alt text](image-18.png)

从累计 RankIC 图像和分层回测结果看，9 个因子整体具备以下特征：

- 累计 IC 曲线整体向上，说明因子在较长时间区间内具备持续预测能力。
- 分层收益具有较好单调性
- 多头组表现强于中低分组
- 单因子年化多头收益多数为正，换手率整体可控。

`两个具体因子的表现`
`市值中性化(ind_zscore(net_profit_deduct_non_recurring_pnl) 对 current_asset_to_total_asset_lf 截面回归取残差)`
![alt text](image-10.png)


 `市值中性化(ind_rank_pct(net_profit_deduct_non_recurring_pnl) 对 (diff_1y(book_value_per_share_ttm) + diff_1y(diluted_earnings_per_share_lyr)) 截面回归取残差)` | 
![alt text](image-11.png)


`模型合成`
对9个因子用lgb滚动训练
![alt text](image-17.png)


## 7.聚焦TMT行业
按照同样的框架对TMT行业挖掘因子，最终得到31个因子，按同样的方法合成信号

![alt text](image-20.png)

