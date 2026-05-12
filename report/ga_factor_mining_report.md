# 基于遗传算法的 TMT 基本面因子挖掘项目汇报

## 0. 摘要

对 `TMT 股票池`的`基本面因子`进行结构化遗传算法搜索，挖掘对未来收益有稳定解释力的因子表达式。遗传算法训练阶段以未来 **20 个交易日收益** 的截面 **RankIC** 表现作为核心目标之一，同时结合 IC 胜率和 NDCG 约束；正式回测阶段使用 **2021-01-01 至 2024-01-01** 作为验证期，并与 label 周期保持一致，采用 **20 个交易日调仓频率**。

当前结果显示，除第 0 个因子方向为负以外，其余入选因子大多呈现稳定正向 RankIC、明显的多空分层和较高的 long-short 夏普。表现最突出的因子集中在盈利质量、盈利能力相对估值、营运资本残差、递延收入/偿债现金流残差等方向。



- 训练与回测口径已经统一为 **未来 20D label + 20D 调仓**，避免了 label 周期与交易周期不一致的问题。
- 回测中大部分正向因子 `ric_20d` 集中在 **0.085 到 0.103**，`rir_20d` 多数高于 **0.70**。
- 多空组合表现较强，多个因子 `longshort_sharpe` 超过 **7**，其中因子 2、3、10 的多空表现最突出。
- 分组收益具备清晰单调性，高分组明显跑赢低分组，说明因子排序信息有效。

## 1. 项目思路与流程

通过参数化遗传算法，在基本面字段上搜索结构化表达式，并在表达式生成后统一进行市值中性化。

### 1.1 表达式搜索空间

当前遗传算法支持以下五类表达式结构：

| 模式 | 表达式结构 | 经济含义 |
|---|---|---|
| `single` | `A` | 单字段信号，例如利润、现金流、资本开支等 |
| `ratio` | `A / B` | 相对强度或估值归一化 |
| `pair_ratio` | `(A +/- B) / (C +/- D)` | 复合比值结构 |
| `resi` | `residual(A ~ B)` | A 对 B 的截面回归残差，用于提取超额信息 |
| `ratio_product` | `(A / B) * (C / D)` | 两组相对信号的乘积 |

算子包括：

| 变换 | 含义 |
|---|---|
| `current` | 当前值 |
| `log` | 对数变换 |
| `zscore` | 截面标准化 |
| `diff_2q` / `diff_1y` | 环比/同比变化 |
| `pct_2q` / `pct_1y` | 环比/同比增长率 |
| `std_2q` / `std_1y` | 过去窗口波动 |

所有表达式最后统一做`市值中性化`

### 1.2 训练目标

遗传算法训练阶段使用 NSGA-II 多目标选择，核心目标为：

| 目标 | 含义 |
|---|---|
| `abs_rank_ic` | 因子与未来 20D 收益的绝对 RankIC |
| `ic_win_rate` | 日度 RankIC 与方向一致的比例 |
| `ndcg_at_k` | 因子对头部收益股票的排序质量 |

本项目的重点是基本面因子，因此使用 **20D 未来收益** 作为主 label，避免 1D 噪声过大，也与华泰遗传规划选股因子研究中使用未来 20 个交易日收益作为优化目标的思路一致。

### 1.3 训练集与回测集切分

训练集：
```text
2016-01-01 <= date < 2020-12-31
```
验证回测期：

```text
2021-01-01 <= date < 2024-01-01
```

label 定义：未来20D的收益率

```python
label_20d = close.shift(-21) / close.shift(-1) - 1
```





## 3. 遗传算法搜索结果概览

### 3.1 不同表达式模式的表现

![alt text](image.png)

从 gene mode 的箱线图看：

- `single`、`resi`、`ratio` 的 `valid_abs_rank_ic` 中位数更高，稳定性也更好。
- `pair_ratio` 和 `ratio_product` 的表达式更复杂，但验证集表现不一定更强，说明过度复杂结构可能带来过拟合。
- `pair_ratio`: 一些算子把不相关的财务指标放在同一个加减结构里，量纲错配，解释性以及效果较差。 


后续搜索可以适当提高 `single`、`resi`、`ratio` 的采样权重，降低过复杂表达式在最终候选中的比例。删除该`pair_ratio`

## 4. 入选因子解释

| 因子表达式 | 经济解释 | ric_1d | ric_5d | ric_20d | rir_20d | longshort_sharpe | pnl_longshort_ann | pnl_longshort_max_drawdown | 综合评价&使用建议 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `市值中性化(market_cap / gross_profit)` | 单位毛利对应市值越高，代表估值越贵 | -0.028 | -0.055 | -0.091 | -0.914 | -5.830 | -0.697 | -0.076 | 全指标为负，方向反向使用，整体表现差 |
| `市值中性化(zscore(r_n_d) / book_to_market_ratio_lf)` | 研发投入相对账面市值比，刻画成长投入与估值折价的结合 | 0.027 | 0.056 | 0.100 | 0.811 | 7.634 | 1.130 | -0.064 | 强正向，IC与夏普稳定，可作成长质量核心因子 |
| `市值中性化(zscore(gross_profit) 对 working_capital_lf 截面回归取残差)` | 剔除营运资本解释后的毛利超额水平 | 0.028 | 0.057 | <span style="color:red;font-weight:bold">0.103</span> | 0.722 | 9.199 | 1.310 | -0.045 | IC、夏普、收益拔尖，回撤控制优秀，综合最优核心候选 |
| `市值中性化(zscore(net_profit_deduct_non_recurring_pnl) / sp_ratio_lyr)` | 扣非净利润相对销售估值，强调主营盈利能力 | 0.029 | 0.059 | 0.102 | <span style="color:red;font-weight:bold">0.873</span> | 9.101 | 1.100 | <span style="color:red;font-weight:bold">-0.034</span> | IC稳定性强、全场回撤最低，收益夏普顶尖，重点纳入核心组合 |
| `市值中性化(zscore(gross_profit) / sp_ratio_ttm)` | 毛利能力相对 PS 估值 | <span style="color:red;font-weight:bold">0.030</span> | <span style="color:red;font-weight:bold">0.060</span> | <span style="color:red;font-weight:bold">0.103</span> | 0.865 | 7.639 | 1.049 | -0.053 | IC顶级，盈利估值逻辑扎实信号稳定，核心稳健因子 |
| `市值中性化(super_quick_ratio_lyr / ps_ratio_ttm)` | 超速动比率相对 PS，偏防御和偿债能力 | 0.018 | 0.030 | 0.040 | 0.567 | 6.624 | 0.722 | -0.038 | IC偏弱，收益夏普尚可，仅作辅助防御类配置 |
| `市值中性化(zscore(ebit_ttm) / sp_ratio_ttm)` | 经营利润相对销售估值 | 0.029 | <span style="color:red;font-weight:bold">0.060</span> | <span style="color:red;font-weight:bold">0.103</span> | 0.850 | 6.744 | 0.931 | -0.063 | 正向有效，但弱于2/3/4，作为次选盈利估值因子 |
| `市值中性化(profit_before_tax 对 zscore(working_capital_lyr) 截面回归取残差)` | 剔除营运资本后的税前利润超额水平 | 0.026 | 0.053 | 0.093 | 0.672 | 6.952 | 1.010 | -0.080 | 整体表现中上，回撤偏高，轻度纳入并控制权重 |
| `市值中性化(net_profit_deduct_non_recurring_pnl 对 working_capital_lyr 截面回归取残差)` | 剔除营运资本后的扣非利润质量 | 0.025 | 0.052 | 0.091 | 0.708 | 7.183 | 1.017 | -0.054 | 各指标稳定中上，适合作为盈利质量补充因子 |
| `市值中性化(zscore(total_equity) / account_receivable_turnover_rate_ttm)` | 净资产规模相对收款周转，反映资产质量和经营效率 | 0.028 | 0.057 | 0.100 | 0.735 | 6.588 | 1.056 | -0.080 | 正向有效，易与规模类因子重叠，需冗余剔除后使用 |
| `市值中性化(deferred_revenue 对 zscore(cash_paid_for_debt) 截面回归取残差)` | 剔除偿债现金流后的递延收入超额，代表订单/预收质量 | 0.021 | 0.046 | 0.091 | 0.724 | <span style="color:red;font-weight:bold">9.500</span> | <span style="color:red;font-weight:bold">1.305</span> | -0.049 | 多空夏普、多头收益夏普全场顶尖，回撤优异，多空&多头最优候选 |
| `市值中性化(working_capital_ttm 对 goodwill 截面回归取残差)` | 剔除商誉影响后的营运资本质量 | 0.027 | 0.053 | 0.092 | 0.781 | 7.109 | 1.066 | -0.096 | 逻辑有效但全场回撤最高，需严格降低配置权重 |
| `市值中性化(zscore(market_cap_3) 对 cash_ratio_ttm 截面回归取残差)` | 剔除现金比率后的规模残差信号 | 0.019 | 0.044 | 0.085 | 0.718 | 7.882 | 1.012 | <span style="color:red;font-weight:bold">-0.042</span> | 回撤极低表现稳健，偏规模风格，适配低波动组合 |
| `市值中性化(zscore(market_cap_3) 对 return_on_invested_capital_ttm 截面回归取残差)` | 剔除 ROIC 后的规模残差信号 | 0.019 | 0.044 | 0.085 | 0.717 | 8.076 | 1.0 | -0.042 | 指标与12高度相近，风格冗余，二选一保留并做去共线性 |



### 5.1 因子回测图像



### 1. `residual(deferred_revenue ~ zscore(cash_paid_for_debt))`:
该因子将递延收入对偿债现金流做回归并取残差，剥离了依靠“借新还旧”维持运转的重资产或伪成长企业。能偶提纯出纯靠硬核产品力让客户提前充值掏钱的优质SaaS或游戏龙头，是TMT核心商业模式的很好体现。

![alt text](image-1.png)

### 2.  `residual(zscore(gross_profit) ~ working_capital_lf)`
将截面标准化的毛利对营运资本取残差，旨在寻找用极少的资金占用撬动了极高毛利的企业。它高度契合TMT的轻资产属性，直接反映了企业在产业链中极强的议价权（如先款后货）与健康的现金流运转效率。
![alt text](image-2.png)

### 3.  `zscore(r_n_d) / book_to_market_ratio_lf`
等价于“标准化研发投入乘以市净率（PB）”，将技术投入壁垒与市场估值溢价结合在了一起。它能有效捕获研发势能强劲且已被资金面认可的科技白马，属于“高质量成长与动能”复合因子。
![alt text](image-3.png)
### 4.  `zscore(net_profit_deduct_non_recurring_pnl) / sp_ratio_lyr`
因子将扣非净利润与市销率（PS）进行动态平衡，完美适应了TMT早期企业利润波动大而常以PS估值的体系。它能够极其稳健地筛选出主业利润丰厚，但相对于其庞大营收规模而言估值尚未泡沫化的中坚力量。
![alt text](image-4.png)

### 5.  `residual(working_capital_ttm ~ goodwill)`
由于TMT行业（尤其是传媒与IT服务）常年存在大量外延式并购，财务并表极易导致营运资本虚高。该因子通过剔除商誉影响后的营运资本，能有效挤出并购带来的虚假繁荣水分，锁定内生造血能力极强的真实龙头企业。
![alt text](image-5.png)











