# behavior_tree_gp 交接文档

## 目标

新增一个独立的 gplearn 树状 GP 侦察框架，用于在“基本面 + 量价”字段上先自由搜索行为金融学因子的结构，再把高频出现且可解释的结构归纳回主 GA 的结构化 mode。

这个目录不替代现有 `core/ga.py` 和 `core/gene.py`。它的定位是 mode discovery / scout。

## 新增内容

- `operators.py`
  - 封装 gplearn 可用的点式安全算子，并补齐《遗传算法赋能交易行为因子》里的量化算子。
  - gplearn 树内点式算子包含 `qdiv`、`qinv`、`slog`、`ssqrt`、`qclip`、`sign`。已删除 `tanh/qtanh`，因为它没有明确量化经济含义。
  - DataFrame 级量化算子包含：
    - 横截面：`cs_rank`、`cs_zscore`、`rank_add`、`ols_residual`
    - 时序：`ts_sum`、`ts_mean`、`ts_median`、`ts_std`、`ts_corr`、`delay`
    - 切割：`rolling_selmean_top`、`rolling_selmean_btm`、`rolling_selmean_diff`
    - 横截面和时序结合：`ts_max_to_min`、`ts_meanrank`
    - 逻辑判断：`sign`、`diff_sign`
  - 横截面 rank、回归、rolling、切割这类算子不能直接放进 gplearn function，因为 gplearn 函数只能看到扁平样本向量；这些算子在 primitives 阶段预计算。

- `features.py`
  - 生成行为金融 primitives。
  - 覆盖价格动量/反转、波动率、下行波动、彩票偏好、成交异常、换手拥挤、历史高低点锚定、VWAP 锚定、行业相对收益/换手、基本面横截面 zscore/rank/delta/pct。
  - 进一步加入报告算子生成的 primitives：
    - `rolling_selmean_diff`：按收盘价/成交量/换手切割收益率或日内振幅
    - `ts_max_to_min`：交易情绪不稳定性代理
    - `ts_corr`：收益和成交变化的时序相关
    - `ts_meanrank`：横截面排序后的时序均值
    - `diff_sign`：相对历史均值的状态转化
    - `rank_add`：切割算子和不稳定性算子的复合
    - `ols_residual`：收益相对换手的横截面残差
  - 输出 `features` 矩阵字典和 `FeatureSpec` catalog，方便追踪每个输入变量的经济含义。

- `scoring.py`
  - 复用主框架 `evaluate_factor` 做验证集 RankICIR、NDCG、coverage。
  - 增加一个轻量的日频截面线性残差化函数，可对 `barra_*` 暴露做验证阶段中性化评分。

- `run_tree_gp.py`
  - 完整运行入口。
  - 默认读取 `data/panels/mock_tmt_daily.parquet` 和 `data/metadata/fixtures/mock_tmt_metadata.json`。
  - 使用 date-aware RankICIR 作为 gplearn 训练 fitness。
  - 通过 `parsimony_coefficient`、较浅 `init_depth`、hoist mutation，以及验证后 `max_program_length` / `max_program_depth` 标记来控制复杂度。
  - 输出：
    - `artifacts/results/behavior_tree_gp/tree_gp_programs.csv`
    - `artifacts/results/behavior_tree_gp/feature_catalog.csv`

## 运行方式

需要先安装 gplearn：

```powershell
conda run -n pytorch pip install gplearn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

从 `E:\实习` 执行：

```powershell
python -m alpha_gen.behavior_tree_gp.run_tree_gp
```

小规模调试可以用：

```powershell
python -m alpha_gen.behavior_tree_gp.run_tree_gp --population-size 80 --generations 2 --n-components 5
```

## 复杂度控制

当前有两层控制：

1. 训练期软约束
   - `--parsimony-coefficient`
   - `--init-depth-min`
   - `--init-depth-max`
   - 较高的 `p_hoist_mutation`

2. 验证期硬标记
   - `--max-program-length`
   - `--max-program-depth`
   - 输出 CSV 中的 `passes_complexity`

gplearn 本身没有稳定的全局 hard max depth 参数，所以当前不会在进化过程中强行拒绝超限树，而是在验证和后续筛选阶段标记。

## 和主 GA mode 的关系

建议流程：

1. 用本目录跑树状 GP。
2. 看 `tree_gp_programs.csv` 中通过复杂度限制且验证集 RankICIR / 中性化 RankICIR 较高的表达式。
3. 人工或脚本归纳常见结构，例如：
   - `sub(fundamental_change, ret_60d)` -> `underreaction`
   - `mul(max_ret_20d, qinv(csz_market_cap))` -> `lottery`
   - `sub(volume_ratio_20d, csz_roe)` -> `sentiment_gap`
   - `qdiv(ret_5d, volume_ts_z_20d)` -> `liquidity_pressure`
4. 将这些结构固化到 `core/gene.py` 的新 mode，而不是直接把复杂树作为最终生产因子。

## 当前限制

- gplearn 未安装时，入口会给出安装提示并退出。
- 训练 fitness 是扁平样本上的 date-aware RankICIR，自定义 fitness 捕获了训练样本日期，因此建议 `--n-jobs 1` 起步。
- 量化时序、横截面回归、切割算子已经预计算为 primitives；gplearn 树内不支持动态 rolling/rank/OLS。
- 中性化评分目前是 pandas/NumPy 版本，适合验证 scout 结果，不是 GPU 主流程。
- 默认 primitives 面向当前 mock TMT 数据字段；换真实数据时需要确认字段名包含 `open/high/low/close/volume/amount/turnover` 以及 metadata 中的基本面规则。

## 后续建议

- 增加 program parser，把 gplearn 表达式自动归类到候选 mode。
- 增加 walk-forward 多窗口验证，避免单窗口过拟合。
- 对输出 program 做相似度去重，减少同义树。
- 将验证评分扩展为 raw RankICIR、NDCG、中性化 RankICIR、turnover、coverage、complexity 的综合排序。
