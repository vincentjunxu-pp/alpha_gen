"""
量化分析工具函数库
包含因子相关性处理、横截面相关系数计算等工具函数
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from alpha_factory.factor.evaluation import (
    FactorQualityAnalyzer,
    ICAnalyzer,
    PNLAnalyzer,
    CorrAnalyzer,
)


class BatchFactorEvaluator:
    """
    批量因子回测评估框架
    用于将长表中的多个因子逐一进行 质量、IC、PNL 及 风险相关性测试，并汇总结果。
    """

    def __init__(self, open_mtx, tradeable_mtx, label_mtx, risk_factor_bank, risk_factor_pnl_bank, figure=False):
        """
        初始化：将所有因子共用的底层行情与风险数据保存在类中，避免重复加载。
        """
        self.open_mtx = open_mtx
        self.tradeable_mtx = tradeable_mtx
        self.label_mtx = label_mtx
        self.risk_factor_bank = risk_factor_bank
        self.risk_factor_pnl_bank = risk_factor_pnl_bank
        self.figure = figure  # 决定是否需要图像

        # 预先实例化 Analyzers（如果它们内部没有状态冲突，可以复用）
        self.fq_analyzer = FactorQualityAnalyzer()
        self.ic_analyzer = ICAnalyzer(label_mtx=self.label_mtx)
        self.pnl_analyzer = PNLAnalyzer(trade_price_mtx=self.open_mtx, trade_freq=1)
        self.risk_analyzer = CorrAnalyzer(
            factor_bank=self.risk_factor_bank,
            factor_pnl_bank=self.risk_factor_pnl_bank,
        )

    def evaluate_single_factor(self, factor_name: str, factor_mtx_1day: pd.DataFrame) -> dict:
        """
        对单个宽表因子矩阵执行完整的评估流水线，返回各类指标的字典。
        """
        res_summary = {"factor_name": factor_name}

        # ========================================
        # 1. 完整性检验（Factor Quality）
        # ========================================
        fq_res_dict = self.fq_analyzer.analyze(
            factor_mtx=factor_mtx_1day,
            tradeable_mtx=self.tradeable_mtx,
            return_figure=self.figure,
        )
        res_summary.update({
            "inf_ratio_mean": fq_res_dict.get("inf_ratio_mean"),
            "outlier_ratio_mean": fq_res_dict.get("outlier_ratio_mean"),
            "coverage_mean": fq_res_dict.get("coverage_mean"),
            "unique_ratio_mean": fq_res_dict.get("unique_ratio_mean"),
        })

        # ========================================
        # 2. IC 检验
        # ========================================
        ic_res_dict_all = self.ic_analyzer.analyze(
            factor_mtx=factor_mtx_1day.astype(float),
            tradeable_mtx=self.tradeable_mtx,
            return_figure=self.figure,
        )
        # 提取全历史(all)的IC指标
        eval_metrics_ic = [
            "ic_1d", "ir_1d", "ric_1d", "rir_1d", "long_ic_1d", "long_ric_1d",
            "ic_5d", "ir_5d", "ric_5d", "rir_5d", "long_ic_5d", "long_ric_5d",
            "ic_20d", "ir_20d", "ric_20d", "rir_20d", "long_ic_20d", "long_ric_20d",
        ]

        for metric in eval_metrics_ic:
            # 根据原代码逻辑，整体指标通常直接通过 metric name 获取
            res_summary[metric] = ic_res_dict_all.get(metric)

        # ========================================
        # 3. PNL 评估
        # ========================================
        pnl_res_dict = self.pnl_analyzer.analyze(
            factor_mtx=factor_mtx_1day,
            n_groups=5,
            tradeable_mtx=self.tradeable_mtx,
            return_figure=self.figure,
        )
        eval_metrics_pnl = [
            "pnl_long_ann", "long_sharpe", "long_fitness", "long_turnover", "pnl_long_max_drawdown",
            "pnl_longshort_ann", "longshort_sharpe", "longshort_fitness", "longshort_turnover", "pnl_longshort_max_drawdown",
            # "fig_pnl_grouped_cumsum", "fig_pnl_yearly_grouped_mean", "fig_pnl_longshort_cumsum"
        ]
        for metric in eval_metrics_pnl:
            res_summary[metric] = pnl_res_dict.get(metric)

        if self.figure:
            # ---- 自动发现所有 horizon（1d/5d/20d...）----
            horizons = sorted(
                {
                    key.replace("ric_series_", "")
                    for key in ic_res_dict_all
                    if key.startswith("ric_series_") and "long_" not in key
                }
            )
            n = len(horizons)
            if n > 0:
                fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
                if n == 1:
                    axes = [axes]

                for i, horizon in enumerate(horizons):
                    ax = axes[i]

                    # RIC
                    s = ic_res_dict_all.get(f"ric_series_{horizon}")
                    if s is not None:
                        s.cumsum().plot(ax=ax, label="RIC", linewidth=1.5)

                    # IC
                    s = ic_res_dict_all.get(f"ic_series_{horizon}")
                    if s is not None:
                        s.cumsum().plot(ax=ax, label="IC", linewidth=1.5)

                    # Long RIC (top 50%)
                    s = ic_res_dict_all.get(f"long_ric_series_{horizon}")
                    if s is not None:
                        s.cumsum().plot(
                            ax=ax, label="Long RIC", linewidth=1.5, linestyle="--"
                        )

                    # Long IC (top 50%)
                    s = ic_res_dict_all.get(f"long_ic_series_{horizon}")
                    if s is not None:
                        s.cumsum().plot(
                            ax=ax, label="Long IC", linewidth=1.5, linestyle="--"
                        )

                    ax.set_title(f"Cumulative IC — {horizon}")
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.grid(True)
                    ax.legend()

                plt.tight_layout()
                plt.show()

        if self.figure:
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            pnl_res_dict["grouped_pnl_cumsum_df"].plot(
                ax=axes[0],
                title="Grouped PnL Cumsum",
                grid=True,
            )
            axes[0].axhline(0, color="black", linewidth=0.8)

            yearly_pnl = pnl_res_dict["grouped_pnl_df"].copy()
            yearly_pnl["Year"] = yearly_pnl.index.year
            yearly_mean = yearly_pnl.groupby("Year").mean()
            yearly_mean.plot(
                kind="bar",
                ax=axes[1],
                title="Yearly Grouped PnL Mean",
                grid=True,
            )

            pnl_res_dict["pnl_longshort_cumsum"].plot(
                ax=axes[2],
                title="Long-Short PnL Cumsum",
                grid=True,
            )
            axes[2].axhline(0, color="black", linewidth=0.8)

            plt.tight_layout()
            plt.show()

        # ========================================
        # 4. 风险因子相关性评估
        # ========================================
        risk_res_dict = self.risk_analyzer.analyze(
            factor_mtx=factor_mtx_1day,
            factor_pnl=pnl_res_dict["pnl_longshort"],
            return_figure=self.figure,
        )
        eval_metrics_corr = [
            "cs_corr_abs_mean", "cs_corr_abs_top1", "cs_corr_abs_mean_top5pct",
            "pnl_corr_abs_mean", "pnl_corr_abs_top1", "pnl_corr_abs_mean_top5pct",
        ]
        for metric in eval_metrics_corr:
            res_summary[metric] = risk_res_dict.get(metric)

        return res_summary

    def run_batch_evaluation(self, long_df: pd.DataFrame, factor_columns: list, output_csv: str = "all_factors_metrics.csv"):
        """
        遍历长表中的所有因子列，执行评估，并将最终结果保存为CSV。
        假设长表的索引为 [datetime, contract] 的 MultiIndex。
        """
        all_results = []

        print(f"开始批量回测，共计 {len(factor_columns)} 个因子...")
        count = 0
        for factor_name in tqdm(factor_columns):
            try:
                # 将长表当前因子列 unstack 转换为宽表矩阵（日期 x 标的）
                factor_mtx_1day = long_df[factor_name].unstack().dropna(axis=0, how="all")

                # 执行单因子流水线
                single_res = self.evaluate_single_factor(factor_name, factor_mtx_1day)
                all_results.append(single_res)
                count += 1
                # if count == 2:break
            except Exception as e:
                print(f"因子 {factor_name} 回测失败，错误信息：{str(e)}")
                # 记录失败因子，保证循环不中断
                all_results.append({"factor_name": factor_name, "error": str(e)})

        # 汇总结果并导出
        final_df = pd.DataFrame(all_results)
        # 将 factor_name 设置为索引，方便查看
        final_df.set_index("factor_name", inplace=True)
        if output_csv:
            final_df.to_csv(output_csv, encoding="utf-8-sig")
        print(f"批量回测完成！结果已保存至 {output_csv}")

        return final_df
