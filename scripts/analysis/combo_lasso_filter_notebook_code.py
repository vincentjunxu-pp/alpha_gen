from __future__ import annotations

# Notebook-style code for filtering highly correlated genes within each combo.
# Fill the factor generation block with your existing single-factor code.

import ast
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from alpha_gen.free_gp_cuda import (
    CudaFactorContext,
    FreeGPSearchConfig,
    Program,
    ProgramGeneratorConfig,
    ProgramScorer,
    ScorerConfig,
)


sns.set_theme(style="whitegrid")
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.max_colwidth", 240)


# ======================
# 1. Config
# ======================
RESULT_PATH = r"E:\实习\alpha_gen\artifacts\results\your_result_table.csv"
PANEL_PATH = r"E:\实习\alpha_gen\data\panels\mock_behavior_daily.parquet"
LABEL_COL = "label_20d"
TRADEABLE_COL = "is_tradeable"
INDUSTRY_COL = "industry_code"
DEVICE = "cuda"
FACTOR_VIEW = "neutralized"
LASSO_START_DATE = None
LASSO_END_DATE = None

MIN_TRAIN_ABS_RANK_IC = 0.05
MIN_TRAIN_RANK_IC_IR = 0.30
MIN_COMBO_GENES = 3
MAX_GENES_PER_COMBO = None
COEF_EPS = 1e-8
BARRA_STYLE_FIELDS = [f"barra_{name}" for name in [
    "size", "beta", "momentum", "liquidity", "non_linear_size",
    "residual_volatility", "leverage", "book_to_price", "earnings_yield", "growth",
]]


# ======================
# 2. Load high-quality genes
# ======================
if RESULT_PATH.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(RESULT_PATH)
else:
    df = pd.read_csv(RESULT_PATH)

for col in [
    "train_abs_rank_ic",
    "train_rank_ic_ir",
    "train_neutralized_abs_rank_ic",
    "train_size",
    "train_depth",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

hi = df[
    (df["train_abs_rank_ic"] > MIN_TRAIN_ABS_RANK_IC)
    & (df["train_rank_ic_ir"] > MIN_TRAIN_RANK_IC_IR)
].copy()

if "train_expression" not in hi.columns and "expression" in hi.columns:
    hi["train_expression"] = hi["expression"]


def parse_list_cell(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
                return list(parsed) if isinstance(parsed, (list, tuple, set)) else [str(parsed)]
            except Exception:
                return [text]
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text] if text else []
    return [str(value)]


for list_col in ["sub_families", "fields", "operators"]:
    if list_col in hi.columns:
        hi[list_col] = hi[list_col].map(parse_list_cell)

hi["gene_id"] = ["g%05d" % i for i in hi.index]
hi["combo_key"] = hi["sub_families"].map(lambda x: " + ".join(x)) if "sub_families" in hi.columns else "unknown"

combo_stats = (
    hi.groupby("combo_key")
    .agg(
        n_genes=("gene_id", "size"),
        mean_train_abs_rank_ic=("train_abs_rank_ic", "mean"),
        median_train_abs_rank_ic=("train_abs_rank_ic", "median"),
        mean_train_rank_ic_ir=("train_rank_ic_ir", "mean"),
    )
    .sort_values(["n_genes", "mean_train_abs_rank_ic"], ascending=[False, False])
    .reset_index()
)

combo_stats.head(30)


# ======================
# 3. Load label panel
# ======================
panel = pd.read_parquet(PANEL_PATH)

if isinstance(panel.index, pd.MultiIndex):
    label_panel = panel[LABEL_COL].unstack("Contract")
else:
    label_panel = panel.pivot(index="Datetime", columns="Contract", values=LABEL_COL)

label_panel = label_panel.sort_index()

if LASSO_START_DATE is not None:
    label_panel = label_panel.loc[label_panel.index >= pd.Timestamp(LASSO_START_DATE)]
if LASSO_END_DATE is not None:
    label_panel = label_panel.loc[label_panel.index <= pd.Timestamp(LASSO_END_DATE)]

label_target = label_panel.rank(axis=1, pct=True)
label_target = label_target.sub(label_target.mean(axis=1), axis=0).div(label_target.std(axis=1).replace(0.0, np.nan), axis=0)


# ======================
# 4. Free GP scorer
# ======================
missing_styles = [field for field in BARRA_STYLE_FIELDS if field not in panel.columns]
if missing_styles:
    print("missing barra style fields:", missing_styles)

ctx = CudaFactorContext(
    panel,
    label_col=LABEL_COL,
    tradeable_col=TRADEABLE_COL,
    industry_col=INDUSTRY_COL,
    device=DEVICE,
)

config = FreeGPSearchConfig(
    population_size=2000,
    generations=10,
    random_seed=42,
    min_coverage=0.30,
    show_progress=True,
    generator_config=ProgramGeneratorConfig(
        max_depth=6,
        max_size=64,
    ),
    scorer_config=ScorerConfig(
        style_fields=[field for field in BARRA_STYLE_FIELDS if field in panel.columns],
        mask_inputs_by_tradeable=True,
        mask_factor_by_tradeable=True,
        min_cross_section_size=2,
        neutralize_industry=True,
        neutralize_styles=True,
        standardize_styles=False,
        ndcg_top_fraction=0.20,
    ),
)

scorer = ProgramScorer(ctx, config.scorer_config)
factor_cache = {}
factor_errors = []


# ======================
# 5. Lasso within each combo
# ======================
all_coef_rows = []
selected_rows = []

combo_order = combo_stats.loc[combo_stats["n_genes"] >= MIN_COMBO_GENES, "combo_key"].tolist()

for combo_key in combo_order:
    part = hi.loc[hi["combo_key"] == combo_key].copy()
    if "program_json" not in part.columns:
        raise KeyError("result table must contain program_json to rebuild factor values")

    part = part[part["program_json"].notna()]
    part = part.sort_values("train_abs_rank_ic", ascending=False)

    if MAX_GENES_PER_COMBO is not None:
        part = part.head(MAX_GENES_PER_COMBO)

    if len(part) < MIN_COMBO_GENES:
        continue

    gene_ids = part["gene_id"].tolist()
    matrices = []
    built_gene_ids = []

    for _, row in part.iterrows():
        gene_id = row["gene_id"]
        if gene_id not in factor_cache:
            try:
                raw_program = row["program_json"]
                if isinstance(raw_program, str):
                    program = Program.from_json(raw_program)
                else:
                    program = Program.from_dict(raw_program)
                values = scorer.factor_values(program, view=FACTOR_VIEW)
                factor_cache[gene_id] = ctx.tensor_to_frame(values)
            except Exception as exc:
                factor_errors.append(
                    {
                        "combo_key": combo_key,
                        "gene_id": gene_id,
                        "train_expression": row.get("train_expression", ""),
                        "error": repr(exc),
                    }
                )
                continue

        factor = factor_cache[gene_id].reindex(index=label_panel.index, columns=label_panel.columns)
        factor = factor.rank(axis=1, pct=True)
        factor = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1).replace(0.0, np.nan), axis=0)
        matrices.append(factor.stack(dropna=False).rename(gene_id))
        built_gene_ids.append(gene_id)

    if len(built_gene_ids) < MIN_COMBO_GENES:
        continue

    x_df = pd.concat(matrices, axis=1)
    y = label_target.stack(dropna=False).rename("label")

    sample = x_df.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    if sample.empty or sample.shape[0] < 200:
        continue

    x = sample[built_gene_ids]
    y = sample["label"]

    valid_std = x.std(axis=0)
    kept_gene_ids = valid_std[valid_std > 1e-10].index.tolist()
    if len(kept_gene_ids) < MIN_COMBO_GENES:
        continue

    x = x[kept_gene_ids]
    sample_dates = pd.Index(sample.index.get_level_values(0))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y_centered = y.to_numpy(dtype=float) - float(y.mean())

    unique_dates = pd.Index(pd.unique(sample_dates)).sort_values()
    if len(unique_dates) < 3:
        continue
    n_splits = min(5, max(2, len(unique_dates) // 40), len(unique_dates) - 1)
    date_tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = []
    for train_date_idx, test_date_idx in date_tscv.split(unique_dates):
        train_dates = set(unique_dates[train_date_idx])
        test_dates = set(unique_dates[test_date_idx])
        train_idx = np.flatnonzero(sample_dates.isin(train_dates))
        test_idx = np.flatnonzero(sample_dates.isin(test_dates))
        if len(train_idx) and len(test_idx):
            cv_splits.append((train_idx, test_idx))

    if not cv_splits:
        continue

    model = LassoCV(
        alphas=np.logspace(-5, -1, 50),
        cv=cv_splits,
        fit_intercept=True,
        max_iter=20000,
        n_jobs=-1,
        random_state=20260604,
    )
    model.fit(x_scaled, y_centered)

    coef = pd.DataFrame(
        {
            "combo_key": combo_key,
            "gene_id": kept_gene_ids,
            "coef": model.coef_,
        }
    )
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.merge(
        part[
            [
                "gene_id",
                "train_expression",
                "train_abs_rank_ic",
                "train_rank_ic_ir",
                *([c for c in ["train_neutralized_abs_rank_ic"] if c in part.columns]),
            ]
        ],
        on="gene_id",
        how="left",
    )
    coef["alpha"] = model.alpha_
    coef["n_samples"] = len(sample)
    coef["n_genes_in_lasso"] = len(kept_gene_ids)
    coef = coef.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    all_coef_rows.append(coef)

    selected = coef.loc[coef["abs_coef"] > COEF_EPS].copy()
    selected_rows.append(selected)

    top_plot = coef.head(40).copy()
    if not top_plot.empty:
        top_plot["label"] = top_plot["gene_id"] + " | IC=" + top_plot["train_abs_rank_ic"].round(4).astype(str)
        plt.figure(figsize=(11, max(4, 0.28 * len(top_plot))))
        sns.barplot(data=top_plot, x="coef", y="label", color="#4C78A8")
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.title(f"Lasso Coefficients: {combo_key} | selected={len(selected)}/{len(coef)}")
        plt.xlabel("Lasso coefficient")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()


lasso_coef = pd.concat(all_coef_rows, ignore_index=True) if all_coef_rows else pd.DataFrame()
lasso_selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
factor_errors_df = pd.DataFrame(factor_errors)

if not factor_errors_df.empty:
    display(factor_errors_df.head(30))

if not lasso_selected.empty:
    lasso_selected.sort_values(["combo_key", "abs_coef"], ascending=[True, False]).head(100)


# ======================
# 6. Diagnostics
# ======================
selection_summary = (
    lasso_coef.assign(selected=lasso_coef["abs_coef"] > COEF_EPS)
    .groupby("combo_key")
    .agg(
        n_input=("gene_id", "size"),
        n_selected=("selected", "sum"),
        alpha=("alpha", "first"),
        n_samples=("n_samples", "first"),
    )
    .reset_index()
    if not lasso_coef.empty
    else pd.DataFrame()
)

if not selection_summary.empty:
    selection_summary["selected_ratio"] = selection_summary["n_selected"] / selection_summary["n_input"]
    selection_summary.sort_values(["n_selected", "n_input"], ascending=[False, False]).head(50)


if not lasso_selected.empty and "fields" in hi.columns:
    selected_gene_ids = set(lasso_selected["gene_id"])
    selected_hi = hi.loc[hi["gene_id"].isin(selected_gene_ids)].copy()
    field_coverage = Counter()
    operator_coverage = Counter()

    for fields in selected_hi["fields"]:
        field_coverage.update(set(fields))
    for operators in selected_hi["operators"]:
        operator_coverage.update(set(operators))

    selected_field_coverage = (
        pd.DataFrame(field_coverage.items(), columns=["field", "gene_coverage"])
        .sort_values("gene_coverage", ascending=False)
        .reset_index(drop=True)
    )
    selected_operator_coverage = (
        pd.DataFrame(operator_coverage.items(), columns=["operator", "gene_coverage"])
        .sort_values("gene_coverage", ascending=False)
        .reset_index(drop=True)
    )

    display(selected_field_coverage.head(30))
    display(selected_operator_coverage.head(30))
