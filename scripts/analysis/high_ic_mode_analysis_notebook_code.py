from __future__ import annotations

# Notebook-style analysis trace for high-train-IC behavior mode exploration.
# This file is intentionally linear: copy cells/blocks into a notebook and edit paths.
# It does not train, validate, or save figures.

import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["axes.unicode_minus"] = False

RESULT_PATH = r"E:\实习\alpha_gen\artifacts\results\your_result_table.csv"
META_PATH = r"E:\实习\alpha_gen\data\metadata\real_behavior_metadata.json"


# ======================
# 1. Load Result Table
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
    (df["train_abs_rank_ic"] > 0.05)
    & (df["train_rank_ic_ir"] > 0.3)
].copy()

hi = hi.sort_values("train_abs_rank_ic", ascending=False).reset_index(drop=True)

print("total genes:", len(df))
print("high train IC genes:", len(hi))
print("ratio:", len(hi) / len(df) if len(df) else 0)

display_cols = [
    c
    for c in ["train_abs_rank_ic", "train_rank_ic_ir", "train_neutralized_abs_rank_ic"]
    if c in hi.columns
]
hi[display_cols].describe()


# ======================
# 2. Load Field Metadata
# ======================
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

field_rules = meta.get("field_rules", {})
behavior_rules = meta.get("behavior_field_rules", {})

field_semantic_rows = []
for field in sorted(set(field_rules) | set(behavior_rules)):
    fr = field_rules.get(field, {})
    br = behavior_rules.get(field, {})
    field_semantic_rows.append(
        {
            "field": field,
            "data_family": br.get("data_family", fr.get("family", "unknown")),
            "sub_family": br.get("sub_family", fr.get("add_group", fr.get("family", "unknown"))),
            "behavior_roles": ",".join(br.get("behavior_roles", [])),
            "direction": br.get("direction", fr.get("direction", None)),
        }
    )

field_meta = pd.DataFrame(field_semantic_rows)
field_meta.head()


# ======================
# 3. Parse Fields/Ops
# ======================
program_col = None
for c in ["program_json", "train_program_json"]:
    if c in hi.columns:
        program_col = c
        break

expr_col = None
for c in ["expression", "train_expression"]:
    if c in hi.columns:
        expr_col = c
        break

operator_vocab = {
    "neg",
    "abs",
    "sign",
    "slog",
    "sqrt_abs",
    "add",
    "sub",
    "mul",
    "qdiv",
    "cs_rank",
    "cs_zscore",
    "cs_demean",
    "cs_winsorize_5pct",
    "cs_resid",
    "delay",
    "ts_delta",
    "ts_return",
    "ts_mean",
    "ts_median",
    "ts_std",
    "ts_zscore",
    "ts_max_to_min",
    "ts_meanrank",
    "diff_sign",
    "ts_corr",
    "rolling_selmean_diff",
    "decay_linear",
    "mask_rank_high_50",
    "mask_rank_high_80",
    "mask_rank_low_20",
    "mask_sign_pos",
    "mask_sign_neg",
    "gate_nan",
    "gate_zero",
}

field_count = Counter()
field_gene_count = Counter()
op_count = Counter()

gene_field_sets = []
gene_op_sets = []

for _, row in hi.iterrows():
    fields_this_gene = []
    ops_this_gene = []

    if program_col is not None and pd.notna(row[program_col]):
        raw = json.loads(row[program_col])
        stack = [raw.get("root", raw)]

        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue

            node_type = node.get("node")

            if node_type == "field":
                field = str(node.get("field"))
                fields_this_gene.append(field)
                field_count[field] += 1

            elif node_type == "unary":
                op = str(node.get("op"))
                ops_this_gene.append(op)
                op_count[op] += 1
                stack.append(node.get("child"))

            elif node_type == "binary":
                op = str(node.get("op"))
                ops_this_gene.append(op)
                op_count[op] += 1
                stack.append(node.get("left"))
                stack.append(node.get("right"))

            elif node_type == "gate":
                op = str(node.get("op"))
                ops_this_gene.append(op)
                op_count[op] += 1
                stack.append(node.get("signal"))
                stack.append(node.get("mask"))

    elif expr_col is not None and pd.notna(row[expr_col]):
        expr = str(row[expr_col])

        for op in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr):
            if op in operator_vocab:
                ops_this_gene.append(op)
                op_count[op] += 1

        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            if token in operator_vocab:
                continue
            if token in {"nan", "inf", "None", "True", "False"}:
                continue
            if token in field_rules or token in behavior_rules:
                fields_this_gene.append(token)
                field_count[token] += 1

    for field in set(fields_this_gene):
        field_gene_count[field] += 1

    gene_field_sets.append(tuple(sorted(set(fields_this_gene))))
    gene_op_sets.append(tuple(sorted(set(ops_this_gene))))

hi["fields"] = gene_field_sets
hi["operators"] = gene_op_sets

hi[display_cols + ["fields", "operators"]].head()


# ======================
# 4. Field Frequency Plots
# ======================
field_freq = (
    pd.DataFrame(field_count.items(), columns=["field", "count"])
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

plt.figure(figsize=(11, 5))
sns.barplot(data=field_freq.head(20), x="field", y="count", color="#4C78A8")
plt.title("High Train IC Group - Field Occurrence Frequency Top20")
plt.xlabel("")
plt.ylabel("count")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.show()

field_gene_freq = (
    pd.DataFrame(field_gene_count.items(), columns=["field", "gene_count"])
    .sort_values("gene_count", ascending=False)
    .reset_index(drop=True)
)

plt.figure(figsize=(11, 5))
sns.barplot(data=field_gene_freq.head(20), x="field", y="gene_count", color="#72B7B2")
plt.title("High Train IC Group - Field Gene Coverage Top20")
plt.xlabel("")
plt.ylabel("number of genes using field")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.show()


# ======================
# 5. Operator Frequency
# ======================
op_freq = (
    pd.DataFrame(op_count.items(), columns=["operator", "count"])
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

plt.figure(figsize=(11, 5))
sns.barplot(data=op_freq, x="operator", y="count", color="#F58518")
plt.title("High Train IC Group - Operator Frequency")
plt.xlabel("")
plt.ylabel("count")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.show()

op_group_map = {}
for op in operator_vocab:
    if op.startswith("cs_"):
        op_group_map[op] = "cross_section"
    elif op.startswith("ts_") or op in {"delay", "decay_linear", "diff_sign", "rolling_selmean_diff"}:
        op_group_map[op] = "time_series"
    elif op.startswith("mask_") or op.startswith("gate_"):
        op_group_map[op] = "state_gate"
    elif op in {"add", "sub", "mul", "qdiv"}:
        op_group_map[op] = "binary_interaction"
    else:
        op_group_map[op] = "pointwise"

op_group_count = Counter()
for op, cnt in op_count.items():
    op_group_count[op_group_map.get(op, "unknown")] += cnt

op_group_freq = (
    pd.DataFrame(op_group_count.items(), columns=["operator_group", "count"])
    .sort_values("count", ascending=False)
)

plt.figure(figsize=(8, 4.5))
sns.barplot(data=op_group_freq, x="operator_group", y="count", color="#FF9DA6")
plt.title("High Train IC Group - Operator Group Frequency")
plt.xlabel("")
plt.ylabel("count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ======================
# 6. Semantic Categories
# ======================
field_to_family = dict(zip(field_meta["field"], field_meta["data_family"]))
field_to_sub_family = dict(zip(field_meta["field"], field_meta["sub_family"]))
field_to_roles = dict(zip(field_meta["field"], field_meta["behavior_roles"]))

gene_family_sets = []
gene_sub_family_sets = []
gene_role_sets = []

for fields in hi["fields"]:
    families = sorted({field_to_family.get(f, "unknown") for f in fields})
    sub_families = sorted({field_to_sub_family.get(f, "unknown") for f in fields})

    roles = set()
    for f in fields:
        role_text = field_to_roles.get(f, "")
        for role in str(role_text).split(","):
            role = role.strip()
            if role:
                roles.add(role)

    gene_family_sets.append(tuple(families))
    gene_sub_family_sets.append(tuple(sub_families))
    gene_role_sets.append(tuple(sorted(roles)))

hi["data_families"] = gene_family_sets
hi["sub_families"] = gene_sub_family_sets
hi["behavior_roles"] = gene_role_sets

hi[["fields", "data_families", "sub_families", "behavior_roles"]].head()


# Family coverage.
family_count = Counter()
for families in hi["data_families"]:
    for family in families:
        family_count[family] += 1

family_freq = (
    pd.DataFrame(family_count.items(), columns=["data_family", "gene_count"])
    .sort_values("gene_count", ascending=False)
)

plt.figure(figsize=(8, 4.5))
sns.barplot(data=family_freq, x="data_family", y="gene_count", color="#54A24B")
plt.title("High Train IC Group - Data Family Coverage")
plt.xlabel("")
plt.ylabel("number of genes")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# Sub-family coverage.
sub_family_count = Counter()
for sub_families in hi["sub_families"]:
    for sub_family in sub_families:
        sub_family_count[sub_family] += 1

sub_family_freq = (
    pd.DataFrame(sub_family_count.items(), columns=["sub_family", "gene_count"])
    .sort_values("gene_count", ascending=False)
)

plt.figure(figsize=(12, 5))
sns.barplot(data=sub_family_freq, x="sub_family", y="gene_count", color="#B279A2")
plt.title("High Train IC Group - Sub Family Coverage")
plt.xlabel("")
plt.ylabel("number of genes")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.show()


# Sub-family co-occurrence heatmap.
all_sub_families = sorted(sub_family_count.keys())
co_matrix = pd.DataFrame(0, index=all_sub_families, columns=all_sub_families, dtype=int)

for sub_families in hi["sub_families"]:
    items = list(sub_families)
    for a in items:
        for b in items:
            co_matrix.loc[a, b] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(co_matrix, cmap="Blues", linewidths=0.3)
plt.title("High Train IC Group - Sub Family Co-occurrence")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()


# Gene x sub-family matrix, sorted by train_abs_rank_ic.
top_sub_families = sub_family_freq["sub_family"].tolist()
gene_category_matrix = pd.DataFrame(0, index=hi.index, columns=top_sub_families)

for idx, sub_families in hi["sub_families"].items():
    for sf in sub_families:
        if sf in gene_category_matrix.columns:
            gene_category_matrix.loc[idx, sf] = 1

gene_category_matrix.index = [
    f"{i}: IC={hi.loc[i, 'train_abs_rank_ic']:.3f}" for i in hi.index
]

plt.figure(figsize=(12, max(5, len(hi) * 0.22)))
sns.heatmap(gene_category_matrix, cmap="Greens", cbar=False, linewidths=0.2)
plt.title("High Train IC Group - Gene x Sub Family Matrix")
plt.xlabel("")
plt.ylabel("genes sorted by train_abs_rank_ic")
plt.tight_layout()
plt.show()


# ======================
# 7. Combo Statistics
# ======================
combo_rows = []

for combo, part in hi.groupby(hi["sub_families"].map(lambda x: " + ".join(x))):
    row = {
        "combo": combo,
        "n_genes": len(part),
        "mean_train_abs_rank_ic": part["train_abs_rank_ic"].mean(),
        "median_train_abs_rank_ic": part["train_abs_rank_ic"].median(),
        "mean_train_rank_ic_ir": part["train_rank_ic_ir"].mean(),
    }
    if "train_neutralized_abs_rank_ic" in part.columns:
        row["mean_train_neutralized_abs_rank_ic"] = part["train_neutralized_abs_rank_ic"].mean()
    combo_rows.append(row)

combo_stats = (
    pd.DataFrame(combo_rows)
    .sort_values(["n_genes", "mean_train_abs_rank_ic"], ascending=[False, False])
    .reset_index(drop=True)
)

combo_stats.head(20)

if "mean_train_neutralized_abs_rank_ic" in combo_stats.columns:
    plot_combo = combo_stats.head(30).copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_combo,
        x="mean_train_abs_rank_ic",
        y="mean_train_neutralized_abs_rank_ic",
        size="n_genes",
        sizes=(80, 800),
        color="#E45756",
        alpha=0.75,
    )

    for _, r in plot_combo.iterrows():
        plt.text(
            r["mean_train_abs_rank_ic"],
            r["mean_train_neutralized_abs_rank_ic"],
            str(r["n_genes"]),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    plt.title("Sub Family Combo: Raw Train IC vs Neutralized Train IC")
    plt.xlabel("mean train_abs_rank_ic")
    plt.ylabel("mean train_neutralized_abs_rank_ic")
    plt.tight_layout()
    plt.show()

    plot_combo[
        [
            "combo",
            "n_genes",
            "mean_train_abs_rank_ic",
            "mean_train_neutralized_abs_rank_ic",
            "mean_train_rank_ic_ir",
        ]
    ].head(20)


# ======================
# 8. Mode Candidate Table
# ======================
mode_candidate_cols = [
    "train_abs_rank_ic",
    "train_rank_ic_ir",
    "train_neutralized_abs_rank_ic",
    "fields",
    "data_families",
    "sub_families",
    "behavior_roles",
    "operators",
]
mode_candidate_cols = [c for c in mode_candidate_cols if c in hi.columns]

mode_candidate_table = hi[mode_candidate_cols].copy()

if "expression" in hi.columns:
    mode_candidate_table["expression"] = hi["expression"]

mode_candidate_table.head(30)
