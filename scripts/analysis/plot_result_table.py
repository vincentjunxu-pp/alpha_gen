from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLS = (
    "train_abs_rank_ic",
    "valid_abs_rank_ic",
    "train_neutralized_abs_rank_ic",
    "train_size",
    "train_depth",
)
OPERATORS = {
    "neg", "abs", "sign", "slog", "sqrt_abs", "add", "sub", "mul", "qdiv",
    "cs_rank", "cs_zscore", "cs_demean", "cs_winsorize_5pct", "cs_resid",
    "delay", "ts_delta", "ts_return", "ts_mean", "ts_median", "ts_std",
    "ts_zscore", "ts_max_to_min", "ts_meanrank", "diff_sign", "ts_corr",
    "rolling_selmean_diff", "decay_linear", "mask_rank_high_50",
    "mask_rank_high_80", "mask_rank_low_20", "mask_sign_pos", "mask_sign_neg",
    "gate_nan", "gate_zero",
}
SKIP_TOKENS = OPERATORS | {
    "nan", "inf", "true", "false", "none", "train", "valid", "rank", "rank_ic",
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def node_counts(node: dict, fields: Counter[str], ops: Counter[str]) -> None:
    kind = node.get("node")
    if kind == "field":
        fields[str(node.get("field", ""))] += 1
    elif kind == "unary":
        ops[str(node.get("op", ""))] += 1
        node_counts(node["child"], fields, ops)
    elif kind == "binary":
        ops[str(node.get("op", ""))] += 1
        node_counts(node["left"], fields, ops)
        node_counts(node["right"], fields, ops)
    elif kind == "gate":
        ops[str(node.get("op", ""))] += 1
        node_counts(node["signal"], fields, ops)
        node_counts(node["mask"], fields, ops)


def count_from_program_json(values: pd.Series) -> tuple[Counter[str], Counter[str]]:
    fields: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    for text in values.dropna().astype(str):
        raw = json.loads(text)
        node_counts(raw["root"], fields, ops)
    return fields, ops


def count_from_expression(values: pd.Series) -> tuple[Counter[str], Counter[str]]:
    fields: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for expr in values.dropna().astype(str):
        for op in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr):
            if op in OPERATORS:
                ops[op] += 1
        for token in ident_re.findall(expr):
            if token in SKIP_TOKENS or token.startswith("barra_") or token.startswith("label_"):
                continue
            if token in OPERATORS:
                continue
            fields[token] += 1
    return fields, ops


def count_fields_ops(df: pd.DataFrame) -> tuple[Counter[str], Counter[str]]:
    if "program_json" in df.columns:
        return count_from_program_json(df["program_json"])
    for col in ("train_program_json", "expression", "train_expression"):
        if col in df.columns:
            return count_from_expression(df[col])
    return Counter(), Counter()


def count_barra(df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    cols = [c for c in df.columns if c.endswith("style_selected_fields") or c.endswith("barra_selected_styles")]
    if not cols:
        cols = [c for c in df.columns if "barra" in c.lower() and "style" in c.lower()]
    for col in cols:
        for text in df[col].dropna().astype(str):
            for item in re.split(r"[,;|]", text):
                item = item.strip()
                if item.startswith("barra_"):
                    counts[item] += 1
    return counts


def style_ax(ax) -> None:
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def save_bar(counter: Counter[str], path: Path, title: str, top: int | None = None) -> None:
    items = counter.most_common(top)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(range(len(labels)), values, color="#4C78A8", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = {"train_abs_rank_ic", "valid_abs_rank_ic"} - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    ordered = df.sort_values("train_abs_rank_ic", ascending=False).reset_index(drop=True)
    paths = [
        out_dir / "01_sorted_train_valid_abs_rank_ic.png",
        out_dir / "02_box_train_valid_abs_rank_ic.png",
        out_dir / "03_box_train_raw_vs_neutralized_abs_rank_ic.png",
        out_dir / "04_field_usage_top20.png",
        out_dir / "05_operator_usage.png",
        out_dir / "06_barra_style_usage.png",
        out_dir / "07_train_size_depth_hist.png",
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ordered.index, ordered["train_abs_rank_ic"], label="train_abs_rank_ic", color="#1F77B4", alpha=0.78)
    ax.plot(ordered.index, ordered["valid_abs_rank_ic"], label="valid_abs_rank_ic", color="#FF7F0E", alpha=0.62)
    ax.set_title("Genes Sorted by Train Abs RankIC")
    ax.set_xlabel("gene position")
    ax.set_ylabel("abs_rank_ic")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([df["train_abs_rank_ic"].dropna(), df["valid_abs_rank_ic"].dropna()], tick_labels=["train", "valid"], patch_artist=True)
    ax.set_title("Train vs Valid Abs RankIC")
    ax.set_ylabel("abs_rank_ic")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=180)
    plt.close(fig)

    if "train_neutralized_abs_rank_ic" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(
            [df["train_abs_rank_ic"].dropna(), df["train_neutralized_abs_rank_ic"].dropna()],
            tick_labels=["raw train", "neutralized train"],
            patch_artist=True,
        )
        ax.set_title("Train Abs RankIC Before vs After Neutralization")
        ax.set_ylabel("abs_rank_ic")
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(paths[2], dpi=180)
        plt.close(fig)

    fields, ops = count_fields_ops(df)
    save_bar(fields, paths[3], "Field Usage Top 20", top=20)
    save_bar(ops, paths[4], "Operator Usage")
    save_bar(count_barra(df), paths[5], "Barra Style Usage")

    if {"train_size", "train_depth"} <= set(df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].hist(df["train_size"].dropna(), bins="auto", color="#54A24B", alpha=0.78)
        axes[0].set_title("Train Size")
        axes[0].set_xlabel("size")
        axes[0].set_ylabel("count")
        style_ax(axes[0])
        axes[1].hist(df["train_depth"].dropna(), bins="auto", color="#E45756", alpha=0.78)
        axes[1].set_title("Train Depth")
        axes[1].set_xlabel("depth")
        axes[1].set_ylabel("count")
        style_ax(axes[1])
        fig.tight_layout()
        fig.savefig(paths[6], dpi=180)
        plt.close(fig)
    return [p for p in paths if p.exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_table", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.result_table.with_suffix("").parent / "plots_from_result_table"
    paths = make_plots(read_table(args.result_table), out_dir)
    print("output_dir", out_dir.resolve())
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
