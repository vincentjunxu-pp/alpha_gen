from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

DEFAULT_META_PATH = (
    REPO_ROOT / "data" / "metadata" / "production" / "real_behavior_metadata.json"
)
DEFAULT_OUTPUT_SUBDIR = "behavior_frequency_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze behavior_gen result tables and export base-field/operator "
            "frequency tables plus summary plots."
        )
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="Behavior GA result table. If omitted, auto-detect the latest CSV with gene_slots.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Behavior metadata JSON used to annotate fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <result-dir>/{DEFAULT_OUTPUT_SUBDIR}.",
    )
    parser.add_argument(
        "--min-train-abs-rank-ic",
        type=float,
        default=0.05,
        help="Keep rows with train_abs_rank_ic above this threshold when the column exists.",
    )
    parser.add_argument(
        "--min-train-rank-ic-ir",
        type=float,
        default=0.30,
        help="Keep rows with train_rank_ic_ir above this threshold when the column exists.",
    )
    parser.add_argument(
        "--passed-validation-only",
        action="store_true",
        help="Keep only rows where passed_validation is true when the column exists.",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable train IC / ICIR threshold filtering.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate genes before analysis.",
    )
    return parser.parse_args()


def read_result_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def detect_latest_behavior_result() -> Path:
    candidates = sorted(
        (REPO_ROOT / "artifacts" / "results").glob("**/*final_population*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        except Exception:
            continue
        if "gene_slots" in columns:
            return path
    raise FileNotFoundError(
        "No behavior_gen final_population CSV with a gene_slots column was found. "
        "Pass --result-path explicitly."
    )


def parse_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def first_existing(row: pd.Series, candidates: tuple[str, ...], default: Any = None) -> Any:
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return row[col]
    return default


def load_field_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["field", "data_family", "sub_family", "sub_type", "behavior_roles"])
    metadata = json.loads(path.read_text(encoding="utf-8"))
    rules = metadata.get("behavior_field_rules") or metadata.get("field_rules") or {}
    rows: list[dict[str, Any]] = []
    for field, rule in rules.items():
        roles = rule.get("behavior_roles", rule.get("behavior_role", ()))
        if isinstance(roles, str):
            role_text = roles
        else:
            role_text = ",".join(str(role) for role in roles)
        rows.append(
            {
                "field": field,
                "data_family": rule.get("data_family", rule.get("family", "unknown")),
                "sub_family": rule.get("sub_family", rule.get("add_group", rule.get("family", "unknown"))),
                "sub_type": rule.get("sub_type", rule.get("sub_family", "unknown")),
                "behavior_roles": role_text,
                "direction": rule.get("direction"),
                "unit_type": rule.get("unit_type", "unknown"),
                "window": rule.get("window", rule.get("period_type", "unknown")),
            }
        )
    return pd.DataFrame(rows)


def field_meta_maps(field_meta: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if field_meta.empty:
        return {}
    return field_meta.set_index("field").to_dict(orient="index")


def metric_filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    numeric_candidates = (
        "train_abs_rank_ic",
        "train_rank_ic_ir",
        "train_neutralized_abs_rank_ic",
        "valid_abs_rank_ic",
        "valid_rank_ic_ir",
    )
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if args.dedupe:
        key_cols = [col for col in ("gene_slots", "gene_conditions", "gene_mode", "gene_combiner") if col in out.columns]
        if key_cols:
            out = out.drop_duplicates(key_cols).copy()
        elif "expression" in out.columns:
            out = out.drop_duplicates("expression").copy()

    if args.passed_validation_only and "passed_validation" in out.columns:
        out = out[out["passed_validation"].astype(str).str.lower().isin({"true", "1"})].copy()

    if args.no_quality_filter:
        return out.reset_index(drop=True)

    masks = []
    if "train_abs_rank_ic" in out.columns and args.min_train_abs_rank_ic is not None:
        masks.append(out["train_abs_rank_ic"] > args.min_train_abs_rank_ic)
    if "train_rank_ic_ir" in out.columns and args.min_train_rank_ic_ir is not None:
        masks.append(out["train_rank_ic_ir"] > args.min_train_rank_ic_ir)
    if masks:
        mask = masks[0]
        for item in masks[1:]:
            mask &= item
        out = out[mask].copy()
    return out.reset_index(drop=True)


def normalize_slot_items(slots: Any) -> list[tuple[str, dict[str, Any]]]:
    parsed = parse_cell(slots)
    if isinstance(parsed, dict):
        output = []
        for name, raw in parsed.items():
            if isinstance(raw, dict):
                output.append((str(name), raw))
            elif isinstance(raw, str):
                output.append((str(name), {"field": raw, "unary_op": "current"}))
        return output
    if isinstance(parsed, list):
        output = []
        for idx, raw in enumerate(parsed):
            if isinstance(raw, dict):
                output.append((str(raw.get("slot", idx)), raw))
        return output
    return []


def normalize_condition_items(conditions: Any) -> list[tuple[str, dict[str, Any]]]:
    parsed = parse_cell(conditions)
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    if not isinstance(parsed, list):
        return []
    output = []
    for idx, raw in enumerate(parsed):
        if isinstance(raw, dict):
            output.append((f"condition_{idx}", raw))
    return output


def extract_from_behavior_schema(
    row: pd.Series,
    factor_id: int,
    meta_by_field: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    field_rows: list[dict[str, Any]] = []
    op_rows: list[dict[str, Any]] = []
    gene_fields: set[str] = set()
    gene_ops: set[str] = set()

    mode = first_existing(row, ("gene_mode", "tree_mode", "mode"), "")
    combiner = first_existing(row, ("gene_combiner", "tree_combiner", "combiner"), "")
    direction_policy = first_existing(row, ("gene_direction_policy", "tree_direction_policy", "direction_policy"), "")
    expression = first_existing(row, ("expression", "train_expression", "valid_expression"), "")

    if mode:
        op_rows.append(
            {
                "factor_id": factor_id,
                "operator": str(mode),
                "operator_type": "mode",
                "source": "gene_mode",
            }
        )
        gene_ops.add(str(mode))
    if combiner:
        op_rows.append(
            {
                "factor_id": factor_id,
                "operator": str(combiner),
                "operator_type": "combiner",
                "source": "gene_combiner",
            }
        )
        gene_ops.add(str(combiner))
    if direction_policy:
        op_rows.append(
            {
                "factor_id": factor_id,
                "operator": str(direction_policy),
                "operator_type": "direction_policy",
                "source": "gene_direction_policy",
            }
        )
        gene_ops.add(str(direction_policy))

    for slot_name, slot in normalize_slot_items(row.get("gene_slots")):
        field = slot.get("field")
        if not field:
            continue
        unary_op = str(slot.get("unary_op", "current"))
        meta = meta_by_field.get(str(field), {})
        field_rows.append(
            {
                "factor_id": factor_id,
                "source": "slot",
                "slot_or_condition": slot_name,
                "field": str(field),
                "unary_op": unary_op,
                "mode": str(mode),
                "combiner": str(combiner),
                "expression": expression,
                **meta,
            }
        )
        op_rows.append(
            {
                "factor_id": factor_id,
                "operator": unary_op,
                "operator_type": "unary",
                "source": f"slot:{slot_name}",
            }
        )
        gene_fields.add(str(field))
        gene_ops.add(unary_op)

    for cond_name, cond in normalize_condition_items(row.get("gene_conditions")):
        field = cond.get("field")
        if not field:
            continue
        unary_op = str(cond.get("unary_op", "current"))
        condition_op = str(cond.get("condition_op", "unknown"))
        meta = meta_by_field.get(str(field), {})
        field_rows.append(
            {
                "factor_id": factor_id,
                "source": "condition",
                "slot_or_condition": cond_name,
                "field": str(field),
                "unary_op": unary_op,
                "condition_op": condition_op,
                "threshold": cond.get("threshold"),
                "mode": str(mode),
                "combiner": str(combiner),
                "expression": expression,
                **meta,
            }
        )
        for op, op_type in ((unary_op, "unary"), (condition_op, "condition")):
            op_rows.append(
                {
                    "factor_id": factor_id,
                    "operator": op,
                    "operator_type": op_type,
                    "source": f"condition:{cond_name}",
                }
            )
            gene_ops.add(op)
        gene_fields.add(str(field))

    parsed = {
        "factor_id": factor_id,
        "mode": str(mode),
        "combiner": str(combiner),
        "direction_policy": str(direction_policy),
        "fields": json.dumps(sorted(gene_fields), ensure_ascii=False),
        "operators": json.dumps(sorted(gene_ops), ensure_ascii=False),
        "expression": expression,
    }
    return field_rows, op_rows, parsed


def parse_behavior_results(
    df: pd.DataFrame,
    field_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_by_field = field_meta_maps(field_meta)
    field_rows: list[dict[str, Any]] = []
    op_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []

    for factor_id, (_, row) in enumerate(df.iterrows()):
        rows, ops, parsed = extract_from_behavior_schema(row, factor_id, meta_by_field)
        field_rows.extend(rows)
        op_rows.extend(ops)
        parsed_rows.append(parsed)

    return pd.DataFrame(field_rows), pd.DataFrame(op_rows), pd.DataFrame(parsed_rows)


def add_frequency_shares(freq: pd.DataFrame, count_col: str, total: int, n_factors: int) -> pd.DataFrame:
    if freq.empty:
        return freq
    out = freq.copy()
    out["count_share"] = out[count_col] / total if total else 0.0
    if "gene_count" in out.columns:
        out["gene_coverage"] = out["gene_count"] / n_factors if n_factors else 0.0
    return out


def build_frequency_tables(
    field_long: pd.DataFrame,
    operator_long: pd.DataFrame,
    parsed_factors: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    n_factors = len(parsed_factors)
    tables: dict[str, pd.DataFrame] = {}

    if field_long.empty:
        tables["field_frequency"] = pd.DataFrame()
        tables["field_by_slot"] = pd.DataFrame()
        tables["semantic_frequency"] = pd.DataFrame()
    else:
        field_freq = (
            field_long.groupby("field", dropna=False)
            .agg(
                count=("field", "size"),
                gene_count=("factor_id", "nunique"),
                slot_count=("source", lambda s: int((s == "slot").sum())),
                condition_count=("source", lambda s: int((s == "condition").sum())),
                data_family=("data_family", "first"),
                sub_family=("sub_family", "first"),
                sub_type=("sub_type", "first"),
                behavior_roles=("behavior_roles", "first"),
            )
            .reset_index()
            .sort_values(["gene_count", "count"], ascending=False)
        )
        tables["field_frequency"] = add_frequency_shares(
            field_freq,
            "count",
            int(field_long.shape[0]),
            n_factors,
        )
        tables["field_by_slot"] = (
            field_long.groupby(["source", "slot_or_condition", "field"], dropna=False)
            .agg(count=("field", "size"), gene_count=("factor_id", "nunique"))
            .reset_index()
            .sort_values(["source", "slot_or_condition", "count"], ascending=[True, True, False])
        )

        semantic_rows = []
        for col in ("data_family", "sub_family", "sub_type"):
            if col in field_long.columns:
                tmp = (
                    field_long.groupby(col, dropna=False)
                    .agg(count=("field", "size"), gene_count=("factor_id", "nunique"))
                    .reset_index()
                    .rename(columns={col: "category"})
                )
                tmp["category_type"] = col
                semantic_rows.append(tmp)
        if "behavior_roles" in field_long.columns:
            role_counts: Counter[str] = Counter()
            role_gene_sets: defaultdict[str, set[int]] = defaultdict(set)
            for _, row in field_long.iterrows():
                for role in str(row.get("behavior_roles", "")).split(","):
                    role = role.strip()
                    if not role:
                        continue
                    role_counts[role] += 1
                    role_gene_sets[role].add(int(row["factor_id"]))
            role_df = pd.DataFrame(
                {
                    "category": role,
                    "count": count,
                    "gene_count": len(role_gene_sets[role]),
                    "category_type": "behavior_role",
                }
                for role, count in role_counts.items()
            )
            semantic_rows.append(role_df)
        if semantic_rows:
            tables["semantic_frequency"] = (
                pd.concat(semantic_rows, ignore_index=True)
                .sort_values(["category_type", "gene_count", "count"], ascending=[True, False, False])
                .reset_index(drop=True)
            )
        else:
            tables["semantic_frequency"] = pd.DataFrame()

    if operator_long.empty:
        tables["operator_frequency"] = pd.DataFrame()
    else:
        op_freq = (
            operator_long.groupby(["operator_type", "operator"], dropna=False)
            .agg(count=("operator", "size"), gene_count=("factor_id", "nunique"))
            .reset_index()
            .sort_values(["count", "gene_count"], ascending=False)
        )
        tables["operator_frequency"] = add_frequency_shares(
            op_freq,
            "count",
            int(operator_long.shape[0]),
            n_factors,
        )

    if parsed_factors.empty:
        tables["mode_combiner_frequency"] = pd.DataFrame()
    else:
        tables["mode_combiner_frequency"] = (
            parsed_factors.groupby(["mode", "combiner"], dropna=False)
            .size()
            .reset_index(name="gene_count")
            .sort_values("gene_count", ascending=False)
        )

    return tables


def save_bar_plot(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str, top: int = 30) -> None:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_df = df.head(top).copy()
    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df = plot_df.iloc[::-1]
    height = max(4.0, 0.32 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(plot_df[label_col], plot_df[value_col], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    source_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    field_long: pd.DataFrame,
    operator_long: pd.DataFrame,
    parsed_factors: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_df.to_csv(output_dir / "filtered_result_rows.csv", index=False, encoding="utf-8-sig")
    field_long.to_csv(output_dir / "field_occurrence_long.csv", index=False, encoding="utf-8-sig")
    operator_long.to_csv(output_dir / "operator_occurrence_long.csv", index=False, encoding="utf-8-sig")
    parsed_factors.to_csv(output_dir / "parsed_factor_fields_operators.csv", index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    field_freq = tables.get("field_frequency", pd.DataFrame())
    op_freq = tables.get("operator_frequency", pd.DataFrame())
    semantic = tables.get("semantic_frequency", pd.DataFrame())
    mode_combiner = tables.get("mode_combiner_frequency", pd.DataFrame())

    save_bar_plot(field_freq, "field", "count", output_dir / "top_fields_by_occurrence.png", "Base Field Occurrence Top30")
    save_bar_plot(field_freq, "field", "gene_count", output_dir / "top_fields_by_gene_coverage.png", "Base Field Gene Coverage Top30")
    save_bar_plot(op_freq, "operator", "count", output_dir / "operator_frequency.png", "Operator Frequency Top30")
    if not semantic.empty:
        family = semantic[semantic["category_type"] == "data_family"].copy()
        save_bar_plot(family, "category", "gene_count", output_dir / "data_family_gene_coverage.png", "Data Family Gene Coverage")
    save_bar_plot(mode_combiner, "combiner", "gene_count", output_dir / "combiner_gene_count.png", "Combiner Gene Count")

    summary = {
        "result_path": str(result_path),
        "metadata_path": str(args.metadata_path),
        "output_dir": str(output_dir),
        "total_rows": int(len(source_df)),
        "filtered_rows": int(len(filtered_df)),
        "filters": {
            "no_quality_filter": bool(args.no_quality_filter),
            "min_train_abs_rank_ic": args.min_train_abs_rank_ic,
            "min_train_rank_ic_ir": args.min_train_rank_ic_ir,
            "passed_validation_only": bool(args.passed_validation_only),
            "dedupe": bool(args.dedupe),
        },
        "field_occurrences": int(len(field_long)),
        "operator_occurrences": int(len(operator_long)),
        "top_fields": field_freq.head(20).to_dict(orient="records") if not field_freq.empty else [],
        "top_operators": op_freq.head(20).to_dict(orient="records") if not op_freq.empty else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_path = args.result_path or detect_latest_behavior_result()
    result_path = result_path.resolve()
    output_dir = args.output_dir or (result_path.parent / DEFAULT_OUTPUT_SUBDIR)

    source_df = read_result_table(result_path)
    filtered_df = metric_filter(source_df, args)
    field_meta = load_field_metadata(args.metadata_path)
    field_long, operator_long, parsed_factors = parse_behavior_results(filtered_df, field_meta)
    tables = build_frequency_tables(field_long, operator_long, parsed_factors)
    write_outputs(
        output_dir=output_dir,
        source_df=source_df,
        filtered_df=filtered_df,
        field_long=field_long,
        operator_long=operator_long,
        parsed_factors=parsed_factors,
        tables=tables,
        args=args,
        result_path=result_path,
    )

    print(f"loaded rows: {len(source_df)}")
    print(f"analyzed rows: {len(filtered_df)}")
    print(f"field occurrences: {len(field_long)}")
    print(f"operator occurrences: {len(operator_long)}")
    print(f"saved analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
