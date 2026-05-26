from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.data.make_metadata_from_columns import build_metadata, write_metadata


ROOT = Path(__file__).resolve().parent
DEFAULT_SCREEN = ROOT / "tmt_fundamental_table_screen.json"
DEFAULT_OUTPUT = ROOT / "candidate_metadata_from_screen.json"

SPECIAL_COLUMNS = [
    "industry_code",
    "is_tradeable",
    "label_20d",
    "label_5d",
    "label_1d",
    "Close",
]


def _append_unique(output: list[str], seen: set[str], values: list[str]) -> None:
    for value in values:
        field = str(value)
        if field in seen:
            continue
        seen.add(field)
        output.append(field)


def columns_from_screen(screen_path: str | Path, *, include_other: bool = False) -> list[str]:
    """Expand the reviewed table-screen JSON into candidate metadata columns.

    The screen file contains table-level white/other/drop decisions, not the
    exact final panel columns. Use this only for candidate metadata. For real
    runs, prefer make_metadata_from_columns.py with final_tmt_panel.columns.
    """

    screen = json.loads(Path(screen_path).read_text(encoding="utf-8"))
    columns: list[str] = []
    seen: set[str] = set()

    for table in screen:
        _append_unique(columns, seen, table.get("white", []))
        if include_other:
            _append_unique(columns, seen, table.get("other", []))

    _append_unique(columns, seen, SPECIAL_COLUMNS)
    return columns


def build_real_metadata(
    screen_path: str | Path = DEFAULT_SCREEN,
    *,
    include_other: bool = False,
    dataset: str = "real_tmt_daily.parquet",
    label_col: str = "label_20d",
    tradeable_col: str = "is_tradeable",
    industry_col: str = "industry_code",
) -> dict[str, object]:
    columns = columns_from_screen(screen_path, include_other=include_other)
    metadata = build_metadata(
        columns,
        dataset=dataset,
        label_col=label_col,
        tradeable_col=tradeable_col,
        industry_col=industry_col,
    )
    metadata["source_columns_file"] = str(Path(screen_path).name)
    metadata["notes"] = [
        *metadata["notes"],
        "Candidate metadata built from tmt_fundamental_table_screen.json, not from exact final panel columns.",
        "Use make_metadata_from_columns.py with final_tmt_panel.columns.tolist() for exact real_metadata.json.",
        "label_20d remains the training target. label_1d and label_5d are kept as data columns for diagnostics only.",
    ]
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate TMT metadata from reviewed screenshot-column screen.")
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="real_tmt_daily.parquet")
    parser.add_argument("--label-col", default="label_20d")
    parser.add_argument("--tradeable-col", default="is_tradeable")
    parser.add_argument("--industry-col", default="industry_code")
    parser.add_argument("--include-other", action="store_true", help="Also include other-review fields from the screen file.")
    parser.add_argument("--expected-columns", type=int, default=None, help="Fail if expanded candidate column count differs.")
    args = parser.parse_args()

    columns = columns_from_screen(args.screen, include_other=args.include_other)
    if args.expected_columns is not None and len(columns) != args.expected_columns:
        raise ValueError(f"expanded candidate columns={len(columns)} differs from expected={args.expected_columns}")

    metadata = build_real_metadata(
        args.screen,
        include_other=args.include_other,
        dataset=args.dataset,
        label_col=args.label_col,
        tradeable_col=args.tradeable_col,
        industry_col=args.industry_col,
    )
    output = write_metadata(metadata, args.output)
    print(f"saved: {output}")
    print(f"field_rules: {len(metadata['field_rules'])}")
    print(f"label_col: {metadata['label_col']}")


if __name__ == "__main__":
    main()
