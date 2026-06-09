from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Gene definition for the structured-expression search.
#
# The GA searches a small set of readable expression templates over raw fields
# plus a per-field unary transform:
#
#   single:     A
#   ratio:      A / B
#   pair_ratio: (A +/- B) / (C +/- D)
#   resi:       residual(A ~ B)
#   resi_pair:  residual(A ~ B + C)
#   multi_resi: residual(A ~ B + C + D)
#   spread:     A / B - C / D
#   style_composite: combine(rank_score(A), rank_score(B))
#
# Every calculated expression is size-neutralized later in factor_calc.py or
# torch_backend.py. This file only controls legal parameter combinations.
# ---------------------------------------------------------------------------

MODE_CHOICES = ("single", "ratio", "pair_ratio", "resi", "resi_pair", "multi_resi", "spread", "style_composite")
PAIR_OP_CHOICES = ("+", "-")
ACCOUNTING_PAIR_TRANSFORMS = ("current", "diff_2q", "diff_1y")
STYLE_COMPOSITE_OP = "+"
TRANSFORM_CHOICES = (
    "current",
    "log",
    "rank_pct",
    "zscore",
    "ind_rank_pct",
    "ind_zscore",
    "diff_2q",
    "diff_1y",
    "pct_2q",
    "pct_1y",
)
TRANSFORM_WINDOWS = {
    "2q": 120,
    "1y": 244,
}

MODE_DESCRIPTION = {
    "single": "单指标：A",
    "ratio": "受限比值：A / B",
    "pair_ratio": "受限组合比值：(A +/- B) / (C +/- D)",
    "resi": "截面残差：residual(A ~ B)",
    "resi_pair": "双项控制截面残差：residual(A ~ B + C)",
    "multi_resi": "三项控制截面残差：residual(A ~ B + C + D)",
    "spread": "比值价差：A / B - C / D",
    "style_composite": "风格复合：rank_score(A) + rank_score(B)",
}

SINGLE_UNIT_TYPES = {"ratio", "rate", "growth", "score", "turnover", "yield"}
RATIO_UNIT_PAIRS = {
    ("currency", "currency"),
    ("ratio", "ratio"),
    ("rate", "rate"),
    ("growth", "growth"),
}
RATIO_ADD_GROUP_PAIRS = {
    ("profit", "market_value"),
    ("cashflow", "market_value"),
    ("revenue", "market_value"),
    ("equity", "market_value"),
    ("asset", "market_value"),
    ("profit", "revenue"),
    ("cashflow", "revenue"),
    ("expense", "revenue"),
    ("debt", "asset"),
    ("cash", "asset"),
    ("cash", "debt"),
    ("working_capital", "asset"),
    ("working_capital", "revenue"),
    ("revenue", "asset"),
    ("profit", "asset"),
    ("profit", "equity"),
    ("cashflow", "debt"),
    ("equity", "asset"),
}
STYLE_FAMILY_PAIRS = {
    frozenset(("quality", "valuation")),
    frozenset(("profitability", "valuation")),
    frozenset(("growth", "valuation")),
    frozenset(("analyst", "valuation")),
    frozenset(("quality", "leverage")),
    frozenset(("quality", "liquidity")),
    frozenset(("cashflow", "profitability")),
    frozenset(("efficiency", "profitability")),
}


@dataclass(frozen=True)
class FieldRule:
    """One input field's allowed roles.

    The original 11-parameter version also used the transform flags below. They
    are kept in metadata compatibility terms because preprocess.py can still
    build old transform caches for experiments, but the new gene templates only
    use current prepared fields. `can_y` controls signal-side fields; `can_x`
    controls denominator/control-side fields.
    """

    can_y: bool
    can_x: bool
    allow_log: bool
    allow_current: bool
    allow_lag: bool
    allow_diff: bool
    allow_pct: bool
    allow_std: bool
    family: str = "other"
    unit_type: str = "unknown"
    statement: str = "other"
    period_type: str = "unknown"
    direction: int = 1
    add_group: str = "unknown"
    allow_industry_relative: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FieldRule":
        """Build a rule from JSON metadata."""

        return cls(
            can_y=bool(raw["can_y"]),
            can_x=bool(raw["can_x"]),
            allow_log=bool(raw["allow_log"]),
            allow_current=bool(raw["allow_current"]),
            allow_lag=bool(raw.get("allow_lag", False)),
            allow_diff=bool(raw["allow_diff"]),
            allow_pct=bool(raw["allow_pct"]),
            allow_std=bool(raw.get("allow_std", False)),
            family=str(raw.get("family", "other")),
            unit_type=str(raw.get("unit_type", "unknown")),
            statement=str(raw.get("statement", "other")),
            period_type=str(raw.get("period_type", "unknown")),
            direction=int(raw.get("direction", 1)),
            add_group=str(raw.get("add_group", raw.get("family", "unknown"))),
            allow_industry_relative=bool(raw.get("allow_industry_relative", True)),
        )


@dataclass(frozen=True)
class FactorGene:
    """Parameters of one structured factor expression.

    Some fields are inactive under simpler modes:
    - single uses only a.
    - ratio/resi use a and b.
    - resi_pair uses a, b and c.
    - multi_resi/spread use a, b, c and d.
    - pair_ratio uses a, b, c, d and both pair operators.
    - style_composite uses a, b and left_op.

    Inactive values are still stored so uniform crossover/mutation can stay
    simple. Semantic de-duplication in ga.py ignores inactive values.
    """

    a: str
    b: str
    c: str
    d: str
    left_op: str
    right_op: str
    mode: str
    a_transform: str = "current"
    b_transform: str = "current"
    c_transform: str = "current"
    d_transform: str = "current"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logging/results."""

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FactorGene":
        """Restore a gene from JSON/CSV records."""

        return cls(
            a=str(raw["a"]),
            b=str(raw["b"]),
            c=str(raw["c"]),
            d=str(raw["d"]),
            left_op=str(raw["left_op"]),
            right_op=str(raw["right_op"]),
            mode=str(raw["mode"]),
            a_transform=str(raw.get("a_transform", "current")),
            b_transform=str(raw.get("b_transform", "current")),
            c_transform=str(raw.get("c_transform", "current")),
            d_transform=str(raw.get("d_transform", "current")),
        )


def allowed_transforms(rule: FieldRule) -> list[str]:
    """Transforms allowed for one field under metadata rules."""

    transforms: list[str] = []
    if rule.allow_current:
        transforms.append("current")
        transforms.append("rank_pct")
        transforms.append("zscore")
        if rule.allow_industry_relative:
            transforms.append("ind_rank_pct")
            transforms.append("ind_zscore")
    if rule.allow_log and rule.allow_current:
        transforms.append("log")
    if rule.allow_diff:
        transforms.extend(["diff_2q", "diff_1y"])
    if rule.allow_pct:
        transforms.extend(["pct_2q", "pct_1y"])
    return transforms


def load_field_rules(metadata_path: str | Path) -> dict[str, FieldRule]:
    """Load field-level legality rules from metadata."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return {field: FieldRule.from_dict(rule) for field, rule in metadata["field_rules"].items()}


def y_fields(field_rules: Mapping[str, FieldRule]) -> list[str]:
    """Fields that may appear on the signal side of an expression."""

    return [field for field, rule in field_rules.items() if rule.can_y and allowed_transforms(rule)]


def x_fields(field_rules: Mapping[str, FieldRule]) -> list[str]:
    """Fields that may appear as denominator/control variables."""

    return [field for field, rule in field_rules.items() if rule.can_x and allowed_transforms(rule)]


def any_fields(field_rules: Mapping[str, FieldRule]) -> list[str]:
    """Fields that may appear in unrestricted expression positions."""

    return [field for field, rule in field_rules.items() if allowed_transforms(rule)]


def _choice(rng: np.random.Generator, values: list[str] | tuple[str, ...]) -> str:
    """Numpy's choice returns np scalar types; convert to plain Python str."""

    return str(rng.choice(values))


def _choice_mode(
    rng: np.random.Generator,
    mode_probabilities: Mapping[str, float] | None = None,
) -> str:
    """Sample one mode, using equal probability unless configured otherwise."""

    if mode_probabilities is None:
        return _choice(rng, MODE_CHOICES)

    unknown = sorted(set(mode_probabilities) - set(MODE_CHOICES))
    if unknown:
        raise ValueError(f"mode_probabilities contains unknown modes: {unknown}")

    weights = np.array([float(mode_probabilities.get(mode, 0.0)) for mode in MODE_CHOICES], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("mode_probabilities must contain finite non-negative weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("mode_probabilities must assign positive total weight")
    probabilities = weights / total
    return str(rng.choice(MODE_CHOICES, p=probabilities))


def _field_error(
    field: str,
    transform: str,
    field_rules: Mapping[str, FieldRule],
    *,
    role: str,
    parameter: str,
) -> str | None:
    """Validate one active field for its expression role."""

    if field not in field_rules:
        return f"unknown {parameter} field: {field!r}"

    rule = field_rules[field]
    if role == "y" and not rule.can_y:
        return f"{parameter} field {field!r} cannot be used as signal side"
    if role == "x" and not rule.can_x:
        return f"{parameter} field {field!r} cannot be used as denominator/control side"
    if role not in {"y", "x", "any"}:
        return f"unknown field role {role!r} for {parameter}"
    if transform not in TRANSFORM_CHOICES:
        return f"{parameter} transform must be one of {TRANSFORM_CHOICES}, got {transform!r}"
    if transform not in allowed_transforms(rule):
        return f"{parameter} field {field!r} does not allow transform {transform!r}"
    return None


def _is_single_metric(rule: FieldRule) -> bool:
    """Whether a raw field is meaningful as a standalone signal."""

    if rule.unit_type in SINGLE_UNIT_TYPES:
        return True
    if rule.family == "valuation" and rule.unit_type in {"ratio", "rate", "yield"}:
        return True
    if rule.family == "analyst" and rule.unit_type == "score":
        return True
    return False


def _is_accounting_transform(transform: str) -> bool:
    """Transforms that preserve accounting-unit arithmetic."""

    return transform in ACCOUNTING_PAIR_TRANSFORMS


def _ratio_pair_error(
    left_field: str,
    left_transform: str,
    right_field: str,
    right_transform: str,
    field_rules: Mapping[str, FieldRule],
    *,
    context: str,
) -> str | None:
    """Validate one semantically meaningful ratio A / B."""

    if left_field == right_field and left_transform == right_transform:
        return f"{context} cannot divide a field by itself"
    if not _is_accounting_transform(left_transform) or not _is_accounting_transform(right_transform):
        return f"{context} only allows accounting-unit transforms {ACCOUNTING_PAIR_TRANSFORMS}"
    if left_transform != right_transform:
        return f"{context} requires numerator and denominator to use the same transform"

    left_rule = field_rules[left_field]
    right_rule = field_rules[right_field]
    unit_pair = (left_rule.unit_type, right_rule.unit_type)
    if unit_pair not in RATIO_UNIT_PAIRS:
        return f"{context} unit pair {unit_pair!r} is not in the ratio whitelist"
    if left_rule.add_group == "score" or right_rule.add_group == "score":
        return f"{context} does not allow score-like fields in accounting ratios"
    add_group_pair = (left_rule.add_group, right_rule.add_group)
    if add_group_pair not in RATIO_ADD_GROUP_PAIRS:
        return f"{context} add_group pair {add_group_pair!r} is not in the ratio template whitelist"
    return None


def _addition_pair_error(
    left_field: str,
    left_transform: str,
    right_field: str,
    right_transform: str,
    op: str,
    field_rules: Mapping[str, FieldRule],
    *,
    context: str,
) -> str | None:
    """Validate A +/- B before it becomes part of a pair_ratio."""

    if left_field == right_field and left_transform == right_transform:
        return f"{context} cannot combine the same field with itself"
    if not _is_accounting_transform(left_transform) or not _is_accounting_transform(right_transform):
        return f"{context} only allows accounting-unit transforms {ACCOUNTING_PAIR_TRANSFORMS}"
    if left_transform != right_transform:
        return f"{context} requires both fields to use the same transform"

    left_rule = field_rules[left_field]
    right_rule = field_rules[right_field]
    if left_rule.unit_type != right_rule.unit_type:
        return f"{context} requires matching unit_type, got {left_rule.unit_type!r} and {right_rule.unit_type!r}"
    if left_rule.add_group != right_rule.add_group:
        return f"{context} requires matching add_group, got {left_rule.add_group!r} and {right_rule.add_group!r}"
    if op == "-" and left_rule.add_group in {"score", "unknown"}:
        return f"{context} subtraction is not allowed for add_group {left_rule.add_group!r}"
    return None


def _pair_ratio_error(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> str | None:
    """Validate the full pair_ratio template."""

    left_error = _addition_pair_error(
        gene.a,
        gene.a_transform,
        gene.b,
        gene.b_transform,
        gene.left_op,
        field_rules,
        context="pair_ratio numerator",
    )
    if left_error:
        return left_error

    right_error = _addition_pair_error(
        gene.c,
        gene.c_transform,
        gene.d,
        gene.d_transform,
        gene.right_op,
        field_rules,
        context="pair_ratio denominator",
    )
    if right_error:
        return right_error

    numerator_rule = field_rules[gene.a]
    denominator_rule = field_rules[gene.c]
    unit_pair = (numerator_rule.unit_type, denominator_rule.unit_type)
    if unit_pair not in RATIO_UNIT_PAIRS:
        return f"pair_ratio unit pair {unit_pair!r} is not in the ratio whitelist"
    add_group_pair = (numerator_rule.add_group, denominator_rule.add_group)
    if add_group_pair not in RATIO_ADD_GROUP_PAIRS:
        return f"pair_ratio add_group pair {add_group_pair!r} is not in the ratio template whitelist"
    return None


def _resi_error(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> str | None:
    """Validate residual(A ~ B) semantics."""

    del field_rules
    if gene.a == gene.b and gene.a_transform == gene.b_transform:
        return "resi cannot regress a field on itself"
    return None


def _additive_control_error(
    controls: list[tuple[str, str]],
    field_rules: Mapping[str, FieldRule],
    *,
    context: str,
) -> str | None:
    """Validate B + C or B + C + D before it is used as one residual control."""

    if len(set(controls)) != len(controls):
        return f"{context} controls must be distinct field/transform pairs"

    first_field, first_transform = controls[0]
    if not _is_accounting_transform(first_transform):
        return f"{context} only allows accounting-unit transforms {ACCOUNTING_PAIR_TRANSFORMS}"

    first_rule = field_rules[first_field]
    for field, transform in controls[1:]:
        if transform != first_transform:
            return f"{context} controls must use the same transform"
        if not _is_accounting_transform(transform):
            return f"{context} only allows accounting-unit transforms {ACCOUNTING_PAIR_TRANSFORMS}"
        rule = field_rules[field]
        if rule.unit_type != first_rule.unit_type:
            return f"{context} requires matching unit_type, got {first_rule.unit_type!r} and {rule.unit_type!r}"
        if rule.add_group != first_rule.add_group:
            return f"{context} requires matching add_group, got {first_rule.add_group!r} and {rule.add_group!r}"
    return None


def _resi_pair_error(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> str | None:
    """Validate residual(A ~ B + C) semantics."""

    target = (gene.a, gene.a_transform)
    controls = [
        (gene.b, gene.b_transform),
        (gene.c, gene.c_transform),
    ]
    if target in controls:
        return "resi_pair cannot use the target as a control"
    return _additive_control_error(controls, field_rules, context="resi_pair additive control")


def _multi_resi_error(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> str | None:
    """Validate residual(A ~ B + C + D) semantics."""

    target = (gene.a, gene.a_transform)
    controls = [
        (gene.b, gene.b_transform),
        (gene.c, gene.c_transform),
        (gene.d, gene.d_transform),
    ]
    if target in controls:
        return "multi_resi cannot use the target as a control"
    if len(set(controls)) != len(controls):
        return "multi_resi controls must be distinct field/transform pairs"
    return _additive_control_error(controls, field_rules, context="multi_resi additive control")


def _spread_error(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> str | None:
    """Validate A / B - C / D semantics."""

    left_error = _ratio_pair_error(
        gene.a,
        gene.a_transform,
        gene.b,
        gene.b_transform,
        field_rules,
        context="spread left ratio",
    )
    if left_error:
        return left_error

    right_error = _ratio_pair_error(
        gene.c,
        gene.c_transform,
        gene.d,
        gene.d_transform,
        field_rules,
        context="spread right ratio",
    )
    if right_error:
        return right_error

    left = (gene.a, gene.a_transform, gene.b, gene.b_transform)
    right = (gene.c, gene.c_transform, gene.d, gene.d_transform)
    if left == right:
        return "spread requires two different ratios"
    return None


def _style_composite_error(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> str | None:
    """Validate rank_score(A) + rank_score(B) as a whitelisted style pair."""

    if gene.a == gene.b:
        return "style_composite requires two different fields"
    if gene.left_op != STYLE_COMPOSITE_OP:
        return "style_composite only allows additive style combination"
    if gene.a_transform != "current" or gene.b_transform != "current":
        return "style_composite stores raw fields and applies rank scoring inside the mode; transforms must be current"
    a_rule = field_rules[gene.a]
    b_rule = field_rules[gene.b]
    pair = frozenset((a_rule.family, b_rule.family))
    if pair not in STYLE_FAMILY_PAIRS:
        return f"style_composite family pair {(a_rule.family, b_rule.family)!r} is not whitelisted"
    if not a_rule.allow_current or not b_rule.allow_current:
        return "style_composite requires current values so rank scores can be computed"
    if a_rule.unit_type == "currency" and b_rule.unit_type == "currency":
        return "style_composite should combine normalized metrics, not two raw currency fields"
    return None


def validate_gene(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> list[str]:
    """Return all legality errors for one gene."""

    errors: list[str] = []

    if gene.mode not in MODE_CHOICES:
        errors.append(f"mode must be one of {MODE_CHOICES}, got {gene.mode!r}")
        return errors

    if gene.left_op not in PAIR_OP_CHOICES:
        errors.append(f"left_op must be one of {PAIR_OP_CHOICES}, got {gene.left_op!r}")
    if gene.right_op not in PAIR_OP_CHOICES:
        errors.append(f"right_op must be one of {PAIR_OP_CHOICES}, got {gene.right_op!r}")

    if gene.mode == "resi":
        active_specs: list[tuple[str, str, str, str]] = [
            ("a", gene.a, gene.a_transform, "any"),
            ("b", gene.b, gene.b_transform, "any"),
        ]
    elif gene.mode == "resi_pair":
        active_specs = [
            ("a", gene.a, gene.a_transform, "any"),
            ("b", gene.b, gene.b_transform, "any"),
            ("c", gene.c, gene.c_transform, "any"),
        ]
    elif gene.mode == "multi_resi":
        active_specs = [
            ("a", gene.a, gene.a_transform, "any"),
            ("b", gene.b, gene.b_transform, "any"),
            ("c", gene.c, gene.c_transform, "any"),
            ("d", gene.d, gene.d_transform, "any"),
        ]
    else:
        active_specs = [("a", gene.a, gene.a_transform, "y")]

    if gene.mode == "ratio":
        active_specs.append(("b", gene.b, gene.b_transform, "x"))
    elif gene.mode == "spread":
        active_specs.extend(
            [
                ("b", gene.b, gene.b_transform, "x"),
                ("c", gene.c, gene.c_transform, "y"),
                ("d", gene.d, gene.d_transform, "x"),
            ]
        )
    elif gene.mode == "style_composite":
        active_specs.append(("b", gene.b, gene.b_transform, "x"))
    elif gene.mode == "pair_ratio":
        # A and B form the left/signal side; C and D form the denominator or
        # regression-control side.
        active_specs.extend(
            [
                ("b", gene.b, gene.b_transform, "y"),
                ("c", gene.c, gene.c_transform, "x"),
                ("d", gene.d, gene.d_transform, "x"),
            ]
        )

    for parameter, field, transform, role in active_specs:
        error = _field_error(field, transform, field_rules, role=role, parameter=parameter)
        if error is not None:
            errors.append(error)

    if errors:
        return errors

    if gene.mode == "single" and not _is_single_metric(field_rules[gene.a]):
        errors.append(
            f"single only allows ratio/rate/growth/score/turnover/valuation-yield fields, "
            f"got family={field_rules[gene.a].family!r}, unit_type={field_rules[gene.a].unit_type!r}"
        )
    elif gene.mode == "ratio":
        error = _ratio_pair_error(
            gene.a,
            gene.a_transform,
            gene.b,
            gene.b_transform,
            field_rules,
            context="ratio",
        )
        if error:
            errors.append(error)
    elif gene.mode == "pair_ratio":
        error = _pair_ratio_error(gene, field_rules)
        if error:
            errors.append(error)
    elif gene.mode == "resi":
        error = _resi_error(gene, field_rules)
        if error:
            errors.append(error)
    elif gene.mode == "resi_pair":
        error = _resi_pair_error(gene, field_rules)
        if error:
            errors.append(error)
    elif gene.mode == "multi_resi":
        error = _multi_resi_error(gene, field_rules)
        if error:
            errors.append(error)
    elif gene.mode == "spread":
        error = _spread_error(gene, field_rules)
        if error:
            errors.append(error)
    elif gene.mode == "style_composite":
        error = _style_composite_error(gene, field_rules)
        if error:
            errors.append(error)

    return errors


def is_valid_gene(gene: FactorGene, field_rules: Mapping[str, FieldRule]) -> bool:
    """Boolean wrapper for callers that only need pass/fail."""

    return not validate_gene(gene, field_rules)


def _repair_field(
    field: str,
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
    *,
    role: str,
) -> str:
    """Keep a legal field if possible; otherwise sample from the right pool."""

    if role == "y":
        pool = y_fields(field_rules)
    elif role == "x":
        pool = x_fields(field_rules)
    elif role == "any":
        pool = any_fields(field_rules)
    else:
        raise ValueError(f"unknown field role {role!r}")
    if not pool:
        raise ValueError(f"no fields can be used as role {role!r}")

    if field in field_rules:
        rule = field_rules[field]
        if allowed_transforms(rule) and (
            (role == "y" and rule.can_y) or (role == "x" and rule.can_x) or role == "any"
        ):
            return field
    return _choice(rng, pool)


def _repair_transform(
    field: str,
    transform: str,
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
) -> str:
    """Keep a legal transform if possible; otherwise sample one."""

    transforms = allowed_transforms(field_rules[field])
    if not transforms:
        raise ValueError(f"field {field!r} has no allowed transforms")
    if transform in transforms:
        return transform
    return _choice(rng, transforms)


_SAMPLER_CACHE: dict[int, dict[str, object]] = {}


def _random_fill_field(
    pool: list[str],
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Sample an inactive field/transform pair for complete gene records."""

    field = _choice(rng, pool)
    return field, _choice(rng, allowed_transforms(field_rules[field]))


def _inactive_field_pool(field_rules: Mapping[str, FieldRule]) -> list[str]:
    """Fields used only to keep inactive gene slots populated."""

    return x_fields(field_rules) or any_fields(field_rules)


def _sampler_cache(field_rules: Mapping[str, FieldRule]) -> dict[str, object]:
    """Build mode-aware candidate pools once for a field-rule mapping."""

    cache_key = id(field_rules)
    if cache_key in _SAMPLER_CACHE:
        return _SAMPLER_CACHE[cache_key]

    y_pool = y_fields(field_rules)
    x_pool = x_fields(field_rules)
    any_pool = any_fields(field_rules)
    single_fields = [field for field in y_pool if _is_single_metric(field_rules[field])]

    ratio_candidates: list[tuple[str, str, str]] = []
    for a in y_pool:
        a_transforms = set(allowed_transforms(field_rules[a]))
        for b in x_pool:
            common_transforms = [
                transform
                for transform in ACCOUNTING_PAIR_TRANSFORMS
                if transform in a_transforms and transform in allowed_transforms(field_rules[b])
            ]
            for transform in common_transforms:
                if _ratio_pair_error(a, transform, b, transform, field_rules, context="ratio") is None:
                    ratio_candidates.append((a, b, transform))

    def addition_groups(
        pool: list[str],
        *,
        ops: tuple[str, ...] = PAIR_OP_CHOICES,
    ) -> dict[tuple[str, str, str], list[tuple[str, str, str, str]]]:
        groups: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = {}
        for left in pool:
            left_transforms = set(allowed_transforms(field_rules[left]))
            for right in pool:
                for transform in ACCOUNTING_PAIR_TRANSFORMS:
                    if transform not in left_transforms or transform not in allowed_transforms(field_rules[right]):
                        continue
                    for op in ops:
                        if _addition_pair_error(left, transform, right, transform, op, field_rules, context="pair") is None:
                            rule = field_rules[left]
                            key = (rule.unit_type, rule.add_group, transform)
                            groups.setdefault(key, []).append((left, right, op, transform))
        return groups

    left_addition_groups = addition_groups(y_pool)
    right_addition_groups = addition_groups(x_pool)
    pair_ratio_group_pairs: list[tuple[tuple[str, str, str], tuple[str, str, str], int]] = []
    for left_key, left_values in left_addition_groups.items():
        left_unit, left_group, _left_transform = left_key
        for right_key, right_values in right_addition_groups.items():
            right_unit, right_group, _right_transform = right_key
            if (left_unit, right_unit) not in RATIO_UNIT_PAIRS:
                continue
            if (left_group, right_group) not in RATIO_ADD_GROUP_PAIRS:
                continue
            pair_ratio_group_pairs.append((left_key, right_key, len(left_values) * len(right_values)))

    resi_targets = [
        (field, transform)
        for field in any_pool
        for transform in allowed_transforms(field_rules[field])
    ]
    resi_controls = [
        (field, transform)
        for field in any_pool
        for transform in allowed_transforms(field_rules[field])
    ]
    additive_control_terms: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    for field in any_pool:
        rule = field_rules[field]
        for transform in ACCOUNTING_PAIR_TRANSFORMS:
            if transform in allowed_transforms(rule):
                key = (rule.unit_type, rule.add_group, transform)
                additive_control_terms.setdefault(key, []).append((field, transform))
    resi_pair_controls: list[tuple[tuple[str, str], tuple[str, str]]] = []
    resi_triple_controls: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]] = []
    for terms in additive_control_terms.values():
        unique_terms = sorted(set(terms))
        for left_idx, left in enumerate(unique_terms):
            for right in unique_terms[left_idx + 1 :]:
                resi_pair_controls.append((left, right))
        for first_idx, first in enumerate(unique_terms):
            for second_idx in range(first_idx + 1, len(unique_terms)):
                second = unique_terms[second_idx]
                for third in unique_terms[second_idx + 1 :]:
                    resi_triple_controls.append((first, second, third))
    spread_candidates: list[tuple[tuple[str, str, str], tuple[str, str, str]]] = []
    for left in ratio_candidates:
        for right in ratio_candidates:
            if left != right:
                spread_candidates.append((left, right))

    style_pairs = [
        (a, b)
        for a in y_pool
        for b in x_pool
        if a != b
        and field_rules[a].allow_current
        and field_rules[b].allow_current
        and frozenset((field_rules[a].family, field_rules[b].family)) in STYLE_FAMILY_PAIRS
        and not (field_rules[a].unit_type == "currency" and field_rules[b].unit_type == "currency")
    ]

    cache: dict[str, object] = {
        "y_pool": y_pool,
        "x_pool": x_pool,
        "any_pool": any_pool,
        "single_fields": single_fields,
        "ratio_candidates": ratio_candidates,
        "left_addition_groups": left_addition_groups,
        "right_addition_groups": right_addition_groups,
        "pair_ratio_group_pairs": pair_ratio_group_pairs,
        "resi_targets": resi_targets,
        "resi_controls": resi_controls,
        "resi_pair_controls": resi_pair_controls,
        "resi_triple_controls": resi_triple_controls,
        "spread_candidates": spread_candidates,
        "style_pairs": style_pairs,
    }
    _SAMPLER_CACHE[cache_key] = cache
    return cache


def _random_gene_for_mode(
    mode: str,
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
) -> FactorGene:
    """Sample one legal gene from mode-specific candidate pools."""

    cache = _sampler_cache(field_rules)
    y_pool = cache["y_pool"]
    x_pool = cache["x_pool"]
    any_pool = cache["any_pool"]
    inactive_pool = x_pool or any_pool
    if not inactive_pool:
        raise ValueError("no fields can be used as inactive stored fields")

    if mode == "single":
        single_fields = cache["single_fields"]
        if not single_fields:
            raise RuntimeError("single mode has no legal field candidates")
        a = _choice(rng, single_fields)
        b, b_transform = _random_fill_field(inactive_pool, field_rules, rng)
        c, c_transform = _random_fill_field(inactive_pool, field_rules, rng)
        d, d_transform = _random_fill_field(inactive_pool, field_rules, rng)
        return FactorGene(
            a=a,
            b=b,
            c=c,
            d=d,
            left_op=_choice(rng, PAIR_OP_CHOICES),
            right_op=_choice(rng, PAIR_OP_CHOICES),
            mode=mode,
            a_transform=_choice(rng, allowed_transforms(field_rules[a])),
            b_transform=b_transform,
            c_transform=c_transform,
            d_transform=d_transform,
        )

    if mode == "ratio":
        ratio_candidates = cache["ratio_candidates"]
        if not ratio_candidates:
            raise RuntimeError("ratio mode has no legal field candidates")
        a, b, transform = ratio_candidates[int(rng.integers(len(ratio_candidates)))]
        c, c_transform = _random_fill_field(inactive_pool, field_rules, rng)
        d, d_transform = _random_fill_field(inactive_pool, field_rules, rng)
        return FactorGene(
            a=a,
            b=b,
            c=c,
            d=d,
            left_op=_choice(rng, PAIR_OP_CHOICES),
            right_op=_choice(rng, PAIR_OP_CHOICES),
            mode=mode,
            a_transform=transform,
            b_transform=transform,
            c_transform=c_transform,
            d_transform=d_transform,
        )

    if mode == "pair_ratio":
        group_pairs = cache["pair_ratio_group_pairs"]
        if not group_pairs:
            raise RuntimeError("pair_ratio mode has no legal field candidates")
        weights = np.array([weight for *_keys, weight in group_pairs], dtype=float)
        group_id = int(rng.choice(len(group_pairs), p=weights / weights.sum()))
        left_key, right_key, _weight = group_pairs[group_id]
        left_values = cache["left_addition_groups"][left_key]
        right_values = cache["right_addition_groups"][right_key]
        a, b, left_op, left_transform = left_values[int(rng.integers(len(left_values)))]
        c, d, right_op, right_transform = right_values[int(rng.integers(len(right_values)))]
        return FactorGene(
            a=a,
            b=b,
            c=c,
            d=d,
            left_op=left_op,
            right_op=right_op,
            mode=mode,
            a_transform=left_transform,
            b_transform=left_transform,
            c_transform=right_transform,
            d_transform=right_transform,
        )

    if mode == "resi":
        targets = cache["resi_targets"]
        controls = cache["resi_controls"]
        if not targets or not controls:
            raise RuntimeError("resi mode has no legal field candidates")
        for _ in range(1_000):
            a, a_transform = targets[int(rng.integers(len(targets)))]
            b, b_transform = controls[int(rng.integers(len(controls)))]
            c, c_transform = _random_fill_field(inactive_pool, field_rules, rng)
            d, d_transform = _random_fill_field(inactive_pool, field_rules, rng)
            gene = FactorGene(
                a=a,
                b=b,
                c=c,
                d=d,
                left_op=_choice(rng, PAIR_OP_CHOICES),
                right_op=_choice(rng, PAIR_OP_CHOICES),
                mode=mode,
                a_transform=a_transform,
                b_transform=b_transform,
                c_transform=c_transform,
                d_transform=d_transform,
            )
            if is_valid_gene(gene, field_rules):
                return gene
        raise RuntimeError("resi mode failed to sample distinct target/control fields")

    if mode == "resi_pair":
        targets = cache["resi_targets"]
        control_pairs = cache["resi_pair_controls"]
        if not targets or not control_pairs:
            raise RuntimeError("resi_pair mode has no legal field candidates")
        for _ in range(1_000):
            a, a_transform = targets[int(rng.integers(len(targets)))]
            (b, b_transform), (c, c_transform) = control_pairs[int(rng.integers(len(control_pairs)))]
            d, d_transform = _random_fill_field(inactive_pool, field_rules, rng)
            gene = FactorGene(
                a=a,
                b=b,
                c=c,
                d=d,
                left_op="+",
                right_op=_choice(rng, PAIR_OP_CHOICES),
                mode=mode,
                a_transform=a_transform,
                b_transform=b_transform,
                c_transform=c_transform,
                d_transform=d_transform,
            )
            if is_valid_gene(gene, field_rules):
                return gene
        raise RuntimeError("resi_pair mode failed to sample distinct target/control fields")

    if mode == "multi_resi":
        targets = cache["resi_targets"]
        control_triples = cache["resi_triple_controls"]
        if not targets or not control_triples:
            raise RuntimeError("multi_resi mode has no legal field candidates")
        for _ in range(1_000):
            a, a_transform = targets[int(rng.integers(len(targets)))]
            (b, b_transform), (c, c_transform), (d, d_transform) = control_triples[
                int(rng.integers(len(control_triples)))
            ]
            gene = FactorGene(
                a=a,
                b=b,
                c=c,
                d=d,
                left_op=_choice(rng, PAIR_OP_CHOICES),
                right_op=_choice(rng, PAIR_OP_CHOICES),
                mode=mode,
                a_transform=a_transform,
                b_transform=b_transform,
                c_transform=c_transform,
                d_transform=d_transform,
            )
            if is_valid_gene(gene, field_rules):
                return gene
        raise RuntimeError("multi_resi mode failed to sample distinct target/control fields")

    if mode == "spread":
        spread_candidates = cache["spread_candidates"]
        if not spread_candidates:
            raise RuntimeError("spread mode has no legal field candidates")
        left, right = spread_candidates[int(rng.integers(len(spread_candidates)))]
        a, b, left_transform = left
        c, d, right_transform = right
        return FactorGene(
            a=a,
            b=b,
            c=c,
            d=d,
            left_op="-",
            right_op=_choice(rng, PAIR_OP_CHOICES),
            mode=mode,
            a_transform=left_transform,
            b_transform=left_transform,
            c_transform=right_transform,
            d_transform=right_transform,
        )

    if mode == "style_composite":
        style_pairs = cache["style_pairs"]
        if not style_pairs:
            raise RuntimeError("style_composite mode has no legal field candidates")
        a, b = style_pairs[int(rng.integers(len(style_pairs)))]
        c, c_transform = _random_fill_field(inactive_pool, field_rules, rng)
        d, d_transform = _random_fill_field(inactive_pool, field_rules, rng)
        return FactorGene(
            a=a,
            b=b,
            c=c,
            d=d,
            left_op=STYLE_COMPOSITE_OP,
            right_op=_choice(rng, PAIR_OP_CHOICES),
            mode=mode,
            a_transform="current",
            b_transform="current",
            c_transform=c_transform,
            d_transform=d_transform,
        )

    raise ValueError(f"unknown mode: {mode!r}")


def _mode_has_candidates(mode: str, cache: Mapping[str, object]) -> bool:
    if mode == "single":
        return bool(cache["single_fields"])
    if mode == "ratio":
        return bool(cache["ratio_candidates"])
    if mode == "pair_ratio":
        return bool(cache["pair_ratio_group_pairs"])
    if mode == "resi":
        return bool(cache["resi_targets"]) and bool(cache["resi_controls"])
    if mode == "resi_pair":
        return bool(cache["resi_targets"]) and bool(cache["resi_pair_controls"])
    if mode == "multi_resi":
        return bool(cache["resi_targets"]) and bool(cache["resi_triple_controls"])
    if mode == "spread":
        return bool(cache["spread_candidates"])
    if mode == "style_composite":
        return bool(cache["style_pairs"])
    return False


def _choice_viable_mode(
    rng: np.random.Generator,
    cache: Mapping[str, object],
    mode_probabilities: Mapping[str, float] | None = None,
) -> str:
    """Sample among modes that have at least one legal candidate."""

    viable_modes = [mode for mode in MODE_CHOICES if _mode_has_candidates(mode, cache)]
    if not viable_modes:
        raise RuntimeError("no mode has legal field candidates under the current metadata")

    if mode_probabilities is None:
        return _choice(rng, viable_modes)

    unknown = sorted(set(mode_probabilities) - set(MODE_CHOICES))
    if unknown:
        raise ValueError(f"mode_probabilities contains unknown modes: {unknown}")

    weights = np.array([float(mode_probabilities.get(mode, 0.0)) for mode in viable_modes], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("mode_probabilities must contain finite non-negative weights")
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError(f"mode_probabilities assign no positive weight to viable modes: {viable_modes}")
    return str(rng.choice(viable_modes, p=weights / total))


def random_gene(
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
    mode_probabilities: Mapping[str, float] | None = None,
) -> FactorGene:
    """Sample one legal structured-expression gene."""

    cache = _sampler_cache(field_rules)
    mode = _choice_viable_mode(rng, cache, mode_probabilities)
    gene = _random_gene_for_mode(mode, field_rules, rng)
    if is_valid_gene(gene, field_rules):
        return gene

    raise RuntimeError("failed to sample a legal gene from mode-specific candidate pools")


def repair_gene(
    gene: FactorGene,
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
    mode_probabilities: Mapping[str, float] | None = None,
) -> FactorGene:
    """Repair a possibly illegal gene after crossover or mutation."""

    mode = gene.mode if gene.mode in MODE_CHOICES else _choice_mode(rng, mode_probabilities)
    left_op = gene.left_op if gene.left_op in PAIR_OP_CHOICES else _choice(rng, PAIR_OP_CHOICES)
    if mode == "style_composite":
        left_op = STYLE_COMPOSITE_OP
    right_op = gene.right_op if gene.right_op in PAIR_OP_CHOICES else _choice(rng, PAIR_OP_CHOICES)

    a_role = "any" if mode in {"resi", "resi_pair", "multi_resi"} else "y"
    a = _repair_field(gene.a, field_rules, rng, role=a_role)
    if mode == "pair_ratio":
        b = _repair_field(gene.b, field_rules, rng, role="y")
    elif mode in {"resi", "resi_pair", "multi_resi"}:
        b = _repair_field(gene.b, field_rules, rng, role="any")
    else:
        b = _repair_field(gene.b, field_rules, rng, role="x")
    if mode == "spread":
        c = _repair_field(gene.c, field_rules, rng, role="y")
        d = _repair_field(gene.d, field_rules, rng, role="x")
    elif mode in {"resi_pair", "multi_resi"}:
        c = _repair_field(gene.c, field_rules, rng, role="any")
        if mode == "multi_resi":
            d = _repair_field(gene.d, field_rules, rng, role="any")
        else:
            inactive_role = "x" if x_fields(field_rules) else "any"
            d = _repair_field(gene.d, field_rules, rng, role=inactive_role)
    else:
        inactive_role = "x" if x_fields(field_rules) else "any"
        c = _repair_field(gene.c, field_rules, rng, role=inactive_role)
        d = _repair_field(gene.d, field_rules, rng, role=inactive_role)
    a_transform = "current" if mode == "style_composite" else _repair_transform(a, gene.a_transform, field_rules, rng)
    b_transform = "current" if mode == "style_composite" else _repair_transform(b, gene.b_transform, field_rules, rng)
    c_transform = _repair_transform(c, gene.c_transform, field_rules, rng)
    d_transform = _repair_transform(d, gene.d_transform, field_rules, rng)

    repaired = FactorGene(
        a=a,
        b=b,
        c=c,
        d=d,
        left_op=left_op,
        right_op=right_op,
        mode=mode,
        a_transform=a_transform,
        b_transform=b_transform,
        c_transform=c_transform,
        d_transform=d_transform,
    )
    if is_valid_gene(repaired, field_rules):
        return repaired

    # Crossover can create semantically invalid combinations even when every
    # individual field is legal for its role. Prefer a legal sample in the same
    # mode; if that mode is unavailable under this metadata, fall back to the
    # configured viable modes so offspring repair remains total.
    try:
        fallback = random_gene(field_rules, rng, mode_probabilities={mode: 1.0})
        if is_valid_gene(fallback, field_rules):
            return fallback
    except RuntimeError:
        fallback = random_gene(field_rules, rng, mode_probabilities=mode_probabilities)
        if is_valid_gene(fallback, field_rules):
            return fallback

    raise RuntimeError("failed to repair gene: " + "; ".join(validate_gene(repaired, field_rules)))


def mutate_one_parameter(
    gene: FactorGene,
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
    mode_probabilities: Mapping[str, float] | None = None,
) -> FactorGene:
    """Mutate one expression parameter and repair the result."""

    parameter = _choice(
        rng,
        (
            "a",
            "b",
            "c",
            "d",
            "left_op",
            "right_op",
            "mode",
            "a_transform",
            "b_transform",
            "c_transform",
            "d_transform",
        ),
    )

    if parameter == "a":
        pool = any_fields(field_rules) if gene.mode in {"resi", "resi_pair", "multi_resi"} else y_fields(field_rules)
        mutated = replace(gene, a=_choice(rng, pool))
    elif parameter == "b":
        if gene.mode == "pair_ratio":
            pool = y_fields(field_rules)
        elif gene.mode in {"resi", "resi_pair", "multi_resi"}:
            pool = any_fields(field_rules)
        else:
            pool = x_fields(field_rules)
        mutated = replace(gene, b=_choice(rng, pool))
    elif parameter == "c":
        if gene.mode == "spread":
            pool = y_fields(field_rules)
        elif gene.mode in {"resi_pair", "multi_resi"}:
            pool = any_fields(field_rules)
        else:
            pool = _inactive_field_pool(field_rules)
        mutated = replace(gene, c=_choice(rng, pool))
    elif parameter == "d":
        if gene.mode == "multi_resi":
            pool = any_fields(field_rules)
        elif gene.mode == "spread":
            pool = x_fields(field_rules)
        else:
            pool = _inactive_field_pool(field_rules)
        mutated = replace(gene, d=_choice(rng, pool))
    elif parameter == "left_op":
        mutated = replace(gene, left_op=_choice(rng, PAIR_OP_CHOICES))
    elif parameter == "right_op":
        mutated = replace(gene, right_op=_choice(rng, PAIR_OP_CHOICES))
    elif parameter == "mode":
        mutated = replace(gene, mode=_choice_mode(rng, mode_probabilities))
    elif parameter == "a_transform":
        mutated = replace(gene, a_transform=_choice(rng, allowed_transforms(field_rules[gene.a])))
    elif parameter == "b_transform":
        mutated = replace(gene, b_transform=_choice(rng, allowed_transforms(field_rules[gene.b])))
    elif parameter == "c_transform":
        mutated = replace(gene, c_transform=_choice(rng, allowed_transforms(field_rules[gene.c])))
    elif parameter == "d_transform":
        mutated = replace(gene, d_transform=_choice(rng, allowed_transforms(field_rules[gene.d])))
    else:
        raise AssertionError(f"unhandled parameter: {parameter}")

    return repair_gene(mutated, field_rules, rng, mode_probabilities=mode_probabilities)


def describe_gene(gene: FactorGene) -> str:
    """Create a human-readable Chinese expression description."""

    def transform_expr(field: str, transform: str) -> str:
        if transform == "current":
            return field
        return f"{transform}({field})"

    a = transform_expr(gene.a, gene.a_transform)
    b = transform_expr(gene.b, gene.b_transform)
    c = transform_expr(gene.c, gene.c_transform)
    d = transform_expr(gene.d, gene.d_transform)
    left = f"{a} {gene.left_op} {b}"
    right = f"{c} {gene.right_op} {d}"

    if gene.mode == "single":
        body = a
    elif gene.mode == "ratio":
        body = f"{a} / {b}"
    elif gene.mode == "pair_ratio":
        body = f"({left}) / ({right})"
    elif gene.mode == "resi":
        body = f"{a} 对 {b} 截面回归取残差"
    elif gene.mode == "resi_pair":
        body = f"{a} 对 ({b} + {c}) 截面回归取残差"
    elif gene.mode == "multi_resi":
        body = f"{a} 对 ({b} + {c} + {d}) 截面回归取残差"
    elif gene.mode == "spread":
        body = f"({a} / {b}) - ({c} / {d})"
    elif gene.mode == "style_composite":
        body = f"rank_score({gene.a}) + rank_score({gene.b})"
    else:
        body = f"<非法 mode={gene.mode}>"

    return f"市值中性化({body})"
