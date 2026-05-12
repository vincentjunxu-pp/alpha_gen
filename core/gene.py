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
#   ratio_product: (A / B) * (C / D)
#
# Every calculated expression is size-neutralized later in factor_calc.py or
# torch_backend.py. This file only controls legal parameter combinations.
# ---------------------------------------------------------------------------

MODE_CHOICES = ("single", "ratio", "pair_ratio", "resi", "ratio_product")
PAIR_OP_CHOICES = ("+", "-")
TRANSFORM_CHOICES = (
    "current",
    "log",
    "zscore",
    "diff_2q",
    "diff_1y",
    "pct_2q",
    "pct_1y",
    "std_2q",
    "std_1y",
)
TRANSFORM_WINDOWS = {
    "2q": 120,
    "1y": 244,
}

MODE_DESCRIPTION = {
    "single": "单字段变换：A",
    "ratio": "双字段比值：A / B",
    "pair_ratio": "组合比值：(A +/- B) / (C +/- D)",
    "resi": "截面残差：residual(A ~ B)",
    "ratio_product": "比值乘积：(A / B) * (C / D)",
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

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FieldRule":
        """Build a rule from JSON metadata."""

        return cls(
            can_y=bool(raw["can_y"]),
            can_x=bool(raw["can_x"]),
            allow_log=bool(raw["allow_log"]),
            allow_current=bool(raw["allow_current"]),
            allow_lag=bool(raw["allow_lag"]),
            allow_diff=bool(raw["allow_diff"]),
            allow_pct=bool(raw["allow_pct"]),
            allow_std=bool(raw["allow_std"]),
        )


@dataclass(frozen=True)
class FactorGene:
    """Parameters of one structured factor expression.

    Some fields are inactive under simpler modes:
    - single uses only a.
    - ratio/resi use a and b.
    - pair_ratio uses a, b, c, d and both pair operators.
    - ratio_product uses a, b, c, d.

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
        transforms.append("zscore")
    if rule.allow_log and rule.allow_current:
        transforms.append("log")
    if rule.allow_diff:
        transforms.extend(["diff_2q", "diff_1y"])
    if rule.allow_pct:
        transforms.extend(["pct_2q", "pct_1y"])
    if rule.allow_std:
        transforms.extend(["std_2q", "std_1y"])
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


def _choice(rng: np.random.Generator, values: list[str] | tuple[str, ...]) -> str:
    """Numpy's choice returns np scalar types; convert to plain Python str."""

    return str(rng.choice(values))


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
    if transform not in TRANSFORM_CHOICES:
        return f"{parameter} transform must be one of {TRANSFORM_CHOICES}, got {transform!r}"
    if transform not in allowed_transforms(rule):
        return f"{parameter} field {field!r} does not allow transform {transform!r}"
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

    active_specs: list[tuple[str, str, str, str]] = [("a", gene.a, gene.a_transform, "y")]
    if gene.mode in {"ratio", "resi"}:
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
    elif gene.mode == "ratio_product":
        active_specs.extend(
            [
                ("b", gene.b, gene.b_transform, "x"),
                ("c", gene.c, gene.c_transform, "y"),
                ("d", gene.d, gene.d_transform, "x"),
            ]
        )

    for parameter, field, transform, role in active_specs:
        error = _field_error(field, transform, field_rules, role=role, parameter=parameter)
        if error is not None:
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

    pool = y_fields(field_rules) if role == "y" else x_fields(field_rules)
    if not pool:
        raise ValueError(f"no fields can be used as role {role!r}")

    if field in field_rules:
        rule = field_rules[field]
        if allowed_transforms(rule) and ((role == "y" and rule.can_y) or (role == "x" and rule.can_x)):
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


def random_gene(field_rules: Mapping[str, FieldRule], rng: np.random.Generator) -> FactorGene:
    """Sample one legal structured-expression gene."""

    y_pool = y_fields(field_rules)
    x_pool = x_fields(field_rules)
    if not y_pool:
        raise ValueError("no fields can be used as signal-side fields")
    if not x_pool:
        raise ValueError("no fields can be used as denominator/control fields")

    for _ in range(1_000):
        mode = _choice(rng, MODE_CHOICES)
        a = _choice(rng, y_pool)
        b = _choice(rng, y_pool if mode == "pair_ratio" else x_pool)
        c = _choice(rng, y_pool if mode == "ratio_product" else x_pool)
        d = _choice(rng, x_pool)
        gene = FactorGene(
            a=a,
            b=b,
            c=c,
            d=d,
            left_op=_choice(rng, PAIR_OP_CHOICES),
            right_op=_choice(rng, PAIR_OP_CHOICES),
            mode=mode,
            a_transform=_choice(rng, allowed_transforms(field_rules[a])),
            b_transform=_choice(rng, allowed_transforms(field_rules[b])),
            c_transform=_choice(rng, allowed_transforms(field_rules[c])),
            d_transform=_choice(rng, allowed_transforms(field_rules[d])),
        )
        if is_valid_gene(gene, field_rules):
            return gene

    raise RuntimeError("failed to sample a legal gene after 1000 attempts")


def repair_gene(gene: FactorGene, field_rules: Mapping[str, FieldRule], rng: np.random.Generator) -> FactorGene:
    """Repair a possibly illegal gene after crossover or mutation."""

    mode = gene.mode if gene.mode in MODE_CHOICES else _choice(rng, MODE_CHOICES)
    left_op = gene.left_op if gene.left_op in PAIR_OP_CHOICES else _choice(rng, PAIR_OP_CHOICES)
    right_op = gene.right_op if gene.right_op in PAIR_OP_CHOICES else _choice(rng, PAIR_OP_CHOICES)

    a = _repair_field(gene.a, field_rules, rng, role="y")
    if mode == "pair_ratio":
        b = _repair_field(gene.b, field_rules, rng, role="y")
    else:
        b = _repair_field(gene.b, field_rules, rng, role="x")
    if mode == "ratio_product":
        c = _repair_field(gene.c, field_rules, rng, role="y")
    else:
        c = _repair_field(gene.c, field_rules, rng, role="x")
    d = _repair_field(gene.d, field_rules, rng, role="x")
    a_transform = _repair_transform(a, gene.a_transform, field_rules, rng)
    b_transform = _repair_transform(b, gene.b_transform, field_rules, rng)
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
    # A remaining error would indicate inconsistent metadata rather than a
    # stochastic repair issue, so expose all errors clearly.
    raise RuntimeError("failed to repair gene: " + "; ".join(validate_gene(repaired, field_rules)))


def mutate_one_parameter(gene: FactorGene, field_rules: Mapping[str, FieldRule], rng: np.random.Generator) -> FactorGene:
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
        mutated = replace(gene, a=_choice(rng, y_fields(field_rules)))
    elif parameter == "b":
        pool = y_fields(field_rules) if gene.mode == "pair_ratio" else x_fields(field_rules)
        mutated = replace(gene, b=_choice(rng, pool))
    elif parameter == "c":
        pool = y_fields(field_rules) if gene.mode == "ratio_product" else x_fields(field_rules)
        mutated = replace(gene, c=_choice(rng, pool))
    elif parameter == "d":
        mutated = replace(gene, d=_choice(rng, x_fields(field_rules)))
    elif parameter == "left_op":
        mutated = replace(gene, left_op=_choice(rng, PAIR_OP_CHOICES))
    elif parameter == "right_op":
        mutated = replace(gene, right_op=_choice(rng, PAIR_OP_CHOICES))
    elif parameter == "mode":
        mutated = replace(gene, mode=_choice(rng, MODE_CHOICES))
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

    return repair_gene(mutated, field_rules, rng)


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
    elif gene.mode == "ratio_product":
        body = f"({a} / {b}) * ({c} / {d})"
    else:
        body = f"<非法 mode={gene.mode}>"

    return f"市值中性化({body})"
