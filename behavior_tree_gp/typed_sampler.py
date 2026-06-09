from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from alpha_gen.behavior_gen.gene import (
    CONDITION_OP_CHOICES,
    DIRECTION_POLICIES,
    MODE_REGISTRY,
    BehaviorFieldRule,
    ModeSpec,
    SlotSpec,
    condition_fields_for_mode,
    fields_for_slot,
)
from alpha_gen.behavior_gen.sampler import viable_modes

from .typed_tree import (
    ROLE_PRESERVING_BINARY_OPS,
    ROLE_PRESERVING_UNARY_OPS,
    BinaryNode,
    ConditionNode,
    FieldNode,
    TemplateTree,
    TreeExpr,
    UnaryNode,
    is_valid_tree,
    node_depth,
    validate_slot_node,
    validate_tree,
)


@dataclass(frozen=True)
class TypedTreeSamplerConfig:
    """Sampling knobs for strongly typed behavior trees."""

    max_slot_depth: int = 3
    max_total_depth: int = 5
    max_nodes: int = 32
    terminal_probability: float = 0.58
    unary_probability: float = 0.20
    optional_slot_probability: float = 0.45
    condition_probability: float = 0.35
    condition_thresholds: tuple[float, ...] = (0.55, 0.60, 0.70, 0.80)
    direction_policies: tuple[str, ...] = ("fixed", "train_ic")
    unary_tree_ops: tuple[str, ...] = ROLE_PRESERVING_UNARY_OPS
    binary_tree_ops: tuple[str, ...] = ROLE_PRESERVING_BINARY_OPS


def _choice(rng: np.random.Generator, values: list[str] | tuple[str, ...]) -> str:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return str(values[int(rng.integers(len(values)))])


def _mode_weights(mode_probabilities: Mapping[str, float] | None, modes: list[str]) -> np.ndarray:
    if mode_probabilities is None:
        return np.full(len(modes), 1.0 / len(modes), dtype=float)
    unknown = sorted(set(mode_probabilities) - set(MODE_REGISTRY))
    if unknown:
        raise ValueError(f"mode_probabilities contains unknown modes: {unknown}")
    weights = np.array([float(mode_probabilities.get(mode, 0.0)) for mode in modes], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("mode_probabilities must be finite non-negative weights")
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("mode_probabilities assign no positive weight to viable modes")
    return weights / total


def _choice_mode(
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    mode_probabilities: Mapping[str, float] | None = None,
) -> str:
    modes = viable_modes(field_rules)
    if not modes:
        raise RuntimeError("no behavior mode has legal candidates under current metadata")
    return str(rng.choice(modes, p=_mode_weights(mode_probabilities, modes)))


def _allowed_unary_ops(rule: BehaviorFieldRule) -> list[str]:
    return [op for op in rule.allowed_unary_ops if op]


def _sample_field_node(
    field_rules: Mapping[str, BehaviorFieldRule],
    slot_name: str,
    slot_spec: SlotSpec,
    rng: np.random.Generator,
) -> FieldNode:
    candidates = fields_for_slot(field_rules, slot_name, slot_spec)
    if not candidates:
        raise RuntimeError(f"slot {slot_name!r} has no legal field candidates")
    # Pick a field that has at least one allowed unary op, falling back
    # to the raw candidate list if every field has an empty unary-op set.
    safe_candidates = [f for f in candidates if _allowed_unary_ops(field_rules[f])]
    if not safe_candidates:
        safe_candidates = candidates
    field = _choice(rng, safe_candidates)
    allowed_ops = _allowed_unary_ops(field_rules[field])
    unary_op = _choice(rng, allowed_ops) if allowed_ops else "rank_pct"
    return FieldNode(
        semantic_type=slot_name,
        field=field,
        unary_op=unary_op,
    )


def sample_slot_tree(
    field_rules: Mapping[str, BehaviorFieldRule],
    slot_name: str,
    slot_spec: SlotSpec,
    rng: np.random.Generator,
    *,
    config: TypedTreeSamplerConfig | None = None,
    remaining_depth: int | None = None,
) -> TreeExpr:
    """Sample one same-type subtree for a mode slot."""

    config = config or TypedTreeSamplerConfig()
    if remaining_depth is None:
        remaining_depth = config.max_slot_depth
    if remaining_depth <= 1 or rng.random() < config.terminal_probability:
        return _sample_field_node(field_rules, slot_name, slot_spec, rng)

    operator_roll = rng.random()
    if operator_roll < config.unary_probability:
        child = sample_slot_tree(
            field_rules,
            slot_name,
            slot_spec,
            rng,
            config=config,
            remaining_depth=remaining_depth - 1,
        )
        return UnaryNode(
            semantic_type=slot_name,
            op=_choice(rng, config.unary_tree_ops),
            child=child,
        )

    left = sample_slot_tree(
        field_rules,
        slot_name,
        slot_spec,
        rng,
        config=config,
        remaining_depth=remaining_depth - 1,
    )
    right = sample_slot_tree(
        field_rules,
        slot_name,
        slot_spec,
        rng,
        config=config,
        remaining_depth=remaining_depth - 1,
    )
    return BinaryNode(
        semantic_type=slot_name,
        op=_choice(rng, config.binary_tree_ops),
        left=left,
        right=right,
    )


def _sample_conditions(
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    config: TypedTreeSamplerConfig,
) -> tuple[ConditionNode, ...]:
    if mode_spec.max_conditions <= 0 or rng.random() >= config.condition_probability:
        return ()
    candidates = condition_fields_for_mode(field_rules, mode_spec)
    if not candidates:
        return ()
    max_count = min(mode_spec.max_conditions, len(candidates))
    n_conditions = int(rng.integers(1, max_count + 1))
    chosen = list(rng.choice(candidates, size=n_conditions, replace=False))
    conditions: list[ConditionNode] = []
    for field in chosen:
        field = str(field)
        rule = field_rules[field]
        condition_op = _choice(rng, CONDITION_OP_CHOICES)
        threshold = 0.0
        if condition_op in {"top_quantile", "bottom_quantile"}:
            threshold = float(config.condition_thresholds[int(rng.integers(len(config.condition_thresholds)))])
        conditions.append(
            ConditionNode(
                field=field,
                unary_op=_choice(rng, _allowed_unary_ops(rule)),
                condition_op=condition_op,
                threshold=threshold,
            )
        )
    return tuple(conditions)


def _sample_slots(
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    config: TypedTreeSamplerConfig,
) -> dict[str, TreeExpr]:
    slots: dict[str, TreeExpr] = {}
    for slot_name, slot_spec in mode_spec.slots.items():
        if slot_spec.required or rng.random() < config.optional_slot_probability:
            if fields_for_slot(field_rules, slot_name, slot_spec):
                slots[slot_name] = sample_slot_tree(field_rules, slot_name, slot_spec, rng, config=config)
    return slots


def random_typed_tree(
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: TypedTreeSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> TemplateTree:
    """Sample one valid strongly typed behavior tree."""

    config = config or TypedTreeSamplerConfig()
    failure_reasons: dict[str, int] = {}
    last_errors: list[str] = []
    for _ in range(200):
        mode = _choice_mode(field_rules, rng, mode_probabilities=mode_probabilities)
        mode_spec = MODE_REGISTRY[mode]
        tree = TemplateTree(
            mode=mode,
            combiner=_choice(rng, mode_spec.allowed_combiners),
            slots=_sample_slots(mode_spec, field_rules, rng, config),
            conditions=_sample_conditions(mode_spec, field_rules, rng, config),
            direction_policy=_choice(rng, config.direction_policies),
        )
        tree_errors = validate_tree(tree, field_rules, max_depth=config.max_total_depth, max_nodes=config.max_nodes,
                                    max_slot_depth=config.max_slot_depth)
        if not tree_errors:
            return tree
        # Aggregate failure reasons for diagnostics
        for err in tree_errors:
            category = err.split(":")[0] if ":" in err else err.split()[0] if " " in err else err
            failure_reasons[category] = failure_reasons.get(category, 0) + 1
        last_errors = tree_errors
    summary = ", ".join(f"{cat}×{cnt}" for cat, cnt in sorted(failure_reasons.items(), key=lambda x: -x[1])[:8])
    detail = "; ".join(last_errors[:5])
    raise RuntimeError(f"failed to sample a valid typed tree after 200 attempts [{summary}]. last tree errors: {detail}")


def _repair_tree(
    tree: TemplateTree,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: TypedTreeSamplerConfig,
    mode_probabilities: Mapping[str, float] | None = None,
) -> TemplateTree:
    if is_valid_tree(tree, field_rules, max_depth=config.max_total_depth, max_nodes=config.max_nodes,
                     max_slot_depth=config.max_slot_depth):
        return tree
    try:
        mode = tree.mode if tree.mode in MODE_REGISTRY else _choice_mode(field_rules, rng, mode_probabilities=mode_probabilities)
        mode_spec = MODE_REGISTRY[mode]
        combiner = tree.combiner if tree.combiner in mode_spec.allowed_combiners else mode_spec.default_combiner
        slots: dict[str, TreeExpr] = {}
        for slot_name, slot_spec in mode_spec.slots.items():
            existing = tree.slots.get(slot_name)
            if existing is not None:
                if len(validate_slot_node(existing, slot_name, slot_spec, field_rules)) == 0:
                    slots[slot_name] = existing
                    continue
            if slot_spec.required or rng.random() < config.optional_slot_probability:
                if fields_for_slot(field_rules, slot_name, slot_spec):
                    slots[slot_name] = sample_slot_tree(field_rules, slot_name, slot_spec, rng, config=config)
        repaired = TemplateTree(
            mode=mode,
            combiner=combiner,
            slots=slots,
            conditions=tree.conditions[: mode_spec.max_conditions],
            direction_policy=tree.direction_policy if tree.direction_policy in DIRECTION_POLICIES else "fixed",
        )
        if is_valid_tree(repaired, field_rules, max_depth=config.max_total_depth, max_nodes=config.max_nodes,
                         max_slot_depth=config.max_slot_depth):
            return repaired
    except Exception:
        pass
    return random_typed_tree(field_rules, rng, config=config, mode_probabilities=mode_probabilities)


def mutate_typed_tree(
    tree: TemplateTree,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: TypedTreeSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> TemplateTree:
    """Mutate one semantic unit and repair the tree."""

    config = config or TypedTreeSamplerConfig()
    parameter = _choice(rng, ("mode", "combiner", "slot_tree", "condition", "direction_policy"))
    if parameter == "mode":
        new_mode = _choice_mode(field_rules, rng, mode_probabilities=mode_probabilities)
        mode_spec = MODE_REGISTRY[new_mode]
        mutated = TemplateTree(
            mode=new_mode,
            combiner=_choice(rng, mode_spec.allowed_combiners),
            slots={},
            conditions=(),
            direction_policy=tree.direction_policy,
        )
    elif parameter == "combiner":
        mode_spec = MODE_REGISTRY[tree.mode]
        mutated = replace(tree, combiner=_choice(rng, mode_spec.allowed_combiners))
    elif parameter == "slot_tree":
        mode_spec = MODE_REGISTRY[tree.mode]
        slot_names = tuple(mode_spec.slots)
        slot_name = _choice(rng, slot_names)
        slots = dict(tree.slots)
        slots[slot_name] = sample_slot_tree(field_rules, slot_name, mode_spec.slots[slot_name], rng, config=config)
        mutated = replace(tree, slots=slots)
    elif parameter == "condition":
        mode_spec = MODE_REGISTRY[tree.mode]
        if tree.conditions and rng.random() < 0.5:
            kept = list(tree.conditions)
            del kept[int(rng.integers(len(kept)))]
            mutated = replace(tree, conditions=tuple(kept))
        else:
            mutated = replace(tree, conditions=_sample_conditions(mode_spec, field_rules, rng, replace(config, condition_probability=1.0)))
    elif parameter == "direction_policy":
        mutated = replace(tree, direction_policy=_choice(rng, config.direction_policies))
    else:
        raise AssertionError(parameter)
    return _repair_tree(mutated, field_rules, rng, config=config, mode_probabilities=mode_probabilities)


def crossover_typed_trees(
    left: TemplateTree,
    right: TemplateTree,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: TypedTreeSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> tuple[TemplateTree, TemplateTree]:
    """Semantic crossover: swap compatible slot subtrees under behavior templates."""

    config = config or TypedTreeSamplerConfig()

    def make_child(primary: TemplateTree, secondary: TemplateTree) -> TemplateTree:
        mode = primary.mode if rng.random() < 0.75 else secondary.mode
        if mode not in MODE_REGISTRY:
            mode = _choice_mode(field_rules, rng, mode_probabilities=mode_probabilities)
        mode_spec = MODE_REGISTRY[mode]
        slots: dict[str, TreeExpr] = {}
        for slot_name, slot_spec in mode_spec.slots.items():
            candidates = []
            if slot_name in primary.slots:
                candidates.append(primary.slots[slot_name])
            if slot_name in secondary.slots:
                candidates.append(secondary.slots[slot_name])
            if candidates:
                slots[slot_name] = candidates[int(rng.integers(len(candidates)))]
            elif slot_spec.required:
                slots[slot_name] = sample_slot_tree(field_rules, slot_name, slot_spec, rng, config=config)
            elif rng.random() < config.optional_slot_probability and fields_for_slot(field_rules, slot_name, slot_spec):
                slots[slot_name] = sample_slot_tree(field_rules, slot_name, slot_spec, rng, config=config)
        combiner_pool = [item for item in (primary.combiner, secondary.combiner) if item in mode_spec.allowed_combiners]
        combiner = _choice(rng, tuple(combiner_pool)) if combiner_pool else mode_spec.default_combiner
        return TemplateTree(
            mode=mode,
            combiner=combiner,
            slots=slots,
            conditions=primary.conditions if rng.random() < 0.5 else secondary.conditions,
            direction_policy=primary.direction_policy if rng.random() < 0.5 else secondary.direction_policy,
            version=max(primary.version, secondary.version),
        )

    child_a = _repair_tree(make_child(left, right), field_rules, rng, config=config, mode_probabilities=mode_probabilities)
    child_b = _repair_tree(make_child(right, left), field_rules, rng, config=config, mode_probabilities=mode_probabilities)
    return child_a, child_b


def random_typed_population(
    field_rules: Mapping[str, BehaviorFieldRule],
    size: int,
    rng: np.random.Generator,
    *,
    config: TypedTreeSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> list[TemplateTree]:
    if size <= 0:
        return []
    config = config or TypedTreeSamplerConfig()
    return [
        random_typed_tree(field_rules, rng, config=config, mode_probabilities=mode_probabilities)
        for _ in range(size)
    ]
