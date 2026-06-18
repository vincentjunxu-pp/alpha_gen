from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .gene import (
    CONDITION_OP_CHOICES,
    DIRECTION_POLICIES,
    MODE_REGISTRY,
    BehaviorFieldRule,
    BehaviorGene,
    ConditionGene,
    SlotGene,
    condition_fields_for_mode,
    fields_for_slot,
    get_mode_spec,
    is_valid_gene,
    validate_gene,
)


@dataclass(frozen=True)
class BehaviorSamplerConfig:
    """Sampling and repair knobs for behavior genes."""

    optional_slot_probability: float = 0.6
    condition_probability: float = 0.45
    condition_thresholds: tuple[float, ...] = (0.55, 0.60, 0.70, 0.80)
    direction_policies: tuple[str, ...] = ("train_ic", "fixed")


def _choice(rng: np.random.Generator, values: list[str] | tuple[str, ...]) -> str:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return str(values[int(rng.integers(len(values)))])


def _mode_weights(
    mode_probabilities: Mapping[str, float] | None,
    viable_modes: list[str],
) -> np.ndarray:
    if mode_probabilities is None:
        return np.full(len(viable_modes), 1.0 / len(viable_modes), dtype=float)

    unknown = sorted(set(mode_probabilities) - set(MODE_REGISTRY))
    if unknown:
        raise ValueError(f"mode_probabilities contains unknown modes: {unknown}")
    weights = np.array([float(mode_probabilities.get(mode, 0.0)) for mode in viable_modes], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("mode_probabilities must contain finite non-negative weights")
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError(f"mode_probabilities assign no positive weight to viable modes: {viable_modes}")
    return weights / total


def _allowed_unary_ops(rule: BehaviorFieldRule) -> list[str]:
    return [op for op in rule.allowed_unary_ops if op]


def _mode_has_candidates(mode_spec: ModeSpec, field_rules: Mapping[str, BehaviorFieldRule]) -> bool:
    for slot_name, slot_spec in mode_spec.slots.items():
        if slot_spec.required and not fields_for_slot(field_rules, slot_name, slot_spec):
            return False
    return bool(mode_spec.allowed_combiners)


def viable_modes(field_rules: Mapping[str, BehaviorFieldRule]) -> list[str]:
    """Return modes whose required slots have at least one candidate field."""

    return [mode for mode, spec in MODE_REGISTRY.items() if _mode_has_candidates(spec, field_rules)]


def _choice_mode(
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    mode_probabilities: Mapping[str, float] | None = None,
) -> str:
    modes = viable_modes(field_rules)
    if not modes:
        raise RuntimeError("no behavior mode has legal candidates under current metadata")
    weights = _mode_weights(mode_probabilities, modes)
    return str(rng.choice(modes, p=weights))


def _sample_slot(
    field_rules: Mapping[str, BehaviorFieldRule],
    slot_name: str,
    slot_spec: SlotSpec,
    rng: np.random.Generator,
    *,
    excluded_fields: set[str] | None = None,
) -> SlotGene:
    candidates = fields_for_slot(field_rules, slot_name, slot_spec)
    if excluded_fields:
        candidates = [field for field in candidates if field not in excluded_fields]
    if not candidates:
        raise RuntimeError(f"slot {slot_name!r} has no legal field candidates")
    field = _choice(rng, candidates)
    return SlotGene(field=field, unary_op=_choice(rng, _allowed_unary_ops(field_rules[field])))


def _slot_gene_is_valid(
    slot: SlotGene,
    slot_name: str,
    slot_spec: SlotSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> bool:
    if slot.field not in field_rules:
        return False
    rule = field_rules[slot.field]
    if slot.unary_op not in rule.allowed_unary_ops:
        return False
    return slot.field in fields_for_slot(field_rules, slot_name, slot_spec)


def _sample_slots(
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    config: BehaviorSamplerConfig,
) -> dict[str, SlotGene]:
    slots: dict[str, SlotGene] = {}
    used_fields: set[str] = set()
    for slot_name, slot_spec in mode_spec.slots.items():
        if slot_spec.required or rng.random() < config.optional_slot_probability:
            candidates = [
                field
                for field in fields_for_slot(field_rules, slot_name, slot_spec)
                if field not in used_fields
            ]
            if candidates:
                slot = _sample_slot(
                    field_rules,
                    slot_name,
                    slot_spec,
                    rng,
                    excluded_fields=used_fields,
                )
                slots[slot_name] = slot
                used_fields.add(slot.field)
    return slots


def _sample_conditions(
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    config: BehaviorSamplerConfig,
) -> tuple[ConditionGene, ...]:
    if mode_spec.max_conditions <= 0 or rng.random() >= config.condition_probability:
        return ()
    candidates = condition_fields_for_mode(field_rules, mode_spec)
    if not candidates:
        return ()
    max_count = min(mode_spec.max_conditions, len(candidates))
    n_conditions = int(rng.integers(1, max_count + 1))
    chosen = list(rng.choice(candidates, size=n_conditions, replace=False))
    conditions: list[ConditionGene] = []
    for field in chosen:
        rule = field_rules[str(field)]
        op = _choice(rng, CONDITION_OP_CHOICES)
        threshold = 0.0
        if op in {"top_quantile", "bottom_quantile"}:
            threshold = float(config.condition_thresholds[int(rng.integers(len(config.condition_thresholds)))])
        conditions.append(
            ConditionGene(
                field=str(field),
                unary_op=_choice(rng, _allowed_unary_ops(rule)),
                condition_op=op,
                threshold=threshold,
            )
        )
    return tuple(conditions)


def random_gene(
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: BehaviorSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> BehaviorGene:
    """Sample one legal behavior gene."""

    config = config or BehaviorSamplerConfig()
    mode = _choice_mode(field_rules, rng, mode_probabilities=mode_probabilities)
    mode_spec = get_mode_spec(mode)
    gene = BehaviorGene(
        mode=mode,
        combiner=_choice(rng, mode_spec.allowed_combiners),
        slots=_sample_slots(mode_spec, field_rules, rng, config),
        conditions=_sample_conditions(mode_spec, field_rules, rng, config),
        direction_policy=_choice(rng, config.direction_policies),
    )
    repaired = repair_gene(gene, field_rules, rng, config=config, mode_probabilities=mode_probabilities)
    if is_valid_gene(repaired, field_rules):
        return repaired
    raise RuntimeError("failed to sample a legal behavior gene: " + "; ".join(validate_gene(repaired, field_rules)))


def _repair_slot(
    slot: SlotGene | None,
    slot_name: str,
    slot_spec: SlotSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    excluded_fields: set[str] | None = None,
) -> SlotGene | None:
    if (
        slot is not None
        and (not excluded_fields or slot.field not in excluded_fields)
        and _slot_gene_is_valid(slot, slot_name, slot_spec, field_rules)
    ):
        return slot
    candidates = [
        field
        for field in fields_for_slot(field_rules, slot_name, slot_spec)
        if not excluded_fields or field not in excluded_fields
    ]
    if not candidates:
        return None
    return _sample_slot(
        field_rules,
        slot_name,
        slot_spec,
        rng,
        excluded_fields=excluded_fields,
    )


def _condition_is_valid(
    condition: ConditionGene,
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
) -> bool:
    if condition.field not in field_rules:
        return False
    if condition.condition_op not in CONDITION_OP_CHOICES:
        return False
    rule = field_rules[condition.field]
    if condition.unary_op not in rule.allowed_unary_ops:
        return False
    allowed = set(mode_spec.allowed_condition_roles)
    if allowed and not allowed.intersection(rule.behavior_roles):
        if "state_signal" not in rule.allowed_slots and "orderbook_filter" not in rule.allowed_slots:
            return False
    if condition.condition_op in {"top_quantile", "bottom_quantile"}:
        return 0.0 < condition.threshold < 1.0
    return True


def _repair_conditions(
    conditions: tuple[ConditionGene, ...],
    mode_spec: ModeSpec,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    config: BehaviorSamplerConfig,
) -> tuple[ConditionGene, ...]:
    kept = [condition for condition in conditions if _condition_is_valid(condition, mode_spec, field_rules)]
    kept = kept[: mode_spec.max_conditions]
    if kept or rng.random() >= config.condition_probability:
        return tuple(kept)
    sampled = _sample_conditions(mode_spec, field_rules, rng, config)
    return sampled[: mode_spec.max_conditions]


def repair_gene(
    gene: BehaviorGene,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: BehaviorSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> BehaviorGene:
    """Repair a possibly illegal behavior gene after crossover or mutation."""

    config = config or BehaviorSamplerConfig()
    mode = gene.mode if gene.mode in MODE_REGISTRY and _mode_has_candidates(MODE_REGISTRY[gene.mode], field_rules) else _choice_mode(
        field_rules,
        rng,
        mode_probabilities=mode_probabilities,
    )
    mode_spec = get_mode_spec(mode)
    combiner = gene.combiner if gene.combiner in mode_spec.allowed_combiners else mode_spec.default_combiner

    slots: dict[str, SlotGene] = {}
    used_fields: set[str] = set()
    for slot_name, slot_spec in mode_spec.slots.items():
        existing = gene.slots.get(slot_name)
        repaired = _repair_slot(
            existing,
            slot_name,
            slot_spec,
            field_rules,
            rng,
            excluded_fields=used_fields,
        )
        if repaired is not None and (slot_spec.required or existing is not None or rng.random() < config.optional_slot_probability):
            slots[slot_name] = repaired
            used_fields.add(repaired.field)

    conditions = _repair_conditions(gene.conditions, mode_spec, field_rules, rng, config)
    direction_policy = gene.direction_policy if gene.direction_policy in config.direction_policies else "train_ic"
    if direction_policy not in DIRECTION_POLICIES:
        direction_policy = "train_ic"
    if direction_policy == "regime_switch" and mode_spec.direction_policy != "regime_switch":
        direction_policy = "train_ic"

    repaired_gene = BehaviorGene(
        mode=mode,
        combiner=combiner,
        slots=slots,
        conditions=conditions,
        direction_policy=direction_policy,
        version=gene.version,
    )
    if is_valid_gene(repaired_gene, field_rules):
        return repaired_gene

    # If optional retained choices still caused an edge case, sample fresh in the
    # selected mode so repair remains total for GA operations.
    fresh_slots: dict[str, SlotGene] = {}
    used_fields = set()
    for slot_name, slot_spec in mode_spec.slots.items():
        if not slot_spec.required:
            continue
        slot = _sample_slot(
            field_rules,
            slot_name,
            slot_spec,
            rng,
            excluded_fields=used_fields,
        )
        fresh_slots[slot_name] = slot
        used_fields.add(slot.field)

    fresh = BehaviorGene(
        mode=mode,
        combiner=mode_spec.default_combiner,
        slots=fresh_slots,
        conditions=(),
        direction_policy="train_ic",
    )
    if is_valid_gene(fresh, field_rules):
        return fresh
    raise RuntimeError("failed to repair behavior gene: " + "; ".join(validate_gene(repaired_gene, field_rules)))


def mutate_one_parameter(
    gene: BehaviorGene,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: BehaviorSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> BehaviorGene:
    """Mutate one semantic parameter and repair the result."""

    config = config or BehaviorSamplerConfig()
    parameter = _choice(
        rng,
        (
            "mode",
            "combiner",
            "slot_field",
            "slot_unary",
            "condition",
            "direction_policy",
        ),
    )

    if parameter == "mode":
        mutated = replace(gene, mode=_choice_mode(field_rules, rng, mode_probabilities=mode_probabilities))
    elif parameter == "combiner":
        mode_spec = get_mode_spec(gene.mode) if gene.mode in MODE_REGISTRY else get_mode_spec(_choice_mode(field_rules, rng))
        mutated = replace(gene, combiner=_choice(rng, mode_spec.allowed_combiners))
    elif parameter in {"slot_field", "slot_unary"}:
        if not gene.slots:
            mutated = gene
        else:
            slot_name = _choice(rng, tuple(gene.slots))
            mode_spec = get_mode_spec(gene.mode)
            slot_spec = mode_spec.slots[slot_name]
            old_slot = gene.slots[slot_name]
            if parameter == "slot_field":
                excluded_fields = {
                    slot.field
                    for name, slot in gene.slots.items()
                    if name != slot_name
                }
                new_slot = _sample_slot(
                    field_rules,
                    slot_name,
                    slot_spec,
                    rng,
                    excluded_fields=excluded_fields,
                )
            else:
                rule = field_rules.get(old_slot.field)
                if rule is None:
                    excluded_fields = {
                        slot.field
                        for name, slot in gene.slots.items()
                        if name != slot_name
                    }
                    new_slot = _sample_slot(
                        field_rules,
                        slot_name,
                        slot_spec,
                        rng,
                        excluded_fields=excluded_fields,
                    )
                else:
                    new_slot = replace(old_slot, unary_op=_choice(rng, _allowed_unary_ops(rule)))
            slots = dict(gene.slots)
            slots[slot_name] = new_slot
            mutated = replace(gene, slots=slots)
    elif parameter == "condition":
        mode_spec = get_mode_spec(gene.mode) if gene.mode in MODE_REGISTRY else get_mode_spec(_choice_mode(field_rules, rng))
        if gene.conditions and rng.random() < 0.5:
            kept = list(gene.conditions)
            del kept[int(rng.integers(len(kept)))]
            mutated = replace(gene, conditions=tuple(kept))
        else:
            sampled = _sample_conditions(mode_spec, field_rules, rng, replace(config, condition_probability=1.0))
            mutated = replace(gene, conditions=sampled)
    elif parameter == "direction_policy":
        mutated = replace(gene, direction_policy=_choice(rng, config.direction_policies))
    else:
        raise AssertionError(f"unhandled mutation parameter: {parameter}")

    return repair_gene(mutated, field_rules, rng, config=config, mode_probabilities=mode_probabilities)


def crossover_genes(
    left: BehaviorGene,
    right: BehaviorGene,
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: BehaviorSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> tuple[BehaviorGene, BehaviorGene]:
    """Slot-aware crossover for behavior genes."""

    config = config or BehaviorSamplerConfig()

    def make_child(primary: BehaviorGene, secondary: BehaviorGene) -> BehaviorGene:
        mode = primary.mode if rng.random() < 0.75 else secondary.mode
        if mode not in MODE_REGISTRY:
            mode = _choice_mode(field_rules, rng, mode_probabilities=mode_probabilities)
        mode_spec = get_mode_spec(mode)

        slots: dict[str, SlotGene] = {}
        for slot_name in mode_spec.slots:
            candidates = []
            if slot_name in primary.slots:
                candidates.append(primary.slots[slot_name])
            if slot_name in secondary.slots:
                candidates.append(secondary.slots[slot_name])
            if candidates:
                slots[slot_name] = candidates[int(rng.integers(len(candidates)))]

        combiner_pool = [combiner for combiner in (primary.combiner, secondary.combiner) if combiner in mode_spec.allowed_combiners]
        combiner = _choice(rng, tuple(combiner_pool)) if combiner_pool else mode_spec.default_combiner

        conditions = primary.conditions if rng.random() < 0.5 else secondary.conditions
        direction_policy = primary.direction_policy if rng.random() < 0.5 else secondary.direction_policy
        return BehaviorGene(
            mode=mode,
            combiner=combiner,
            slots=slots,
            conditions=conditions,
            direction_policy=direction_policy,
            version=max(primary.version, secondary.version),
        )

    child_a = repair_gene(
        make_child(left, right),
        field_rules,
        rng,
        config=config,
        mode_probabilities=mode_probabilities,
    )
    child_b = repair_gene(
        make_child(right, left),
        field_rules,
        rng,
        config=config,
        mode_probabilities=mode_probabilities,
    )
    return child_a, child_b


def random_population(
    field_rules: Mapping[str, BehaviorFieldRule],
    size: int,
    rng: np.random.Generator,
    *,
    config: BehaviorSamplerConfig | None = None,
    mode_probabilities: Mapping[str, float] | None = None,
) -> list[BehaviorGene]:
    """Sample a population of legal behavior genes."""

    if size <= 0:
        return []
    return [
        random_gene(
            field_rules,
            rng,
            config=config,
            mode_probabilities=mode_probabilities,
        )
        for _ in range(size)
    ]
