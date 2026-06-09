from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.behavior_gen.gene import (
    MODE_REGISTRY,
    BehaviorGene,
    SlotGene,
    load_behavior_field_rules,
    validate_behavior_field_rules,
    validate_gene,
    validate_mode_registry,
)
from alpha_gen.behavior_gen.sampler import crossover_genes, mutate_one_parameter, random_gene


META_PATH = (
    WORKSPACE_ROOT
    / "alpha_gen"
    / "data"
    / "metadata"
    / "production"
    / "real_behavior_metadata.json"
)


def _assert_unique_slot_fields(gene: BehaviorGene) -> None:
    fields = [slot.field for slot in gene.slots.values()]
    assert len(fields) == len(set(fields)), f"duplicate slot fields in {gene.mode}: {fields}"


def test_behavior_metadata_and_modes_are_valid() -> None:
    rules = load_behavior_field_rules(META_PATH)
    assert not validate_mode_registry()
    assert not validate_behavior_field_rules(rules)


def test_every_mode_samples_valid_unique_genes() -> None:
    rules = load_behavior_field_rules(META_PATH)
    rng = np.random.default_rng(20260609)

    for mode in MODE_REGISTRY:
        for _ in range(20):
            gene = random_gene(rules, rng, mode_probabilities={mode: 1.0})
            assert not validate_gene(gene, rules)
            _assert_unique_slot_fields(gene)


def test_duplicate_slot_field_is_rejected() -> None:
    rules = load_behavior_field_rules(META_PATH)
    gene = BehaviorGene(
        mode="anchor_momentum",
        combiner="anchor_confirm",
        slots={
            "price_anchor": SlotGene(field="MA5"),
            "price_momentum": SlotGene(field="MA5"),
            "fund_support": SlotGene(field="buy_lg_amount"),
        },
    )

    errors = validate_gene(gene, rules)
    assert any("reuses fields across slots" in error for error in errors)


def test_mutation_and_crossover_preserve_contract() -> None:
    rules = load_behavior_field_rules(META_PATH)
    rng = np.random.default_rng(20260610)
    population = [random_gene(rules, rng) for _ in range(40)]

    for gene in population:
        mutated = mutate_one_parameter(gene, rules, rng)
        assert not validate_gene(mutated, rules)
        _assert_unique_slot_fields(mutated)

    for left, right in zip(population[::2], population[1::2]):
        children = crossover_genes(left, right, rules, rng)
        for child in children:
            assert not validate_gene(child, rules)
            _assert_unique_slot_fields(child)
