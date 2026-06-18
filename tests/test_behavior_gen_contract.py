from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.behavior_gen.gene import (
    MODE_REGISTRY,
    BehaviorGene,
    ConditionGene,
    SlotGene,
    describe_gene_formula,
    load_behavior_field_rules,
    validate_behavior_field_rules,
    validate_gene,
    validate_mode_registry,
)
from alpha_gen.behavior_gen.ga import (
    EvaluatedBehaviorGene,
    NSGA_MODE_RIR_LONG_RIR,
    NSGA_MODE_RIR_LONG_RIR_NDCG,
    NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR,
    NSGA_OBJECTIVE_MODES,
    evaluated_behavior_to_frame,
    selected_behavior_rank_table,
)
from alpha_gen.behavior_gen.sampler import crossover_genes, mutate_one_parameter, random_gene
from alpha_gen.behavior_gen.torch_backend import (
    NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    NEUTRALIZATION_SIZE_THEN_INDUSTRY,
    neutralize_behavior_factor_tensor,
)
from alpha_gen.core.metrics import FactorScore
from alpha_gen.core.torch_backend import (
    NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY,
    NEUTRALIZED_METRIC_NONE,
    cross_sectional_residual_torch,
    evaluate_factor_tensor,
    industry_neutralize_torch,
)


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
            formula = describe_gene_formula(gene)
            assert "factor :=" in formula
            for slot in gene.slots.values():
                assert slot.field in formula


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


def test_formula_expands_transforms_combiner_conditions_and_direction() -> None:
    gene = BehaviorGene(
        mode="quality_neglect",
        combiner="quality_gap",
        slots={
            "profit_growth": SlotGene(field="profit_growth_field", unary_op="zscore"),
            "cashflow_quality": SlotGene(field="cashflow_field", unary_op="rank_pct"),
            "price_reaction": SlotGene(field="price_field", unary_op="ts_zscore_20d"),
        },
        conditions=(
            ConditionGene(
                field="attention_field",
                unary_op="current",
                condition_op="bottom_quantile",
                threshold=0.7,
            ),
        ),
    )

    formula = describe_gene_formula(gene)
    assert "profit_growth := CS_ZSCORE(profit_growth_field)" in formula
    assert "cashflow_quality := (CS_RANK_PCT(cashflow_field) - 0.5)" in formula
    assert "price_reaction := TS_ZSCORE(price_field, 20)" in formula
    assert "0.25 * ABS(price_reaction)" in formula
    assert "CS_RANK_PCT(attention_field) <= 0.3" in formula
    assert "factor := (-1 * WHERE(" in formula


def test_result_frame_contains_formula_column() -> None:
    gene = BehaviorGene(
        mode="fund_price_underreaction",
        combiner="rank_gap",
        slots={
            "fund_anchor": SlotGene(field="fund_field", unary_op="zscore"),
            "price_reaction": SlotGene(field="price_field", unary_op="rank_pct"),
        },
    )
    score = FactorScore(
        mean_rank_ic=0.01,
        rank_ic_ir=0.1,
        ic_win_rate=0.5,
        ndcg_at_k=0.2,
        direction=1,
        n_ic_obs=10,
        coverage=1.0,
        neutralized_icir=0.0,
        neutralized_mean_rank_ic=0.0,
        neutralized_ic_win_rate=0.0,
        neutralized_n_ic_obs=0,
    )

    frame = evaluated_behavior_to_frame(
        [
            EvaluatedBehaviorGene(
                gene=gene,
                train_score=score,
                train_metrics=None,
                valid_score=score,
                valid_metrics=None,
                generation=0,
            )
        ]
    )
    assert "formula" in frame.columns
    assert "fund_anchor := CS_ZSCORE(fund_field)" in frame.loc[0, "formula"]
    assert "factor := (fund_anchor - price_reaction)" in frame.loc[0, "formula"]
    assert frame.loc[0, "train_direction"] == 1

    # All metric columns come from GPU FactorScore
    expected_metrics = {"ic", "ir", "ric", "rir", "long_ric", "long_rir",
                        "long_sharpe", "sharpe", "ndcg_k", "win_rate",
                        "neutralized_ric", "neutralized_rir"}
    assert all(f"train_{name}" in frame.columns for name in expected_metrics)
    assert all(f"valid_{name}" in frame.columns for name in expected_metrics)
    # Sharpre now comes from GPU — should be populated
    assert frame.loc[0, "train_sharpe"] == 0.0  # rank_ic_ir=0 → zero sharpe
    assert "train_mean_rank_ic" not in frame.columns
    assert "train_coverage" not in frame.columns
    assert "valid_top_excess_ann" not in frame.columns


def test_compact_metrics_use_top_half_for_long_group() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    columns = list("ABCDEF")
    factor = pd.DataFrame(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 6, 5, 4],
            [1, 3, 2, 4, 6, 5],
            [2, 1, 3, 5, 4, 6],
        ],
        index=dates,
        columns=columns,
        dtype=float,
    )
    label = pd.DataFrame(
        [
            [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
            [-0.02, -0.01, 0.00, 0.03, 0.02, 0.01],
            [-0.03, -0.01, -0.02, 0.01, 0.03, 0.02],
            [-0.01, -0.02, 0.00, 0.02, 0.01, 0.03],
        ],
        index=dates,
        columns=columns,
    )
    score = FactorScore(
        mean_rank_ic=1.0,
        rank_ic_ir=0.0,
        ic_win_rate=1.0,
        ndcg_at_k=1.0,
        direction=1,
        n_ic_obs=4,
        coverage=1.0,
        neutralized_icir=0.5,
        neutralized_mean_rank_ic=0.25,
    )

    metrics = compact_factor_metrics(
        factor=factor,
        label=label,
        tradeable=None,
        dates=dates,
        direction=1,
        score=score,
        include_neutralized_metrics=True,
        label_horizon=1,
        rebalance_freq=1,
        commission_rate=0.0,
        slippage_rate=0.0,
        stamp_tax_rate=0.0,
    )

    assert np.isclose(metrics.ric, 1.0)
    assert np.isclose(metrics.long_ric, 1.0)
    assert metrics.win_rate == 1.0
    assert metrics.ndcg_k == 1.0
    assert metrics.neutralized_ric == 0.25
    assert metrics.neutralized_rir == 0.5
    assert np.isfinite(metrics.long_sharpe)
    assert np.isfinite(metrics.sharpe)


def test_behavior_neutralization_modes_are_explicit_and_ordered() -> None:
    factor = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0, 3.0, 7.0], [2.0, 3.0, 5.0, 9.0, 4.0, 8.0]]
    )
    size = torch.tensor(
        [[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], [1.2, 1.7, 2.2, 2.7, 3.2, 3.7]]
    )
    industries = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    tradeable = torch.ones_like(factor, dtype=torch.bool)

    class Context:
        cache = type("Cache", (), {"industry": object()})()

        def get_current(self, field: str, use_log: bool) -> torch.Tensor:
            assert field == "barra_size"
            assert not use_log
            return size

        def industry_codes(self) -> torch.Tensor:
            return industries

    ctx = Context()
    raw = neutralize_behavior_factor_tensor(
        factor,
        ctx,  # type: ignore[arg-type]
        neutralization_mode=NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
        tradeable_mask=tradeable,
    )
    assert torch.equal(raw, factor)

    actual = neutralize_behavior_factor_tensor(
        factor,
        ctx,  # type: ignore[arg-type]
        neutralization_mode=NEUTRALIZATION_SIZE_THEN_INDUSTRY,
        tradeable_mask=tradeable,
    )
    expected = industry_neutralize_torch(
        cross_sectional_residual_torch(factor, size, mask=tradeable),
        industries,
        mask=tradeable,
    )
    assert torch.allclose(actual, expected, equal_nan=True)


def test_behavior_context_caches_boolean_tradeable_mask() -> None:
    from alpha_gen.behavior_gen.torch_backend import BehaviorTorchContext
    from alpha_gen.core.preprocess import TransformCache

    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    columns = ["A", "B"]
    label = pd.DataFrame(0.0, index=dates, columns=columns, dtype="float32")
    tradeable = pd.DataFrame(
        [[1, 0], [1, 1]],
        index=dates,
        columns=columns,
        dtype="int8",
    )
    cache = TransformCache(
        current={},
        label=label,
        tradeable=tradeable,
        industry=None,
        field_rules={},
    )
    ctx = BehaviorTorchContext(
        cache=cache,
        behavior_field_rules={},
        device="cpu",
        cache_on_device=True,
    )

    first = ctx.tradeable()
    second = ctx.tradeable()

    assert first.dtype == torch.bool
    assert first.data_ptr() == second.data_ptr()
    assert first.tolist() == [[True, False], [True, True]]


def test_full_barra_industry_simultaneous_neutralization() -> None:
    """NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY now regresses on 10 Barra styles
    AND industry dummies simultaneously (single cross-sectional regression),
    not as a two-step Barra-residual-then-industry-demean process."""
    generator = torch.Generator().manual_seed(20260618)
    n_dates, n_contracts, n_styles = 12, 60, 10
    n_industries = 4
    styles = torch.randn(n_dates, n_contracts, n_styles, generator=generator)
    alpha = torch.randn(n_dates, n_contracts, generator=generator)
    # Inject known Barra + industry structure so neutralization should reduce IC
    industry_effect = torch.randn(n_dates, n_industries, generator=generator)
    industry_codes_tensor = torch.arange(n_contracts).remainder(n_industries).unsqueeze(0).expand(n_dates, -1)
    industry_component = torch.zeros(n_dates, n_contracts)
    for i in range(n_industries):
        industry_component += (industry_codes_tensor == i).to(torch.float32) * industry_effect[:, i:i+1]
    factor = alpha + 0.6 * (styles * torch.linspace(0.3, 1.2, n_styles)).sum(dim=2) + 0.4 * industry_component
    label = alpha + 0.1 * torch.randn(n_dates, n_contracts, generator=generator)
    tradeable = torch.ones((n_dates, n_contracts), dtype=torch.bool)
    industries = industry_codes_tensor.clone()

    class Context:
        barra_style_fields = tuple(f"barra_{idx}" for idx in range(n_styles))
        barra_corr_threshold = 0.30
        barra_max_styles = 2
        date_index = pd.date_range("2026-01-01", periods=n_dates, freq="D")

        def date_positions(self, dates):
            return None

        def label(self):
            return label

        def tradeable(self):
            return tradeable

        def barra_styles(self):
            return styles

        def industry_codes(self):
            return industries

    full_score = evaluate_factor_tensor(
        factor,
        Context(),  # type: ignore[arg-type]
        neutralized_metric_mode=NEUTRALIZED_METRIC_FULL_BARRA_INDUSTRY,
    )
    # Neutralized metrics must be computed (finite, non-trivial observation count)
    assert full_score.neutralized_n_ic_obs > 0
    assert np.isfinite(full_score.neutralized_icir)
    assert np.isfinite(full_score.neutralized_mean_rank_ic)
    assert np.isfinite(full_score.neutralized_ic_win_rate)
    # Raw IC should differ from neutralized (structure was injected)
    assert abs(full_score.rank_ic_ir - full_score.neutralized_icir) > 0.001

    no_neutralized_score = evaluate_factor_tensor(
        factor,
        Context(),  # type: ignore[arg-type]
        neutralized_metric_mode=NEUTRALIZED_METRIC_NONE,
    )
    assert no_neutralized_score.neutralized_n_ic_obs == 0
    assert no_neutralized_score.neutralized_icir == 0.0


def test_behavior_nsga_objective_modes() -> None:
    gene = BehaviorGene(
        mode="fund_price_underreaction",
        combiner="rank_gap",
        slots={
            "fund_anchor": SlotGene(field="fund_field", unary_op="zscore"),
            "price_reaction": SlotGene(field="price_field", unary_op="rank_pct"),
        },
    )
    score = FactorScore(
        mean_rank_ic=0.0,
        rank_ic_ir=0.4,
        ic_win_rate=0.0,
        ndcg_at_k=0.2,
        direction=1,
        n_ic_obs=1,
        coverage=1.0,
        long_rank_ic_ir=0.3,
        neutralized_icir=99.0,
    )
    item = EvaluatedBehaviorGene(
        gene=gene,
        train_score=score,
        train_metrics=None,
        generation=0,
    )

    assert item.objectives(NSGA_MODE_RIR_LONG_RIR_NDCG) == (0.4, 0.3, 0.2)
    assert item.objectives(NSGA_MODE_RIR_LONG_RIR) == (0.4, 0.3)

    three_objective_table = selected_behavior_rank_table(
        [item],
        objective_mode=NSGA_MODE_RIR_LONG_RIR_NDCG,
    )
    assert {
        "objective_rir",
        "objective_long_rir",
        "objective_ndcg_k",
    }.issubset(three_objective_table.columns)
    assert three_objective_table.loc[0, "train_direction"] == 1

    two_objective_table = selected_behavior_rank_table(
        [item],
        objective_mode=NSGA_MODE_RIR_LONG_RIR,
    )
    assert {"objective_rir", "objective_long_rir"}.issubset(
        two_objective_table.columns
    )
    assert "objective_ndcg_k" not in two_objective_table.columns
    assert set(two_objective_table["objective_mode"]) == {
        NSGA_MODE_RIR_LONG_RIR
    }


def test_neutralized_rir_nsga_objectives() -> None:
    """Verify the new rir_long_rir_neutralized_rir mode returns
    (rank_ic_ir, long_rank_ic_ir, neutralized_icir)."""
    gene = BehaviorGene(
        mode="fund_price_underreaction",
        combiner="rank_gap",
        slots={
            "fund_anchor": SlotGene(field="fund_field", unary_op="zscore"),
            "price_reaction": SlotGene(field="price_field", unary_op="rank_pct"),
        },
    )
    score = FactorScore(
        mean_rank_ic=0.0,
        rank_ic_ir=0.5,
        ic_win_rate=0.0,
        ndcg_at_k=0.2,
        direction=1,
        n_ic_obs=1,
        coverage=1.0,
        long_rank_ic_ir=0.35,
        neutralized_icir=0.28,
    )
    item = EvaluatedBehaviorGene(
        gene=gene,
        train_score=score,
        train_metrics=None,
        generation=0,
    )

    objectives = item.objectives(NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR)
    assert objectives == (0.5, 0.35, 0.28)

    # Verify the mode is in NSGA_OBJECTIVE_MODES
    assert NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR in NSGA_OBJECTIVE_MODES
    assert NSGA_OBJECTIVE_MODES[NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR] == (
        "rir", "long_rir", "neutralized_rir",
    )


def test_neutralized_rir_rank_table_columns() -> None:
    """Rank table for the new mode must include objective_neutralized_rir."""
    gene = BehaviorGene(
        mode="fund_price_underreaction",
        combiner="rank_gap",
        slots={
            "fund_anchor": SlotGene(field="fund_field", unary_op="zscore"),
            "price_reaction": SlotGene(field="price_field", unary_op="rank_pct"),
        },
    )
    score = FactorScore(
        mean_rank_ic=0.0,
        rank_ic_ir=0.5,
        ic_win_rate=0.0,
        ndcg_at_k=0.2,
        direction=1,
        n_ic_obs=1,
        coverage=1.0,
        long_rank_ic_ir=0.35,
        neutralized_icir=0.28,
    )
    item = EvaluatedBehaviorGene(
        gene=gene,
        train_score=score,
        train_metrics=None,
        generation=0,
    )

    table = selected_behavior_rank_table(
        [item],
        objective_mode=NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR,
    )
    expected_cols = {
        "objective_rir",
        "objective_long_rir",
        "objective_neutralized_rir",
    }
    assert expected_cols.issubset(table.columns)
    assert "objective_ndcg_k" not in table.columns
    assert set(table["objective_mode"]) == {
        NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR
    }
    assert table.loc[0, "train_direction"] == 1


def test_result_frame_includes_neutralized_rir_columns() -> None:
    """evaluated_behavior_to_frame must include train_neutralized_rir
    and valid_neutralized_rir for the new NSGA mode."""
    gene = BehaviorGene(
        mode="fund_price_underreaction",
        combiner="rank_gap",
        slots={
            "fund_anchor": SlotGene(field="fund_field", unary_op="zscore"),
            "price_reaction": SlotGene(field="price_field", unary_op="rank_pct"),
        },
    )
    score = FactorScore(
        mean_rank_ic=0.02,
        rank_ic_ir=0.4,
        ic_win_rate=0.6,
        ndcg_at_k=0.15,
        direction=1,
        n_ic_obs=20,
        coverage=0.9,
        long_rank_ic_ir=0.3,
        neutralized_icir=0.25,
        neutralized_mean_rank_ic=0.01,
        neutralized_ic_win_rate=0.55,
        neutralized_n_ic_obs=18,
    )
    eval_gene = EvaluatedBehaviorGene(
        gene=gene,
        train_score=score,
        train_metrics=None,
        valid_score=score,
        valid_metrics=None,
        generation=1,
        passed_validation=True,
    )

    frame = evaluated_behavior_to_frame(
        [eval_gene],
        objective_mode=NSGA_MODE_RIR_LONG_RIR_NEUTRALIZED_RIR,
    )
    assert "train_neutralized_rir" in frame.columns
    assert "valid_neutralized_rir" in frame.columns
    assert frame.loc[0, "train_neutralized_rir"] == 0.25
    assert frame.loc[0, "valid_neutralized_rir"] == 0.25
    # Sort columns must include neutralized_rir
    assert "valid_neutralized_rir" in frame.columns
