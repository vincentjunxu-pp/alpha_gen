from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from alpha_gen.core.metrics import FactorScore, factor_group_pnl
from alpha_gen.core.nsga2 import nsga2_select, rank_table

from .gene import BehaviorFieldRule, BehaviorGene, describe_gene, gene_key
from .sampler import (
    BehaviorSamplerConfig,
    crossover_genes,
    mutate_one_parameter,
    random_gene,
)
from .torch_backend import (
    BehaviorTorchContext,
    calculate_behavior_factor_tensor,
    score_behavior_factor_tensor,
)


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = "", leave: bool = False):
    """Wrap an iterable with tqdm when progress display is requested."""

    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


@dataclass(frozen=True)
class BehaviorGAConfig:
    """Search parameters for behavior-finance genes.

    The behavior framework is GPU-first because every candidate factor is a
    full Datetime x Contract tensor. Set require_cuda=False only for very small
    debugging jobs.
    """

    population_size: int = 80
    generations: int = 3
    crossover_prob: float = 0.85
    mutation_prob: float = 0.25
    random_seed: int = 20260529
    ndcg_k: int | None = None
    ndcg_top_fraction: float = 0.10
    min_coverage: float = 0.30
    mode_probabilities: dict[str, float] | None = None
    sampler_config: BehaviorSamplerConfig = field(default_factory=BehaviorSamplerConfig)
    neutralize_size: bool = True
    neutralize_industry: bool = True
    size_field: str = "barra_size"
    require_cuda: bool = True
    show_progress: bool = False


@dataclass(frozen=True)
class BehaviorValidationCriteria:
    """Validation filters for a final behavior-factor population."""

    min_abs_rank_ic: float = 0.02
    min_ic_win_rate: float = 0.55
    min_top_excess_ann: float = 0.00
    min_coverage: float = 0.30


@dataclass
class EvaluatedBehaviorGene:
    """One behavior gene plus training and optional validation diagnostics."""

    gene: BehaviorGene
    train_score: FactorScore
    generation: int
    error: str = ""
    valid_score: FactorScore | None = None
    valid_top_excess_ann: float | None = None
    valid_pnl_metrics: dict[str, float | int] | None = None
    passed_validation: bool | None = None

    @property
    def objectives(self) -> tuple[float, float, float]:
        return self.train_score.objectives


@dataclass
class BehaviorSearchResult:
    """Result of one behavior-finance GA run."""

    config: BehaviorGAConfig
    final_population: list[EvaluatedBehaviorGene]
    history: list[EvaluatedBehaviorGene]


def _empty_score() -> FactorScore:
    return FactorScore(
        mean_rank_ic=0.0,
        abs_rank_ic=0.0,
        rank_ic_ir=0.0,
        ic_win_rate=0.0,
        ndcg_at_k=0.0,
        direction=1,
        n_ic_obs=0,
        coverage=0.0,
        neutralized_icir=0.0,
        neutralized_mean_rank_ic=0.0,
        neutralized_abs_rank_ic=0.0,
        neutralized_ic_win_rate=0.0,
        neutralized_n_ic_obs=0,
    )


def _n_groups_from_top_fraction(top_fraction: float) -> int:
    return max(2, int(round(1.0 / top_fraction))) if 0 < top_fraction <= 1 else 5


def _scalar_pnl_metrics(pnl_result: dict[str, object]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key, value in pnl_result.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            output[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    return output


def _score_direction(gene: BehaviorGene) -> int | None:
    # Fixed-policy genes are already oriented by their economic mode direction.
    return 1 if gene.direction_policy == "fixed" else None


def _deduplicate_genes(genes: Iterable[BehaviorGene]) -> list[BehaviorGene]:
    seen: set[tuple[object, ...]] = set()
    unique: list[BehaviorGene] = []
    for gene in genes:
        key = gene_key(gene)
        if key in seen:
            continue
        seen.add(key)
        unique.append(gene)
    return unique


def _deduplicate_evaluated(evaluated: Iterable[EvaluatedBehaviorGene]) -> list[EvaluatedBehaviorGene]:
    seen: set[tuple[object, ...]] = set()
    unique: list[EvaluatedBehaviorGene] = []
    for item in evaluated:
        key = gene_key(item.gene)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _try_random_gene(
    field_rules: Mapping[str, BehaviorFieldRule],
    rng: np.random.Generator,
    *,
    config: BehaviorSamplerConfig,
    mode_probabilities: Mapping[str, float] | None = None,
) -> BehaviorGene | None:
    try:
        return random_gene(
            field_rules,
            rng,
            config=config,
            mode_probabilities=mode_probabilities,
        )
    except Exception:
        return None


def _refill_population(
    genes: list[BehaviorGene],
    field_rules: Mapping[str, BehaviorFieldRule],
    target_size: int,
    rng: np.random.Generator,
    *,
    sampler_config: BehaviorSamplerConfig,
    mode_probabilities: Mapping[str, float] | None = None,
) -> list[BehaviorGene]:
    output = _deduplicate_genes(genes)
    attempts = 0
    max_attempts = max(1_000, target_size * 100)
    while len(output) < target_size and attempts < max_attempts:
        sampled = _try_random_gene(
            field_rules,
            rng,
            config=sampler_config,
            mode_probabilities=mode_probabilities,
        )
        if sampled is not None:
            output = _deduplicate_genes(output + [sampled])
        attempts += 1
    while len(output) < target_size:
        sampled = _try_random_gene(
            field_rules,
            rng,
            config=sampler_config,
            mode_probabilities=mode_probabilities,
        )
        if sampled is not None:
            output.append(sampled)
        elif output:
            output.append(output[int(rng.integers(len(output)))])
        else:
            raise RuntimeError("failed to sample any legal behavior gene for population refill")
    return output[:target_size]


def make_behavior_offspring(
    parents: list[BehaviorGene],
    field_rules: Mapping[str, BehaviorFieldRule],
    config: BehaviorGAConfig,
    rng: np.random.Generator,
) -> list[BehaviorGene]:
    """Create one behavior offspring generation from the current population."""

    if not parents:
        raise ValueError("parents cannot be empty")

    order = rng.permutation(len(parents))
    shuffled = [parents[i] for i in order]
    children: list[BehaviorGene] = []

    for i in range(0, len(shuffled), 2):
        left = shuffled[i]
        right = shuffled[(i + 1) % len(shuffled)]

        if rng.random() < config.crossover_prob:
            try:
                child_a, child_b = crossover_genes(
                    left,
                    right,
                    field_rules,
                    rng,
                    config=config.sampler_config,
                    mode_probabilities=config.mode_probabilities,
                )
            except Exception:
                child_a, child_b = left, right
        else:
            child_a, child_b = left, right

        if rng.random() < config.mutation_prob:
            try:
                child_a = mutate_one_parameter(
                    child_a,
                    field_rules,
                    rng,
                    config=config.sampler_config,
                    mode_probabilities=config.mode_probabilities,
                )
            except Exception:
                child_a = left
        if rng.random() < config.mutation_prob:
            try:
                child_b = mutate_one_parameter(
                    child_b,
                    field_rules,
                    rng,
                    config=config.sampler_config,
                    mode_probabilities=config.mode_probabilities,
                )
            except Exception:
                child_b = right

        children.extend([child_a, child_b])

    return children[: len(parents)]


def evaluate_behavior_gene_on_train(
    gene: BehaviorGene,
    ctx: BehaviorTorchContext,
    train_dates: pd.DatetimeIndex,
    config: BehaviorGAConfig,
    generation: int,
    score_cache: dict[tuple[object, ...], tuple[FactorScore, str]],
) -> EvaluatedBehaviorGene:
    """Calculate and score one behavior gene on training dates."""

    key = gene_key(gene)
    if key in score_cache:
        score, error = score_cache[key]
        return EvaluatedBehaviorGene(gene=gene, train_score=score, generation=generation, error=error)

    try:
        factor = calculate_behavior_factor_tensor(
            gene,
            ctx,
            neutralize_size=config.neutralize_size,
            neutralize_industry=config.neutralize_industry,
            size_field=config.size_field,
        )
        score = score_behavior_factor_tensor(
            factor,
            ctx,
            dates=train_dates,
            ndcg_k=config.ndcg_k,
            ndcg_top_fraction=config.ndcg_top_fraction,
            direction=_score_direction(gene),
        )
        error = ""
        if score.coverage < config.min_coverage:
            coverage = score.coverage
            score = _empty_score()
            error = f"coverage below threshold: {coverage:.4f}"
    except Exception as exc:
        score = _empty_score()
        error = f"{type(exc).__name__}: {exc}"

    score_cache[key] = (score, error)
    return EvaluatedBehaviorGene(gene=gene, train_score=score, generation=generation, error=error)


def evaluate_behavior_population_on_train(
    genes: list[BehaviorGene],
    ctx: BehaviorTorchContext,
    train_dates: pd.DatetimeIndex,
    config: BehaviorGAConfig,
    generation: int,
    score_cache: dict[tuple[object, ...], tuple[FactorScore, str]],
) -> list[EvaluatedBehaviorGene]:
    """Evaluate a behavior population and drop exact duplicate genes."""

    unique_genes = _deduplicate_genes(genes)
    evaluated: list[EvaluatedBehaviorGene] = []
    iterator = _progress_iter(
        unique_genes,
        enabled=config.show_progress,
        total=len(unique_genes),
        desc=f"evaluate behavior gen {generation}",
    )
    for gene in iterator:
        evaluated.append(
            evaluate_behavior_gene_on_train(
                gene=gene,
                ctx=ctx,
                train_dates=train_dates,
                config=config,
                generation=generation,
                score_cache=score_cache,
            )
        )
    return evaluated


def _validate_config(config: BehaviorGAConfig, ctx: BehaviorTorchContext) -> None:
    if config.population_size <= 0:
        raise ValueError("population_size must be positive")
    if config.generations < 0:
        raise ValueError("generations must be non-negative")
    if not 0 <= config.crossover_prob <= 1:
        raise ValueError("crossover_prob must be in [0, 1]")
    if not 0 <= config.mutation_prob <= 1:
        raise ValueError("mutation_prob must be in [0, 1]")
    if not 0 < config.ndcg_top_fraction <= 1:
        raise ValueError("ndcg_top_fraction must be in (0, 1]")
    if not 0 <= config.min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    if config.require_cuda and ctx.device.type != "cuda":
        raise RuntimeError(f"Behavior GA requires CUDA, got device={ctx.device}")


def run_behavior_ga_search(
    ctx: BehaviorTorchContext,
    train_dates: pd.DatetimeIndex,
    config: BehaviorGAConfig | None = None,
) -> BehaviorSearchResult:
    """Run one multi-objective behavior-finance genetic search."""

    config = config or BehaviorGAConfig()
    _validate_config(config, ctx)
    rng = np.random.default_rng(config.random_seed)
    field_rules = ctx.behavior_field_rules

    population: list[BehaviorGene] = []
    init_attempts = 0
    max_init_attempts = max(1_000, config.population_size * 100)
    while len(population) < config.population_size and init_attempts < max_init_attempts:
        sampled = _try_random_gene(
            field_rules,
            rng,
            config=config.sampler_config,
            mode_probabilities=config.mode_probabilities,
        )
        if sampled is not None:
            population.append(sampled)
        init_attempts += 1
    if not population:
        raise RuntimeError("failed to sample any legal behavior gene for initial population")

    population = _refill_population(
        population,
        field_rules,
        config.population_size,
        rng,
        sampler_config=config.sampler_config,
        mode_probabilities=config.mode_probabilities,
    )

    score_cache: dict[tuple[object, ...], tuple[FactorScore, str]] = {}
    evaluated_population = evaluate_behavior_population_on_train(
        genes=population,
        ctx=ctx,
        train_dates=train_dates,
        config=config,
        generation=0,
        score_cache=score_cache,
    )
    history: list[EvaluatedBehaviorGene] = list(evaluated_population)

    generations = _progress_iter(
        range(1, config.generations + 1),
        enabled=config.show_progress,
        total=config.generations,
        desc="behavior GA generations",
        leave=True,
    )
    for generation in generations:
        parent_genes = [item.gene for item in evaluated_population]
        offspring = make_behavior_offspring(parent_genes, field_rules, config, rng)
        offspring = _refill_population(
            offspring,
            field_rules,
            config.population_size,
            rng,
            sampler_config=config.sampler_config,
            mode_probabilities=config.mode_probabilities,
        )

        evaluated_offspring = evaluate_behavior_population_on_train(
            genes=offspring,
            ctx=ctx,
            train_dates=train_dates,
            config=config,
            generation=generation,
            score_cache=score_cache,
        )
        history.extend(evaluated_offspring)

        combined = _deduplicate_evaluated(evaluated_population + evaluated_offspring)
        selected_idx = nsga2_select([item.objectives for item in combined], config.population_size)
        evaluated_population = [combined[idx] for idx in selected_idx]

    return BehaviorSearchResult(config=config, final_population=evaluated_population, history=history)


def validate_behavior_population(
    evaluated_population: list[EvaluatedBehaviorGene],
    ctx: BehaviorTorchContext,
    valid_dates: pd.DatetimeIndex,
    criteria: BehaviorValidationCriteria | None = None,
    *,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    label_horizon: int = 20,
    rebalance_freq: int | None = None,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    stamp_tax_rate: float = 0.001,
    neutralize_size: bool = True,
    neutralize_industry: bool = True,
    size_field: str = "barra_size",
    show_progress: bool = False,
) -> list[EvaluatedBehaviorGene]:
    """Evaluate final behavior genes on validation dates and flag survivors."""

    criteria = criteria or BehaviorValidationCriteria()
    output: list[EvaluatedBehaviorGene] = []
    pnl_n_groups = _n_groups_from_top_fraction(ndcg_top_fraction)
    cache = ctx.cache

    iterator = _progress_iter(
        evaluated_population,
        enabled=show_progress,
        total=len(evaluated_population),
        desc="validate behavior population",
    )
    for item in iterator:
        try:
            factor = calculate_behavior_factor_tensor(
                item.gene,
                ctx,
                neutralize_size=neutralize_size,
                neutralize_industry=neutralize_industry,
                size_field=size_field,
            )
            valid_score = score_behavior_factor_tensor(
                factor,
                ctx,
                dates=valid_dates,
                ndcg_k=ndcg_k,
                ndcg_top_fraction=ndcg_top_fraction,
                direction=item.train_score.direction,
            )
            pnl_result = factor_group_pnl(
                factor=ctx.tensor_to_frame(factor),
                label=cache.label,
                tradeable=cache.tradeable,
                dates=valid_dates,
                direction=item.train_score.direction,
                n_groups=pnl_n_groups,
                label_horizon=label_horizon,
                rebalance_freq=rebalance_freq,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                stamp_tax_rate=stamp_tax_rate,
            )
            pnl_metrics = _scalar_pnl_metrics(pnl_result)
            top_excess_ann = float(pnl_metrics.get("pnl_long_excess_ann", 0.0))
            passed = (
                valid_score.abs_rank_ic >= criteria.min_abs_rank_ic
                and valid_score.ic_win_rate >= criteria.min_ic_win_rate
                and top_excess_ann >= criteria.min_top_excess_ann
                and valid_score.coverage >= criteria.min_coverage
            )
            item.valid_score = valid_score
            item.valid_top_excess_ann = top_excess_ann
            item.valid_pnl_metrics = pnl_metrics
            item.passed_validation = passed
        except Exception as exc:
            item.valid_score = _empty_score()
            item.valid_top_excess_ann = 0.0
            item.valid_pnl_metrics = {}
            item.passed_validation = False
            item.error = f"validation {type(exc).__name__}: {exc}"
        output.append(item)

    return output


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def evaluated_behavior_to_frame(evaluated: list[EvaluatedBehaviorGene]) -> pd.DataFrame:
    """Flatten evaluated behavior genes into a readable DataFrame."""

    rows: list[dict[str, object]] = []
    for item in evaluated:
        gene_dict = item.gene.to_dict()
        row: dict[str, object] = {
            "generation": item.generation,
            "expression": describe_gene(item.gene),
            "error": item.error,
            "gene_mode": gene_dict["mode"],
            "gene_combiner": gene_dict["combiner"],
            "gene_direction_policy": gene_dict["direction_policy"],
            "gene_slots": _json_cell(gene_dict["slots"]),
            "gene_conditions": _json_cell(gene_dict["conditions"]),
            "gene_version": gene_dict["version"],
            **{f"train_{key}": value for key, value in item.train_score.to_dict().items()},
        }

        if item.valid_score is not None:
            row.update({f"valid_{key}": value for key, value in item.valid_score.to_dict().items()})
            row["valid_top_excess_ann"] = item.valid_top_excess_ann
            if item.valid_pnl_metrics is not None:
                row.update({f"valid_pnl_{key}": value for key, value in item.valid_pnl_metrics.items()})
            row["passed_validation"] = item.passed_validation

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        objective_cols = ["train_rank_ic_ir", "train_ndcg_at_k", "train_neutralized_icir"]
        existing = [col for col in objective_cols if col in df.columns]
        df = df.sort_values(existing, ascending=[False] * len(existing)).reset_index(drop=True)
    return df


def selected_behavior_rank_table(evaluated: list[EvaluatedBehaviorGene]) -> pd.DataFrame:
    """NSGA-II diagnostic table for an evaluated behavior population."""

    if not evaluated:
        return pd.DataFrame()
    objectives = [item.objectives for item in evaluated]
    table = rank_table(objectives)
    table["expression"] = [describe_gene(evaluated[idx].gene) for idx in table.index]
    return table


def export_behavior_search_result(
    result: BehaviorSearchResult,
    output_dir: str | Path,
    *,
    prefix: str = "behavior_ga",
) -> dict[str, Path]:
    """Write behavior GA results to CSV/JSON files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    history_path = output / f"{prefix}_history.csv"
    final_path = output / f"{prefix}_final_population.csv"
    rank_path = output / f"{prefix}_nsga2_rank.csv"
    config_path = output / f"{prefix}_config.json"

    evaluated_behavior_to_frame(result.history).to_csv(history_path, index=False, encoding="utf-8-sig")
    evaluated_behavior_to_frame(result.final_population).to_csv(final_path, index=False, encoding="utf-8-sig")
    selected_behavior_rank_table(result.final_population).to_csv(rank_path, index=True, encoding="utf-8-sig")
    config_path.write_text(json.dumps(asdict(result.config), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "history": history_path,
        "final_population": final_path,
        "rank_table": rank_path,
        "config": config_path,
    }
