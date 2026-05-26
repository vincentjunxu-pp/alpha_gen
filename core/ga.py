from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .factor_calc import calculate_factor
from .gene import (
    FactorGene,
    FieldRule,
    describe_gene,
    mutate_one_parameter,
    random_gene,
    repair_gene,
)
from .metrics import FactorScore, evaluate_factor, factor_group_pnl
from .nsga2 import nsga2_select, rank_table
from .preprocess import TransformCache


# ---------------------------------------------------------------------------
# Genetic search loop for alpha_gen's structured factor expressions.
#
# This is intentionally small and explicit:
#   1. Randomly initialize legal genes.
#   2. Evaluate each gene on the training dates.
#   3. Generate offspring by crossover and mutation.
#   4. Combine parents and offspring.
#   5. Use NSGA-II to keep the next generation.
#   6. Evaluate the final population on validation dates.
#
# The code avoids a generic framework because the gene structure is fixed and
# readable; all details should be easy to inspect before moving to real data.
# ---------------------------------------------------------------------------


GENE_FIELDS = (
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
class GAConfig:
    """Local-debug search parameters.

    The report uses a much larger population and more seeds. These defaults are
    deliberately small so a laptop can finish a smoke run quickly.
    """

    population_size: int = 80
    generations: int = 3
    crossover_prob: float = 0.85
    mutation_prob: float = 0.25
    random_seed: int = 1
    ndcg_k: int | None = None
    ndcg_top_fraction: float = 0.10
    min_coverage: float = 0.30
    mode_probabilities: dict[str, float] | None = None
    size_field: str = "barra_size"
    use_log_size: bool | None = None
    industry_scope: str | Sequence[str] | None = None
    barra_style_fields: Sequence[str] | str = ()
    barra_corr_threshold: float = 0.30
    barra_max_styles: int = 2
    use_gpu: bool = False
    device: str = "auto"
    cache_on_device: bool = True
    show_progress: bool = False


@dataclass(frozen=True)
class ValidationCriteria:
    """Validation filters inspired by the report."""

    min_abs_rank_ic: float = 0.02
    min_ic_win_rate: float = 0.55
    min_top_excess_ann: float = 0.02
    min_coverage: float = 0.30


@dataclass
class EvaluatedGene:
    """One gene plus its training score and optional validation diagnostics."""

    gene: FactorGene
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
class SearchResult:
    """Result of one GA run."""

    config: GAConfig
    final_population: list[EvaluatedGene]
    history: list[EvaluatedGene]


def should_neutralize_industry(industry_scope: str | Sequence[str] | None) -> bool:
    """Return whether a universe should be neutralized by industry.

    A concrete single industry, e.g. "electronics", is treated as a
    single-industry universe and skipped. "all", "multi", or a sequence with
    more than one distinct industry means the factor is searched on a
    cross-industry universe and should be industry-neutralized before size
    neutralization.
    """

    if industry_scope is None:
        return False
    if isinstance(industry_scope, str):
        text = industry_scope.strip().lower()
        if not text or text in {"none", "single", "specific", "one"}:
            return False
        return text in {"all", "all_industries", "all-industry", "multi", "multiple", "全行业", "多行业"}

    values = {str(value).strip() for value in industry_scope if str(value).strip()}
    if not values:
        return False
    lowered = {value.lower() for value in values}
    if lowered & {"all", "all_industries", "all-industry", "multi", "multiple", "全行业", "多行业"}:
        return True
    return len(values) > 1


def normalize_barra_style_fields(fields: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize user-provided Barra style field names for Torch evaluation."""

    if fields is None:
        return ()
    if isinstance(fields, str):
        values = [fields]
    else:
        values = list(fields)
    return tuple(
        dict.fromkeys(
            text
            for field in values
            if field is not None
            for text in [str(field).strip()]
            if text
        )
    )


def gene_key(gene: FactorGene) -> tuple[object, ...]:
    """Stable semantic key used for de-duplicating genes.

    Some stored parameters are inactive under simpler modes. Addition is
    commutative inside one side of a pair expression, so `A+B` and `B+A` share
    the same key. Subtraction is order-sensitive and must be preserved.
    """

    def pair_key(left: tuple[str, str], right: tuple[str, str], op: str) -> tuple[object, ...]:
        fields = tuple(sorted((left, right))) if op == "+" else (left, right)
        return (op, *fields)

    if gene.mode == "single":
        return (gene.mode, gene.a, gene.a_transform)
    if gene.mode in {"ratio", "resi"}:
        return (gene.mode, gene.a, gene.a_transform, gene.b, gene.b_transform)
    if gene.mode == "resi_pair":
        controls = tuple(sorted(((gene.b, gene.b_transform), (gene.c, gene.c_transform))))
        return (gene.mode, gene.a, gene.a_transform, controls)
    if gene.mode == "multi_resi":
        controls = tuple(
            sorted(
                (
                    (gene.b, gene.b_transform),
                    (gene.c, gene.c_transform),
                    (gene.d, gene.d_transform),
                )
            )
        )
        return (gene.mode, gene.a, gene.a_transform, controls)
    if gene.mode == "spread":
        return (
            gene.mode,
            (gene.a, gene.a_transform, gene.b, gene.b_transform),
            (gene.c, gene.c_transform, gene.d, gene.d_transform),
        )
    if gene.mode == "pair_ratio":
        return (
            gene.mode,
            pair_key((gene.a, gene.a_transform), (gene.b, gene.b_transform), gene.left_op),
            pair_key((gene.c, gene.c_transform), (gene.d, gene.d_transform), gene.right_op),
        )
    if gene.mode == "style_composite":
        terms = tuple(sorted((gene.a, gene.b))) if gene.left_op == "+" else (gene.a, gene.b)
        return (gene.mode, gene.left_op, *terms)
    return tuple(gene.to_dict().get(field) for field in GENE_FIELDS)


def _empty_score() -> FactorScore:
    """Worst-case score used when a factor calculation fails."""

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


def _deduplicate_genes(genes: Iterable[FactorGene]) -> list[FactorGene]:
    """Keep the first copy of each gene."""

    seen: set[tuple[object, ...]] = set()
    unique: list[FactorGene] = []
    for gene in genes:
        key = gene_key(gene)
        if key in seen:
            continue
        seen.add(key)
        unique.append(gene)
    return unique


def _deduplicate_evaluated(evaluated: Iterable[EvaluatedGene]) -> list[EvaluatedGene]:
    """Keep the first evaluated copy of each semantic gene."""

    seen: set[tuple[object, ...]] = set()
    unique: list[EvaluatedGene] = []
    for item in evaluated:
        key = gene_key(item.gene)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _n_groups_from_top_fraction(top_fraction: float) -> int:
    """Convert a top fraction into alpha_factory-style quantile group count."""

    return max(2, int(round(1.0 / top_fraction))) if 0 < top_fraction <= 1 else 5


def _scalar_pnl_metrics(pnl_result: dict[str, object]) -> dict[str, float | int]:
    """Keep scalar PnL diagnostics for CSV/JSON style outputs."""

    output: dict[str, float | int] = {}
    for key, value in pnl_result.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            output[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    return output


def crossover_genes(
    left: FactorGene,
    right: FactorGene,
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
    *,
    mode_probabilities: Mapping[str, float] | None = None,
) -> tuple[FactorGene, FactorGene]:
    """Uniform crossover over the structured gene parameters.

    Every parameter is independently swapped with probability 0.5. The
    offspring are repaired afterward because a field can be legal on one side of
    an expression and illegal on another.
    """

    left_dict = left.to_dict()
    right_dict = right.to_dict()

    child_a = left_dict.copy()
    child_b = right_dict.copy()
    for field in GENE_FIELDS:
        if rng.random() < 0.5:
            child_a[field], child_b[field] = child_b[field], child_a[field]

    repaired_a = repair_gene(FactorGene.from_dict(child_a), field_rules, rng, mode_probabilities=mode_probabilities)
    repaired_b = repair_gene(FactorGene.from_dict(child_b), field_rules, rng, mode_probabilities=mode_probabilities)
    return repaired_a, repaired_b


def _try_random_gene(
    field_rules: Mapping[str, FieldRule],
    rng: np.random.Generator,
    *,
    mode_probabilities: Mapping[str, float] | None = None,
) -> FactorGene | None:
    """Sample a gene, returning None instead of aborting the GA on sampler errors."""

    try:
        return random_gene(field_rules, rng, mode_probabilities=mode_probabilities)
    except Exception:
        return None


def make_offspring(
    parents: list[FactorGene],
    field_rules: Mapping[str, FieldRule],
    config: GAConfig,
    rng: np.random.Generator,
) -> list[FactorGene]:
    """Create one offspring generation from the current population."""

    if not parents:
        raise ValueError("parents cannot be empty")

    order = rng.permutation(len(parents))
    shuffled = [parents[i] for i in order]
    children: list[FactorGene] = []

    # Pair adjacent parents. If population size is odd, the last parent pairs
    # with the first one; this keeps the generation size stable.
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
                    mode_probabilities=config.mode_probabilities,
                )
            except Exception:
                child_b = right

        children.extend([child_a, child_b])

    return children[: len(parents)]


def evaluate_gene_on_train(
    gene: FactorGene,
    cache: TransformCache,
    train_dates: pd.DatetimeIndex,
    config: GAConfig,
    generation: int,
    score_cache: dict[tuple[object, ...], tuple[FactorScore, str]],
    eval_context: object | None = None,
) -> EvaluatedGene:
    """Calculate and score one gene on training dates."""

    key = gene_key(gene)
    if key in score_cache:
        score, error = score_cache[key]
        return EvaluatedGene(gene=gene, train_score=score, generation=generation, error=error)

    try:
        neutralize_industry = should_neutralize_industry(config.industry_scope)
        if eval_context is None:
            if normalize_barra_style_fields(config.barra_style_fields):
                raise RuntimeError(
                    "Barra-neutralized NSGA objective requires TorchEvalContext; "
                    "set use_gpu=True or pass eval_context explicitly"
                )
            factor = calculate_factor(
                gene,
                cache,
                neutralize_industry=neutralize_industry,
                size_field=config.size_field,
                use_log_size=config.use_log_size,
            )
            score = evaluate_factor(
                factor=factor,
                label=cache.label,
                tradeable=cache.tradeable,
                dates=train_dates,
                ndcg_k=config.ndcg_k,
                ndcg_top_fraction=config.ndcg_top_fraction,
            )
        else:
            from .torch_backend import calculate_factor_tensor, evaluate_factor_tensor

            factor = calculate_factor_tensor(
                gene,
                eval_context,
                neutralize_industry=neutralize_industry,
                size_field=config.size_field,
                use_log_size=config.use_log_size,
            )
            score = evaluate_factor_tensor(
                factor=factor,
                ctx=eval_context,
                dates=train_dates,
                ndcg_k=config.ndcg_k,
                ndcg_top_fraction=config.ndcg_top_fraction,
            )
        error = ""
        if score.coverage < config.min_coverage:
            # Keep the diagnostics, but prevent sparse factors from dominating.
            coverage = score.coverage
            score = _empty_score()
            error = f"coverage below threshold: {coverage:.4f}"
    except Exception as exc:
        score = _empty_score()
        error = f"{type(exc).__name__}: {exc}"

    score_cache[key] = (score, error)
    return EvaluatedGene(gene=gene, train_score=score, generation=generation, error=error)


def evaluate_population_on_train(
    genes: list[FactorGene],
    cache: TransformCache,
    train_dates: pd.DatetimeIndex,
    config: GAConfig,
    generation: int,
    score_cache: dict[tuple[object, ...], tuple[FactorScore, str]],
    eval_context: object | None = None,
) -> list[EvaluatedGene]:
    """Evaluate a population and drop exact duplicate genes."""

    unique_genes = _deduplicate_genes(genes)
    evaluated: list[EvaluatedGene] = []
    iterator = _progress_iter(
        unique_genes,
        enabled=config.show_progress,
        total=len(unique_genes),
        desc=f"evaluate gen {generation}",
    )
    for gene in iterator:
        evaluated.append(
            evaluate_gene_on_train(
                gene=gene,
                cache=cache,
                train_dates=train_dates,
                config=config,
                generation=generation,
                score_cache=score_cache,
                eval_context=eval_context,
            )
        )
    return evaluated


def _refill_population(
    genes: list[FactorGene],
    field_rules: Mapping[str, FieldRule],
    target_size: int,
    rng: np.random.Generator,
    *,
    mode_probabilities: Mapping[str, float] | None = None,
) -> list[FactorGene]:
    """Add random legal genes if de-duplication made a population too small."""

    output = _deduplicate_genes(genes)
    attempts = 0
    max_attempts = max(1_000, target_size * 100)
    while len(output) < target_size and attempts < max_attempts:
        sampled = _try_random_gene(
            field_rules,
            rng,
            mode_probabilities=mode_probabilities,
        )
        if sampled is not None:
            output = _deduplicate_genes(output + [sampled])
        attempts += 1
    while len(output) < target_size:
        # The semantic search space can be smaller than a requested production
        # population after de-duplication. Allow duplicate fillers rather than
        # looping forever; score_cache still prevents repeated expensive work.
        sampled = _try_random_gene(
            field_rules,
            rng,
            mode_probabilities=mode_probabilities,
        )
        if sampled is not None:
            output.append(sampled)
        elif output:
            output.append(output[int(rng.integers(len(output)))])
        else:
            raise RuntimeError("failed to sample any legal gene for population refill")
    return output[:target_size]


def run_ga_search(
    cache: TransformCache,
    field_rules: Mapping[str, FieldRule],
    train_dates: pd.DatetimeIndex,
    config: GAConfig | None = None,
    eval_context: object | None = None,
) -> SearchResult:
    """Run one multi-objective genetic search."""

    config = config or GAConfig()
    rng = np.random.default_rng(config.random_seed)
    barra_style_fields = normalize_barra_style_fields(config.barra_style_fields)

    if config.use_gpu and eval_context is None:
        from .torch_backend import TorchEvalContext

        eval_context = TorchEvalContext(
            cache=cache,
            device=config.device,
            cache_on_device=config.cache_on_device,
            barra_style_fields=barra_style_fields,
            barra_corr_threshold=config.barra_corr_threshold,
            barra_max_styles=config.barra_max_styles,
        )
    elif eval_context is None and barra_style_fields:
        raise ValueError(
            "barra_style_fields were provided but no TorchEvalContext is active; "
            "set GAConfig(use_gpu=True, barra_style_fields=...) for a real neutralized_icir objective"
        )

    population: list[FactorGene] = []
    init_attempts = 0
    max_init_attempts = max(1_000, config.population_size * 100)
    while len(population) < config.population_size and init_attempts < max_init_attempts:
        sampled = _try_random_gene(
            field_rules,
            rng,
            mode_probabilities=config.mode_probabilities,
        )
        if sampled is not None:
            population.append(sampled)
        init_attempts += 1
    if not population:
        raise RuntimeError("failed to sample any legal gene for initial population")
    population = _refill_population(
        population,
        field_rules,
        config.population_size,
        rng,
        mode_probabilities=config.mode_probabilities,
    )

    score_cache: dict[tuple[object, ...], tuple[FactorScore, str]] = {}
    evaluated_population = evaluate_population_on_train(
        genes=population,
        cache=cache,
        train_dates=train_dates,
        config=config,
        generation=0,
        score_cache=score_cache,
        eval_context=eval_context,
    )
    history: list[EvaluatedGene] = list(evaluated_population)

    generations = _progress_iter(
        range(1, config.generations + 1),
        enabled=config.show_progress,
        total=config.generations,
        desc="GA generations",
        leave=True,
    )
    for generation in generations:
        parent_genes = [item.gene for item in evaluated_population]
        offspring = make_offspring(parent_genes, field_rules, config, rng)
        offspring = _refill_population(
            offspring,
            field_rules,
            config.population_size,
            rng,
            mode_probabilities=config.mode_probabilities,
        )

        evaluated_offspring = evaluate_population_on_train(
            genes=offspring,
            cache=cache,
            train_dates=train_dates,
            config=config,
            generation=generation,
            score_cache=score_cache,
            eval_context=eval_context,
        )
        history.extend(evaluated_offspring)

        combined = _deduplicate_evaluated(evaluated_population + evaluated_offspring)
        selected_idx = nsga2_select([item.objectives for item in combined], config.population_size)
        evaluated_population = [combined[idx] for idx in selected_idx]

    return SearchResult(config=config, final_population=evaluated_population, history=history)


def validate_population(
    evaluated_population: list[EvaluatedGene],
    cache: TransformCache,
    valid_dates: pd.DatetimeIndex,
    criteria: ValidationCriteria | None = None,
    *,
    ndcg_k: int | None = None,
    ndcg_top_fraction: float = 0.10,
    label_horizon: int = 20,
    rebalance_freq: int | None = None,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    stamp_tax_rate: float = 0.001,
    size_field: str = "barra_size",
    use_log_size: bool | None = None,
    industry_scope: str | Sequence[str] | None = None,
    eval_context: object | None = None,
    show_progress: bool = False,
) -> list[EvaluatedGene]:
    """Evaluate final genes on validation dates and flag passing factors.

    Validation uses the direction learned on the training set. This avoids
    silently flipping a factor direction after seeing validation performance.
    """

    criteria = criteria or ValidationCriteria()
    output: list[EvaluatedGene] = []
    pnl_n_groups = _n_groups_from_top_fraction(ndcg_top_fraction)
    neutralize_industry = should_neutralize_industry(industry_scope)

    iterator = _progress_iter(
        evaluated_population,
        enabled=show_progress,
        total=len(evaluated_population),
        desc="validate population",
    )
    for item in iterator:
        try:
            if eval_context is None:
                factor = calculate_factor(
                    item.gene,
                    cache,
                    neutralize_industry=neutralize_industry,
                    size_field=size_field,
                    use_log_size=use_log_size,
                )
                valid_score = evaluate_factor(
                    factor=factor,
                    label=cache.label,
                    tradeable=cache.tradeable,
                    dates=valid_dates,
                    ndcg_k=ndcg_k,
                    ndcg_top_fraction=ndcg_top_fraction,
                    direction=item.train_score.direction,
                )
                pnl_result = factor_group_pnl(
                    factor=factor,
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
            else:
                from .torch_backend import (
                    calculate_factor_tensor,
                    evaluate_factor_tensor,
                )

                factor = calculate_factor_tensor(
                    item.gene,
                    eval_context,
                    neutralize_industry=neutralize_industry,
                    size_field=size_field,
                    use_log_size=use_log_size,
                )
                valid_score = evaluate_factor_tensor(
                    factor=factor,
                    ctx=eval_context,
                    dates=valid_dates,
                    ndcg_k=ndcg_k,
                    ndcg_top_fraction=ndcg_top_fraction,
                    direction=item.train_score.direction,
                )
                factor_frame = pd.DataFrame(
                    factor.detach().cpu().numpy(),
                    index=eval_context.cache.label.index,
                    columns=eval_context.cache.label.columns,
                )
                pnl_result = factor_group_pnl(
                    factor=factor_frame,
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


def evaluated_to_frame(evaluated: list[EvaluatedGene]) -> pd.DataFrame:
    """Flatten evaluated genes into a readable DataFrame."""

    rows: list[dict[str, object]] = []
    for item in evaluated:
        row: dict[str, object] = {
            "generation": item.generation,
            "expression": describe_gene(item.gene),
            "error": item.error,
            **{f"gene_{key}": value for key, value in item.gene.to_dict().items()},
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


def selected_rank_table(evaluated: list[EvaluatedGene]) -> pd.DataFrame:
    """NSGA-II diagnostic table for an evaluated population."""

    objectives = [item.objectives for item in evaluated]
    table = rank_table(objectives)
    table["expression"] = [describe_gene(evaluated[idx].gene) for idx in table.index]
    return table


def export_search_result(
    result: SearchResult,
    output_dir: str | Path,
    *,
    prefix: str = "mock_ga",
) -> dict[str, Path]:
    """Write GA results to CSV/JSON files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    history_path = output / f"{prefix}_history.csv"
    final_path = output / f"{prefix}_final_population.csv"
    config_path = output / f"{prefix}_config.json"

    evaluated_to_frame(result.history).to_csv(history_path, index=False, encoding="utf-8-sig")
    evaluated_to_frame(result.final_population).to_csv(final_path, index=False, encoding="utf-8-sig")
    config_path.write_text(json.dumps(asdict(result.config), indent=2), encoding="utf-8")

    return {
        "history": history_path,
        "final_population": final_path,
        "config": config_path,
    }
