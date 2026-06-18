from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from alpha_gen.core.metrics import FactorScore
from alpha_gen.core.nsga2 import nsga2_select, rank_table

from .gene import BehaviorFieldRule, BehaviorGene, describe_gene, describe_gene_formula, gene_key
from .sampler import (
    BehaviorSamplerConfig,
    crossover_genes,
    mutate_one_parameter,
    random_gene,
)
from .torch_backend import (
    BEHAVIOR_NEUTRALIZATION_MODES,
    NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    NEUTRALIZATION_SIZE_THEN_INDUSTRY,
    BehaviorTorchContext,
    calculate_behavior_factor_tensor,
    score_behavior_factor_tensor,
    validate_neutralization_requirements,
)


NSGA_MODE_RIR_LONG_RIR_NDCG = "rir_long_rir_ndcg"
NSGA_MODE_RIR_LONG_RIR = "rir_long_rir"
NSGA_OBJECTIVE_MODES: dict[str, tuple[str, ...]] = {
    NSGA_MODE_RIR_LONG_RIR_NDCG: ("rir", "long_rir", "ndcg_k"),
    NSGA_MODE_RIR_LONG_RIR: ("rir", "long_rir"),
}


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
    nsga_objective_mode: str = NSGA_MODE_RIR_LONG_RIR_NDCG
    min_coverage: float = 0.30
    mode_probabilities: dict[str, float] | None = None
    sampler_config: BehaviorSamplerConfig = field(default_factory=BehaviorSamplerConfig)
    neutralization_mode: str = NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY
    size_field: str = "barra_size"
    label_horizon: int = 20
    rebalance_freq: int | None = None
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002
    stamp_tax_rate: float = 0.001
    require_cuda: bool = True
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if not 0 <= self.crossover_prob <= 1:
            raise ValueError("crossover_prob must be in [0, 1]")
        if not 0 <= self.mutation_prob <= 1:
            raise ValueError("mutation_prob must be in [0, 1]")
        if not 0 < self.ndcg_top_fraction <= 1:
            raise ValueError("ndcg_top_fraction must be in (0, 1]")
        if self.ndcg_k is not None and self.ndcg_k <= 0:
            raise ValueError("ndcg_k must be a positive integer when provided")
        if self.nsga_objective_mode not in NSGA_OBJECTIVE_MODES:
            raise ValueError(
                f"nsga_objective_mode must be one of {tuple(NSGA_OBJECTIVE_MODES)}"
            )
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        if self.neutralization_mode not in BEHAVIOR_NEUTRALIZATION_MODES:
            raise ValueError(
                f"neutralization_mode must be one of {BEHAVIOR_NEUTRALIZATION_MODES}"
            )
        if self.label_horizon <= 0:
            raise ValueError("label_horizon must be positive")
        if self.rebalance_freq is not None and self.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be positive when provided")
        if min(self.commission_rate, self.slippage_rate, self.stamp_tax_rate) < 0:
            raise ValueError("transaction cost rates must be non-negative")


@dataclass(frozen=True)
class BehaviorValidationCriteria:
    """Validation filters for a final behavior-factor population."""

    min_rank_ic: float = 0.02
    min_ic_win_rate: float = 0.55
    min_coverage: float = 0.30




@dataclass
class EvaluatedBehaviorGene:
    """One behavior gene plus training and optional validation diagnostics."""

    gene: BehaviorGene
    train_score: FactorScore
    generation: int
    error: str = ""
    train_metrics: object | None = None   # always None — all metrics on GPU
    valid_score: FactorScore | None = None
    valid_metrics: object | None = None   # always None — all metrics on GPU
    passed_validation: bool | None = None

    def objectives(self, objective_mode: str) -> tuple[float, ...]:
        """Return configured maximization objectives from GPU train score.

        Reads directly from *train_score* (FactorScore, computed entirely on
        GPU) instead of *train_metrics*.
        This keeps the training loop GPU-bound.
        """

        if objective_mode not in NSGA_OBJECTIVE_MODES:
            raise ValueError(
                f"objective_mode must be one of {tuple(NSGA_OBJECTIVE_MODES)}"
            )
        # Map NSGA objective names → FactorScore field names
        _NAME_MAP = {
            "rir": "rank_ic_ir",
            "long_rir": "long_rank_ic_ir",
            "ndcg_k": "ndcg_at_k",
        }
        return tuple(
            float(getattr(self.train_score, _NAME_MAP[name]))
            for name in NSGA_OBJECTIVE_MODES[objective_mode]
        )


@dataclass
class BehaviorSearchResult:
    """Result of one behavior-finance GA run."""

    config: BehaviorGAConfig
    final_population: list[EvaluatedBehaviorGene]
    history: list[EvaluatedBehaviorGene]


def _empty_score() -> FactorScore:
    return FactorScore(
        mean_rank_ic=0.0,
        rank_ic_ir=0.0,
        ic_win_rate=0.0,
        ndcg_at_k=0.0,
        direction=1,
        n_ic_obs=0,
        coverage=0.0,
        neutralized_icir=0.0,
        neutralized_mean_rank_ic=0.0,
        neutralized_ic_win_rate=0.0,
        neutralized_n_ic_obs=0,
    )


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
    if len(output) >= target_size:
        return output[:target_size]
    # Last-resort fallback: if we still don't have enough unique genes,
    # clone existing genes up to target_size. Each iteration adds 1,
    # so the loop is bounded by (target_size - len(output)) iterations.
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
    score_cache: dict[
        tuple[object, ...],
        tuple[FactorScore, FactorScore | None, str],
    ],
    *,
    valid_dates: pd.DatetimeIndex | None = None,
) -> EvaluatedBehaviorGene:
    """Calculate and score one behavior gene — **GPU only**.

    Both train and validation scoring run entirely on GPU via
    :func:`score_behavior_factor_tensor`.  NSGA objectives are read
    from the resulting :class:`FactorScore` objects — no CPU
    ``compact_factor_metrics`` is computed here.
    """

    key = gene_key(gene)
    if key in score_cache:
        train_s, valid_s, error = score_cache[key]
        return EvaluatedBehaviorGene(
            gene=gene, train_score=train_s, train_metrics=None,
            generation=generation, error=error,
            valid_score=valid_s, valid_metrics=None, passed_validation=None,
        )

    valid_score = None
    try:
        factor = calculate_behavior_factor_tensor(
            gene, ctx,
            neutralization_mode=config.neutralization_mode,
            size_field=config.size_field,
        )
        score = score_behavior_factor_tensor(
            factor, ctx, dates=train_dates,
            ndcg_k=config.ndcg_k,
            ndcg_top_fraction=config.ndcg_top_fraction,
            direction=_score_direction(gene),
            neutralization_mode=config.neutralization_mode,
        )
        error = ""
        if score.coverage < config.min_coverage:
            error = f"coverage below threshold: {score.coverage:.4f}"
            score = _empty_score()

        # ---- validation: GPU only (same factor tensor) ----------
        if valid_dates is not None and not error:
            try:
                valid_score = score_behavior_factor_tensor(
                    factor, ctx, dates=valid_dates,
                    ndcg_k=config.ndcg_k,
                    ndcg_top_fraction=config.ndcg_top_fraction,
                    direction=score.direction,
                    neutralization_mode=config.neutralization_mode,
                )
            except Exception as exc:
                valid_score = _empty_score()
                if not error:
                    error = f"valid {type(exc).__name__}: {exc}"

        # ---- GPU memory cleanup ---------------------------------
        del factor
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        score = _empty_score()
        error = f"{type(exc).__name__}: {exc}"

    # train_metrics / valid_metrics always None — NSGA reads from
    # train_score / valid_score (GPU FactorScore).
    score_cache[key] = (score, valid_score, error)
    return EvaluatedBehaviorGene(
        gene=gene, train_score=score, train_metrics=None,
        generation=generation, error=error,
        valid_score=valid_score, valid_metrics=None,
        passed_validation=None,
    )


def evaluate_behavior_population_on_train(
    genes: list[BehaviorGene],
    ctx: BehaviorTorchContext,
    train_dates: pd.DatetimeIndex,
    config: BehaviorGAConfig,
    generation: int,
    score_cache: dict[
        tuple[object, ...],
        tuple[FactorScore, FactorScore | None, str],
    ],
    *,
    valid_dates: pd.DatetimeIndex | None = None,
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
                valid_dates=valid_dates,
            )
        )
    return evaluated


def _validate_config(config: BehaviorGAConfig, ctx: BehaviorTorchContext) -> None:
    # Config-level validations are handled by BehaviorGAConfig.__post_init__.
    # This function only validates context-dependent requirements.
    error = validate_neutralization_requirements(
        config.neutralization_mode,
        barra_style_fields=ctx.barra_style_fields,
        has_industry=ctx.cache.industry is not None,
    )
    if error is not None:
        raise ValueError(error)
    if config.neutralization_mode == NEUTRALIZATION_SIZE_THEN_INDUSTRY and (config.size_field, False) not in ctx.cache.current:
        raise ValueError(
            f"size_then_industry requires size field {config.size_field!r} in cache.current"
        )
    if config.require_cuda and ctx.device.type != "cuda":
        raise RuntimeError(f"Behavior GA requires CUDA, got device={ctx.device}")


def run_behavior_ga_search(
    ctx: BehaviorTorchContext,
    train_dates: pd.DatetimeIndex,
    config: BehaviorGAConfig | None = None,
    *,
    valid_dates: pd.DatetimeIndex | None = None,
) -> BehaviorSearchResult:
    """Run one multi-objective behavior-finance genetic search.

    When *valid_dates* is provided, every gene is also scored on the
    validation window from the **same** factor tensor.  This populates
    ``valid_score`` / ``valid_metrics`` on every history entry and
    eliminates the need for a separate :func:`validate_behavior_population`
    pass later.
    """

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

    score_cache: dict[
        tuple[object, ...],
        tuple[FactorScore, FactorScore | None, str],
    ] = {}
    evaluated_population = evaluate_behavior_population_on_train(
        genes=population,
        ctx=ctx,
        train_dates=train_dates,
        config=config,
        generation=0,
        score_cache=score_cache,
        valid_dates=valid_dates,
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
            valid_dates=valid_dates,
        )
        history.extend(evaluated_offspring)

        combined = _deduplicate_evaluated(evaluated_population + evaluated_offspring)
        selected_idx = nsga2_select(
            [
                item.objectives(config.nsga_objective_mode)
                for item in combined
            ],
            config.population_size,
        )
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
    neutralization_mode: str = NEUTRALIZATION_RAW_FULL_BARRA_INDUSTRY,
    size_field: str = "barra_size",
    show_progress: bool = False,
    max_validation_genes: int | None = None,
) -> list[EvaluatedBehaviorGene]:
    """Evaluate every unique gene on validation dates — GPU only.

    Input is de-duplicated by gene structure before any tensor is
    allocated.  When *max_validation_genes* is set, genes are pre-filtered
    by train ``rir`` to bound GPU work.
    """

    criteria = criteria or BehaviorValidationCriteria()
    if neutralization_mode not in BEHAVIOR_NEUTRALIZATION_MODES:
        raise ValueError(
            f"neutralization_mode must be one of {BEHAVIOR_NEUTRALIZATION_MODES}"
        )
    error = validate_neutralization_requirements(
        neutralization_mode,
        barra_style_fields=ctx.barra_style_fields,
        has_industry=ctx.cache.industry is not None,
    )
    if error is not None:
        raise ValueError(error)

    use_cuda = ctx.device.type == "cuda"

    # ---- 1. De-duplicate on gene structure BEFORE any tensor work ----------
    unique_genes = _deduplicate_evaluated(evaluated_population)
    if show_progress and len(unique_genes) < len(evaluated_population):
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            pass
        else:
            _tqdm.write(
                f"validate: deduplicated {len(evaluated_population)} → "
                f"{len(unique_genes)} unique genes"
            )

    # ---- 2. Pre-filter by train rir (memory safeguard) ----------------------
    if max_validation_genes is not None and max_validation_genes > 0 and len(unique_genes) > max_validation_genes:
        # Sort descending by train_rir.  Genes without train_metrics (errors)
        # are placed last so they are dropped first.
        def _train_rir(item: EvaluatedBehaviorGene) -> float:
            return float(item.train_score.rank_ic_ir)

        unique_genes.sort(key=_train_rir, reverse=True)
        unique_genes = unique_genes[:max_validation_genes]
        if show_progress:
            try:
                from tqdm.auto import tqdm as _tqdm
            except ImportError:
                pass
            else:
                _tqdm.write(
                    f"validate: pre-filtered to {len(unique_genes)} genes "
                    f"by train_rir"
                )

    # ---- 3. Score each unique gene on the validation window ----------------
    # Free training-phase GPU cache to make room for validation tensors.
    # With real data (5000 stocks × 1000 days × 80 fields), the cache can
    # hold >2 GB; clearing it prevents CUDA allocator crashes.
    ctx.clear_tensor_cache()

    validated_all: list[EvaluatedBehaviorGene] = []
    error_count = 0
    iterator = _progress_iter(
        unique_genes,
        enabled=show_progress,
        total=len(unique_genes),
        desc="validate behavior population",
    )
    for item in iterator:
        try:
            factor = calculate_behavior_factor_tensor(
                item.gene, ctx,
                neutralization_mode=neutralization_mode,
                size_field=size_field,
            )
            valid_score = score_behavior_factor_tensor(
                factor, ctx,
                dates=valid_dates,
                ndcg_k=ndcg_k,
                ndcg_top_fraction=ndcg_top_fraction,
                direction=item.train_score.direction,
                neutralization_mode=neutralization_mode,
            )
            passed = (
                abs(valid_score.mean_rank_ic) >= criteria.min_rank_ic
                and valid_score.ic_win_rate >= criteria.min_ic_win_rate
                and valid_score.coverage >= criteria.min_coverage
            )
            item.valid_score = valid_score
            item.valid_metrics = None
            item.passed_validation = passed

            del factor
            if ctx.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:
            item.valid_score = _empty_score()
            item.valid_metrics = None
            item.passed_validation = False
            item.error = f"validation {type(exc).__name__}: {exc}"
            error_count += 1
            # Attempt cleanup — but NVML crashes are fatal at C++ level
            if use_cuda:
                torch.cuda.empty_cache()

        validated_all.append(item)

    if show_progress and error_count > 0:
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            pass
        else:
            _tqdm.write(
                f"validate: {len(validated_all)} genes scored, "
                f"{error_count} errors, "
                f"{sum(1 for g in validated_all if g.passed_validation)} passed criteria"
            )

    return validated_all


def select_validation_population(
    validated_genes: list[EvaluatedBehaviorGene],
    population_size: int,
    *,
    nsga_objective_mode: str = NSGA_MODE_RIR_LONG_RIR_NDCG,
) -> list[EvaluatedBehaviorGene]:
    """Run NSGA‑II on validation GPU scores to select the final population.

    Reads objectives directly from ``valid_score`` (:class:`FactorScore`).
    Genes whose evaluation errored or whose *valid_score* is ``None`` are
    excluded.
    """

    if nsga_objective_mode not in NSGA_OBJECTIVE_MODES:
        raise ValueError(
            f"nsga_objective_mode must be one of {tuple(NSGA_OBJECTIVE_MODES)}"
        )
    _NAME_MAP = {
        "rir": "rank_ic_ir",
        "long_rir": "long_rank_ic_ir",
        "ndcg_k": "ndcg_at_k",
    }
    objective_names = NSGA_OBJECTIVE_MODES[nsga_objective_mode]

    clean = [
        item for item in validated_genes
        if item.valid_score is not None and not item.error
    ]
    if not clean:
        return []

    if len(clean) <= population_size:
        return clean

    objectives = [
        tuple(
            float(getattr(item.valid_score, _NAME_MAP[name]))
            for name in objective_names
        )
        for item in clean
    ]
    selected_idx = nsga2_select(objectives, population_size)
    return [clean[idx] for idx in selected_idx]


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _metrics_from_score(score: FactorScore) -> dict[str, float]:
    """GPU :class:`FactorScore` → flat value dict for DataFrame columns."""
    return {
        "ric": score.mean_rank_ic * score.direction,
        "rir": score.rank_ic_ir,
        "ic": score.ic,
        "ir": score.ir,
        "long_ric": score.long_rank_ic,
        "long_rir": score.long_rank_ic_ir,
        "long_ic": score.long_ic,
        "long_ir": score.long_ir,
        "ndcg_k": score.ndcg_at_k,
        "win_rate": score.ic_win_rate,
        "coverage": score.coverage,
        "n_ic_obs": float(score.n_ic_obs),
        "neutralized_ric": score.neutralized_mean_rank_ic * score.direction,
        "neutralized_rir": score.neutralized_icir,
        "neutralized_win_rate": score.neutralized_ic_win_rate,
        "neutralized_n_ic_obs": float(score.neutralized_n_ic_obs),
    }


def evaluated_behavior_to_frame(
    evaluated: list[EvaluatedBehaviorGene],
    *,
    objective_mode: str = NSGA_MODE_RIR_LONG_RIR_NDCG,
) -> pd.DataFrame:
    """Flatten evaluated behavior genes into a readable DataFrame.

    When *train_metrics* is ``None`` (training phase — GPU-only NSGA),
    the essential columns are backfilled from *train_score* (FactorScore).
    PnL columns (sharpe, long_sharpe) will be NaN until a
    post-training ``compact_factor_metrics`` pass fills them.
    """

    if objective_mode not in NSGA_OBJECTIVE_MODES:
        raise ValueError(
            f"objective_mode must be one of {tuple(NSGA_OBJECTIVE_MODES)}"
        )
    rows: list[dict[str, object]] = []
    for item in evaluated:
        gene_dict = item.gene.to_dict()
        train_values = _metrics_from_score(item.train_score)
        row: dict[str, object] = {
            "generation": item.generation,
            "expression": describe_gene(item.gene),
            "formula": describe_gene_formula(item.gene),
            "error": item.error,
            "gene_mode": gene_dict["mode"],
            "gene_combiner": gene_dict["combiner"],
            "gene_direction_policy": gene_dict["direction_policy"],
            "gene_slots": _json_cell(gene_dict["slots"]),
            "gene_conditions": _json_cell(gene_dict["conditions"]),
            "gene_version": gene_dict["version"],
            "train_direction": int(item.train_score.direction),
            **{f"train_{key}": value for key, value in train_values.items()},
        }

        # Valid columns: use valid_score (GPU) when available,
        # fall back to NaN for genes that errored before validation.
        if item.valid_score is not None:
            valid_values = _metrics_from_score(item.valid_score)
        else:
            valid_values = {key: np.nan for key in _metrics_from_score(_empty_score())}
        row.update({f"valid_{key}": value for key, value in valid_values.items()})
        row["passed_validation"] = item.passed_validation

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Prefer validation columns when available; fall back to train.
        train_cols = [f"train_{name}" for name in NSGA_OBJECTIVE_MODES[objective_mode]]
        valid_cols = [f"valid_{name}" for name in NSGA_OBJECTIVE_MODES[objective_mode]]
        sort_cols = (
            valid_cols if all(c in df.columns for c in valid_cols) else train_cols
        )
        existing = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(existing, ascending=[False] * len(existing)).reset_index(drop=True)
    return df


def selected_behavior_rank_table(
    evaluated: list[EvaluatedBehaviorGene],
    *,
    objective_mode: str = NSGA_MODE_RIR_LONG_RIR_NDCG,
) -> pd.DataFrame:
    """NSGA-II diagnostic table for an evaluated behavior population."""

    if not evaluated:
        return pd.DataFrame()
    if objective_mode not in NSGA_OBJECTIVE_MODES:
        raise ValueError(
            f"objective_mode must be one of {tuple(NSGA_OBJECTIVE_MODES)}"
        )
    objective_names = NSGA_OBJECTIVE_MODES[objective_mode]
    objectives = [item.objectives(objective_mode) for item in evaluated]
    table = rank_table(objectives)
    table = table.rename(
        columns={
            f"objective_{idx}": f"objective_{name}"
            for idx, name in enumerate(objective_names)
        }
    )
    table.insert(0, "objective_mode", objective_mode)
    table["expression"] = [describe_gene(evaluated[idx].gene) for idx in table.index]
    table["formula"] = [describe_gene_formula(evaluated[idx].gene) for idx in table.index]
    table["train_direction"] = [
        int(evaluated[idx].train_score.direction)
        for idx in table.index
    ]
    for metric in _metrics_from_score(_empty_score()):
        table[f"train_{metric}"] = [
            _metrics_from_score(evaluated[idx].train_score)[metric]
            for idx in table.index
        ]
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

    evaluated_behavior_to_frame(
        result.history,
        objective_mode=result.config.nsga_objective_mode,
    ).to_csv(history_path, index=False, encoding="utf-8-sig")
    evaluated_behavior_to_frame(
        result.final_population,
        objective_mode=result.config.nsga_objective_mode,
    ).to_csv(final_path, index=False, encoding="utf-8-sig")
    selected_behavior_rank_table(
        result.final_population,
        objective_mode=result.config.nsga_objective_mode,
    ).to_csv(rank_path, index=True, encoding="utf-8-sig")
    config_path.write_text(json.dumps(asdict(result.config), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "history": history_path,
        "final_population": final_path,
        "rank_table": rank_path,
        "config": config_path,
    }
