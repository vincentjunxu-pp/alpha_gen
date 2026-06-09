from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .context import CudaFactorContext
from .evaluation_metrics import empty_factor_score
from .generator import (
    ProgramGeneratorConfig,
    crossover_programs,
    mutate_program,
    random_program,
)
from .nsga2 import nsga2_select, rank_table
from .program import Program, program_key
from .scorer import ProgramScorer, ScoredProgram, ScorerConfig


@dataclass(frozen=True)
class FreeGPSearchConfig:
    population_size: int = 80
    generations: int = 3
    crossover_probability: float = 0.85
    mutation_probability: float = 0.25
    random_seed: int = 1
    min_coverage: float = 0.30
    generator_config: ProgramGeneratorConfig = field(default_factory=ProgramGeneratorConfig)
    scorer_config: ScorerConfig = field(default_factory=ScorerConfig)
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be in [0, 1]")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0, 1]")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")


@dataclass(frozen=True)
class ValidationCriteria:
    min_abs_rank_ic: float = 0.02
    min_ic_win_rate: float = 0.55
    min_coverage: float = 0.30
    min_ndcg_at_k: float = 0.0


@dataclass
class EvaluatedProgram:
    program: Program
    train_score: ScoredProgram
    generation: int
    valid_score: ScoredProgram | None = None
    passed_validation: bool | None = None

    @property
    def objectives(self) -> tuple[float, float, float]:
        return self.train_score.objectives

    @property
    def error(self) -> str:
        return self.train_score.error

    def to_dict(self, *, include_program_json: bool = False) -> dict[str, object]:
        row: dict[str, object] = {
            "generation": self.generation,
            **{f"train_{key}": value for key, value in self.train_score.to_dict().items()},
        }
        if self.valid_score is not None:
            row.update({f"valid_{key}": value for key, value in self.valid_score.to_dict().items()})
            row["passed_validation"] = self.passed_validation
        if include_program_json:
            row["program_json"] = self.program.to_json()
        return row


@dataclass(frozen=True)
class SearchResult:
    config: FreeGPSearchConfig
    final_population: list[EvaluatedProgram]
    history: list[EvaluatedProgram]


def _normalize_dates(dates: Iterable[object] | pd.DatetimeIndex | None) -> pd.DatetimeIndex | None:
    if dates is None:
        return None
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = ""):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _replace_score(scored: ScoredProgram, *, error: str) -> ScoredProgram:
    return ScoredProgram(
        program=scored.program,
        score=empty_factor_score(coverage=scored.score.coverage),
        expression=scored.expression,
        size=scored.size,
        depth=scored.depth,
        complexity_cost=scored.complexity_cost,
        error=error,
    )


def _score_one(
    program: Program,
    scorer: ProgramScorer,
    *,
    dates: pd.DatetimeIndex | None,
    generation: int,
    min_coverage: float,
    score_cache: dict[tuple[object, ...], ScoredProgram],
) -> EvaluatedProgram:
    key = program_key(program)
    scored = score_cache.get(key)
    if scored is None:
        scored = scorer.score_program(program, dates=dates, raise_errors=False)
        if not scored.error and scored.score.coverage < min_coverage:
            scored = _replace_score(
                scored,
                error=f"coverage below threshold: {scored.score.coverage:.4f} < {min_coverage:.4f}",
            )
        score_cache[key] = scored
    return EvaluatedProgram(program=program, train_score=scored, generation=generation)


def _deduplicate_evaluated(evaluated: Iterable[EvaluatedProgram]) -> list[EvaluatedProgram]:
    seen: set[tuple[object, ...]] = set()
    output: list[EvaluatedProgram] = []
    for item in evaluated:
        key = program_key(item.program)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _initialize_population(
    fields: Sequence[str],
    config: FreeGPSearchConfig,
    rng: random.Random,
) -> list[Program]:
    programs: list[Program] = []
    seen: set[tuple[object, ...]] = set()
    max_attempts = max(config.population_size * config.generator_config.max_attempts, config.population_size)
    attempts = 0
    while len(programs) < config.population_size and attempts < max_attempts:
        program = random_program(fields, config=config.generator_config, random_state=rng)
        key = program_key(program)
        if key not in seen:
            seen.add(key)
            programs.append(program)
        attempts += 1
    if not programs:
        raise RuntimeError("failed to initialize any valid GP program")
    return programs


def _refill_programs(
    programs: list[Program],
    fields: Sequence[str],
    config: FreeGPSearchConfig,
    rng: random.Random,
) -> list[Program]:
    output = list(programs)
    seen = {program_key(program) for program in output}
    attempts = 0
    max_attempts = max(config.population_size * config.generator_config.max_attempts, config.population_size)
    while len(output) < config.population_size and attempts < max_attempts:
        program = random_program(fields, config=config.generator_config, random_state=rng)
        key = program_key(program)
        if key not in seen:
            seen.add(key)
            output.append(program)
        attempts += 1
    return output


def _make_offspring(
    parents: list[Program],
    fields: Sequence[str],
    config: FreeGPSearchConfig,
    rng: random.Random,
) -> list[Program]:
    if not parents:
        raise ValueError("parents must not be empty")
    offspring: list[Program] = []
    while len(offspring) < config.population_size:
        if len(parents) >= 2 and rng.random() < config.crossover_probability:
            left, right = rng.sample(parents, 2)
            child = crossover_programs(left, right, fields, config=config.generator_config, random_state=rng)
        else:
            child = parents[rng.randrange(len(parents))]

        if rng.random() < config.mutation_probability:
            child = mutate_program(child, fields, config=config.generator_config, random_state=rng)
        offspring.append(child)
    return offspring


def _evaluate_population(
    programs: Sequence[Program],
    scorer: ProgramScorer,
    *,
    dates: pd.DatetimeIndex | None,
    generation: int,
    min_coverage: float,
    score_cache: dict[tuple[object, ...], ScoredProgram],
    show_progress: bool,
) -> list[EvaluatedProgram]:
    iterator = _progress_iter(
        programs,
        enabled=show_progress,
        total=len(programs),
        desc=f"score generation {generation}",
    )
    return [
        _score_one(
            program,
            scorer,
            dates=dates,
            generation=generation,
            min_coverage=min_coverage,
            score_cache=score_cache,
        )
        for program in iterator
    ]


def _select_population(evaluated: list[EvaluatedProgram], n_select: int) -> list[EvaluatedProgram]:
    unique = _deduplicate_evaluated(evaluated)
    complexities = [item.train_score.complexity_cost for item in unique]
    selected_idx = nsga2_select(
        [item.objectives for item in unique],
        n_select,
        complexities=complexities,
    )
    return [unique[idx] for idx in selected_idx]


def _ensure_population_size(
    evaluated: list[EvaluatedProgram],
    fields: Sequence[str],
    scorer: ProgramScorer,
    config: FreeGPSearchConfig,
    rng: random.Random,
    *,
    dates: pd.DatetimeIndex | None,
    generation: int,
    score_cache: dict[tuple[object, ...], ScoredProgram],
) -> tuple[list[EvaluatedProgram], list[EvaluatedProgram]]:
    if len(evaluated) >= config.population_size:
        return evaluated[: config.population_size], []

    base_programs = [item.program for item in evaluated]
    refilled = _refill_programs(base_programs, fields, config, rng)
    extra_programs = refilled[len(base_programs):]
    if not extra_programs:
        return evaluated, []

    extra_evaluated = _evaluate_population(
        extra_programs,
        scorer,
        dates=dates,
        generation=generation,
        min_coverage=config.min_coverage,
        score_cache=score_cache,
        show_progress=config.show_progress,
    )
    return (evaluated + extra_evaluated)[: config.population_size], extra_evaluated


def run_free_gp_search(
    ctx: CudaFactorContext,
    *,
    train_dates: Iterable[object] | pd.DatetimeIndex | None = None,
    candidate_fields: Sequence[str] | None = None,
    config: FreeGPSearchConfig | None = None,
    scorer: ProgramScorer | None = None,
) -> SearchResult:
    config = config or FreeGPSearchConfig()
    rng = random.Random(config.random_seed)
    fields = tuple(candidate_fields or ctx.searchable_fields)
    if not fields:
        raise ValueError("candidate_fields or ctx.searchable_fields must contain at least one field")
    dates = _normalize_dates(train_dates)
    scorer = scorer or ProgramScorer(ctx, config.scorer_config)

    population = _initialize_population(fields, config, rng)
    score_cache: dict[tuple[object, ...], ScoredProgram] = {}
    evaluated = _evaluate_population(
        population,
        scorer,
        dates=dates,
        generation=0,
        min_coverage=config.min_coverage,
        score_cache=score_cache,
        show_progress=config.show_progress,
    )
    history: list[EvaluatedProgram] = list(evaluated)
    evaluated = _select_population(evaluated, config.population_size)
    evaluated, extra = _ensure_population_size(
        evaluated,
        fields,
        scorer,
        config,
        rng,
        dates=dates,
        generation=0,
        score_cache=score_cache,
    )
    history.extend(extra)

    generations = _progress_iter(
        range(1, config.generations + 1),
        enabled=config.show_progress,
        total=config.generations,
        desc="free GP generations",
    )
    for generation in generations:
        parent_programs = [item.program for item in evaluated]
        offspring = _make_offspring(parent_programs, fields, config, rng)
        offspring = _refill_programs(offspring, fields, config, rng)
        evaluated_offspring = _evaluate_population(
            offspring,
            scorer,
            dates=dates,
            generation=generation,
            min_coverage=config.min_coverage,
            score_cache=score_cache,
            show_progress=config.show_progress,
        )
        history.extend(evaluated_offspring)
        evaluated = _select_population(evaluated + evaluated_offspring, config.population_size)
        evaluated, extra = _ensure_population_size(
            evaluated,
            fields,
            scorer,
            config,
            rng,
            dates=dates,
            generation=generation,
            score_cache=score_cache,
        )
        history.extend(extra)

    return SearchResult(config=config, final_population=evaluated, history=history)


def validate_population(
    evaluated_population: list[EvaluatedProgram],
    ctx: CudaFactorContext,
    valid_dates: Iterable[object] | pd.DatetimeIndex,
    *,
    config: FreeGPSearchConfig | None = None,
    criteria: ValidationCriteria | None = None,
    scorer: ProgramScorer | None = None,
) -> list[EvaluatedProgram]:
    config = config or FreeGPSearchConfig()
    criteria = criteria or ValidationCriteria()
    scorer = scorer or ProgramScorer(ctx, config.scorer_config)
    dates = _normalize_dates(valid_dates)

    output: list[EvaluatedProgram] = []
    for item in evaluated_population:
        valid_score = scorer.score_program(
            item.program,
            dates=dates,
            direction=item.train_score.score.direction,
            raise_errors=False,
        )
        score = valid_score.score
        passed = (
            not valid_score.error
            and score.abs_rank_ic >= criteria.min_abs_rank_ic
            and score.ic_win_rate >= criteria.min_ic_win_rate
            and score.coverage >= criteria.min_coverage
            and score.ndcg_at_k >= criteria.min_ndcg_at_k
        )
        item.valid_score = valid_score
        item.passed_validation = passed
        output.append(item)
    return output


def evaluated_to_frame(evaluated: Sequence[EvaluatedProgram], *, include_program_json: bool = False) -> pd.DataFrame:
    rows = [item.to_dict(include_program_json=include_program_json) for item in evaluated]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_cols = [col for col in ("train_rank_ic_ir", "train_ndcg_at_k", "train_neutralized_icir") if col in frame]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    return frame


def selected_rank_table(evaluated: Sequence[EvaluatedProgram]) -> pd.DataFrame:
    complexities = [item.train_score.complexity_cost for item in evaluated]
    table = rank_table(
        [item.objectives for item in evaluated],
        complexities=complexities,
    )
    table["expression"] = [evaluated[idx].train_score.expression for idx in table.index]
    return table


def export_search_result(
    result: SearchResult,
    output_dir: str | Path,
    *,
    prefix: str = "free_gp_cuda",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_path = output_path / f"{prefix}_final_population.csv"
    history_path = output_path / f"{prefix}_history.csv"
    config_path = output_path / f"{prefix}_config.json"

    evaluated_to_frame(result.final_population, include_program_json=True).to_csv(final_path, index=False)
    evaluated_to_frame(result.history, include_program_json=True).to_csv(history_path, index=False)
    config_path.write_text(
        json.dumps(asdict(result.config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "final_population": final_path,
        "history": history_path,
        "config": config_path,
    }


__all__ = [
    "FreeGPSearchConfig",
    "ValidationCriteria",
    "EvaluatedProgram",
    "SearchResult",
    "run_free_gp_search",
    "validate_population",
    "evaluated_to_frame",
    "selected_rank_table",
    "export_search_result",
]
