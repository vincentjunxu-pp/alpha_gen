"""GA main loop for typed tree GP behavior-finance factor search.

Reuses the evaluation infrastructure from behavior_gen (BehaviorTorchContext,
evaluate_factor_tensor) and NSGA-II selection from core.nsga2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from alpha_gen.core.metrics import FactorScore, factor_group_pnl
from alpha_gen.core.nsga2 import nsga2_select, rank_table

from .torch_backend import (
    calculate_tree_factor_tensor,
    score_tree_factor_tensor,
)
from .typed_sampler import (
    TypedTreeSamplerConfig,
    crossover_typed_trees,
    mutate_typed_tree,
    random_typed_tree,
)
from .typed_tree import (
    TemplateTree,
    tree_depth,
    tree_key,
    tree_size,
)


def _progress_iter(iterable, *, enabled: bool, total: int | None = None, desc: str = "", leave: bool = False):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TreeGAConfig:
    """Search parameters for typed tree GP."""

    population_size: int = 80
    generations: int = 3
    crossover_prob: float = 0.85
    mutation_prob: float = 0.25
    random_seed: int = 20260529
    ndcg_k: int | None = None
    ndcg_top_fraction: float = 0.10
    min_coverage: float = 0.30
    mode_probabilities: dict[str, float] | None = None
    sampler_config: TypedTreeSamplerConfig = field(default_factory=TypedTreeSamplerConfig)
    neutralize_size: bool = True
    neutralize_industry: bool = True
    size_field: str = "barra_size"
    require_cuda: bool = True
    show_progress: bool = False
    parsimony_coefficient: float = 0.001


@dataclass(frozen=True)
class TreeValidationCriteria:
    min_abs_rank_ic: float = 0.02
    min_ic_win_rate: float = 0.55
    min_top_excess_ann: float = 0.00
    min_coverage: float = 0.30


# ═══════════════════════════════════════════════════════════════
# Evaluated tree container
# ═══════════════════════════════════════════════════════════════


@dataclass
class EvaluatedTree:
    tree: TemplateTree
    train_score: FactorScore
    generation: int
    tree_depth_val: int = 0
    tree_size_val: int = 0
    error: str = ""
    selection_objectives: tuple[float, float, float] | None = None
    valid_score: FactorScore | None = None
    valid_top_excess_ann: float | None = None
    valid_pnl_metrics: dict[str, float | int] | None = None
    passed_validation: bool | None = None

    @property
    def objectives(self) -> tuple[float, float, float]:
        return self.selection_objectives or self.train_score.objectives


@dataclass
class TreeSearchResult:
    config: TreeGAConfig
    final_population: list[EvaluatedTree]
    history: list[EvaluatedTree]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


# Sentinel used for invalid / failed evaluations so NSGA-II always
# discards them regardless of which fields FactorScore.objectives picks.
_INVALID_SCORE = float("-inf")
_INVALID_OBJECTIVES = (_INVALID_SCORE, _INVALID_SCORE, _INVALID_SCORE)


def _empty_score() -> FactorScore:
    """Return a score where all performance metrics are dominated.

    An invalid tree must never outrank a weak-but-valid tree, so we use
    ``-inf`` instead of 0.0.  Non-metric fields (direction, n_ic_obs,
    coverage, barra counts) stay at neutral defaults.
    """
    return FactorScore(
        mean_rank_ic=_INVALID_SCORE,
        abs_rank_ic=_INVALID_SCORE,
        rank_ic_ir=_INVALID_SCORE,
        ic_win_rate=_INVALID_SCORE,
        ndcg_at_k=_INVALID_SCORE,
        direction=1,
        n_ic_obs=0,
        coverage=0.0,
        neutralized_icir=_INVALID_SCORE,
        neutralized_mean_rank_ic=_INVALID_SCORE,
        neutralized_abs_rank_ic=_INVALID_SCORE,
        neutralized_ic_win_rate=_INVALID_SCORE,
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


def _score_direction(tree: TemplateTree) -> int | None:
    return 1 if tree.direction_policy == "fixed" else None


def _selection_objectives_with_parsimony(
    score: FactorScore,
    tree: TemplateTree,
    coefficient: float,
) -> tuple[float, float, float]:
    """Return selection objectives after complexity penalty.

    Keep ``FactorScore`` itself raw for reporting/export.  Only NSGA-II
    objectives should receive the parsimony penalty.
    """
    if coefficient <= 0:
        return score.objectives
    penalty = coefficient * tree_size(tree)
    return (
        score.rank_ic_ir - penalty,
        score.ndcg_at_k - penalty,
        score.neutralized_icir - penalty,
    )


def _deduplicate_trees(trees: Iterable[TemplateTree]) -> list[TemplateTree]:
    seen: set[tuple[object, ...]] = set()
    unique: list[TemplateTree] = []
    for tree in trees:
        key = tree_key(tree)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tree)
    return unique


def _deduplicate_evaluated(evaluated: Iterable[EvaluatedTree]) -> list[EvaluatedTree]:
    seen: set[tuple[object, ...]] = set()
    unique: list[EvaluatedTree] = []
    for item in evaluated:
        key = tree_key(item.tree)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _validate_config(config: TreeGAConfig, ctx) -> None:
    from alpha_gen.behavior_gen.torch_backend import BehaviorTorchContext as BTC

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
        raise RuntimeError(f"Tree GA requires CUDA, got device={ctx.device}")


def _release_cuda_temporaries(ctx) -> None:
    """Return unused CUDA blocks to PyTorch's allocator between tree evals."""
    try:
        device = getattr(ctx, "device", None)
        if getattr(device, "type", None) != "cuda":
            return
        import torch

        torch.cuda.empty_cache()
    except Exception:
        return


# ═══════════════════════════════════════════════════════════════
# Population ops
# ═══════════════════════════════════════════════════════════════


def _try_random_tree(
    field_rules, rng, *, config, mode_probabilities=None,
) -> TemplateTree | None:
    try:
        return random_typed_tree(field_rules, rng, config=config, mode_probabilities=mode_probabilities)
    except Exception:
        return None


def _refill_tree_population(
    trees: list[TemplateTree],
    field_rules,
    target_size: int,
    rng: np.random.Generator,
    *,
    sampler_config: TypedTreeSamplerConfig,
    mode_probabilities=None,
) -> list[TemplateTree]:
    output = _deduplicate_trees(trees)
    attempts = 0
    max_attempts = max(1_000, target_size * 100)
    while len(output) < target_size and attempts < max_attempts:
        sampled = _try_random_tree(field_rules, rng, config=sampler_config, mode_probabilities=mode_probabilities)
        if sampled is not None:
            output = _deduplicate_trees(output + [sampled])
        attempts += 1
    # Exhausted primary attempts — keep trying but always dedup so the
    # effective population size is not silently diluted by duplicates.
    shortage_attempts = 0
    max_shortage_attempts = max(5_000, target_size * 50)
    while len(output) < target_size and shortage_attempts < max_shortage_attempts:
        sampled = _try_random_tree(field_rules, rng, config=sampler_config, mode_probabilities=mode_probabilities)
        if sampled is not None:
            key = tree_key(sampled)
            if not any(tree_key(t) == key for t in output):
                output.append(sampled)
        shortage_attempts += 1
    if len(output) < target_size:
        raise RuntimeError(
            f"failed to refill population to {target_size} "
            f"(got {len(output)} unique trees after {attempts + shortage_attempts} attempts). "
            f"Consider relaxing depth/size constraints or increasing the field candidate pool."
        )
    return output[:target_size]


def make_tree_offspring(
    parents: list[TemplateTree],
    field_rules,
    config: TreeGAConfig,
    rng: np.random.Generator,
) -> list[TemplateTree]:
    if not parents:
        raise ValueError("parents cannot be empty")
    order = rng.permutation(len(parents))
    shuffled = [parents[i] for i in order]
    children: list[TemplateTree] = []

    for i in range(0, len(shuffled), 2):
        left = shuffled[i]
        right = shuffled[(i + 1) % len(shuffled)]
        if rng.random() < config.crossover_prob:
            try:
                child_a, child_b = crossover_typed_trees(
                    left, right, field_rules, rng,
                    config=config.sampler_config,
                    mode_probabilities=config.mode_probabilities,
                )
            except Exception:
                child_a, child_b = left, right
        else:
            child_a, child_b = left, right
        if rng.random() < config.mutation_prob:
            try:
                child_a = mutate_typed_tree(child_a, field_rules, rng, config=config.sampler_config, mode_probabilities=config.mode_probabilities)
            except Exception:
                pass  # keep pre-mutation child_a (which may be a crossover result)
        if rng.random() < config.mutation_prob:
            try:
                child_b = mutate_typed_tree(child_b, field_rules, rng, config=config.sampler_config, mode_probabilities=config.mode_probabilities)
            except Exception:
                pass  # keep pre-mutation child_b (which may be a crossover result)
        children.extend([child_a, child_b])
    return children[: len(parents)]


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════


def evaluate_tree_on_train(
    tree: TemplateTree,
    ctx,
    train_dates: pd.DatetimeIndex,
    config: TreeGAConfig,
    generation: int,
    score_cache: dict[tuple[object, ...], tuple[FactorScore, str, tuple[float, float, float] | None]],
) -> EvaluatedTree:
    key = tree_key(tree)
    if key in score_cache:
        score, error, selection_objectives = score_cache[key]
        return EvaluatedTree(
            tree=tree, train_score=score, generation=generation,
            tree_depth_val=tree_depth(tree), tree_size_val=tree_size(tree),
            error=error, selection_objectives=selection_objectives,
        )
    factor = None
    try:
        factor = calculate_tree_factor_tensor(
            tree, ctx,
            neutralize_size=config.neutralize_size,
            neutralize_industry=config.neutralize_industry,
            size_field=config.size_field,
        )
        score = score_tree_factor_tensor(
            factor, ctx,
            dates=train_dates,
            ndcg_k=config.ndcg_k,
            ndcg_top_fraction=config.ndcg_top_fraction,
            direction=_score_direction(tree),
        )
        selection_objectives = _selection_objectives_with_parsimony(
            score, tree, config.parsimony_coefficient,
        )
        error = ""
        if score.coverage < config.min_coverage:
            coverage = score.coverage
            selection_objectives = _INVALID_OBJECTIVES
            error = f"coverage below threshold: {coverage:.4f}"
    except Exception as exc:
        logger.warning("tree evaluation failed (gen %s, mode=%s): %s: %s", generation, tree.mode, type(exc).__name__, exc)
        score = _empty_score()
        selection_objectives = _INVALID_OBJECTIVES
        error = f"{type(exc).__name__}: {exc}"
    finally:
        del factor
        _release_cuda_temporaries(ctx)
    score_cache[key] = (score, error, selection_objectives)
    return EvaluatedTree(
        tree=tree, train_score=score, generation=generation,
        tree_depth_val=tree_depth(tree), tree_size_val=tree_size(tree),
        error=error, selection_objectives=selection_objectives,
    )


def evaluate_tree_population_on_train(
    trees: list[TemplateTree],
    ctx,
    train_dates: pd.DatetimeIndex,
    config: TreeGAConfig,
    generation: int,
    score_cache: dict[tuple[object, ...], tuple[FactorScore, str, tuple[float, float, float] | None]],
) -> list[EvaluatedTree]:
    unique_trees = _deduplicate_trees(trees)
    evaluated: list[EvaluatedTree] = []
    iterator = _progress_iter(unique_trees, enabled=config.show_progress, total=len(unique_trees), desc=f"evaluate tree gen {generation}")
    for tree in iterator:
        evaluated.append(evaluate_tree_on_train(tree, ctx, train_dates, config, generation, score_cache))
    return evaluated


# ═══════════════════════════════════════════════════════════════
# GA main loop
# ═══════════════════════════════════════════════════════════════


def run_tree_ga_search(
    ctx,
    train_dates: pd.DatetimeIndex,
    config: TreeGAConfig | None = None,
) -> TreeSearchResult:
    config = config or TreeGAConfig()
    _validate_config(config, ctx)
    rng = np.random.default_rng(config.random_seed)
    field_rules = ctx.behavior_field_rules

    # Init population
    population: list[TemplateTree] = []
    init_attempts = 0
    max_init_attempts = max(1_000, config.population_size * 100)
    while len(population) < config.population_size and init_attempts < max_init_attempts:
        sampled = _try_random_tree(field_rules, rng, config=config.sampler_config, mode_probabilities=config.mode_probabilities)
        if sampled is not None:
            population.append(sampled)
        init_attempts += 1
    if not population:
        raise RuntimeError("failed to sample any legal typed tree for initial population")

    population = _refill_tree_population(
        population, field_rules, config.population_size, rng,
        sampler_config=config.sampler_config, mode_probabilities=config.mode_probabilities,
    )

    score_cache: dict[tuple[object, ...], tuple[FactorScore, str, tuple[float, float, float] | None]] = {}
    evaluated_population = evaluate_tree_population_on_train(
        population, ctx, train_dates, config, 0, score_cache,
    )
    history: list[EvaluatedTree] = list(evaluated_population)

    generations = _progress_iter(
        range(1, config.generations + 1), enabled=config.show_progress,
        total=config.generations, desc="tree GA generations", leave=True,
    )
    for generation in generations:
        parent_trees = [item.tree for item in evaluated_population]
        offspring = make_tree_offspring(parent_trees, field_rules, config, rng)
        offspring = _refill_tree_population(
            offspring, field_rules, config.population_size, rng,
            sampler_config=config.sampler_config, mode_probabilities=config.mode_probabilities,
        )
        evaluated_offspring = evaluate_tree_population_on_train(
            offspring, ctx, train_dates, config, generation, score_cache,
        )
        history.extend(evaluated_offspring)
        combined = _deduplicate_evaluated(evaluated_population + evaluated_offspring)
        selected_idx = nsga2_select([item.objectives for item in combined], config.population_size)
        evaluated_population = [combined[idx] for idx in selected_idx]

    return TreeSearchResult(config=config, final_population=evaluated_population, history=history)


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════


def validate_tree_population(
    evaluated_population: list[EvaluatedTree],
    ctx,
    valid_dates: pd.DatetimeIndex,
    criteria: TreeValidationCriteria | None = None,
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
) -> list[EvaluatedTree]:
    criteria = criteria or TreeValidationCriteria()
    output: list[EvaluatedTree] = []
    pnl_n_groups = _n_groups_from_top_fraction(ndcg_top_fraction)
    cache = ctx.cache

    iterator = _progress_iter(evaluated_population, enabled=show_progress, total=len(evaluated_population), desc="validate tree population")
    for item in iterator:
        factor = None
        try:
            factor = calculate_tree_factor_tensor(
                item.tree, ctx,
                neutralize_size=neutralize_size, neutralize_industry=neutralize_industry,
                size_field=size_field,
            )
            valid_score = score_tree_factor_tensor(
                factor, ctx, dates=valid_dates,
                ndcg_k=ndcg_k, ndcg_top_fraction=ndcg_top_fraction,
                direction=item.train_score.direction,
            )
            pnl_result = factor_group_pnl(
                factor=ctx.tensor_to_frame(factor),
                label=cache.label, tradeable=cache.tradeable,
                dates=valid_dates, direction=item.train_score.direction,
                n_groups=pnl_n_groups, label_horizon=label_horizon,
                rebalance_freq=rebalance_freq,
                commission_rate=commission_rate, slippage_rate=slippage_rate,
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
        finally:
            del factor
            _release_cuda_temporaries(ctx)
        output.append(item)
    return output


# ═══════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def describe_tree(tree: TemplateTree) -> str:
    """Create a compact description for logs and result tables."""
    from alpha_gen.behavior_gen.gene import MODE_REGISTRY

    if tree.mode in MODE_REGISTRY:
        mode_text = MODE_REGISTRY[tree.mode].description
    else:
        mode_text = f"unknown mode {tree.mode}"
    slot_text = ", ".join(
        f"{name}={node_expression(node)}"
        for name, node in sorted(tree.slots.items())
    )
    condition_text = ""
    if tree.conditions:
        condition_text = "; if " + " & ".join(
            f"{c.condition_op}({c.unary_op}({c.field}), {c.threshold:.2f})"
            for c in tree.conditions
        )
    return f"{tree.mode}/{tree.combiner}: {mode_text}; slots: {slot_text}{condition_text}"


from .typed_tree import node_expression


def evaluated_tree_to_frame(evaluated: list[EvaluatedTree]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in evaluated:
        tree_dict = item.tree.to_dict()
        row: dict[str, object] = {
            "generation": item.generation,
            "expression": describe_tree(item.tree),
            "error": item.error,
            "tree_mode": tree_dict["mode"],
            "tree_combiner": tree_dict["combiner"],
            "tree_depth": item.tree_depth_val,
            "tree_size": item.tree_size_val,
            "tree_direction_policy": tree_dict["direction_policy"],
            "tree_slots": _json_cell(tree_dict["slots"]),
            "tree_conditions": _json_cell(tree_dict.get("conditions", [])),
            **{f"train_{key}": value for key, value in item.train_score.to_dict().items()},
        }
        row.update({
            "selection_rank_ic_ir": item.objectives[0],
            "selection_ndcg_at_k": item.objectives[1],
            "selection_neutralized_icir": item.objectives[2],
        })
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


def export_tree_search_result(
    result: TreeSearchResult,
    output_dir: str | Path,
    *,
    prefix: str = "tree_gp",
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    history_path = output / f"{prefix}_history.csv"
    final_path = output / f"{prefix}_final_population.csv"
    rank_path = output / f"{prefix}_nsga2_rank.csv"
    config_path = output / f"{prefix}_config.json"

    evaluated_tree_to_frame(result.history).to_csv(history_path, index=False, encoding="utf-8-sig")
    evaluated_tree_to_frame(result.final_population).to_csv(final_path, index=False, encoding="utf-8-sig")

    if result.final_population:
        objectives = [item.objectives for item in result.final_population]
        table = rank_table(objectives)
        table["expression"] = [describe_tree(result.final_population[idx].tree) for idx in table.index]
        table.to_csv(rank_path, index=True, encoding="utf-8-sig")

    config_json = asdict(result.config)
    config_json["sampler_config"] = asdict(result.config.sampler_config)
    config_path.write_text(json.dumps(config_json, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "history": history_path,
        "final_population": final_path,
        "rank_table": rank_path,
        "config": config_path,
    }
