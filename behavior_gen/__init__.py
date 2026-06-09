"""Behavior-finance genetic search package."""

from .ga import (
    BehaviorGAConfig,
    BehaviorSearchResult,
    BehaviorValidationCriteria,
    EvaluatedBehaviorGene,
    evaluated_behavior_to_frame,
    export_behavior_search_result,
    run_behavior_ga_search,
    validate_behavior_population,
)
from .gene import (
    BehaviorFieldRule,
    BehaviorGene,
    ConditionGene,
    ModeSpec,
    SlotGene,
    SlotSpec,
    describe_gene,
    load_behavior_field_rules,
    validate_gene,
)
from .sampler import BehaviorSamplerConfig, random_gene, random_population
from .torch_backend import BehaviorTorchContext, calculate_behavior_factor_tensor, score_behavior_factor_tensor


__all__ = [
    "BehaviorFieldRule",
    "BehaviorGAConfig",
    "BehaviorGene",
    "BehaviorSamplerConfig",
    "BehaviorSearchResult",
    "BehaviorTorchContext",
    "BehaviorValidationCriteria",
    "ConditionGene",
    "EvaluatedBehaviorGene",
    "ModeSpec",
    "SlotGene",
    "SlotSpec",
    "calculate_behavior_factor_tensor",
    "describe_gene",
    "evaluated_behavior_to_frame",
    "export_behavior_search_result",
    "load_behavior_field_rules",
    "random_gene",
    "random_population",
    "run_behavior_ga_search",
    "score_behavior_factor_tensor",
    "validate_behavior_population",
    "validate_gene",
]
