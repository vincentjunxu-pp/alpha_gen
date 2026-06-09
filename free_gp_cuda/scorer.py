from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import pandas as pd
import torch

from .context import CudaFactorContext
from .evaluation_metrics import FactorScore, empty_factor_score, evaluate_factor_tensor
from .evaluator import ProgramEvaluator
from .neutralizer import neutralize_factor_tensor
from .program import Program, program_expression


FactorView = Literal["raw", "neutralized"]


def _normalize_fields(fields: Sequence[str] | str | None) -> tuple[str, ...]:
    if fields is None:
        return ()
    if isinstance(fields, str):
        raw_values = [fields]
    else:
        raw_values = list(fields)
    return tuple(
        dict.fromkeys(
            text
            for field in raw_values
            if field is not None
            for text in [str(field).strip()]
            if text
        )
    )


def _normalize_dates(dates: Iterable[object] | pd.DatetimeIndex | None) -> pd.DatetimeIndex | None:
    if dates is None:
        return None
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


@dataclass(frozen=True)
class ScorerConfig:
    ndcg_k: int | None = None
    ndcg_top_fraction: float = 0.10
    n_groups: int = 10
    direction: int | None = None
    min_cross_section_size: int = 2
    style_fields: Sequence[str] | str = ()
    neutralize_industry: bool = True
    neutralize_styles: bool = True
    style_corr_threshold: float = 0.30
    style_max_fields: int = 2
    standardize_styles: bool = True
    mask_inputs_by_tradeable: bool = True
    mask_factor_by_tradeable: bool = True
    tradeable_only: bool = True
    use_subtree_cache: bool = True

    def __post_init__(self) -> None:
        if self.ndcg_k is not None and self.ndcg_k < 1:
            raise ValueError("ndcg_k must be positive or None")
        if not 0 < self.ndcg_top_fraction <= 1:
            raise ValueError("ndcg_top_fraction must be in (0, 1]")
        if self.n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if self.direction is not None and self.direction not in {-1, 1}:
            raise ValueError("direction must be -1, 1, or None")
        if self.min_cross_section_size < 2:
            raise ValueError("min_cross_section_size must be at least 2 (pandas corr minimum)")
        if self.style_corr_threshold < 0:
            raise ValueError("style_corr_threshold must be non-negative")
        if self.style_max_fields <= 0:
            raise ValueError("style_max_fields must be positive")
        object.__setattr__(self, "style_fields", _normalize_fields(self.style_fields))


@dataclass(frozen=True)
class ScoredProgram:
    program: Program
    score: FactorScore
    expression: str
    size: int
    depth: int
    complexity_cost: float
    error: str = ""

    @property
    def objectives(self) -> tuple[float, float, float]:
        return self.score.objectives

    def to_dict(self, *, include_program_json: bool = False) -> dict[str, float | int | str]:
        output: dict[str, float | int | str] = {
            "expression": self.expression,
            "size": self.size,
            "depth": self.depth,
            "complexity_cost": self.complexity_cost,
            "error": self.error,
            **self.score.to_dict(),
        }
        if include_program_json:
            output["program_json"] = self.program.to_json()
        return output


@dataclass
class ProgramScorer:
    ctx: CudaFactorContext
    config: ScorerConfig = field(default_factory=ScorerConfig)
    evaluator: ProgramEvaluator | None = None
    _style_cache: torch.Tensor | None = field(default=None, init=False)
    _industry_cache: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.evaluator is None:
            self.evaluator = ProgramEvaluator(
                self.ctx,
                mask_inputs_by_tradeable=self.config.mask_inputs_by_tradeable,
                tradeable_only=self.config.tradeable_only,
                use_subtree_cache=self.config.use_subtree_cache,
            )

    def clear_cache(self) -> None:
        self._style_cache = None
        self._industry_cache = None
        if self.evaluator is not None:
            self.evaluator.clear_cache()

    def _style_tensors(self) -> torch.Tensor | None:
        fields = tuple(self.config.style_fields)
        if not fields:
            return None
        if self._style_cache is not None:
            return self._style_cache
        tensors = [self.ctx.get_field(field) for field in fields]
        self._style_cache = torch.stack(tensors, dim=2)
        return self._style_cache

    def _industry_codes(self) -> torch.Tensor | None:
        if not self.config.neutralize_industry:
            return None
        if self._industry_cache is None:
            self._industry_cache = self.ctx.industry_codes()
        return self._industry_cache

    def neutralize_factor(
        self,
        factor: torch.Tensor,
        *,
        dates: Iterable[object] | pd.DatetimeIndex | None = None,
    ) -> torch.Tensor:
        date_index = _normalize_dates(dates)
        if date_index is not None and tuple(factor.shape) == self.ctx.shape:
            factor = self.ctx.take_dates(factor, date_index)
        tradeable = self.ctx.take_dates(self.ctx.tradeable(), date_index)
        industry_codes = self._industry_codes()
        if industry_codes is not None:
            industry_codes = self.ctx.take_dates(industry_codes, date_index)
        style_tensors = self._style_tensors()
        if style_tensors is not None:
            style_tensors = self.ctx.take_dates(style_tensors, date_index)
        return neutralize_factor_tensor(
            factor,
            industry_codes=industry_codes,
            style_tensors=style_tensors,
            mask=tradeable,
            style_fields=tuple(self.config.style_fields),
            neutralize_industry=self.config.neutralize_industry,
            neutralize_styles=self.config.neutralize_styles,
            standardize_styles=self.config.standardize_styles,
        ).residual_factor

    def score_factor(
        self,
        factor: torch.Tensor,
        *,
        dates: Iterable[object] | pd.DatetimeIndex | None = None,
        direction: int | None = None,
    ) -> FactorScore:
        date_index = _normalize_dates(dates)
        if date_index is not None and tuple(factor.shape) == self.ctx.shape:
            factor = self.ctx.take_dates(factor, date_index)
        label = self.ctx.take_dates(self.ctx.label(), date_index)
        tradeable = self.ctx.take_dates(self.ctx.tradeable(), date_index)
        industry_codes = self._industry_codes()
        if industry_codes is not None:
            industry_codes = self.ctx.take_dates(industry_codes, date_index)
        style_tensors = self._style_tensors()
        if style_tensors is not None:
            style_tensors = self.ctx.take_dates(style_tensors, date_index)
        return evaluate_factor_tensor(
            factor,
            label,
            tradeable=tradeable,
            industry_codes=industry_codes,
            style_tensors=style_tensors,
            style_fields=tuple(self.config.style_fields),
            neutralize_industry=self.config.neutralize_industry,
            neutralize_styles=self.config.neutralize_styles,
            standardize_styles=self.config.standardize_styles,
            mask_factor_by_tradeable=self.config.mask_factor_by_tradeable,
            ndcg_k=self.config.ndcg_k,
            ndcg_top_fraction=self.config.ndcg_top_fraction,
            n_groups=self.config.n_groups,
            direction=self.config.direction if direction is None else direction,
            min_cross_section_size=self.config.min_cross_section_size,
            style_corr_threshold=self.config.style_corr_threshold,
            style_max_fields=self.config.style_max_fields,
        )

    def factor_values(
        self,
        program: Program,
        *,
        view: FactorView = "raw",
        dates: Iterable[object] | pd.DatetimeIndex | None = None,
        available_fields: Iterable[str] | None = None,
        validate: bool = True,
        clear_eval_cache: bool = True,
    ) -> torch.Tensor:
        if view not in {"raw", "neutralized"}:
            raise ValueError("view must be 'raw' or 'neutralized'")
        assert self.evaluator is not None
        date_index = _normalize_dates(dates)
        factor = self.evaluator.evaluate(
            program,
            dates=date_index,
            validate=validate,
            available_fields=available_fields,
            clear_cache=clear_eval_cache,
        )
        if view == "raw":
            return factor
        return self.neutralize_factor(factor, dates=date_index)

    def score_program(
        self,
        program: Program,
        *,
        dates: Iterable[object] | pd.DatetimeIndex | None = None,
        direction: int | None = None,
        available_fields: Iterable[str] | None = None,
        validate: bool = True,
        raise_errors: bool = False,
        clear_eval_cache: bool = True,
    ) -> ScoredProgram:
        expression = self._safe_expression(program)
        try:
            assert self.evaluator is not None
            date_index = _normalize_dates(dates)
            factor = self.evaluator.evaluate(
                program,
                dates=date_index,
                validate=validate,
                available_fields=available_fields,
                clear_cache=clear_eval_cache,
            )
            score = self.score_factor(factor, dates=date_index, direction=direction)
            error = ""
        except Exception as exc:
            if raise_errors:
                raise
            score = empty_factor_score()
            error = str(exc)
        return ScoredProgram(
            program=program,
            score=score,
            expression=expression,
            size=program.size,
            depth=program.depth,
            complexity_cost=program.complexity_cost,
            error=error,
        )

    def score_programs(
        self,
        programs: Iterable[Program],
        *,
        dates: Iterable[object] | pd.DatetimeIndex | None = None,
        available_fields: Iterable[str] | None = None,
        validate: bool = True,
        raise_errors: bool = False,
    ) -> list[ScoredProgram]:
        date_index = _normalize_dates(dates)
        return [
            self.score_program(
                program,
                dates=date_index,
                available_fields=available_fields,
                validate=validate,
                raise_errors=raise_errors,
                clear_eval_cache=True,
            )
            for program in programs
        ]

    @staticmethod
    def _safe_expression(program: Program) -> str:
        try:
            return program_expression(program)
        except Exception:
            return "<invalid program>"


def score_factor(
    factor: torch.Tensor,
    ctx: CudaFactorContext,
    *,
    dates: Iterable[object] | pd.DatetimeIndex | None = None,
    config: ScorerConfig | None = None,
    direction: int | None = None,
) -> FactorScore:
    scorer = ProgramScorer(ctx, config=config or ScorerConfig())
    return scorer.score_factor(factor, dates=dates, direction=direction)


def factor_values(
    program: Program,
    ctx: CudaFactorContext,
    *,
    view: FactorView = "raw",
    dates: Iterable[object] | pd.DatetimeIndex | None = None,
    config: ScorerConfig | None = None,
    available_fields: Iterable[str] | None = None,
    validate: bool = True,
) -> torch.Tensor:
    scorer = ProgramScorer(ctx, config=config or ScorerConfig())
    return scorer.factor_values(
        program,
        view=view,
        dates=dates,
        available_fields=available_fields,
        validate=validate,
    )


def score_program(
    program: Program,
    ctx: CudaFactorContext,
    *,
    dates: Iterable[object] | pd.DatetimeIndex | None = None,
    config: ScorerConfig | None = None,
    direction: int | None = None,
    available_fields: Iterable[str] | None = None,
    validate: bool = True,
    raise_errors: bool = False,
) -> ScoredProgram:
    scorer = ProgramScorer(ctx, config=config or ScorerConfig())
    return scorer.score_program(
        program,
        dates=dates,
        direction=direction,
        available_fields=available_fields,
        validate=validate,
        raise_errors=raise_errors,
    )


__all__ = [
    "FactorView",
    "ScorerConfig",
    "ScoredProgram",
    "ProgramScorer",
    "factor_values",
    "score_factor",
    "score_program",
]
