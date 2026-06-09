from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import torch

from .context import CudaFactorContext
from .program import (
    BinaryNode,
    ConstNode,
    FieldNode,
    GateNode,
    Program,
    TreeNode,
    UnaryNode,
    node_key,
    node_output_type,
    validate_program,
)
from .registry import get_operator


def _nan_like(values: torch.Tensor) -> torch.Tensor:
    return torch.full_like(values, float("nan"))


def _apply_tradeable(values: torch.Tensor, tradeable: torch.Tensor) -> torch.Tensor:
    return torch.where(tradeable & torch.isfinite(values), values, _nan_like(values))


def _normalize_mask(values: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.isfinite(values) & (values != 0),
        torch.ones_like(values),
        torch.zeros_like(values),
    )


@dataclass
class ProgramEvaluator:
    """Recursive CUDA evaluator for free GP programs.

    The evaluator is intentionally thin: it validates the tree, resolves fields
    from ``CudaFactorContext``, dispatches registered operators, and optionally
    caches repeated subtrees during one evaluation pass. It does not score or
    evolve programs.
    """

    ctx: CudaFactorContext
    mask_inputs_by_tradeable: bool = True
    tradeable_only: bool = True
    use_subtree_cache: bool = True
    _subtree_cache: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict, init=False)
    _tradeable_cache: torch.Tensor | None = field(default=None, init=False)

    def clear_cache(self) -> None:
        self._subtree_cache.clear()
        self._tradeable_cache = None

    def cache_info(self) -> dict[str, object]:
        return {
            "enabled": self.use_subtree_cache,
            "size": len(self._subtree_cache),
            "keys": tuple(self._subtree_cache.keys()),
        }

    def _tradeable(self) -> torch.Tensor:
        if self._tradeable_cache is None:
            self._tradeable_cache = self.ctx.tradeable()
        return self._tradeable_cache

    def _validate_shape(self, values: torch.Tensor, *, path: str) -> torch.Tensor:
        if tuple(values.shape) != self.ctx.shape:
            raise ValueError(f"{path}: evaluated tensor shape {tuple(values.shape)} does not match context shape {self.ctx.shape}")
        if values.device != self.ctx.device:
            values = values.to(device=self.ctx.device)
        if values.dtype != self.ctx.dtype and values.dtype != torch.bool:
            values = values.to(dtype=self.ctx.dtype)
        return values

    def evaluate_node(self, node: TreeNode) -> torch.Tensor:
        key = node_key(node)
        if self.use_subtree_cache and key in self._subtree_cache:
            return self._subtree_cache[key]

        values = self._evaluate_node_uncached(node)
        values = self._validate_shape(values, path="node")
        if node_output_type(node) == "mask":
            values = _normalize_mask(values)

        if self.use_subtree_cache:
            self._subtree_cache[key] = values
        return values

    def _evaluate_node_uncached(self, node: TreeNode) -> torch.Tensor:
        if isinstance(node, FieldNode):
            values = self.ctx.get_field(node.field)
            if self.mask_inputs_by_tradeable:
                values = _apply_tradeable(values, self._tradeable())
            return values

        if isinstance(node, ConstNode):
            return self.ctx.const_like(node.value)

        if isinstance(node, UnaryNode):
            spec = get_operator(node.op)
            child = self.evaluate_node(node.child)
            return spec.func(child)

        if isinstance(node, BinaryNode):
            spec = get_operator(node.op)
            left = self.evaluate_node(node.left)
            right = self.evaluate_node(node.right)
            return spec.func(left, right)

        if isinstance(node, GateNode):
            spec = get_operator(node.op)
            signal = self.evaluate_node(node.signal)
            mask = self.evaluate_node(node.mask)
            return spec.func(signal, mask)

        raise TypeError(type(node).__name__)

    def evaluate(
        self,
        program: Program,
        *,
        dates: Iterable[object] | pd.DatetimeIndex | None = None,
        validate: bool = True,
        available_fields: Iterable[str] | None = None,
        clear_cache: bool = True,
    ) -> torch.Tensor:
        if clear_cache:
            self.clear_cache()
        if validate:
            if available_fields is None:
                available_fields = self.ctx.searchable_fields
            errors = validate_program(program, available_fields=available_fields)
            if errors:
                raise ValueError("invalid program: " + "; ".join(errors))

        values = self.evaluate_node(program.root)
        if self.tradeable_only:
            values = _apply_tradeable(values, self._tradeable())
        values = self.ctx.take_dates(values, dates)
        return values


def evaluate_node(
    node: TreeNode,
    ctx: CudaFactorContext,
    *,
    mask_inputs_by_tradeable: bool = True,
    use_subtree_cache: bool = True,
) -> torch.Tensor:
    evaluator = ProgramEvaluator(
        ctx,
        mask_inputs_by_tradeable=mask_inputs_by_tradeable,
        tradeable_only=False,
        use_subtree_cache=use_subtree_cache,
    )
    return evaluator.evaluate_node(node)


def evaluate_program(
    program: Program,
    ctx: CudaFactorContext,
    *,
    dates: Iterable[object] | pd.DatetimeIndex | None = None,
    validate: bool = True,
    available_fields: Iterable[str] | None = None,
    mask_inputs_by_tradeable: bool = True,
    tradeable_only: bool = True,
    use_subtree_cache: bool = True,
) -> torch.Tensor:
    evaluator = ProgramEvaluator(
        ctx,
        mask_inputs_by_tradeable=mask_inputs_by_tradeable,
        tradeable_only=tradeable_only,
        use_subtree_cache=use_subtree_cache,
    )
    return evaluator.evaluate(
        program,
        dates=dates,
        validate=validate,
        available_fields=available_fields,
    )


__all__ = [
    "ProgramEvaluator",
    "evaluate_node",
    "evaluate_program",
]
