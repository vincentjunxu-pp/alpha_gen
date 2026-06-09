from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import operators as ops


WINDOWS = (5, 20, 60, 120)
SELECT_COUNTS = (1, 3, 5)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    category: str
    output_type: str
    cost: float
    func: Callable


def _windowed(func: Callable, window: int) -> Callable:
    def wrapped(*args):
        return func(*args, window=window)

    wrapped.__name__ = f"{func.__name__}_{window}"
    return wrapped


def _windowed_select(func: Callable, window: int, n: int) -> Callable:
    def wrapped(*args):
        return func(*args, window=window, n=n)

    wrapped.__name__ = f"{func.__name__}_{window}_n{n}"
    return wrapped


def _register(registry: dict[str, OperatorSpec], spec: OperatorSpec) -> None:
    if spec.name in registry:
        raise ValueError(f"duplicate operator name: {spec.name}")
    registry[spec.name] = spec


def _build_registry() -> dict[str, OperatorSpec]:
    registry: dict[str, OperatorSpec] = {}

    for name, arity, cost in [
        ("neg", 1, 0.20),
        ("abs", 1, 0.20),
        ("sign", 1, 0.30),
        ("slog", 1, 0.40),
        ("sqrt_abs", 1, 0.40),
        ("add", 2, 0.30),
        ("sub", 2, 0.30),
        ("mul", 2, 0.40),
        ("qdiv", 2, 0.60),
    ]:
        _register(
            registry,
            OperatorSpec(
                name=name,
                arity=arity,
                category="pointwise",
                output_type="numeric",
                cost=cost,
                func=getattr(ops, name),
            ),
        )

    for name, arity, cost in [
        ("cs_rank", 1, 1.00),
        ("cs_zscore", 1, 1.00),
        ("cs_demean", 1, 0.80),
        ("cs_winsorize_5pct", 1, 1.20),
        ("cs_resid", 2, 3.00),
    ]:
        _register(
            registry,
            OperatorSpec(
                name=name,
                arity=arity,
                category="cross_section",
                output_type="numeric",
                cost=cost,
                func=getattr(ops, name),
            ),
        )

    for name, arity, cost in [
        ("mask_rank_high_50", 1, 0.90),
        ("mask_rank_high_80", 1, 1.00),
        ("mask_rank_low_20", 1, 1.00),
        ("mask_sign_pos", 1, 0.30),
        ("mask_sign_neg", 1, 0.30),
    ]:
        _register(
            registry,
            OperatorSpec(
                name=name,
                arity=arity,
                category="mask",
                output_type="mask",
                cost=cost,
                func=getattr(ops, name),
            ),
        )

    for name, arity, cost in [
        ("gate_nan", 2, 0.50),
        ("gate_zero", 2, 0.50),
    ]:
        _register(
            registry,
            OperatorSpec(
                name=name,
                arity=arity,
                category="gate",
                output_type="numeric",
                cost=cost,
                func=getattr(ops, name),
            ),
        )

    for window in WINDOWS:
        for name, arity, cost in [
            ("delay", 1, 0.40),
            ("ts_delta", 1, 0.60),
            ("ts_return", 1, 0.70),
            ("ts_mean", 1, 1.00),
            ("ts_median", 1, 1.40),
            ("ts_std", 1, 1.20),
            ("ts_zscore", 1, 1.40),
            ("ts_max_to_min", 1, 1.30),
            ("ts_meanrank", 1, 1.60),
            ("diff_sign", 1, 1.20),
            ("decay_linear", 1, 1.30),
            ("ts_corr", 2, 2.00),
        ]:
            registered_name = f"{name}_{window}"
            _register(
                registry,
                OperatorSpec(
                    name=registered_name,
                    arity=arity,
                    category="time_series",
                    output_type="numeric",
                    cost=cost + window / 120.0,
                    func=_windowed(getattr(ops, name), window),
                ),
            )
        for n in SELECT_COUNTS:
            if n > window:
                continue
            for name, cost in [
                ("rolling_selmean_diff", 2.80),
            ]:
                registered_name = f"{name}_{window}_n{n}"
                _register(
                    registry,
                    OperatorSpec(
                        name=registered_name,
                        arity=2,
                        category="time_series",
                        output_type="numeric",
                        cost=cost + window / 80.0 + n / 10.0,
                        func=_windowed_select(getattr(ops, name), window, n),
                    ),
                )

    return registry


OPERATOR_REGISTRY = _build_registry()


def get_operator(name: str) -> OperatorSpec:
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown operator: {name!r}") from exc


def list_operators(*, category: str | None = None, output_type: str | None = None) -> list[OperatorSpec]:
    specs = list(OPERATOR_REGISTRY.values())
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    if output_type is not None:
        specs = [spec for spec in specs if spec.output_type == output_type]
    return specs


__all__ = [
    "WINDOWS",
    "SELECT_COUNTS",
    "OperatorSpec",
    "OPERATOR_REGISTRY",
    "get_operator",
    "list_operators",
]
