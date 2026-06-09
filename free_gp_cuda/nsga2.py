from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NSGA2Rank:
    index: int
    front: int
    crowding_distance: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "front": self.front,
            "crowding_distance": self.crowding_distance,
        }


def as_objective_array(objectives: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(objectives, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"objectives must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("objectives must contain at least one individual")
    if arr.shape[1] == 0:
        raise ValueError("objectives must contain at least one objective")
    return np.where(np.isfinite(arr), arr, -np.inf)


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    left_arr = np.where(np.isfinite(left_arr), left_arr, -np.inf)
    right_arr = np.where(np.isfinite(right_arr), right_arr, -np.inf)
    return bool(np.all(left_arr >= right_arr) and np.any(left_arr > right_arr))


def fast_non_dominated_sort(objectives: Sequence[Sequence[float]]) -> list[list[int]]:
    obj = as_objective_array(objectives)
    n = obj.shape[0]
    dominated_by_count = np.zeros(n, dtype=int)
    dominates_list: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(obj[i], obj[j]):
                dominates_list[i].append(j)
                dominated_by_count[j] += 1
            elif dominates(obj[j], obj[i]):
                dominates_list[j].append(i)
                dominated_by_count[i] += 1

    fronts: list[list[int]] = [np.where(dominated_by_count == 0)[0].tolist()]
    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for individual in fronts[current]:
            for dominated in dominates_list[individual]:
                dominated_by_count[dominated] -= 1
                if dominated_by_count[dominated] == 0:
                    next_front.append(dominated)
        if next_front:
            fronts.append(next_front)
        current += 1
    return fronts


def crowding_distance(objectives: Sequence[Sequence[float]], front: Sequence[int]) -> dict[int, float]:
    obj = as_objective_array(objectives)
    front = list(front)
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    front_obj = obj[front]
    distances = np.zeros(len(front), dtype=float)
    for objective_id in range(front_obj.shape[1]):
        values = front_obj[:, objective_id]
        order = np.argsort(values, kind="mergesort")
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")

        value_min = values[order[0]]
        value_max = values[order[-1]]
        if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max == value_min:
            continue
        scale = value_max - value_min
        for pos in range(1, len(front) - 1):
            if np.isinf(distances[order[pos]]):
                continue
            prev_value = values[order[pos - 1]]
            next_value = values[order[pos + 1]]
            distances[order[pos]] += (next_value - prev_value) / scale
    return {idx: float(distance) for idx, distance in zip(front, distances)}


def rank_population(
    objectives: Sequence[Sequence[float]],
    complexities: Sequence[float] | None = None,
) -> list[NSGA2Rank]:
    ranks: list[NSGA2Rank] = []
    for front_id, front in enumerate(fast_non_dominated_sort(objectives)):
        distances = crowding_distance(objectives, front)
        for idx in front:
            ranks.append(
                NSGA2Rank(
                    index=idx,
                    front=front_id,
                    crowding_distance=distances[idx],
                )
            )
    if complexities is not None:
        ranks.sort(key=lambda item: (item.front, complexities[item.index], -item.crowding_distance, item.index))
    else:
        ranks.sort(key=lambda item: (item.front, -item.crowding_distance, item.index))
    return ranks


def nsga2_select(
    objectives: Sequence[Sequence[float]],
    n_select: int,
    complexities: Sequence[float] | None = None,
) -> list[int]:
    """NSGA-II elite selection with optional complexity-aware intra-front ordering.

    When ``complexities`` is provided, individuals within the same Pareto front
    are sorted by **lower complexity first**, with crowding distance acting as a
    secondary tiebreaker. This biases evolution toward simpler expressions without
    altering the dominance (Pareto) structure.
    """
    if n_select <= 0:
        raise ValueError("n_select must be positive")
    obj = as_objective_array(objectives)
    n_select = min(n_select, obj.shape[0])

    if complexities is not None:
        if len(complexities) != obj.shape[0]:
            raise ValueError(
                f"complexities length {len(complexities)} does not match objectives length {obj.shape[0]}"
            )

    selected: list[int] = []
    for front in fast_non_dominated_sort(obj):
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
            continue
        distances = crowding_distance(obj, front)
        remaining = n_select - len(selected)
        if complexities is not None:
            # Within the same front: lower complexity first, then higher crowding distance
            selected.extend(
                sorted(
                    front,
                    key=lambda idx: (complexities[idx], -distances[idx], idx),
                )[:remaining]
            )
        else:
            selected.extend(
                sorted(front, key=lambda idx: (-distances[idx], idx))[:remaining]
            )
        break
    return selected


def rank_table(
    objectives: Sequence[Sequence[float]],
    complexities: Sequence[float] | None = None,
) -> pd.DataFrame:
    obj = as_objective_array(objectives)
    ranks = rank_population(obj, complexities=complexities)
    rank_df = pd.DataFrame([rank.to_dict() for rank in ranks]).set_index("index")
    objective_df = pd.DataFrame(obj, columns=[f"objective_{i}" for i in range(obj.shape[1])])
    if complexities is not None:
        rank_df["complexity_cost"] = [complexities[i] for i in rank_df.index]
    return objective_df.join(rank_df).sort_values(["front", "crowding_distance"], ascending=[True, False])


__all__ = [
    "NSGA2Rank",
    "as_objective_array",
    "dominates",
    "fast_non_dominated_sort",
    "crowding_distance",
    "rank_population",
    "nsga2_select",
    "rank_table",
]
