from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NSGA-II selection utilities.
#
# The report uses NSGA-II because the three objectives are not combined into a
# single weighted score. This keeps different kinds of good factors alive:
#   - high |IC| factors,
#   - high IC-win-rate factors,
#   - high NDCG@k factors.
#
# All objectives in this reproduction are maximized.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NSGA2Rank:
    """Rank information for one individual."""

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
    """Convert objective records to a clean 2D float array.

    Non-finite values are treated as the worst possible value, because an
    invalid factor should not dominate a valid one. We use -inf because all
    objectives are maximized.
    """

    arr = np.asarray(objectives, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"objectives must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("objectives must contain at least one individual")
    if arr.shape[1] == 0:
        raise ValueError("objectives must contain at least one objective")
    return np.where(np.isfinite(arr), arr, -np.inf)


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether `left` Pareto-dominates `right`.

    For maximization, left dominates right when it is no worse on every
    objective and strictly better on at least one objective.
    """

    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    left_arr = np.where(np.isfinite(left_arr), left_arr, -np.inf)
    right_arr = np.where(np.isfinite(right_arr), right_arr, -np.inf)
    return bool(np.all(left_arr >= right_arr) and np.any(left_arr > right_arr))


def fast_non_dominated_sort(objectives: Sequence[Sequence[float]]) -> list[list[int]]:
    """Split individuals into Pareto fronts.

    Front 0 is the best non-dominated frontier. Front 1 is obtained after
    removing front 0, and so on. The implementation follows the standard
    NSGA-II bookkeeping but keeps variable names explicit for readability:
    - dominated_by_count[i]: how many individuals dominate i.
    - dominates_list[i]: which individuals are dominated by i.
    """

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
    current_front = 0

    while current_front < len(fronts) and fronts[current_front]:
        next_front: list[int] = []
        for individual in fronts[current_front]:
            for dominated in dominates_list[individual]:
                dominated_by_count[dominated] -= 1
                if dominated_by_count[dominated] == 0:
                    next_front.append(dominated)
        if next_front:
            fronts.append(next_front)
        current_front += 1

    return fronts


def crowding_distance(objectives: Sequence[Sequence[float]], front: Sequence[int]) -> dict[int, float]:
    """Calculate crowding distance for one Pareto front.

    Crowding distance measures how isolated an individual is inside the same
    front. Larger distance means the individual helps preserve diversity more.
    Boundary individuals on every objective receive +inf, as in standard
    NSGA-II.
    """

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

        # The two extremes in each objective are always kept if the partial
        # front needs truncation, so they receive infinite distance.
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


def rank_population(objectives: Sequence[Sequence[float]]) -> list[NSGA2Rank]:
    """Return NSGA-II front and crowding distance for every individual."""

    fronts = fast_non_dominated_sort(objectives)
    ranks: list[NSGA2Rank] = []

    for front_id, front in enumerate(fronts):
        distances = crowding_distance(objectives, front)
        for idx in front:
            ranks.append(
                NSGA2Rank(
                    index=idx,
                    front=front_id,
                    crowding_distance=distances[idx],
                )
            )

    # Deterministic order: better front first, then less crowded first, then
    # original index as a stable tiebreaker.
    ranks.sort(key=lambda item: (item.front, -item.crowding_distance, item.index))
    return ranks


def nsga2_select(objectives: Sequence[Sequence[float]], n_select: int) -> list[int]:
    """Select elite individuals according to NSGA-II.

    Whole fronts are accepted until adding the next front would exceed
    `n_select`. The last partial front is sorted by crowding distance descending.
    """

    if n_select <= 0:
        raise ValueError("n_select must be positive")

    obj = as_objective_array(objectives)
    n_select = min(n_select, obj.shape[0])

    selected: list[int] = []
    for front in fast_non_dominated_sort(obj):
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
            continue

        distances = crowding_distance(obj, front)
        remaining = n_select - len(selected)
        partial = sorted(front, key=lambda idx: (-distances[idx], idx))[:remaining]
        selected.extend(partial)
        break

    return selected


def rank_table(objectives: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Return a small diagnostic table for notebooks and logs."""

    obj = as_objective_array(objectives)
    ranks = rank_population(obj)
    rank_df = pd.DataFrame([rank.to_dict() for rank in ranks]).set_index("index")
    objective_df = pd.DataFrame(obj, columns=[f"objective_{i}" for i in range(obj.shape[1])])
    return objective_df.join(rank_df).sort_values(["front", "crowding_distance"], ascending=[True, False])
