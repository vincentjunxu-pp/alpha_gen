from __future__ import annotations

import unittest

from alpha_gen.free_gp_cuda.nsga2 import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    nsga2_select,
    rank_population,
    rank_table,
)


class NSGA2Tests(unittest.TestCase):
    def test_dominates_for_maximization(self) -> None:
        self.assertTrue(dominates((2.0, 1.0), (1.0, 1.0)))
        self.assertFalse(dominates((1.0, 2.0), (2.0, 1.0)))
        self.assertFalse(dominates((1.0, 1.0), (1.0, 1.0)))

    def test_sort_and_select(self) -> None:
        objectives = [
            (3.0, 1.0),
            (1.0, 3.0),
            (2.0, 2.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ]
        fronts = fast_non_dominated_sort(objectives)
        self.assertEqual(set(fronts[0]), {0, 1, 2})

        selected = nsga2_select(objectives, 3)
        self.assertEqual(set(selected), {0, 1, 2})

        ranks = rank_population(objectives)
        self.assertEqual({rank.index for rank in ranks[:3]}, {0, 1, 2})

    def test_crowding_and_rank_table(self) -> None:
        objectives = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        distance = crowding_distance(objectives, [0, 1, 2])
        self.assertEqual(distance[0], float("inf"))
        self.assertEqual(distance[2], float("inf"))

        table = rank_table(objectives)
        self.assertIn("front", table.columns)
        self.assertIn("crowding_distance", table.columns)


if __name__ == "__main__":
    unittest.main()
