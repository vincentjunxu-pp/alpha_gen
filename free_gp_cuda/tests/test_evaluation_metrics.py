from __future__ import annotations

import unittest

import torch

from alpha_gen.free_gp_cuda.evaluation_metrics import (
    daily_ic,
    daily_rank_ic,
    dynamic_style_neutralize,
    evaluate_factor_tensor,
    factor_coverage,
    factor_stability,
    ndcg_at_k,
    top_turnover,
)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda metric tests")
    return torch.device("cuda")


def _tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=_device())


class EvaluationMetricTests(unittest.TestCase):
    def test_daily_ic_and_rank_ic(self) -> None:
        factor = _tensor([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, float("nan"), 2.0, 3.0],
        ])
        label = _tensor([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ])

        pearson = daily_ic(factor, label, min_cross_section_size=4)
        rank_ic = daily_rank_ic(factor, label, min_cross_section_size=4)

        self.assertEqual(tuple(pearson.shape), (2,))
        self.assertTrue(torch.allclose(pearson.detach().cpu(), torch.tensor([1.0, -1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(rank_ic.detach().cpu(), torch.tensor([1.0, -1.0]), atol=1e-6))

    def test_score_auto_direction_and_ndcg(self) -> None:
        label = _tensor([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
        ])
        factor = -label
        score = evaluate_factor_tensor(
            factor,
            label,
            ndcg_k=2,
            n_groups=4,
            min_cross_section_size=4,
        )

        self.assertEqual(score.direction, -1)
        self.assertAlmostEqual(score.abs_rank_ic, 1.0, places=6)
        self.assertAlmostEqual(score.ic_win_rate, 1.0, places=6)
        self.assertAlmostEqual(score.ndcg_at_k, 1.0, places=6)

    def test_coverage_turnover_and_stability(self) -> None:
        factor = _tensor([
            [1.0, 2.0, float("nan"), 4.0],
            [1.0, 3.0, 2.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])
        label = _tensor([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ])
        tradeable = torch.ones_like(factor, dtype=torch.bool)
        tradeable[0, 2] = False

        self.assertAlmostEqual(factor_coverage(factor, label, tradeable), 1.0, places=6)
        self.assertGreaterEqual(top_turnover(factor, tradeable=tradeable, k=2), 0.0)
        self.assertLessEqual(top_turnover(factor, tradeable=tradeable, k=2), 1.0)
        self.assertGreater(factor_stability(factor, tradeable=tradeable, min_cross_section_size=3), -1.01)

    def test_ndcg_perfect_beats_reversed(self) -> None:
        label = _tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ])
        perfect = ndcg_at_k(label, label, k=2, n_groups=5)
        reversed_score = ndcg_at_k(-label, label, k=2, n_groups=5)

        self.assertAlmostEqual(perfect, 1.0, places=6)
        self.assertLess(reversed_score, perfect)

    def test_dynamic_style_neutralize_selects_and_residualizes(self) -> None:
        factor = _tensor([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ])
        style = factor.unsqueeze(2)
        result = dynamic_style_neutralize(
            factor,
            style,
            style_fields=("style_a",),
            corr_threshold=0.10,
            max_styles=1,
        )

        self.assertEqual(int(result.selected_mask.sum().detach().cpu().item()), 1)
        self.assertEqual(result.selected_fields, ("style_a",))
        self.assertLess(torch.nan_to_num(result.residual_factor.abs()).max().detach().cpu().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
