from __future__ import annotations

import unittest

import pandas as pd
import torch

from alpha_gen.free_gp_cuda.context import CudaFactorContext
from alpha_gen.free_gp_cuda.program import BinaryNode, FieldNode, Program, UnaryNode
from alpha_gen.free_gp_cuda.scorer import ProgramScorer, ScorerConfig, factor_values, score_factor, score_program


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda scorer tests")
    return torch.device("cuda")


def _make_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01 15:00:00", periods=5, freq="B", name="Datetime")
    contracts = pd.Index(["A", "B", "C", "D"], name="Contract")
    index = pd.MultiIndex.from_product([dates, contracts], names=["Datetime", "Contract"])
    rows = []
    for date_id, _date in enumerate(dates):
        for contract_id, contract in enumerate(contracts):
            value = float(date_id * 10 + contract_id + 1)
            rows.append(
                {
                    "feature_a": value,
                    "feature_b": value * 2.0,
                    "style_a": value,
                    "label_20d": value,
                    "is_tradeable": 0 if contract == "D" and date_id == 2 else 1,
                    "industry_code": "tech" if contract in {"A", "B"} else "finance",
                }
            )
    return pd.DataFrame(rows, index=index)


def _ctx() -> CudaFactorContext:
    return CudaFactorContext(_make_panel(), device=_device())


class ScorerTests(unittest.TestCase):
    def test_score_program_basic(self) -> None:
        ctx = _ctx()
        program = Program(FieldNode("feature_a"))
        scored = score_program(
            program,
            ctx,
            config=ScorerConfig(ndcg_k=2, n_groups=4, min_cross_section_size=4),
            raise_errors=True,
        )

        self.assertEqual(scored.error, "")
        self.assertEqual(scored.expression, "feature_a")
        self.assertEqual(scored.size, 1)
        self.assertAlmostEqual(scored.score.abs_rank_ic, 1.0, places=6)
        self.assertAlmostEqual(scored.score.ndcg_at_k, 1.0, places=6)
        self.assertGreater(scored.score.coverage, 0.90)

    def test_score_program_with_style_neutralization(self) -> None:
        ctx = _ctx()
        scorer = ProgramScorer(
            ctx,
            ScorerConfig(
                ndcg_k=2,
                n_groups=4,
                min_cross_section_size=4,
                style_fields=("style_a",),
                style_corr_threshold=0.10,
                style_max_fields=1,
            ),
        )
        scored = scorer.score_program(Program(FieldNode("feature_a")), raise_errors=True)

        self.assertEqual(scored.error, "")
        self.assertEqual(scored.score.style_selected_count, 1)
        self.assertEqual(scored.score.style_selected_fields, ("style_a",))
        self.assertGreaterEqual(scored.score.style_max_abs_corr, 0.99)

    def test_factor_values_raw_or_neutralized_view(self) -> None:
        ctx = _ctx()
        program = Program(FieldNode("feature_a"))
        config = ScorerConfig(style_fields=("style_a",), min_cross_section_size=3)
        raw = factor_values(program, ctx, view="raw", config=config)
        neutralized = factor_values(program, ctx, view="neutralized", config=config)

        self.assertEqual(tuple(raw.shape), ctx.shape)
        self.assertEqual(tuple(neutralized.shape), ctx.shape)
        self.assertGreater(torch.nan_to_num(raw.abs()).max().detach().cpu().item(), 1.0)
        self.assertLess(torch.nan_to_num(neutralized.abs()).max().detach().cpu().item(), 1e-4)

    def test_score_factor_accepts_full_factor_with_dates(self) -> None:
        ctx = _ctx()
        dates = ctx.dates[[3, 4]]
        factor = ctx.get_field("feature_a")
        score = score_factor(
            factor,
            ctx,
            dates=dates,
            config=ScorerConfig(ndcg_k=2, n_groups=4, min_cross_section_size=4),
        )

        self.assertEqual(score.n_ic_obs, 2)
        self.assertAlmostEqual(score.abs_rank_ic, 1.0, places=6)

    def test_invalid_program_returns_empty_score_unless_raised(self) -> None:
        ctx = _ctx()
        program = Program(FieldNode("label_20d"))
        scored = score_program(program, ctx)

        self.assertNotEqual(scored.error, "")
        self.assertEqual(scored.score.n_ic_obs, 0)
        with self.assertRaises(ValueError):
            score_program(program, ctx, raise_errors=True)

    def test_score_programs_batch(self) -> None:
        ctx = _ctx()
        scorer = ProgramScorer(ctx, ScorerConfig(ndcg_k=2, n_groups=4, min_cross_section_size=4))
        programs = [
            Program(FieldNode("feature_a")),
            Program(BinaryNode("sub", FieldNode("feature_b"), FieldNode("feature_a"))),
            Program(UnaryNode("ts_mean_5", FieldNode("feature_a"))),
        ]
        scored = scorer.score_programs(programs)

        self.assertEqual(len(scored), 3)
        self.assertTrue(all(item.expression for item in scored))
        self.assertTrue(all(isinstance(item.to_dict()["rank_ic_ir"], float) for item in scored))


if __name__ == "__main__":
    unittest.main()
