from __future__ import annotations

import unittest

import pandas as pd
import torch

from alpha_gen.free_gp_cuda.context import CudaFactorContext
from alpha_gen.free_gp_cuda.generator import ProgramGeneratorConfig
from alpha_gen.free_gp_cuda.scorer import ScorerConfig
from alpha_gen.free_gp_cuda.search import (
    FreeGPSearchConfig,
    evaluated_to_frame,
    run_free_gp_search,
    selected_rank_table,
    validate_population,
)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda search tests")
    return torch.device("cuda")


def _make_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01 15:00:00", periods=8, freq="B", name="Datetime")
    contracts = pd.Index(["A", "B", "C", "D", "E"], name="Contract")
    index = pd.MultiIndex.from_product([dates, contracts], names=["Datetime", "Contract"])
    rows = []
    for date_id, _date in enumerate(dates):
        for contract_id, contract in enumerate(contracts):
            base = float(date_id * 10 + contract_id + 1)
            rows.append(
                {
                    "feature_a": base,
                    "feature_b": base * 0.5 + contract_id,
                    "feature_c": (-1.0 if contract_id % 2 else 1.0) * base,
                    "label_20d": base,
                    "is_tradeable": 0 if contract == "E" and date_id == 3 else 1,
                    "industry_code": "tech" if contract in {"A", "B", "C"} else "finance",
                }
            )
    return pd.DataFrame(rows, index=index)


def _ctx() -> CudaFactorContext:
    return CudaFactorContext(_make_panel(), device=_device())


def _config() -> FreeGPSearchConfig:
    return FreeGPSearchConfig(
        population_size=8,
        generations=1,
        random_seed=11,
        min_coverage=0.0,
        generator_config=ProgramGeneratorConfig(max_depth=3, max_size=24, terminal_probability=0.25),
        scorer_config=ScorerConfig(ndcg_k=2, n_groups=5, min_cross_section_size=4),
    )


class SearchTests(unittest.TestCase):
    def test_run_search_returns_population_and_history(self) -> None:
        ctx = _ctx()
        result = run_free_gp_search(
            ctx,
            candidate_fields=("feature_a", "feature_b", "feature_c"),
            train_dates=ctx.dates[:6],
            config=_config(),
        )

        self.assertEqual(len(result.final_population), result.config.population_size)
        self.assertGreaterEqual(len(result.history), result.config.population_size)
        self.assertTrue(all(item.train_score.expression for item in result.final_population))

        frame = evaluated_to_frame(result.final_population, include_program_json=True)
        self.assertIn("train_rank_ic_ir", frame.columns)
        self.assertIn("program_json", frame.columns)

        ranks = selected_rank_table(result.final_population)
        self.assertIn("expression", ranks.columns)

    def test_validate_population_sets_valid_score(self) -> None:
        ctx = _ctx()
        result = run_free_gp_search(
            ctx,
            candidate_fields=("feature_a", "feature_b", "feature_c"),
            train_dates=ctx.dates[:6],
            config=_config(),
        )
        validated = validate_population(result.final_population, ctx, ctx.dates[6:], config=result.config)

        self.assertEqual(len(validated), len(result.final_population))
        self.assertTrue(all(item.valid_score is not None for item in validated))


if __name__ == "__main__":
    unittest.main()
