from __future__ import annotations

import unittest

import pandas as pd
import torch

from alpha_gen.free_gp_cuda.context import CudaFactorContext
from alpha_gen.free_gp_cuda.evaluator import ProgramEvaluator, evaluate_node, evaluate_program
from alpha_gen.free_gp_cuda.program import BinaryNode, ConstNode, FieldNode, GateNode, Program, UnaryNode


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda evaluator tests")
    return torch.device("cuda")


def _make_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01 15:00:00", periods=4, freq="B", name="Datetime")
    contracts = pd.Index(["A", "B", "C"], name="Contract")
    index = pd.MultiIndex.from_product([dates, contracts], names=["Datetime", "Contract"])
    rows = []
    for date_id, _date in enumerate(dates):
        for contract_id, contract in enumerate(contracts):
            value = float(date_id * 10 + contract_id)
            rows.append(
                {
                    "feature_a": value,
                    "feature_b": value * 2.0,
                    "label_20d": value / 100.0,
                    "is_tradeable": 0 if contract == "C" and date_id == 2 else 1,
                    "industry_code": "tech" if contract in {"A", "B"} else "finance",
                }
            )
    return pd.DataFrame(rows, index=index)


def _ctx(**kwargs) -> CudaFactorContext:
    return CudaFactorContext(_make_panel(), device=_device(), **kwargs)


def _tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=_device())


def _assert_close(testcase: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    testcase.assertTrue(
        torch.allclose(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=atol, equal_nan=True),
        msg=f"\nactual={actual.detach().cpu()}\nexpected={expected.detach().cpu()}",
    )


class EvaluatorTests(unittest.TestCase):
    def test_simple_numeric_program(self) -> None:
        ctx = _ctx()
        program = Program(BinaryNode("sub", FieldNode("feature_b"), FieldNode("feature_a")))
        result = evaluate_program(program, ctx)
        expected = _tensor([
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, float("nan")],
            [30.0, 31.0, 32.0],
        ])
        _assert_close(self, result, expected)

    def test_gate_program_uses_mask_tree(self) -> None:
        ctx = _ctx()
        program = Program(
            GateNode(
                "gate_nan",
                signal=FieldNode("feature_a"),
                mask=UnaryNode("mask_rank_high_80", FieldNode("feature_b")),
            )
        )
        result = evaluate_program(program, ctx)
        expected = _tensor([
            [float("nan"), float("nan"), 2.0],
            [float("nan"), float("nan"), 12.0],
            [float("nan"), 21.0, float("nan")],
            [float("nan"), float("nan"), 32.0],
        ])
        _assert_close(self, result, expected)

    def test_dates_subset_keeps_full_history_before_slice(self) -> None:
        ctx = _ctx()
        program = Program(UnaryNode("ts_mean_5", FieldNode("feature_a")))
        result = evaluate_program(program, ctx, dates=ctx.dates[[2, 3]])
        self.assertEqual(tuple(result.shape), (2, 3))
        expected = _tensor([
            [float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan")],
        ])
        _assert_close(self, result, expected)

    def test_subtree_cache_reuses_shared_nodes(self) -> None:
        ctx = _ctx()
        shared = UnaryNode("slog", FieldNode("feature_a"))
        program = Program(BinaryNode("add", shared, shared))
        evaluator = ProgramEvaluator(ctx)
        result = evaluator.evaluate(program, clear_cache=True)
        expected = torch.sign(ctx.get_field("feature_a")) * torch.log1p(ctx.get_field("feature_a").abs()) * 2.0
        expected = torch.where(ctx.tradeable() & torch.isfinite(expected), expected, torch.full_like(expected, float("nan")))
        _assert_close(self, result, expected)
        self.assertLess(len(evaluator.cache_info()["keys"]), program.size)
        self.assertIn(("unary", "slog", ("field", "feature_a")), evaluator.cache_info()["keys"])

    def test_default_validation_blocks_non_searchable_fields(self) -> None:
        ctx = _ctx()
        program = Program(FieldNode("label_20d"))
        with self.assertRaises(ValueError):
            evaluate_program(program, ctx)
        result = evaluate_program(program, ctx, available_fields=ctx.available_columns)
        expected = torch.where(ctx.tradeable(), ctx.label(), torch.full_like(ctx.label(), float("nan")))
        _assert_close(self, result, expected)

    def test_evaluate_node_mask_and_invalid_program(self) -> None:
        ctx = _ctx()
        mask = evaluate_node(UnaryNode("mask_rank_high_80", FieldNode("feature_b")), ctx)
        self.assertEqual(tuple(mask.shape), ctx.shape)
        self.assertTrue(torch.all((mask == 0) | (mask == 1)).detach().cpu().item())

        with self.assertRaises(ValueError):
            evaluate_program(Program(UnaryNode("mask_rank_high_80", FieldNode("feature_b"))), ctx)
        with self.assertRaises(ValueError):
            evaluate_program(Program(BinaryNode("add", FieldNode("feature_a"), UnaryNode("mask_rank_high_80", FieldNode("feature_b")))), ctx)

    def test_constants_and_tradeable_flags(self) -> None:
        ctx = _ctx()
        program = Program(BinaryNode("add", FieldNode("feature_a"), ConstNode(1.5)))
        result = evaluate_program(program, ctx)
        expected = _tensor([
            [1.5, 2.5, 3.5],
            [11.5, 12.5, 13.5],
            [21.5, 22.5, float("nan")],
            [31.5, 32.5, 33.5],
        ])
        _assert_close(self, result, expected)

        unmasked = evaluate_program(program, ctx, mask_inputs_by_tradeable=False, tradeable_only=False)
        self.assertTrue(torch.isfinite(unmasked[2, 2]).detach().cpu().item())


if __name__ == "__main__":
    unittest.main()
