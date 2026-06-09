from __future__ import annotations

import math
import re
import unittest

import torch

from alpha_gen.free_gp_cuda import operators as ops
from alpha_gen.free_gp_cuda.registry import OPERATOR_REGISTRY, SELECT_COUNTS, WINDOWS


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda operator tests")
    return torch.device("cuda")


def _tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=_device())


def _assert_close(testcase: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    testcase.assertTrue(
        torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=atol, equal_nan=True),
        msg=f"\nactual={actual_cpu}\nexpected={expected_cpu}",
    )


class CudaOperatorTests(unittest.TestCase):
    def test_device_preserved(self) -> None:
        x = _tensor([[1.0, 2.0, float("nan")]])
        out = ops.neg(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(ops.mask_sign_pos(x).device.type, "cuda")

    def test_pointwise_boundaries(self) -> None:
        x = _tensor([[1.0, -2.0, float("nan"), float("inf")]])
        y = _tensor([[1.0, 0.0, 2.0, 4.0]])
        expected_slog = _tensor([[math.log1p(1.0), -math.log1p(2.0), float("nan"), float("nan")]])
        expected_qdiv = _tensor([[1.0, float("nan"), float("nan"), float("nan")]])
        _assert_close(self, ops.slog(x), expected_slog)
        _assert_close(self, ops.qdiv(x, y), expected_qdiv)

    def test_cross_section_rank_zscore_and_winsor(self) -> None:
        x = _tensor([
            [1.0, 2.0, 2.0, float("nan"), 5.0],
            [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ])
        expected_rank = _tensor([
            [0.25, 0.625, 0.625, float("nan"), 1.0],
            [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            [0.6, 0.6, 0.6, 0.6, 0.6],
        ])
        _assert_close(self, ops.cs_rank(x), expected_rank)

        row_std = math.sqrt(3.0)
        expected_z = _tensor([
            [(1.0 - 2.5) / row_std, (2.0 - 2.5) / row_std, (2.0 - 2.5) / row_std, float("nan"), (5.0 - 2.5) / row_std],
            [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
        ])
        _assert_close(self, ops.cs_zscore(x), expected_z)

        w = _tensor([[0.0, 1.0, 2.0, 3.0, 100.0], [1.0, float("nan"), float("nan"), 2.0, float("nan")]])
        expected_w = _tensor([[0.2, 1.0, 2.0, 3.0, 80.6], [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]])
        _assert_close(self, ops.cs_winsorize_5pct(w), expected_w)

    def test_cross_section_regression_and_industry(self) -> None:
        x = _tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y = _tensor([[3.0, 5.0, 7.0, 9.0, 11.0]])
        _assert_close(self, ops.cs_resid(y, x), _tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]), atol=1e-4)

    def test_time_series_no_future(self) -> None:
        x = _tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        _assert_close(self, ops.delay(x, 2), _tensor([[float("nan")], [float("nan")], [1.0], [2.0], [3.0]]))
        _assert_close(self, ops.ts_mean(x, 3), _tensor([[float("nan")], [float("nan")], [2.0], [3.0], [4.0]]))
        _assert_close(self, ops.ts_median(x, 3), _tensor([[float("nan")], [float("nan")], [2.0], [3.0], [4.0]]))
        _assert_close(self, ops.ts_std(x, 3), _tensor([[float("nan")], [float("nan")], [1.0], [1.0], [1.0]]))
        _assert_close(self, ops.ts_delta(x, 2), _tensor([[float("nan")], [float("nan")], [2.0], [2.0], [2.0]]))
        _assert_close(self, ops.ts_max_to_min(x, 3), _tensor([[float("nan")], [float("nan")], [2.0], [2.0], [2.0]]))
        _assert_close(self, ops.diff_sign(x, 3), _tensor([[float("nan")], [float("nan")], [1.0], [1.0], [1.0]]))

        y = x * 2.0
        _assert_close(self, ops.ts_corr(x, y, 3), _tensor([[float("nan")], [float("nan")], [1.0], [1.0], [1.0]]), atol=1e-4)

    def test_cross_section_time_series_combo_and_selection(self) -> None:
        x = _tensor([
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ])
        expected_meanrank = _tensor([
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [2.0 / 3.0, 5.0 / 6.0],
            [5.0 / 6.0, 2.0 / 3.0],
        ])
        _assert_close(self, ops.ts_meanrank(x, 3), expected_meanrank)

        sel_x = _tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
        sel_y = _tensor([[3.0], [1.0], [2.0], [5.0], [4.0]])
        expected_top = _tensor([[float("nan")], [float("nan")], [10.0], [40.0], [40.0]])
        expected_btm = _tensor([[float("nan")], [float("nan")], [20.0], [20.0], [30.0]])
        expected_diff = _tensor([[float("nan")], [float("nan")], [-10.0], [20.0], [10.0]])
        _assert_close(self, ops.rolling_selmean_diff(sel_x, sel_y, 3, 1), expected_diff)
        self.assertFalse(hasattr(ops, "rolling_selmean_top"))
        self.assertFalse(hasattr(ops, "rolling_selmean_btm"))

        del expected_top, expected_btm

    def test_masks_and_gates(self) -> None:
        x = _tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        high80 = ops.mask_rank_high_80(x)
        expected_high80 = _tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
        _assert_close(self, high80, expected_high80)

        high50 = ops.mask_rank_high_50(x)
        expected_high50 = _tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
        _assert_close(self, high50, expected_high50)

        low20 = ops.mask_rank_low_20(x)
        expected_low20 = _tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        _assert_close(self, low20, expected_low20)

        signed = _tensor([[-1.0, 0.0, 2.0, float("nan")]])
        _assert_close(self, ops.mask_sign_pos(signed), _tensor([[0.0, 0.0, 1.0, 0.0]]))
        _assert_close(self, ops.mask_sign_neg(signed), _tensor([[1.0, 0.0, 0.0, 0.0]]))

        _assert_close(self, ops.gate_nan(x, high80), _tensor([[float("nan"), float("nan"), float("nan"), 4.0, 5.0]]))
        _assert_close(self, ops.gate_zero(x, high80), _tensor([[0.0, 0.0, 0.0, 4.0, 5.0]]))
        bad = _tensor([[float("nan"), 2.0]])
        true_gate = _tensor([[1.0, 0.0]])
        _assert_close(self, ops.gate_zero(bad, true_gate), _tensor([[float("nan"), 0.0]]))

    def test_registry_metadata(self) -> None:
        self.assertEqual(WINDOWS, (5, 20, 60, 120))
        self.assertEqual(SELECT_COUNTS, (1, 3, 5))
        self.assertNotIn("ts_mean_10", OPERATOR_REGISTRY)
        self.assertNotIn("rolling_selmean_top_20_n10", OPERATOR_REGISTRY)
        self.assertNotIn("rolling_selmean_top_20_n3", OPERATOR_REGISTRY)
        self.assertIn("rolling_selmean_diff_20_n3", OPERATOR_REGISTRY)
        for removed in [
            "identity",
            "div",
            "max2",
            "min2",
            "cs_rank_centered",
            "ts_sum_20",
            "cs_beta",
            "industry_demean",
            "industry_zscore",
            "positive",
            "negative",
            "top_quantile",
            "bottom_quantile",
            "above_median",
            "below_median",
            "where",
            "cond_and",
            "ts_cov_20",
        ]:
            self.assertNotIn(removed, OPERATOR_REGISTRY)
        for kept in ["mask_rank_high_50", "mask_rank_high_80", "mask_rank_low_20", "mask_sign_pos", "mask_sign_neg"]:
            self.assertIn(kept, OPERATOR_REGISTRY)
        for name, spec in OPERATOR_REGISTRY.items():
            self.assertEqual(name, spec.name)
            self.assertGreaterEqual(spec.arity, 1)
            self.assertTrue(spec.category)
            self.assertIn(spec.output_type, {"numeric", "mask"})
            self.assertGreater(spec.cost, 0)
            self.assertTrue(callable(spec.func))

        suffixes = {
            int(match.group(1))
            for name, spec in OPERATOR_REGISTRY.items()
            for match in [re.search(r"_(\d+)$", name)]
            if match and spec.category == "time_series"
        }
        self.assertEqual(suffixes, set(WINDOWS))

        x = _tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        _assert_close(self, OPERATOR_REGISTRY["delay_5"].func(x), _tensor([[float("nan")], [float("nan")], [float("nan")], [float("nan")], [float("nan")], [1.0]]))
        _assert_close(self, OPERATOR_REGISTRY["ts_mean_5"].func(x), _tensor([[float("nan")], [float("nan")], [float("nan")], [float("nan")], [3.0], [4.0]]))


if __name__ == "__main__":
    unittest.main()
