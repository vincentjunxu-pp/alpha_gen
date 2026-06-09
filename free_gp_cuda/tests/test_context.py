from __future__ import annotations

import unittest

import pandas as pd
import torch

from alpha_gen.free_gp_cuda.context import CudaFactorContext, validate_long_panel


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda context tests")
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
                    "note": "x",
                }
            )
    return pd.DataFrame(rows, index=index)


def _assert_close(testcase: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor) -> None:
    testcase.assertTrue(
        torch.allclose(actual.detach().cpu(), expected.detach().cpu(), equal_nan=True),
        msg=f"\nactual={actual.detach().cpu()}\nexpected={expected.detach().cpu()}",
    )


class ContextTests(unittest.TestCase):
    def test_long_panel_validation(self) -> None:
        panel = _make_panel()
        validate_long_panel(panel)
        with self.assertRaises(TypeError):
            validate_long_panel(panel.reset_index(drop=True))
        duplicate = pd.concat([panel, panel.iloc[[0]]])
        with self.assertRaises(ValueError):
            validate_long_panel(duplicate)

    def test_context_axes_fields_and_tensors(self) -> None:
        panel = _make_panel()
        ctx = CudaFactorContext(panel, device=_device(), max_cached_tensors=3)
        self.assertEqual(ctx.shape, (4, 3))
        self.assertEqual(ctx.searchable_fields, ("feature_a", "feature_b"))

        feature = ctx.get_field("feature_a")
        self.assertEqual(feature.device.type, "cuda")
        self.assertEqual(tuple(feature.shape), (4, 3))
        expected = torch.tensor(
            [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]],
            dtype=torch.float32,
            device=_device(),
        )
        _assert_close(self, feature, expected)
        self.assertEqual(ctx.get_field("feature_a").data_ptr(), feature.data_ptr())

        label = ctx.label()
        _assert_close(self, label, expected / 100.0)

    def test_tradeable_industry_dates_and_frame_roundtrip(self) -> None:
        ctx = CudaFactorContext(_make_panel(), device=_device())
        tradeable = ctx.tradeable()
        self.assertEqual(tradeable.dtype, torch.bool)
        self.assertFalse(bool(tradeable[2, 2].detach().cpu().item()))
        self.assertTrue(bool(tradeable[0, 0].detach().cpu().item()))

        industry = ctx.industry_codes()
        self.assertEqual(industry.dtype, torch.long)
        self.assertEqual(tuple(industry.shape), ctx.shape)
        self.assertEqual(set(ctx.industry_labels), {"finance", "tech"})
        self.assertEqual(int(industry[0, 0].detach().cpu().item()), int(industry[0, 1].detach().cpu().item()))
        self.assertNotEqual(int(industry[0, 0].detach().cpu().item()), int(industry[0, 2].detach().cpu().item()))

        positions = ctx.date_positions(ctx.dates[[1, 3]])
        self.assertEqual(positions.detach().cpu().tolist(), [1, 3])
        taken = ctx.take_dates(ctx.get_field("feature_a"), ctx.dates[[1, 3]])
        self.assertEqual(tuple(taken.shape), (2, 3))
        with self.assertRaises(KeyError):
            ctx.date_positions(pd.DatetimeIndex(["1999-01-01 15:00:00"]))

        frame = ctx.tensor_to_frame(ctx.get_field("feature_a"))
        self.assertEqual(frame.index.name, "Datetime")
        self.assertEqual(frame.columns.name, "Contract")
        self.assertEqual(float(frame.loc[ctx.dates[3], "C"]), 32.0)

    def test_lru_cache_and_cache_disabled(self) -> None:
        ctx = CudaFactorContext(_make_panel(), device=_device(), max_cached_tensors=2)
        ctx.get_field("feature_a")
        ctx.get_field("feature_b")
        self.assertEqual(len(ctx.cache_keys()), 2)
        ctx.label()
        self.assertEqual(len(ctx.cache_keys()), 2)
        self.assertNotIn(("field", "feature_a"), ctx.cache_keys())

        ctx.clear_cache()
        self.assertEqual(ctx.cache_keys(), ())

        uncached = CudaFactorContext(_make_panel(), device=_device(), cache_on_device=False)
        first = uncached.get_field("feature_a")
        second = uncached.get_field("feature_a")
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertFalse(uncached.cache_info()["enabled"])

    def test_candidate_field_validation_and_missing_columns(self) -> None:
        panel = _make_panel()
        ctx = CudaFactorContext(panel, device=_device(), candidate_fields=("feature_b", "feature_a"))
        self.assertEqual(ctx.searchable_fields, ("feature_b", "feature_a"))

        with self.assertRaises(KeyError):
            CudaFactorContext(panel.drop(columns=["label_20d"]), device=_device())
        with self.assertRaises(KeyError):
            CudaFactorContext(panel, device=_device(), candidate_fields=("missing",))
        with self.assertRaises(TypeError):
            CudaFactorContext(panel, device=_device(), candidate_fields=("note",))
        with self.assertRaises(TypeError):
            ctx.get_field("note")
        with self.assertRaises(KeyError):
            ctx.get_field("missing")

    def test_missing_industry_configuration(self) -> None:
        ctx = CudaFactorContext(_make_panel().drop(columns=["industry_code"]), device=_device(), industry_col=None)
        with self.assertRaises(ValueError):
            ctx.industry_codes()


if __name__ == "__main__":
    unittest.main()
