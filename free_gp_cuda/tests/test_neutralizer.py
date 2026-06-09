from __future__ import annotations

import unittest

import torch

from alpha_gen.free_gp_cuda.neutralizer import industry_demean, neutralize_factor_tensor


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for free_gp_cuda neutralizer tests")
    return torch.device("cuda")


def _tensor(values, *, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=_device())


class NeutralizerTests(unittest.TestCase):
    def test_industry_demean_removes_group_means(self) -> None:
        values = _tensor([[1.0, 3.0, 10.0, 14.0], [2.0, 4.0, 20.0, 24.0]])
        industry = _tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
        demeaned = industry_demean(values, industry)

        for code in (0, 1):
            group = industry == code
            group_values = torch.where(group, demeaned, torch.full_like(demeaned, float("nan")))
            mean = torch.nanmean(group_values, dim=1)
            self.assertLess(mean.abs().max().detach().cpu().item(), 1e-6)

    def test_full_neutralization_removes_industry_and_all_styles(self) -> None:
        industry = _tensor([
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ], dtype=torch.long)
        style = _tensor([
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
        ])
        factor = industry.to(torch.float32) * 100.0 + style * 2.0
        result = neutralize_factor_tensor(
            factor,
            style_tensors=style.unsqueeze(2),
            industry_codes=industry,
            style_fields=("style_a",),
            neutralize_industry=True,
            neutralize_styles=True,
        )

        self.assertEqual(result.selected_fields, ("style_a",))
        self.assertEqual(int(result.selected_mask.sum().detach().cpu().item()), 1)
        self.assertLess(torch.nan_to_num(result.residual_factor.abs()).max().detach().cpu().item(), 1e-4)

    def test_constant_style_does_not_invalidate_whole_row(self) -> None:
        industry = _tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
        style = torch.ones((1, 6, 1), dtype=torch.float32, device=_device())
        factor = _tensor([[1.0, 2.0, 3.0, 11.0, 12.0, 13.0]])
        result = neutralize_factor_tensor(
            factor,
            style_tensors=style,
            industry_codes=industry,
            style_fields=("constant_style",),
            neutralize_industry=True,
            neutralize_styles=True,
        )

        self.assertTrue(torch.isfinite(result.residual_factor).any().detach().cpu().item())
        self.assertEqual(result.selected_fields, ("constant_style",))

    def test_barra_residual_is_industry_demeaned_after_regression(self) -> None:
        industry = _tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
        style = _tensor([[[1.0], [2.0], [4.0], [10.0], [20.0], [40.0]]])
        factor = _tensor([[1.0, 2.0, 4.0, 10.0, 20.0, 40.0]])

        result = neutralize_factor_tensor(
            factor,
            style_tensors=style,
            industry_codes=industry,
            neutralize_industry=True,
            neutralize_styles=True,
        )

        raw_mean = style.mean(dim=1, keepdim=True)
        raw_std = style.std(dim=1, keepdim=True, unbiased=True)
        zscore = (style - raw_mean) / raw_std
        design = torch.cat([torch.ones((1, 6, 1), dtype=torch.float32, device=_device()), zscore], dim=2)
        y = factor.unsqueeze(2)
        beta = torch.linalg.solve(design.transpose(1, 2).matmul(design), design.transpose(1, 2).matmul(y))
        style_residual = factor - design.matmul(beta).squeeze(2)
        expected_residual = industry_demean(style_residual, industry)

        self.assertLess(
            torch.nan_to_num(result.residual_factor - expected_residual).abs().max().detach().cpu().item(),
            1e-4,
        )
        for code in (0, 1):
            group = industry == code
            group_values = torch.where(group, result.residual_factor, torch.full_like(result.residual_factor, float("nan")))
            self.assertLess(torch.nanmean(group_values, dim=1).abs().max().detach().cpu().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
