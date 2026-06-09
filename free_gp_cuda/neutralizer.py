from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


Tensor = torch.Tensor
EPS = 1e-12


def _nan_like(values: Tensor) -> Tensor:
    return torch.full_like(values, float("nan"))


def _ensure_2d(values: Tensor, name: str) -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D [date, contract], got shape {tuple(values.shape)}")


def _as_bool_mask(mask: Tensor | None, like: Tensor, *, name: str = "mask") -> Tensor | None:
    if mask is None:
        return None
    if tuple(mask.shape) != tuple(like.shape):
        raise ValueError(f"{name} must share shape {tuple(like.shape)}, got {tuple(mask.shape)}")
    if mask.dtype == torch.bool:
        return mask
    return torch.isfinite(mask) & (mask != 0)


def _style_names(style_fields: Sequence[str], n_styles: int) -> tuple[str, ...]:
    names = tuple(str(item) for item in style_fields)
    return tuple(names[idx] if idx < len(names) else f"style_{idx}" for idx in range(n_styles))


def _common_valid(
    factor: Tensor,
    *,
    style_tensors: Tensor | None = None,
    industry_codes: Tensor | None = None,
    mask: Tensor | None = None,
    require_industry: bool = False,
) -> Tensor:
    valid = torch.isfinite(factor)
    bool_mask = _as_bool_mask(mask, factor)
    if bool_mask is not None:
        valid = valid & bool_mask
    if require_industry:
        if industry_codes is None:
            raise ValueError("industry_codes are required when neutralize_industry=True")
        if tuple(industry_codes.shape) != tuple(factor.shape):
            raise ValueError("industry_codes must share factor shape")
        valid = valid & (industry_codes >= 0)
    if style_tensors is not None:
        if style_tensors.ndim != 3:
            raise ValueError("style_tensors must be 3D [date, contract, style]")
        if tuple(style_tensors.shape[:2]) != tuple(factor.shape):
            raise ValueError("style_tensors must share factor date/contract dimensions")
        valid = valid & torch.isfinite(style_tensors).all(dim=2)
    return valid


def industry_demean(
    values: Tensor,
    industry_codes: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Demean a [date, contract] tensor within each date/industry group."""

    _ensure_2d(values, "values")
    if tuple(industry_codes.shape) != tuple(values.shape):
        raise ValueError("industry_codes must share values shape")
    valid = torch.isfinite(values) & (industry_codes >= 0)
    bool_mask = _as_bool_mask(mask, values)
    if bool_mask is not None:
        valid = valid & bool_mask

    output = _nan_like(values)
    unique_codes = torch.unique(industry_codes[valid])
    for code in unique_codes:
        group_valid = valid & (industry_codes == code)
        n = group_valid.sum(dim=1, keepdim=True).to(values.dtype)
        values0 = torch.where(group_valid, values, torch.zeros_like(values))
        mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
        output = torch.where(group_valid & (n > 0), values - mean, output)
    return output


def industry_demean_3d(
    values: Tensor,
    industry_codes: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Demean a [date, contract, style] tensor within date/industry groups."""

    if values.ndim != 3:
        raise ValueError(f"values must be 3D [date, contract, style], got shape {tuple(values.shape)}")
    if tuple(industry_codes.shape) != tuple(values.shape[:2]):
        raise ValueError("industry_codes must share the first two values dimensions")
    valid = torch.isfinite(values) & (industry_codes >= 0).unsqueeze(2)
    if mask is not None:
        if tuple(mask.shape) != tuple(values.shape[:2]):
            raise ValueError("mask must share the first two values dimensions")
        valid = valid & mask.to(torch.bool).unsqueeze(2)

    output = torch.full_like(values, float("nan"))
    base_valid = valid.any(dim=2)
    unique_codes = torch.unique(industry_codes[base_valid & (industry_codes >= 0)])
    for code in unique_codes:
        group = industry_codes == code
        group_valid = valid & group.unsqueeze(2)
        n = group_valid.sum(dim=1, keepdim=True).to(values.dtype)
        values0 = torch.where(group_valid, values, torch.zeros_like(values))
        mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
        output = torch.where(group_valid & (n > 0), values - mean, output)
    return output


def _row_zscore_3d(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    valid = torch.isfinite(values) & mask.unsqueeze(2)
    n = valid.sum(dim=1, keepdim=True).to(values.dtype)
    values0 = torch.where(valid, values, torch.zeros_like(values))
    mean = values0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    variance = (centered * centered).sum(dim=1, keepdim=True) / torch.clamp(n - 1.0, min=1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    zscore = (values - mean) / (std + eps)
    return torch.where(valid & (n >= 2) & (std > eps), zscore, torch.full_like(values, float("nan")))


def _row_corr_against_styles(factor: Tensor, styles: Tensor, valid: Tensor) -> Tensor:
    if styles.shape[2] == 0:
        return torch.empty((0,), dtype=factor.dtype, device=factor.device)
    factor0 = torch.where(valid, factor, torch.zeros_like(factor))
    n = valid.sum(dim=1, keepdim=True).to(factor.dtype)
    factor_mean = factor0.sum(dim=1, keepdim=True) / torch.clamp(n, min=1.0)
    factor_centered = torch.where(valid, factor - factor_mean, torch.zeros_like(factor))
    style_valid = torch.isfinite(styles) & valid.unsqueeze(2)
    styles0 = torch.where(style_valid, styles, torch.zeros_like(styles))
    style_n = style_valid.sum(dim=1, keepdim=True).to(styles.dtype)
    style_mean = styles0.sum(dim=1, keepdim=True) / torch.clamp(style_n, min=1.0)
    style_centered = torch.where(style_valid, styles - style_mean, torch.zeros_like(styles))

    common = torch.isfinite(factor_centered).unsqueeze(2) & style_valid
    cov = torch.where(common, factor_centered.unsqueeze(2) * style_centered, torch.zeros_like(style_centered)).sum(dim=1)
    var_factor = torch.where(valid, factor_centered * factor_centered, torch.zeros_like(factor_centered)).sum(dim=1)
    var_style = torch.where(style_valid, style_centered * style_centered, torch.zeros_like(style_centered)).sum(dim=1)
    denom = torch.sqrt(var_factor.unsqueeze(1) * var_style)
    daily_corr = torch.where(denom > EPS, cov / denom, torch.full_like(cov, float("nan")))
    finite = torch.isfinite(daily_corr)
    obs = finite.sum(dim=0)
    mean_corr = torch.where(
        obs > 0,
        torch.where(finite, daily_corr, torch.zeros_like(daily_corr)).sum(dim=0)
        / torch.clamp(obs.to(factor.dtype), min=1.0),
        torch.zeros((styles.shape[2],), dtype=factor.dtype, device=factor.device),
    )
    return mean_corr.abs()


@dataclass(frozen=True)
class NeutralizationResult:
    residual_factor: Tensor
    industry_neutral_factor: Tensor
    abs_mean_corr: Tensor
    selected_mask: Tensor
    selected_fields: tuple[str, ...] = ()


def neutralize_factor_tensor(
    factor: Tensor,
    *,
    style_tensors: Tensor | None = None,
    industry_codes: Tensor | None = None,
    mask: Tensor | None = None,
    style_fields: Sequence[str] = (),
    neutralize_industry: bool = True,
    neutralize_styles: bool = True,
    standardize_styles: bool = True,
    ridge: float = 1e-6,
) -> NeutralizationResult:
    """Remove Barra style exposures first, then neutralize the residual by industry.

    Barra fields are only cross-sectionally centered/standardized on the common
    valid universe. They are not industry-neutralized; industry exposure is
    removed from the factor residual after the Barra regression.
    """

    _ensure_2d(factor, "factor")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if not neutralize_industry and not neutralize_styles:
        valid = _common_valid(factor, style_tensors=style_tensors, mask=mask)
        cleaned = torch.where(valid, factor, _nan_like(factor))
        n_styles = 0 if style_tensors is None else style_tensors.shape[2]
        return NeutralizationResult(
            residual_factor=cleaned,
            industry_neutral_factor=cleaned,
            abs_mean_corr=torch.empty((0,), dtype=factor.dtype, device=factor.device),
            selected_mask=torch.zeros((n_styles,), dtype=torch.bool, device=factor.device),
            selected_fields=(),
        )

    common_valid = _common_valid(
        factor,
        style_tensors=style_tensors if neutralize_styles else None,
        industry_codes=industry_codes,
        mask=mask,
        require_industry=neutralize_industry,
    )
    y = torch.where(common_valid, factor, _nan_like(factor))

    n_styles = 0 if style_tensors is None else int(style_tensors.shape[2])
    selected_mask = torch.zeros((n_styles,), dtype=torch.bool, device=factor.device)
    empty_corr = torch.empty((0,), dtype=factor.dtype, device=factor.device)
    if not neutralize_styles or style_tensors is None or n_styles == 0:
        residual = industry_demean(y, industry_codes, mask=common_valid) if neutralize_industry else y
        return NeutralizationResult(
            residual_factor=residual,
            industry_neutral_factor=residual,
            abs_mean_corr=empty_corr,
            selected_mask=selected_mask,
            selected_fields=(),
        )

    if standardize_styles:
        x = _row_zscore_3d(style_tensors, common_valid)
    else:
        x = torch.where(common_valid.unsqueeze(2), style_tensors, torch.full_like(style_tensors, float("nan")))
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    valid = torch.isfinite(y) & common_valid
    selected_mask[:] = True
    selected_fields = _style_names(style_fields, n_styles)
    abs_mean_corr = _row_corr_against_styles(y, x, valid)

    params = n_styles + 1
    enough = valid.sum(dim=1, keepdim=True) >= (params + 1)
    intercept = torch.ones((*factor.shape, 1), dtype=factor.dtype, device=factor.device)
    design = torch.cat([intercept, x], dim=2)

    design = torch.where(valid.unsqueeze(2) & torch.isfinite(design), design, torch.zeros_like(design))
    y0 = torch.where(valid, y, torch.zeros_like(y)).unsqueeze(2)
    xt = design.transpose(1, 2)
    xtx = torch.matmul(xt, design)
    xty = torch.matmul(xt, y0)
    eye = torch.eye(design.shape[2], dtype=factor.dtype, device=factor.device).unsqueeze(0)
    beta = torch.linalg.solve(xtx + eye * ridge, xty)
    fitted = torch.matmul(design, beta).squeeze(2)
    style_residual = y - fitted
    style_residual = torch.where(valid & enough, style_residual, _nan_like(factor))
    residual = industry_demean(style_residual, industry_codes, mask=common_valid) if neutralize_industry else style_residual
    return NeutralizationResult(
        residual_factor=residual,
        industry_neutral_factor=residual,
        abs_mean_corr=abs_mean_corr,
        selected_mask=selected_mask,
        selected_fields=selected_fields,
    )


__all__ = [
    "NeutralizationResult",
    "industry_demean",
    "industry_demean_3d",
    "neutralize_factor_tensor",
]
