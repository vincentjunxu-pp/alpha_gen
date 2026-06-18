"""Compare GPU vs CPU metrics at the lowest level to pinpoint divergences.

USAGE:  conda activate pytorch && python _compare_gpu_cpu.py
"""
import json, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.7")
sys.path.insert(0, r'E:\实习')

import numpy as np, pandas as pd, torch

# ---- Load data ----
ROOT = r"E:\实习\alpha_gen"
META = ROOT + r"\data\metadata\fixtures\mock_behavior_metadata.json"
md = json.loads(open(META, encoding="utf-8").read())
size_field = md.get("size_field", "barra_size")
barra_fields = tuple(md.get("barra_style_fields", ()))
from alpha_gen.core.gene import load_field_rules
from alpha_gen.core.preprocess import build_transform_cache, load_panel
from alpha_gen.core.utils import get_rolling_windows
field_rules = load_field_rules(META)
panel = load_panel(ROOT + r"\data\panels\mock_behavior_daily.parquet")
cache = build_transform_cache(panel, field_rules,
    label_col="label_20d", tradeable_col="is_tradeable", industry_col="industry_code",
    extra_current_fields=[size_field, *barra_fields], show_progress=False)
train_dates, _ = get_rolling_windows(
    cache.label.index[:-20], train_start_date="20230718", test_start_date="20241231",
    stride=120000, horizon=20)[0]

# ---- Gen some factors via GA (gives realistic factor patterns) ----
from alpha_gen.behavior_gen.gene import load_behavior_field_rules
from alpha_gen.behavior_gen.torch_backend import BehaviorTorchContext, calculate_behavior_factor_tensor
from alpha_gen.behavior_gen.ga import BehaviorGAConfig, run_behavior_ga_search
behavior_rules = load_behavior_field_rules(META)
bctx = BehaviorTorchContext(cache=cache, behavior_field_rules=behavior_rules,
    device="cuda", cache_on_device=False, barra_style_fields=barra_fields)
config = BehaviorGAConfig(
    population_size=5, generations=1, random_seed=42,
    min_coverage=0.0, size_field=size_field, require_cuda=True, show_progress=False,
    ndcg_top_fraction=0.20,
)
r = run_behavior_ga_search(ctx=bctx, train_dates=train_dates, config=config)

# ---------------------------------------------------------------------------
# For each gene, compare GPU and CPU at the raw-tensor level
# ---------------------------------------------------------------------------
from alpha_gen.core.torch_backend import (
    daily_rank_ic_torch, daily_ic_torch, _row_corr, nan_rank_torch,
    cs_rank_pct_torch, _take_dates, _apply_mask,
)
from alpha_gen.core.metrics import daily_rank_ic, daily_ic, _daily_corr_series

THRESH = 0.01  # 1% relative diff = suspicious
print(f"\n{'='*70}")
print(f"Comparing GPU vs CPU per-gene (threshold={THRESH*100:.0f}%)")
print(f"{'='*70}")

disc = 0
for gi, g in enumerate(r.history[:5]):
    if g.error:
        continue

    # GPU factor tensor
    ft = calculate_behavior_factor_tensor(g.gene, bctx,
        neutralization_mode=config.neutralization_mode, size_field=config.size_field)

    # CPU factor DataFrame
    fc = pd.DataFrame(ft.detach().cpu().numpy(), index=cache.label.index, columns=cache.label.columns)

    # Crop to train dates, apply tradeable
    fg = _take_dates(ft, bctx.date_positions(train_dates))
    lg = _take_dates(bctx.label(), bctx.date_positions(train_dates))
    tg = _take_dates(bctx.tradeable(), bctx.date_positions(train_dates))
    fg = _apply_mask(fg, tg)
    lg = _apply_mask(lg, tg)

    fc_train = fc.loc[train_dates]
    lc_train = cache.label.loc[train_dates]
    tc_train = cache.tradeable.loc[train_dates]
    fc_train = fc_train.where(tc_train.replace([np.inf, -np.inf], np.nan).fillna(0).gt(0))
    lc_train = lc_train.where(tc_train.replace([np.inf, -np.inf], np.nan).fillna(0).gt(0))

    def check(name, gpu_val, cpu_val, tol=THRESH):
        global disc
        if abs(cpu_val) < 1e-8 and abs(gpu_val) < 1e-8:
            return
        d = abs(gpu_val - cpu_val) / max(abs(cpu_val), 1e-8)
        if d > tol:
            print(f"  ⚠ Gene{gi} {name:20s} GPU={gpu_val:+.6f} CPU={cpu_val:+.6f} Δ={d:.4f}")
            disc += 1

    # 1. Daily RankIC series
    g_ric = daily_rank_ic_torch(fg, lg).detach().cpu().numpy()
    c_ric = daily_rank_ic(fc_train, lc_train).values
    if len(g_ric) == len(c_ric):
        max_d = np.max(np.abs(g_ric - c_ric))
        corr = np.corrcoef(g_ric, c_ric)[0, 1] if len(g_ric) > 1 else 1.0
        if max_d > 0.001 or corr < 0.999:
            print(f"  ⚠ Gene{gi} RankIC-series  maxΔ={max_d:.6f} corr={corr:.6f}")
            disc += 1

    # 2. Daily Pearson IC series
    g_pic = daily_ic_torch(fg, lg).detach().cpu().numpy()
    c_pic = daily_ic(fc_train, lc_train).values
    if len(g_pic) == len(c_pic):
        max_d = np.max(np.abs(g_pic - c_pic))
        corr = np.corrcoef(g_pic, c_pic)[0, 1] if len(g_pic) > 1 else 1.0
        if max_d > 0.001 or corr < 0.999:
            print(f"  ⚠ Gene{gi} PearsonIC-series maxΔ={max_d:.6f} corr={corr:.6f}")
            disc += 1

    # 3. Summary metrics
    ts = g.train_score
    cs = r.history[gi].train_score  # same object

    # GPU rank_ic_ir vs CPU (compute from CPU daily_rank_ic)
    c_ric_mean = c_ric.mean(); c_ric_std = c_ric.std(ddof=1) if len(c_ric) > 1 else 0.0
    c_rir = c_ric_mean / c_ric_std if c_ric_std > 0 else 0.0
    check("rir", ts.rank_ic_ir, c_rir)

    # GPU mean_rank_ic
    check("mean_rank_ic", ts.mean_rank_ic, float(c_ric_mean))

    # GPU ic_win_rate vs CPU
    c_win = float((c_ric > 0).mean())
    check("ic_win_rate", ts.ic_win_rate, c_win)

    # Pearson IC/IR
    c_pic_mean = c_pic.mean(); c_pic_std = c_pic.std(ddof=1) if len(c_pic) > 1 else 0.0
    c_ir = c_pic_mean / c_pic_std if c_pic_std > 0 else 0.0
    check("ic", ts.ic, float(c_pic_mean))
    check("ir", ts.ir, float(c_ir))

    # Coverage
    g_cov = torch.isfinite(fg).sum(dim=1).float() / tg.sum(dim=1).clamp(min=1)
    g_cov = g_cov[tg.sum(dim=1) > 0].mean().item()
    c_cov_val = fc_train.notna().sum(axis=1) / tc_train.replace([np.inf,-np.inf],np.nan).fillna(0).gt(0).sum(axis=1)
    c_cov_val = c_cov_val[tc_train.replace([np.inf,-np.inf],np.nan).fillna(0).gt(0).sum(axis=1) > 0].mean()
    check("coverage", ts.coverage, float(c_cov_val))

    # Long metrics — compare GPU long_rir computation with CPU equivalent
    g_rank_pct = cs_rank_pct_torch(fg, mask=tg)
    g_long_mask = tg & (g_rank_pct >= 0.5)
    g_long_f = torch.where(g_long_mask, fg, torch.full_like(fg, float("nan")))
    g_long_l = torch.where(g_long_mask, lg, torch.full_like(lg, float("nan")))
    g_long_ric = daily_rank_ic_torch(g_long_f, g_long_l)
    if g_long_ric.numel() > 1:
        g_lr = float((g_long_ric.mean() / g_long_ric.std(unbiased=True)).detach().cpu().item()) if g_long_ric.std(unbiased=True) > 0 else 0.0
    else:
        g_lr = 0.0

    # CPU long_rir
    from alpha_gen.core.metrics import _top_half_mask
    c_long_mask = _top_half_mask(fc_train, lc_train)
    c_long_ric = daily_rank_ic(fc_train.where(c_long_mask), lc_train.where(c_long_mask))
    c_lr = float(c_long_ric.mean() / c_long_ric.std(ddof=1)) if len(c_long_ric) > 1 and c_long_ric.std(ddof=1) > 0 else 0.0
    check("long_rir", ts.long_rank_ic_ir, c_lr)

    del ft, fc

if disc == 0:
    print("✅ All GPU/CPU metrics consistent")
else:
    print(f"\n⚠ {disc} discrepancies found")

del bctx; torch.cuda.empty_cache()
