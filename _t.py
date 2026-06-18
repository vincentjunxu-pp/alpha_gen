import os, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128,garbage_collection_threshold:0.7')
sys.path.insert(0,r'E:/实习')
import torch, numpy as np, pandas as pd
from alpha_gen.core.torch_backend import daily_rank_ic_torch, daily_ic_torch, _apply_mask
from alpha_gen.core.metrics import daily_rank_ic as cpu_rank_ic, daily_ic as cpu_ic

torch.manual_seed(1)
n_dates, n_stocks = 100, 200
factor = torch.randn(n_dates, n_stocks, device='cuda')
label = torch.randn(n_dates, n_stocks, device='cuda') * 0.5 + factor * 0.2
tradeable = torch.ones(n_dates, n_stocks, dtype=torch.bool, device='cuda')
tradeable[:, :10] = False

f_masked = _apply_mask(factor, tradeable)
l_masked = _apply_mask(label, tradeable)

g_ric = daily_rank_ic_torch(f_masked, l_masked).cpu().numpy()
g_pic = daily_ic_torch(f_masked, l_masked).cpu().numpy()

fc = pd.DataFrame(factor.cpu().numpy()).where(tradeable.cpu().numpy())
lc = pd.DataFrame(label.cpu().numpy()).where(tradeable.cpu().numpy())
c_ric = cpu_rank_ic(fc, lc).values
c_pic = cpu_ic(fc, lc).values

print(f'RankIC len: GPU={len(g_ric)} CPU={len(c_ric)}')
if len(g_ric) == len(c_ric):
    max_d = abs(g_ric - c_ric).max()
    corr = float(np.corrcoef(g_ric, c_ric)[0,1]) if len(g_ric)>1 else 1.0
    print(f'RankIC: maxΔ={max_d:.6f} corr={corr:.6f}')
    # Show a few samples
    for i in range(min(5, len(g_ric))):
        print(f'  day{i}: GPU={g_ric[i]:+.6f} CPU={c_ric[i]:+.6f}')

print(f'PearsonIC len: GPU={len(g_pic)} CPU={len(c_pic)}')
if len(g_pic) == len(c_pic):
    max_d = abs(g_pic - c_pic).max()
    corr = float(np.corrcoef(g_pic, c_pic)[0,1]) if len(g_pic)>1 else 1.0
    print(f'PearsonIC: maxΔ={max_d:.6f} corr={corr:.6f}')
