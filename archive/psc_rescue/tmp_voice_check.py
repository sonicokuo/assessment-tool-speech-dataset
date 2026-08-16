import sys, glob, numpy as np
sys.path.insert(0,'src')
import feature_extractor_mix as fx
SH='/ocean/projects/cis260125p/shared'
stems=sorted(glob.glob(SH+'/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/s1/*.wav'))[:300]
jl,jr=[],[]
for s in stems:
    try:
        jj=fx.compute_jitter(s); jl.append(jj.get('jitter_local_pct')); jr.append(jj.get('jitter_rap_pct'))
    except Exception: pass
def rep(n,v):
    a=np.array([x for x in v if x is not None],float); valid=a[~np.isnan(a)]
    if len(valid)==0: print(f'  {n}: ALL NaN -> DEAD'); return
    print(f'  {n}: N={len(v)} valid={len(valid)} nan%={100*(len(v)-len(valid))/max(len(v),1):.0f} mean={np.mean(valid):.4f} std={np.std(valid):.4f} cv={np.std(valid)/abs(np.mean(valid)+1e-9):.2f} range=[{np.min(valid):.3f},{np.max(valid):.3f}]')
print('=== JITTER GT (correct keys, 300 clean s1 stems) ===')
rep('jitter_local_pct',jl); rep('jitter_rap_pct',jr)
