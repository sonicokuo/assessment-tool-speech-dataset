import sys, glob, numpy as np
sys.path.insert(0,'src')
import feature_extractor_mix as fx
m=fx.load_srmr_model(getattr(fx,'SRMR_CONFIG',{}))
ws=sorted(glob.glob('/ocean/projects/cis260125p/shared/data/chime6_sample/*.wav'))
srmrs=np.array([v for v in (fx.compute_srmr(w,m) for w in ws) if v is not None])
print(f'CHiME-6 SRMR (n={len(srmrs)}): {np.round(srmrs,2)}', flush=True)
print(f'  mean={srmrs.mean():.2f} range=[{srmrs.min():.2f},{srmrs.max():.2f}]', flush=True)
print(f'  AUG-train support [1.5,6.2] -> interpolation={((srmrs>=1.5)&(srmrs<=6.2)).mean():.2f}', flush=True)
