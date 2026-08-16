import soundfile as sf, os, numpy as np, sys, tempfile
sys.path.insert(0,'src')
from datasets import load_dataset, Audio
import feature_extractor_mix as fx
srmr_model=fx.load_srmr_model(getattr(fx,'SRMR_CONFIG',{}))
ds=load_dataset('vidalfernando/chime6_eval', split='eval', streaming=True)
ds=ds.cast_column('audio', Audio(sampling_rate=16000))
tmp=tempfile.mkdtemp(); srmrs=[]; n=0
for s in ds:
    if n>=120: break
    try:
        a=s['audio']; arr=np.asarray(a['array'],dtype=np.float64); sr=a['sampling_rate']
        if len(arr)<sr: continue
        p=os.path.join(tmp,'c.wav'); sf.write(p,arr,sr)
        v=fx.compute_srmr(p,srmr_model)
        if v is not None: srmrs.append(v)
        n+=1
        if n%30==0: print(n,'done',flush=True)
    except Exception as e:
        if n<2: print('skip',repr(e)[:80],flush=True)
srmrs=np.array(srmrs)
print(f'CHiME-6 SRMR (n={len(srmrs)}): mean={srmrs.mean():.2f} p5/50/95=[{np.percentile(srmrs,5):.1f}/{np.percentile(srmrs,50):.1f}/{np.percentile(srmrs,95):.1f}]',flush=True)
import pandas as pd
tr=pd.read_csv('/ocean/projects/cis260125p/shared/data/features_corrected_merged/train-100.csv')
tr=tr[~tr.filename.str.contains('s1clean')]
t5,t95=np.percentile(pd.to_numeric(tr.srmr,errors='coerce').dropna(),[5,95])
print(f'AUG-train SRMR [p5,p95]=[{t5:.1f},{t95:.1f}]',flush=True)
print(f'-> {((srmrs>=t5)&(srmrs<=t95)).mean():.2f} of CHiME-6 SRMR INSIDE aug-train support (1.0=interpolation)',flush=True)
