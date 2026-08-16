import sys, glob, os, tempfile
sys.path.insert(0,'src')
import numpy as np, soundfile as sf, scipy.signal as sps, pyroomacoustics as pra, pandas as pd
from scipy.stats import spearmanr
import feature_extractor_mix as fx
srmr_cfg = getattr(fx,'SRMR_CONFIG',{})
srmr_model = fx.load_srmr_model(srmr_cfg)
def rand_rir(sr,rng):
    dims=[rng.uniform(4,9),rng.uniform(3,7),rng.uniform(2.6,4)]; rt60=rng.uniform(0.3,0.8)
    e,mo=pra.inverse_sabine(rt60,dims)
    room=pra.ShoeBox(dims,fs=sr,materials=pra.Material(e),max_order=int(min(mo,40)))
    room.add_source([rng.uniform(0.5,dims[0]-0.5),rng.uniform(0.5,dims[1]-0.5),rng.uniform(1,2)])
    room.add_microphone([rng.uniform(0.5,dims[0]-0.5),rng.uniform(0.5,dims[1]-0.5),rng.uniform(1,2)])
    room.compute_rir(); return room.rir[0][0]
clips=sorted(glob.glob('/ocean/projects/cis260125p/shared/data/clean_control_audio/*.wav'))[:800]
print('clips:',len(clips),flush=True)
rng=np.random.default_rng(0); tmp=tempfile.mkdtemp(); rows=[]
for i,c in enumerate(clips):
    try:
        x,sr=sf.read(c); x=x.mean(1) if x.ndim>1 else x; x=x.astype(np.float64)
        rir=rand_rir(sr,rng); xr=sps.fftconvolve(x,rir)[:len(x)]
        rp=os.path.join(tmp,'rev.wav'); sf.write(rp,xr,sr)
        srmr=fx.compute_srmr(rp,srmr_model)              # SRMR on REVERBERANT (real reverb)
        f0=fx.compute_f0_variation(c)                    # f0 on CLEAN stem
        spk=fx.compute_praat_speaking_rate(c)            # rate on CLEAN stem
        pau=fx.compute_praat_pause_patterns(c)           # pauses on CLEAN stem
        if i==0: print('f0 keys:',list(f0)[:6],'| spk:',list(spk)[:4],'| pau:',list(pau)[:5],flush=True)
        rows.append(dict(snr_db=rng.uniform(0,40),       # INDEPENDENT snr
            srmr=srmr, f0_mean_hz=f0.get('f0_mean_hz'),
            praat_speaking_rate_syl_sec=spk.get('praat_speaking_rate_syl_sec'),
            praat_pause_count=pau.get('praat_pause_count'),
            praat_pause_rate_per_min=pau.get('praat_pause_rate_per_min')))
    except Exception as e:
        if i<3: print('skip',repr(e)[:90],flush=True)
    if (i+1)%100==0: print(f'{i+1}/{len(clips)} kept {len(rows)}',flush=True)
df=pd.DataFrame(rows).dropna(); df.to_csv('/ocean/projects/cis260125p/shared/data/features_corrected_subset.csv',index=False)
print(f'N={len(df)}',flush=True)
def sp(a,b): return spearmanr(df[a],df[b]).correlation
print('=== CORRECTED confound matrix  (original in parens) ===',flush=True)
print(f'  srmr <-> f0            : {sp("srmr","f0_mean_hz"):+.3f}   (orig +0.80)',flush=True)
print(f'  snr  <-> pause_count   : {sp("snr_db","praat_pause_count"):+.3f}   (orig +0.57)',flush=True)
print(f'  snr  <-> pause_rate    : {sp("snr_db","praat_pause_rate_per_min"):+.3f}   (orig +0.66)',flush=True)
print(f'  snr  <-> speaking_rate : {sp("snr_db","praat_speaking_rate_syl_sec"):+.3f}   (orig -0.58)',flush=True)
