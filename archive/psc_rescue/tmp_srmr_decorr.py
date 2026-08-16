import numpy as np, glob
import soundfile as sf, scipy.signal as sps, parselmouth, pyroomacoustics as pra
from scipy.stats import spearmanr
from versa.utterance_metrics.srmr import srmr_metric
def srmr_of(a, sr): return srmr_metric(a, sr, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False)['srmr']
def f0_of(a, sr):
    p = parselmouth.Sound(a, sampling_frequency=sr).to_pitch(); v = p.selected_array['frequency']; v=v[v>0]
    return float(np.mean(v)) if len(v) else np.nan
def rand_rir(sr, rng):
    dims=[rng.uniform(4,9), rng.uniform(3,7), rng.uniform(2.6,4)]
    rt60=rng.uniform(0.3,0.8)                                  # realistic far-field RT60
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room=pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=min(max_order,40))
    room.add_source([rng.uniform(0.5,dims[0]-0.5),rng.uniform(0.5,dims[1]-0.5),rng.uniform(1,2)])
    room.add_microphone([rng.uniform(0.5,dims[0]-0.5),rng.uniform(0.5,dims[1]-0.5),rng.uniform(1,2)])
    room.compute_rir(); return room.rir[0][0], rt60
clips=sorted(glob.glob('/ocean/projects/cis260125p/shared/data/clean_control_audio/*.wav'))[:160]
rng=np.random.default_rng(0); f0s=[]; sa=[]; sv=[]; rts=[]
for i,c in enumerate(clips):
    try:
        x,sr=sf.read(c); x=x.mean(1) if x.ndim>1 else x; x=x.astype(np.float64)
        f0=f0_of(x,sr)
        if not (f0==f0): continue
        a=srmr_of(x,sr); rir,rt=rand_rir(sr,rng); xr=sps.fftconvolve(x,rir)[:len(x)]; v=srmr_of(xr,sr)
        f0s.append(f0); sa.append(a); sv.append(v); rts.append(rt)
    except Exception as e:
        if i<3: print('skip',repr(e)[:70],flush=True)
    if (i+1)%40==0: print(f'{i+1}/{len(clips)} kept {len(f0s)}',flush=True)
print(f'N={len(f0s)}  mean RT60={np.mean(rts):.2f}s',flush=True)
print(f'SRMR_anechoic    mean={np.mean(sa):.2f}  vs f0 SRCC={spearmanr(sa,f0s).correlation:+.3f}',flush=True)
print(f'SRMR_reverberant mean={np.mean(sv):.2f}  vs f0 SRCC={spearmanr(sv,f0s).correlation:+.3f}',flush=True)
