import sys, glob, os, numpy as np, torch
sys.path.insert(0,'src')
import feature_extractor_mix as fx
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
SH='/ocean/projects/cis260125p/shared'
PT=SH+'/data/processed_clean_train_s1clean'; S1=SH+'/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/s1'
N=1200
pts=sorted(glob.glob(PT+'/*.pt'))[:N]
X=[];Yj=[];Ys=[];Yh=[]
for p in pts:
    try:
        d=torch.load(p,map_location='cpu'); feat=d['audio_features'].float().mean(0).numpy()
        wav=os.path.join(S1, d['filename'].replace('_s1clean.wav','.wav'))
        if not os.path.exists(wav): continue
        jj=fx.compute_jitter(wav); j=jj.get('jitter_local_pct'); s=fx.compute_shimmer(wav); h=fx.compute_hnr(wav)
        if None in (j,s,h) or any(np.isnan([j,s,h])): continue
        X.append(feat);Yj.append(j);Ys.append(s);Yh.append(h)
    except Exception: continue
X=np.array(X); n=len(X); print('probe N=',n,flush=True)
rng=np.random.RandomState(0); perm=rng.permutation(n); X=X[perm]
ntr=int(0.8*n); Xtr,Xte=X[:ntr],X[ntr:]
sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
print('=== LINEAR PROBE (frozen WavLM -> voice quality, held-out SRCC) ===',flush=True)
for name,Y in [('jitter',Yj),('shimmer',Ys),('hnr',Yh)]:
    Y=np.array(Y)[perm]; Ytr,Yte=Y[:ntr],Y[ntr:]
    m=Ridge(alpha=10.0).fit(Xtr,Ytr); r=spearmanr(m.predict(Xte),Yte).correlation
    print(f'  {name}: SRCC={r:+.3f} (n_test={len(Yte)})',flush=True)
