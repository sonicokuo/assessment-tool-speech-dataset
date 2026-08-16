import sys, numpy as np, io, soundfile as sf, os, tempfile, glob
sys.path.insert(0,'src')
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
import feature_extractor_mix as fx
print('downloading parquet to disk...', flush=True)
d=snapshot_download('vidalfernando/chime6_eval', repo_type='dataset', allow_patterns=['*.parquet'])
pfs=sorted(glob.glob(d+'/**/*.parquet', recursive=True))
print('parquet files:', len(pfs), flush=True)
m=fx.load_srmr_model(getattr(fx,'SRMR_CONFIG',{})); tmp=tempfile.mkdtemp()
srmrs=[]; n=0
for pf in pfs:
    for batch in pq.ParquetFile(pf).iter_batches(batch_size=40, columns=['audio']):
        for a in batch.column('audio').to_pylist():
            if n>=120: break
            b=a.get('bytes') if isinstance(a,dict) else None
            if not b: continue
            try:
                arr,sr=sf.read(io.BytesIO(b))
                if arr.ndim>1: arr=arr.mean(1)
                if len(arr)<sr: continue
                p=os.path.join(tmp,'c.wav'); sf.write(p,arr,sr)
                v=fx.compute_srmr(p,m)
                if v is not None: srmrs.append(v); n+=1
                if n%30==0: print(n,'srmr done',flush=True)
            except Exception: continue
        if n>=120: break
    if n>=120: break
srmrs=np.array(srmrs)
print(f'CHiME-6 SRMR (n={len(srmrs)}): mean={srmrs.mean():.2f} p5/50/95=[{np.percentile(srmrs,5):.1f}/{np.percentile(srmrs,50):.1f}/{np.percentile(srmrs,95):.1f}]', flush=True)
print(f'-> interpolation vs aug-train [1.5,6.2] = {((srmrs>=1.5)&(srmrs<=6.2)).mean():.2f}', flush=True)
