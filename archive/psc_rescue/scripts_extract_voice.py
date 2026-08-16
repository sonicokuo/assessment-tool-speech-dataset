import sys, glob, os, argparse, pandas as pd
sys.path.insert(0,'src')
from multiprocessing import Pool
_G={}
def _init():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import feature_extractor_mix as fx; _G['fx']=fx
def _proc(wav):
    fx=_G['fx']; stem=os.path.basename(wav)
    try:
        jj=fx.compute_jitter(wav)
        return dict(filename=stem, jitter_local_pct=jj.get('jitter_local_pct'),
                    jitter_rap_pct=jj.get('jitter_rap_pct'), shimmer=fx.compute_shimmer(wav), hnr=fx.compute_hnr(wav))
    except Exception:
        return dict(filename=stem, jitter_local_pct=None, jitter_rap_pct=None, shimmer=None, hnr=None)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--s1_dir'); ap.add_argument('--out_csv'); ap.add_argument('--workers',type=int,default=10)
    a=ap.parse_args(); wavs=sorted(glob.glob(a.s1_dir+'/*.wav')); print('voice extract:',len(wavs),flush=True)
    rows=[]
    with Pool(a.workers, initializer=_init) as pool:
        for i,r in enumerate(pool.imap_unordered(_proc,wavs,chunksize=8)):
            rows.append(r)
            if (i+1)%2000==0: print(f'{i+1}/{len(wavs)}',flush=True)
    pd.DataFrame(rows).to_csv(a.out_csv,index=False); print('WROTE',a.out_csv,'N=',len(rows),flush=True)
if __name__=='__main__': main()
