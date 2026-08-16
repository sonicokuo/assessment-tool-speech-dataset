import json, os, pandas as pd, numpy as np
SH='/ocean/projects/cis260125p/shared'
OUT=SH+'/data/features_corrected_merged'; os.makedirs(OUT,exist_ok=True)
VOICE=['jitter_local_pct','jitter_rap_pct','shimmer','hnr']
OVL=['overlap_ratio','overlap_segments','overlap_segments_vad','overlap_ratio_vad']
S1={'train':SH+'/data/features_corrected_s1clean.csv','dev':SH+'/data/features_corrected_dev_s1clean.csv','test':SH+'/data/features_corrected_test_s1clean.csv'}
for csv_sp,out_sp in [('train','train-100'),('dev','dev'),('test','test')]:
    f0=json.load(open(f'{SH}/data/clean_f0_{csv_sp}.json'))
    voice=pd.read_csv(f'{SH}/data/voice_{csv_sp}.csv').set_index('filename')
    ovl_df=pd.read_csv(f'{SH}/data/features/{out_sp}.csv')
    have_ovl=[c for c in OVL if c in ovl_df.columns]
    ovl=ovl_df.set_index('filename')[have_ovl] if have_ovl else None
    print(f'{out_sp}: overlap cols available={have_ovl}',flush=True)
    rows=[]
    def add(csv_path, suffix, zero_ovl):
        df=pd.read_csv(csv_path)
        for _,r in df.iterrows():
            fn=r['filename']; d=dict(r)
            d['f0_sd_hz']=f0.get(fn,{}).get('f0_sd_hz')
            for c in VOICE: d[c]=(voice.loc[fn,c] if fn in voice.index else None)
            if zero_ovl:
                d['overlap_ratio']=0.0; d['overlap_segments']=''; d['overlap_segments_vad']=''; d['overlap_ratio_vad']=0.0
            elif ovl is not None:
                for c in have_ovl: d[c]=(ovl.loc[fn,c] if fn in ovl.index else '')
            if suffix: d['filename']=fn.replace('.wav',suffix+'.wav')
            rows.append(d)
    add(f'{SH}/data/features_corrected_{csv_sp}.csv','',False)   # mix: real overlap
    add(S1[csv_sp],'_s1clean',True)                              # s1clean: zero overlap + suffix
    out=pd.DataFrame(rows); out.to_csv(f'{OUT}/{out_sp}.csv',index=False)
    print(f'{out_sp}: {len(out)} rows',flush=True)
