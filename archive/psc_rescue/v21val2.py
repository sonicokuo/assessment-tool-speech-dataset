import json, csv, os, re
import numpy as np
from scipy.stats import spearmanr
SH = "/ocean/projects/cis260125p/shared"

def norm(k):
    return k[:-4] if k.endswith(".wav") else k

cf = {norm(k): v for k, v in json.load(open(SH + "/data/clean_features_dev.json")).items()}
CLEANKEY = {"srmr":"srmr","snr":"snr_db","speaking_rate":"praat_speaking_rate_syl_sec",
            "pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}
def fnum(d,k):
    try: return float(d[k]) if d.get(k) not in (None,"") else None
    except: return None
def srcc(items):
    a=[x for x,y in items]; b=[y for x,y in items]
    if len(a)<8: return None,len(a)
    return spearmanr(a,b)[0], len(a)

def parse(text):
    out={}
    m=re.search(r"(-?\d+\.?\d*)\s*dB", text);  out["snr"]=float(m.group(1)) if m else None
    m=re.search(r"SRMR\s+of\s+(-?\d+\.?\d*)", text); out["srmr"]=float(m.group(1)) if m else None
    m=re.search(r"speaking rate (?:is|of|runs to (?:a )?\w+)?\s*(-?\d+\.?\d*)\s*syl/sec", text)
    if not m: m=re.search(r"(-?\d+\.?\d*)\s*syl/sec", text)
    out["speaking_rate"]=float(m.group(1)) if m else None
    m=re.search(r"pause count is\s+(\d+)", text); out["pause_count"]=float(m.group(1)) if m else None
    m=re.search(r"pause rate is\s+(-?\d+\.?\d*)\s*per min", text); out["pause_rate"]=float(m.group(1)) if m else None
    return out

def analyze_prose(run, eps):
    print("\n############", run, "(parse generated vs clean) ############")
    for ep in eps:
        path = SH + "/checkpoints/%s/val_samples/epoch_00%d.json" % (run, ep)
        if not os.path.exists(path): continue
        dd = json.load(open(path))
        sel=["srmr","speaking_rate","pause_count","pause_rate"]
        rows={}
        for feat in sel+["snr"]:
            pc=[]; pt=[]
            for clip in dd:
                fn=norm(clip["filename"])
                g=parse(clip.get("generated",""))
                t=parse(clip.get("target",""))
                pr=g.get(feat); gt=t.get(feat)
                if pr is not None and gt is not None: pt.append((pr,gt))
                cv=fnum(cf[fn],CLEANKEY[feat]) if fn in cf else None
                if pr is not None and cv is not None: pc.append((pr,cv))
            rows[feat]=(srcc(pt),srcc(pc))
        mt=np.mean([rows[f][0][0] for f in sel if rows[f][0][0] is not None])
        mc=np.mean([rows[f][1][0] for f in sel if rows[f][1][0] is not None])
        sr=rows["srmr"][1][0]
        ncpc=rows["srmr"][1][1]
        print("  ep%d (n_clips=%d): PRED~target=%.3f  PRED~clean=%.3f  srmr~clean=%.3f (n=%d)" % (ep,len(dd),mt,mc,sr,ncpc))

analyze_prose("v21_observability",[1,2,3,4,5])
analyze_prose("v21repro_nomap",[1,2,3,4,5])
