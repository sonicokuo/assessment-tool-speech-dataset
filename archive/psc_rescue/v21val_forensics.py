import json, csv, os
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

# detect val_samples schema for v21
p = SH + "/checkpoints/v21_observability/val_samples/epoch_004.json"
d = json.load(open(p))
print("v21 val_samples epoch4: type", type(d), "n", len(d))
print("clip0 keys:", list(d[0].keys()))
pfa = d[0].get("per_feature_abs_error")
print("has per_feature_abs_error:", pfa is not None)
if pfa: print("pfa srmr:", pfa.get("srmr"))

def analyze(run, eps):
    print("\n############", run, "############")
    for ep in eps:
        path = SH + "/checkpoints/%s/val_samples/epoch_00%d.json" % (run, ep)
        if not os.path.exists(path): continue
        dd = json.load(open(path))
        sel = ["srmr","speaking_rate","pause_count","pause_rate"]
        rows={}
        for feat in sel+["snr","overlap_ratio"]:
            pt=[]; pc=[]
            for clip in dd:
                fn=norm(clip["filename"]); pf=clip.get("per_feature_abs_error",{})
                if feat not in pf: continue
                pr=pf[feat]["pred"]; gt=pf[feat]["gt"]
                if pr is not None and gt is not None: pt.append((pr,gt))
                cv = fnum(cf[fn], CLEANKEY[feat]) if (feat in CLEANKEY and fn in cf) else None
                if pr is not None and cv is not None: pc.append((pr,cv))
            rows[feat]=(srcc(pt), srcc(pc))
        mt=np.mean([rows[f][0][0] for f in sel if rows[f][0][0] is not None])
        mc=np.mean([rows[f][1][0] for f in sel if rows[f][1][0] is not None])
        srmr_clean = rows["srmr"][1][0]
        print("  ep%d: mean-excl-snr PRED~target=%.3f  PRED~clean=%.3f  (srmr~clean=%.3f)" % (ep, mt, mc, srmr_clean))

analyze("v21_observability", [1,2,3,4,5])
analyze("v21repro_nomap", [1,2,3,4,5])
