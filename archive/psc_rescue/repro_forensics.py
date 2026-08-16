import json, csv, os
import numpy as np
from scipy.stats import spearmanr
SH = "/ocean/projects/cis260125p/shared"

def norm(k):
    return k[:-4] if k.endswith(".wav") else k

cf = {norm(k): v for k, v in json.load(open(SH + "/data/clean_features_dev.json")).items()}
f0 = {norm(k): v for k, v in json.load(open(SH + "/data/clean_f0_dev.json")).items()}
mixc = {}
with open(SH + "/data/features_pyannote/dev_cleanf0.csv") as fh:
    for row in csv.DictReader(fh):
        mixc[norm(row["filename"])] = row

CLEANKEY = {"srmr":"srmr","snr":"snr_db","speaking_rate":"praat_speaking_rate_syl_sec",
            "pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}

def fnum(d, k):
    try:
        return float(d[k]) if d.get(k) not in (None, "") else None
    except Exception:
        return None

def srcc(items):
    a=[x for x,y in items]; b=[y for x,y in items]
    if len(a)<8: return None,len(a)
    return spearmanr(a,b)[0], len(a)

for run in ["v21repro_nomap", "v21repro_map"]:
    print("\n############", run, "############")
    for ep in [4, 5]:
        path = SH + "/checkpoints/%s/val_samples/epoch_00%d.json" % (run, ep)
        if not os.path.exists(path): continue
        d = json.load(open(path))
        feats = ["srmr","snr","speaking_rate","pause_count","pause_rate","overlap_ratio"]
        # PRED~target(gt in file)  and  PRED~clean
        rows = {}
        for feat in feats:
            pt=[]; pc=[]
            for clip in d:
                fn = norm(clip["filename"])
                pfa = clip.get("per_feature_abs_error", {})
                if feat not in pfa: continue
                pr = pfa[feat]["pred"]; gt = pfa[feat]["gt"]
                if pr is not None and gt is not None: pt.append((pr,gt))
                # clean
                cv=None
                if feat=="overlap_ratio":
                    cv=None  # no clean GT
                elif feat in CLEANKEY and fn in cf:
                    cv=fnum(cf[fn], CLEANKEY[feat])
                if pr is not None and cv is not None: pc.append((pr,cv))
            rt,nt=srcc(pt); rc,nc=srcc(pc)
            rows[feat]=(rt,nt,rc,nc)
        print("  epoch %d:" % ep)
        print("    %-14s %16s %16s" % ("feat","PRED~target(gt)","PRED~clean"))
        for feat in feats:
            rt,nt,rc,nc = rows[feat]
            st = ("%.3f(n=%d)"%(rt,nt)) if rt is not None else "NA"
            sc = ("%.3f(n=%d)"%(rc,nc)) if rc is not None else "NA"
            print("    %-14s %16s %16s" % (feat, st, sc))
        # mean-excl-snr,excl-overlap (the 4 features that have clean GT: srmr,speaking,pause_count,pause_rate)
        sel=["srmr","speaking_rate","pause_count","pause_rate"]
        mt=np.mean([rows[f][0] for f in sel if rows[f][0] is not None])
        mc=np.mean([rows[f][2] for f in sel if rows[f][2] is not None])
        # the in-training metric is mean over selection excl snr -> includes overlap_ratio? compute both
        sel_incl_ov=["srmr","speaking_rate","pause_count","pause_rate","overlap_ratio"]
        mt_ov=np.mean([rows[f][0] for f in sel_incl_ov if rows[f][0] is not None])
        print("    -> mean-excl-snr PRED~target  (4 feats) = %.3f ; (incl overlap=%.3f)" % (mt, mt_ov))
        print("    -> mean-excl-snr PRED~clean   (4 feats) = %.3f" % mc)

# does v21repro emit f0 at all? scan epoch4 generated text
print("\n=== f0 emission check (v21repro_nomap epoch4 generated) ===")
d = json.load(open(SH + "/checkpoints/v21repro_nomap/val_samples/epoch_004.json"))
import re
nf0 = sum(1 for c in d if re.search(r"F0 mean is", c.get("generated","")))
print("  clips emitting 'F0 mean is':", nf0, "/", len(d))
print("  sample generated[0]:", d[0]["generated"][:400])
