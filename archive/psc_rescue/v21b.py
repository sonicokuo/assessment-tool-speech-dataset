import json, csv, re
import numpy as np
from scipy.stats import spearmanr
SH = "/ocean/projects/cis260125p/shared"
def norm(k): return k[:-4] if k.endswith(".wav") else k

res = json.load(open(SH + "/checkpoints/v21_observability/inference_results.json"))
cf = {norm(k): v for k, v in json.load(open(SH + "/data/clean_features_test.json")).items()}
mixc = {}
with open(SH + "/data/features_pyannote/test_cleanf0.csv") as fh:
    for row in csv.DictReader(fh):
        mixc[norm(row["filename"])] = row

# obs test target parse
def parse(text):
    out={}
    m=re.search(r"(-?\d+\.?\d*)\s*dB", text); out["snr"]=float(m.group(1)) if m else None
    m=re.search(r"SRMR\s+of\s+(-?\d+\.?\d*)", text); out["srmr"]=float(m.group(1)) if m else None
    m=re.search(r"speaking rate (?:is|of)\s+(-?\d+\.?\d*)", text)
    if not m: m=re.search(r"(-?\d+\.?\d*)\s*syl/sec", text)
    out["speaking_rate"]=float(m.group(1)) if m else None
    m=re.search(r"pause count is\s+(\d+)", text); out["pause_count"]=float(m.group(1)) if m else None
    m=re.search(r"pause rate is\s+(-?\d+\.?\d*)\s*per min", text); out["pause_rate"]=float(m.group(1)) if m else None
    return out

CK={"srmr":"srmr","snr":"snr_db","speaking_rate":"praat_speaking_rate_syl_sec","pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}
def fnum(d,k):
    try: return float(d[k]) if d.get(k) not in (None,"") else None
    except: return None

# (b) Is v21 stored 'actual' == obs target == mix CSV? identity check
feats=["srmr","snr","speaking_rate","pause_count","pause_rate"]
print("=== (b) v21 inference stored 'actual'  vs  obs-target  vs  mix-CSV(test_cleanf0)  vs  clean ===")
print("%-14s %18s %16s %14s" % ("feat","sAct==mixCSV(exact)","sAct~mixCSV","sAct~clean"))
for feat in feats:
    nx=ntot=0; sm=[]; sc=[]
    for r in res:
        fn=norm(r["filename"])
        sact=None
        for pf in r["per_feature"]:
            if pf["feature"]==feat: sact=pf["actual"]
        if sact is None: continue
        mv=fnum(mixc.get(fn,{}), CK[feat]); cv=fnum(cf.get(fn,{}), CK[feat])
        if mv is not None:
            ntot+=1
            if abs(sact-mv)<0.02: nx+=1
            sm.append((sact,mv))
        if cv is not None: sc.append((sact,cv))
    def g(items):
        a=[x for x,y in items];b=[y for x,y in items]
        return spearmanr(a,b)[0] if len(a)>=10 else float("nan")
    print("%-14s %18s %16.3f %14.3f" % (feat, "%d/%d=%.3f"%(nx,ntot,nx/ntot if ntot else 0), g(sm), g(sc)))
