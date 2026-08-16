import json, re, csv
from scipy.stats import spearmanr
SH = "/ocean/projects/cis260125p/shared"

def norm(k):
    return k[:-4] if k.endswith(".wav") else k

# ---- v21 inference results (TEST, 3000) ----
res = json.load(open(SH + "/checkpoints/v21_observability/inference_results.json"))
print("v21 inference_results: n=", len(res))
r0 = res[0]
print("keys:", list(r0.keys()))
print("per_feature[0]:", r0["per_feature"][0])

# Build per-clip dicts: pred (claimed) and stored-actual, keyed by feature
pred = {}   # fname -> {feat: claimed}
sact = {}   # fname -> {feat: actual}
for r in res:
    f = norm(r["filename"])
    pred[f] = {}
    sact[f] = {}
    for pf in r["per_feature"]:
        pred[f][pf["feature"]] = pf["claimed"]
        sact[f][pf["feature"]] = pf["actual"]

# ---- clean GT (TEST) ----
cf = {norm(k): v for k, v in json.load(open(SH + "/data/clean_features_test.json")).items()}
# ---- mix CSV (TEST) used to build obs target ----
mixc = {}
import os
# pick the test mix csv that matches obs test target; try test_cleanf0 then test
for cand in ["test_cleanf0.csv", "test.csv"]:
    p = SH + "/data/features_pyannote/" + cand
    if os.path.exists(p):
        d = {}
        with open(p) as fh:
            for row in csv.DictReader(fh):
                d[norm(row["filename"])] = row
        mixc[cand] = d
# ---- obs test target (parse) ----
def parse_obs(text):
    out = {}
    m = re.search(r"(-?\d+\.?\d*)\s*dB", text)
    if m: out["snr"] = float(m.group(1))
    m = re.search(r"SRMR\s+of\s+(-?\d+\.?\d*)", text)
    if m: out["srmr"] = float(m.group(1))
    m = re.search(r"speaking rate (?:is|of)\s+(-?\d+\.?\d*)", text)
    if m: out["speaking_rate"] = float(m.group(1))
    m = re.search(r"pause count is\s+(\d+)", text)
    if m: out["pause_count"] = float(m.group(1))
    m = re.search(r"pause rate is\s+(-?\d+\.?\d*)\s*per min", text)
    if m: out["pause_rate"] = float(m.group(1))
    return out
obs = {norm(k): parse_obs(v) for k, v in json.load(open(SH + "/data/descriptions_observability_test.json")).items()}

def fnum(d, k):
    try:
        return float(d[k]) if d.get(k) not in (None, "") else None
    except Exception:
        return None

CLEANKEY = {"srmr":"srmr","snr":"snr_db","speaking_rate":"praat_speaking_rate_syl_sec",
            "pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}
MIXKEY = {"srmr":"srmr","snr":"snr_db","speaking_rate":"praat_speaking_rate_syl_sec",
          "pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}

def srcc(items):
    a = [x for x, y in items]; b = [y for x, y in items]
    if len(a) < 10: return None, len(a)
    return spearmanr(a, b)[0], len(a)

feats = ["srmr","snr","speaking_rate","pause_count","pause_rate"]
fns = set(pred) & set(cf) & set(obs)
print("\ncommon test clips (pred/clean/obs):", len(fns))

print("\n%-14s %12s %12s %12s %12s %14s" % ("feat","PRED~clean","PRED~sAct","PRED~obs","sAct~clean","obs~clean"))
agg = {"PRED~clean":[], "PRED~obs":[]}
for feat in feats:
    ck = CLEANKEY[feat]; mk = MIXKEY[feat]
    pc=[]; ps=[]; po=[]; sc=[]; oc=[]
    for fn in fns:
        p = pred.get(fn,{}).get(feat); s = sact.get(fn,{}).get(feat)
        c = fnum(cf[fn], ck); o = obs[fn].get(feat)
        if p is not None and c is not None: pc.append((p,c))
        if p is not None and s is not None: ps.append((p,s))
        if p is not None and o is not None: po.append((p,o))
        if s is not None and c is not None: sc.append((s,c))
        if o is not None and c is not None: oc.append((o,c))
    def g(items):
        r,n = srcc(items); return ("%.3f(n=%d)"%(r,n)) if r is not None else "NA"
    print("%-14s %12s %12s %12s %12s %14s" % (feat, g(pc), g(ps), g(po), g(sc), g(oc)))

# headline mean-excl-snr PRED~clean
import numpy as np
def meancorr(target_fn, label):
    vals=[]
    for feat in feats:
        if feat=="snr": continue
        ck=CLEANKEY[feat]
        items=[]
        for fn in fns:
            p = pred.get(fn,{}).get(feat); t = target_fn(fn,feat,ck)
            if p is not None and t is not None: items.append((p,t))
        r,n=srcc(items)
        if r is not None: vals.append(r)
    print("  mean-excl-snr %s = %.3f over %d feats" % (label, np.mean(vals), len(vals)))

print("\n=== v21 headline reconstruction ===")
meancorr(lambda fn,feat,ck: fnum(cf[fn],ck), "PRED~clean")
meancorr(lambda fn,feat,ck: sact.get(fn,{}).get(feat), "PRED~storedActual")
