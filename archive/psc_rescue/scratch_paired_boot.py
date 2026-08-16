import json, random, sys
sys.path.insert(0, "src")
from scipy.stats import spearmanr
import csv
import eval.sfs as sfs

SH = "/ocean/projects/cis260125p/shared"
P = sfs.ClaimParser()
def parse_first(text):
    out = {}
    for c in P.parse(text or ""):
        if c.feature not in out: out[c.feature] = c.value
    return out

COLS = {"snr":"snr_db","srmr":"srmr","speaking_rate":"praat_speaking_rate_syl_sec",
        "pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}
ROBUST = list(COLS)
gt = {}
for row in csv.DictReader(open(f"{SH}/data/features_corrected_merged/test.csv")):
    stem = (row.get("filename") or "").rsplit(".",1)[0]
    d = {}
    for f,c in COLS.items():
        v = row.get(c)
        if v not in (None,"","nan"):
            try:
                fv=float(v)
                if fv==fv: d[f]=fv
            except ValueError: pass
    gt[stem]=d

def load(path):
    d = json.load(open(path))
    return {str(r["filename"]).replace(".wav",""): parse_first(r.get("generated","")) for r in d}

bl = load(f"{SH}/checkpoints/full/bsigma_attnconcat_seed73/inference_results.json")
ao = load(f"{SH}/scratch_ao_on_bl1250.json")
stems = [s for s in bl if s in ao and s in gt]
print("paired stems:", len(stems))

def srcc_robust(preds, ss):
    vals=[]
    for f in ROBUST:
        xs=[preds[s][f] for s in ss if f in preds[s] and f in gt[s]]
        ys=[gt[s][f]    for s in ss if f in preds[s] and f in gt[s]]
        if len(xs)>=10: vals.append(float(spearmanr(xs,ys).correlation))
    return sum(vals)/len(vals)

d_full = srcc_robust(ao, stems) - srcc_robust(bl, stems)
print(f"full-1250 paired delta (audioonly - oracle, free decode): {d_full:+.4f}")

rng = random.Random(1)
# paired delta at n=200 (val-sized subsets of the SAME clips for both arms)
for n in (200,):
    deltas=[]
    for _ in range(400):
        ss = rng.sample(stems, n)
        deltas.append(srcc_robust(ao, ss) - srcc_robust(bl, ss))
    mu=sum(deltas)/len(deltas)
    sd=(sum((x-mu)**2 for x in deltas)/(len(deltas)-1))**0.5
    deltas.sort()
    frac_pos = sum(1 for x in deltas if x>0)/len(deltas)
    print(f"paired delta @n={n}: mean={mu:+.4f} SD={sd:.4f} CI95=[{deltas[9]:+.4f},{deltas[389]:+.4f}] P(delta>0)={frac_pos:.3f}")
# bootstrap CI on the full-1250 delta
deltas=[]
for _ in range(400):
    ss=[stems[rng.randrange(len(stems))] for _ in range(len(stems))]
    deltas.append(srcc_robust(ao, ss) - srcc_robust(bl, ss))
deltas.sort()
print(f"delta CI95 @n=1250 (clip bootstrap): [{deltas[9]:+.4f},{deltas[389]:+.4f}]")
