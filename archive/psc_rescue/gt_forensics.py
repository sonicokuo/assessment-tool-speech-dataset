import json, re, csv
from scipy.stats import spearmanr
SH = "/ocean/projects/cis260125p/shared"

def norm(k):
    return k[:-4] if k.endswith(".wav") else k

def parse_obs(text):
    d = {}
    m = re.search(r"(-?\d+\.?\d*)\s*dB", text)
    if m: d["snr"] = float(m.group(1))
    m = re.search(r"SRMR\s+of\s+(-?\d+\.?\d*)", text)
    if m: d["srmr"] = float(m.group(1))
    m = re.search(r"speaking rate (?:is|of)\s+(-?\d+\.?\d*)", text)
    if m: d["speaking_rate"] = float(m.group(1))
    m = re.search(r"pause count is\s+(\d+)", text)
    if m: d["pause_count"] = float(m.group(1))
    m = re.search(r"pause rate is\s+(-?\d+\.?\d*)\s*per min", text)
    if m: d["pause_rate"] = float(m.group(1))
    return d

obs = {norm(k): parse_obs(v) for k, v in json.load(open(SH + "/data/descriptions_observability_dev.json")).items()}
cf = {norm(k): v for k, v in json.load(open(SH + "/data/clean_features_dev.json")).items()}

mix = {}
with open(SH + "/data/features_pyannote/dev.csv") as f:
    for row in csv.DictReader(f):
        mix[norm(row["filename"])] = row
mixc = {}
with open(SH + "/data/features_pyannote/dev_cleanf0.csv") as f:
    for row in csv.DictReader(f):
        mixc[norm(row["filename"])] = row

def fnum(d, k):
    try:
        return float(d[k]) if d.get(k) not in (None, "") else None
    except Exception:
        return None

def srcc(akeys, fa, fb):
    a = []; b = []
    for k in akeys:
        va = fa(k); vb = fb(k)
        if va is None or vb is None: continue
        a.append(va); b.append(vb)
    if len(a) < 10: return None, len(a)
    return spearmanr(a, b)[0], len(a)

keys = set(obs) & set(cf) & set(mix) & set(mixc)
print("common keys (obs/clean/mix/mixc):", len(keys))
specs = [
    ("srmr", "srmr", "srmr", "srmr"),
    ("snr", "snr", "snr_db", "snr_db"),
    ("speaking_rate", "speaking_rate", "praat_speaking_rate_syl_sec", "praat_speaking_rate_syl_sec"),
    ("pause_count", "pause_count", "praat_pause_count", "praat_pause_count"),
    ("pause_rate", "pause_rate", "praat_pause_rate_per_min", "praat_pause_rate_per_min"),
]
def fmt(x): return ("%.3f" % x) if x is not None else "  NA "
print("%-14s %15s %13s %10s %16s" % ("feat", "obs~mixCLEANF0", "obs~mix(dev)", "obs~clean", "clean~mixCLEANF0"))
for name, ko, kc, km in specs:
    r1, n1 = srcc(keys, lambda k: obs[k].get(ko), lambda k: fnum(mixc[k], km))
    r2, _ = srcc(keys, lambda k: obs[k].get(ko), lambda k: fnum(mix[k], km))
    r3, _ = srcc(keys, lambda k: obs[k].get(ko), lambda k: fnum(cf[k], kc))
    r4, _ = srcc(keys, lambda k: fnum(mixc[k], km), lambda k: fnum(cf[k], kc))
    print("%-14s %15s %13s %10s %16s  (n=%d)" % (name, fmt(r1), fmt(r2), fmt(r3), fmt(r4), n1))

# exact-match fraction obs vs mixc (identity test)
print("\n=== exact-value match: obs target vs dev_cleanf0.csv (identity check) ===")
for name, ko, km in [("srmr","srmr","srmr"),("snr","snr","snr_db"),("speaking_rate","speaking_rate","praat_speaking_rate_syl_sec"),("pause_rate","pause_rate","praat_pause_rate_per_min")]:
    nmatch = ntot = 0
    for k in keys:
        vo = obs[k].get(ko); vm = fnum(mixc[k], km)
        if vo is None or vm is None: continue
        ntot += 1
        if abs(vo - vm) < 0.01: nmatch += 1
    print("  %-14s exact(|d|<0.01): %d/%d = %.3f" % (name, nmatch, ntot, nmatch/ntot if ntot else 0))
