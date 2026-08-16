import json, re, sys
d = json.load(open(sys.argv[1]))
rows = d if isinstance(d, list) else list(d.values())
tot = len(rows)
def gen(r):
    if isinstance(r, str): return r
    for k in ("generated", "generated_text", "prediction", "output"):
        if isinstance(r, dict) and r.get(k): return r[k]
    return ""
n_f0 = sum(1 for r in rows if re.search(r"[Ff]0\b|fundamental frequency|pitch", gen(r)))
n_hedge = sum(1 for r in rows if re.search(r"cannot|unable|not reliab|unreliab|withh|too much overlap", gen(r), re.I))
print(f"  clips: {tot}")
print(f"  mention f0/pitch: {n_f0} ({100*n_f0/max(tot,1):.1f}%)")
print(f"  contain a hedge/abstention phrase: {n_hedge} ({100*n_hedge/max(tot,1):.1f}%)")
print("  --- sample generation ---")
print("  ", gen(rows[0])[:420].replace("\n", " "))
