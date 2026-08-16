import json, re, sys
d = json.load(open(sys.argv[1]))
rows = d if isinstance(d, list) else list(d.values())
def gen(r):
    if isinstance(r, str): return r
    for k in ("generated","generated_text","prediction","output"):
        if isinstance(r, dict) and r.get(k): return r[k]
    return ""
def name(r):
    if isinstance(r, dict):
        for k in ("filename","file","stem","name"):
            if r.get(k): return str(r[k])
    return ""
HEDGE = re.compile(r"cannot be reliab|not reported|unable|unreliab", re.I)
VALUE = re.compile(r"F0[^.]{0,40}?is\s+[0-9]+\.?[0-9]*|fundamental frequency[^.]{0,40}?[0-9]+\.?[0-9]*", re.I)
for label, pred in (("MIXTURE", lambda n: not n.endswith("_s1clean")),
                    ("CLEAN  ", lambda n: n.endswith("_s1clean"))):
    sub = [r for r in rows if pred(name(r).replace(".wav","").replace(".pt",""))]
    if not sub: continue
    h = sum(1 for r in sub if HEDGE.search(gen(r)))
    v = sum(1 for r in sub if VALUE.search(gen(r)))
    print(f"  {label} n={len(sub):>5}  hedged f0: {h:>5} ({100*h/len(sub):5.1f}%)   "
          f"emitted an f0 VALUE: {v:>5} ({100*v/len(sub):5.1f}%)")
