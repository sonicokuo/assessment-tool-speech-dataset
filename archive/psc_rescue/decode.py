import json
V21="/ocean/projects/cis260125p/shared/checkpoints/v21_observability/inference_results.json"
A  ="/ocean/projects/cis260125p/shared/checkpoints/v21repl_A/inference_results.json"
def look(p,tag):
    d=json.load(open(p))
    e=d[0]
    print("="*20,tag,"="*20)
    print("top-level keys per clip:", list(e.keys()))
    print("has target:", bool(e.get("target")))
    print("target[:160]:", repr(e.get("target","")[:160]))
    # correlation of f0 emission with overlap hedging
    f0=sum(1 for x in d if any(f=="f0_mean" for f,_ in x["claims"]))
    ov=sum(1 for x in d if any(f=="overlap_ratio" for f,_ in x["claims"]))
    print(f"f0_mean emitted: {f0}/3000 ; overlap_ratio emitted: {ov}/3000")
look(V21,"v21"); look(A,"A")
# Are the TARGETS the same across the two json (same GT prose)? if so GT is shared.
d1={x["filename"]:x.get("target","") for x in json.load(open(V21))}
d2={x["filename"]:x.get("target","") for x in json.load(open(A))}
same=sum(1 for fn in d1 if fn in d2 and d1[fn]==d2[fn])
print(f"\nIdentical target prose across the two json: {same}/3000")
# show one where they differ
for fn in d1:
    if fn in d2 and d1[fn]!=d2[fn]:
        print("DIFF example:",fn)
        print("  v21 target:",repr(d1[fn][:200]))
        print("  A   target:",repr(d2[fn][:200]))
        break
