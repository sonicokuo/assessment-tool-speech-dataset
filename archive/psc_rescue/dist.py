import json
from collections import Counter
V21="/ocean/projects/cis260125p/shared/checkpoints/v21_observability/inference_results.json"
A  ="/ocean/projects/cis260125p/shared/checkpoints/v21repl_A/inference_results.json"
def claims(p):
    d=json.load(open(p)); out={}
    for e in d:
        cd={}
        for f,v in e["claims"]:
            if f not in cd: cd[f]=v
        out[e["filename"]]=cd
    return out, d
v21,dv=claims(V21); a,da=claims(A)

for feat in ["pause_count","srmr","snr","speaking_rate"]:
    print("="*20, feat, "="*20)
    for tag,c in [("v21",v21),("A",a)]:
        vals=[c[fn][feat] for fn in c if feat in c[fn]]
        if feat=="pause_count":
            print(f"  {tag}: n={len(vals)} top-value-counts:", Counter(round(x) for x in vals).most_common(8))
        else:
            vals_s=sorted(vals)
            n=len(vals)
            q=lambda p: vals_s[int(p*(n-1))]
            top=Counter(round(x,2) for x in vals).most_common(5)
            print(f"  {tag}: n={n} min={vals_s[0]:.3f} q25={q(.25):.3f} med={q(.5):.3f} q75={q(.75):.3f} max={vals_s[-1]:.3f} | top5 rounded: {top}")

# How many clips does A even mention pause_count vs emit overlap_ratio (hedging)?
print("\n=== feature emission counts (per model) ===")
for tag,d in [("v21",dv),("A",da)]:
    cnt=Counter()
    for e in d:
        seen=set()
        for f,v in e["claims"]:
            if f not in seen: cnt[f]+=1; seen.add(f)
    print(f"  {tag}:", dict(sorted(cnt.items(), key=lambda x:-x[1])))
