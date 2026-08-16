import json, sys
d = json.load(open("/ocean/projects/cis260125p/shared/layer_sweep_full.json"))
pl = d["per_layer"]
FE = ["snr","srmr","f0_mean","f0_sd","speaking_rate","pause_count","pause_rate","jitter","shimmer","hnr","overlap_ratio"]
print("Best layer PER FEATURE (full-train sweep, 27800/6000):\n")
print(f"{'feature':<15}{'best layer':>11}{'best SRCC':>11}{'layer24':>10}{'gain':>9}")
for f in FE:
    vals = [(int(k), pl[k].get(f)) for k in pl if pl[k].get(f) is not None]
    if not vals: continue
    bl, bv = max(vals, key=lambda kv: kv[1])
    l24 = pl["24"].get(f)
    print(f"{f:<15}{bl:>11}{bv:>11.3f}{l24:>10.3f}{bv-l24:>+9.3f}")
print()
ls = sorted(int(k) for k in pl)
print("robust5 / ill-posed by layer:")
for k in ls:
    print(f"  L{k:<3} robust5 {pl[str(k)]['_robust5']:.4f}   ill-posed {pl[str(k)]['_illposed']:.4f}")
