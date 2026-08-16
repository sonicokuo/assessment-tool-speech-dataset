import json, os, statistics as st
SH = "/ocean/projects/cis260125p/shared"
R5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
IP = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
runs = {"fw2_s73": "aux_l7_fw2_s73.json", "fw2_s42": "aux_l7_fw2_s42.json",
        "fw_s73(old)": "aux_l7audioonly_s73.json"}
tab = {}
for k, f in runs.items():
    p = os.path.join(SH, f)
    if not os.path.exists(p):
        print("MISSING", f); continue
    d = json.load(open(p))
    pf = d["summary"]["native"]["per_feature"]
    tab[k] = {n: pf.get(n) for n in R5 + IP}
ks = list(tab)
hdr = "feature".ljust(15) + "".join(k.rjust(14) for k in ks)
print(hdr); print("-" * len(hdr))
for n in R5 + IP:
    line = n.ljust(15)
    for k in ks:
        v = tab[k].get(n)
        line += ("%14.4f" % v) if isinstance(v, (int, float)) else "           n/a"
    print(line)
for pan, fs in (("ROBUST5", R5), ("ILL-POSED", IP)):
    line = pan.ljust(15)
    for k in ks:
        vals = [tab[k][n] for n in fs if isinstance(tab[k].get(n), (int, float))]
        line += ("%14.4f" % st.fmean(vals)) if vals else "           n/a"
    print(line)
print()
print("ref: tuned L7 ridge  robust5 0.7035 | ill-posed 0.5153  (layer sweep, 27800 clips)")
