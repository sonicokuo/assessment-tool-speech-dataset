import json, os, statistics as st
SH = "/ocean/projects/cis260125p/shared"
R5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
IP = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
runs = {"unmasked_s73": "aux_l7_unmasked_s73.json",
        "fw2_s73": "aux_l7_fw2_s73.json",
        "fw2_s42": "aux_l7_fw2_s42.json"}
tab = {}
for k, f in runs.items():
    p = os.path.join(SH, f)
    if not os.path.exists(p):
        print("MISSING", f); continue
    tab[k] = json.load(open(p))["summary"]["native"]["per_feature"]
ks = list(tab)
print("%-15s%14s%14s%14s%12s" % ("feature", *ks, "unmask-fw2"))
print("-" * 71)
for n in R5 + IP:
    row = "%-15s" % n
    for k in ks:
        v = tab[k].get(n)
        row += ("%14.4f" % v) if isinstance(v, (int, float)) else "%14s" % "--"
    d = tab["unmasked_s73"].get(n), tab["fw2_s73"].get(n)
    row += ("%+12.4f" % (d[0] - d[1])) if all(isinstance(x, (int, float)) for x in d) else "%12s" % "--"
    print(row)
print()
RIDGE = {"ROBUST5": 0.7035, "ILL-POSED": 0.5153}
for pan, fs in (("ROBUST5", R5), ("ILL-POSED", IP)):
    vals = {k: st.fmean([tab[k][n] for n in fs if isinstance(tab[k].get(n), (int, float))]) for k in ks}
    line = "%-15s" % pan
    for k in ks:
        line += "%14.4f" % vals[k]
    line += "%+12.4f" % (vals["unmasked_s73"] - vals["fw2_s73"])
    print(line)
    gap = vals["unmasked_s73"] - RIDGE[pan]
    print("    vs TUNED L7 RIDGE %.4f  ->  unmasked margin %+.4f  (%s)"
          % (RIDGE[pan], gap, "WE WIN" if gap > 0 else "RIDGE WINS"))
