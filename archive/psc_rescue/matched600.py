import json, statistics as st
SH = "/ocean/projects/cis260125p/shared"
R5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
IP = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
m = json.load(open(f"{SH}/aux_l7_fw2_s73_MATCHED600.json"))["summary"]["native"]["per_feature"]
f = json.load(open(f"{SH}/aux_l7_fw2_s73.json"))["summary"]["native"]["per_feature"]
# LM free-decode values read off the scored panel (n=600, instrument GT)
lm = {"snr": 0.972, "srmr": 0.926, "speaking_rate": 0.683,
      "pause_count": 0.661, "pause_rate": 0.612, "f0_mean": 0.433}
print("%-15s%11s%12s%10s%9s" % ("feature", "AUX n=600", "AUX n=6000", "LM n=600", "LM-AUX"))
print("-" * 57)
for n in R5 + IP:
    a600, a6k, l = m.get(n), f.get(n), lm.get(n)
    row = "%-15s" % n
    row += ("%11.4f" % a600) if a600 is not None else "%11s" % "--"
    row += ("%12.4f" % a6k) if a6k is not None else "%12s" % "--"
    row += ("%10.4f" % l) if l is not None else "%10s" % "--"
    row += ("%+9.4f" % (l - a600)) if (l is not None and a600 is not None) else "%9s" % "--"
    print(row)
a = st.fmean([m[n] for n in R5]); b = st.fmean([f[n] for n in R5])
ai = st.fmean([m[n] for n in IP]); bi = st.fmean([f[n] for n in IP])
print()
print("ROBUST5   aux n=600 %.4f | aux n=6000 %.4f | LM free-decode n=600 0.7710" % (a, b))
print("ILL-POSED aux n=600 %.4f | aux n=6000 %.4f" % (ai, bi))
print()
print("subset effect on AUX (n=600 minus n=6000): %+.4f" % (a - b))
print("LM minus MATCHED aux on the SAME 600 clips: %+.4f" % (0.7710 - a))
