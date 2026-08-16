"""Direct, convention-free check of the SNR gain-slope directions.

The regression applied a sign to the x-axis, which is exactly how I mislabelled the
targets. So read the RAW predictions instead: for a POSITIVE dB step, did the model's
SNR estimate go UP or DOWN in each condition? Physical truth:
    noise  louder -> true SNR DOWN   -> faithful model predicts LOWER
    speech louder -> true SNR UP     -> faithful model predicts HIGHER
A loudness reader gives the SAME direction for both.
"""
import json, numpy as np
SH = "/ocean/projects/cis260125p/shared"
rows = json.load(open(f"{SH}/snr_gain_slopes.json"))
print(f"n clips = {len(rows)}\n")
print(f"{'condition':<14}{'dB step':>9}{'mean d_yhat':>14}{'frac moving DOWN':>20}   n")
for kind in ("noise", "speech", "both"):
    for db in ("-6.0", "-3.0", "3.0", "6.0"):
        d = [r[kind][db] - r["base"] for r in rows if db in r.get(kind, {})]
        if len(d) < 20:
            continue
        d = np.array(d, float)
        print(f"{kind:<14}{db:>9}{d.mean():>14.4f}{(d < 0).mean():>20.3f}   {d.size}")
    print()
print("EXPECTED for a genuine ratio reader:")
print("  noise  +dB -> d_yhat NEGATIVE (true SNR fell)")
print("  speech +dB -> d_yhat POSITIVE (true SNR rose)")
print("A loudness tracker moves the SAME way in both.")
