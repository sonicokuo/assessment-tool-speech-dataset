"""Settle the f0 asymmetry: heavy tail or outliers?

§1.9 recorded f0_mean with err_red +0.526 attributed vs -0.061 random while its WIN RATE was
only 0.534. A win rate near chance with a large mean gap means most clips are coin flips and a
minority carry the effect. Mean alone cannot distinguish a genuine heavy tail (real, and the
win-rate statistic is simply the wrong lens) from a handful of outliers (not real).

Three views: mean (outlier-sensitive), median (robust), and a paired sign test on the per-clip
difference (distribution-free, tests whether attributed beats random MORE OFTEN than chance).
"""
import json, sys
from math import erfc, sqrt
import numpy as np

d = json.load(open(sys.argv[1]))
print("f0 ASYMMETRY — mean vs median vs paired sign test\n")
hdr = ("feature", "win", "mean_att", "med_att", "mean_rnd", "med_rnd", "sign p")
print(f"{hdr[0]:<15}{hdr[1]:>7}{hdr[2]:>10}{hdr[3]:>9}{hdr[4]:>10}{hdr[5]:>9}{hdr[6]:>9}")
print("-" * 69)
for f, v in d["per_feature"].items():
    a = np.array(v.get("per_clip_red_attributed", []))
    r = np.array(v.get("per_clip_red_random", []))
    if len(a) == 0:
        continue
    diff = a - r
    n = len(diff)
    pos = int((diff > 0).sum())
    z = (pos - n / 2) / sqrt(n / 4) if n else 0.0
    p = erfc(abs(z) / sqrt(2))
    print(f"{f:<15}{v['win_rate']:>7.3f}{v['err_reduction_attributed']:>10.3f}"
          f"{v['median_red_attributed']:>9.3f}{v['err_reduction_random']:>10.3f}"
          f"{v['median_red_random']:>9.3f}{p:>9.3f}")
print("\nREAD: if MEDIAN att >> MEDIAN rnd and sign p < 0.05 -> genuine, win rate was the wrong")
print("lens. If medians are equal and only MEANS differ -> outliers, and the null stands.")
