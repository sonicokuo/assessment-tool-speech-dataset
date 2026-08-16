"""Is the LOW window actually low-overlap? With 99.1% of mixtures at overlap>=0.5,
there may be no genuinely clean window, in which case HIGH vs LOW is not a control."""
import csv, json, sys, numpy as np
sys.path.insert(0, "src")
from data.f0_clean import parse_overlap_windows_samples
SR = 16000
SH = "/ocean/projects/cis260125p/shared"
rows = {r["filename"]: r for r in csv.DictReader(open(f"{SH}/data/features_corrected_merged/test.csv"))}
res = json.load(open(f"{SH}/causal_remix.json"))
import soundfile as sf, os
hi, lo = [], []
for r in res[:200]:
    fn = r["filename"]
    row = rows.get(fn)
    if not row: continue
    col = "overlap_segments_vad" if row.get("overlap_segments_vad") else "overlap_segments"
    win = parse_overlap_windows_samples(row.get(col) or "", SR)
    p = os.path.join(f"{SH}/data/audio_corrected/test", fn)
    if not os.path.exists(p): continue
    n = sf.info(p).frames
    ovl = np.zeros(n, dtype=np.float32)
    for (s, e) in win:
        ovl[max(0,int(s*SR)):min(n,int(e*SR))] = 1.0
    L = int(1.5 * SR)
    if n <= L: continue
    cs = np.concatenate([[0.0], np.cumsum(ovl)])
    st = np.arange(0, n - L, max(1, L // 4))
    mass = (cs[st + L] - cs[st]) / L
    hi.append(float(mass.max())); lo.append(float(mass.min()))
hi, lo = np.array(hi), np.array(lo)
print(f"n={len(hi)}")
print(f"HIGH window overlap fraction: median {np.median(hi):.3f}  mean {hi.mean():.3f}")
print(f"LOW  window overlap fraction: median {np.median(lo):.3f}  mean {lo.mean():.3f}")
print(f"clips where LOW window is >50% overlapped: {(lo>0.5).mean()*100:.1f}%")
print(f"clips where LOW window is <10% overlapped: {(lo<0.1).mean()*100:.1f}%")
