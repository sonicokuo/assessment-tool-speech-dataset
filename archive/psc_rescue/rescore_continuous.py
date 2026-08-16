"""D-item 28 — re-score attribution specificity against the CONTINUOUS overlap target.

Every specificity number recorded so far (§1.5) scored `|contribution|` against
`overlap_info[:,0]`, a BINARISED per-frame overlap flag. But
`compute_overlap_map_targets.py` already produced a CONTINUOUS per-frame overlap fraction
(verified: 7 distinct levels in [0,1]) which has been sitting unused. Continuous is strictly
the better reference: it distinguishes barely-overlapping from fully-overlapping frames instead
of crushing both into one bucket, so partial-overlap frames stop being mislabelled.

This cannot change the CAUSAL nulls (those used no reference at all). It only sharpens the
CORRELATIONAL specificity table.
"""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from eval.attribution_metrics import score_all  # noqa: E402

SH = "/ocean/projects/cis260125p/shared"
npz = np.load(f"{SH}/attribution_fw2.npz", allow_pickle=True)
C, OV_BIN, FEATS = npz["contrib"], npz["overlap"], [str(x) for x in npz["features"]]
names = [str(x) for x in npz["names"]]
print(f"[data] {C.shape[0]} clips, {len(FEATS)} features", flush=True)

tdir = {}
for split in ["test", "dev", "train"]:
    for f in glob.glob(f"{SH}/data/overlap_map_targets/{split}/*.pt"):
        tdir[os.path.splitext(os.path.basename(f))[0]] = f

OV_CONT = np.full_like(OV_BIN, np.nan)
hit = 0
for i, nm in enumerate(names):
    stem = os.path.splitext(os.path.basename(nm))[0]
    p = tdir.get(stem)
    if p is None:
        continue
    t = torch.load(p, map_location="cpu", weights_only=False)["overlap_map_target"].squeeze(-1)
    N = int(np.isfinite(OV_BIN[i]).sum())
    if N < 2:
        continue
    # pool the frame-rate target down to the prefix-token grid, matching how the binary
    # channel was pooled in attribution_capture.py
    pooled = torch.nn.functional.adaptive_avg_pool1d(t[None, None, :].float(), N)[0, 0].numpy()
    OV_CONT[i, :N] = pooled
    hit += 1
print(f"[match] continuous target found for {hit}/{len(names)} clips", flush=True)

for label, OV in (("BINARY (as recorded)", OV_BIN), ("CONTINUOUS (corrected)", OV_CONT)):
    r = score_all(C, OV, FEATS, n_others=16, n_perm=3, seed=0)
    oc = r["__ORACLE__"]["spec_twoside"]
    print(f"\n=== {label} ===  oracle ceiling {oc:.4f}")
    print(f"{'feature':<15}{'spec':>8}{'% ceil':>9}{'devR':>9}")
    for f in FEATS:
        print(f"{f:<15}{r[f]['spec_twoside']:>8.3f}{100*r[f]['spec_twoside']/oc:>8.0f}%"
              f"{r[f]['dev_corr']:>9.3f}")
