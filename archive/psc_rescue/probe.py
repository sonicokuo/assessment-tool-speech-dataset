import torch, os, glob, json
d1="/ocean/projects/cis260125p/shared/data/processed_clean_control"
d2="/ocean/projects/cis260125p/shared/data/processed_pyannote/test"
for tag,d in [("clean_control",d1),("pyannote_test",d2)]:
    pts=sorted(glob.glob(os.path.join(d,"*.pt")))
    print(f"== {tag}: {len(pts)} .pt files, dir={d}")
    if not pts: continue
    # load a known clip
    fn="1089-134686-0000_121-127105-0031"
    cand=[p for p in pts if os.path.basename(p).startswith(fn)]
    p=cand[0] if cand else pts[0]
    x=torch.load(p, map_location="cpu", weights_only=False)
    oi=x["overlap_info"]
    print(f"   sample={os.path.basename(p)} overlap_info shape={tuple(oi.shape)}")
    print(f"   overlap_info col-means:", [round(float(oi[:,c].mean()),4) for c in range(oi.shape[1])])
    print(f"   is_overlap frac (col0):", round(float((oi[:,0]>0.5).float().mean()),4))
    print(f"   keys:", list(x.keys()))
