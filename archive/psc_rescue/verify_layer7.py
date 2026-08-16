"""BLOCKING integrity check for processed_layer7 (EXPLAINABILITY §1.17).

A concurrency bug briefly let two extractors target the same temp path. The crash mode was
safe, but interleaved writes to a shared temp are not detectable after the fact except by
loading every file. The alignment assert is the important one: overlap_info is COPIED VERBATIM
from the layer-24 cache, so a length mismatch would silently misalign the overlap channels
against the audio features -- a bug that trains fine and produces wrong science.
"""
import glob, os, sys, torch
B = "/ocean/projects/cis260125p/shared/data/processed_layer7"
REF = "/ocean/projects/cis260125p/shared/data/processed_corrected"
REQ = {"audio_features", "overlap_info", "overlap_segments", "filename"}
bad = []
for split in ["test", "val", "train"]:
    fs = sorted(glob.glob(f"{B}/{split}/*.pt"))
    stale = glob.glob(f"{B}/{split}/*.tmp")
    for i, f in enumerate(fs):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            assert REQ <= set(d), "missing keys"
            af, oi = d["audio_features"], d["overlap_info"]
            assert af.shape[1] == 1024, f"dim {af.shape}"
            assert af.shape[0] == oi.shape[0], f"MISALIGNED {af.shape[0]} vs {oi.shape[0]}"
            assert torch.isfinite(af).all(), "non-finite features"
        except Exception as e:
            bad.append((f, f"{type(e).__name__}: {e}"))
        if (i + 1) % 5000 == 0:
            print(f"  {split} {i+1}/{len(fs)} bad={len(bad)}", flush=True)
    print(f"{split}: {len(fs)} files, {len(bad)} bad so far, {len(stale)} stale .tmp", flush=True)
print(f"\nTOTAL CORRUPT: {len(bad)}")
for f, e in bad[:10]:
    print("  ", os.path.basename(f), e)
if bad:
    with open("/ocean/projects/cis260125p/shared/layer7_bad.txt", "w") as fh:
        fh.write("\n".join(f for f, _ in bad))
    print("  -> wrote layer7_bad.txt; delete those and re-run extract_layer7.py (it self-heals)")
    sys.exit(1)
print("PASS — layer-7 cache is safe to train on")
