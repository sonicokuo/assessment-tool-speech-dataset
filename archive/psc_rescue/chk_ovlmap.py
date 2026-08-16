import torch, glob, os
d = "/ocean/projects/cis260125p/shared/data/overlap_map_targets/test"
fs = sorted(glob.glob(f"{d}/*.pt"))
x = torch.load(fs[0], map_location="cpu", weights_only=False)
print("files:", len(fs), "type:", type(x).__name__)
if isinstance(x, dict):
    for k, v in x.items():
        sh = tuple(v.shape) if hasattr(v, "shape") else v
        dt = str(v.dtype) if hasattr(v, "dtype") else ""
        print(f"  {k}: {sh} {dt}")
        if hasattr(v, "numel") and v.numel() > 4 and v.dtype.is_floating_point:
            print(f"      min={float(v.min()):.4f} max={float(v.max()):.4f} "
                  f"uniq={len(torch.unique(v))}  (>2 uniq = genuinely CONTINUOUS)")
