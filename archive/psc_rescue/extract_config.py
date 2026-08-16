import sys, json, glob, os
import torch

CKPT_ROOT = "/ocean/projects/cis260125p/shared/checkpoints"

def load_cfg(run):
    for fn in ["best.pt", "last.pt"]:
        p = os.path.join(CKPT_ROOT, run, fn)
        if os.path.islink(p) and not os.path.exists(p):
            return {"_error": f"{fn} dangling symlink -> {os.readlink(p)}"}
        if os.path.exists(p):
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:
                return {"_error": f"load fail {fn}: {e}"}
            cfg = d.get("config", {})
            extra = {k: d.get(k) for k in ("epoch","best_val_sfs_f1","best_val_composite","wandb_run_id") if k in d}
            sd_keys = [k for k in d.keys() if k.endswith("_state_dict")]
            return {"_file": fn, "_extra": extra, "_state_dicts": sd_keys, "config": cfg}
    return {"_error": "no best.pt/last.pt"}

runs = sys.argv[1:]
out = {}
for r in runs:
    out[r] = load_cfg(r)
print(json.dumps(out, default=str, indent=1))
