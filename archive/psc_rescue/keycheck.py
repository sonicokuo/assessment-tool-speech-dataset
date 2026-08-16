import numpy as np
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_events_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
print("oracle event maps: clips =", len(names))
print("  example name:", names[0])
print("  feature keys:", sorted({k.split("/")[0] for k in o.files if "/" in k}))
sr = [k for k in o.files if k.startswith("speaking_rate/")]
v = np.asarray(o[sr[0]], dtype=float)
print(f"  speaking_rate map: len={v.size} nonzero={int((v>0).sum())} sum={v.sum():.4f} (expect 1.0)")
pc = np.asarray(o[[k for k in o.files if k.startswith("pause_count/")][0]], dtype=float)
print(f"  pause_count density: nonzero={int((pc>0).sum())} sum={pc.sum():.4f}")
s = np.load(f"{SH}/v2_grad.npz")
sal_stems = {k.split("/", 1)[1] for k in s.files if k.startswith("f0_sd/")}
print("  stem overlap oracle-vs-saliency:", len(set(names) & sal_stems), "of", len(names))
