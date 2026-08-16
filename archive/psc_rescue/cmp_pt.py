import torch, json, numpy as np, os
stem = "1089-134686-0000_121-127105-0031"
paths = {
  "pyannote": f"/ocean/projects/cis260125p/shared/data/processed_pyannote/test/{stem}.pt",
  "aug": f"/ocean/projects/cis260125p/shared/data/processed_aug/test/{stem}.pt",
  "plain": f"/ocean/projects/cis260125p/shared/data/processed/test/{stem}.pt",
}
ds = {}
for k,p in paths.items():
    if not os.path.exists(p):
        print(k, "MISSING"); continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    ds[k] = d
    af = d["audio_features"]; oi = d.get("overlap_info")
    print(f"{k}: audio_features {tuple(af.shape)} mean={float(af.float().mean()):.5f} std={float(af.float().std()):.5f}; "
          f"overlap_info {tuple(oi.shape) if oi is not None else None}; keys={list(d.keys())}")
# compare audio_features between pyannote and aug
if "pyannote" in ds and "aug" in ds:
    a = ds["pyannote"]["audio_features"].float(); b = ds["aug"]["audio_features"].float()
    if a.shape == b.shape:
        print("pyannote vs aug audio_features: identical=", bool(torch.allclose(a,b)), " max_abs_diff=", float((a-b).abs().max()))
    else:
        print("pyannote vs aug audio_features SHAPE DIFFERS", a.shape, b.shape)
    oa = ds["pyannote"]["overlap_info"].float(); ob = ds["aug"]["overlap_info"].float()
    if oa.shape==ob.shape:
        print("overlap_info identical=", bool(torch.allclose(oa,ob)), " max_abs_diff=", float((oa-ob).abs().max()))
    else:
        print("overlap_info SHAPE DIFFERS", oa.shape, ob.shape)

# Check descriptions_aug.json has this clip + count test coverage
with open("/ocean/projects/cis260125p/shared/data/descriptions_aug.json") as f:
    aug = json.load(f)
with open("/ocean/projects/cis260125p/shared/data/descriptions_untagged_noseg.json") as f:
    v9desc = json.load(f)
print("\ndescriptions_aug.json: total keys=", len(aug))
print("descriptions_untagged_noseg.json: total keys=", len(v9desc))
# test stems
test_stems = set(os.path.splitext(f)[0] for f in os.listdir("/ocean/projects/cis260125p/shared/data/processed_aug/test") if f.endswith(".pt"))
def keyforms(stem):
    return [stem, stem+".wav"]
def covered(desc, stems):
    c=0
    for s in stems:
        if s in desc or (s+".wav") in desc: c+=1
    return c
print("aug covers test stems:", covered(aug, test_stems), "/", len(test_stems))
print("v9desc covers test stems:", covered(v9desc, test_stems), "/", len(test_stems))
# show this clip's target in each
for name,desc in [("aug",aug),("v9desc",v9desc)]:
    v = desc.get(stem) or desc.get(stem+".wav")
    print(f"\n{name}[{stem}] = {repr(v[:300]) if v else None}")
