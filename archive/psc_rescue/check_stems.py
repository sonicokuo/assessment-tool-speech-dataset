import os
dirs = {
  "processed/test": "/ocean/projects/cis260125p/shared/data/processed/test",
  "processed_pyannote/test": "/ocean/projects/cis260125p/shared/data/processed_pyannote/test",
  "processed_aug/test": "/ocean/projects/cis260125p/shared/data/processed_aug/test",
}
sets = {}
for name, d in dirs.items():
    stems = set(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".pt"))
    sets[name] = stems
    print(f"{name}: {len(stems)} stems")
names = list(sets)
base = sets[names[0]]
for n in names[1:]:
    inter = base & sets[n]
    only_a = base - sets[n]
    only_b = sets[n] - base
    print(f"{names[0]} vs {n}: intersection={len(inter)}, only_{names[0]}={len(only_a)}, only_{n}={len(only_b)}")
# sample a few stems
print("sample stems processed/test:", sorted(list(sets[names[0]]))[:3])
