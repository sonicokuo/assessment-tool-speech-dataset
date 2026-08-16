import torch, glob
a = sorted(glob.glob("/ocean/projects/cis260125p/shared/data/processed_corrected/test/*.pt"))[0]
b = a.replace("processed_corrected", "processed_layer7")
da = torch.load(a, map_location="cpu", weights_only=False)
db = torch.load(b, map_location="cpu", weights_only=False)
print("  overlap_info identical? ", torch.equal(da["overlap_info"], db["overlap_info"]), " (MUST be True)")
print("  filename identical?     ", da["filename"] == db["filename"])
print("  T aligned?              ", db["audio_features"].shape[0] == db["overlap_info"].shape[0])
