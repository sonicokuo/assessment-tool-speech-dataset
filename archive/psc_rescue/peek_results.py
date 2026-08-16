import json, os
def peek(path, label):
    print(f"\n===== {label}: {path} =====")
    if not os.path.exists(path):
        print("  MISSING"); return
    sz = os.path.getsize(path)
    print(f"  size={sz} bytes")
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print("  JSON load error:", e); return
    print(f"  n_entries={len(data)}")
    if data:
        e0 = data[0]
        print("  keys:", list(e0.keys()))
        print("  filename:", e0.get("filename"))
        tgt = e0.get("target","")
        gen = e0.get("generated","")
        print("  --- target[:400] ---")
        print("   ", repr(tgt[:400]))
        print("  --- generated[:300] ---")
        print("   ", repr(gen[:300]))
        print("  overlap_segments present:", "overlap_segments" in e0, e0.get("overlap_segments"))
peek("/ocean/projects/cis260125p/shared/checkpoints/v9_lora_8b_dur/inference_results.json", "v9 EXISTING")
peek("/ocean/projects/cis260125p/shared/checkpoints/v17_decoupled/inference_results.json", "v17")
peek("/ocean/projects/cis260125p/shared/checkpoints/v14_aug/inference_results.json", "v14")
