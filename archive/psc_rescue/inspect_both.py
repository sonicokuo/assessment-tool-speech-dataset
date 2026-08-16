import torch, json, os, sys

def show(ck, label):
    print(f"\n===== {label}: {ck} =====")
    if not os.path.exists(ck):
        print("  MISSING"); return None
    d = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = d.get("config", {})
    keys = ["descriptions_path","descriptions_json","descriptions","data_dir","test_dir",
            "train_dir","val_dir","lm_name","adapter_variant","use_sections","tagged_mode",
            "beats_cached","section_query_mode","max_target_length","lambda_prose","lambda_nums",
            "lambda_mse","lora_r","lora_alpha","wandb_run_id","score_overlap_spans",
            "use_decoupled","decoupled","clean_f0","spec_augment","augment"]
    for k in keys:
        if k in cfg:
            print(f"  {k}: {cfg.get(k)}")
    # print any path-like config values not covered
    print("  -- other path-ish keys --")
    for k in sorted(cfg):
        if k in keys: continue
        v = cfg[k]
        if isinstance(v,str) and ("/" in v or v.endswith(".json") or "desc" in k.lower() or "dir" in k.lower() or "path" in k.lower()):
            print(f"  {k}: {v}")
    print("  epoch:", d.get("epoch"), "step:", d.get("step"))
    return cfg

show("/ocean/projects/cis260125p/shared/checkpoints/v9_lora_8b_dur/best.pt", "v9")
show("/ocean/projects/cis260125p/shared/checkpoints/v17_decoupled/best.pt", "v17")
