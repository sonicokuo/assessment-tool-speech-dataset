import torch, json, sys
ck = "/ocean/projects/cis260125p/shared/checkpoints/v9_lora_8b_dur/best.pt"
d = torch.load(ck, map_location="cpu", weights_only=False)
print("top-level keys:", list(d.keys()))
cfg = d.get("config", {})
print("=== full config (scalars) ===")
for k in sorted(cfg.keys()):
    v = cfg[k]
    if isinstance(v,(str,int,float,bool)) or v is None:
        print(f"  {k}: {v}")
print("=== selected ===")
for k in ["descriptions_path","descriptions_json","descriptions","data_dir","test_dir","train_dir","val_dir","lm_name","adapter_variant","use_sections","tagged_mode","beats_cached","section_query_mode","max_target_length","lambda_prose","lambda_nums","lambda_mse","lora_r","lora_alpha","wandb_run_id","spec_augment","use_bertscore"]:
    print(f"  {k}: {cfg.get(k, '<MISSING>')}")
print("epoch:", d.get("epoch"), "step:", d.get("step"))
