import sys, os, json, torch
r=sys.argv[1]
base="/ocean/projects/cis260125p/shared/checkpoints"
p=os.path.join(base,r,"best.pt")
keys=["lm_name","adapter_variant","use_lora","lora_r","lora_rank","r","lora_alpha","lambda_prose","lambda_nums","lambda_mse","use_sections","tagged_mode","use_duration","max_target_length","descriptions_path","beats_cached","epochs","num_epochs"]
try:
    ck=torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    cfg=ck.get("config",{}) if isinstance(ck,dict) else {}
    ep=ck.get("epoch") if isinstance(ck,dict) else None
    sub={k:cfg.get(k) for k in keys if k in cfg}
    # also dump any key containing lora
    lora_keys={k:v for k,v in cfg.items() if "lora" in k.lower() or k=="r"}
    print(r, "epoch="+str(ep), json.dumps(sub), "| LORA:", json.dumps(lora_keys))
except Exception as e:
    print(r, "ERR", repr(e)[:160])
