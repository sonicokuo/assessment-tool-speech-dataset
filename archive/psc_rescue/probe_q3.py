import json, os
base="/ocean/projects/cis260125p/shared/checkpoints"
# inference_results are lists of per-clip records (no config). Check if any has config / look at test_dir from summary
# Identify q3_8b family LM by inspecting val_samples or a meta. Try inference_summary "config" key.
for r in ["q3_8b_concat","q3_8b_concat_v2","q3_8b_film_attn","q3_8b_film_attn_2L","q3_8b_film_attn_v2","q3_8b_film_attn_v3","q3_8b_film_mamba","q3_8b_film_mamba_v2","q3_8b_qformer","q3_8b_qformer_v2","v7_lora_8b"]:
    s=os.path.join(base,r,"inference_summary.json")
    cfg=None
    if os.path.exists(s):
        d=json.load(open(s))
        cfg=d.get("config")
    # also check val_samples dir for a config
    print(r, "summary_config:", json.dumps(cfg) if cfg else "NONE", "| test_dir:", (json.load(open(s)).get("test_dir") if os.path.exists(s) else "?"))
