# Archived configs — RUN RECORDS, never edited, never re-run

Moved 2026-08-16. These are the ONLY configs that enabled the legacy model paths
(`use_sections: true`, `tagged_mode: true`), i.e. the only way `SectionQueryHead`, `SpecEncoder`,
`SupervisedSNRMapHead` and `TokenGroundingHead` are reachable at runtime.

**They are dead three times over:**
1. their checkpoints were deleted 2026-08-16 (`v13_section_warmup`, `v15v3_l02`,
   `v15_aug_2dmap` — evidence preserved in `results/archive_runs/`);
2. their data lineage no longer exists (`processed_pyannote`, `processed_aug_noise`,
   `processed_clean_train_s1clean` — the symlink farms pointing at them are broken, 3000 dead
   links per split);
3. `lambda_snr_map` and `lambda_token_grounding` are declared in 21 and 16 configs respectively
   and enabled in NONE, so those two heads were never live under any config in the tree.

**Do not edit these to "fix" them.** They are run records for weights that existed. Editing a
config after the fact is exactly how the fw/fw2 confusion was manufactured: a `.fw2.yaml` written
a day after three runs finished, repointing only eval, left three arms believed to have trained
on fw2 while their checkpoints record fw (`src/inference.py:607-646`).

## Why the legacy MODULES were left in src/

`src/model/{section_query,section_readout,spec_encoder,snr_map_head,token_grounding_head}.py`,
`src/training/decoupled_grounding.py` and `src/data/{section_tags,feature_tags}.py` remain in the
live tree even though nothing can now reach them. They are imported at MODULE TOP by
`src/train.py` and `src/inference.py`, so archiving them requires making ~10 imports lazy inside
config-guarded branches — surgery on the training entry point whose only payoff is tidiness, and
whose failure mode is a wasted PSC allocation. Archiving the configs above removes the misleading
part (a live-looking path to a dead lineage) at zero risk. Revisit only if train.py is being
refactored for another reason.
