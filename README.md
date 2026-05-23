# AQUA-NL — Adaptive Audio Quality Assessment and Description in Natural Language

CMU 11-785 IDL project targeting EMNLP. Speech-quality reasoning with
**evidence-grounded explanations**: take a mixture waveform, generate a
section-structured natural-language description of quality, and produce one
cross-attention map per section back to the input spectrogram.

```
audio  →  WavLM-Large (frozen)  →  Adapter (Conv8x + Mamba + FiLM)  →  audio prefix tokens
                                                                          ↓
                              Qwen3-8B + LoRA r=16  →  natural-language description
                                                                          ↓
                            (post-hoc) attention extraction from LM layers
                                                                          ↓
                                       per-section attention maps  +  prose
```

Two contributions:

1. **Overlap-aware quality description.** FiLM conditioning modulates the
   audio prefix using per-frame VAD-derived overlap context, so the model
   hedges ("F0 estimates unreliable") during overlap rather than
   hallucinating values. Per-section attention maps for the paper figure
   are extracted post-hoc from the LM's native attention layers (the
   earlier `<sec_*>`-tag + `SectionQueryHead` route is preserved as an
   optional ablation toggle — see `use_sections: true`).
2. **Signal Faithfulness Score (SFS).** Regex-parses numerical claims
   from the generated prose and scores them against per-feature ground-truth
   tolerances.

## Quick start (PSC Bridges-2)

```bash
# 1. Grab a GPU
interact -p GPU-shared --gres=gpu:h100-80:1 -t 8:00:00 -A cis260125p

# 2. Activate env + move into repo
module load anaconda3
conda activate /ocean/projects/cis260125p/shared/envs/project
export PYTHONNOUSERSITE=1
export SHARED=/ocean/projects/cis260125p/shared
cd $SHARED/assessment-tool-speech-dataset
git pull --ff-only origin main

# 3. Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 4. Run the test suites
python -m pytest tests/ -v
python scripts/test_fix_descriptions.py
python scripts/test_fix_overlap_csv.py
python scripts/test_build_descriptions_deterministic.py
```

Shorter conda activation name (one-time):

```bash
conda config --append envs_dirs /ocean/projects/cis260125p/shared/envs
# from now on: conda activate project
```

Allocation check:

```bash
groups | tr ' ' '\n' | grep cis260125p
```

If empty, the PI must add you to `cis260125p` before `interact -A cis260125p` works.

Valid GPU types on Bridges-2:

| Tag | Memory |
|---|---|
| `v100-16` | 16 GB |
| `v100-32` | 32 GB |
| `l40s-48` | 48 GB |
| `h100-80` | 80 GB |

The current pipeline assumes `h100-80` at `batch_size=6` with Qwen3-8B + LoRA r=16 + gradient checkpointing. Smaller GPUs need `--batch_size 4 --gradient_accumulation_steps 2`. The 8B frozen backbone (~16 GB in bf16) is the memory floor; LoRA itself only adds ~45M trainable params.

## Architecture

| Component | Choice | Why |
|---|---|---|
| LM | Qwen/Qwen3-8B + LoRA r=16 (default) | LoRA preserves the LM's instruction-following prior so it doesn't mode-collapse into off-topic ramble on weakly-conditioning clips; ~45M trainable on top of 8B frozen pretrained weights. v6 (1.7B + full FT) showed this is essential at 19,900-clip dataset scale. |
| Audio prefix encoder | WavLM-Large (frozen) | Produces the LM audio prefix |
| Adapter | Conv8× compression + Mamba context + FiLM overlap conditioning | Compresses WavLM frames 8× while injecting per-frame overlap-reliability signal |
| Per-section attention | Post-hoc extraction from LM's native attention layers (default) | Parses generated prose for section spans, aggregates LM attention from those spans back to audio prefix tokens. Cleaner than the original `SectionQueryHead` path, which destabilized training when coupled with `tagged_mode=true` |
| Spec encoder (optional, ablation row) | BEATs (Microsoft, vendored under `src/beats/`) | Only loaded when `use_sections=true`. Frozen, 2D patch grid for the legacy cross-attention design. |
| Loss | `λ_prose · lm_ce(prose) + λ_nums · lm_ce(nums) + λ_mse · masked_mse(scalar regression)` | Three concurrent signals; nums target concentrates digit-tokenization gradient; MSE bypasses LM entirely to feed the adapter clean scalar gradient |

### Quality-feature taxonomy (6 sections, 8 SFS-scored scalars)

| Section | Inner feature tag(s) | CSV column |
|---|---|---|
| `<sec_noise>`   | `<f_snr>` | `snr_db` |
| `<sec_reverb>`  | `<f_srmr>` | `srmr` |
| `<sec_pitch>`   | `<f_f0_mean>`, `<f_f0_sd>` | `f0_mean_hz`, `f0_sd_hz` |
| `<sec_tempo>`   | `<f_speaking_rate>` | `praat_speaking_rate_syl_sec` |
| `<sec_pauses>`  | `<f_pause_count>`, `<f_pause_rate>` | `praat_pause_count`, `praat_pause_rate_per_min` |
| `<sec_overlap>` | `<f_overlap_ratio>`, `<f_overlap_segments>` (with `<r>` per range) | `overlap_ratio_vad`, `overlap_segments_vad` |

Special-token vocabulary added before training: 6 section opens + 1 shared `</sec>` close + 9 feature opens + 1 shared `</f>` close + `<r>`/`</r>` = 19 tokens. Registered via `tokenizer.add_tokens(special_tokens=False)` so they survive `decode(skip_special_tokens=True)` (parser and attention hook both need them visible).

### Overlap source-of-truth split

Two distinct columns in the feature CSV:

| Column | Source | Used for |
|---|---|---|
| `overlap_segments` / `overlap_ratio` | Pyannote on the **mix** | Model **input** (`overlap_info` channels written by `src/preprocess.py`). Same distribution at inference time on cross-domain audio (AMI etc.) where stems aren't available. |
| `overlap_segments_vad` / `overlap_ratio_vad` | Silero VAD on s1 + s2 **stems** | Description **GT** (read by `scripts/build_descriptions_deterministic.py`). Oracle labels; the model must learn to bridge from noisy pyannote input to clean VAD output. |

This avoids the trivial-copy data leakage where the model could memorize its own input channels and inflate SFS on overlap features.

## Data pipeline

5 steps. All scripts are idempotent except step 3 (which always overwrites `descriptions.json`).

```
audio  → [feature_extractor_mix]  → features_pyannote/<split>.csv
                                            │
       [fix_overlap_csv]   (adds *_vad columns from Silero on s1/s2 stems)
                                            │
[build_descriptions_deterministic]  →  descriptions.json   (19,900 entries; no LLM in the loop)
                                            │
                  [preprocess]    →  processed_pyannote/{train,val,test}/*.pt
                                            │           (audio_features + overlap_info)
            [preprocess_beats]    →  same .pt files (adds beats_patches key)
```

### Step 1 — feature extraction

Produces pyannote-derived overlap as the input distribution. Already done on PSC; rebuild only if recomputing from scratch.

```bash
# Run from inside an H100 interact (compute_overlap_pyannote uses CUDA)
for split in train-100 dev test; do
  python src/feature_extractor_mix.py \
    --audio_dir      $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split/mix_clean \
    --libri2mix_root $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split \
    --overlap        pyannote --hf_token $HF_TOKEN \
    --output         $SHARED/data/features_pyannote/${split}.csv
done
```

Needs `HF_TOKEN` (accept terms once at https://hf.co/pyannote/segmentation-3.0).

### Step 2 — add VAD-derived overlap columns

Silero VAD on `s1`/`s2` stems, written to **new** columns `overlap_segments_vad` and `overlap_ratio_vad` alongside the pyannote ones. Idempotent: skips rows that already have `overlap_ratio_vad` populated. Checkpoints every 500 rows via atomic `.tmp + rename`, so a killed run resumes cleanly.

```bash
for split in train-100 dev test; do
  case "$split" in
    train-100) libri="$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100" ;;
    dev)       libri="$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/dev" ;;
    test)      libri="$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test" ;;
  esac
  python scripts/fix_overlap_csv.py \
    --csv            $SHARED/data/features_pyannote/${split}.csv \
    --libri2mix_root $libri
done
```

CPU only. ~15 minutes total for all 19,900 clips.

### Step 3 — build `descriptions.json` deterministically

No LLM. Reads the VAD columns when present, falls back to pyannote columns with a one-shot warning. ~10 seconds end-to-end.

**Default for the LoRA+8B pipeline (v9 recipe):**

```bash
python scripts/build_descriptions_deterministic.py --all --untagged \
  --no-overlap-segments --no-duration \
  --features-dir $SHARED/data/features_pyannote \
  --output       $SHARED/data/descriptions_untagged_noseg_nodur.json
```

Sample entry (8 SFS-scored features, no duration / no overlap segments):

```
The signal-to-noise ratio SNR is 13.17 dB. The SRMR is 5.3646. The F0 mean is 152.48 Hz and the F0 standard deviation SD is 62.88 Hz. The speaking rate is 6.888 syl/sec. The pause count is 1 and the pause rate is 15.306 per min. The overlap ratio is 0.7908. F0 and formant estimates are unreliable during overlap windows.
```

Flag rationale:
- **`--untagged`** strips the `<sec_*>`, `<f_*>`, `<r>` special-token wrappers from the prose. The LoRA path doesn't register them in the tokenizer (`tagged_mode: false`), and post-hoc attention extraction parses the prose directly for section spans.
- **`--no-overlap-segments`** drops the "overlap segments are present at 0.5-3.6s, ..." sentence. WavLM features are temporally pooled and the LM cannot learn precise time-stamp emission, so without this flag it hallucinates a stereotyped tile-the-clip pattern that the IoU≥0.8 SFS scorer rejects. The scalar `overlap_ratio` is still emitted.
- **`--no-duration`** drops the leading "The recording is X s long." sentence. Duration is trivially measurable from the wav header (`audio.shape[0] / sample_rate`) — the model has no genuine learning task there, and including it as a scored claim would either inflate SFS for models that emit a correct auto-prepend or penalize honest abstention. SFS now scores audio-QUALITY features only. Duration is stored as a separate `measured_duration_sec` sidecar field at inference time and can be rendered alongside the quality prose by downstream UIs.

> **Note:** the YAML's `descriptions_path` should point at the file you just built. If you're using a different name, override via `--descriptions_path $SHARED/data/<your>.json` on the train CLI.

**Tagged variant** (for the `use_sections=true` ablation row only — registers the 19 special tokens):

```bash
python scripts/build_descriptions_deterministic.py --all \
  --features-dir $SHARED/data/features_pyannote \
  --output       $SHARED/data/descriptions_tagged.json
```

Sample tagged entry:

```
The recording is 3.920 s long.
<sec_noise><f_snr>The signal-to-noise ratio SNR is 13.17 dB</f></sec>.
<sec_reverb><f_srmr>The SRMR is 5.3646</f></sec>.
<sec_pitch><f_f0_mean>The F0 mean is 152.48 Hz</f> and
           <f_f0_sd>the F0 standard deviation SD is 62.88 Hz</f></sec>.
<sec_overlap><f_overlap_ratio>The overlap ratio is 0.5102</f> and
             <f_overlap_segments>overlap segments are present at <r>1.0-3.0s</r></f></sec>.
F0 and formant estimates are unreliable during overlap windows.
```

To rebuild only one third for distributed work, add `--part 1` (or 2, 3) instead of `--all`.

### Step 4 — WavLM features + overlap context → `.pt`

Per clip: `audio_features (T, 1024)`, `overlap_info (T, 4)`, `overlap_segments`, `filename`.

```bash
for src in train-100:train dev:val test:test; do
  IN=${src%%:*}; OUT=${src##*:}
  python src/preprocess.py \
    --audio_dir    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$IN/mix_clean \
    --features_csv $SHARED/data/features_pyannote/${IN}.csv \
    --output_dir   $SHARED/data/processed_pyannote/$OUT
done
```

Needs an H100 (WavLM-Large forward). ~30 minutes for train, ~6 min each for val/test.

**If only the overlap clamping changed and you want to preserve cached BEATs patches in existing `.pt` files**, use the surgical refresh:

```bash
for split in train val test; do
  case "$split" in
    train) csv="$SHARED/data/features_pyannote/train-100.csv" ;;
    val)   csv="$SHARED/data/features_pyannote/dev.csv" ;;
    test)  csv="$SHARED/data/features_pyannote/test.csv" ;;
  esac
  python scripts/refresh_overlap_info.py \
    --pt_dir       $SHARED/data/processed_pyannote/$split \
    --features_csv $csv
done
```

CPU only. ~15 minutes total. Only mutates `overlap_info` and `overlap_segments`; `audio_features` and `beats_patches` survive byte-identical.

### Step 5 — cache BEATs patches into each `.pt` *(optional — only needed for the `use_sections=true` ablation row)*

The headline LoRA + 8B run uses post-hoc attention extraction from the LM's native layers and does NOT need BEATs. Only run this step if you plan to train the `section-head` or `static-queries` design-ablation rows (which re-enable the legacy `SectionQueryHead` path).

Adds the `beats_patches` and `beats_grid_meta` keys to every `.pt`. Idempotent: clips that already have `beats_patches` are skipped unless `--overwrite`.

```bash
for split in train val test; do
  case "$split" in
    train) audio="$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/mix_clean" ;;
    val)   audio="$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/dev/mix_clean" ;;
    test)  audio="$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean" ;;
  esac
  python scripts/preprocess_beats.py \
    --audio_dir       $audio \
    --pt_dir          $SHARED/data/processed_pyannote/$split \
    --checkpoint_name BEATs_iter3_plus_AS2M.pt
done
```

H100 strongly recommended. ~3-4 hours for train, ~1 hour each for val/test.

## Training

### Wandb (one-time)

Runs live at `https://wandb.ai/speech_quality_adapter/aqua-nl-emnlp`.

The team entity and project are **pinned in `configs/config.psc.emnlp.yaml`** (`wandb_entity: speech_quality_adapter`, `wandb_project: aqua-nl-emnlp`) and read by `train.py` at `wandb.init` time. You don't need to export `WANDB_ENTITY` each session. Just authenticate once:

```bash
python -m wandb login
# paste your key from https://wandb.ai/authorize when prompted —
# it lands in ~/.netrc (mode 600), persists across shells
```

Override the entity or project from the CLI per-run if you ever need to:

```bash
python src/train.py --config configs/config.psc.emnlp.yaml \
  --wandb_entity  <other> \
  --wandb_project <other>
```

Sanity-check the URL line in the run's startup banner:

```
wandb: 🚀 View run at https://wandb.ai/speech_quality_adapter/aqua-nl-emnlp/runs/...
                         ^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^
                         team entity              right project
```

### Main run

Uses `configs/config.psc.emnlp.yaml` defaults: **Qwen3-8B + LoRA r=16**, `film-mamba` adapter, `use_sections: false`, `tagged_mode: false`, `beats_cached: false`, `batch_size: 6`, `epochs: 15`, `--no-duration` targets.

Foreground with log capture via `tee` (output streams to your terminal AND is written to `$LOG`):

```bash
mkdir -p $SHARED/logs $SHARED/checkpoints/v9_lora_8b_nodur
LOG=$SHARED/logs/v9-lora-8b-nodur_$(date +%Y%m%d_%H%M%S).log

python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $SHARED/data/descriptions_untagged_noseg_nodur.json \
  --wandb_run_name    v9-lora-8b-nodur \
  --save_dir          $SHARED/checkpoints/v9_lora_8b_nodur \
  2>&1 | tee "$LOG"
```

~33 min / epoch × 15 epochs ≈ 8 hours on a single H100. The launch keeps your shell occupied; if you need to detach without killing, wrap the command in `screen -S v9` or `tmux new -s v9` first (Ctrl+A D / Ctrl+B D to detach, re-attach later with `screen -r v9` / `tmux attach -t v9`).

The `python -u` flag is **critical** when piping through `tee` — it disables Python's stdout buffering so prints land in the terminal and the log file in real time, not in 8 KB chunks every few minutes.

What you should see:
- ~1 min in: model construction banner (`[LoRA] rank=16 alpha=32`, `Parameters — LM total: 8.23B | LoRA trainable: 43.6M | adapter trainable: 51.0M | grand total trainable: 94.7M (1.150% of LM)`), dataset load (`Loaded: train=13900, val=3000`), and the wandb URL.
- After ~30 s of warmup: loss prints every `log_every` steps (default 10), starts near 9 and falls toward 1-3 over the first epoch.
- After each epoch: a val pass with BLEU / ROUGE / SFS, then a checkpoint write to `<save_dir>` (atomic — `.tmp` + `os.replace`). When val_sfs_f1 improves, `best.pt` is also uploaded to wandb as an Artifact (`upload_ckpt_to_wandb: true` default).

If output stops changing for >5 min after the wandb URL appears, the loop is stalled — check `pgrep -af train.py` and the very tail of the log for a traceback.

### Resuming

Every epoch writes `<save_dir>/last.pt` (latest) and updates `<save_dir>/best.pt` (best-val-so-far). Resume restores adapter weights, optimizer, scheduler, epoch counter, and the wandb run ID (same run page continues).

```bash
python src/train.py --config configs/config.psc.emnlp.yaml \
  --resume_from $SHARED/checkpoints/film_mamba__v1/last.pt
```

If `last.pt` is corrupted (e.g., session timed out mid-save — the file size will be much smaller than `best.pt`), resume from `best.pt` instead. The trainer writes `last.pt` non-atomically, so a SIGKILL during the save truncates it; `best.pt` only updates after a complete write of `last.pt`, so it's the safer fallback.

### Smoke run (50 micro-batches, no wandb)

The trainer accepts an inline `max_steps` cap via the config layer. Useful for verifying model construction + dataloader wiring before committing GPU hours:

```bash
WANDB_MODE=disabled python src/train.py --config configs/config.psc.emnlp.yaml --max_steps 50
```

Walltime ~1-2 min after the first-time Qwen3-8B HF download (~16 GB across 5 safetensors shards). Use `HF_HOME=$SHARED/hf_cache` to keep the cache on shared storage so other users / sessions don't re-download.

### Config overrides on the command line

`train.py` coerces any `--key value` pair against the YAML's type (bool / int / float; strings pass through). Useful for ablations without YAML edits:

```bash
python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-attn \
  --batch_size      4 \
  --gradient_accumulation_steps 2 \
  --save_dir        $SHARED/checkpoints/film_attn_ablation \
  --wandb_run_name  film-attn-ablation
```

## Ablations

The headline run uses the main config (`film-mamba`). Two ablation families. Save_dirs and wandb run names mirror the adapter / knob being tested so you can read a checkpoint path and know what it is.

**All ablations inherit the LoRA + 8B recipe from `configs/config.psc.emnlp.yaml`** — no per-command `--lm_name` / `--lora_rank` flags needed. To run a row with the OLD full-FT 1.7B recipe for comparison (e.g., to quantify how much LoRA matters at this dataset scale), append:

```
--lm_name Qwen/Qwen3-1.7B --lora_rank 0
```

to any of the commands below. Save it under a separate `__fullft` suffix to keep checkpoints distinct (e.g., `--save_dir $SHARED/checkpoints/film_mamba__fullft__v1`).

### Adapter-architecture sweep

The paper's primary comparison table. Same data, same seed, same epochs, same section / BEATs / dynamic-query setup — **only `adapter_variant` changes**. The six variants we compare:

| adapter_variant | Conditioning | Temporal mixer | Notes |
|---|---|---|---|
| `film-mamba` | FiLM | Mamba SSM (1 layer) | Headline result (current default) |
| `concat-only` | None (concat) | None | Baseline — argues for FiLM + mixer |
| `sigmoid-gate` | Sigmoid gate | None | Lighter gating alternative |
| `film` | FiLM | None | Isolates the temporal-mixer contribution |
| `film-attn` | FiLM | Self-attention (1 layer) | Argues Mamba vs attn |
| `qformer` | Q-Former cross-attention | (implicit) | Alternative architecture |

Naming convention: `--wandb_run_name v9-<variant>` and `--save_dir $SHARED/checkpoints/v9_<variant>`. The `v9` prefix marks the recipe generation (LoRA + 8B + no-duration + no-overlap-segments). When you re-tune hyperparams within v9, append `__v2` etc. to the save_dir.

**Scheduling**: each run takes ~8h on one H100. The 8 variants in sequence is ~64 h. To parallelize, allocate one H100 per variant in separate `interact` sessions (or `sbatch` jobs) — **never launch two trainings on the same GPU**, they OOM each other instantly. Each command below assumes you're inside a fresh `interact -p GPU-shared --gres=gpu:h100-80:1 -t 8:00:00 -A cis260125p` session.

```bash
# Shared setup — set once per session before launching ANY variant
DESC=$SHARED/data/descriptions_untagged_noseg_nodur.json
mkdir -p $SHARED/logs

# Pre-warm the 8B blobs (one-time per node — second run on same node is instant)
time cat $SHARED/hf_cache/hub/models--Qwen--Qwen3-8B/blobs/* > /dev/null
```

```bash
# v9-film-mamba (headline, this is the main config — no --adapter_variant needed)
LOG=$SHARED/logs/v9-film-mamba_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $DESC \
  --save_dir          $SHARED/checkpoints/v9_film_mamba \
  --wandb_run_name    v9-film-mamba \
  2>&1 | tee "$LOG"
```

```bash
# v9-concat-only — baseline without FiLM or temporal mixer
LOG=$SHARED/logs/v9-concat-only_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $DESC \
  --adapter_variant   concat-only \
  --save_dir          $SHARED/checkpoints/v9_concat_only \
  --wandb_run_name    v9-concat-only \
  2>&1 | tee "$LOG"
```

```bash
# v9-sigmoid-gate — lighter gating than FiLM
LOG=$SHARED/logs/v9-sigmoid-gate_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $DESC \
  --adapter_variant   sigmoid-gate \
  --save_dir          $SHARED/checkpoints/v9_sigmoid_gate \
  --wandb_run_name    v9-sigmoid-gate \
  2>&1 | tee "$LOG"
```

```bash
# v9-film (FiLM only, no temporal mixer) — isolates the FiLM contribution
LOG=$SHARED/logs/v9-film_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $DESC \
  --adapter_variant   film \
  --save_dir          $SHARED/checkpoints/v9_film \
  --wandb_run_name    v9-film \
  2>&1 | tee "$LOG"
```

```bash
# v9-film-attn (FiLM + 1-layer self-attention mixer)
LOG=$SHARED/logs/v9-film-attn_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $DESC \
  --adapter_variant   film-attn \
  --save_dir          $SHARED/checkpoints/v9_film_attn \
  --wandb_run_name    v9-film-attn \
  2>&1 | tee "$LOG"
```

```bash
# v9-qformer — Q-Former cross-attention alternative
LOG=$SHARED/logs/v9-qformer_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $DESC \
  --adapter_variant   qformer \
  --save_dir          $SHARED/checkpoints/v9_qformer \
  --wandb_run_name    v9-qformer \
  2>&1 | tee "$LOG"
```

To run multiple variants sequentially in one shell (~64h end-to-end), chain them with `&&` between the python commands so the next one only starts if the previous succeeded.

Each command keeps your shell occupied for ~8h. To detach safely use `screen -S adapter` / `tmux new -s adapter` before launching, then Ctrl+A D / Ctrl+B D to detach (training continues), and `screen -r` / `tmux attach -t adapter` to re-attach.

> **Note on the currently-running training:** if you launched the headline with `--wandb_run_name v9-lora-8b-nodur --save_dir $SHARED/checkpoints/v9_lora_8b_nodur` (older convention), that's fine — it's the equivalent of `v9-film-mamba` since `film-mamba` is the YAML default `adapter_variant`. Future ablation rows should use the `v9-<variant>` naming above so they line up cleanly in the wandb run list.

### Design ablations (orthogonal to the adapter sweep)

For probing the section-attention design itself. Note that since v7 the YAML defaults are `use_sections: false` + `tagged_mode: false`, so the headline row already runs the "free-form prose" config. These ablations turn the section-tag path *back on*, to compare against the post-hoc attention extraction approach that's now the default:

| Knob | Override (relative to YAML defaults) | Run name |
|---|---|---|
| Section-head ON + tagged-prose (legacy EMNLP-rework path) | `--use_sections true --tagged_mode true --beats_cached true --descriptions_path $SHARED/data/descriptions_tagged.json` | `v9-section-head` |
| Section-head ON, but static queries (learnable lookup, no LM-derived query) | `--use_sections true --tagged_mode true --beats_cached true --descriptions_path $SHARED/data/descriptions_tagged.json --section_query_mode static` | `v9-static-queries` |
| Full-FT comparison (Qwen3-1.7B, no LoRA) | `--lm_name Qwen/Qwen3-1.7B --lora_rank 0` | `v9-fullft-1.7b` |

Section-head rows require the tagged JSON — rebuild via `scripts/build_descriptions_deterministic.py --all` (without `--untagged`/`--no-overlap-segments`/`--no-duration`) into a separate output file.

Same scheduling rule applies: one variant at a time, foreground with `tee`. Use `screen` / `tmux` if you need to detach.

```bash
# v9-section-head (legacy EMNLP-rework path — needs descriptions_tagged.json)
LOG=$SHARED/logs/v9-section-head_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --use_sections true --tagged_mode true --beats_cached true \
  --descriptions_path $SHARED/data/descriptions_tagged.json \
  --save_dir          $SHARED/checkpoints/v9_section_head \
  --wandb_run_name    v9-section-head \
  2>&1 | tee "$LOG"
```

```bash
# v9-static-queries (static section queries instead of LM-derived dynamic queries)
LOG=$SHARED/logs/v9-static-queries_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --use_sections true --tagged_mode true --beats_cached true \
  --descriptions_path  $SHARED/data/descriptions_tagged.json \
  --section_query_mode static \
  --save_dir           $SHARED/checkpoints/v9_static_queries \
  --wandb_run_name     v9-static-queries \
  2>&1 | tee "$LOG"
```

```bash
# v9-fullft-1.7b (full FT comparison — the original v6 recipe before LoRA).
# This one ALSO needs --descriptions_path explicit since v6 used a different
# target format. Uses the same no-duration / no-overlap-segments JSON for fair
# comparison; only the LM + LoRA-rank flip.
LOG=$SHARED/logs/v9-fullft-1.7b_$(date +%Y%m%d_%H%M%S).log
python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --descriptions_path $SHARED/data/descriptions_untagged_noseg_nodur.json \
  --lm_name           Qwen/Qwen3-1.7B \
  --lora_rank         0 \
  --save_dir          $SHARED/checkpoints/v9_fullft_17b \
  --wandb_run_name    v9-fullft-1.7b \
  2>&1 | tee "$LOG"
```

### Compute budget

Each run is ~8 h on one H100. Adapter sweep is 8 runs = ~64 h sequential, ~8 h fully-parallel-on-8-GPUs. Design ablations are 2 more runs.

## Inference + evaluation

Greedy decoding over the test set (`--top_k 1` is deterministic, matches paper-table numbers). No `--lm_name` / `--adapter_variant` flags — `inference.py` reads them from the checkpoint's embedded config.

**Run one ablation at a time** — inference loads the 8B LM (~16 GB GPU memory), so two concurrent runs on the same GPU will OOM. ~30-90 min per ablation on H100.

```bash
# Headline (v9-film-mamba)
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_film_mamba/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-concat-only
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_concat_only/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-sigmoid-gate
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_sigmoid_gate/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-film
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_film/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-film-attn
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_film_attn/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-qformer
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_qformer/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

**Design ablations:**

```bash
# v9-section-head
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_section_head/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-static-queries
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_static_queries/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

```bash
# v9-fullft-1.7b
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_fullft_17b/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

To save your shell session for ~8-12h of sequential inference across all 11 ablations, wrap each command in `screen` / `tmux` and detach.

### Outputs

Written **next to the checkpoint** (`dirname(--checkpoint)`):

- `inference_results.json` — per-clip records flushed every 50 clips via atomic tmp+rename. Keys per clip:
  - `filename`
  - `generated` — model's audio-quality prose (no duration, no overlap segments)
  - `measured_duration_sec` — sidecar metadata, computed from `audio_features.shape[0] / 50` Hz. Downstream renderers can join with `generated` if they want a duration-headed display
  - `target`, `claims` (parsed `(feature, value)` pairs)
  - `sfs_precision / sfs_recall / sfs_f1`, `per_feature` breakdown
  - `attention_maps` — **only populated when `use_sections=true`** (the section-head ablation row). The headline LoRA + 8B path uses post-hoc attention extraction (see `scripts/extract_attention.py` — separate run after inference).
- `inference_summary.json` — aggregate `sfs_precision / recall / f1`, `per_feature_accuracy`, `gen_metrics: {bleu, rouge_l, bertscore_f1}`.

The same wandb run page used at training time gets `test/sfs_*` and `test/{bleu,rouge_l,bertscore_f1}` (because the checkpoint embeds `wandb_run_id`).

### Re-score against the no-duration target

`inference.py` reads its target / SFS ground truth from `descriptions_path`. Any run scored against a target that still carries a leading duration sentence (the old `descriptions_untagged_noseg.json`) is depressed two ways: `duration_sec` lands in the SFS recall denominator where it can never be matched, and the duration sentence sits unmatched in the BLEU/ROUGE/BERTScore reference. Generation is greedy and target-independent, so you do **not** re-run the model — re-score the saved `inference_results.json` against the no-duration target:

```bash
for V in v9_film_mamba v9_film_attn v9_concat_only v9_sigmoid_gate v9_film v9_qformer; do
  python scripts/rescore_nodur.py \
    --inference_results $SHARED/checkpoints/$V/inference_results.json \
    --descriptions      $SHARED/data/descriptions_untagged_noseg_nodur.json
done
# writes inference_summary_rescored.json next to each inference_results.json
```

Only needed for runs inferred **before** `descriptions_path` was pointed at the no-duration target; runs done after are already correct (the scorer also now drops any GT feature without a tolerance, so duration cannot inflate recall regardless). To collate the corrected numbers, point the snippet below at `inference_summary_rescored.json`.

### Performance table (collate the adapter sweep)

Each variant's `inference_summary.json` (or `inference_summary_rescored.json`, see above) is one row of the paper's performance table. After running inference for every variant, collate them into the table (SFS-F1 / P / R / BLEU / ROUGE-L / BERTScore):

```bash
python3 - <<'PY'
import json, glob, os
pat = os.path.expandvars("$SHARED/checkpoints/v9_*/inference_summary.json")
print(f"{'variant':22}{'SFS-F1':>8}{'P':>7}{'R':>7}{'BLEU':>7}{'ROUGE':>7}{'BERT':>7}")
for f in sorted(glob.glob(pat)):
    d = json.load(open(f)); g = d.get("gen_metrics", {})
    n = os.path.basename(os.path.dirname(f))
    print(f"{n:22}{d['sfs_f1']:8.3f}{d['sfs_precision']:7.3f}{d['sfs_recall']:7.3f}"
          f"{(g.get('bleu') or 0):7.2f}{(g.get('rouge_l') or 0):7.3f}{(g.get('bertscore_f1') or 0):7.3f}")
PY
```

The adapter variants (`v9_concat_only`, `v9_sigmoid_gate`, `v9_film`, `v9_film_attn`, `v9_qformer`, `v9_film_mamba`) are the rows of the adapter-comparison table. The design-ablation dirs (`v9_section_head`, `v9_static_queries`, `v9_fullft_1.7b`) are reported separately, not in that table. Add the no-audio baseline row with `scripts/zero_shot_baseline.py` (see below).

### Run the tests

```bash
python -m pytest tests/ -v        # full suite (SFS, text metrics, feature set, section-query, FiLM init, overlap-info)
```

See the [Testing](#testing) section for dependencies, standalone scripts, and the 5-minute end-to-end smoke.

### Analysis scripts (run after inference)

All read `inference_results.json` and produce paper-ready outputs alongside it.

```bash
RES=$SHARED/checkpoints/v9_lora_8b_nodur/inference_results.json
CKPT=$SHARED/checkpoints/v9_lora_8b_nodur/best.pt

# Per-feature SFS table — P/R/F1 per feature, sorted by F1.
# Surfaces which features are strong (pause_count, overlap_ratio) vs weak
# (f0_mean, f0_sd). Writes per_feature_sfs.md + .json next to the input.
python scripts/per_feature_sfs.py --inference_results $RES

# Overlap-aware hedging contingency table.
# Buckets clips by ground-truth overlap_ratio (none / low / mid / high / v_high)
# and reports the rate at which the model emits the unreliability hedge.
# Use --features_csv as a fallback when targets lack overlap_ratio (e.g., the
# clean-speech control set).
python scripts/overlap_hedging_compare.py --inference_results $RES

# Zero-shot baseline — bare Qwen3-8B (no adapter, no LoRA, no audio prefix).
# Establishes the floor that any audio-conditioned model must beat.
python scripts/zero_shot_baseline.py --config configs/config.psc.emnlp.yaml \
  --test_dir $SHARED/data/processed_pyannote/test \
  --output_dir $SHARED/checkpoints/zero_shot_baseline \
  --start 0 --end 100

# Post-hoc per-section attention extraction (paper figure).
mkdir -p $(dirname $RES)/attention
python scripts/extract_attention.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $CKPT \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --filenames  "<clip1>.wav,<clip2>.wav,<clip3>.wav" \
  --output_dir $(dirname $RES)/attention

# Spectrogram + per-section heatmap overlay PDFs.
python scripts/plot_attention.py \
  --attention_dir $(dirname $RES)/attention \
  --audio_dir     $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --output_dir    docs/figures/

# Faithfulness study — mask top-K attended frames, re-run inference, measure
# SFS drop per section vs random masking. Proves attention is CAUSAL.
python scripts/faithfulness_study.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $CKPT \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --attention_dir $(dirname $RES)/attention \
  --output     $(dirname $RES)/faithfulness.json --top_k_mask 5

# SFS-vs-human correlation — validates that SFS tracks human judgement.
# Two phases:
python scripts/sfs_human_correlation.py --mode prepare \
  --inference_results $RES \
  --output_csv $(dirname $RES)/human_ratings.csv --n 50
# (open the CSV, fill the `human_rating` column 1-5, then:)
python scripts/sfs_human_correlation.py --mode analyze \
  --rated_csv  $(dirname $RES)/human_ratings.csv \
  --output_md  $(dirname $RES)/sfs_human_correlation.md
```

### Range / resume / parallelism

Three thousand test clips → ~60-90 min on Qwen3-1.7B. The script auto-resumes (`inference_results.json` is consulted before each clip).

| You want to… | Flags |
|---|---|
| Process the whole test set (default) | *(none)* |
| Process clips 0..499 | `--start 0 --end 500` |
| Process clips 500..2999 | `--start 500 --end 3000` |
| Quick 50-clip smoke | `--start 0 --end 50` |
| Resume a crashed full run | *(same command — auto-skips done clips)* |

Different `save_dir` ↔ independent `inference_results.json` files, so range-parallel across ablations is fine. **Don't** range-parallel on the same checkpoint — both shards flush to one file and the later flush overwrites work in flight.

### Attention figures (paper centerpiece)

Two paths produce per-section attention maps:

| Path | Used when | Script |
|---|---|---|
| **Post-hoc** (default) | LoRA + 8B headline run (no section_head) | `scripts/extract_attention.py` + `scripts/plot_attention.py` |
| **Built-in attention_maps** | Section-head ablation row (`use_sections=true`) | `scripts/plot_attention_spectrograms.py` reads from `inference_results.json::attention_maps` |

The post-hoc path is shown above in "Analysis scripts". For the section-head ablation:

```bash
python scripts/plot_attention_spectrograms.py \
  --results   $SHARED/checkpoints/section_head__v1/inference_results.json \
  --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --output    docs/figures/attention_overlays \
  --limit     10 \
  --format    pdf
```

Each PDF row corresponds to an emitted section + one per `<r>` range, log-mel in greyscale + attention map overlaid as a hot heatmap.

## Metric semantics

| Metric | Measures | When to trust |
|---|---|---|
| **SFS-F1** (primary) | Numerical faithfulness: regex-parses tagged claims (e.g. `<f_snr>The SNR is 15.63 dB</f>`) and checks each within per-feature tolerance | The paper's headline. High SFS = model is reading the audio, not hallucinating numbers. |
| **BLEU-4** | n-gram precision vs reference | Surface-form fluency / template conformance |
| **ROUGE-L F1** | longest common subsequence | Robust-to-reorder structure overlap |
| **BERTScore-F1** | RoBERTa-large semantic similarity | Catches faithful paraphrases that BLEU/ROUGE miss |

Interpretation grid:

| SFS | BLEU/ROUGE | Diagnosis |
|---|---|---|
| high | high | Fluent + factual — ideal |
| high | low | Factual but unusual phrasing — acceptable |
| low | high | Fluent hallucination — train longer, fix conditioning |
| low | low | Generator broken — degenerate decoding or alignment bug |

Checkpoint selection is on `val_loss` only — saving-best on a similarity metric biases toward reference copying and *lowers* SFS.

## Testing

### Dependencies

```bash
pip install pytest sacrebleu rouge-score bert-score librosa scipy
```

`sacrebleu`, `rouge-score`, `bert-score` are only needed for the corresponding metric — missing deps cause that one metric to silently skip. `librosa` is required by `scripts/plot_attention.py` for the spectrogram. `scipy` is required by `scripts/sfs_human_correlation.py` for Spearman/Pearson.

### Full pytest suite

```bash
python -m pytest tests/ -v
```

Covers:

| Test file | What it covers |
|---|---|
| `test_sfs.py` | `ClaimParser`, `TaggedClaimParser`, `HybridClaimParser`, `SFSScorer` |
| `test_section_query.py` | `SectionQueryHead` static + dynamic modes, BEATs integration |
| `test_compute_loss_b_full.py` | B-full dual-prompt loss with mocked LM |
| `test_text_metrics.py` | BLEU / ROUGE / BERTScore fail-soft behavior |
| `test_build_overlap_info.py` | overlap clamping and segment-merge invariants |
| `test_refresh_overlap_info.py` | surgical `.pt` refresh keys-preserved invariants |
| `test_feature_set.py` | canonical 8 SFS scalars + per-feature scales |
| `test_feature_tags.py` | tag catalog + tag-token registration |
| `test_film_init_diagnostic.py` | FiLM identity-init bug regression test |

### Standalone test scripts (no pytest)

Some tests are easier to run as plain scripts:

```bash
python scripts/test_fix_descriptions.py
python scripts/test_fix_overlap_csv.py
python scripts/test_build_descriptions_deterministic.py
```

### BEATs integration test (gated, downloads ~370 MB)

```bash
RUN_BEATS_TEST=1 python -m pytest tests/test_section_query.py::TestBEATsIntegration -v
```

### End-to-end smoke (validate the whole stack in ~5 minutes)

```bash
# 1. Build a 50-clip descriptions JSON for a smoke run
python scripts/build_descriptions_deterministic.py --part 1 --untagged \
  --no-overlap-segments --no-duration \
  --features-dir $SHARED/data/features_pyannote \
  --output       /tmp/smoke_descriptions.json

# 2. 50-step training run (no wandb, no checkpoints flushed)
WANDB_MODE=disabled python -u src/train.py \
  --config configs/config.psc.emnlp.yaml \
  --descriptions_path /tmp/smoke_descriptions.json \
  --max_steps 50 \
  --save_dir /tmp/smoke_ckpt

# 3. 10-clip inference smoke (needs a real checkpoint — point at v9 if running)
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/v9_lora_8b_nodur/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --start 0 --end 10 --top_k 1

# 4. Analysis scripts work on the 10-clip output
python scripts/per_feature_sfs.py \
  --inference_results $SHARED/checkpoints/v9_lora_8b_nodur/inference_results.json
python scripts/overlap_hedging_compare.py \
  --inference_results $SHARED/checkpoints/v9_lora_8b_nodur/inference_results.json
```

If all four steps complete without errors, the full pipeline is wired correctly.

## Project layout

```
src/
  feature_extractor.py     — Pyannote feature extractor (cross-domain datasets)
  feature_extractor_mix.py — Libri2Mix extractor; supports both overlap modes
  preprocess.py            — WavLM features + 4-dim overlap context → .pt
  adapter.py               — Conv8x + Mamba + FiLM (+ ablation variants)
  dataset.py               — Dataset + collate (pads variable-length BEATs patches)
  train.py                 — Training loop (static + dynamic section queries, B-full loss)
  inference.py             — Generation + SFS + per-section attention capture
  sfs.py                   — TaggedClaimParser + HybridClaimParser + SFSScorer
  text_metrics.py          — BLEU-4 / ROUGE-L / BERTScore-F1 (fail-soft on missing dep)
  feature_set.py           — Canonical 8 SFS-scored scalars + per-feature scales
  section_tags.py          — 6 sections, 9 inner tags, <r> markers (SoT, 19 special tokens)
  section_query.py         — SectionQueryHead (static + dynamic cross-attention)
  spec_encoder.py          — SpecEncoder wrapper for BEATs / AST
  beats/                   — Vendored Microsoft BEATs encoder

scripts/
  fix_overlap_csv.py                 — Adds *_vad columns from Silero-on-stems
  build_descriptions_deterministic.py — Composes descriptions.json from CSV (no LLM).
                                       Flags: --untagged / --no-overlap-segments / --no-duration
  refresh_overlap_info.py            — Surgical overlap_info refresh in existing .pt files
  preprocess_beats.py                — Caches BEATs patches into per-clip .pt files

  # Post-inference analysis (consume inference_results.json):
  per_feature_sfs.py                 — Per-feature P/R/F1 table + JSON
  overlap_hedging_compare.py         — Hedge-rate × overlap-bucket contingency
  zero_shot_baseline.py              — Bare-LM baseline (no adapter, no audio prefix)

  # Attention figure (LoRA + 8B headline path, post-hoc):
  extract_attention.py               — Per-section attention vectors via Qwen3 native attn
  plot_attention.py                  — Spectrogram + per-section heatmap overlay PDFs

  # Section-head ablation path (use_sections=true):
  plot_attention_spectrograms.py     — Renders attention_maps embedded in inference_results.json

  # Causal study + metric validity:
  faithfulness_study.py              — Mask top-K attended frames → measure SFS drop
  sfs_human_correlation.py           — Prepare/analyze SFS-vs-human ratings (Spearman + Pearson)

  test_*.py                          — Standalone test scripts

configs/
  config.psc.emnlp.yaml    — Current default: Qwen3-8B + LoRA r=16, untagged-noseg targets, no section_head
  config.psc.yaml          — Older baseline (Qwen3-8B + LoRA + earlier description format), kept for reference

tests/
  test_sfs.py                          — TaggedClaimParser + scorer
  test_section_query.py                — SectionQueryHead static + dynamic + BEATs integration
  test_compute_loss_b_full.py          — B-full multi-task loss with mocked LM
  test_text_metrics.py                 — BLEU / ROUGE / BERTScore
  test_build_overlap_info.py           — Overlap clamping (de49d6a)
  test_refresh_overlap_info.py         — Surgical refresh keys-preserved invariants

descriptions.json                      — 19,900 GT entries (regenerated by build_descriptions_deterministic.py)
```

## Conventions and gotchas

- **`Qwen/Qwen3-8B` is the default LM as of v7.** Earlier configs used 1.7B + full FT, which catastrophically forgot the LM's instruction-following prior on the ~6M target-token training set. LoRA on 8B keeps the prior frozen as an anchor — ~45M trainable params instead of 1.7B. Both `Qwen/Qwen3-8B` and `Qwen/Qwen3-1.7B` are the instruct-tuned variants (no `-Instruct` suffix in Qwen3).
- **`overlap_info` in `.pt` files comes from `overlap_segments` (pyannote-on-mix), not `overlap_segments_vad`.** That's the intended input distribution. The description GT reads VAD-on-stems. Different signals on purpose, to prevent the trivial-copy data leakage.
- **Special tokens (`<sec_*>`, `<f_*>`, `<r>`) are registered only when `tagged_mode: true`** (section-head ablation rows). They're added with `special_tokens=False` so `tokenizer.decode(..., skip_special_tokens=True)` keeps them in the output string — required for both the SFS parser and the section_head's attention hook. With `tagged_mode: false` (default), these tokens never enter the vocab and prose is generated as plain text.
- **`max_target_length: 512`** covers the full untagged-noseg description format (p99 ~380 tokens, max ~430). The earlier 224 cap was set for a trimmed catalog and silently truncated tails; 512 absorbs the longest clip with margin.
- **`gradient_checkpointing` is required at 8B + LoRA** to stay under H100-80 memory at `batch_size: 6`. The 8B frozen weights alone are ~16 GB in bf16; activations need to be checkpointed.
- **`--no-overlap-segments` is critical for SFS.** Without it, the model hallucinates a stereotyped 5-7-segment "tile-the-clip" pattern at the end of each description. The IoU≥0.8 SFS scorer rejects almost all of these → overlap_span recall craters to 0. Verified in v6 → v7 transition: dropping the segments sentence took overlap_span recall from 0.0 to ~0.65.
- **Atomic writes everywhere.** `preprocess_beats.py`, `refresh_overlap_info.py`, `fix_overlap_csv.py`, `inference.py`, AND `train.py`'s `last.pt` / `best.pt` saves all use `.tmp` + `os.replace`. A SIGKILL / OOM / quota truncate / Lustre writeback failure during a save leaves an ignorable `.tmp` while the canonical path stays at either the previous full content or the new full content. (Pre-2026-05-17 builds did not have atomic save on the train-loop checkpoints — we lost v7-lora-8b's `best.pt` to a 0-byte truncation that the atomic save would have prevented.)
- **Best.pt is mirrored to wandb as a versioned Artifact** (`upload_ckpt_to_wandb: true` by default). Each improvement uploads a new version named `best-<run_name>`. Off-site backup against local disk loss. Disable per-run with `--upload_ckpt_to_wandb false`. Download from another machine with `wandb.use_artifact("speech_quality_adapter/aqua-nl-emnlp/best-<run_name>:latest")`.
- **Duration is metadata, not a quality claim.** `--no-duration` strips it from training targets; `src/inference.py` measures it from the wav header and stores it on the output JSON's `measured_duration_sec` sidecar. The `generated` field contains only the model's audio-quality prose. SFS no longer scores duration (`duration_sec` was removed from `TOLERANCES`).
