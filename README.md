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

**Default for the LoRA+8B pipeline** (matches what `configs/config.psc.emnlp.yaml::descriptions_path` points at):

```bash
python scripts/build_descriptions_deterministic.py --all --untagged --no-overlap-segments \
  --features-dir $SHARED/data/features_pyannote \
  --output       $SHARED/data/descriptions_untagged_noseg.json
```

Sample entry (untagged + no overlap segments):

```
The recording is 3.920 s long. The signal-to-noise ratio SNR is 13.17 dB. The SRMR is 5.3646. The F0 mean is 152.48 Hz and the F0 standard deviation SD is 62.88 Hz. The speaking rate is 6.888 syl/sec. The pause count is 1 and the pause rate is 15.306 per min. The overlap ratio is 0.7908. F0 and formant estimates are unreliable during overlap windows.
```

Flag rationale:
- **`--untagged`** strips the `<sec_*>`, `<f_*>`, `<r>` special-token wrappers from the prose. The LoRA path doesn't register them in the tokenizer (`tagged_mode: false`), and post-hoc attention extraction parses the prose directly for section spans.
- **`--no-overlap-segments`** drops the "overlap segments are present at 0.5-3.6s, ..." sentence. WavLM features are temporally pooled and the LM cannot learn precise time-stamp emission, so without this flag it hallucinates a stereotyped tile-the-clip pattern that the IoU≥0.8 SFS scorer rejects. The scalar `overlap_ratio` is still emitted.

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

Uses `configs/config.psc.emnlp.yaml` defaults: **Qwen3-8B + LoRA r=16**, `film-mamba` adapter, `use_sections: false`, `tagged_mode: false`, `beats_cached: false`, `batch_size: 6`, `epochs: 15`.

Foreground (interactive shell visible — output streams to your terminal):

```bash
python src/train.py --config configs/config.psc.emnlp.yaml
```

Detached (survives SSH disconnect; output goes to a log file). Put `LOG=...` on its own line so the variable lands in your current shell — chaining it after `&&` in front of `nohup ... &` groups everything into a subshell and `$LOG` becomes empty in the outer shell:

```bash
mkdir -p $SHARED/logs
LOG=$SHARED/logs/film_mamba_$(date +%Y%m%d_%H%M%S).log

nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --wandb_run_name film-mamba \
  --save_dir       $SHARED/checkpoints/film_mamba__v1 \
  > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID  log=$LOG"
```

~33 min / epoch × 15 epochs ≈ 8 hours on a single H100. (Roughly the same as the old Qwen3-1.7B + full FT timing — the larger 8B forward is offset by LoRA's smaller backward + optimizer-state footprint.)

**Follow progress in real time while detached**:

```bash
tail -f $LOG          # ctrl-C to stop following (the run keeps going)
```

The `python -u` flag in the launch command above is **critical** — without it, when stdout is redirected to a file (as `nohup ... > $LOG` does), Python block-buffers stdout in ~8 KB chunks and the log file looks empty for minutes. `-u` forces unbuffered output so every print lands in `$LOG` immediately and `tail -f` shows real-time progress.

What you should see:
- ~1 min in: model construction banner (`[LoRA] rank=16 alpha=32`, `Parameters — LM total: 8.23B | LoRA trainable: 43.6M | adapter trainable: 51.0M | grand total trainable: 94.7M (1.150% of LM)`), dataset load (`Loaded: train=13900, val=3000`), and the wandb URL.
- After ~30 s of warmup: loss prints every `log_every` steps (default 10), starts near 9 and falls toward 1-3 over the first epoch.
- After each epoch: a val pass with BLEU / ROUGE / SFS, then a checkpoint write to `<save_dir>`.

If `tail -f` shows nothing changing for >5 min after the wandb URL appears, the loop is stalled — check `pgrep -af train.py` and the very tail of the log for a traceback.

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

The paper's primary comparison table. Same data, same seed, same epochs, same section / BEATs / dynamic-query setup — **only `adapter_variant` changes**. Eight variants in `src/adapter.py::build_adapter`:

| adapter_variant | Conditioning | Temporal mixer | Notes |
|---|---|---|---|
| `film-mamba` | FiLM | Mamba SSM (1 layer) | Headline result (current default) |
| `concat-only` | None (concat) | None | Baseline — argues for FiLM + mixer |
| `sigmoid-gate` | Sigmoid gate | None | Lighter gating alternative |
| `film` | FiLM | None | Isolates the temporal-mixer contribution |
| `film-attn` | FiLM | Self-attention (1 layer) | Argues Mamba vs attn |
| `film-attn-2L` | FiLM | Self-attention (2 layers) | Depth ablation for attn |
| `film-mamba-2L` | FiLM | Mamba SSM (2 layers) | Depth ablation for Mamba |
| `qformer` | Q-Former cross-attention | (implicit) | Alternative architecture |

Naming convention: `--wandb_run_name <variant>` and `--save_dir $SHARED/checkpoints/<variant>__v1`. Reading a path tells you the variant. Bump to `__v2`, `__v3` etc. when re-tuning hyperparams.

```bash
# film-mamba (headline, this is the main config — no --adapter_variant needed)
LOG=$SHARED/logs/film_mamba_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --save_dir       $SHARED/checkpoints/film_mamba__v1 \
  --wandb_run_name film-mamba \
  > "$LOG" 2>&1 &

# concat-only
LOG=$SHARED/logs/concat_only_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant concat-only \
  --save_dir        $SHARED/checkpoints/concat_only__v1 \
  --wandb_run_name  concat-only \
  > "$LOG" 2>&1 &

# sigmoid-gate
LOG=$SHARED/logs/sigmoid_gate_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant sigmoid-gate \
  --save_dir        $SHARED/checkpoints/sigmoid_gate__v1 \
  --wandb_run_name  sigmoid-gate \
  > "$LOG" 2>&1 &

# film (FiLM only, no temporal mixer)
LOG=$SHARED/logs/film_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film \
  --save_dir        $SHARED/checkpoints/film__v1 \
  --wandb_run_name  film \
  > "$LOG" 2>&1 &

# film-attn
LOG=$SHARED/logs/film_attn_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-attn \
  --save_dir        $SHARED/checkpoints/film_attn__v1 \
  --wandb_run_name  film-attn \
  > "$LOG" 2>&1 &

# film-attn-2L
LOG=$SHARED/logs/film_attn_2L_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-attn-2L \
  --save_dir        $SHARED/checkpoints/film_attn_2L__v1 \
  --wandb_run_name  film-attn-2L \
  > "$LOG" 2>&1 &

# film-mamba-2L
LOG=$SHARED/logs/film_mamba_2L_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-mamba-2L \
  --save_dir        $SHARED/checkpoints/film_mamba_2L__v1 \
  --wandb_run_name  film-mamba-2L \
  > "$LOG" 2>&1 &

# qformer
LOG=$SHARED/logs/qformer_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant qformer \
  --save_dir        $SHARED/checkpoints/qformer__v1 \
  --wandb_run_name  qformer \
  > "$LOG" 2>&1 &
```

**Don't run more than one of these in parallel on the same GPU** — they'd OOM each other. Either sequence them (drop the `&`, run one at a time, ~8 h each = ~64 h end-to-end), or grab additional H100s in separate interact allocations.

### Design ablations (orthogonal to the adapter sweep)

For probing the section-attention design itself. Note that since v7 the YAML defaults are `use_sections: false` + `tagged_mode: false`, so the headline row already runs the "free-form prose" config. These ablations turn the section-tag path *back on*, to compare against the post-hoc attention extraction approach that's now the default:

| Knob | Override (relative to YAML defaults) | Run name |
|---|---|---|
| Section-head ON + tagged-prose (legacy EMNLP-rework path) | `--use_sections true --tagged_mode true --beats_cached true --descriptions_path $SHARED/data/descriptions_tagged.json` | `section-head` |
| Section-head ON, but static queries (learnable lookup, no LM-derived query) | `--use_sections true --tagged_mode true --beats_cached true --descriptions_path $SHARED/data/descriptions_tagged.json --section_query_mode static` | `static-queries` |
| Full-FT comparison (Qwen3-1.7B, no LoRA) | `--lm_name Qwen/Qwen3-1.7B --lora_rank 0` | `fullft-1.7b` |

Section-head rows require the tagged JSON — rebuild via `scripts/build_descriptions_deterministic.py --all` (without `--untagged`/`--no-overlap-segments`) into a separate output file.

```bash
# section-head (legacy EMNLP-rework path — needs descriptions_tagged.json)
LOG=$SHARED/logs/section_head_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --use_sections true --tagged_mode true --beats_cached true \
  --descriptions_path $SHARED/data/descriptions_tagged.json \
  --save_dir          $SHARED/checkpoints/section_head__v1 \
  --wandb_run_name    section-head \
  > "$LOG" 2>&1 &

# static-queries (static section queries instead of LM-derived dynamic queries)
LOG=$SHARED/logs/static_queries_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --use_sections true --tagged_mode true --beats_cached true \
  --descriptions_path  $SHARED/data/descriptions_tagged.json \
  --section_query_mode static \
  --save_dir           $SHARED/checkpoints/static_queries__v1 \
  --wandb_run_name     static-queries \
  > "$LOG" 2>&1 &

# fullft-1.7b (full FT comparison — the original v6 recipe before LoRA)
LOG=$SHARED/logs/fullft_17b_$(date +%Y%m%d_%H%M%S).log
nohup python -u src/train.py --config configs/config.psc.emnlp.yaml \
  --lm_name    Qwen/Qwen3-1.7B \
  --lora_rank  0 \
  --save_dir   $SHARED/checkpoints/fullft_17b__v1 \
  --wandb_run_name fullft-1.7b \
  > "$LOG" 2>&1 &
```

### Compute budget

Each run is ~8 h on one H100. Adapter sweep is 8 runs = ~64 h sequential, ~8 h fully-parallel-on-8-GPUs. Design ablations are 2 more runs.

## Inference + evaluation

Greedy decoding over the test set (`--top_k 1` is deterministic, matches paper-table numbers). No `--lm_name` / `--adapter_variant` flags — `inference.py` reads them from the checkpoint's embedded config.

```bash
python src/inference.py --config configs/config.psc.emnlp.yaml \
  --checkpoint $SHARED/checkpoints/a0__main__v1/best.pt \
  --test_dir   $SHARED/data/processed_pyannote/test \
  --top_k      1
```

Loop over all adapter ablations:

```bash
for variant in film_mamba concat_only sigmoid_gate film \
               film_attn film_attn_2L film_mamba_2L qformer; do
  python src/inference.py --config configs/config.psc.emnlp.yaml \
    --checkpoint $SHARED/checkpoints/${variant}__v1/best.pt \
    --test_dir   $SHARED/data/processed_pyannote/test \
    --top_k      1
done
```

For the design-ablation checkpoints (`no_sections__v1`, `static_queries__v1`), same pattern.

### Outputs

Written **next to the checkpoint** (`dirname(--checkpoint)`):

- `inference_results.json` — per-clip records flushed every 50 clips via atomic tmp+rename. Keys per clip:
  - `generated`, `target`, `claims` (parsed `(feature, value)` pairs)
  - `sfs_precision / sfs_recall / sfs_f1`, `per_feature` breakdown
  - `attention_maps` — **only populated when `use_sections=true`** (the section-head ablation row). The headline LoRA + 8B path uses post-hoc attention extraction (see `scripts/extract_attention.py` — separate run after inference).
- `inference_summary.json` — aggregate `sfs_precision / recall / f1`, `per_feature_accuracy`, `gen_metrics: {bleu, rouge_l, bertscore_f1}`.

After inference, run the analysis scripts for paper-ready tables:

```bash
# Per-feature SFS breakdown (precision/recall/F1 per feature)
python scripts/per_feature_sfs.py \
  --inference_results $SHARED/checkpoints/film_mamba__v1/inference_results.json

# Overlap hedging contingency table (evidence for contribution #1)
python scripts/overlap_hedging_compare.py \
  --inference_results $SHARED/checkpoints/film_mamba__v1/inference_results.json
```

The same wandb run page used at training time gets `test/sfs_*` and `test/{bleu,rouge_l,bertscore_f1}` (because the checkpoint embeds `wandb_run_id`).

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

### Attention figures

**Section-head ablation row only** (`use_sections=true`). `scripts/plot_attention_spectrograms.py` reads the `attention_maps` field from `inference_results.json`, which is only populated when section_head fired during inference:

```bash
python scripts/plot_attention_spectrograms.py \
  --results   $SHARED/checkpoints/section_head__v1/inference_results.json \
  --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --output    docs/figures/attention_overlays \
  --limit     10 \
  --format    pdf
```

Each PDF has one row per emitted section + one per `<r>` range, log-mel in greyscale + attention map overlaid as a hot heatmap. `PatchGrid.reshape_attention` reshapes the flat per-section vector back to the 2D `(T_p, F_p)` grid before plotting.

**For the headline LoRA + 8B run** (no section_head), per-section attention maps are recovered post-hoc from the LM's native attention layers. `scripts/extract_attention.py` + `scripts/plot_attention.py` (in progress — task #59 / #60) do this without the section_head path. Same paper figure, no training-time coupling.

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

```bash
pip install pytest sacrebleu rouge-score bert-score
python -m pytest tests/ -v                           # full unit suite

# Standalone test scripts (no pytest needed)
python scripts/test_fix_descriptions.py
python scripts/test_fix_overlap_csv.py
python scripts/test_build_descriptions_deterministic.py
python -c "
import sys; sys.path[:0]=['src','tests']
import importlib
m = importlib.import_module('test_build_overlap_info')
n=ok=0
for x in dir(m):
    if x.startswith('test_'):
        n += 1
        try: getattr(m, x)(); ok += 1
        except AssertionError as e: print(f'FAIL {x}: {e}')
print(f'{ok}/{n} passed')
"
```

`sacrebleu`, `rouge-score`, `bert-score` are only needed for the corresponding metric — missing deps cause that one metric to silently skip.

The BEATs integration test downloads a ~370 MB checkpoint on first run; gated behind an env var:

```bash
RUN_BEATS_TEST=1 python -m pytest tests/test_section_query.py::TestBEATsIntegration -v
```

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
  build_descriptions_deterministic.py — Composes descriptions.json from CSV (no LLM)
  refresh_overlap_info.py            — Surgical overlap_info refresh in existing .pt files
  preprocess_beats.py                — Caches BEATs patches into per-clip .pt files
  plot_attention_spectrograms.py     — Renders per-clip attention overlays
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
- **Atomic writes everywhere except `last.pt`.** `preprocess_beats.py`, `refresh_overlap_info.py`, `fix_overlap_csv.py`, and `inference.py` all use `.tmp` + `os.replace` so a SIGKILL mid-write can't corrupt artifacts. `last.pt` from `train.py` does NOT yet use atomic write — if the session times out mid-save, `last.pt` is truncated and resume must use `best.pt` instead.
