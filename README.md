# AQUA-NL — Adaptive Audio Quality Assessment and Description in Natural Language

CMU 11-785 IDL project targeting EMNLP. Speech-quality reasoning with
**evidence-grounded explanations**: take a mixture waveform, generate a
section-structured natural-language description of quality, and produce one
cross-attention map per section back to the input spectrogram.

```
audio  →  WavLM-Large (frozen)  →  Adapter (Conv8x + Mamba + FiLM)  →  audio prefix tokens
                                                                          ↓
                            BEATs (frozen, vendored)  →  spec patches  →  SectionQueryHead
                                                                          ↓
                                Qwen3-1.7B (full FT)  →  "<sec_noise><f_snr>…"
                                                                          ↓
                                       per-section / per-range attention maps  +  prose
```

Two contributions:

1. **Evidence-grounded section design.** Every `<sec_*>` token triggers a
   cross-attention query against BEATs patches, producing a time-frequency
   attention map for that claim. Multi-value features (overlap segments)
   are split via `<r>` markers, yielding per-range maps.
2. **Signal Faithfulness Score (SFS).** Regex-parses tagged numerical
   claims and scores them against per-feature ground-truth tolerances.

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

The current pipeline assumes `h100-80` at `batch_size=6` with full FT + gradient checkpointing. Smaller GPUs need `--batch_size 4 --gradient_accumulation_steps 2` and/or `--section_query_mode static`.

## Architecture

| Component | Choice | Why |
|---|---|---|
| LM | Qwen/Qwen3-1.7B, full FT | Small enough to fully fine-tune on one H100 |
| Spec encoder | BEATs (Microsoft, vendored under `src/beats/`) | Pretrained on AudioSet, 2D patch grid preserved through the encoder so cross-attention has time-frequency localization |
| Audio prefix encoder | WavLM-Large (frozen) | Used only to produce the LM audio prefix; not the evidence path |
| Adapter | Conv8× compression + Mamba context + FiLM overlap conditioning | Compresses WavLM frames 8× while injecting per-frame overlap-reliability signal |
| Section queries | Dynamic — query = `W_q · h_t` where `h_t` is the LM hidden state at each `<sec_*>` position | Matches the professor's "the model generates a query" framing; static lookup mode also available |
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

No LLM. Reads the VAD columns when present, falls back to pyannote columns with a one-shot warning. Produces structurally perfect output: every clip has all 6 sections in canonical order, every numerical claim wrapped in `<f_*>`, every overlap range wrapped in `<r>`. ~10 seconds end-to-end.

```bash
python scripts/build_descriptions_deterministic.py --all \
  --features-dir $SHARED/data/features_pyannote \
  --output       $SHARED/data/descriptions.json
```

Sample entry:

```
The recording is 3.920 s long.
<sec_noise><f_snr>The signal-to-noise ratio SNR is 13.17 dB</f></sec>.
<sec_reverb><f_srmr>The SRMR is 5.3646</f></sec>.
<sec_pitch><f_f0_mean>The F0 mean is 152.48 Hz</f> and
           <f_f0_sd>the F0 standard deviation SD is 62.88 Hz</f></sec>.
<sec_tempo><f_speaking_rate>The speaking rate is 6.888 syl/sec</f></sec>.
<sec_pauses><f_pause_count>The pause count is 1</f> and
            <f_pause_rate>the pause rate is 15.306 per min</f></sec>.
<sec_overlap><f_overlap_ratio>The overlap ratio is 0.5102</f> and
             <f_overlap_segments>overlap segments are present at <r>1.0-3.0s</r></f></sec>.
F0 and formant estimates are unreliable during overlap windows.
```

To rebuild only one third for distributed work:

```bash
python scripts/build_descriptions_deterministic.py --part 1 \
  --output $SHARED/data/descriptions.part1.json
```

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

### Step 5 — cache BEATs patches into each `.pt`

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

Uses `configs/config.psc.emnlp.yaml` defaults: Qwen3-1.7B full FT, `film-mamba` adapter, `use_sections: true`, `section_query_mode: dynamic`, BEATs cached, `batch_size: 6`, `epochs: 15`.

Foreground (interactive shell visible — output streams to your terminal):

```bash
python src/train.py --config configs/config.psc.emnlp.yaml
```

Detached (survives SSH disconnect; output goes to a log file). Put `LOG=...` on its own line so the variable lands in your current shell — chaining it after `&&` in front of `nohup ... &` groups everything into a subshell and `$LOG` becomes empty in the outer shell:

```bash
mkdir -p $SHARED/logs
LOG=$SHARED/logs/a0_main_$(date +%Y%m%d_%H%M%S).log

nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --wandb_run_name a0-main \
  --save_dir       $SHARED/checkpoints/a0__main__v1 \
  > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID  log=$LOG"
```

~33 min / epoch × 15 epochs ≈ 8 hours on a single H100.

**Follow progress in real time while detached**:

```bash
tail -f $LOG          # ctrl-C to stop following (the run keeps going)
```

What you should see:
- ~1 min in: model construction banner (`[tagged-mode]`, `[full-FT]`, `[sections] 6 sections registered`), parameter counts, dataset load (`Loaded: train=13900, val=3000`), and the wandb URL.
- After ~30 s of warmup: loss prints every `log_every` steps (default 10), starts near 9 and falls toward 1-3 over the first epoch.
- After each epoch: a val pass with BLEU / ROUGE / SFS, then a checkpoint write to `<save_dir>`.

If `tail -f` shows nothing changing for >5 min after the wandb URL appears, the loop is stalled — check `pgrep -af train.py` and the very tail of the log for a traceback.

### Resuming

Every epoch writes `<save_dir>/last.pt` (latest) and updates `<save_dir>/best.pt` (best-val-so-far). Resume restores adapter weights, optimizer, scheduler, epoch counter, and the wandb run ID (same run page continues).

```bash
python src/train.py --config configs/config.psc.emnlp.yaml \
  --resume_from $SHARED/checkpoints/qwen3_17b_full_ft_tagged_v1/last.pt
```

### Smoke run (50 micro-batches, no wandb)

The trainer accepts an inline `max_steps` cap via the config layer. Useful for verifying model construction + dataloader wiring before committing GPU hours:

```bash
WANDB_MODE=disabled python src/train.py --config configs/config.psc.emnlp.yaml --max_steps 50
```

Walltime ~1-2 min after the first-time Qwen3-1.7B HF download (~3.4 GB).

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

The headline run (`a0__main__v1`) uses the main config. Two ablation families:

### Adapter-architecture sweep

The paper's primary comparison table. Same data, same seed, same epochs, same section / BEATs / dynamic-query setup — **only `adapter_variant` changes**. Eight variants in `src/adapter.py::build_adapter`:

| ID | adapter_variant | Conditioning | Temporal mixer | Notes |
|---|---|---|---|---|
| **A0** | `film-mamba` | FiLM | Mamba SSM (1 layer) | Headline result (current default) |
| **A1** | `concat-only` | None (concat) | None | Baseline — argues for FiLM + mixer |
| **A2** | `sigmoid-gate` | Sigmoid gate | None | Lighter gating alternative |
| **A3** | `film` | FiLM | None | Isolates the temporal-mixer contribution |
| **A4** | `film-attn` | FiLM | Self-attention (1 layer) | Argues Mamba vs attn |
| **A5** | `film-attn-2L` | FiLM | Self-attention (2 layers) | Depth ablation for attn |
| **A6** | `film-mamba-2L` | FiLM | Mamba SSM (2 layers) | Depth ablation for Mamba |
| **A7** | `qformer` | Q-Former cross-attention | (implicit) | Alternative architecture |

```bash
# A0 main (film-mamba, headline)
LOG=$SHARED/logs/a0_main_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --save_dir       $SHARED/checkpoints/a0__main__v1 \
  --wandb_run_name a0-main \
  > "$LOG" 2>&1 &

# A1 concat-only
LOG=$SHARED/logs/a1_concat_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant concat-only \
  --save_dir        $SHARED/checkpoints/a1__concat_only__v1 \
  --wandb_run_name  a1-concat-only \
  > "$LOG" 2>&1 &

# A2 sigmoid-gate
LOG=$SHARED/logs/a2_sigmoid_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant sigmoid-gate \
  --save_dir        $SHARED/checkpoints/a2__sigmoid_gate__v1 \
  --wandb_run_name  a2-sigmoid-gate \
  > "$LOG" 2>&1 &

# A3 film (FiLM only, no temporal mixer)
LOG=$SHARED/logs/a3_film_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film \
  --save_dir        $SHARED/checkpoints/a3__film__v1 \
  --wandb_run_name  a3-film \
  > "$LOG" 2>&1 &

# A4 film-attn
LOG=$SHARED/logs/a4_film_attn_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-attn \
  --save_dir        $SHARED/checkpoints/a4__film_attn__v1 \
  --wandb_run_name  a4-film-attn \
  > "$LOG" 2>&1 &

# A5 film-attn-2L
LOG=$SHARED/logs/a5_film_attn_2L_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-attn-2L \
  --save_dir        $SHARED/checkpoints/a5__film_attn_2L__v1 \
  --wandb_run_name  a5-film-attn-2L \
  > "$LOG" 2>&1 &

# A6 film-mamba-2L
LOG=$SHARED/logs/a6_film_mamba_2L_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant film-mamba-2L \
  --save_dir        $SHARED/checkpoints/a6__film_mamba_2L__v1 \
  --wandb_run_name  a6-film-mamba-2L \
  > "$LOG" 2>&1 &

# A7 qformer
LOG=$SHARED/logs/a7_qformer_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --adapter_variant qformer \
  --save_dir        $SHARED/checkpoints/a7__qformer__v1 \
  --wandb_run_name  a7-qformer \
  > "$LOG" 2>&1 &
```

**Don't run more than one of these in parallel on the same GPU** — they'd OOM each other. Either sequence them (drop the `&`, run one at a time, ~8 h each = ~64 h end-to-end), or grab additional H100s in separate interact allocations.

### Design ablations (orthogonal to the adapter sweep)

For probing the section-attention design itself. Each toggles **one** of the EMNLP-novelty knobs while keeping the adapter at the headline `film-mamba`:

| ID | Knob | Override |
|---|---|---|
| **D1** | no sections, free-form prose | `--use_sections false --tagged_mode false` |
| **D2** | static section queries (learnable lookup) | `--section_query_mode static` |

Run these only after the adapter sweep is done, to argue novelty of the section-query design itself — not strictly needed for the paper's main result table.

```bash
# D1 no-sections
LOG=$SHARED/logs/d1_no_sections_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --use_sections false --tagged_mode false \
  --save_dir        $SHARED/checkpoints/d1__no_sections__v1 \
  --wandb_run_name  d1-no-sections \
  > "$LOG" 2>&1 &

# D2 static-queries
LOG=$SHARED/logs/d2_sec_static_$(date +%Y%m%d_%H%M%S).log
nohup python src/train.py --config configs/config.psc.emnlp.yaml \
  --section_query_mode static \
  --save_dir        $SHARED/checkpoints/d2__sec_static__v1 \
  --wandb_run_name  d2-sec-static \
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
for a in a0__main a1__concat_only a2__sigmoid_gate a3__film \
         a4__film_attn a5__film_attn_2L a6__film_mamba_2L a7__qformer; do
  python src/inference.py --config configs/config.psc.emnlp.yaml \
    --checkpoint $SHARED/checkpoints/${a}__v1/best.pt \
    --test_dir   $SHARED/data/processed_pyannote/test \
    --top_k      1
done
```

For the design-ablation checkpoints (`d1__no_sections__v1`, `d2__sec_static__v1`), same pattern.

### Outputs

Written **next to the checkpoint** (`dirname(--checkpoint)`):

- `inference_results.json` — per-clip records flushed every 50 clips via atomic tmp+rename. Keys per clip:
  - `generated`, `target`, `claims` (parsed `(feature, value)` pairs)
  - `sfs_precision / sfs_recall / sfs_f1`, `per_feature` breakdown
  - `attention_maps` — one flat-vector per section + one per `<r>` range:
    - `noise`, `reverb`, `pitch`, `tempo`, `pauses`, `overlap`
    - `overlap@<start>-<end>s` per emitted `<r>` range
- `inference_summary.json` — aggregate `sfs_precision / recall / f1`, `per_feature_accuracy`, `gen_metrics: {bleu, rouge_l, bertscore_f1}`.

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

```bash
python scripts/plot_attention_spectrograms.py \
  --results   $SHARED/checkpoints/a0__main__v1/inference_results.json \
  --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --output    docs/figures/attention_overlays \
  --limit     10 \
  --format    pdf
```

Each PDF has one row per emitted section + one per `<r>` range, log-mel in greyscale + attention map overlaid as a hot heatmap. `PatchGrid.reshape_attention` reshapes the flat per-section vector back to the 2D `(T_p, F_p)` grid before plotting.

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
  config.psc.emnlp.yaml    — Current default: Qwen3-1.7B full FT + sections + BEATs + dynamic
  config.psc.yaml          — Older baseline (Qwen3-8B + LoRA + tagged_mode: false), kept for reference

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

- **`Qwen/Qwen3-1.7B` is the instruct-tuned model.** Qwen3 doesn't have a separate `-Instruct` suffix; bare name is the instruct variant, `-Base` is the SSL-only one. Don't reintroduce the `-Instruct` suffix.
- **`overlap_info` in `.pt` files comes from `overlap_segments` (pyannote-on-mix), not `overlap_segments_vad`.** That's the intended input distribution. The description GT reads VAD-on-stems. Different signals on purpose, to prevent the trivial-copy data leakage.
- **Special tokens are added with `special_tokens=False`** so `tokenizer.decode(..., skip_special_tokens=True)` keeps them in the output string — required for both the SFS parser and the attention hook to find section positions.
- **`max_target_length: 224`** matches the trimmed 8-feature catalog (~9 spans, ~180-200 tokens median). The trimmed verbalization output is shorter than the 22-feature catalog the older recipes used.
- **gradient_checkpointing is mandatory at full FT** on H100-80 to stay under the memory budget at batch_size=6.
- **`<r>` markers exist inside `<f_overlap_segments>` only.** All other inner spans hold a single value and don't use `<r>`.
- **Atomic writes everywhere.** `preprocess_beats.py`, `refresh_overlap_info.py`, `fix_overlap_csv.py`, and `inference.py` all use `.tmp` + `os.replace` so a SIGKILL mid-write can't corrupt artifacts.
