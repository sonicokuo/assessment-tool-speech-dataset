# assessment-tool-speech-dataset

Overlap-aware speech quality assessment pipeline: extract audio features, verbalize them into natural language descriptions, and train an adapter (WavLM + FiLM conditioning) that bridges audio representations into a causal LM for generating quality descriptions.

## Environment Setup

Requires **Python 3.10 or 3.11** and a CUDA-capable GPU (for mamba-ssm and training).

Activate the shared env on PSC Bridges-2:

```bash
module load anaconda3
conda activate /ocean/projects/cis260125p/shared/envs/project
export PYTHONNOUSERSITE=1   # stop ~/.local from shadowing the shared env
```

See [PSC Team Workflow](#psc-team-workflow) below for interact-node and allocation details.

### Optional Dependencies

- **VERSA** (for SRMR reverberation metric): [github.com/wavlab-speech/versa](https://github.com/wavlab-speech/versa)
- **Ollama + gemma4:e2b** (for feature verbalization): [ollama.com](https://ollama.com)

## Pipeline

Six stages end-to-end — four data-prep stages, then training and evaluation. See [Data pipeline (Bridges-2)](#data-pipeline-bridges-2), [Training](#training), and [Evaluation](#evaluation) below for exact commands.

```
Step 1: Feature extraction         src/feature_extractor_mix.py        →  features/{split}.csv       (includes overlap_segments from VAD on s1/s2 stems)
Step 2: Verbalize (Ollama)         scripts/feature_verbalization.py    →  verbalized/{split}.csv
Step 3: Concatenate + JSON         scripts/merge_verbalized_to_json.py →  descriptions.json
Step 4: Preprocess audio (WavLM)   src/preprocess.py                   →  processed/{train,val,test}/*.pt
Step 5: Train                      src/train.py                        →  checkpoints/{best,last}.pt  + wandb
Step 6: Evaluate                   src/inference.py                    →  inference_{results,summary}.json + wandb test/*
```

Libri2Mix ships with speaker-disjoint `train-clean-100 / dev-clean / test-clean` splits, so `scripts/split_data.py` is **not** used in this workflow.

## Project Structure

```
src/
  feature_extractor.py      - Pyannote-based feature extractor (for cross-domain datasets without clean stems)
  feature_extractor_mix.py  - Libri2Mix extractor; default overlap method is Silero VAD on s1/s2 stems (oracle labels)
  preprocess.py             - WavLM features + 5-dim per-frame overlap context from the feature CSV → .pt files
  adapter.py                - Reliability-aware adapter with FiLM conditioning + ablation variants
  dataset.py                - Shared dataset and collate utilities
  train.py                  - Training script (adapter + LoRA)
  inference.py              - Generation + SFS evaluation
  sfs.py                    - Signal Faithfulness Score metric (regex-based claim parser)
  text_metrics.py           - BLEU-4 / ROUGE-L / BERTScore-F1 helpers (complement SFS)
scripts/
  feature_verbalization.py         - LLM-based feature verbalization (Ollama/gemma4)
  audit_verbalized_batches.py      - Scans verbalized CSVs for [ERROR] rows, gaps, tail-missing ranges
  merge_verbalized_to_json.py      - Concatenates verbalized CSVs across splits into descriptions.json
  csv_to_json.py                   - Legacy single-CSV → JSON (kept for smoke tests)
  split_data.py                    - Speaker-disjoint train/val/test splits (unused for Libri2Mix)
configs/
  config.yaml               - Training configuration (toy/laptop defaults)
  config.psc.yaml           - PSC Bridges-2 configuration (shared-storage paths + Qwen model)
experiments/
  Feature_Extractor_Final.ipynb    - Original feature extraction notebook
tests/
  test_sfs.py               - Tests for SFS claim parser and scorer
  test_text_metrics.py      - Tests for BLEU / ROUGE-L / BERTScore wrapper
```

## Testing

```bash
pip install pytest sacrebleu rouge-score bert-score
python -m pytest tests/ -v
```

`sacrebleu`, `rouge-score`, and `bert-score` are only needed for the BLEU / ROUGE-L / BERTScore metrics reported alongside SFS at inference and val-generation time (see [Evaluation](#evaluation) below). If any are missing, the relevant metric is silently skipped.

## PSC Team Workflow

The team shares one conda environment on PSC Bridges-2 at:

```
/ocean/projects/cis260125p/shared/envs/project
```

### Daily workflow

```bash
# 1. Grab an H100 node (H100s on Bridges-2 are 80 GB, partition tag is h100-80)
interact -p GPU-shared --gres=gpu:h100-80:1 -t 8:00:00 -A cis260125p
# Alternatives if H100s are full:
#   --gres=gpu:v100-32:1
#   --gres=gpu:l40s-48:1
# If `-p GPU-shared` errors for H100, try `-p GPU` instead.

# 2. Activate the shared env — see Environment Setup at the top of this README.

# 3. Move into the shared repo
cd /ocean/projects/cis260125p/shared/assessment-tool-speech-dataset
git pull origin main   # grab latest code

# 4. Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Shorter activation name (one-time per teammate)

Append the shared envs directory so `conda activate project` works instead of the full path:

```bash
conda config --append envs_dirs /ocean/projects/cis260125p/shared/envs
# from now on:
conda activate project
```

### Group membership check

Confirm you're on the project allocation:

```bash
groups | tr ' ' '\n' | grep cis260125p
```

If this prints nothing, the PI needs to add you to the `cis260125p` allocation before `interact -A cis260125p` will work.

### Valid Bridges-2 GPU types (for `--gres=gpu:<type>:N`)

| Tag | Memory |
|---|---|
| `v100-16` | 16 GB |
| `v100-32` | 32 GB |
| `l40s-48` | 48 GB |
| `h100-80` | 80 GB |

### Shared data layout

Generated datasets and pipeline artifacts live under `/ocean/projects/cis260125p/shared/data/`:

```
/ocean/projects/cis260125p/shared/data/
├── Libri2Mix/Libri2Mix/wav16k/min/{train-100,dev,test}/mix_clean/*.wav  # input audio
├── wham_noise/                                                           # WHAM source noise
├── features/{train-100,dev,test}.csv          # step 1 output
├── verbalized/{train-100,dev,test}.csv        # step 2 output
├── verbalized_all.csv                         # concatenated (for step 3)
├── descriptions.json                          # step 3 output — consumed by training
└── processed/{train,val,test}/*.pt            # step 4 output — consumed by training
```

Checkpoints land in `/ocean/projects/cis260125p/shared/checkpoints/`.

## Data pipeline (Bridges-2)

Activate the shared env first (see [Environment Setup](#environment-setup) above), then:

```bash
cd /ocean/projects/cis260125p/shared/assessment-tool-speech-dataset
export SHARED=/ocean/projects/cis260125p/shared
mkdir -p $SHARED/data/features $SHARED/data/verbalized
```

### Step 1 — feature extraction

Uses Silero VAD on the Libri2Mix `s1/`/`s2/` stems for overlap detection — no HF token needed.

```bash
for split in train-100 dev test; do
  python src/feature_extractor_mix.py \
    --audio_dir      $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split/mix_clean \
    --libri2mix_root $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split \
    --output         $SHARED/data/features/${split}.csv
done
```

> If you want Pyannote-based overlap instead (on the mix itself, no stems), pass `--overlap pyannote --hf_token $HF_TOKEN`. The default `min_max_vad` is more accurate for Libri2Mix because it uses the clean speaker sources.

### Step 2 — verbalization

Needs Ollama running with `gemma4:e2b`. Ollama is already installed at `$SHARED/ollama/` and the model is cached in `$SHARED/ollama_models/`. Each new compute-node session needs to activate the server:

```bash
# Activate Ollama (every new interact node)
export PATH=$SHARED/ollama/bin:$PATH
export LD_LIBRARY_PATH=$SHARED/ollama/lib:$LD_LIBRARY_PATH
export OLLAMA_MODELS=$SHARED/ollama_models

# Start the server in the background if not already running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama-$USER.log 2>&1 &
  sleep 3
fi

# Verbalize each split
for split in train-100 dev test; do
  python scripts/feature_verbalization.py \
    --input  $SHARED/data/features/${split}.csv \
    --output $SHARED/data/verbalized/${split}.csv
done
```

### Step 3 — concatenate and build descriptions JSON

```bash
# Optional audit first — lists [ERROR] rows, gaps, and tail-missing ranges
python scripts/audit_verbalized_batches.py $SHARED/data/verbalized

# Consolidate all splits (incl. any leftover train-100_*_*.csv batches) into one JSON
python scripts/merge_verbalized_to_json.py \
  --verbalized_dir $SHARED/data/verbalized \
  --output         $SHARED/data/descriptions.json
#   --require_all   fail if any [ERROR] rows remain
```

### Step 4 — preprocess audio to `.pt`

`preprocess.py` reads the `overlap_segments` column of each split's feature CSV (the oracle VAD labels from Step 1) and writes a per-clip `.pt` with `audio_features (T, 1024)` + `overlap_info (T, 5)`. The five per-frame overlap channels are: `is_overlap`, `segment_duration_s`, `frac_through_segment`, `clip_overlap_ratio`, `density_300ms`.

```bash
# Output dir names follow train.py's convention (train / val / test); rename dev→val here.
python src/preprocess.py \
  --audio_dir    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/mix_clean \
  --features_csv $SHARED/data/features/train-100.csv \
  --output_dir   $SHARED/data/processed/train

python src/preprocess.py \
  --audio_dir    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/dev/mix_clean \
  --features_csv $SHARED/data/features/dev.csv \
  --output_dir   $SHARED/data/processed/val

python src/preprocess.py \
  --audio_dir    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --features_csv $SHARED/data/features/test.csv \
  --output_dir   $SHARED/data/processed/test
```

## Training

`train.py` accepts any `--key value` override for config entries — **no YAML edits needed per run**. The most-swapped keys:

- `--lm_name <HF_model_id>` — swap the causal LM backbone
- `--adapter_variant <name>` — swap the audio→LM adapter architecture
- `--save_dir <path>`, `--wandb_run_name <label>` — isolate per-run checkpoints + logs

**Any config key** (`--batch_size`, `--epochs`, `--lr_adapter`, `--lora_rank`, …) can be overridden the same way — the override logic in `train.py` coerces bool/int/float values based on the YAML type. Strings pass through verbatim.

### wandb setup (one-time per teammate)

Team runs live at **https://wandb.ai/speech_quality_adapter/idl-ablation** (entity `speech_quality_adapter`, project `idl-ablation`).

```bash
# 1. Log in (paste API key from https://wandb.ai/authorize)
python -m wandb login
python -m wandb status      # confirm "Currently logged in as: <you>"

# 2. Route runs to the team entity — persist in ~/.bashrc so you never forget
echo 'export WANDB_ENTITY=speech_quality_adapter' >> ~/.bashrc
source ~/.bashrc
echo $WANDB_ENTITY          # must print: speech_quality_adapter
```

> ⚠️ **If `WANDB_ENTITY` is not set**, `wandb.init()` defaults to your **personal account** and the run lands at `wandb.ai/<your-username>/idl-ablation` — not the team. This happens silently; the URL in the run's startup banner is the only signal. Always eyeball it after launch.

Verify the routing is right after launching any run — the line the script prints:

```
wandb: 🚀 View run at https://wandb.ai/speech_quality_adapter/idl-ablation/runs/...
                         ^^^^^^^^^^^^^^^^^^^^^^^
                         must be the team, NOT your username
```

If it shows your username instead, Ctrl+C the run, `export WANDB_ENTITY=speech_quality_adapter`, and relaunch. (Wandb does not support moving runs between entities — delete the misrouted one from the UI after.)

`wandb_project` is already set to `idl-ablation` in `config.psc.yaml`; no override needed.

### Sanity test (smoke run, ~3-5 min)

Before committing H100 hours to the full 3-variant sweep, run a tiny end-to-end check. Confirms model loads, adapter builds, LoRA applies, train step runs, val + generation + SFS logs to wandb, checkpoint saves.

```bash
# Build a 20/10/10 .pt subset once
mkdir -p $SHARED/data/processed_smoke/{train,val,test}
cp $(ls $SHARED/data/processed/train/*.pt | head -20) $SHARED/data/processed_smoke/train/
cp $(ls $SHARED/data/processed/val/*.pt   | head -10) $SHARED/data/processed_smoke/val/
cp $(ls $SHARED/data/processed/test/*.pt  | head -10) $SHARED/data/processed_smoke/test/

# Launch — a smoke wandb run named "sanity-check"
python src/train.py --config configs/config.psc.yaml \
  --lm_name          Qwen/Qwen3-8B \
  --adapter_variant  film-attn \
  --data_dir         $SHARED/data/processed_smoke \
  --batch_size       4 \
  --epochs           1 \
  --save_dir         $SHARED/checkpoints/sanity_check \
  --wandb_run_name   sanity-check
```

Watch https://wandb.ai/speech_quality_adapter/idl-ablation — the `sanity-check` run should appear within ~30 sec and show:
- `params/lm_total ≈ 8-9B`, `params/trainable_total ≈ 90-100M` (adapter + LoRA), `params/trainable_pct_of_lm ≈ 1-1.2%` in the run overview.
- `val_samples` table with 8 greedy-decoded samples + per-sample SFS after the val pass.
- `val_sfs_f1`, `val_loss`, `train_loss_step` scalar panels populating.

If the smoke run completes cleanly, delete the smoke artifacts and proceed to the real sweep:

```bash
rm -rf $SHARED/data/processed_smoke $SHARED/checkpoints/sanity_check
```

### Available LMs (cached in `/ocean/projects/cis260125p/shared/hf_cache`)

| `--lm_name` | Size (bf16) | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B` | ~15 GB | Dense, 28 layers. Lightest; fits bs=8 on H100-80 without gradient checkpointing. |
| `Qwen/Qwen3-8B` | ~16 GB | Dense, Qwen3 family. **Default for IDL report runs.** Fits bs=6 on H100-80 (bs=8 OOMs without gradient checkpointing). |
| `Qwen/Qwen3.5-9B` | ~18 GB | Dense, Qwen3.5 family. Needs `bs=4 + grad_accum=2` and/or `--gradient_checkpointing true`. |
| `Qwen/Qwen3.6-35B-A3B` | ~70 GB | Sparse MoE. **Needs 4-bit quantization** (bitsandbytes) — straight bf16 will OOM even on H100-80. |

Swapping to a model not in that list will trigger a one-time HuggingFace download to the shared cache.

### Available `--adapter_variant` values

Built in `src/adapter.py::build_adapter`:

- `concat-only` — baseline: concat audio + overlap features, no conditioning.
- `sigmoid-gate` — sigmoid-gated overlap-aware mixing.
- `film` — FiLM conditioning, no sequential context mixer.
- `film-attn` / `film-attn-2L` — FiLM + self-attention context (1 or 2 layers).
- `film-mamba` / `film-mamba-2L` — FiLM + Mamba SSM context (1 or 2 layers). **Default in `build_adapter`.**
- `qformer` — Q-Former style cross-attention (alternative architecture).

### Baseline launch (uses YAML defaults)

Uses whatever `lm_name` and `adapter_variant` are in `configs/config.psc.yaml`:

```bash
python src/train.py --config configs/config.psc.yaml
```

### Phase-2 training recipe (B-full + Pyannote inputs, default as of 2026-04-25)

The current `configs/config.psc.yaml` defaults run **multi-task B-full** training: each batch
computes a prose CE loss, a bare-numbers CE loss, and an auxiliary-regression MSE loss, blended
as `loss = lambda_prose * lm_ce_prose + lambda_nums * lm_ce_nums + lambda_mse * masked_mse`.
The audio→numerical-feature mapping gets a direct, undiluted gradient via the aux head (bypassing
the LM and the digit-subword tokenizer).

Three orthogonal interventions stack in this single retraining sweep:
1. **FiLM init fix** (`src/adapter.py:80`) — `gamma.weight` init changed from `zeros_` to `normal_(0, 0.01)`
   so overlap signal has nonzero gradient through FiLM at step 0. (Was making FiLM-* variants
   blind to overlap at init while concat-only saw it directly.)
2. **B-full multi-task supervision** — see `compute_loss` in `src/train.py`. Two LM forwards per
   batch share the same audio prefix; the bare-numbers target's CE is ~75% digit tokens vs the prose
   target's ~12%, concentrating numerical-grounding gradient on the same output channel SFS evaluates.
3. **Pyannote-on-mix overlap inputs** — `overlap_info` channels come from Pyannote (4 channels:
   `is_overlap`, `segment_duration_s`, `frac_through_segment`, `density_300ms`). The `clip_overlap_ratio`
   channel was removed because it's an SFS-evaluated feature; feeding it as model input was data leakage.

**One-time preprocessing on PSC** (~3-5 hours; needs `HF_TOKEN` for gated Pyannote repos):

```bash
export HF_TOKEN=hf_...
bash scripts/run_pyannote_preprocessing.sh
```

Outputs land at `$SHARED/data/features_pyannote/` and `$SHARED/data/processed_pyannote/`. The
original `features/` and `processed/` directories are not touched, so legacy training paths still
work if you point the YAML back at them.

**Retrain a variant** (use a fresh `save_dir` to keep legacy 5-channel runs separate):

```bash
python src/train.py --config configs/config.psc.yaml \
  --adapter_variant film-mamba \
  --save_dir        $SHARED/checkpoints/q3_8b_film_mamba_v2 \
  --wandb_run_name  q3_8b-film-mamba-v2
```

Diagnostic wandb scalars per epoch: `train_loss_lm_prose`, `train_loss_lm_nums`, `train_loss_mse`,
plus their `val_*` counterparts. If `val_loss_mse` drops fast but `val_sfs_f1` stays flat → the
adapter encodes numbers but the LM isn't reading the prefix; investigate prompt format / LoRA target
modules. If both stay flat → optimizer/data issue.

### Three-run ablation recipe for the IDL report

Each teammate runs one line on their own H100 — separate `save_dir` keeps checkpoints from clobbering each other:

```bash
# Person 1 — concat-only baseline
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant concat-only \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_concat \
  --wandb_run_name  q3_8b-concat-only

# Person 2 — Q-Former alternative
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant qformer \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_qformer \
  --wandb_run_name  q3_8b-qformer

# Person 3 — FiLM + attention (proposed)
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant film-attn \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_film_attn \
  --wandb_run_name  q3_8b-film-attn
```

> Qwen3-8B OOMs at bs=8 on H100-80; bs=6 is the tested-safe setting. If you prefer the default bs=8 from the config, swap to Qwen2.5-7B or add `--gradient_checkpointing true`.

Naming convention: `<lm-slug>_<variant>` — makes checkpoints self-describing across a 3×3 LM × variant sweep.

### Extended ablation — the remaining adapter variants

The three-run recipe above covers the minimum story (baseline / popular alt / FiLM). If you have H100-hours to spare, run the remaining variants from `build_adapter` for a stronger paper table. All use the same LM, bs, and training budget so the comparison stays apples-to-apples.

```bash
# sigmoid-gate — a lighter overlap-aware mixing alternative to FiLM
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant sigmoid-gate \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_sigmoid_gate \
  --wandb_run_name  q3_8b-sigmoid-gate

# film — FiLM conditioning only, no sequential context mixer
# Isolates the contribution of the temporal mixer (attn vs mamba vs nothing).
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant film \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_film \
  --wandb_run_name  q3_8b-film

# film-mamba — FiLM + Mamba SSM context (1 layer) — the default in build_adapter
# The "proposed" variant for the paper's main claim. Worth running if mamba-ssm installs cleanly.
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant film-mamba \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_film_mamba \
  --wandb_run_name  q3_8b-film-mamba

# film-attn-2L — FiLM + self-attention context (2 layers)
# Tests whether deeper context helps over 1-layer film-attn.
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant film-attn-2L \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_film_attn_2L \
  --wandb_run_name  q3_8b-film-attn-2L

# film-mamba-2L — FiLM + Mamba (2 layers)
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant film-mamba-2L \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_film_mamba_2L \
  --wandb_run_name  q3_8b-film-mamba-2L
```

Suggested filtering by story:
- **Ablate the temporal mixer** (FiLM fixed, swap mixer): `film`, `film-attn`, `film-mamba` — argues for Mamba over attention.
- **Ablate the conditioning mechanism** (mixer fixed, swap conditioning): `concat-only`, `sigmoid-gate`, `film-attn` — argues for FiLM over naive concat/gate.
- **Depth ablation**: `film-attn` vs `film-attn-2L`, `film-mamba` vs `film-mamba-2L` — argues 1 layer is enough / 2 layers help.

Pick whichever sub-table strengthens your paper's thesis; you don't need to publish all 8 variants.

### Resuming a crashed or preempted run

Every epoch writes `$SAVE_DIR/last.pt` (latest state) and updates `$SAVE_DIR/best.pt` (best-val-so-far). Resume by passing `--resume_from <path_to_last.pt>` — adapter + LoRA weights, optimizer state, scheduler state, epoch counter, best-val-loss, and the wandb run ID are all restored (so the same wandb run continues, not a new one).

```bash
# Person 1 — resume concat-only
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant concat-only \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_concat \
  --wandb_run_name  q3_8b-concat-only \
  --resume_from     $SHARED/checkpoints/q3_8b_concat/last.pt

# Person 2 — resume qformer
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant qformer \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_qformer \
  --wandb_run_name  q3_8b-qformer \
  --resume_from     $SHARED/checkpoints/q3_8b_qformer/last.pt

# Person 3 — resume film-attn
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen3-8B \
  --adapter_variant film-attn \
  --batch_size      6 \
  --save_dir        $SHARED/checkpoints/q3_8b_film_attn \
  --wandb_run_name  q3_8b-film-attn \
  --resume_from     $SHARED/checkpoints/q3_8b_film_attn/last.pt
```

Extended-ablation resumes follow the same pattern — add `--resume_from $SHARED/checkpoints/<save_dir>/last.pt` to the matching launch line.

Typical reasons to resume: OOM mid-epoch, srun/sbatch time limit hit, node preemption, intentional restart with changed hyperparameters. To extend training beyond the original `epochs`, add `--epochs N` alongside `--resume_from`.

## Evaluation

Inference evaluates the best checkpoint on the test set with **four complementary metrics**.

### What the metrics measure

| Metric | What it measures | When to trust it |
|---|---|---|
| **SFS-F1** (primary) | Numerical faithfulness — regex-extracts claims like `"SNR of 15.63 dB"` from the generated text and checks each against the ground-truth measurement within per-feature tolerances. Covers 13 features (SNR, HNR, F0 mean/SD, jitter, shimmer, SRMR, duration, overlap ratio, overlap spans (IoU≥0.8), speaking/articulation rate, pause count/rate, sample rate). | This is the metric the paper is about. If SFS-F1 is high, the model actually "reads" the audio rather than hallucinating numbers. `SFS-precision` = fraction of claims correct; `SFS-recall` = fraction of ground-truth features mentioned. |
| **BLEU-4** | Surface n-gram precision vs the reference description. Rewards exact wording and n-gram overlap. | Sanity check for fluency and template conformance. Low BLEU + high SFS usually means the model uses different phrasing but stays factual (often good). Low BLEU + low SFS means broken output. |
| **ROUGE-L (F1)** | Longest common subsequence between hyp and ref. More robust than BLEU to reordering. | Complements BLEU for summarization-like overlap; interpret as "how much of the reference structure survived." |
| **BERTScore-F1** | Embedding-similarity (RoBERTa-large) averaged token-by-token. Captures semantic equivalence even when wording differs. | The right metric for catching paraphrases. High BERTScore + low BLEU = paraphrased but faithful. Low BERTScore = the model is saying something unrelated to the reference. |

### Interpretation grid

Useful when writing the results discussion:

| SFS-F1 | BLEU/ROUGE | Diagnosis |
|---|---|---|
| high | high | Fluent *and* factual — ideal. |
| high | low | Factual but uses unusual phrasing. Acceptable; no fix needed. |
| low | high | Fluent hallucinations — model reproduces reference templates but gets numbers wrong. Train longer / improve conditioning. |
| low | low | Generator broken — decoding degenerate, checkpoint regressed, or GT/pred misalignment bug. |

### Test / inference commands for the report

Each teammate runs inference on **their own** trained checkpoint. The `save_dir` here must match the one passed at training time — `best.pt` inside that directory is what gets evaluated. Greedy decoding (`--top_k 1`) is used so the paper numbers are deterministic.

> **No `--lm_name` / `--adapter_variant` needed.** `inference.py` reads the training config embedded in the checkpoint and auto-sets `lm_name`, `adapter_variant`, and the LoRA hyperparameters. You'll see a `[config] <key>: old → new (from checkpoint)` line per substitution in the console at startup.

**Three-run ablation (matches the training recipe):**

```bash
# Person 1 — concat-only baseline
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_concat/best.pt \
  --test_dir   $SHARED/data/processed/test \
  --top_k      1

# Person 2 — Q-Former alternative
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_qformer/best.pt \
  --test_dir   $SHARED/data/processed/test \
  --top_k      1

# Person 3 — FiLM + attention (proposed)
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film_attn/best.pt \
  --test_dir   $SHARED/data/processed/test \
  --top_k      1
```

**Extended ablation (matches the extended training block):**

```bash
# sigmoid-gate
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_sigmoid_gate/best.pt \
  --test_dir   $SHARED/data/processed/test --top_k 1

# film (FiLM only, no mixer)
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film/best.pt \
  --test_dir   $SHARED/data/processed/test --top_k 1

# film-mamba (proposed variant with Mamba mixer)
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film_mamba/best.pt \
  --test_dir   $SHARED/data/processed/test --top_k 1

# film-attn-2L (deeper attn mixer)
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film_attn_2L/best.pt \
  --test_dir   $SHARED/data/processed/test --top_k 1

# film-mamba-2L (deeper Mamba mixer)
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film_mamba_2L/best.pt \
  --test_dir   $SHARED/data/processed/test --top_k 1
```

**Useful decoding flags:**
- `--top_k 1` → greedy (recommended for the paper table; deterministic).
- `--temperature 0.7 --top_p 0.9` → diverse sampling (only for qualitative inspection).
- `--checkpoint_device cpu` → load checkpoint via CPU before moving to GPU (for smaller GPUs).

#### Running by range + resuming crashes

Full test set is **3000 clips** → ~60-90 min on Qwen3-8B. You can slice that up with `--start N --end M` (half-open: includes `N`, excludes `M`) and rerun safely — the script auto-resumes.

**Quick reference — find your case, copy the flags:**

| You want to… | Flags |
|---|---|
| Process the whole test set (default) | *(no flags)* |
| Process the first 500 clips | `--start 0 --end 500` |
| Process clips 500–999 (500 total) | `--start 500 --end 1000` |
| Resume a crashed full-set run | *(no flags — same command again)* |
| Finish the remainder after doing 0-500 | `--start 500 --end 3000` |
| Quick 50-clip smoke check | `--start 0 --end 50` |

**Full copy-paste example:**

```bash
# First chunk — clips 0..499
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film_attn/best.pt \
  --test_dir   $SHARED/data/processed/test \
  --top_k 1 --start 0 --end 500

# Later — clips 500..2999 (auto-skips the 500 already done)
python src/inference.py --config configs/config.psc.yaml \
  --checkpoint $SHARED/checkpoints/q3_8b_film_attn/best.pt \
  --test_dir   $SHARED/data/processed/test \
  --top_k 1 --start 500 --end 3000
```

**What to expect in the console:**

- **At startup** (on a rerun): `[resume] Found N already-scored clips in .../inference_results.json` — confirms the auto-skip is active.
- **During the loop**: `X/end done (range); Y/3000 total on disk` every 10 clips. `inference_results.json` is flushed every **50 clips** via atomic `tmp → rename`, so a crash costs at most 50 clips of work.
- **At the end**: if the run didn't cover the full test set, you'll see `[partial] N/3000 clips scored so far …`. The aggregate in `inference_summary.json` and on wandb is computed over whatever is on disk at that moment — final numbers are only trustworthy once `N == 3000`.

> **Don't run two range jobs on the same checkpoint simultaneously.** Both would flush to the same `inference_results.json` and race — the later flush overwrites in-flight work from the other. Range-parallel across *different* ablation checkpoints (different `save_dir`s) is fine. Within one checkpoint, go sequential.

**Loop form** if one teammate runs all eight at once:

```bash
for variant in concat qformer film_attn sigmoid_gate film film_mamba film_attn_2L film_mamba_2L; do
  python src/inference.py --config configs/config.psc.yaml \
    --checkpoint $SHARED/checkpoints/q3_8b_${variant}/best.pt \
    --test_dir   $SHARED/data/processed/test \
    --top_k      1
done
```

### Where results land

Inference outputs are written **next to the checkpoint** (i.e. `dirname(--checkpoint)`), not to the YAML's `save_dir` — so each ablation's results sit beside its own `best.pt`:

- **`inference_results.json`** — per-clip records (flushed every 50 clips during the loop): `filename`, `generated`, `target`, extracted `claims`, `sfs_precision/recall/f1`, and `per_feature` breakdown.
- **`inference_summary.json`** — aggregate numbers for the paper table: `sfs_precision`, `sfs_recall`, `sfs_f1`, `per_feature_accuracy`, and `gen_metrics: {bleu, rouge_l, bertscore_f1}`. Written at the end of each invocation over whatever is currently on disk.

On wandb (default-on; disable with `wandb_log_test: false` in the config): the same run page that has the training curves gets new test-set scalars under `test/*`:

- `test/sfs_precision`, `test/sfs_recall`, `test/sfs_f1`
- `test/bleu`, `test/rouge_l`, `test/bertscore_f1`

Because `best.pt` stores the original `wandb_run_id`, inference resumes the same wandb run instead of creating a new one — so train, val, and test metrics sit side-by-side on one page.

### Training-time monitoring

The same metrics are logged every epoch on the 8-sample val slice (`src/train.py`). BLEU and ROUGE-L are always-on (near-free on 8 samples); BERTScore is opt-in via `use_bertscore: true` in the config (it downloads a ~1 GB RoBERTa-large model on first use). Scalars: `val_sfs_precision/recall/f1`, `val_bleu`, `val_rouge_l`, `val_bertscore_f1`. Per-epoch JSON dumps of the 8 samples also land at `$SAVE_DIR/val_samples/epoch_NNN.json` for offline inspection.

> Training checkpoint selection is driven by **val_loss only** — BLEU/ROUGE/BERTScore are reported for diagnostics, not for `best.pt` selection. Saving-best on a similarity metric would push the model toward reference-copying and *lower* SFS.
