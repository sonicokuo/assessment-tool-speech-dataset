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

Five stages end-to-end. **See [Pipeline commands (Bridges-2)](#pipeline-commands-bridges-2) below for the exact commands with PSC paths.**

```
Step 1: Feature extraction         src/feature_extractor_mix.py     →  features/{split}.csv       (includes overlap_segments from VAD on s1/s2 stems)
Step 2: Verbalize (Ollama)         scripts/feature_verbalization.py →  verbalized/{split}.csv
Step 3: Concatenate + JSON         scripts/merge_verbalized_to_json.py →  descriptions.json
Step 4: Preprocess audio (WavLM)   src/preprocess.py                →  processed/{train,val,test}/*.pt
Step 5: Train / evaluate           src/train.py, src/inference.py
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
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

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

### Pipeline commands (Bridges-2)

Activate the shared env first (see Environment Setup above), then:

```bash
cd /ocean/projects/cis260125p/shared/assessment-tool-speech-dataset
export SHARED=/ocean/projects/cis260125p/shared
mkdir -p $SHARED/data/features $SHARED/data/verbalized
```

**Step 1 — feature extraction** (uses Silero VAD on the Libri2Mix `s1/`/`s2/` stems for overlap detection — no HF token needed):

```bash
for split in train-100 dev test; do
  python src/feature_extractor_mix.py \
    --audio_dir      $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split/mix_clean \
    --libri2mix_root $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split \
    --output         $SHARED/data/features/${split}.csv
done
```

> If you want Pyannote-based overlap instead (on the mix itself, no stems), pass `--overlap pyannote --hf_token $HF_TOKEN`. The default `min_max_vad` is more accurate for Libri2Mix because it uses the clean speaker sources.

**Step 2 — verbalization (needs Ollama running with `gemma4:e2b`):**

Ollama is already installed at `$SHARED/ollama/` and the model is cached in `$SHARED/ollama_models/`. Each new compute-node session needs to activate the server:

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

**Step 3 — concatenate and build descriptions JSON:**

```bash
# Optional audit first — lists [ERROR] rows, gaps, and tail-missing ranges
python scripts/audit_verbalized_batches.py $SHARED/data/verbalized

# Consolidate all splits (incl. any leftover train-100_*_*.csv batches) into one JSON
python scripts/merge_verbalized_to_json.py \
  --verbalized_dir $SHARED/data/verbalized \
  --output         $SHARED/data/descriptions.json
#   --require_all   fail if any [ERROR] rows remain
```

**Step 4 — preprocess audio to `.pt` (VAD overlap ground truth from feature CSV):**

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

**Step 5 — train / evaluate with the committed PSC config:**

#### wandb setup (one-time per teammate)

Runs log to the team entity `speech_quality_adapter`, project `idl-ablation` → viewed at https://wandb.ai/speech_quality_adapter/idl-ablation.

```bash
# 1. Log in (paste API key from https://wandb.ai/authorize)
python -m wandb login
python -m wandb status      # confirm "Currently logged in as: <you>"

# 2. Route runs to the team entity (add to ~/.bashrc to persist across sessions)
echo 'export WANDB_ENTITY=speech_quality_adapter' >> ~/.bashrc
source ~/.bashrc
echo $WANDB_ENTITY          # should print: speech_quality_adapter
```

Without `WANDB_ENTITY`, runs go to your personal wandb account instead of the team. `wandb_project` is already set to `idl-ablation` in `config.psc.yaml`; no override needed.

#### Sanity test (smoke run, ~3-5 min)

Before committing H100 hours to the full 3-variant sweep, run a tiny end-to-end check. Confirms model loads, adapter builds, LoRA applies, train step runs, val + generation + SFS logs to wandb, checkpoint saves.

```bash
# Build a 20/10/10 .pt subset once
mkdir -p $SHARED/data/processed_smoke/{train,val,test}
cp $(ls $SHARED/data/processed/train/*.pt | head -20) $SHARED/data/processed_smoke/train/
cp $(ls $SHARED/data/processed/val/*.pt   | head -10) $SHARED/data/processed_smoke/val/
cp $(ls $SHARED/data/processed/test/*.pt  | head -10) $SHARED/data/processed_smoke/test/

# Launch — a smoke wandb run named "sanity-check"
python src/train.py --config configs/config.psc.yaml \
  --lm_name          Qwen/Qwen2.5-7B \
  --adapter_variant  film-attn \
  --data_dir         $SHARED/data/processed_smoke \
  --batch_size       4 \
  --epochs           1 \
  --save_dir         $SHARED/checkpoints/sanity_check \
  --wandb_run_name   sanity-check
```

Watch https://wandb.ai/speech_quality_adapter/idl-ablation — the `sanity-check` run should appear within ~30 sec and show:
- `params/lm_total ≈ 7.6B`, `params/trainable_total ≈ 55M`, `params/trainable_pct_of_lm ≈ 0.7%` in the run overview.
- `val_samples` table with 8 greedy-decoded samples + per-sample SFS after the val pass.
- `val_sfs_f1`, `val_loss`, `train_loss_step` scalar panels populating.

If the smoke run completes cleanly, delete the smoke artifacts and proceed to the real sweep:

```bash
rm -rf $SHARED/data/processed_smoke $SHARED/checkpoints/sanity_check
```

#### Baseline launch (uses whatever `lm_name` and `adapter_variant` are in the YAML)

```bash
python src/train.py     --config configs/config.psc.yaml
python src/inference.py --config configs/config.psc.yaml \
                        --checkpoint $SHARED/checkpoints/best.pt \
                        --test_dir   $SHARED/data/processed/test
```

### Swapping models and adapter variants via CLI

`train.py` accepts any `--key value` override for config entries — **no YAML edits needed per run**. The two most-swapped keys:

- `--lm_name <HF_model_id>` — swap the causal LM backbone
- `--adapter_variant <name>` — swap the audio→LM adapter architecture
- `--save_dir <path>`, `--wandb_run_name <label>` — isolate per-run checkpoints + logs

#### Available LMs (cached in `/ocean/projects/cis260125p/shared/hf_cache`)

| `--lm_name` | Size (bf16) | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B` | ~15 GB | Dense; safest fit on a single H100-80. Default for IDL report runs. |
| `Qwen/Qwen3.5-9B` | ~18 GB | Dense, Qwen3.5 family; matches `configs/config.yaml`'s original default. |
| `Qwen/Qwen3.6-35B-A3B` | ~70 GB | Sparse MoE. **Needs 4-bit quantization** (bitsandbytes) — straight bf16 will OOM even on H100-80. |

Swapping to a model not in that list will trigger a one-time HuggingFace download to the shared cache.

#### Available `--adapter_variant` values

Built in `src/adapter.py::build_adapter`:

- `concat-only` — baseline: concat audio + overlap features, no conditioning.
- `sigmoid-gate` — sigmoid-gated overlap-aware mixing.
- `film` — FiLM conditioning, no sequential context mixer.
- `film-attn` / `film-attn-2L` — FiLM + self-attention context (1 or 2 layers).
- `film-mamba` / `film-mamba-2L` — FiLM + Mamba SSM context (1 or 2 layers). **Default in `build_adapter`.**
- `qformer` — Q-Former style cross-attention (alternative architecture).

#### Three-run ablation recipe for the IDL report

Each teammate runs one line on their own H100 — separate `save_dir` keeps checkpoints from clobbering each other:

```bash
# Person 1 — concat-only baseline
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen2.5-7B \
  --adapter_variant concat-only \
  --save_dir        $SHARED/checkpoints/q25_7b_concat \
  --wandb_run_name  q25_7b-concat-only

# Person 2 — Q-Former alternative
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen2.5-7B \
  --adapter_variant qformer \
  --save_dir        $SHARED/checkpoints/q25_7b_qformer \
  --wandb_run_name  q25_7b-qformer

# Person 3 — FiLM + attention (proposed)
python src/train.py --config configs/config.psc.yaml \
  --lm_name         Qwen/Qwen2.5-7B \
  --adapter_variant film-attn \
  --save_dir        $SHARED/checkpoints/q25_7b_film_attn \
  --wandb_run_name  q25_7b-film-attn
```

Naming convention: `<lm-slug>_<variant>` — makes checkpoints self-describing across a 3×3 LM × variant sweep.

**Any config key** (`--batch_size`, `--epochs`, `--lr_adapter`, `--lora_rank`, …) can be overridden the same way — the override logic in `train.py` coerces bool/int/float values based on the YAML type. Strings pass through verbatim.
