# assessment-tool-speech-dataset

Overlap-aware speech quality assessment pipeline: extract audio features, verbalize them into natural language descriptions, and train an adapter (WavLM + FiLM conditioning) that bridges audio representations into a causal LM for generating quality descriptions.

## Environment Setup

Requires **Python 3.10 or 3.11** and a CUDA-capable GPU (for mamba-ssm and training).

```bash
# Create conda environment
conda create -n project python=3.10
conda activate project

# Install PyTorch with CUDA (match your system's CUDA version)
# Check with: nvcc --version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # adjust cu126 to match

# On HPC clusters, load CUDA toolkit before installing mamba-ssm
module load cuda
module load gcc/13.2.1  # needs GCC >= 9

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

- **VERSA** (for SRMR reverberation metric): [github.com/wavlab-speech/versa](https://github.com/wavlab-speech/versa)
- **Ollama + gemma4:e2b** (for feature verbalization): [ollama.com](https://ollama.com)

## Pipeline

```
Step 1: Extract features → CSV
  python src/feature_extractor.py --audio_dir ./data/raw --output features.csv

Step 2: Split audio by speaker (prevents data leakage)
  python scripts/split_data.py --audio_dir ./data/raw --output_dir ./data/raw

Step 3: Verbalize features → natural language descriptions (requires Ollama)
  python scripts/feature_verbalization.py --input features.csv --output verbalized.csv

Step 4: Convert to JSON training targets
  python scripts/csv_to_json.py --input verbalized.csv --output data/descriptions.json

Step 5: Preprocess audio → .pt files (WavLM + overlap)
  python src/preprocess.py --audio_dir ./data/raw/train --output_dir ./data/processed/train
  python src/preprocess.py --audio_dir ./data/raw/val   --output_dir ./data/processed/val
  python src/preprocess.py --audio_dir ./data/raw/test  --output_dir ./data/processed/test

Step 6: Train
  python src/train.py --config configs/config.yaml

Step 7: Evaluate
  python src/inference.py --config configs/config.yaml --checkpoint ./checkpoints/best.pt --test_dir ./data/processed/test
```

## Project Structure

```
src/
  feature_extractor.py  - Audio feature extraction (SNR, overlap, F0, HNR, jitter, shimmer, pauses, speaking rate)
  preprocess.py         - WavLM feature extraction + Pyannote overlap → .pt files
  adapter.py            - Reliability-aware adapter with FiLM conditioning + ablation variants
  dataset.py            - Shared dataset and collate utilities
  train.py              - Training script (adapter + LoRA)
  inference.py          - Generation + SFS evaluation
  sfs.py                - Signal Faithfulness Score metric
scripts/
  feature_verbalization.py - LLM-based feature verbalization (Ollama/gemma4)
  csv_to_json.py           - Convert verbalized CSV to descriptions.json
  split_data.py            - Speaker-disjoint train/val/test splits
configs/
  config.yaml              - Training configuration
experiments/
  Feature_Extractor_Final.ipynb - Original feature extraction notebook
tests/
  test_sfs.py              - Tests for SFS claim parser and scorer
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

# 2. Activate the shared env
module load anaconda3
conda deactivate
conda activate /ocean/projects/cis260125p/shared/envs/project

# 3. Stop your ~/.local site-packages from shadowing the shared env
export PYTHONNOUSERSITE=1

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

Run these from the shared repo (`/ocean/projects/cis260125p/shared/assessment-tool-speech-dataset`) after activating the shared env.

```bash
export SHARED=/ocean/projects/cis260125p/shared
mkdir -p $SHARED/data/features $SHARED/data/verbalized
```

**Step 1 — feature extraction (needs `HF_TOKEN` for Pyannote overlap):**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx    # your HF token
for split in train-100 dev test; do
  python src/feature_extractor.py \
    --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/$split/mix_clean \
    --output    $SHARED/data/features/${split}.csv
done
```

**Step 2 — verbalization (needs Ollama running with `gemma4:e2b`):**

```bash
# One-time Ollama setup (user-local, no sudo):
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma4:e2b

# Verbalize each split
for split in train-100 dev test; do
  python scripts/feature_verbalization.py \
    --input  $SHARED/data/features/${split}.csv \
    --output $SHARED/data/verbalized/${split}.csv
done
```

**Step 3 — concatenate and build descriptions JSON:**

```bash
head -1     $SHARED/data/verbalized/train-100.csv  > $SHARED/data/verbalized_all.csv
tail -n +2 -q $SHARED/data/verbalized/*.csv       >> $SHARED/data/verbalized_all.csv

python scripts/csv_to_json.py \
  --input  $SHARED/data/verbalized_all.csv \
  --output $SHARED/data/descriptions.json
```

**Step 4 — preprocess audio to `.pt` (rename `dev` → `val`):**

```bash
python src/preprocess.py --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/mix_clean \
                         --output_dir $SHARED/data/processed/train
python src/preprocess.py --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/dev/mix_clean \
                         --output_dir $SHARED/data/processed/val
python src/preprocess.py --audio_dir $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
                         --output_dir $SHARED/data/processed/test
```

**Step 5 — train / evaluate with the committed PSC config:**

```bash
python src/train.py     --config configs/config.psc.yaml
python src/inference.py --config configs/config.psc.yaml \
                        --checkpoint $SHARED/checkpoints/best.pt \
                        --test_dir   $SHARED/data/processed/test
```
