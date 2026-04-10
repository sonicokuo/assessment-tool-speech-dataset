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
