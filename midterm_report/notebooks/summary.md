# Dataset Quality Scoring for Speaker Model Training

## What this notebook does

We built a toolkit that analyzes audio datasets and scores them by quality, so other teams can pick the best training data for their speaker models. The pipeline uses pyannote segmentation-3.0 to detect overlapping speech, then combines that with SNR, duration, and spectral features to produce a quality score for each audio sample.

## How it works

1. Create synthetic audio samples at four overlap levels from LibriSpeech
2. Run pyannote's pretrained overlap detector on each sample
3. Extract additional quality signals like SNR and spectral flatness
4. Score each sample using two methods and rank the datasets

## Two experiments

Experiment 1 scores quality based only on detected overlap. Clean speech scores highest, heavy overlap scores lowest.

Experiment 2 combines overlap with SNR, duration, and spectral clarity into a weighted score. The idea is that overlap alone doesn't capture everything about audio quality.

## Results

The pipeline correctly ranks datasets from cleanest to most overlapped. Both scoring methods produce the right ordering, with Experiment 1 showing stronger separation between categories.

We also ran a filtering analysis showing how quality thresholds affect dataset composition, a weight sensitivity study across five configurations, and statistical significance tests between profiles.

## Limitations

The results are on synthetic data, not real-world recordings. We created overlapping audio by adding two LibriSpeech utterances together, but pyannote was trained on real conversations with natural turn-taking and interruptions. This means the model overestimates overlap on clean audio and has a compressed detection range across profiles. The multi-signal scoring doesn't help much here because all samples come from the same source, so SNR and duration barely vary.

## Next steps

Run the same pipeline on VoxBlink and VoxCeleb data on the PSC cluster. These are real recordings with natural overlaps and diverse recording conditions, which is what pyannote was trained on. We also plan to add speaker embedding purity signals and connect this to the difficulty scoring from our proposal.

## Files

- `osd_baseline_colab.ipynb` is the main notebook, runs on Colab with A100 GPU
- `figures/` has all six plots for the report
- `results.json` has the raw numbers
- `requirements.txt` lists dependencies
