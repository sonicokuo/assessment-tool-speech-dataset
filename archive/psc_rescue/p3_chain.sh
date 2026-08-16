#!/bin/bash
SH=/ocean/projects/cis260125p/shared
# wait for the full-train layer sweep to WRITE ITS JSON, then confirm the file is settled
# (an earlier chain raced savez_compressed and read a half-written 1.1 GB file)
for i in $(seq 1 200); do
  if [ -f "$SH/layer_sweep_full.json" ]; then
    s1=$(stat -c %s "$SH/layer_sweep_full.json"); sleep 5
    s2=$(stat -c %s "$SH/layer_sweep_full.json")
    [ "$s1" = "$s2" ] && break
  fi
  sleep 30
done
cd $SH/repo_verify
$SH/envs/project/bin/python -u $SH/mel_ridge.py \
  "$SH/data/audio_corrected/train-100,$SH/data/audio_corrected/train-100-s1clean" \
  "$SH/data/audio_corrected/test,$SH/data/audio_corrected/test-s1clean" \
  $SH/data/features_corrected_merged/train-100.csv \
  $SH/data/features_corrected_merged/test.csv \
  $SH/mel_ridge.json
