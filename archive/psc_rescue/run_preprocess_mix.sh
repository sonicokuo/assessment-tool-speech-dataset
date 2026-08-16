export HF_HOME=/ocean/projects/cis260125p/shared/hf_cache HF_HUB_OFFLINE=1
SH=/ocean/projects/cis260125p/shared
cd $SH/cur_train
for pair in train-100:train dev:dev test:test; do
  aud=${pair%%:*}; out=${pair##*:}
  echo "### PREPROCESS $out (mix) ###"
  $SH/envs/project/bin/python -u src/preprocess.py \
    --audio_dir    $SH/data/audio_corrected/$aud \
    --features_csv $SH/data/features_corrected_merged/$aud.csv \
    --output_dir   $SH/data/processed_corrected/$out
done
echo "### MIX-PREPROCESS DONE ###"
