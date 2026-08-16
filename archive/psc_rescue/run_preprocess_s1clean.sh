export HF_HOME=/ocean/projects/cis260125p/shared/hf_cache HF_HUB_OFFLINE=1
SH=/ocean/projects/cis260125p/shared
cd $SH/cur_train
for pair in train-100-s1clean:train:train-100 dev-s1clean:dev:dev test-s1clean:test:test; do
  aud=$(echo $pair | cut -d: -f1); out=$(echo $pair | cut -d: -f2); csv=$(echo $pair | cut -d: -f3)
  echo "### PREPROCESS-S1CLEAN $out ###"
  $SH/envs/project/bin/python -u src/preprocess.py \
    --audio_dir    $SH/data/audio_corrected/$aud \
    --features_csv $SH/data/features_corrected_merged/$csv.csv \
    --output_dir   $SH/data/processed_corrected/$out
done
echo "### S1CLEAN-PREPROCESS DONE ###"
