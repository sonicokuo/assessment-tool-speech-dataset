#!/bin/bash
SH=/ocean/projects/cis260125p/shared
C=$SH/repo_verify/configs
NOTE=$SH/l7_note.txt
cat > $NOTE << "NOTEEOF"

# -- LAYER-7 + AUDIO-ONLY ARM (2026-08-08) --------------------------------------
# TWO changes from config.bsigma.attnconcat.yaml, both justified by measurement:
#  1. data_dir -> processed_layer7. The full-train 25-layer sweep (sanity-passed: layer 24
#     reproduces the recorded 0.6887 to 4 decimals) shows layer 7 beats layer 24 by +0.0148
#     robust5 and +0.0650 on the ill-posed panel, and the STRONGEST probe we could build
#     (tuned ridge, layer 7) reaches 0.7200 -- above our audio-only aux head 0.7066. Reporting
#     a layer-24 system against a layer-7 baseline would be self-sabotage.
#  2. zero_overlap_input -> overlap_info comes from ORACLE VAD on the CLEAN STEMS, which do not
#     exist at test time. Removing it is what makes "given only the waveform" true.
# NOT single-knob, deliberately: P1 (layer24 + audio-only) isolates knob 2 and the recorded
# baseline (layer24 + oracle, val 0.7194) anchors both. This is the arm we would SHIP.
# The layer-7 cache copies overlap_info/overlap_segments/filename VERBATIM and replaces only
# audio_features, so the layer is the only thing that differs.
zero_overlap_input: true
NOTEEOF
for SEED in 73 42; do
  sed -e "s|^seed: .*|seed: ${SEED}|" \
      -e "s|^data_dir: .*|data_dir: ${SH}/data/processed_layer7|" \
      -e "s|bsigma_attnconcat_seed73|l7audioonly_attnconcat_seed${SEED}|g" \
      $C/config.bsigma.attnconcat.yaml > $C/config.l7audioonly.s${SEED}.yaml
  cat $NOTE >> $C/config.l7audioonly.s${SEED}.yaml
done
rm -f $NOTE
echo "=== diff vs baseline ==="
diff $C/config.bsigma.attnconcat.yaml $C/config.l7audioonly.s73.yaml | grep -E "^[<>]" | head -8
echo "=== s42 seed ==="; grep -n "^seed:" $C/config.l7audioonly.s42.yaml
