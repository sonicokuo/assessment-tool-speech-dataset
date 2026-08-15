#!/bin/bash
# Verify every analysis script this session depends on is IDENTICAL local vs PSC.
# Twice today a local fix was compiled, grepped, described as done -- and never scp'd, so
# the remote ran stale code and produced plausible-but-wrong output (the "signed" captures
# were byte-identical to unsigned). Checksums make that impossible to miss.
REMOTE=/ocean/projects/cis260125p/shared/repo_verify/scripts
FILES="capture_saliency.py score_attribution.py build_oracle_maps.py build_oracle_maps_events.py
        build_snr_map.py snr_gain_slopes.py causal_dose.py causal_dose_analyze.py causal_remix.py
        causal_remix_analyze.py snr_triviality_panel.py"
printf '%-34s %-10s %s\n' FILE STATUS NOTE
for f in $FILES; do
  [ -f "scripts/$f" ] || { printf '%-34s %-10s %s\n' "$f" "LOCAL-MISS" "-"; continue; }
  L=$(md5 -q "scripts/$f" 2>/dev/null || md5sum "scripts/$f" | cut -d' ' -f1)
  R=$(ssh -o ConnectTimeout=20 -i ~/.ssh/psc_key slin32@bridges2.psc.edu \
        "md5sum $REMOTE/$f 2>/dev/null | cut -d' ' -f1" 2>/dev/null | tr -d '\r')
  if [ -z "$R" ]; then printf '%-34s %-10s %s\n' "$f" "REMOTE-MISS" "never transferred"
  elif [ "$L" = "$R" ]; then printf '%-34s %-10s %s\n' "$f" "OK" ""
  else printf '%-34s %-10s %s\n' "$f" "**STALE**" "remote differs - scp before running"
  fi
done
