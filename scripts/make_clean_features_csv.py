#!/usr/bin/env python3
"""Splice clean-stem GT (from compute_clean_features.py, and optionally clean-frame
F0 from compute_clean_f0.py) into a single clean-GT features CSV.

The shipped features_pyannote/{dev,test,train-100}.csv have snr/srmr/rate/pause
measured on the 2-speaker MIXTURE (corrupted by overlap). This substitutes the
well-posed clean-stem values so build_descriptions_observability.py rebuilds the
RECOVERABLE numbers from clean GT, and SFS scores against clean GT.

Columns substituted from --clean_features:
    snr_db, srmr,
    praat_speaking_rate_syl_sec, praat_articulation_rate_syl_sec,
    praat_pause_count, praat_pause_rate_per_min,
    praat_mean_pause_dur_sec, praat_total_pause_dur_sec, praat_pause_to_speech_ratio
Columns optionally substituted from --clean_f0 (same as make_clean_f0_csv.py):
    f0_mean_hz, f0_sd_hz   (blanked when the clean-frame F0 is undefined)

Never touched: overlap_ratio / overlap_ratio_vad / overlap_segments* (the
VAD-on-stems oracle is already correct), duration_sec, sample_rate_hz.

SNR is BLANKED for a clip whose clean entry has snr_no_interferer=True (a clean
single-speaker clip with no s2 stem -> no interferer to define a ratio against).
A blank cell makes SFS skip the feature for that clip rather than score a bogus
number. SRMR left blank (e.g. a --skip_srmr login-node pass) is preserved from the
original CSV so the column is never silently zeroed.

Usage:
  python scripts/make_clean_features_csv.py \
    --in_csv          features_pyannote/test.csv \
    --clean_features  clean_features_test.json \
    [--clean_f0       clean_f0_test.json] \
    --out_csv         features_pyannote/test_clean.csv
"""
import argparse
import csv
import json
import math


# columns lifted from the clean-features JSON
_CLEAN_FEATURE_KEYS = (
    "snr_db", "srmr",
    "praat_speaking_rate_syl_sec", "praat_articulation_rate_syl_sec",
    "praat_pause_count", "praat_pause_rate_per_min",
    "praat_mean_pause_dur_sec", "praat_total_pause_dur_sec",
    "praat_pause_to_speech_ratio",
)


def _undef(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or x == ""


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--clean_features", required=True)
    ap.add_argument("--clean_f0", default=None,
                    help="optional clean-frame F0 JSON to also substitute f0_mean_hz/f0_sd_hz")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    clean = json.loads(open(args.clean_features).read())
    cf0 = json.loads(open(args.clean_f0).read()) if args.clean_f0 else {}

    rows = list(csv.DictReader(open(args.in_csv)))
    cols = list(rows[0].keys()) if rows else []

    n_snr = n_srmr = n_rate = n_snr_blank = n_f0_sub = n_f0_blank = 0
    for r in rows:
        fn = (r.get("filename") or "").strip()
        cl = clean.get(fn)
        if cl is not None:
            # SNR (blank when no interferer was available)
            if cl.get("snr_no_interferer"):
                if "snr_db" in r:
                    r["snr_db"] = ""
                    n_snr_blank += 1
            elif not _undef(cl.get("snr_db")) and "snr_db" in r:
                r["snr_db"] = _fmt(cl["snr_db"])
                n_snr += 1
            # SRMR (preserve original if this pass skipped it)
            if not _undef(cl.get("srmr")) and "srmr" in r:
                r["srmr"] = _fmt(cl["srmr"])
                n_srmr += 1
            # rate + pauses
            touched_rate = False
            for k in _CLEAN_FEATURE_KEYS:
                if k in ("snr_db", "srmr"):
                    continue
                if k in r and not _undef(cl.get(k)):
                    r[k] = _fmt(cl[k])
                    touched_rate = True
            if touched_rate:
                n_rate += 1

        # optional clean-frame F0 (same semantics as make_clean_f0_csv.py)
        c = cf0.get(fn)
        if c is not None:
            cm, cs = c.get("f0_mean_hz"), c.get("f0_sd_hz")
            if _undef(cm):
                if "f0_mean_hz" in r:
                    r["f0_mean_hz"] = ""
                if "f0_sd_hz" in r:
                    r["f0_sd_hz"] = ""
                n_f0_blank += 1
            else:
                if "f0_mean_hz" in r:
                    r["f0_mean_hz"] = f"{round(float(cm), 2)}"
                if "f0_sd_hz" in r and not _undef(cs):
                    r["f0_sd_hz"] = f"{round(float(cs), 2)}"
                n_f0_sub += 1

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}: {len(rows)} rows")
    print(f"  clean SNR substituted {n_snr} (blanked-no-interferer {n_snr_blank}), "
          f"clean SRMR {n_srmr}, clean rate/pause {n_rate}")
    if args.clean_f0:
        print(f"  clean-F0 substituted {n_f0_sub}, blanked-unmeasurable {n_f0_blank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
