"""Prepare AMI IHM (individual headset, close-talk) audio MATCHED to the SDM test clips.

Purpose: the AMI cross-domain GT for the intrinsic features (f0, speaking_rate, pauses)
was computed with Praat on the *SDM* (far-field, reverberant) channel, where Praat measures
reverberation rather than speech -> those GT columns are noise. This script re-derives clean
GT from the IHM (headset) channel of the *same* utterances.

The model input stays SDM (unchanged); only the GT source changes. We select the IHM
utterances corresponding to the SDM test clips (matched by meeting/speaker/begin/end, which are
identical across mics for the same AMI manual segment) and write each IHM wav named by the
SDM clip stem -> feature_extractor then produces a CSV keyed identically to the SDM model outputs.

Usage:
  python scripts/prepare_ami_ihm.py \
    --sdm_gt_csv  $SHARED/data/features_ami_sdm_test.csv \
    --out_wav_dir $SHARED/data/ami_ihm_matched/wav \
    --manifest    $SHARED/data/ami_ihm_matched/manifest.csv
"""
import argparse
import csv
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO_ID = "edinburghcstr/ami"
CONFIG = "ihm"
N_SHARDS = 4
TARGET_SR = 16000


def parse_key(audio_id: str):
    """AMI_<meeting>_<mic>_<speaker>_<begin_cs>_<end_cs> -> (meeting, speaker, begin_cs, end_cs).

    The mic field differs across configs (sdm vs H0x) but meeting/speaker/begin/end are the
    same for a given AMI manual segment, so this key aligns SDM and IHM clips 1:1.
    """
    p = audio_id.split("_")
    if len(p) < 6 or p[0] != "AMI":
        return None
    return (p[1], p[3], p[4], p[5])  # meeting, speaker, begin_cs, end_cs  (mic p[2] ignored)


def download_shards(split: str) -> list:
    from huggingface_hub import hf_hub_download
    paths = []
    for i in range(N_SHARDS):
        fn = f"{CONFIG}/{split}-{i:05d}-of-{N_SHARDS:05d}.parquet"
        print(f"  downloading {fn} ...", flush=True)
        paths.append(hf_hub_download(REPO_ID, fn, repo_type="dataset"))
    return paths


def to_mono16k(a, sr):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != TARGET_SR:
        import torch
        import torchaudio
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, TARGET_SR).numpy().astype(np.float32)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdm_gt_csv", required=True,
                    help="features_ami_sdm_test.csv; its 'filename' column defines the SDM clips to match")
    ap.add_argument("--out_wav_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    os.makedirs(args.out_wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)

    import pandas as pd

    # needed SDM clips -> {alignment key: sdm stem}
    sdm = pd.read_csv(args.sdm_gt_csv)
    need = {}
    for fn in sdm["filename"].astype(str):
        stem = fn[:-4] if fn.endswith(".wav") else fn
        k = parse_key(stem)
        if k:
            need[k] = stem
    print(f"{len(need)} SDM clips to match", flush=True)

    paths = download_shards(args.split)

    # pass 1: which IHM audio_ids match a needed SDM clip
    meta = pd.concat([pq.read_table(p, columns=["audio_id"]).to_pandas() for p in paths], ignore_index=True)
    sel = {}  # ihm audio_id -> sdm stem
    for aid in meta["audio_id"]:
        k = parse_key(aid)
        if k in need:
            sel[aid] = need[k]
    print(f"matched {len(sel)}/{len(need)} IHM utterances", flush=True)

    # pass 2: decode selected IHM audio, write wav named by SDM stem
    rows = []
    written = 0
    for p in paths:
        if written >= len(sel):
            break
        cols = pq.read_table(p).to_pydict()
        for j in range(len(cols["audio_id"])):
            aid = cols["audio_id"][j]
            if aid not in sel:
                continue
            try:
                arr, sr = sf.read(io.BytesIO(cols["audio"][j]["bytes"]))
            except Exception as e:
                print(f"  [skip] {aid}: decode failed ({e})")
                continue
            arr = to_mono16k(arr, sr)
            stem = sel[aid]
            sf.write(os.path.join(args.out_wav_dir, f"{stem}.wav"), arr, TARGET_SR)
            rows.append({
                "filename": f"{stem}.wav",
                "ihm_audio_id": aid,
                "meeting_id": cols["meeting_id"][j],
                "speaker_id": cols["speaker_id"][j],
                "begin_time": cols["begin_time"][j],
                "end_time": cols["end_time"][j],
                "duration_sec": round(len(arr) / TARGET_SR, 3),
                "text": cols["text"][j],
            })
            written += 1
            if written % 200 == 0:
                print(f"  wrote {written}/{len(sel)}", flush=True)

    rows.sort(key=lambda r: r["filename"])
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "ihm_audio_id", "meeting_id", "speaker_id",
                                          "begin_time", "end_time", "duration_sec", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. {written} IHM wavs -> {args.out_wav_dir}", flush=True)
    print(f"  manifest: {args.manifest}")
    if written < len(need):
        print(f"  [WARN] {len(need) - written} SDM clips had no IHM match")


if __name__ == "__main__":
    sys.exit(main())
