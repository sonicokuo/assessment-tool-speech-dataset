"""Prepare an AMI SDM (single-distant-mic, far-field) cross-domain test set.

Downloads the `edinburghcstr/ami` `sdm` config test split (4 parquet shards,
~1.24 GB), extracts utterance-level clips using AMI's manual segment boundaries,
and writes them as 16 kHz mono wavs in the same layout the Libri2Mix pipeline
expects (so src/feature_extractor.py / src/preprocess.py run unchanged).

Outputs:
  <out_wav_dir>/<audio_id>.wav          one clip per selected utterance
  <manifest>                            filename,meeting_id,speaker_id,begin_time,end_time,duration_sec,text
  <segments_csv>                        ALL test-split segments (every utterance, no
                                        duration filter) -> needed by
                                        compute_ami_overlap_gt.py for oracle overlap.

The clip filename is the AMI `audio_id` (globally unique, encodes
meeting/mic/speaker/begin/end), so the stem flows consistently through
feature_extractor -> build_descriptions -> preprocess -> inference.

Usage (gate, 20 clips):
  python scripts/prepare_ami_sdm.py --limit 20 \
    --out_wav_dir $SHARED/data/ami_sdm/wav/gate \
    --manifest    $SHARED/data/ami_sdm/manifest_gate.csv \
    --segments_csv $SHARED/data/ami_sdm/segments_test.csv

Full test set:
  python scripts/prepare_ami_sdm.py \
    --out_wav_dir $SHARED/data/ami_sdm/wav/test \
    --manifest    $SHARED/data/ami_sdm/manifest_test.csv \
    --segments_csv $SHARED/data/ami_sdm/segments_test.csv
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
CONFIG = "sdm"
N_SHARDS = 4
TARGET_SR = 16000
META_COLS = ["meeting_id", "audio_id", "speaker_id", "begin_time", "end_time", "text"]


def download_shards(split: str) -> list[str]:
    """Fetch the parquet shards for one split into the HF cache; return local paths."""
    from huggingface_hub import hf_hub_download

    paths = []
    for i in range(N_SHARDS):
        fn = f"{CONFIG}/{split}-{i:05d}-of-{N_SHARDS:05d}.parquet"
        print(f"  downloading {fn} ...", flush=True)
        paths.append(hf_hub_download(REPO_ID, fn, repo_type="dataset"))
    return paths


def to_mono16k(audio_array: np.ndarray, sr: int) -> np.ndarray:
    audio_array = np.asarray(audio_array, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != TARGET_SR:
        import torch
        import torchaudio

        audio_array = (
            torchaudio.functional.resample(torch.from_numpy(audio_array), sr, TARGET_SR)
            .numpy()
            .astype(np.float32)
        )
    return audio_array


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_wav_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--segments_csv", required=True)
    ap.add_argument("--min_dur", type=float, default=3.0)
    ap.add_argument("--max_dur", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of extracted clips (stratified round-robin across "
                         "meetings for variety); default = all eligible clips")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.segments_csv)), exist_ok=True)

    paths = download_shards(args.split)

    # ---- Pass 1: cheap metadata table (no audio) for segments + clip selection ----
    print("Reading segment metadata (no audio) ...", flush=True)
    import pandas as pd

    meta = pd.concat(
        [pq.read_table(p, columns=META_COLS).to_pandas() for p in paths],
        ignore_index=True,
    )
    print(f"  {len(meta)} total segments across {meta['meeting_id'].nunique()} meetings")
    assert meta["audio_id"].is_unique, "audio_id is not unique — filename collisions possible"

    # segments.csv: every utterance (full set, used for oracle overlap GT).
    seg_out = meta[["meeting_id", "speaker_id", "begin_time", "end_time", "audio_id"]].copy()
    seg_out.sort_values(["meeting_id", "begin_time"]).to_csv(args.segments_csv, index=False)
    print(f"  wrote {args.segments_csv}")

    # Eligible clips: duration window.
    meta["duration_sec"] = (meta["end_time"] - meta["begin_time"]).round(3)
    elig = meta[(meta["duration_sec"] >= args.min_dur) & (meta["duration_sec"] <= args.max_dur)].copy()
    print(f"  {len(elig)} clips in [{args.min_dur},{args.max_dur}] s window")

    # Selection: stratified round-robin across meetings (variety incl. overlap).
    if args.limit is not None and args.limit < len(elig):
        rng = np.random.default_rng(args.seed)
        by_meeting = {m: g.sample(frac=1.0, random_state=args.seed).to_dict("records")
                      for m, g in elig.groupby("meeting_id")}
        order = sorted(by_meeting)
        selected = []
        while len(selected) < args.limit and any(by_meeting.values()):
            for m in order:
                if by_meeting[m]:
                    selected.append(by_meeting[m].pop())
                    if len(selected) >= args.limit:
                        break
        sel_df = pd.DataFrame(selected)
    else:
        sel_df = elig
    sel_ids = set(sel_df["audio_id"])
    print(f"  selected {len(sel_ids)} clips for extraction")

    # ---- Pass 2: stream shards WITH audio, decode + write only selected clips ----
    manifest_rows = []
    written = 0
    for p in paths:
        if written >= len(sel_ids):
            break
        tbl = pq.read_table(p)
        cols = tbl.to_pydict()
        n = len(cols["audio_id"])
        for j in range(n):
            aid = cols["audio_id"][j]
            if aid not in sel_ids:
                continue
            audio = cols["audio"][j]
            try:
                arr, sr = sf.read(io.BytesIO(audio["bytes"]))
            except Exception as e:
                print(f"  [skip] {aid}: decode failed ({e})")
                continue
            arr = to_mono16k(arr, sr)
            out_path = os.path.join(args.out_wav_dir, f"{aid}.wav")
            sf.write(out_path, arr, TARGET_SR)
            manifest_rows.append({
                "filename": f"{aid}.wav",
                "meeting_id": cols["meeting_id"][j],
                "speaker_id": cols["speaker_id"][j],
                "begin_time": cols["begin_time"][j],
                "end_time": cols["end_time"][j],
                "duration_sec": round(len(arr) / TARGET_SR, 3),
                "text": cols["text"][j],
            })
            written += 1
            if written % 100 == 0:
                print(f"  wrote {written}/{len(sel_ids)} clips", flush=True)

    manifest_rows.sort(key=lambda r: r["filename"])
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "meeting_id", "speaker_id",
                                          "begin_time", "end_time", "duration_sec", "text"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nDone. wrote {written} wavs to {args.out_wav_dir}")
    print(f"  manifest: {args.manifest}")
    if written < len(sel_ids):
        print(f"  [WARN] {len(sel_ids) - written} selected clips were not found/decoded")


if __name__ == "__main__":
    sys.exit(main())
