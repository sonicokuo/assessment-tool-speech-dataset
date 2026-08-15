#!/usr/bin/env python3
"""merge_inference_shards.py — combine parallel `inference.py --out` shards into one results file.

WHY SHARDS EXIST
`inference.py` writes to a single fixed path and resumes by skipping filenames already present.
Two processes on the same checkpoint therefore RACE on the atomic tmp-then-rename and can clobber
each other, so a 6000-clip eval was stuck on one node for ~24 h. With `--out` each node writes a
disjoint `--start/--end` range to its own shard; this merges them.

SAFETY CHECKS (each corresponds to a way this has actually gone wrong before):
  * checkpoint fingerprint must agree across shards — merging generations produced by DIFFERENT
    weights is the exact silent corruption `inference.py`'s F12 guard exists to prevent.
  * duplicate filenames are reported, not silently deduped: overlap means the ranges were not
    disjoint, so the run needs fixing rather than papering over.
  * the merged count is checked against the expected clip total.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob, e.g. '<dir>/inference_shard_*.json'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=0, help="expected clip count (0 = skip check)")
    a = ap.parse_args()

    paths = sorted(glob.glob(a.shards))
    if not paths:
        print(f"[fatal] no shards matched {a.shards!r}")
        return 1

    merged, fps = [], set()
    for p in paths:
        try:
            rows = json.load(open(p))
        except Exception as e:                                   # noqa: BLE001
            print(f"[fatal] {p}: {type(e).__name__}: {e}")
            return 1
        for r in rows:
            fp = r.get("_checkpoint_fingerprint")
            if fp:
                fps.add(fp)
        merged.extend(rows)
        print(f"  {p}: {len(rows)} rows")

    if len(fps) > 1:
        print(f"[fatal] shards came from DIFFERENT checkpoints: {sorted(fps)}")
        print("[fatal] refusing to merge — this is the F12 silent-corruption case.")
        return 1

    names = [r.get("filename") for r in merged]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    if dupes:
        print(f"[warn] {len(dupes)} DUPLICATE filenames — shard ranges were not disjoint")
        print(f"[warn] first few: {dupes[:5]}")
        seen, dedup = set(), []
        for r in merged:
            if r.get("filename") in seen:
                continue
            seen.add(r.get("filename"))
            dedup.append(r)
        merged = dedup

    json.dump(merged, open(a.out, "w"))
    print(f"\nwrote {a.out}  clips={len(merged)}  shards={len(paths)}  "
          f"fingerprint={sorted(fps)[0] if fps else 'none'}")
    if a.expect and len(merged) != a.expect:
        print(f"[warn] expected {a.expect} clips, merged {len(merged)} — "
              f"{a.expect - len(merged)} missing; check shard ranges cover the full set")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
