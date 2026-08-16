"""results_io — write a result file that can always be traced back to what produced it.

WHY THIS EXISTS
`results/MANIFEST.tsv` found that 28 of 47 live result files cannot be traced to the arm that
produced them. That set includes `metric_validity.json` (contribution I's load-bearing result),
`residual_head_s73.json`, `oracle_error_ceiling.json`, every causal result, and **every ridge
baseline**.

The ridge group is not hypothetical. The project record carries two values for "the tuned layer-7
ridge" (0.7200 and 0.7035, the latter being the UNTUNED one), and quoting the wrong one inflated
the unmasked arm's ill-posed margin from **+0.0100 to +0.0319** — reported before it was caught.
A provenance block on that file would have made the error mechanical to detect instead of
dependent on someone remembering which sweep produced which number.

`aux_repool.py` already writes its checkpoint path, which is exactly why all six `aux_*` rows in
the manifest resolve and the other 28 do not. This generalises that one good habit.

WHAT IT GUARANTEES
1. **Provenance travels with the number.** Producer script, checkpoint, clip count and a stamp go
   inside the file, so the manifest is derived rather than curated.
2. **No partial file is ever readable as a result.** Writes go to `<path>.inprogress` and are
   renamed on success. `src/inference.py` flushes every 50 clips, and a scorer reading one of
   those mid-flight once produced a fully formatted panel — coverage 1.0000, populated CI column —
   from n=50. Its only tell was the per-feature `n`.
3. **The clip count is recorded, not inferred.** Two numbers computed on different clip subsets
   are not comparable, and a prefix is not a random sample: the first 600 test clips are +0.034
   EASIER than the full set on robust5 and 0.047 HARDER on the ill-posed panel.

USAGE
    from eval.results_io import write_result
    write_result(summary, out_path=args.out, producer="uq_ridge_baseline",
                 checkpoint=args.checkpoint, n=len(clips))
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

PROVENANCE_KEY = "_provenance"


def provenance(producer: str, checkpoint: str | None, n: int | None,
               **extra: Any) -> dict:
    """The block that makes a number traceable. `checkpoint` may be a path or a fingerprint."""
    ckpt = checkpoint or ""
    return {
        "producer": producer,
        "checkpoint": ckpt,
        # the run directory is what a reader recognises, so surface it explicitly
        "arm": os.path.basename(os.path.dirname(ckpt)) if "/" in ckpt else ckpt,
        "n_clips": n,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def write_result(obj: Any, *, out_path: str, producer: str,
                 checkpoint: str | None = None, n: int | None = None,
                 **extra: Any) -> str:
    """Write `obj` as JSON with a provenance block, atomically.

    A dict gets the block under `_provenance`. A list (some producers emit per-clip arrays) is
    wrapped as {"_provenance": ..., "results": [...]} rather than silently losing the metadata —
    callers that need the bare list can read ["results"].
    """
    prov = provenance(producer, checkpoint, n, **extra)
    if isinstance(obj, dict):
        payload = dict(obj)
        payload[PROVENANCE_KEY] = prov
    else:
        payload = {PROVENANCE_KEY: prov, "results": obj}

    tmp = out_path + ".inprogress"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)          # atomic: a reader sees the old file or the new one
    return out_path


def read_provenance(path: str) -> dict | None:
    """Return the provenance block, or None if this file predates it (i.e. is untraceable)."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:                                            # noqa: BLE001
        return None
    return d.get(PROVENANCE_KEY) if isinstance(d, dict) else None
