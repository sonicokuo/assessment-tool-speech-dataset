#!/usr/bin/env python3
"""build_results_manifest.py — one row per number-bearing file under $SHARED/results.

WHY THIS EXISTS
Every wrong-artefact trap in this project had the same shape: two files whose names did not say
which arm, which clip count, or which producer they came from, sitting next to each other.
  * `saliency_test_zeroovl.npz` vs `saliency_zeroovl_test.npz` — word order only, different
    checkpoints, 21.5 MB vs 1.65 MB.
  * `inference_results.CONTAMINATED_oracle_overlap.json` vs `inference_results.json` — 5 KB apart
    out of 23.5 MB, adjacent in any listing.
  * `layer_sweep.json` (12k-clip subset, answers best_layer 4) vs `layer_sweep_full.json`
    (full train, answers 7).
  * a scored panel reporting robust5 0.5961 at n=50 whose successor at n=600 says 0.7710.
None of those is detectable by reading the filename, and each one has been quoted or nearly quoted.

The fix is not discipline, it is a lookup table. This walks results/ and emits a TSV with, per
file: its category, size, mtime, the producing script where that is recoverable, the checkpoint
FINGERPRINT the tooling already writes into most result JSONs, and the clip count. A number can
then be traced to its arm without opening anything.

WHAT IT DOES NOT DO
It does not guess. A field it cannot establish is written as `?` rather than inferred from the
filename — filenames are exactly the signal that has proven unreliable here. `?` in the producer
column is a real finding: it means that result cannot currently be traced to the code that made it.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

# result-JSON keys that identify provenance, in priority order
CKPT_KEYS = ("_checkpoint_fingerprint", "checkpoint_fingerprint", "ckpt", "checkpoint")
N_KEYS = ("n", "n_clips", "clips", "num_clips")


def probe_json(p: pathlib.Path) -> dict:
    """Read provenance out of a result JSON without assuming a schema."""
    out = {"producer": "?", "ckpt": "?", "n": "?"}
    try:
        with p.open() as fh:
            d = json.load(fh)
    except Exception:                                            # noqa: BLE001
        return out
    if isinstance(d, list):
        out["n"] = str(len(d))
        d = d[0] if d and isinstance(d[0], dict) else {}
    if not isinstance(d, dict):
        return out
    # NEW-STYLE: a _provenance block written by eval.results_io.write_result is authoritative
    # and needs no guessing. Files predating it fall through to the heuristics below.
    prov = d.get("_provenance")
    if isinstance(prov, dict):
        out["producer"] = str(prov.get("producer") or "?")
        out["ckpt"] = str(prov.get("arm") or prov.get("checkpoint") or "?")
        if prov.get("n_clips") is not None:
            out["n"] = str(prov["n_clips"])
        return out

    for k in CKPT_KEYS:
        if k in d and d[k]:
            v = str(d[k])
            out["ckpt"] = os.path.basename(os.path.dirname(v)) if "/" in v else v
            break
    for k in N_KEYS:
        if k in d and isinstance(d[k], (int, float)):
            out["n"] = str(int(d[k])); break
    if out["n"] == "?":
        # summary-shaped files carry n inside the per-readout blocks
        for v in d.values():
            if isinstance(v, dict):
                for kk in N_KEYS:
                    if isinstance(v.get(kk), (int, float)):
                        out["n"] = str(int(v[kk])); break
            if out["n"] != "?":
                break
    for k in ("producer", "script", "_script"):
        if isinstance(d.get(k), str):
            out["producer"] = d[k]; break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default=None, help="default: <results_dir>/MANIFEST.tsv")
    a = ap.parse_args()

    root = pathlib.Path(a.results_dir)
    out_p = pathlib.Path(a.out) if a.out else root / "MANIFEST.tsv"
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.tsv":
            continue
        rel = p.relative_to(root)
        cat = rel.parts[0] if len(rel.parts) > 1 else "(top)"
        info = probe_json(p) if p.suffix == ".json" else {"producer": "?", "ckpt": "?", "n": "?"}
        st = p.stat()
        rows.append([str(rel), cat, f"{st.st_size}",
                     __import__("time").strftime("%Y-%m-%d", __import__("time").localtime(st.st_mtime)),
                     info["ckpt"], info["n"], info["producer"]])

    hdr = ["path", "category", "bytes", "mtime", "checkpoint", "n_clips", "producer"]
    with out_p.open("w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")

    untraceable = sum(1 for r in rows if r[4] == "?" and r[0].endswith(".json"))
    print(f"wrote {out_p}  ({len(rows)} files)")
    print(f"  JSON results with NO recoverable checkpoint fingerprint: {untraceable}")
    print("  '?' is not a formatting gap — it means that number cannot currently be traced to")
    print("  the arm that produced it. Those are the rows worth fixing at the producer.")
    by = {}
    for r in rows:
        by[r[1]] = by.get(r[1], 0) + 1
    print("  by category:", by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
