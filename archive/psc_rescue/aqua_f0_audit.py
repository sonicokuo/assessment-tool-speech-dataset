"""Audit: is the audio-only arm's f0 coverage collapse real, or an artifact?

Run on PSC login node (CPU-light: JSON + regex only).
  $SH/envs/project/bin/python aqua_f0_audit.py
"""
import json
import re
import statistics
import sys

SH = "/ocean/projects/cis260125p/shared"
sys.path.insert(0, f"{SH}/repo_verify/src")
from eval.sfs import ClaimParser  # noqa: E402

P = ClaimParser()

FEATS = ["snr", "srmr", "hnr", "f0_mean", "f0_sd", "jitter", "shimmer",
         "speaking_rate", "pause_count", "pause_rate", "overlap_ratio"]

# ────────────────────────────────────────────────────────────────────────────
# PART 1 — TRAINING TARGETS: fraction of targets with an f0 VALUE vs a hedge,
# split clean vs mixture, for fw (what both arms trained on) and fw2.
# ────────────────────────────────────────────────────────────────────────────
print("#" * 78)
print("# PART 1 — TRAINING-TARGET CONTENT (canonical descriptions JSONs)")
print("#" * 78)
for fname in ["descriptions_corrected_fw.json", "descriptions_corrected_fw2.json"]:
    try:
        d = json.load(open(f"{SH}/data/{fname}"))
    except Exception as e:  # noqa: BLE001
        print(fname, "LOAD FAIL", e)
        continue
    stats = {g: dict(n=0, f0_val=0, f0_sd_val=0, hedge=0, hnr_val=0,
                     shim_val=0, jit_val=0) for g in ("clean", "mix")}
    sample_clean = None
    for stem, txt in d.items():
        g = "clean" if "_s1clean" in stem else "mix"
        s = stats[g]
        s["n"] += 1
        if re.search(r"The F0 mean is \d", txt):
            s["f0_val"] += 1
        if re.search(r"The F0 standard deviation SD is \d", txt):
            s["f0_sd_val"] += 1
        if "cannot be reliably estimated" in txt:
            s["hedge"] += 1
        if re.search(r"The HNR is \d", txt):
            s["hnr_val"] += 1
        if re.search(r"The shimmer is \d", txt):
            s["shim_val"] += 1
        if re.search(r"The jitter is \d", txt):
            s["jit_val"] += 1
        if g == "clean" and sample_clean is None:
            sample_clean = (stem, txt)
    print(f"\n== {fname}  total={len(d)}")
    for g, s in stats.items():
        n = max(1, s["n"])
        print(f"  {g:6s} n={s['n']:6d}"
              f"  f0_val={s['f0_val']:6d} ({s['f0_val'] / n:6.1%})"
              f"  f0_sd_val={s['f0_sd_val']:6d}"
              f"  hedge={s['hedge']:6d} ({s['hedge'] / n:6.1%})"
              f"  hnr={s['hnr_val']:6d} ({s['hnr_val'] / n:5.1%})"
              f"  shim={s['shim_val']:6d} ({s['shim_val'] / n:5.1%})"
              f"  jit={s['jit_val']:6d} ({s['jit_val'] / n:5.1%})")
    if sample_clean:
        print(f"  sample CLEAN target [{sample_clean[0]}]:\n    {sample_clean[1][:500]}")

# ────────────────────────────────────────────────────────────────────────────
# PART 2 — GENERATIONS: repo ClaimParser (authoritative) on all three arms,
# split clean/mix: claim counts per feature, hedges, degeneration stats.
# ────────────────────────────────────────────────────────────────────────────
print()
print("#" * 78)
print("# PART 2 — GENERATIONS (repo ClaimParser, NOT ad-hoc regex)")
print("#" * 78)

FILES = {
    "audioonly_free (audioonly_attnconcat_seed73)":
        f"{SH}/checkpoints/full/audioonly_attnconcat_seed73/inference_results.json",
    "oracle_free (bsigma_attnconcat_seed73)":
        f"{SH}/checkpoints/full/bsigma_attnconcat_seed73/inference_results.json",
    "oracle_slotverified (slotdecode/attnconcat_verified)":
        f"{SH}/slotdecode/attnconcat_verified/inference_results.json",
}


def rep_n(text, n=4):
    w = text.split()
    if len(w) < n:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


for name, path in FILES.items():
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        print(f"\n===== {name}: LOAD FAIL {e}")
        continue
    recs = d if isinstance(d, list) else list(d.values())
    print(f"\n===== {name}\n  n={len(recs)}  record_keys={sorted(recs[0].keys())}")
    agg = {g: dict(n=0, hedge_f0=0, hedge_attr=0, lens=[], reps=[],
                   ends_punct=0, has_ovl_clause=0,
                   counts={f: 0 for f in FEATS}) for g in ("clean", "mix")}
    samples = {"clean": [], "mix": []}
    for r in recs:
        stem = str(r.get("filename", "")).replace(".wav", "")
        gen = r.get("generated", "") or ""
        g = "clean" if "_s1clean" in stem else "mix"
        a = agg[g]
        a["n"] += 1
        found = set(c.feature for c in P.parse(gen))
        for f in FEATS:
            if f in found:
                a["counts"][f] += 1
        h = P.parse_hedges(gen)
        if "f0_mean" in h:
            a["hedge_f0"] += 1
            if h["f0_mean"]["attributed"]:
                a["hedge_attr"] += 1
        a["lens"].append(len(gen))
        a["reps"].append(rep_n(gen))
        if gen.rstrip().endswith((".", "!", "?")):
            a["ends_punct"] += 1
        if "overlap ratio" in gen.lower():
            a["has_ovl_clause"] += 1
        if len(samples[g]) < 3:
            samples[g].append((stem, gen))
    for g, a in agg.items():
        n = max(1, a["n"])
        print(f"  -- {g}: n={a['n']}"
              f"  f0_HEDGE={a['hedge_f0']} ({a['hedge_f0'] / n:.1%},"
              f" attributed {a['hedge_attr']})"
              f"  meanlen={statistics.mean(a['lens']):.0f}ch"
              f"  rep4={statistics.mean(a['reps']):.3f}"
              f"  ends_punct={a['ends_punct'] / n:.1%}"
              f"  overlap_clause={a['has_ovl_clause'] / n:.1%}")
        print("     claims: " + "  ".join(
            f"{f}={a['counts'][f]}({a['counts'][f] / n:.0%})" for f in FEATS))
    # Samples: audio-only gets full text (the arm under audit); others truncated.
    full = name.startswith("audioonly")
    for g in ("clean", "mix"):
        for stem, gen in samples[g][:2 if full else 1]:
            body = gen if full else gen[:600]
            print(f"\n  --- SAMPLE [{name}] {g} {stem}:\n  {body}")

print("\nDONE")
