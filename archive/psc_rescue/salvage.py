"""Phase 0 offline salvage (S1-S6) from val_samples/epoch_*.json — zero node-hours.

Replicas of src/eval/ckpt_selection.py rep_n/degeneration_stats/passes_degeneration_guard
(copied exactly; PSC cur_train is flat so we can't import the subpackage)."""
import json, glob, os, sys
import numpy as np
from scipy.stats import spearmanr

B = "/ocean/projects/cis260125p/shared"
ROBUST5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
EXTRA = ["f0_mean", "f0_sd"]
N_BOOT = 1000
RNG = np.random.default_rng(0)

# ---- exact replicas -------------------------------------------------------
def rep_n(text, n=4):
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return 1.0 - len(set(grams))/len(grams)

def nonascii_frac(text):
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127)/len(text)

def degeneration_stats(texts, n=4, rep_clip_thresh=0.5):
    texts = [t or "" for t in texts]
    reps = [rep_n(t, n) for t in texts]
    total = sum(len(t) for t in texts)
    na = sum(1 for t in texts for ch in t if ord(ch) > 127)
    n_na_clips = sum(1 for t in texts if any(ord(ch) > 127 for ch in t))
    n_hi = sum(1 for r in reps if r > rep_clip_thresh)
    return {"rep_n_mean": sum(reps)/len(reps), "rep_n_max": max(reps),
            "frac_clips_high_rep": n_hi/len(texts),
            "nonascii_frac": (na/total) if total else 0.0,
            "frac_clips_nonascii": n_na_clips/len(texts)}

def passes_guard(bleu, best_bleu, st, bleu_rel_floor=0.6, rep_n_thresh=0.95,
                 nonascii_thresh=0.05, clip_nonascii_thresh=0.15, clip_rep_thresh=0.50):
    if bleu is not None and best_bleu is not None and best_bleu > 0 and bleu < bleu_rel_floor*best_bleu:
        return False, f"BLEU_FLOOR bleu {bleu:.2f} < 0.6*best {best_bleu:.2f}"
    if st["frac_clips_high_rep"] > clip_rep_thresh:
        return False, f"FRAC_HIGH_REP {st['frac_clips_high_rep']:.3f} > 0.50"
    if st["rep_n_max"] > rep_n_thresh:
        return False, f"REP_N_MAX {st['rep_n_max']:.3f} > 0.95"
    if st["nonascii_frac"] > nonascii_thresh:
        return False, f"NONASCII {st['nonascii_frac']:.4f}"
    if st["frac_clips_nonascii"] > clip_nonascii_thresh:
        return False, f"FRAC_NONASCII_CLIPS {st['frac_clips_nonascii']:.3f}"
    return True, "clean"

# ---- load -----------------------------------------------------------------
def load(arm):
    out = {}
    for f in sorted(glob.glob(f"{B}/checkpoints/full/{arm}_grounded/val_samples/epoch_*.json")):
        ep = int(os.path.basename(f).split("_")[1].split(".")[0])
        out[ep] = {e["filename"]: e for e in json.load(open(f))}
    return out

M, C = load("m3b"), load("b1")
eps = sorted(set(M) & set(C))
print(f"epochs both arms: {eps}")

def pf(entry, feat):
    d = entry.get("per_feature_abs_error", {}) or {}
    v = d.get(feat)
    if not v:
        return None, None
    return v.get("pred"), v.get("gt")

def srcc(pairs):
    if len(pairs) < 10:
        return None
    p, g = zip(*pairs)
    r = spearmanr(p, g).correlation
    return None if np.isnan(r) else float(r)

def matched_stats(me, ce, feats, fns):
    """per-feature matched SRCC + per-arm coverage; returns dict."""
    out = {}
    for ft in feats:
        both, m_only, c_only = [], 0, 0
        for fn in fns:
            mp, mg = pf(me[fn], ft); cp, cg = pf(ce[fn], ft)
            if mp is not None: m_only += 1
            if cp is not None: c_only += 1
            if mp is not None and cp is not None:
                both.append((fn, mp, cp, mg))
        out[ft] = {"n_matched": len(both),
                   "cov_m3b": m_only/len(fns), "cov_b1": c_only/len(fns),
                   "srcc_m3b": srcc([(p, g) for _, p, _, g in both]),
                   "srcc_b1":  srcc([(p, g) for _, _, p, g in both])}
    return out

def robust_mean(d, feats, key):
    vs = [d[f][key] for f in feats if d.get(f, {}).get(key) is not None]
    return float(np.mean(vs)) if vs else None

try:
    import sacrebleu
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False

report = {"epochs": {}}
best_bleu = {"m3b": None, "b1": None}
best_metric = {"m3b": -1e9, "b1": -1e9}
print("\n================ PER-EPOCH SALVAGE ================")
for ep in eps:
    me, ce = M[ep], C[ep]
    fns = sorted(set(me) & set(ce))
    row = {"n_clips": len(fns)}

    # S5 degeneration + S1 guard replay per arm
    for arm, d in (("m3b", me), ("b1", ce)):
        gens = [d[fn].get("generated", "") or "" for fn in fns]
        refs = [d[fn].get("target", "") or "" for fn in fns]
        st = degeneration_stats(gens)
        bleu = sacrebleu.corpus_bleu(gens, [refs]).score if HAS_BLEU else None
        ok, reason = passes_guard(bleu, best_bleu[arm], st)
        # selector metric proxy: unmatched srcc_robust from the same JSON
        un = {}
        for ft in ROBUST5:
            pairs = [(p, g) for fn in fns for p, g in [pf(d[fn], ft)] if p is not None]
            un[ft] = srcc(pairs)
        sel = float(np.mean([v for v in un.values() if v is not None]))
        improved = sel > best_metric[arm]
        saved = improved and ok
        if saved:
            best_metric[arm] = sel
        if ok and bleu is not None:   # best_bleu updates only on clean epochs (mirror)
            best_bleu[arm] = bleu if best_bleu[arm] is None else max(best_bleu[arm], bleu)
        row[arm] = {"rep_mean": round(st["rep_n_mean"], 3), "rep_max": round(st["rep_n_max"], 3),
                    "frac_high_rep": round(st["frac_clips_high_rep"], 3),
                    "bleu": round(bleu, 2) if bleu else None,
                    "guard": reason, "sel_unmatched": round(sel, 4),
                    "would_save_best": saved}

    # S2/S3 matched re-score
    ms = matched_stats(me, ce, ROBUST5 + EXTRA, fns)
    row["matched"] = {ft: {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in ms[ft].items()} for ft in ms}
    r_m = robust_mean(ms, ROBUST5, "srcc_m3b"); r_c = robust_mean(ms, ROBUST5, "srcc_b1")
    ns = [f for f in ROBUST5 if f != "snr"]
    rn_m = robust_mean(ms, ns, "srcc_m3b"); rn_c = robust_mean(ms, ns, "srcc_b1")
    row["matched_robust"] = {"m3b": round(r_m, 4), "b1": round(r_c, 4), "delta": round(r_m-r_c, 4)}
    row["matched_nosnr"] = {"m3b": round(rn_m, 4), "b1": round(rn_c, 4), "delta": round(rn_m-rn_c, 4)}

    # S4 paired bootstrap on the matched delta
    deltas, deltas_ns = [], []
    fn_arr = np.array(fns)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(fns), len(fns))
        bs_fns = fn_arr[idx]
        bd = matched_stats(me, ce, ROBUST5, list(bs_fns))
        bm = robust_mean(bd, ROBUST5, "srcc_m3b"); bc = robust_mean(bd, ROBUST5, "srcc_b1")
        if bm is not None and bc is not None:
            deltas.append(bm-bc)
        bmn = robust_mean(bd, ns, "srcc_m3b"); bcn = robust_mean(bd, ns, "srcc_b1")
        if bmn is not None and bcn is not None:
            deltas_ns.append(bmn-bcn)
    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        row["delta_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    if deltas_ns:
        lo, hi = np.percentile(deltas_ns, [2.5, 97.5])
        row["delta_nosnr_ci95"] = [round(float(lo), 4), round(float(hi), 4)]

    # non-degenerate-only rescore (both arms rep<0.5)
    nd = [fn for fn in fns
          if rep_n(me[fn].get("generated", "") or "") <= 0.5
          and rep_n(ce[fn].get("generated", "") or "") <= 0.5]
    nds = matched_stats(me, ce, ROBUST5, nd)
    row["nondegen"] = {"n": len(nd),
                       "m3b": robust_mean(nds, ROBUST5, "srcc_m3b"),
                       "b1": robust_mean(nds, ROBUST5, "srcc_b1")}

    # S5 loop-rate difference bootstrap
    mrep = np.array([rep_n(me[fn].get("generated", "") or "") > 0.5 for fn in fns])
    crep = np.array([rep_n(ce[fn].get("generated", "") or "") > 0.5 for fn in fns])
    diffs = [float(np.mean(mrep[RNG.integers(0, len(fns), len(fns))]) -
                   np.mean(crep[RNG.integers(0, len(fns), len(fns))])) for _ in range(N_BOOT)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    row["looprate"] = {"m3b": round(float(mrep.mean()), 3), "b1": round(float(crep.mean()), 3),
                       "diff_ci95": [round(float(lo), 3), round(float(hi), 3)]}

    report["epochs"][ep] = row
    print(f"\n--- epoch {ep} (n={len(fns)}) ---")
    for arm in ("m3b", "b1"):
        r = row[arm]
        print(f"  {arm}: sel={r['sel_unmatched']} bleu={r['bleu']} rep_max={r['rep_max']} "
              f"frac_hi={r['frac_high_rep']} guard={r['guard']} save_best={r['would_save_best']}")
    print(f"  MATCHED robust: m3b={row['matched_robust']['m3b']} b1={row['matched_robust']['b1']} "
          f"delta={row['matched_robust']['delta']} CI95={row.get('delta_ci95')}")
    print(f"  MATCHED no-snr: delta={row['matched_nosnr']['delta']} CI95={row.get('delta_nosnr_ci95')}")
    print(f"  per-feat matched: " + " ".join(
        f"{ft}:m={row['matched'][ft]['srcc_m3b']}/b={row['matched'][ft]['srcc_b1']}"
        f"(cov {row['matched'][ft]['cov_m3b']:.2f}/{row['matched'][ft]['cov_b1']:.2f})"
        for ft in ROBUST5))
    print(f"  f0 matched: " + " ".join(
        f"{ft}:m={row['matched'][ft]['srcc_m3b']}/b={row['matched'][ft]['srcc_b1']}" for ft in EXTRA))
    print(f"  non-degen(n={row['nondegen']['n']}): m3b={row['nondegen']['m3b']} b1={row['nondegen']['b1']}")
    print(f"  looprate: m3b={row['looprate']['m3b']} b1={row['looprate']['b1']} "
          f"diffCI={row['looprate']['diff_ci95']}")

# S6 epoch parity of last.pt
print("\n================ S6 last.pt epoch parity ================")
try:
    import torch
    for arm in ("m3b", "b1"):
        p = f"{B}/checkpoints/full/{arm}_grounded/last.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        keys = [k for k in ck.keys() if not hasattr(ck[k], "shape")]
        print(f"  {arm}: epoch={ck.get('epoch')} keys={sorted(keys)[:12]} "
              f"has_best_val_bleu={'best_val_bleu' in ck} wandb={ck.get('wandb_run_id')}")
        del ck
except Exception as e:
    print("  S6 failed:", type(e).__name__, e)

out = f"{B}/salvage_report_2026-07-16.json"
json.dump(report, open(out, "w"), indent=1)
print(f"\nreport saved: {out}")
