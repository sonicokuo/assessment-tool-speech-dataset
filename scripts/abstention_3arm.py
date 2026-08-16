#!/usr/bin/env python3
"""abstention_3arm.py — the matched comparison that decides whether contribution III is OURS.

WHY THIS EXISTS
`residual_error_head.py` showed a post-hoc GBM error-predictor beating our LEARNED sigma head on
10 of 11 features (mean within-condition ranking 0.214 -> 0.300), which closes the per-claim gap
the plan called "not closable at current sigma quality". That is good news about the SYSTEM and
bad news about the STORY: if the ranking ability comes from a post-hoc predictor rather than the
sigma head, then a reviewer immediately asks the obvious question —

    "You compared your SIGMA to the ridge's POST-HOC predictor. Give the ridge the same
     post-hoc predictor you just gave yourself. Now who wins?"

That comparison has never been run. Both prior scripts are half of it:
  uq_ridge_baseline.py   ours=SIGMA        vs ridge=GBM     <- machinery MISMATCHED, flatters ridge
  residual_error_head.py ours=SIGMA        vs ours=GBM      <- no baseline at all

THE THREE ARMS (identical clips, identical folds, identical GBM hyperparameters)
  A  ours  + sigma head        the learned confidence, as shipped
  B  ours  + post-hoc GBM      same predictions, confidence upgraded
  C  ridge + post-hoc GBM      the hostile baseline, given EXACTLY B's machinery

B vs C is the only comparison that isolates OUR REPRESENTATION, because the confidence
machinery is held fixed across them. A vs B measures what the sigma head is worth.

READING (pre-registered, so the result cannot be rationalised after the fact):
  B > C on the ill-posed panel  -> the representation carries the observability signal;
                                   contribution III holds, restated as "our representation
                                   supports better error ranking", with the sigma head demoted
                                   from mechanism to ablation row.
  B ~ C                         -> abstention quality is a property of frozen WavLM, not of our
                                   model. Contribution III is NOT ours. Report it as a negative
                                   and let the ceiling + hedging carry the paper.
  A > B                         -> the sigma head is genuinely doing the work; keep it as the
                                   mechanism and report B/C as robustness.

TWO METRICS, BOTH REPORTED, EACH WITH ITS LOSS STATED
  within-condition SRCC : does confidence rank errors AMONG comparable clips? This is the
                          PER-CLAIM question. Condition-level separation is excluded by
                          construction, which is the whole point — sigma scored rho 1.000
                          across the overlap ladder but 0.05-0.26 within a condition, so the
                          across-condition number was never evidence of per-claim ability.
  E-AURC                : selective risk with the perfect-ordering AURC subtracted
                          (Geifman, Uziel & El-Yaniv, ICLR 2019, arXiv:1805.08206), so a more
                          ACCURATE arm is not credited for accuracy it already had.

LEAK GUARDS (each one is a defect that actually fired in this project)
  * the error predictor sees pooled features + the point estimate, NEVER y. Given y and yhat it
    computes |yhat - y| exactly and scores ~1.0 by construction.
  * cells whose GT is constant within a condition are VOIDED: there err = |yhat - const| is a
    deterministic function of an input. This scored 0.965 on overlap_ratio in oracle_error_ceiling.
  * every arm is cross-validated on the SAME folds, so no arm sees more data than another.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import ILL_POSED_UNDER_OVERLAP_FEATURES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from eval.results_io import write_result  # noqa: E402

# Panels derived from the single sources of truth, not retyped. These literals were
# byte-identical to canonical when this was written (verified 2026-08-16) — the point is that a
# future edit to feature_set.py can no longer silently disagree with three separate copies.
# Four defects in this repo came from exactly that: hnr_db/shimmer_pct in the target builder, a
# short-name GT lookup, slot_frames() ordering, and FEATS in score_matched_test.py which
# silently reported 7 of 11 features.
ROBUST5 = list(HEADLINE_FEATURES)
ILLPOSED = sorted(ILL_POSED_UNDER_OVERLAP_FEATURES)



def srcc(a, b) -> float:
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(x.size, float); sx = x[o]; i = 0
        while i < x.size:
            j = i + 1
            while j < x.size and sx[j] == sx[i]:
                j += 1
            r[o[i:j]] = 0.5 * (i + j - 1); i = j
        return r
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 30 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def eaurc(conf: np.ndarray, err: np.ndarray) -> float:
    """E-AURC: AURC minus the AURC a PERFECT ordering would achieve on the same errors."""
    ok = np.isfinite(conf) & np.isfinite(err)
    conf, err = conf[ok], err[ok]
    if conf.size < 30:
        return float("nan")
    def auc(e):
        return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean())
    return auc(err[np.argsort(conf)]) - auc(np.sort(err))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux_sigma", required=True, help="dump_aux_sigma.py output (ours)")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--pt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_pca", type=int, default=256,
                    help="PCA dims for the error predictor (cost control; "
                         "applied identically to arms B and C)")
    a = ap.parse_args()

    import torch
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    n_pca = a.n_pca

    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    cols = [f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0]
            for f in SUPERVISED_FEATURES]

    gt = {}
    for r in csv.DictReader(open(a.features_csv)):
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        row = {}
        for nm, cl in zip(names, cols):
            try:
                row[nm] = float(r.get(cl, r.get(nm, "")))
            except (TypeError, ValueError):
                pass
        gt[stem] = row

    keep = []
    for r in json.load(open(a.aux_sigma)):
        if not r.get("aux_mean"):
            continue
        stem = r["filename"]
        g = gt.get(stem) or gt.get(stem[:-8] if stem.endswith("_s1clean") else stem) or {}
        if g:
            keep.append({"stem": stem, "aux": np.asarray(r["aux_mean"], float),
                         "sigma": np.asarray(r.get("sigma") or [], float), "gt": g})
    print(f"clips: {len(keep)}", flush=True)

    X = []
    for i, r in enumerate(keep):
        af = torch.load(os.path.join(a.pt_dir, r["stem"] + ".pt"),
                        map_location="cpu", weights_only=False)["audio_features"].float()
        X.append(np.concatenate([af.mean(0).numpy(), af.std(0).numpy()]).astype(np.float32))
        # heartbeat: the dispatcher treats a stale log as a dead job and relaunches it, which
        # spawned a duplicate writer on the same --out path twice. Loading is minutes of silence.
        if (i + 1) % 250 == 0:
            print(f"  loaded {i + 1}/{len(keep)}", flush=True)
    X = np.stack(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    is_clean = np.array([r["stem"].endswith("_s1clean") for r in keep])
    print(f"pooled  (clean {int(is_clean.sum())} / mix {int((~is_clean).sum())})", flush=True)

    def posthoc(Xf, yhat, err, kf):
        """Cross-validated |error| predictor. Sees the estimate and the features, NEVER y.

        PCA is a COST control, not a modelling choice. Boosting over the raw 2048-dim pooled
        encoder costs ~30 min per feature (10 fits each scanning 2049 columns x 200 iters),
        which is ~5h for the panel and leaves no slack on a single allocation. Projecting to
        `n_pca` components first is ~8x cheaper.

        IT CANNOT BIAS B vs C: both arms are projected with the SAME basis and the SAME
        hyperparameters, and the projection is UNSUPERVISED (fit on training-fold X only, never
        on y and never on the held-out fold). If the reduction costs any accuracy it costs both
        arms alike, which SHRINKS |B-C| — conservative for the claim we would like to make.
        """
        c = np.zeros_like(err)
        for k, (tr, te) in enumerate(kf.split(Xf)):
            # Per-FOLD heartbeat, not just per-feature. A single feature's fits take ~12 min, so
            # a per-feature heartbeat still leaves the log silent long enough for the dispatcher
            # to call the job dead and relaunch it — which cost ~15 min of work once already.
            print(f"      fold {k + 1}/{kf.get_n_splits()}", flush=True)
            pca = PCA(n_components=min(n_pca, Xf[tr].shape[0], Xf.shape[1]),
                      svd_solver="randomized", random_state=0).fit(Xf[tr])
            # the point estimate is appended AFTER projection so it is never diluted by it
            Etr = np.column_stack([pca.transform(Xf[tr]), yhat[tr]])
            Ete = np.column_stack([pca.transform(Xf[te]), yhat[te]])
            c[te] = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, max_depth=6,
                random_state=0).fit(Etr, err[tr]).predict(Ete)
        return c

    def within(conf, err, cl):
        """Mean SRCC computed SEPARATELY inside each condition, then averaged.

        Conditions whose GT is constant are already voided upstream; here we additionally skip a
        condition with too few clips to rank.
        """
        vals = [srcc(conf[m], err[m]) for m in (cl, ~cl) if m.sum() > 50]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    hdr = (f"{'feature':<15}{'A sig':>8}{'B ours+GBM':>12}{'C ridge+GBM':>13}{'B-C':>8}"
           f"{'|':>3}{'A eauc':>9}{'B eauc':>9}{'C eauc':>9}{'winner':>10}")
    print("\n" + hdr); print("-" * len(hdr))
    summary, kf = {}, KFold(n_splits=a.folds, shuffle=True, random_state=0)
    # Resume: reload any completed features so a relaunch continues instead of restarting. Folds
    # are seeded (random_state=0) so resumed and fresh rows are computed identically.
    if os.path.exists(a.out + ".partial"):
        try:
            summary = json.load(open(a.out + ".partial"))
            if summary:
                print(f"[resume] {len(summary)} features already done: {sorted(summary)}", flush=True)
        except Exception:                                        # noqa: BLE001
            summary = {}

    for j, nm in enumerate(names):
        if nm in summary:
            continue                                             # already computed, see [resume]
        ok = np.array([nm in r["gt"] and np.isfinite(r["gt"][nm]) for r in keep])
        if ok.sum() < 300:
            continue
        y = np.array([r["gt"][nm] for r, k in zip(keep, ok) if k])
        yh_ours = np.array([r["aux"][j] for r, k in zip(keep, ok) if k])
        sg = np.array([r["sigma"][j] if r["sigma"].size > j else np.nan
                       for r, k in zip(keep, ok) if k])
        Xf, cl = X[ok], is_clean[ok]
        print(f"  [{j + 1}/{len(names)}] fitting {nm} (n={int(ok.sum())}) ...",
              flush=True)

        # VOID a feature whose GT is constant in EITHER condition: there err = |yhat - const| is
        # a deterministic function of the predictor's own input, and every arm scores spuriously.
        if ((cl.sum() > 50 and float(np.std(y[cl])) < 1e-6)
                or ((~cl).sum() > 50 and float(np.std(y[~cl])) < 1e-6)):
            print(f"{nm:<15}  VOID — GT constant within a condition (err = f(input))")
            summary[nm] = {"voided": "constant GT within condition"}
            continue

        # ⚠️ THE BASELINE MUST BE TUNED OR THE COMPARISON IS RIGGED. An untuned alpha=10 ridge is
        # NOT the baseline of record — the project's own layer sweep found a TUNED layer-7 ridge
        # beats our aux head (margin -0.0134), so handicapping arm C here would manufacture a
        # win for arm B. alpha is selected per feature on an INNER split of the training fold
        # only, so the test fold stays untouched and no arm sees more data than another.
        yh_ridge = np.zeros_like(y)
        for tr, te in kf.split(Xf):
            n_in = int(0.8 * tr.size)
            fit_i, val_i = tr[:n_in], tr[n_in:]
            best_a, best_e = 10.0, np.inf
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                e = float(np.mean(np.abs(
                    Ridge(alpha=alpha).fit(Xf[fit_i], y[fit_i]).predict(Xf[val_i]) - y[val_i])))
                if e < best_e:
                    best_a, best_e = alpha, e
            yh_ridge[te] = Ridge(alpha=best_a).fit(Xf[tr], y[tr]).predict(Xf[te])

        err_o, err_r = np.abs(yh_ours - y), np.abs(yh_ridge - y)
        confB = posthoc(Xf, yh_ours, err_o, kf)
        confC = posthoc(Xf, yh_ridge, err_r, kf)

        wA = within(sg, err_o, cl) if np.isfinite(sg).any() else float("nan")
        wB, wC = within(confB, err_o, cl), within(confC, err_r, cl)
        # E-AURC on each arm's OWN error, normalised by that arm's own mean error, so the
        # statistic is about ORDERING alone and the more accurate arm gets no free credit.
        eA = eaurc(sg, err_o / (err_o.mean() or 1.0)) if np.isfinite(sg).any() else float("nan")
        eB = eaurc(confB, err_o / (err_o.mean() or 1.0))
        eC = eaurc(confC, err_r / (err_r.mean() or 1.0))
        win = "OURS" if np.isfinite(wB) and np.isfinite(wC) and wB > wC else "ridge+GBM"

        print(f"{nm:<15}{wA:8.3f}{wB:12.3f}{wC:13.3f}{wB - wC:+8.3f}{'|':>3}"
              f"{eA:9.3f}{eB:9.3f}{eC:9.3f}{win:>10}")
        summary[nm] = {"within_A_sigma": wA, "within_B_ours_gbm": wB, "within_C_ridge_gbm": wC,
                       "within_B_minus_C": wB - wC, "eaurc_A": eA, "eaurc_B": eB, "eaurc_C": eC,
                       "winner": win}
        # Dump AFTER recording, so the partial always reflects completed work. Each feature costs
        # ~12 min; without this a restart discards everything done so far, which happened twice.
        json.dump(summary, open(a.out + ".partial", "w"), indent=2)

    for panel, feats in (("ROBUST5", ROBUST5), ("ILL-POSED", ILLPOSED)):
        have = [f for f in feats if f in summary and "within_B_ours_gbm" in summary[f]]
        if not have:
            continue
        mA = np.nanmean([summary[f]["within_A_sigma"] for f in have])
        mB = np.nanmean([summary[f]["within_B_ours_gbm"] for f in have])
        mC = np.nanmean([summary[f]["within_C_ridge_gbm"] for f in have])
        print(f"\n{panel:<10} within-condition:  A sigma {mA:.4f} | B ours+GBM {mB:.4f} | "
              f"C ridge+GBM {mC:.4f}   ->  B-C = {mB - mC:+.4f} "
              f"({'OURS' if mB > mC else 'RIDGE'})")
        summary[f"_panel_{panel}"] = {"A_sigma": float(mA), "B_ours_gbm": float(mB),
                                      "C_ridge_gbm": float(mC), "B_minus_C": float(mB - mC)}

    write_result(summary, out_path=a.out, producer="abstention_3arm",
                 checkpoint=a.aux_sigma, n=len(keep))
    print(f"\nwrote {a.out}")
    print("READ: B vs C is the ONLY arm pair that isolates our representation — same clips, same")
    print("folds, same GBM. If B ~ C, abstention quality belongs to frozen WavLM and not to us;")
    print("say so. A vs B says what the LEARNED sigma head is worth over a post-hoc predictor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
