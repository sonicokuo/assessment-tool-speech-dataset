#!/usr/bin/env python3
"""score_attribution.py — the attribution evaluation (items 0.9, 0.10, 0.17).

Scores a candidate per-frame map against the EXACT oracle contribution map, and
reports it against the two denominators an attribution number is meaningless
without:

  1. NULLS  — what a map that knows nothing still scores
  2. CEILING — what a PERFECT map scores after pooling to the token grid

NULLS
  N1 random          uniform noise                     -> "anything above zero"
  N2 flat            constant map                      -> "featureless does as well"
  N3 permutation     ANOTHER clip's oracle map         -> "population-generic, not clip-specific"
  N4 time-shuffled   this clip's map, time-permuted    -> "marginal stats, not placement"
  N5 overlap map     this clip's own overlap layout    -> "purchasable from the overlap INPUT alone"

N5 is the decisive one. `overlap_info[:,0]` is a model INPUT (verified rho 0.9999
with the GT column), so any map that merely echoes it reproduces every
correlational signature we previously reported. N5 prices that echo exactly.

U-1 makes N5 sharp rather than approximate: the f0 map support is
`voiced AND NOT overlap`, which factorises into an input-derivable term (NOT
overlap) and an audio-only term (voiced). N5 measures the first, so the margin
ABOVE N5 is the genuinely audio-derived part.

CLEAN-CLIP PANEL (0.10): on `_s1clean` twins the overlap set is EMPTY, so the map
collapses to the voiced mask -- entirely audio-derived -- and N5 is VACUOUS. Any
alignment there is evidence-tracking with the echo confound STRUCTURALLY absent.
This is the sharpest condition and the old evaluation discarded these clips.

NOISE LADDER (0.17): score `(1-a)*phi + a*shuffle(phi)` for a in [0,1]. This
interpolates from the perfect map (a=0) to exactly N4 (a=1), so the endpoints are
the ceiling and a principled null. It validates the metric (must be monotone) and
makes scores INTERPRETABLE: if the model scores what a=0.6 scores, its map is
~40% signal.

Usage:
  python scripts/score_attribution.py --oracle $SH/oracle_maps_test.npz \
      [--model_maps saliency.npz] [--feature f0_mean] [--boot 2000]
"""
from __future__ import annotations

import argparse

import numpy as np

TIME_STEP = 0.01           # oracle maps live on a 100 Hz grid
TOKEN_RATE = 6.25          # arm of record


def _avg_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = _avg_rank(a) - _avg_rank(a).mean(), _avg_rank(b) - _avg_rank(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def pool(phi: np.ndarray, tgt_rate: float = TOKEN_RATE) -> np.ndarray:
    """Block-sum onto the token grid, DISCARDING any incomplete final block.

    Zero-PADDING the tail instead would give both series an artificially small
    final element, correlating them for free: it put N1-random at +0.0487 with a
    CI excluding zero, which a random map cannot legitimately achieve. Truncation
    costs <1 token and removes the artifact.
    """
    f = max(1, int(round((1.0 / TIME_STEP) / tgt_rate)))
    n = (phi.size // f) * f
    if n == 0:
        return np.zeros(0, dtype=float)
    return phi[:n].reshape(-1, f).sum(axis=1)


def align_to_oracle(x: np.ndarray, offset_frames: int = 2) -> np.ndarray:
    """D7: Praat's pitch grid does not start at t=0.

    `p.xs()[0]` is about half the analysis window (~20 ms at a 75 Hz floor), so the oracle
    map's index 0 corresponds to t~0.02 s while a model map's index 0 is t=0. Ceiling, N1,
    N3 and N4 are computed with the SAME grid on both sides and are unaffected; only the
    MODEL map and N5 carry the offset, so the model is systematically penalised ~2 frames
    per 16-frame block against the ceiling it is divided by. Shifting the candidate removes
    a one-directional bias against the thing we are measuring.
    """
    if offset_frames <= 0:
        return x
    return np.concatenate([x[offset_frames:], np.zeros(offset_frames, dtype=x.dtype)])


def degrade(x: np.ndarray, tgt_rate: float = TOKEN_RATE) -> np.ndarray:
    """Apply the model's OWN resolution handicap: pool to the token grid, re-expand.

    Every null and the ceiling must carry the same handicap as the model map, or
    they are not comparable. Scoring nulls at 100 Hz while scoring the ceiling at
    token rate made |N5| (0.6967) EXCEED its own ceiling (0.6679) for f0_mean —
    an impossibility that revealed the mismatch.
    """
    f = max(1, int(round((1.0 / TIME_STEP) / tgt_rate)))
    p = pool(x, tgt_rate)
    return np.repeat(p / f, f)


def mass_concentration(cand: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of |cand| mass landing inside ref's support, MINUS the support's
    own share of the timeline (item 0.30).

    Spearman is the wrong instrument when the reference is two-valued: the
    f0_mean oracle map is BINARY, and its noise ladder PLATEAUS (a=0.1->0.3 moved
    0.6745->0.6715), so rank correlation cannot separate 10% from 30% corruption.
    Mass concentration asks the question a support mask actually poses -- does the
    candidate put its weight where the evidence is -- and the baseline subtraction
    makes 0.0 mean chance for any support size.
    """
    n = min(cand.size, ref.size)
    if n < 3:
        return float("nan")
    c, r = np.abs(cand[:n]), np.abs(ref[:n])
    tot = c.sum()
    sup = r > 0
    if tot <= 0 or not sup.any() or sup.all():
        return float("nan")
    return float(c[sup].sum() / tot - sup.mean())


def boot_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 0):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--model_maps", default="")
    ap.add_argument("--feature", default="f0_mean")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--rate", type=float, default=TOKEN_RATE)
    a = ap.parse_args()

    z = np.load(a.oracle, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    key = lambda f, n: f"{f}/{n}"                                   # noqa: E731
    have = [n for n in names if key(a.feature, n) in z]
    if not have:
        print(f"no maps for feature {a.feature}")
        return 1

    # clean twins carry the _s1clean suffix -> the panel where N5 is vacuous
    clean = [n for n in have if n.endswith("_s1clean")]
    mixed = [n for n in have if not n.endswith("_s1clean")]
    rng = np.random.default_rng(0)

    def panel(subset, label):
        if len(subset) < 5:
            print(f"\n--- {label}: n={len(subset)} too few, skipped")
            return
        ceil, n1, n2, n3, n4, n5 = [], [], [], [], [], []
        for i, n in enumerate(subset):
            phi = np.asarray(z[key(a.feature, n)], dtype=float)
            if phi.size < 8 or np.allclose(phi, phi[0]):
                continue
            # EVERY series carries the model's OWN resolution handicap and is scored
            # against the NATIVE-resolution oracle. Same footing throughout — scoring
            # nulls at token rate while scoring the ceiling at 100 Hz made |N5| exceed
            # its own ceiling for f0_mean, which is impossible.
            ref = phi
            ceil.append(spearman(degrade(phi, a.rate), ref))
            n1.append(spearman(degrade(rng.random(phi.size), a.rate), ref))
            n2.append(float("nan"))                                  # constant map: no ranks
            other = np.asarray(z[key(a.feature, subset[(i + 7) % len(subset)])], dtype=float)
            n3.append(spearman(degrade(other, a.rate)[: ref.size], ref))
            n4.append(spearman(degrade(rng.permutation(phi), a.rate), ref))
            ok = key("overlap_ratio", n)
            if ok in z:
                ov = np.asarray(z[ok], dtype=float)
                n5.append(spearman(degrade(ov, a.rate), ref) if not np.allclose(ov, 0) else float("nan"))

        print(f"\n--- {label}  (n={len(subset)}, feature={a.feature}, rate={a.rate} Hz)")
        for nm, v in (("CEILING (perfect map)", ceil), ("N1 random", n1), ("N3 permutation", n3),
                      ("N4 time-shuffled", n4), ("N5 overlap map", n5)):
            m, lo, hi = boot_ci(np.array(v, dtype=float), a.boot)
            n_ok = int(np.isfinite(np.array(v, dtype=float)).sum())
            if n_ok == 0:
                print(f"    {nm:<24} VACUOUS (undefined on this panel)")
            else:
                print(f"    {nm:<24} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   n={n_ok}")
        print("    N2 flat                  VACUOUS (constant map has no ranks)")

    panel(mixed, "MIXTURES")
    panel(clean, "CLEAN TWINS  <- N5 vacuous here; confound structurally absent")

    # ---------------- noise ladder (0.17): metric self-validation ----------------
    print(f"\n=== NOISE LADDER (0.17) — (1-a)*phi + a*shuffle(phi), pooled to {a.rate} Hz ===")
    print("    a=0 is the ceiling, a=1 is exactly N4. Must be MONOTONE decreasing.")
    sub = have[: min(300, len(have))]
    prev, mono = None, True
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        sc = []
        for n in sub:
            phi = np.asarray(z[key(a.feature, n)], dtype=float)
            if phi.size < 8 or np.allclose(phi, phi[0]):
                continue
            noisy = (1 - alpha) * phi + alpha * rng.permutation(phi)
            sc.append(spearman(degrade(noisy, a.rate), phi))
        m, lo, hi = boot_ci(np.array(sc, dtype=float), 500)
        flag = ""
        if prev is not None and np.isfinite(m) and np.isfinite(prev) and m > prev + 0.02:
            flag, mono = "  <-- NON-MONOTONE", False
        print(f"    a={alpha:<4} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")
        prev = m
    print(f"    ladder monotone: {'YES — metric behaves' if mono else 'NO — METRIC IS BROKEN, fix before use'}")

    if a.model_maps:
        mz = np.load(a.model_maps, allow_pickle=True)
        sc = []
        for n in have:
            mk = f"{a.feature}/{n}"
            if mk not in mz:
                continue
            ref = np.asarray(z[key(a.feature, n)], dtype=float)
            mm = np.asarray(mz[mk], dtype=float)
            sc.append(spearman(degrade(mm, a.rate)[: ref.size], ref))
        m, lo, hi = boot_ci(np.array(sc, dtype=float), a.boot)
        print(f"\n=== MODEL MAP: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={len(sc)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
