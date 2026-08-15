"""Draw the CORRECTED per-feature evidence maps.

The old figure plotted |contribution|, which is dominated by the broadcast clip-level
value (DC) and therefore looks flat for every feature no matter what the model learned.
This one plots the DEVIATION map (c - mean_t c), which is where the clip-specific signal
lives, and shades each clip's own overlap regions so the reader can check the sign
directly: for a quantity that is UNRECOVERABLE under overlap (f0, jitter, shimmer, hnr),
the deviation must DIP inside the shaded spans.

Usage:
    python scripts/plot_attribution_maps.py attribution_fw2.npz out.png [corrected_fw2.json]
"""
from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

TOKEN_MS = 160  # one prefix token covers 160 ms at 6.25 tok/s

# features whose deviation should be NEGATIVE inside overlap (ill-posed there)
ILL_POSED = {"f0_mean", "f0_sd", "jitter", "shimmer", "hnr", "pause_rate", "pause_count"}


def _valid(c: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(c) & np.isfinite(o)
    return c[m], o[m]


def main(npz_path: str, out_path: str, json_path: str | None = None) -> None:
    d = np.load(npz_path, allow_pickle=True)
    C, OV = d["contrib"], d["overlap"]
    FEATS = [str(x) for x in d["features"]]
    scores = json.load(open(json_path)) if json_path else None

    # Pick clips CLOSEST TO 50% overlap, not the most overlapped. A 94%-overlapped clip
    # has no clean region to contrast against, so the map looks uninformative however
    # good it is. Maximum contrast is what makes the sign visible.
    frac = np.array([np.nanmean(o > 0.5) for o in OV])
    ok = np.where(np.isfinite(frac) & (frac > 0.05) & (frac < 0.95))[0]
    if len(ok) < 3:
        ok = np.argsort(-frac)[:3]
    picks = ok[np.argsort(np.abs(frac[ok] - 0.5))][:3]

    fig = plt.figure(figsize=(14, 11.5))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.25, 1.0], hspace=0.42, wspace=0.28)

    # ---- ROW 1: the money shot. f0 deviation vs overlap, one panel per clip ----
    fi = FEATS.index("f0_mean") if "f0_mean" in FEATS else 0
    for k, ci in enumerate(picks):
        ax = fig.add_subplot(gs[0, k])
        c, o = _valid(C[ci, fi], OV[ci])
        dev = c - c.mean()
        t = np.arange(len(dev)) * TOKEN_MS / 1000.0
        # shade this clip's own overlapped spans
        inside = o > 0.5
        edges = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
        labelled = False
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color="0.82", zorder=0,
                       label=None if labelled else "overlapped")
            labelled = True
        ax.axhline(0, color="0.4", lw=0.8, zorder=1)
        ax.plot(t, dev, color="#c0392b", lw=1.9, zorder=3)
        r = np.corrcoef(dev, o)[0, 1] if o.std() > 1e-9 and dev.std() > 1e-9 else np.nan
        ax.set_title(f"clip {ci} — {100*np.nanmean(o>0.5):.0f}% overlapped   (r = {r:+.2f})",
                     fontsize=10)
        ax.set_xlabel("time (s)", fontsize=9)
        if k == 0:
            ax.set_ylabel("f0_mean deviation\n(c − mean)", fontsize=9)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.tick_params(labelsize=8)

    # ---- ROW 2: all features, deviation heatmap, one clip ----
    ci = picks[0]
    ax = fig.add_subplot(gs[1, :])
    good = np.isfinite(C[ci]).all(0) & np.isfinite(OV[ci])
    M = C[ci][:, good]
    D = M - M.mean(axis=1, keepdims=True)
    D = D / (np.abs(D).max(axis=1, keepdims=True) + 1e-12)
    lim = 1.0
    im = ax.imshow(D, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest",
                   extent=[0, D.shape[1] * TOKEN_MS / 1000.0, len(FEATS) - 0.5, -0.5])
    o = OV[ci][good]
    inside = o > 0.5
    edges = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
    tt = np.arange(len(o)) * TOKEN_MS / 1000.0
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        ax.axvspan(tt[s], tt[min(e, len(tt) - 1)], facecolor="none", edgecolor="k",
                   lw=2.0, ls="--", zorder=5)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels([f + ("  *" if f in ILL_POSED else "") for f in FEATS], fontsize=9)
    ax.set_xlabel("time (s)   —   dashed boxes = this clip's overlapped spans", fontsize=9)
    ax.set_title("Deviation maps, all features, one clip.  "
                 "* = ill-posed under overlap, so BLUE inside the boxes is correct.",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01,
                 label="deviation (row-normalised)")

    # ---- ROW 3: the metric correction + ceiling ----
    if scores:
        ax = fig.add_subplot(gs[2, :2])
        names = [f for f in FEATS]
        old = [scores[f]["spec_oneside_BUGGY"] for f in names]
        new = [scores[f]["spec_twoside"] for f in names]
        ceil = scores["__ORACLE__"]["spec_twoside"]
        x = np.arange(len(names))
        ax.bar(x - 0.2, old, 0.4, color="#b0b0b0", label="one-sided (BUGGY — reported before)")
        ax.bar(x + 0.2, new, 0.4, color="#2c7fb8", label="two-sided (corrected)")
        ax.axhline(ceil, color="#e67e22", ls="--", lw=1.8,
                   label=f"oracle ceiling = {ceil:.3f}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
        ax.set_ylabel("clip-specificity", fontsize=9)
        ax.set_title("The sign bug: anti-aligned (physically correct) maps scored NEGATIVE",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[2, 2])
        r = [scores[f]["dev_corr"] for f in names]
        col = ["#c0392b" if v < 0 else "#27ae60" for v in r]
        ax.barh(np.arange(len(names)), r, color=col)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("corr(deviation, overlap)", fontsize=9)
        ax.set_title("Signed: negative = avoids\noverlap (correct for ill-posed)", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("Per-feature evidence maps — corrected metrics", fontsize=13, y=0.985)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
