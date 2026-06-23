"""uq_bakeoff.py — the UNCERTAINTY-QUANTIFICATION BAKE-OFF harness (kill-fast gate).

THE QUESTION
------------
Does a CHEAP UQ head (MDN / MC-dropout / deep-ensemble; src/uq_heads.py) beat the
INCUMBENT heteroscedastic-sigma head (src/reliability_head.py, the predicted log-var)
at predicting per-frame SNR-map ERROR, measured by risk-coverage AURC? If a cheap head
already matches the gain a diffusion model could plausibly buy, diffusion is unjustified
and the cheap head IS the paper. This module is the gate that decides GO / NO-GO.

WHAT IT SCORES
--------------
Given, for one method, per-frame arrays {prediction, uncertainty, true_snr, mask} it
computes:

  1. RISK-COVERAGE AURC with CONTINUOUS RISK = |pred - true| (NOT a 0/1 band cut).
     Frames are ranked by uncertainty; we ABSTAIN on the HIGHEST-uncertainty frames
     first and sweep coverage from 1 (keep all) down to 0, accumulating the running
     mean |error| of the retained frames; the area under that risk-vs-coverage curve is
     the AURC. A good uncertainty puts its big values on the big-error frames, so the
     retained set's mean error falls fast -> low AURC. We reuse metrics_calibrated.aurc,
     which already integrates ARBITRARY per-item losses (we feed the continuous |error|,
     not a binary indicator) ranked by DESCENDING confidence = ASCENDING uncertainty
     (confidence := -uncertainty).
  2. SPEARMAN(uncertainty, |error|) — a tolerance-free check that the uncertainty RANKS
     the errors. Reuses metrics_calibrated.spearman.
  3. CALIBRATION — for the variance/sigma methods, treat the per-frame predictive law as
     Gaussian(pred, uncertainty) and report EMPIRICAL coverage at nominal 50/80/90/95
     (fraction of frames with |error| <= z_{1-(1-q)/2} * sigma). An OVERCONFIDENCE flag
     fires when empirical coverage falls materially below nominal (intervals too tight —
     the known diffusion/NN failure).
  4. PAIRED BOOTSTRAP 95% CI of AURC(method) - AURC(heteroscedastic), resampling FRAMES
     (the incumbent and the challenger re-scored on the IDENTICAL resampled frames so the
     shared base-rate variance cancels). Plus the +/-0.054 Hoeffding half-width at 3000
     clips printed as the PUBLISHABILITY FLOOR: a method must beat the incumbent by more
     than the floor to be a real, reportable win.

GATE LOGIC (printed in the table footer)
----------------------------------------
A challenger is a PUBLISHABLE WIN over the incumbent iff
    AURC(incumbent) - AURC(challenger) > Hoeffding floor (~0.054 @ 3000 clips)
    AND the paired-bootstrap 95% CI of (challenger - incumbent) AURC excludes 0 (< 0).
If the MDN already clears that bar, diffusion's plausible ceiling is already met by a
head that trains in minutes -> NO-GO on diffusion (the MDN is the contribution).

EVERYTHING here is pure-Python / numpy / torch-optional: the scoring core takes plain
float arrays so it is unit-testable on CPU with synthetic risks. A thin extractor
(extract_method_arrays) turns a trained-checkpoint dir + processed clips + oracle SNR
targets into the per-method arrays, mirroring snr_map_validate.py's .pt orchestration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the AURC / Hoeffding / Spearman / normal-quantile primitives — do NOT reinvent.
try:  # package-relative when imported as src.uq_bakeoff
    from .metrics_calibrated import aurc, hoeffding_halfwidth, spearman, normal_quantile
except ImportError:  # flat import when src/ is on sys.path (matches the repo style)
    from metrics_calibrated import aurc, hoeffding_halfwidth, spearman, normal_quantile


# The four uncertainty channels the gate compares. The first is the INCUMBENT.
METHODS = ("heteroscedastic_sigma", "mdn_var", "mcdropout_var", "ensemble_var")
INCUMBENT = "heteroscedastic_sigma"

# Nominal calibration levels reported by the reliability summary.
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)

# Overconfidence flag: empirical coverage this far (abs) below nominal at the 90% level
# => predicted intervals too tight. 0.10 is a deliberately loose, first-principles
# threshold (a 90% interval covering < 80% is materially overconfident).
OVERCONFIDENCE_GAP: float = 0.10

# The publishability floor stated in the task: the +/-0.054 finite-sample Hoeffding
# half-width at 3000 clips (two curves, delta=0.05). Recomputed exactly below for the
# actual n, but kept as a labeled constant for the table footer.
HOEFFDING_FLOOR_3000: float = 0.054


# ════════════════════════════════════════════════════════════════════════════════
# continuous-risk AURC (the mandatory NON-band-cut risk)
# ════════════════════════════════════════════════════════════════════════════════
def continuous_risk_aurc(
    uncertainty: list[float], abs_error: list[float]
) -> float:
    """Risk-coverage AURC with CONTINUOUS risk = |pred - true| (NOT a 0/1 band cut).

    We abstain on the highest-uncertainty frames FIRST, so the per-frame CONFIDENCE fed
    to metrics_calibrated.aurc is -uncertainty (higher confidence = lower uncertainty =
    retained earlier as coverage shrinks). The LOSS fed is the raw continuous |error| of
    each frame, NOT a thresholded indicator: aurc accumulates the running MEAN of those
    real-valued losses over the most-confident k and averages across coverage. So a
    method whose uncertainty concentrates on the large-|error| frames sheds error fast as
    coverage drops -> lower AURC. Lower is better.
    """
    if len(uncertainty) != len(abs_error):
        raise ValueError("uncertainty and abs_error must be aligned (same length)")
    if not uncertainty:
        return 0.0
    confidence = [-u for u in uncertainty]
    return aurc(confidence, abs_error)


def spearman_unc_err(
    uncertainty: list[float], abs_error: list[float]
) -> float | None:
    """Spearman(uncertainty, |error|): does the uncertainty RANK the errors? None if
    undefined (n<2 or a degenerate constant series)."""
    return spearman(uncertainty, abs_error)


# ════════════════════════════════════════════════════════════════════════════════
# calibration / reliability summary + overconfidence flag
# ════════════════════════════════════════════════════════════════════════════════
def _z_for_level(q: float) -> float:
    """Two-sided z so that P(|N(0,1)| <= z) = q, i.e. z = Phi^{-1}((1+q)/2)."""
    return normal_quantile((1.0 + q) / 2.0)


def calibration_summary(
    abs_error: list[float],
    sigma: list[float],
    levels: tuple[float, ...] = NOMINAL_LEVELS,
) -> dict:
    """Empirical coverage of Gaussian predictive intervals at each nominal level.

    Treats each frame's predictive law as N(pred, sigma^2): the nominal-q central
    interval is +/- z_q * sigma with z_q = Phi^{-1}((1+q)/2). Empirical coverage at q is
    the fraction of frames with |error| <= z_q * sigma. A well-calibrated uncertainty has
    empirical ~= nominal at every level; empirical << nominal means the intervals are too
    TIGHT (overconfident). `sigma` here is the predictive STANDARD DEVIATION (sqrt of the
    variance UQ signal). Returns per-level coverage + the gaps + an overconfidence flag
    (fires at the 90% level when coverage is OVERCONFIDENCE_GAP below nominal).
    """
    n = len(abs_error)
    if n == 0 or len(sigma) != n:
        return {
            "empirical_coverage": {f"{int(q*100)}": None for q in levels},
            "coverage_gap": {f"{int(q*100)}": None for q in levels},
            "overconfident": False,
            "n": n,
        }
    emp: dict[str, float] = {}
    gap: dict[str, float] = {}
    for q in levels:
        z = _z_for_level(q)
        covered = sum(1 for e, s in zip(abs_error, sigma) if s > 0 and e <= z * s)
        # frames with sigma == 0 only count as covered if their error is exactly 0
        covered += sum(1 for e, s in zip(abs_error, sigma) if s <= 0 and e <= 1e-12)
        cov = covered / n
        emp[f"{int(q * 100)}"] = cov
        gap[f"{int(q * 100)}"] = cov - q  # negative => overconfident at this level
    over = (emp.get("90") is not None) and (gap["90"] < -OVERCONFIDENCE_GAP)
    return {
        "empirical_coverage": emp,
        "coverage_gap": gap,
        "overconfident": bool(over),
        "n": n,
    }


# ════════════════════════════════════════════════════════════════════════════════
# paired bootstrap of AURC(method) - AURC(incumbent)
# ════════════════════════════════════════════════════════════════════════════════
def paired_bootstrap_aurc_delta(
    unc_method: list[float],
    unc_incumbent: list[float],
    abs_error: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Seeded paired-bootstrap CI of delta = AURC(method) - AURC(incumbent).

    Both uncertainty channels and the SHARED per-frame |error| are resampled on the
    IDENTICAL bootstrap indices each draw (paired), so the common base-rate variance
    cancels and the CI is on the DIFFERENCE. Negative delta = method beats the incumbent
    (lower AURC). Returns {point, lo, hi, p_value, n, n_boot, alpha} where p_value is the
    one-sided bootstrap probability that delta >= 0 (method NOT better). Same seed =>
    identical interval.
    """
    n = len(abs_error)
    if not (len(unc_method) == len(unc_incumbent) == n):
        raise ValueError("all three arrays must be aligned (same length)")
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0,
                "n": 0, "n_boot": n_boot, "alpha": alpha}
    point = continuous_risk_aurc(unc_method, abs_error) - \
        continuous_risk_aurc(unc_incumbent, abs_error)
    rng = random.Random(seed)
    stats = []
    n_ge0 = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        e = [abs_error[i] for i in idx]
        um = [unc_method[i] for i in idx]
        ui = [unc_incumbent[i] for i in idx]
        d = continuous_risk_aurc(um, e) - continuous_risk_aurc(ui, e)
        stats.append(d)
        if d >= 0:
            n_ge0 += 1
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot))]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"point": point, "lo": lo, "hi": hi, "p_value": n_ge0 / n_boot,
            "n": n, "n_boot": n_boot, "alpha": alpha}


# ════════════════════════════════════════════════════════════════════════════════
# per-method scoring + the full bake-off table
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class MethodScore:
    method: str
    n: int
    aurc: float
    spearman: float | None
    calibration: dict = field(default_factory=dict)
    # vs-incumbent fields (None for the incumbent itself)
    aurc_delta: float | None = None
    delta_lo: float | None = None
    delta_hi: float | None = None
    delta_p: float | None = None
    beats_floor: bool | None = None
    ci_excludes_zero: bool | None = None
    publishable_win: bool | None = None


def score_method(
    uncertainty: list[float],
    abs_error: list[float],
    method: str,
    sigma_for_calib: list[float] | None = None,
) -> dict:
    """Score ONE method's per-frame arrays: AURC + Spearman + calibration.

    `uncertainty` is the per-frame UQ signal used to RANK (a variance, or |sigma| for the
    incumbent). `sigma_for_calib` is the predictive STANDARD DEVIATION used for the
    Gaussian coverage check; defaults to sqrt(uncertainty) when not given (treats the
    uncertainty as a variance). `abs_error` is the per-frame |pred - true|.
    """
    if sigma_for_calib is None:
        sigma_for_calib = [math.sqrt(u) if u > 0 else 0.0 for u in uncertainty]
    return {
        "method": method,
        "n": len(abs_error),
        "aurc": continuous_risk_aurc(uncertainty, abs_error),
        "spearman": spearman_unc_err(uncertainty, abs_error),
        "calibration": calibration_summary(abs_error, sigma_for_calib),
    }


def run_bakeoff(
    method_arrays: dict[str, dict],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    incumbent: str = INCUMBENT,
) -> dict:
    """Score every method and compare each challenger to the incumbent.

    `method_arrays`: {method_name: {"uncertainty": [...], "abs_error": [...],
    optional "sigma": [...] for calibration}}. Every method MUST share the SAME
    `abs_error` length (they are scored on the same frames); the incumbent's abs_error is
    the reference. Returns a dict with per-method MethodScore-like records + the global
    Hoeffding floor + the GO/NO-GO verdict for the MDN.
    """
    if incumbent not in method_arrays:
        raise ValueError(f"incumbent {incumbent!r} not in method_arrays")
    ref_err = method_arrays[incumbent]["abs_error"]
    n = len(ref_err)
    floor = hoeffding_halfwidth(n, delta=alpha, n_curves=2) if n > 0 else float("inf")

    scores: dict[str, MethodScore] = {}
    for name, arr in method_arrays.items():
        s = score_method(
            arr["uncertainty"], arr["abs_error"], name,
            sigma_for_calib=arr.get("sigma"),
        )
        scores[name] = MethodScore(
            method=name, n=s["n"], aurc=s["aurc"],
            spearman=s["spearman"], calibration=s["calibration"],
        )

    inc_unc = method_arrays[incumbent]["uncertainty"]
    for name, arr in method_arrays.items():
        if name == incumbent:
            continue
        if len(arr["abs_error"]) != n:
            raise ValueError(
                f"method {name!r} has {len(arr['abs_error'])} frames; incumbent has {n}"
            )
        bs = paired_bootstrap_aurc_delta(
            arr["uncertainty"], inc_unc, ref_err,
            n_boot=n_boot, alpha=alpha, seed=seed,
        )
        sc = scores[name]
        sc.aurc_delta = bs["point"]
        sc.delta_lo = bs["lo"]
        sc.delta_hi = bs["hi"]
        sc.delta_p = bs["p_value"]
        # PUBLISHABLE WIN: beats incumbent by more than the finite-sample floor AND the
        # paired-bootstrap 95% CI of the (challenger - incumbent) AURC excludes 0 (< 0).
        sc.beats_floor = (sc.aurc_delta is not None) and (-sc.aurc_delta > floor)
        sc.ci_excludes_zero = (sc.delta_hi is not None) and (sc.delta_hi < 0.0)
        sc.publishable_win = bool(sc.beats_floor and sc.ci_excludes_zero)

    verdict = _gate_verdict(scores, floor)
    return {
        "n_frames": n,
        "hoeffding_floor": floor,
        "hoeffding_floor_3000_stated": HOEFFDING_FLOOR_3000,
        "incumbent": incumbent,
        "methods": {k: _score_to_dict(v) for k, v in scores.items()},
        "verdict": verdict,
    }


def _score_to_dict(s: MethodScore) -> dict:
    return {
        "method": s.method, "n": s.n, "aurc": s.aurc, "spearman": s.spearman,
        "calibration": s.calibration, "aurc_delta_vs_incumbent": s.aurc_delta,
        "delta_lo": s.delta_lo, "delta_hi": s.delta_hi, "delta_p": s.delta_p,
        "beats_floor": s.beats_floor, "ci_excludes_zero": s.ci_excludes_zero,
        "publishable_win": s.publishable_win,
    }


def _gate_verdict(scores: dict[str, MethodScore], floor: float) -> dict:
    """The GO/NO-GO on diffusion. If the MDN is a publishable win over the incumbent,
    diffusion's plausible ceiling is already met by a minutes-to-train head -> NO-GO on
    diffusion (the MDN is the paper). Otherwise no cheap head clears the bar and the
    diffusion experiment is JUSTIFIED (GO)."""
    mdn = scores.get("mdn_var")
    any_cheap_win = any(
        v.publishable_win for k, v in scores.items()
        if k != INCUMBENT and v.publishable_win
    )
    mdn_win = bool(mdn is not None and mdn.publishable_win)
    if mdn_win:
        decision = "NO-GO on diffusion: MDN already a publishable win over incumbent"
    elif any_cheap_win:
        decision = ("LEAN NO-GO: a cheap head (not MDN) beats incumbent; "
                    "diffusion must beat THAT head, not just sigma")
    else:
        decision = ("GO on diffusion: no cheap head beats the incumbent by the "
                    "publishability floor; the diffusion experiment is justified")
    return {
        "mdn_publishable_win": mdn_win,
        "any_cheap_publishable_win": bool(any_cheap_win),
        "floor": floor,
        "decision": decision,
    }


# ════════════════════════════════════════════════════════════════════════════════
# pretty-printed table
# ════════════════════════════════════════════════════════════════════════════════
def format_table(report: dict) -> str:
    """A clean fixed-width table of the bake-off, incumbent row first."""
    lines = []
    n = report["n_frames"]
    floor = report["hoeffding_floor"]
    inc = report["incumbent"]
    lines.append(f"UQ BAKE-OFF  (n_frames={n}, incumbent={inc})")
    lines.append(f"  publishability floor (Hoeffding, 2 curves, 95%) = {floor:.4f}"
                 f"   [stated +/-{report['hoeffding_floor_3000_stated']:.3f} @ 3000 clips]")
    lines.append("")
    hdr = (f"{'method':22s} {'AURC':>8s} {'dAURC':>8s} {'95% CI':>18s} "
           f"{'Spearman':>9s} {'cov90':>6s} {'over?':>5s} {'WIN?':>5s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    # incumbent first, then the rest in METHODS order
    order = [inc] + [m for m in METHODS if m != inc and m in report["methods"]]
    order += [m for m in report["methods"] if m not in order]
    for name in order:
        m = report["methods"][name]
        aurc_s = f"{m['aurc']:.4f}"
        if name == inc:
            d_s, ci_s, win_s = "  (ref)", "        --        ", "  -"
        else:
            d = m["aurc_delta_vs_incumbent"]
            d_s = f"{d:+.4f}" if d is not None else "   n/a"
            lo, hi = m["delta_lo"], m["delta_hi"]
            ci_s = (f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "       n/a       ")
            win_s = "YES" if m["publishable_win"] else " no"
        sp = m["spearman"]
        sp_s = f"{sp:.3f}" if sp is not None else "  n/a"
        cov = m["calibration"].get("empirical_coverage", {}).get("90")
        cov_s = f"{cov:.2f}" if cov is not None else " n/a"
        over_s = "YES" if m["calibration"].get("overconfident") else " no"
        lines.append(f"{name:22s} {aurc_s:>8s} {d_s:>8s} {ci_s:>18s} "
                     f"{sp_s:>9s} {cov_s:>6s} {over_s:>5s} {win_s:>5s}")
    lines.append("")
    lines.append("VERDICT: " + report["verdict"]["decision"])
    lines.append("  (lower AURC better; dAURC = method - incumbent, negative = better;")
    lines.append("   WIN = beats incumbent by > floor AND 95% CI excludes 0.)")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# checkpoint -> per-method per-frame arrays  (runnable once heads are trained)
# ════════════════════════════════════════════════════════════════════════════════
def extract_method_arrays(
    checkpoint_dir: str,
    processed_dir: str,
    snr_map_dir: str,
    max_clips: int = 0,
    mc_samples: int = 20,
    device: str = "cpu",
) -> dict[str, dict]:
    """Turn a trained-checkpoint dir + processed clips + oracle SNR targets into the
    per-method per-frame {uncertainty, abs_error, sigma} arrays the bake-off consumes.

    Expects, under `checkpoint_dir`, any of:
      reliability.pt        -> a {'reliability_head_state_dict' OR adapter ckpt} giving
                               the INCUMBENT per-frame sigma. (See note below.)
      mdn.pt                -> {'mdn_head_state_dict', 'config'} for MDNSNRMapHead.
      mcdropout.pt          -> {'mcdropout_head_state_dict', 'config'} for the MC head.
      ensemble/*.pt         -> N point-regression heads (SupervisedSNRMapHead-shaped)
                               whose disagreement is the ensemble variance.
    For each processed clip it reads `audio_features`, predicts with every available
    head, looks up the oracle per-frame SNR target via snr_map_dir/manifest.json, and
    accumulates, OVER SUPERVISED (mask=True) FRAMES, the per-frame |pred - true| and each
    method's uncertainty. Returns {method: {"uncertainty":[...], "abs_error":[...],
    "sigma":[...]}}; methods whose checkpoint is missing are simply omitted (the harness
    scores whatever is present, as long as the incumbent is there).

    NOTE: this is the GLUE for the actual RUN; it imports torch lazily so the scoring
    core (and its CPU unit tests) never need torch. The incumbent sigma extraction
    depends on how the reliability head is exposed per-frame in your trained model; the
    block below handles a dedicated per-frame sigma head if you train one, else raises a
    clear message so you wire it to your adapter's reliability output.
    """
    import torch  # lazy
    from snr_map_head import SupervisedSNRMapHead
    from uq_heads import (
        MDNSNRMapHead, MCDropoutSNRMapHead, ensemble_uncertainty,
    )

    def _load(path):
        return torch.load(path, map_location=device, weights_only=False)

    manifest = json.load(open(os.path.join(snr_map_dir, "manifest.json")))
    pts = sorted(f for f in os.listdir(processed_dir) if f.endswith(".pt"))
    if max_clips:
        pts = pts[:max_clips]

    # ── build whatever heads exist ──
    heads: dict[str, object] = {}
    mdn_path = os.path.join(checkpoint_dir, "mdn.pt")
    if os.path.exists(mdn_path):
        ck = _load(mdn_path)
        cfg = ck.get("config", {})
        h = MDNSNRMapHead(
            audio_dim=int(cfg.get("snr_map_audio_dim", 1024)),
            n_components=int(cfg.get("mdn_components", 3)),
            hidden=int(cfg.get("snr_map_hidden", 256)),
            kernel_size=int(cfg.get("snr_map_kernel_size", 5)),
        ).to(device)
        h.load_state_dict(ck["mdn_head_state_dict"])
        h.eval()
        heads["mdn_var"] = h

    mc_path = os.path.join(checkpoint_dir, "mcdropout.pt")
    if os.path.exists(mc_path):
        ck = _load(mc_path)
        cfg = ck.get("config", {})
        h = MCDropoutSNRMapHead(
            audio_dim=int(cfg.get("snr_map_audio_dim", 1024)),
            hidden=int(cfg.get("snr_map_hidden", 256)),
            kernel_size=int(cfg.get("snr_map_kernel_size", 5)),
            p=float(cfg.get("mc_dropout_p", 0.1)),
        ).to(device)
        h.load_state_dict(ck["mcdropout_head_state_dict"])
        h.eval()
        heads["mcdropout_var"] = h

    ens_dir = os.path.join(checkpoint_dir, "ensemble")
    ens_heads = []
    if os.path.isdir(ens_dir):
        for f in sorted(os.listdir(ens_dir)):
            if not f.endswith(".pt"):
                continue
            ck = _load(os.path.join(ens_dir, f))
            cfg = ck.get("config", {})
            hh = SupervisedSNRMapHead(
                audio_dim=int(cfg.get("snr_map_audio_dim", 1024)),
                hidden=int(cfg.get("snr_map_hidden", 256)),
                kernel_size=int(cfg.get("snr_map_kernel_size", 5)),
            ).to(device)
            hh.load_state_dict(ck["snr_map_head_state_dict"])
            hh.eval()
            ens_heads.append(hh)

    # Incumbent: a dedicated per-frame heteroscedastic SNR head exposing (mean, log_var).
    # We support a sibling checkpoint snr_sigma.pt with {'sigma_head_state_dict','config'}
    # built from a 2-output SupervisedSNRMapHead variant; if absent, the incumbent must be
    # supplied via --methods_json instead. (Kept flexible: the exact per-frame sigma wiring
    # is model-specific; the harness only needs the resulting per-frame sigma array.)
    inc_path = os.path.join(checkpoint_dir, "snr_sigma.pt")
    inc_head = None
    if os.path.exists(inc_path):
        ck = _load(inc_path)
        cfg = ck.get("config", {})
        # a 2-channel SupervisedSNRMapHead: out_proj -> 2 (mean, log_var). We re-use the
        # MDN K=1 head as the carrier of (mean, log_sigma) since K=1 mixture == a single
        # heteroscedastic Gaussian (pi=1), which is exactly the incumbent per frame.
        inc_head = MDNSNRMapHead(
            audio_dim=int(cfg.get("snr_map_audio_dim", 1024)),
            n_components=1,
            hidden=int(cfg.get("snr_map_hidden", 256)),
            kernel_size=int(cfg.get("snr_map_kernel_size", 5)),
        ).to(device)
        inc_head.load_state_dict(ck["sigma_head_state_dict"])
        inc_head.eval()

    out: dict[str, dict] = {m: {"uncertainty": [], "abs_error": [], "sigma": []}
                            for m in (
                                ["heteroscedastic_sigma"] if inc_head is not None else []
                            ) + list(heads.keys()) + (["ensemble_var"] if ens_heads else [])}

    for ptname in pts:
        cached = _load(os.path.join(processed_dir, ptname))
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        rel = manifest.get(filename)
        if rel is None:
            continue
        tgt = _load(os.path.join(snr_map_dir, rel))
        af = cached["audio_features"].to(device).unsqueeze(0).float()      # (1,T,D)
        target = tgt["snr_map_target"].to(device).float()                 # (T,)
        mask = tgt.get("snr_map_mask", torch.ones_like(target)).to(device).bool()
        T = min(af.shape[1], target.shape[0])
        af, target, mask = af[:, :T], target[:T], mask[:T]
        m_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if m_idx.numel() == 0:
            continue

        with torch.no_grad():
            # incumbent (heteroscedastic sigma): K=1 mixture -> (mean, var); sigma=sqrt(var)
            if inc_head is not None:
                params = inc_head.forward(af)
                mean, var = inc_head.predict(params)                       # (1,T)
                pm, pv = mean[0, :T], var[0, :T]
                err = (pm - target).abs()
                sig = pv.clamp(min=0).sqrt()
                _append(out["heteroscedastic_sigma"], pv[m_idx], err[m_idx], sig[m_idx])

            if "mdn_var" in heads:
                params = heads["mdn_var"].forward(af)
                mean, var = MDNSNRMapHead.predict(params)
                pm, pv = mean[0, :T], var[0, :T]
                err = (pm - target).abs()
                _append(out["mdn_var"], pv[m_idx], err[m_idx], pv[m_idx].clamp(min=0).sqrt())

            if "mcdropout_var" in heads:
                mean, var = heads["mcdropout_var"].sample(af, n=mc_samples)
                pm, pv = mean[0, :T], var[0, :T]
                err = (pm - target).abs()
                _append(out["mcdropout_var"], pv[m_idx], err[m_idx], pv[m_idx].clamp(min=0).sqrt())

            if ens_heads:
                preds = [h.forward_timeline(af)[0, :T] for h in ens_heads]
                mean, var = ensemble_uncertainty(preds)
                err = (mean - target).abs()
                _append(out["ensemble_var"], var[m_idx], err[m_idx], var[m_idx].clamp(min=0).sqrt())

    return out


def _append(d: dict, unc, err, sig) -> None:
    """Extend a method's running arrays with the supervised-frame tensors (as floats)."""
    d["uncertainty"].extend([float(x) for x in unc.tolist()])
    d["abs_error"].extend([float(x) for x in err.tolist()])
    d["sigma"].extend([float(x) for x in sig.tolist()])


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════
def _load_methods_json(path: str) -> dict[str, dict]:
    """Load {method: {"uncertainty":[...], "abs_error":[...], optional "sigma":[...]}}."""
    data = json.load(open(path))
    for name, arr in data.items():
        if "uncertainty" not in arr or "abs_error" not in arr:
            raise ValueError(f"method {name!r} needs 'uncertainty' and 'abs_error'")
    return data


def _load_npz_methods(specs: list[str]):
    """--npz heteroscedastic_sigma=path.npz ... ; each npz has uncertainty/abs_error
    (and optionally sigma) arrays."""
    import numpy as np  # lazy
    out: dict[str, dict] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--npz needs method=path, got {spec!r}")
        name, path = spec.split("=", 1)
        z = np.load(path)
        rec = {"uncertainty": [float(x) for x in z["uncertainty"]],
               "abs_error": [float(x) for x in z["abs_error"]]}
        if "sigma" in z:
            rec["sigma"] = [float(x) for x in z["sigma"]]
        out[name] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--methods_json", help="JSON of {method:{uncertainty,abs_error,[sigma]}}")
    src.add_argument("--npz", nargs="+", help="method=path.npz pairs")
    src.add_argument("--checkpoint_dir", help="trained-head dir (with mdn.pt etc.)")
    ap.add_argument("--processed_dir", help="processed .pt clips (with --checkpoint_dir)")
    ap.add_argument("--snr_map_dir", help="oracle SNR targets dir (with --checkpoint_dir)")
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    if args.methods_json:
        method_arrays = _load_methods_json(args.methods_json)
    elif args.npz:
        method_arrays = _load_npz_methods(args.npz)
    else:
        if not (args.processed_dir and args.snr_map_dir):
            ap.error("--checkpoint_dir needs --processed_dir and --snr_map_dir")
        method_arrays = extract_method_arrays(
            args.checkpoint_dir, args.processed_dir, args.snr_map_dir,
            max_clips=args.max_clips, mc_samples=args.mc_samples, device=args.device,
        )

    report = run_bakeoff(method_arrays, n_boot=args.n_boot, alpha=args.alpha,
                         seed=args.seed)
    print(format_table(report))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\n[wrote {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
