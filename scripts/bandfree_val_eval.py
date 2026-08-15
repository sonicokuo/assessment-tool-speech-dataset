#!/usr/bin/env python
"""Band-free per-feature SRCC/nMAE/coverage on a run's val_samples (the NTL-vs-noNTL eval).

Parses each generated description, joins to clean GT, scores the 12 canonical features.
overlap_ratio GT comes from the reference 'target' text (a mix property). Features whose
clean GT is unavailable on this split are reported with coverage only.

HEADLINE FREEZE (fix F8, 2026-07-15 metric-consistency audit): the headline is the mean
SRCC over EXACTLY selection_metric.HEADLINE_FEATURES (frozen ROBUST5, snr INCLUDED),
imported from its single source of truth — the same set the val selector uses. All other
features (f0_mean, f0_sd, hnr, jitter, shimmer, articulation_rate, overlap_ratio) are
printed in a separate "ill-posed / abstention panel" and are NEVER averaged into the
headline. A no-snr secondary mean (SNR-circularity check) and a headline-level mean
coverage are always printed alongside.
"""
import json, sys, glob, os

# PSC shared checkout first as a fallback, then the repo-relative src (inserted last so
# it wins) — the script now also imports cleanly when run/tested from the repo itself.
sys.path.insert(0, '/ocean/projects/cis260125p/shared/assessment-tool-redirect/src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data.feature_set import FEATURE_NAMES  # canonical 11-feature list (single source)
from eval.sfs import HybridClaimParser, SFSScorer
from eval.selection_metric import HEADLINE_FEATURES  # frozen ROBUST5 (F8)

DATA = '/ocean/projects/cis260125p/shared/data'
# feature -> (gt_source, gt_col)  ; gt_source in {feat, f0, target}
FEAT = {
    'snr': ('feat', 'snr_db'), 'srmr': ('feat', 'srmr'), 'hnr': ('feat', 'hnr_db'),
    'f0_mean': ('f0', 'f0_mean_hz'), 'f0_sd': ('f0', 'f0_sd_hz'),
    'jitter': ('feat', 'jitter_local_pct'), 'shimmer': ('feat', 'shimmer_pct'),
    'speaking_rate': ('feat', 'praat_speaking_rate_syl_sec'),
    'articulation_rate': ('feat', 'praat_articulation_rate_syl_sec'),
    'pause_count': ('feat', 'praat_pause_count'),
    'pause_rate': ('feat', 'praat_pause_rate_per_min'),
    'overlap_ratio': ('target', None),
}

# Panel = every scored feature that is NOT in the frozen headline. f0 is diagnosed
# ill-posed / mode-collapsing (2026-07-13); overlap_ratio is the FiLM conditioning
# input (leak); articulation_rate was dropped from the supervised set 2026-06-24.
PANEL_ORDER = [f for f in ('hnr', 'f0_mean', 'f0_sd', 'jitter', 'shimmer',
                           'articulation_rate', 'overlap_ratio') if f in FEAT]
assert set(HEADLINE_FEATURES) <= set(FEAT), 'headline features need a GT mapping in FEAT'


def fnum(x):
    try:
        v = float(x); return v if v == v else None
    except Exception:
        return None

def load_gt(split):
    cf, c0 = {}, {}
    for k, v in json.load(open(f'{DATA}/clean_features_{split}.json')).items():
        cf[k] = v
    p0 = f'{DATA}/clean_f0_{split}.json'
    if os.path.exists(p0):
        for k, v in json.load(open(p0)).items():
            c0[k] = v
    return cf, c0

P = HybridClaimParser()

# LATENT-COUPLING FIX (2026-07-29): this allowlist used to read
# `SFSScorer.TOLERANCES`, i.e. the BAND-FREE metric's feature scope was defined by the
# RETIRED band metric's tolerance dict. Anyone tidying or trimming TOLERANCES would
# silently change which features this scorer reports, with no error — the same
# rename-shaped failure that erased hnr/shimmer from every target for two weeks. The
# canonical feature list is feature_set.FEATURE_NAMES; depend on that instead.
# Union with TOLERANCES keys so nothing currently reported silently disappears
# (TOLERANCES carries a few extras such as duration_sec/sample_rate from the old parser).
_ALLOWED = set(FEATURE_NAMES) | set(SFSScorer.TOLERANCES)


def claims(t):
    o = {}
    for c in P.parse(t or ''):
        if c.feature in _ALLOWED and c.feature not in o:
            o[c.feature] = c.value
    return o

def score_run(val_json, split):
    import numpy as np                 # lazy: keep module importable without numpy/scipy
    from scipy.stats import spearmanr
    cf, c0 = load_gt(split)
    res = json.load(open(val_json))
    total = len(res)
    mc = {e['filename']: claims(e.get('generated') or '') for e in res}
    mt = {e['filename']: claims(e.get('target') or '') for e in res}  # for overlap_ratio GT

    def gt_for(feat, fn):
        src, col = FEAT[feat]
        if src == 'target':
            return mt.get(fn, {}).get(feat)
        d = (cf if src == 'feat' else c0).get(fn)
        # try filename and stem keys
        if d is None:
            stem = os.path.splitext(fn)[0]
            d = (cf if src == 'feat' else c0).get(stem) or (cf if src == 'feat' else c0).get(stem + '.wav')
        return fnum(d.get(col)) if d else None

    out = {}
    for feat in FEAT:
        xs, ys = [], []
        for fn, cl in mc.items():
            if feat in cl:
                g = gt_for(feat, fn)
                if g is not None:
                    xs.append(cl[feat]); ys.append(g)
        # Coverage sits NEXT TO the SRCC everywhere (F10): SRCC pairs form only where
        # a claim parses, so under asymmetric degeneration two models are silently
        # scored on different clip subsets (2026-07-15 audit risk 3).
        cov = len(xs) / total if total else 0.0
        if len(xs) >= 8:
            xs = np.array(xs); ys = np.array(ys)
            sr = spearmanr(xs, ys).correlation
            sd = float(np.std(ys)); mae = float(np.mean(np.abs(xs - ys)))
            out[feat] = {'srcc': None if sr != sr else round(float(sr), 3),
                         'nmae': round(mae / sd, 3) if sd > 1e-9 else None,
                         'cov': round(cov, 3), 'n': len(xs)}
        else:
            out[feat] = {'srcc': None, 'nmae': None, 'cov': round(cov, 3), 'n': len(xs)}
    return out, total


def headline_summary(scores):
    """Frozen-headline aggregates from a {feat: {'srcc':.., 'cov':..}} score dict.

    Pure python (no numpy) so it is unit-testable without the PSC deps. Returns:
      mean_srcc        — mean SRCC over exactly HEADLINE_FEATURES with a defined SRCC
                         (frozen ROBUST5; snr INCLUDED — F8). None if none defined.
      mean_srcc_no_snr — the same mean without snr (SNR-circularity check, standard
                         secondary line).
      mean_cov         — mean coverage over ALL headline features, defined even where
                         SRCC is None (a feature that degenerated to 0 emissions still
                         drags this down — that is the point; F10 / audit risk 3).
      n_feats          — headline features contributing a defined SRCC (max 5).
    """
    srccs = [scores[f]['srcc'] for f in HEADLINE_FEATURES
             if f in scores and scores[f].get('srcc') is not None]
    no_snr = [scores[f]['srcc'] for f in HEADLINE_FEATURES
              if f != 'snr' and f in scores and scores[f].get('srcc') is not None]
    covs = [scores[f].get('cov') or 0.0 for f in HEADLINE_FEATURES if f in scores]
    _m = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {'mean_srcc': _m(srccs), 'mean_srcc_no_snr': _m(no_snr),
            'mean_cov': _m(covs), 'n_feats': len(srccs)}


def main(argv):
    runs = argv or [
        '/ocean/projects/cis260125p/shared/checkpoints/newproj_full12/val_samples',
        '/ocean/projects/cis260125p/shared/checkpoints/newproj_full12_noNTL/val_samples',
    ]
    split = 'dev'
    print(f'{"feature":18}' + ''.join(f'{os.path.basename(os.path.dirname(r))[:14]:>16}' for r in runs))
    allscores = {}
    for r in runs:
        vj = sorted(glob.glob(r + '/epoch_*.json'))[-1]
        sc, total = score_run(vj, split)
        allscores[r] = sc

    def row(feat):
        line = f'{feat:18}'
        for r in runs:
            m = allscores[r][feat]
            cell = (f'{m["srcc"]:.3f}/c{m["cov"]:.2f}' if m['srcc'] is not None
                    else f'  -/c{m["cov"]:.2f}/n{m["n"]}')
            line += f'{cell:>16}'
        print(line)

    print('-- headline (frozen ROBUST5 = selection_metric.HEADLINE_FEATURES) --')
    for feat in HEADLINE_FEATURES:
        row(feat)
    print('-- ill-posed / abstention panel (never in the headline mean) --')
    for feat in PANEL_ORDER:
        row(feat)

    # Frozen headline (F8): mean SRCC over exactly HEADLINE_FEATURES — snr INCLUDED
    # since the B2 flip; f0/overlap_ratio/articulation_rate never in this mean.
    print()
    for r in runs:
        h = headline_summary(allscores[r])
        name = os.path.basename(os.path.dirname(r))
        print(f'  {name}: HEADLINE mean SRCC (ROBUST5, snr INCLUDED) = {h["mean_srcc"]}'
              f'  [{h["n_feats"]}/{len(HEADLINE_FEATURES)} feats]')
        print(f'      no-snr secondary (circularity check) = {h["mean_srcc_no_snr"]}'
              f'   headline mean coverage = {h["mean_cov"]}')


if __name__ == '__main__':
    main(sys.argv[1:])
