#!/usr/bin/env python
"""Band-free per-feature SRCC/nMAE/coverage on a run's val_samples (the NTL-vs-noNTL eval).

Parses each generated description, joins to clean GT, scores the 12 canonical features.
overlap_ratio GT comes from the reference 'target' text (a mix property). Features whose
clean GT is unavailable on this split are reported with coverage only.
"""
import json, sys, glob, os
sys.path.insert(0, '/ocean/projects/cis260125p/shared/assessment-tool-redirect/src')
from sfs import HybridClaimParser, SFSScorer
import numpy as np
from scipy.stats import spearmanr

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
def claims(t):
    o = {}
    for c in P.parse(t or ''):
        if c.feature in SFSScorer.TOLERANCES and c.feature not in o:
            o[c.feature] = c.value
    return o

def score_run(val_json, split):
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

RELIABLE = ['srmr', 'speaking_rate', 'articulation_rate', 'pause_count', 'pause_rate', 'overlap_ratio']  # snr excluded (degenerate)
ORDER = ['snr', 'srmr', 'hnr', 'f0_mean', 'f0_sd', 'jitter', 'shimmer',
         'speaking_rate', 'articulation_rate', 'pause_count', 'pause_rate', 'overlap_ratio']

runs = sys.argv[1:] or [
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
for feat in ORDER:
    row = f'{feat:18}'
    for r in runs:
        m = allscores[r][feat]
        cell = (f'{m["srcc"]:.3f}/c{m["cov"]:.2f}' if m['srcc'] is not None else f'  -/c{m["cov"]:.2f}/n{m["n"]}')
        row += f'{cell:>16}'
    print(row)
# mean SRCC over reliable scorable
print()
for r in runs:
    srccs = [allscores[r][f]['srcc'] for f in RELIABLE if allscores[r][f].get('srcc') is not None]
    mean = round(float(np.mean(srccs)), 3) if srccs else None
    print(f'  {os.path.basename(os.path.dirname(r))}: mean SRCC (reliable, excl snr) = {mean}  [{len(srccs)} feats]')
