#!/usr/bin/env python
"""Band-free SFS comparison across versions (decision 2026-06-23).

Retires band precision/F1. Reports, per feature on STATED claims vs CLEAN GT:
  SRCC (Spearman), nMAE = MAE/std(GT), coverage = mentioned/total.
Headline scalar = mean SRCC over SCORABLE features (GT variance; SNR excluded as
degenerate s1-vs-s2 near-constant, flagged separately, never averaged in).

Single deterministic driver run uniformly over every version -> one comparable table.
Usage: python bandfree_compare.py [v1 v2 ...]   (default: curated main + ablation set)
"""
import json, sys, os
sys.path.insert(0, '/ocean/projects/cis260125p/shared/assessment-tool-redirect/src')
from sfs import HybridClaimParser, SFSScorer
import numpy as np
from scipy.stats import spearmanr, pearsonr

SHARED = '/ocean/projects/cis260125p/shared'
DATA = f'{SHARED}/data'
SPLITS = ('test', 'dev')

# feature -> (claim_key, gt_source, clean_gt_col)
FEAT = {
    'snr':               ('snr_db', 'feat', 'snr_db'),
    'srmr':              ('srmr', 'feat', 'srmr'),
    'f0_mean':           ('f0_mean_hz', 'f0', 'f0_mean_hz'),
    'f0_sd':             ('f0_sd_hz', 'f0', 'f0_sd_hz'),
    'speaking_rate':     ('praat_speaking_rate_syl_sec', 'feat', 'praat_speaking_rate_syl_sec'),
    'articulation_rate': ('praat_articulation_rate_syl_sec', 'feat', 'praat_articulation_rate_syl_sec'),
    'pause_count':       ('praat_pause_count', 'feat', 'praat_pause_count'),
    'pause_rate':        ('praat_pause_rate_per_min', 'feat', 'praat_pause_rate_per_min'),
}
# SNR excluded from the headline mean: s1-vs-s2 mix-clean SNR is near-constant (degenerate GT).
EXCLUDE_FROM_MEAN = {'snr'}

DEFAULT_VERSIONS = [
    # main progression
    'v7_lora_8b', 'v9_lora_8b_dur', 'v9_rescore_cleanf0', 'v11_section_head_lora',
    'v14_aug', 'v17_decoupled', 'v21_observability', 'qwen3_4b_full_ft_tagged_v1',
    # adapter ablations (8B)
    'q3_8b_concat_v2', 'q3_8b_film_attn_v3', 'q3_8b_film_mamba_v2', 'q3_8b_qformer_v2',
]

def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None

# clean GT (test + dev)
cf, c0 = {}, {}
for sp in SPLITS:
    for k, v in json.load(open(f'{DATA}/clean_features_{sp}.json')).items():
        cf[k] = v
    for k, v in json.load(open(f'{DATA}/clean_f0_{sp}.json')).items():
        c0[k] = v

def cgt(feat, fn):
    _, src, ck = FEAT[feat]
    d = (cf if src == 'feat' else c0).get(fn)
    return fnum(d.get(ck)) if d else None

P = HybridClaimParser()
def claims(t):
    o = {}
    for c in P.parse(t):
        if c.feature in SFSScorer.TOLERANCES and c.feature not in o:
            o[c.feature] = c.value
    return o

def score_version(v):
    pred = f'{SHARED}/checkpoints/{v}/inference_results.json'
    if not os.path.exists(pred):
        return None
    res = json.load(open(pred))
    total = len(res)
    mc = {e['filename']: claims(e.get('generated') or e.get('generated_clean') or '') for e in res}
    feats = {}
    for feat in FEAT:
        xs, ys = [], []
        for fn, cl in mc.items():
            if feat in cl:
                g = cgt(feat, fn)
                if g is not None:
                    xs.append(cl[feat]); ys.append(g)
        cov = len(xs) / total if total else 0.0
        if len(xs) >= 10:
            xs = np.array(xs); ys = np.array(ys)
            sr = spearmanr(xs, ys).correlation
            sd = float(np.std(ys)); mae = float(np.mean(np.abs(xs - ys)))
            feats[feat] = {'srcc': None if sr != sr else round(float(sr), 3),
                           'nmae': round(mae / sd, 3) if sd > 1e-9 else None,
                           'cov': round(cov, 3), 'n': len(xs), 'gt_sd': round(sd, 3)}
        else:
            feats[feat] = {'srcc': None, 'nmae': None, 'cov': round(cov, 3), 'n': len(xs)}
    srccs = [m['srcc'] for f, m in feats.items()
             if f not in EXCLUDE_FROM_MEAN and m.get('srcc') is not None and (m.get('gt_sd') or 0) > 1e-6]
    covs = [m['cov'] for f, m in feats.items() if f not in EXCLUDE_FROM_MEAN]
    return {'n_eval': total, 'features': feats,
            'mean_srcc_excl_snr': round(float(np.mean(srccs)), 3) if srccs else None,
            'mean_coverage_excl_snr': round(float(np.mean(covs)), 3) if covs else None}

VERSIONS = sys.argv[1:] or DEFAULT_VERSIONS
out = {}
for v in VERSIONS:
    r = score_version(v)
    if r is not None:
        out[v] = r

os.makedirs(f'{SHARED}/rescore_v21', exist_ok=True)
json.dump(out, open(f'{SHARED}/rescore_v21/bandfree_all_versions.json', 'w'), indent=1)

# ---- comparable table ----
order = ['srmr', 'pause_count', 'pause_rate', 'speaking_rate', 'articulation_rate', 'f0_mean', 'f0_sd', 'snr']
hdr = f'{"version":24} {"n":>5} {"meanSRCC":>8} ' + ' '.join(f'{f[:8]:>8}' for f in order)
print('=== BAND-FREE SFS (SRCC per feature; mean excl. SNR) ===')
print(hdr)
for v, r in out.items():
    row = f'{v:24} {r["n_eval"]:>5} {(("%.3f"%r["mean_srcc_excl_snr"]) if r["mean_srcc_excl_snr"] is not None else "  -  "):>8} '
    row += ' '.join(f'{(("%.3f"%r["features"][f]["srcc"]) if r["features"][f].get("srcc") is not None else "  -"):>8}' for f in order)
    print(row)
print()
print('=== nMAE = MAE/std(GT) per feature (lower=better; >1 worse than predicting the mean) ===')
print(f'{"version":24} ' + ' '.join(f'{f[:8]:>8}' for f in order))
for v, r in out.items():
    print(f'{v:24} ' + ' '.join(f'{(("%.2f"%r["features"][f]["nmae"]) if r["features"][f].get("nmae") is not None else "  -"):>8}' for f in order))
print()
print('=== coverage = fraction of clips making a claim, per feature ===')
print(f'{"version":24} {"meanCov":>7} ' + ' '.join(f'{f[:8]:>8}' for f in order))
for v, r in out.items():
    print(f'{v:24} {(("%.2f"%r["mean_coverage_excl_snr"]) if r["mean_coverage_excl_snr"] is not None else "  -"):>7} '
          + ' '.join(f'{("%.2f"%r["features"][f]["cov"]):>8}' for f in order))
