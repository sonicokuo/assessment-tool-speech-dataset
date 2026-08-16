# PSC rescue — loose code from $SHARED top level, 2026-08-16

**What this is.** Every loose `*.py` (98) and `*.sh` (41) that was sitting at the top level of
`/ocean/projects/cis260125p/shared`. None of it was in version control. It is committed here
UNSORTED AND UNMODIFIED, byte-for-byte as found, so that the reorganisation of `$SHARED` cannot
lose anything. Sorting into `scripts/` happens as a separate, reviewable step.

**Why this was urgent.** These files include the producers of numbers currently quoted in the
plan and in CLAUDE.md, and they existed in exactly one place on a filesystem with no backup:

| script | produced |
|---|---|
| `aurc_eval.py` | the AURC figures quoted in CLAUDE.md (ill-posed +12.2%, robust5 +19.9%, srmr +38.3%) |
| `attribution_capture.py` | `attribution_fw2.npz` / `attribution_lsm.npz`, feeding contribution III |
| `aux_repool.py` | every aux-head ROBUST5 / ILL-POSED panel number |
| `analyze.py` | the corrected correlational panel; "sigma separates conditions, not claims" |
| `aqua_f0_audit.py` | the fw-vs-fw2 target-content evidence |
| `graded_overlap.py` | f0 rho +1.000 -> -0.900 blinded |
| `remix_error.py`, `erf_per_feature.py` | the causal NULL and the ~1 s ERF |
| `layer_ridge.py`, `probe_family.py`, `skyline.py` | the ridge baselines we benchmark against |

**Do not run anything from this directory.** Several are path-bound to `$SHARED` and several are
superseded by twins (see the `_INVALID` list in the PSC plan). This is a preservation copy.

**Known caller coupling — must be fixed before the originals move on PSC.**
`repo_verify/scripts/dispatcher.py` invokes `aux_repool.py` by bare filename from `$SHARED` at
lines 110, 237, 246, 310 and 416 (`cd {SH} && {PY} -u aux_repool.py`). Six further scripts are
hardcoded in `.sh` launchers. Move the file and fix its caller in the SAME change, or leave a
symlink for one transition window.
