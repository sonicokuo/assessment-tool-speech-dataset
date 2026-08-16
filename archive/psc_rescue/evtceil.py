"""Resolution ceilings for the event features + the event-duration statistic that
should predict them. A model score is meaningless without its ceiling."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, mass_concentration
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_events_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]][:800]
TOKEN_MS = 160.0
for feat in ("speaking_rate", "pause_count"):
    ceil_sp, ceil_mc, runlen = [], [], []
    for n in names:
        k = f"{feat}/{n}"
        if k not in o:
            continue
        ref = np.asarray(o[k], dtype=float)
        if ref.size < 8 or not (ref > 0).any() or (ref > 0).all():
            continue
        ceil_sp.append(spearman(degrade(ref), ref))
        ceil_mc.append(mass_concentration(degrade(ref), ref))
        # contiguous run lengths of the support, in ms (grid is 10 ms)
        sup = (ref > 0).astype(int)
        d = np.diff(np.concatenate([[0], sup, [0]]))
        runs = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) * 10.0
        runlen.extend(runs.tolist())
    s, sl, sh = boot_ci(np.array(ceil_sp, float), 500)
    m, ml, mh = boot_ci(np.array(ceil_mc, float), 500)
    rl = np.array(runlen, float)
    print(f"{feat:<14} CEILING spearman {s:+.4f} [{sl:+.4f},{sh:+.4f}]   "
          f"mass-conc {m:+.4f} [{ml:+.4f},{mh:+.4f}]")
    print(f"{'':<14} support-event duration: median {np.median(rl):.0f} ms, "
          f"{100*(rl < TOKEN_MS).mean():.0f}% SHORTER than one {TOKEN_MS:.0f} ms token   n_events={rl.size}")
