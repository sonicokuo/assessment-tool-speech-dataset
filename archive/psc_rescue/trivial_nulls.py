"""D3 — price every feature against TRIVIAL-INFORMATION nulls.

We ran the energy-envelope null only for snr (0.586 vs a 0.674 ceiling = 87%, correctly
treated as damning) and never applied it to our OWN headline. That is the gap this closes.

WHY EACH FEATURE IS EXPOSED
  pause_count / pause_rate : intervals are defined by an INTENSITY THRESHOLD (-25 dB below
      peak), so the reference is a near-deterministic function of the energy envelope. A
      "-energy" map may score much of the 0.917 ceiling with ZERO model involvement.
  speaking_rate            : nuclei are intensity-contour PEAKS. Same exposure.
  f0_mean / f0_sd          : on clean twins the reference support IS the voiced mask, so
      alignment can only ever certify SUPPORT DETECTION, never value grounding. The right
      null is the VOICING MASK itself (N6).

Nulls computed here, all model-free and derived from the audio the model actually sees:
  N1  random
  N6a -energy  (negative envelope: high where the signal is quiet -> pauses)
  N6b +energy  (the envelope itself -> speech-active frames, nuclei)
  N6c voicing  (|energy| above the same -25 dB rule: a crude VAD, no pitch notion at all)
Everything carries the model's own resolution handicap via degrade(), same footing rule.
"""
import os, sys, numpy as np, soundfile as sf
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, mass_concentration
SH = "/ocean/projects/cis260125p/shared"
AUD = {"mix": f"{SH}/data/audio_corrected/test", "clean": f"{SH}/data/audio_corrected/test-s1clean"}
FRAME = 160          # 10 ms at 16 kHz -> matches the 100 Hz oracle grid

def envelope(stem):
    p = os.path.join(AUD["clean"] if stem.endswith("_s1clean") else AUD["mix"], stem + ".wav")
    if not os.path.exists(p):
        return None
    x, _ = sf.read(p)
    x = x.mean(1) if x.ndim > 1 else x
    nf = len(x) // FRAME
    if nf < 8:
        return None
    e = (x[:nf * FRAME].reshape(nf, FRAME) ** 2).mean(axis=1) + 1e-12
    return 10 * np.log10(e)          # dB envelope on the oracle grid

SRC = [("oracle_maps_test.npz", ("f0_mean", "f0_sd")),
       ("oracle_events_test.npz", ("speaking_rate", "pause_count"))]
rng = np.random.default_rng(0)
print(f"{'feature':<14}{'panel':<7}{'CEILING':>10}{'N1 rand':>10}{'-energy':>10}"
      f"{'+energy':>10}{'voicing':>10}   n")
for f_npz, feats in SRC:
    z = np.load(f"{SH}/{f_npz}", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    for feat in feats:
        for panel, sel in (("MIX", lambda n: not n.endswith("_s1clean")),
                           ("CLEAN", lambda n: n.endswith("_s1clean"))):
            C, R, NE, PE, V = [], [], [], [], []
            for n in names:
                k = f"{feat}/{n}"
                if k not in z or not sel(n):
                    continue
                ref = np.asarray(z[k], dtype=float)
                if ref.size < 8 or np.allclose(ref, ref[0]):
                    continue
                edb = envelope(n)
                if edb is None:
                    continue
                m = min(edb.size, ref.size)
                ref, edb = ref[:m], edb[:m]
                vad = (edb > (edb.max() - 25.0)).astype(float)   # the instrument's own rule
                C.append(spearman(degrade(ref), ref))
                R.append(spearman(degrade(rng.random(m)), ref))
                NE.append(spearman(degrade(-edb), ref))
                PE.append(spearman(degrade(edb), ref))
                V.append(spearman(degrade(vad), ref))
                if len(C) >= 700:
                    break
            if len(C) < 30:
                continue
            g = lambda v: boot_ci(np.array(v, float), 400)[0]     # noqa: E731
            print(f"{feat:<14}{panel:<7}{g(C):>10.3f}{g(R):>10.3f}{g(NE):>10.3f}"
                  f"{g(PE):>10.3f}{g(V):>10.3f}   {len(C)}")
print("\nA null close to the CEILING means that feature's map row cannot distinguish")
print("model grounding from trivial energy/voicing structure -- the snr situation (87%).")
