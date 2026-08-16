import soundfile as sf, os, numpy as np
d = "/ocean/projects/cis260125p/shared/data/audio_corrected/test"
f = sorted(os.listdir(d))[0]
i = sf.info(os.path.join(d, f))
print("file:", f)
print("subtype:", i.subtype, " samplerate:", i.samplerate, " frames:", i.frames)
print("16-bit step 2^-15 =", 2 ** -15)
print("observed max|diff| = 3.052e-05   ratio =", 3.052e-05 / (2 ** -15))
