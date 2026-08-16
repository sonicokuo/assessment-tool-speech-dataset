import torch, os, glob
d1="/ocean/projects/cis260125p/shared/data/processed_clean_control"
d2="/ocean/projects/cis260125p/shared/data/processed_pyannote/test"
s1=set(os.path.basename(p) for p in glob.glob(os.path.join(d1,"*.pt")))
s2=set(os.path.basename(p) for p in glob.glob(os.path.join(d2,"*.pt")))
print("clean_control n:",len(s1),"pyannote n:",len(s2),"intersection:",len(s1&s2),"only_clean:",len(s1-s2),"only_pya:",len(s2-s1))
# audio_features identical?
fn="1089-134686-0000_121-127105-0031.pt"
a=torch.load(os.path.join(d1,fn),map_location="cpu",weights_only=False)
b=torch.load(os.path.join(d2,fn),map_location="cpu",weights_only=False)
af1,af2=a["audio_features"],b["audio_features"]
print("audio_features shapes:",tuple(af1.shape),tuple(af2.shape))
if af1.shape==af2.shape:
    print("audio_features max abs diff:", float((af1-af2).abs().max()))
print("overlap_segments clean:",a["overlap_segments"][:3])
print("overlap_segments pya  :",b["overlap_segments"][:3])
# check overlap_info zero across MANY clean_control clips
import random
zero=0;tot=0
for p in random.sample(list(s1&s2), 40):
    x=torch.load(os.path.join(d1,p),map_location="cpu",weights_only=False)
    oi=x["overlap_info"]
    if float(oi.abs().sum())==0: zero+=1
    tot+=1
print(f"clean_control: {zero}/{tot} sampled clips have ALL-ZERO overlap_info")
