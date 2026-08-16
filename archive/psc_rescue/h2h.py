import json, csv, math
from collections import defaultdict

V21="/ocean/projects/cis260125p/shared/checkpoints/v21_observability/inference_results.json"
A  ="/ocean/projects/cis260125p/shared/checkpoints/v21repl_A/inference_results.json"
CJ ="/ocean/projects/cis260125p/shared/data/clean_features_test.json"
CSV="/ocean/projects/cis260125p/shared/data/features_pyannote/test_cleanf0.csv"

FEATS=["srmr","speaking_rate","pause_count","pause_rate","snr"]
# GT key in clean_features_test.json
GTKEY={"srmr":"srmr","speaking_rate":"praat_speaking_rate_syl_sec",
       "pause_count":"praat_pause_count","pause_rate":"praat_pause_rate_per_min"}

def load_claims(p):
    d=json.load(open(p))
    out={}  # filename -> {feature: claimed}
    perf_actual={}  # filename -> {feature: actual}
    for e in d:
        fn=e["filename"]
        cd={}
        for f,v in e["claims"]:
            # keep first occurrence
            if f not in cd: cd[f]=v
        out[fn]=cd
        pa={}
        for pf in e["per_feature"]:
            pa[pf["feature"]]=pf["actual"]
        perf_actual[fn]=pa
    return out, perf_actual

v21c,v21a=load_claims(V21)
ac,aa=load_claims(A)

# GT for srmr/speaking/pause from clean_features_test.json
cj=json.load(open(CJ))
# GT snr from csv
csvsnr={}
with open(CSV) as f:
    for row in csv.DictReader(f):
        try: csvsnr[row["filename"]]=float(row["snr_db"])
        except: pass

def gt_val(fn, feat):
    if feat=="snr":
        return csvsnr.get(fn)
    k=GTKEY[feat]
    rec=cj.get(fn)
    if rec is None: return None
    v=rec.get(k)
    return float(v) if v is not None else None

def srcc(xs, ys):
    # Spearman via rank Pearson, average ties
    n=len(xs)
    if n<3: return float('nan')
    def ranks(a):
        order=sorted(range(n), key=lambda i:a[i])
        r=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and a[order[j+1]]==a[order[i]]: j+=1
            avg=(i+j)/2.0+1
            for k in range(i,j+1): r[order[k]]=avg
            i=j+1
        return r
    rx=ranks(xs); ry=ranks(ys)
    mx=sum(rx)/n; my=sum(ry)/n
    cov=sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    vx=math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    vy=math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    if vx==0 or vy==0: return float('nan')
    return cov/(vx*vy)

def pearson(xs,ys):
    n=len(xs)
    if n<3: return float('nan')
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    vx=math.sqrt(sum((xs[i]-mx)**2 for i in range(n)))
    vy=math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
    if vx==0 or vy==0: return float('nan')
    return cov/(vx*vy)

def stats(vals):
    n=len(vals)
    if n==0: return (float('nan'),float('nan'),0)
    m=sum(vals)/n
    sd=math.sqrt(sum((x-m)**2 for x in vals)/n) if n>1 else 0.0
    return (m,sd,len(set(round(x,6) for x in vals)))

print(f"v21 clips emitting: total {len(v21c)}")
print(f"A   clips emitting: total {len(ac)}")
print()
hdr=f"{'feat':<14}{'n_int':>6}{'v21_mean':>10}{'A_mean':>10}{'v21_SD':>9}{'A_SD':>9}{'SDr(v21/A)':>11}{'v21_nuq':>8}{'A_nuq':>8}{'SRCC_v21':>10}{'SRCC_A':>9}{'r(claim)':>10}"
print(hdr)
print("-"*len(hdr))
hl21=[]; hlA=[]
for feat in FEATS:
    # intersection of clips both emitted a value AND GT exists
    xs21=[]; xsA=[]; gts=[]; fns=[]
    for fn in v21c:
        if feat not in v21c[fn]: continue
        if fn not in ac or feat not in ac[fn]: continue
        g=gt_val(fn,feat)
        if g is None: continue
        xs21.append(v21c[fn][feat]); xsA.append(ac[fn][feat]); gts.append(g); fns.append(fn)
    n=len(xs21)
    m21,sd21,nu21=stats(xs21)
    mA,sdA,nuA=stats(xsA)
    s21=srcc(xs21,gts); sA=srcc(xsA,gts)
    rclaim=pearson(xs21,xsA)
    sdr=(sd21/sdA) if sdA>0 else float('nan')
    print(f"{feat:<14}{n:>6}{m21:>10.3f}{mA:>10.3f}{sd21:>9.3f}{sdA:>9.3f}{sdr:>11.3f}{nu21:>8}{nuA:>8}{s21:>10.3f}{sA:>9.3f}{rclaim:>10.3f}")
    hl21.append(s21); hlA.append(sA)
print("-"*len(hdr))
print(f"HEADLINE mean SRCC (5-feat, on intersections): v21={sum(hl21)/len(hl21):.3f}  A={sum(hlA)/len(hlA):.3f}")
