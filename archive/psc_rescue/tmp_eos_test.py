import sys, torch, json
sys.path.insert(0,'src')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
from train import _tokenize_with_eos
d = json.load(open('/ocean/projects/cis260125p/shared/data/descriptions_observability_test.json'))
targets = list(d.values())[:3]
ids, attn = _tokenize_with_eos(tok, targets, 512, 'cpu')
eos = tok.eos_token_id; pad = tok.pad_token_id
print(f'eos_id={eos} pad_id={pad}  (pad==eos? {pad==eos})')
allok=True
for i in range(len(targets)):
    n=int(attn[i].sum()); last=ids[i,n-1].item()
    labels=ids[i].clone(); labels[attn[i]==0]=-100
    sup = labels[n-1].item()==eos
    ok=(last==eos) and sup; allok&=ok
    print(f'  target{i}: content_len={n} EOS_appended={last==eos} EOS_supervised(not -100)={sup} -> {"OK" if ok else "BUG"}  last5={tok.decode(ids[i,max(0,n-5):n])!r}')
print('EOS VERDICT:', 'CORRECT (appended + supervised in loss)' if allok else 'BUG')
