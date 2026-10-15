"""V3 limited representation candidate on frozen B0-DB features.

This is deliberately a cached-feature head experiment: the B0 checkpoint is
frozen, stage4 magnitude statistics receive a rank-4 residual adapter, and
six independent label-conditioned evidence projections are trained with a
feature-nearest hard-negative ranking term. No power/weak-mask/pair metadata
is supplied to the model.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DATA=Path(r'E:\LJQ_code\干扰数据\dual-axis-frequency-representation-v3')
VH=DATA/'sawfm_level_05__jnr_30dB/closedset_6class_train.mat'
VL=DATA/'sawfm_level_03__jnr_20dB/closedset_6class_train.mat'
AUDIT=ROOT/'outputs/route_a_frequency_representation_v3/stage1_data_audit'
BASE=ROOT/'outputs/route_a_frequency_representation_v3/stage2_formal'
OUT=ROOT/'outputs/route_a_frequency_representation_v3/stage3_candidate'
SEEDS=(261021,261022,261023,261024,261025)
REGIMES=('FB','LabelIF20','ComboIF20')
CLASS_NAMES=('FMB','SawFM','SinFM','MFT','VFT','ISRJ')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read_labels(path):
    from cjlt.data.v73_dataset import read_labels
    return read_labels(path)
def load_roles():
    z=np.load(AUDIT/'role_indices.npz'); return {k:z[k].astype(np.int64) for k in z.files}
def load_manifest(regime):
    if regime=='FB': return np.asarray(json.loads((AUDIT/'frequency_balanced_manifest.json').read_text())['selected_indices'],dtype=np.int64)
    p=AUDIT/('label_if20_manifest.json' if regime=='LabelIF20' else 'combination_if20_manifest.json')
    return np.asarray(json.loads(p.read_text())['selected_indices'],dtype=np.int64)

class ResidualEvidence(nn.Module):
    def __init__(self, baseline_w, baseline_b):
        super().__init__(); d=128; r=4
        self.u=nn.Parameter(torch.zeros(d,r)); self.v=nn.Parameter(torch.zeros(r,d))
        self.proj=nn.ModuleList([nn.Sequential(nn.Linear(d,16),nn.ReLU(),nn.Linear(16,1)) for _ in range(6)])
        for block in self.proj:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
        self.register_buffer('bw',torch.as_tensor(baseline_w,dtype=torch.float32)); self.register_buffer('bb',torch.as_tensor(baseline_b,dtype=torch.float32))
        self.scale=0.25
    def forward(self,g,s):
        adapted=g + self.scale*torch.tanh(s @ self.v.t() @ self.u.t())
        base=F.linear(adapted,self.bw,self.bb)
        residual=torch.cat([m(adapted) for m in self.proj],dim=1)
        return base + self.scale*torch.tanh(residual), adapted

def balanced_bce(logits,y):
    losses=[]
    for j in range(y.shape[1]):
        pos=y[:,j]>0.5; neg=~pos
        if pos.any() and neg.any():
            losses.append(0.5*F.binary_cross_entropy_with_logits(logits[pos,j],y[pos,j])+0.5*F.binary_cross_entropy_with_logits(logits[neg,j],y[neg,j]))
        else: losses.append(F.binary_cross_entropy_with_logits(logits[:,j],y[:,j]))
    return torch.stack(losses).mean()
def rank_loss(logits,feat,y,margin=0.2):
    vals=[]
    norm=F.normalize(feat.detach(),dim=1); sim=norm@norm.t(); n=len(y)
    for j in range(y.shape[1]):
        pos=torch.where(y[:,j]>0.5)[0]; neg=torch.where(y[:,j]<0.5)[0]
        if len(pos)==0 or len(neg)==0: continue
        for i in pos:
            k=neg[torch.argmax(sim[i,neg])]
            vals.append(F.softplus(torch.as_tensor(margin,device=logits.device)-logits[i,j]+logits[k,j]))
    return torch.stack(vals).mean() if vals else logits.sum()*0.0
def metrics(y,logits):
    per={n:float(average_precision_score(y[:,j],logits[:,j])) for j,n in enumerate(CLASS_NAMES)}
    return {'macro_aupr':float(np.mean(list(per.values()))),'per_label_aupr':per}

def extract(model,path,indices,device,batch=128):
    from cjlt.data.v73_dataset import V73SignalDataset
    ds=V73SignalDataset(path,indices,class_mode='six',normalization='complex_rms')
    dl=DataLoader(ds,batch_size=batch,num_workers=0,pin_memory=True)
    gs=[]; ss=[]; ys=[]
    with torch.no_grad():
        for b in dl:
            sig=b['signal'].to(device,non_blocking=True)
            x=model.forward_structure_maps(sig)
            g=x['global']; amp=torch.sqrt(x['stage4_real']**2+x['stage4_imag']**2+1e-8)
            s=torch.cat([amp.mean(-1),amp.std(-1)],dim=1)
            gs.append(g.cpu().numpy()); ss.append(s.cpu().numpy()); ys.append(b['label'].numpy())
    ds.close()
    return np.concatenate(gs),np.concatenate(ss),np.concatenate(ys)

def cache_for(seed,regime,device,indices):
    from cjlt.models.backbone_adapter import build_backbone
    ck=BASE/f'seed_{seed}'/regime/'F1_B0_DB.pt'
    model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(device)
    state=torch.load(ck,map_location='cpu')['model_state_dict']; model.load_state_dict(state,strict=True); model.eval()
    out={}
    for name,path in (('VH',VH),('VL',VL)):
        out[name]=extract(model,path,indices,device)
    return out,sha(ck)

def run_cell(seed,regime,device,roles,epochs=60):
    t=time.time(); selected=load_manifest(regime); opt=selected; early=roles['early']; ev=roles['evaluation']
    indices=np.concatenate([opt,early,ev]); order={int(x):i for i,x in enumerate(indices)}
    cache,cksha=cache_for(seed,regime,device,indices)
    pos={k:np.asarray([order[int(x)] for x in arr],dtype=np.int64) for k,arr in [('opt',opt),('early',early),('eval',ev)]}
    g,s,y=cache['VH']; bidx=pos['opt']; eidx=pos['early']; vidx=pos['eval']
    # extract F1's frozen linear head from checkpoint model state
    st=torch.load(BASE/f'seed_{seed}'/regime/'F1_B0_DB.pt',map_location='cpu')['model_state_dict'];
    bw=st['classifier.1.weight'].numpy(); bb=st['classifier.1.bias'].numpy()
    torch.manual_seed(seed); head=ResidualEvidence(bw,bb).to(device)
    optm=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    tx=torch.as_tensor(g[bidx],device=device); sx=torch.as_tensor(s[bidx],device=device); ty=torch.as_tensor(y[bidx],dtype=torch.float32,device=device)
    ex=torch.as_tensor(g[eidx],device=device); es=torch.as_tensor(s[eidx],device=device); ey=torch.as_tensor(y[eidx],dtype=torch.float32,device=device)
    best=None; stale=0; hist=[]
    gen=torch.Generator(device=device).manual_seed(seed+7000)
    for ep in range(1,epochs+1):
        head.train(); perm=torch.randperm(len(tx),generator=gen,device=device); losses=[]
        for start in range(0,len(perm),256):
            ii=perm[start:start+256]; logits,af=head(tx[ii],sx[ii]); l=balanced_bce(logits,ty[ii])+0.20*rank_loss(logits,af,ty[ii])+5e-4*((logits-(F.linear(tx[ii],head.bw,head.bb)))**2).mean()+1e-4*(head.u.pow(2).mean()+head.v.pow(2).mean()); optm.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.0); optm.step(); losses.append(float(l.detach().cpu()))
        head.eval();
        with torch.no_grad(): el=head(ex,es)[0].cpu().numpy()
        m=metrics(ey.cpu().numpy(),el); key=(m['macro_aupr'],m['per_label_aupr']['SawFM']); hist.append({'epoch':ep,'loss':float(np.mean(losses)),'early_macro_aupr':key[0],'early_sawfm_aupr':key[1]})
        if best is None or key>best[0]: best=(key,{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},ep); stale=0
        else: stale+=1
        if stale>=12: break
    head.load_state_dict(best[1]); head.eval();
    rows={}
    for view in ('VH','VL'):
        gg,ss,yy=cache[view]; with_g=torch.as_tensor(gg[vidx],device=device); with_s=torch.as_tensor(ss[vidx],device=device)
        with torch.no_grad(): ll=head(with_g,with_s)[0].cpu().numpy()
        rows[view]=metrics(yy[vidx],ll)
    p=OUT/f'seed_{seed}'/regime; p.mkdir(parents=True,exist_ok=True)
    write(p/'history.json',hist); torch.save({'state_dict':head.state_dict(),'best_epoch':best[2]},p/'candidate.pt')
    result={'seed':seed,'regime':regime,'status':'completed','checkpoint_sha256':sha(p/'candidate.pt'),'base_checkpoint_sha256':cksha,'best_epoch':best[2],'by_view':rows,'parameter_count':sum(x.numel() for x in head.parameters() if x.requires_grad),'elapsed_seconds':time.time()-t,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False}
    write(p/'summary.json',result); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS)); ap.add_argument('--regimes',nargs='+',default=list(REGIMES)); ap.add_argument('--epochs',type=int,default=60); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device); roles=load_roles(); rows=[]
    for seed in args.seeds:
        for regime in args.regimes:
            rows.append(run_cell(seed,regime,device,roles,args.epochs)); print(json.dumps(rows[-1],ensure_ascii=False))
    write(OUT/'run_index.json',{'status':'completed','runs':rows,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False})
if __name__=='__main__': main()
