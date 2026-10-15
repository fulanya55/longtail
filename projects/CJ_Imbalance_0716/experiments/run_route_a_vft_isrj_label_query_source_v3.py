"""Frozen F2 VFT label-query versus position-pooling source-v3 screen."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels,read_parameters
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import _extract as eg
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_model_from_state
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals
from experiments.run_route_a_vft_isrj_raw_source_diagnostic import _hard_source_view
class Head(nn.Module):
 def __init__(self,c,t,kind):super().__init__();self.kind=kind;self.key=nn.Parameter(torch.zeros(c));self.fc=nn.Linear(2*c,1)
 def forward(self,x):
  z=(x*torch.softmax(self.key,0)[None,:,None]).sum(1) if self.kind=='query' else (x*self.key[None,:,None]).mean(2);z=z if self.kind=='mean' else (x*torch.softmax(z,1)[:,None,:]).sum(2);return self.fc(torch.cat([z,x.mean(2)],1)).squeeze(1)
def extract(m,x,d,b):
 out=[];m.eval()
 with torch.inference_mode():
  for i in range(0,len(x),b):out.append(m.forward_temporal_features(torch.from_numpy(x[i:i+b]).to(d)).cpu())
 return torch.cat(out)
def scores(kind,tr_x,tr_y,va_x,va_y,te_x,te_y,seed,d):
 torch.manual_seed(seed);m=Head(tr_x.shape[1],tr_x.shape[2],kind).to(d);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);best=(-1,None);stale=0
 for ep in range(30):
  m.train();order=torch.randperm(len(tr_y));
  for ix in order.split(64):opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(m(tr_x[ix].to(d)),tr_y[ix].float().to(d));loss.backward();opt.step()
  m.eval();
  with torch.inference_mode():v=m(va_x.to(d)).cpu().numpy()
  a=roc_auc_score(va_y.numpy(),v)
  if a>best[0]:best=(a,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()});stale=0
  else:stale+=1
  if stale>=5:break
 m.load_state_dict(best[1]);m.eval()
 with torch.inference_mode():s=m(te_x.to(d)).cpu().numpy()
 return {'auroc':float(roc_auc_score(te_y.numpy(),s)),'aupr':float(average_precision_score(te_y.numpy(),s)),'epochs':ep+1,'parameters':sum(p.numel() for p in m.parameters())}
def run(a):
 audit=json.loads(a.audit.read_text(encoding='utf8'));seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'))
 if not audit['passed'] or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('audit/seal failed')
 y=read_labels(a.data);p,_=read_parameters(a.data);weak=compute_weak_component_mask(y,_component_power(p),10.);man=json.loads(a.manifest.read_text(encoding='utf8'));d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[]
 for seed in a.seeds:
  r=np.load(a.role_root/f'seed_{seed}'/'role_indices.npz');state=torch.load(man['seed_artifacts'][str(seed)]['focal_checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict'];m=_load_model_from_state(a.f2_root/'runs/F2_DFMS_Full'/f'seed_{seed}',d,state);z={};g={}
  for role in ('fit','validation','early'):
   ix=r[role];x=_load_signals(a.data,ix,a.batch);t=extract(m,x,d,a.batch);sel,tar,_=_hard_source_view(y[ix],weak[ix]);z[role]=(t[sel],torch.from_numpy(tar.astype(np.float32)));g[role]=(eg(m,x,d,a.batch)[sel],tar)
  del m
  gr=RidgeClassifier(alpha=10.,class_weight='balanced').fit(g['fit'][0],g['fit'][1]);gs=gr.decision_function(g['early'][0]);rec={'seed':seed,'global':{'auroc':float(roc_auc_score(g['early'][1],gs)),'aupr':float(average_precision_score(g['early'][1],gs))}}
  for kind in ('query','mean'):
   rec[kind]=scores(kind,z['fit'][0],z['fit'][1],z['validation'][0],z['validation'][1],z['early'][0],z['early'][1],seed,d)
  rows.append(rec)
 qg=np.array([r['query']['auroc']-r['global']['auroc'] for r in rows]);qp=np.array([r['query']['auroc']-r['mean']['auroc'] for r in rows]);gates={'query_global_ge_0p010_4of5':bool(qg.mean()>=.01 and(qg>0).sum()>=4),'query_mean_ge_0p015_4of5':bool(qp.mean()>=.015 and(qp>0).sum()>=4),'test_v3_sealed':True};o={'status':'completed','experiment':'RouteA_VFTISRJ_LabelQuerySourceV3','rows':rows,'query_minus_global':{'mean':float(qg.mean()),'positive':int((qg>0).sum())},'query_minus_mean':{'mean':float(qp.mean()),'positive':int((qp>0).sum())},'gates':gates,'passed':bool(all(gates.values())),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(o,ensure_ascii=False,indent=2));return o
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_vft_isrj_token_source_v3\closedset_6class_train.mat'));p.add_argument('--role-root',type=Path,default=ROOT/'outputs/tapc/route_a_vft_isrj_token_source_v3_generation');p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_vft_isrj_token_source_v3_generation/source_v1_identity_role_audit.json');p.add_argument('--manifest',type=Path,default=ROOT/'frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json');p.add_argument('--f2-root',type=Path,default=ROOT/'outputs/tapc/route_a_dfms_iq_structure_screen');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_vft_isrj_label_query_source_v3/summary.json');p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]);p.add_argument('--batch',type=int,default=64);run(p.parse_args())
if __name__=='__main__':main()
