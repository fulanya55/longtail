"""Frozen F2 token-order identifiability screen on fresh source-v2."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels,read_parameters
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import _extract as _extract_global
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_model_from_state
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals
from experiments.run_route_a_vft_isrj_raw_source_diagnostic import _hard_source_view
def temporal(model,x,device,batch=64):
 out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(x),batch):out.append(model.forward_temporal_features(torch.from_numpy(x[i:i+batch]).to(device)).cpu())
 return torch.cat(out).numpy()
def metric(train_x,train_y,test_x,test_y):
 m=RidgeClassifier(alpha=10.,class_weight='balanced').fit(train_x,train_y);s=m.decision_function(test_x);return {'auroc':float(roc_auc_score(test_y,s)),'aupr':float(average_precision_score(test_y,s))}
def run(a):
 audit=json.loads(a.audit.read_text(encoding='utf8'));seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'))
 if not audit['passed'] or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('audit/seal failed')
 data=a.data_path; labels=read_labels(data);p,_=read_parameters(data);weak=compute_weak_component_mask(labels,_component_power(p),10.);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');manifest=json.loads(a.frozen_manifest.read_text(encoding='utf8'));rows=[]
 for seed in a.seeds:
  roles=np.load(a.role_root/f'seed_{seed}'/'role_indices.npz'); feat={}
  state=torch.load(manifest['seed_artifacts'][str(seed)]['focal_checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict'];model=_load_model_from_state(a.f2_root/'runs/F2_DFMS_Full'/f'seed_{seed}',device,state)
  for role in ('fit','validation','early'):
   idx=roles[role];x=_load_signals(data,idx,a.batch);g=_extract_global(model,x,device,a.batch);t=temporal(model,x,device,a.batch); b=F.adaptive_avg_pool1d(torch.from_numpy(t),12).flatten(1).numpy(); rng=np.random.default_rng(20803000+seed+len(role)); order=np.argsort(rng.random(t.shape),axis=-1);ts=np.take_along_axis(t,order,axis=-1); sh=F.adaptive_avg_pool1d(torch.from_numpy(ts),12).flatten(1).numpy();sel,y,_=_hard_source_view(labels[idx],weak[idx]);feat[role]=(g[sel],b[sel],sh[sel],y)
  del model
  rec={'seed':seed,'roles':{}}
  for name,pos in [('Global128',0),('Token12',1),('ShuffledToken12',2)]:
   rec['roles'][name]={r:metric(feat['fit'][pos],feat['fit'][3],feat[r][pos],feat[r][3]) for r in ('validation','early')}
  rows.append(rec)
 delta=np.array([r['roles']['Token12']['early']['auroc']-r['roles']['Global128']['early']['auroc'] for r in rows]);shuffle=np.array([r['roles']['Token12']['early']['auroc']-r['roles']['ShuffledToken12']['early']['auroc'] for r in rows]);gap=np.array([abs(r['roles']['Token12']['validation']['auroc']-r['roles']['Token12']['early']['auroc']) for r in rows]);gates={'token_vs_global_ge_0p010_4of5':bool(delta.mean()>=.01 and (delta>0).sum()>=4),'token_vs_shuffle_ge_0p020_4of5':bool(shuffle.mean()>=.02 and (shuffle>0).sum()>=4),'token_val_early_gap_lt_0p150':bool(gap.max()<.15),'test_v3_sealed':True};o={'status':'completed','experiment':'RouteA_VFTISRJ_TokenSourceV2','evidence_role':'fresh_frozen_token_identifiability','rows':rows,'early_token_minus_global':{'mean':float(delta.mean()),'positive_seeds':int((delta>0).sum())},'early_token_minus_shuffle':{'mean':float(shuffle.mean()),'positive_seeds':int((shuffle>0).sum())},'gates':gates,'passed':bool(all(gates.values())),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(o,ensure_ascii=False,indent=2));return o
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_vft_isrj_token_source_v2\closedset_6class_train.mat'));p.add_argument('--role-root',type=Path,default=ROOT/'outputs/tapc/route_a_vft_isrj_token_source_v2_generation');p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_vft_isrj_token_source_v2_generation/source_v1_identity_role_audit.json');p.add_argument('--frozen-manifest',type=Path,default=ROOT/'frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json');p.add_argument('--f2-root',type=Path,default=ROOT/'outputs/tapc/route_a_dfms_iq_structure_screen');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_vft_isrj_token_source_v2/summary.json');p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]);p.add_argument('--batch',type=int,default=64);run(p.parse_args())
if __name__=='__main__':main()
