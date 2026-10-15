"""Fresh five-seed TS-CLE raw-I/Q expert screen with frozen global safety path."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import h5py,numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels,read_parameters
from cjlt.models.target_specific_complex_evidence import TARGET_INDICES,TargetSpecificComplexEvidenceClassifier
from experiments.run_route_a_controlled_lt import _mean_ci,_read,_write
from experiments.run_route_a_development_v16_isrj_width_probe import _load_signals
from experiments.run_route_a_development_v2_h5 import _sha256
from experiments.run_route_a_development_v2_shared_gradient import _component_power

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def batches(n,batch,seed):
 return torch.randperm(n,generator=torch.Generator().manual_seed(seed)).split(batch)
def predict(model,x,d,batch,evidence=False):
 model.eval();rows=[];entropy=[]
 with torch.no_grad():
  for i in range(0,len(x),batch):
   z=model(torch.from_numpy(x[i:i+batch]).to(d),return_evidence=evidence);rows.append((z['logits'] if evidence else z).cpu().numpy())
   if evidence:entropy.append((-(torch.softmax(z['local_evidence'],dim=-1)*torch.log_softmax(z['local_evidence'],dim=-1)).sum(-1)).cpu().numpy())
 return np.concatenate(rows), (np.concatenate(entropy) if evidence else None)
def train_global(model,xfit,yfit,xearly,yearly,d,args,seed):
 for n,p in model.named_parameters():p.requires_grad=('stem' in n or 'global_classifier' in n)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.global_lr,weight_decay=args.weight_decay);best=(float('inf'),0,None)
 for ep in range(1,args.global_epochs+1):
  model.train()
  for ix in batches(len(xfit),args.batch_size,seed+ep):opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(torch.from_numpy(xfit[ix]).to(d)),torch.from_numpy(yfit[ix]).to(d));loss.backward();opt.step()
  with torch.no_grad():v=float(F.binary_cross_entropy_with_logits(model(torch.from_numpy(xearly).to(d)),torch.from_numpy(yearly).to(d)).cpu())
  if v<best[0]:best=(v,ep,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
 model.load_state_dict(best[2]);return best
def train_local(model,xfit,yfit,xearly,yearly,d,args,seed):
 for n,p in model.named_parameters():p.requires_grad=('local_' in n or 'evidence_heads' in n)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.local_lr,weight_decay=args.weight_decay);best=(float('inf'),0,None);hist=[]
 targets=list(TARGET_INDICES)
 for ep in range(1,args.local_epochs+1):
  model.train();losses=[]
  for ix in batches(len(xfit),args.batch_size,seed+700+ep):opt.zero_grad(set_to_none=True);z=model(torch.from_numpy(xfit[ix]).to(d));loss=F.binary_cross_entropy_with_logits(z[:,targets],torch.from_numpy(yfit[ix][:,targets]).to(d));loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
  with torch.no_grad():v=float(F.binary_cross_entropy_with_logits(model(torch.from_numpy(xearly).to(d))[:,targets],torch.from_numpy(yearly[:,targets]).to(d)).cpu())
  hist.append({'epoch':ep,'fit_target_bce':float(np.mean(losses)),'early_target_bce':v})
  if v<best[0]:best=(v,ep,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
 model.load_state_dict(best[2]);return best,hist
def composition_view(labels,weak,t):
 other=np.delete(labels,t,axis=1);codes=(other*np.array([1,2,4,8,16],dtype=np.uint8)).sum(1);valid=set(codes[labels[:,t]==0]).intersection(set(codes[labels[:,t]==1]));pos=(labels[:,t]==1)&weak[:,t]&np.isin(codes,list(valid));neg=(labels[:,t]==0)&np.isin(codes,list(valid));return pos,neg
def metric(labels,weak,scores,t):
 pos,neg=composition_view(labels,weak,t);y=np.r_[np.ones(pos.sum(),dtype=np.uint8),np.zeros(neg.sum(),dtype=np.uint8)];s=np.r_[scores[pos,t],scores[neg,t]];strong=(labels[:,t]==1)&~weak[:,t]&np.isin(np.delete(labels,t,axis=1)@np.array([1,2,4,8,16]),list(set((np.delete(labels,t,axis=1)@np.array([1,2,4,8,16]))[labels[:,t]==0])));ys=np.r_[np.ones(strong.sum(),dtype=np.uint8),np.zeros(neg.sum(),dtype=np.uint8)];ss=np.r_[scores[strong,t],scores[neg,t]];return {'weak_count':int(pos.sum()),'negative_count':int(neg.sum()),'weak_auroc':float(roc_auc_score(y,s)),'weak_aupr':float(average_precision_score(y,s)),'strong_auroc':float(roc_auc_score(ys,ss))}
def one(args,x,y,weak,ids,seed,d):
 root=args.output_root/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True);fit,early,ev=(ids[k] for k in ('fit','early_stop','evaluation'));seed_all(seed);base=TargetSpecificComplexEvidenceClassifier(channels=args.channels,topk=args.topk).to(d);g=train_global(base,x[fit],y[fit],x[early],y[early],d,args,seed);global_state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()};variants={};global_pred,_=predict(base,x[ev],d,args.eval_batch)
 for name,control in (('TS_CLE',False),('CyclicSourceControl',True)):
  model=TargetSpecificComplexEvidenceClassifier(channels=args.channels,topk=args.topk,source_control=control).to(d);model.load_state_dict(global_state);best,hist=train_local(model,x[fit],y[fit],x[early],y[early],d,args,seed);score,entropy=predict(model,x[ev],d,args.eval_batch,True);np.savez_compressed(root/f'{name}_predictions.npz',sample_ids=ev,logits=score,weak=weak[ev].astype(np.uint8),targets=y[ev].astype(np.uint8));torch.save({'state_dict':model.state_dict(),'global_best_epoch':g[1],'local_best_epoch':best[1],'global_early_bce':g[0],'local_early_bce':best[0]},root/f'{name}_best.pth');_write(root/f'{name}_history.json',hist);variants[name]={'metrics':{str(t):metric(y[ev],weak[ev],score,t) for t in TARGET_INDICES},'entropy_mean':[float(q) for q in entropy.mean(0)],'global_max_abs_delta':float(np.abs(score-global_pred).max())}
 a=np.load(root/'TS_CLE_predictions.npz')['logits'];b=np.load(root/'CyclicSourceControl_predictions.npz')['logits'];safe=(0,1,2);safe_exact=bool(np.array_equal(a[:,safe],b[:,safe]));return {'seed':seed,'global_best_epoch':g[1],'variants':variants,'safe_logits_exact':safe_exact,'parameter_count':sum(p.numel() for p in base.parameters()),'local_trainable_parameters':sum(p.numel() for n,p in base.named_parameters() if ('local_' in n or 'evidence_heads' in n)),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v25_used':False}
def run(a):
 audit=_read(a.audit);seal=_read(a.test_v3_manifest)
 if not audit['identity_audit_passed'] or audit['test_files_opened'] or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('audit or sealed test-v3 check failed')
 if _sha256(a.data_path).lower()!=audit['sha256'].lower():raise ValueError('v25 source hash changed')
 ids={k:np.asarray(np.load(a.roles,allow_pickle=False)[k],dtype=np.int64) for k in ('fit','early_stop','evaluation')};y=read_labels(a.data_path).astype(np.float32);p,_=read_parameters(a.data_path);weak=compute_weak_component_mask(y.astype(np.uint8),_component_power(p),a.weak_threshold_db);d=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'));allids=np.sort(np.concatenate(tuple(ids.values())));raw=_load_signals(a.data_path,allids);x=np.zeros((len(y),2,raw.shape[-1]),dtype=np.float32);x[allids]=raw;a.output_root.mkdir(parents=True,exist_ok=True);rows=[one(a,x,y,weak,ids,s,d) for s in a.seeds]
 deltas=[];strong=[]
 for r in rows:
  vals=[]
  for t in TARGET_INDICES:vals.append(r['variants']['TS_CLE']['metrics'][str(t)]['weak_auroc']-r['variants']['CyclicSourceControl']['metrics'][str(t)]['weak_auroc']);strong.extend([r['variants']['TS_CLE']['metrics'][str(t)]['strong_auroc']-r['variants']['CyclicSourceControl']['metrics'][str(t)]['strong_auroc'] for t in TARGET_INDICES])
  deltas.append(vals)
 mean_seed=np.asarray(deltas).mean(1);per_target={str(t):_mean_ci([r['variants']['TS_CLE']['metrics'][str(t)]['weak_auroc']-r['variants']['CyclicSourceControl']['metrics'][str(t)]['weak_auroc'] for r in rows],25000+t) for t in TARGET_INDICES};macro=_mean_ci(mean_seed,25100);checks={'macro_weak_auroc_ge_0p015':bool(macro['mean']>=.015),'macro_weak_auroc_ci_positive':bool(macro['bootstrap_95_ci'][0]>0),'macro_positive_4of5':bool(sum(mean_seed>0)>=4),'every_target_mean_nonnegative':bool(all(x['mean']>=0 for x in per_target.values())),'strong_safe_every_seed':bool(all(x>=-.01 for x in strong)),'safe_logits_exact':bool(all(r['safe_logits_exact'] for r in rows)),'matched_parameter_contract':bool(len({r['parameter_count'] for r in rows})==1 and len({r['local_trainable_parameters'] for r in rows})==1),'test_v3_sealed':True};passed=bool(all(checks.values()));out={'status':'completed','experiment':'route_a_v25_target_specific_complex_evidence_probe','rows':rows,'macro_weak_auroc_delta':macro,'per_target_weak_auroc_delta':per_target,'gate':{'checks':checks,'passed':passed,'decision':'permit_fresh_six_label_integration_design' if passed else 'stop_target_specific_complex_evidence'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v25_used':False};_write(a.output_root/'route_a_v25_target_specific_complex_evidence_probe_summary.json',out);_write(a.output_root/'run_receipt.json',{'status':'completed','gate_passed':passed,'decision':out['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v25_used':False});print(json.dumps({'passed':passed,'checks':checks,'output':str(a.output_root)},ensure_ascii=False,indent=2))
def args():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v25_target_complex_evidence\closedset_6class_train.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_v25_target_complex_evidence_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_v25_target_complex_evidence_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output-root',type=Path,default=ROOT/'outputs/tapc/route_a_v25_target_specific_complex_evidence_probe');p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]);p.add_argument('--device',default=None);p.add_argument('--channels',type=int,default=8);p.add_argument('--topk',type=int,default=16);p.add_argument('--global-epochs',type=int,default=20);p.add_argument('--local-epochs',type=int,default=20);p.add_argument('--global-lr',type=float,default=3e-4);p.add_argument('--local-lr',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--eval-batch',type=int,default=128);p.add_argument('--weak-threshold-db',type=float,default=10.);return p.parse_args()
if __name__=='__main__':run(args())
