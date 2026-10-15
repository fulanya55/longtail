"""Five-seed H0e single-view CVMC screen on fresh counterfactual pairs."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.counterfactual_visibility import CounterfactualVisibilityProbe
from experiments.run_route_a_controlled_lt import _mean_ci,_read,_write
TARGETS=(3,4,5)
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def _seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def _state_hash(model):
 h=hashlib.sha256()
 for value in model.state_dict().values():h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def _load(path):
 with h5py.File(path,'r') as h:
  raw=np.asarray(h['iq_complex']); x=np.stack([raw['real'],raw['imag']],axis=1).astype(np.float32); y=np.asarray(h['label_selected'],dtype=np.uint8); pair=np.asarray(h['pair_id']).reshape(-1).astype(np.int64); target=np.asarray(h['pair_target_index']).reshape(-1).astype(np.int64)-1; side=np.asarray(h['pair_side']).reshape(-1).astype(np.int64)
 x/=np.sqrt(np.maximum(np.square(x).sum(1).mean(1),1e-12))[:,None,None]
 if x.shape!=(4800,2,3000) or y.shape!=(4800,6):raise ValueError('H0e I/Q contract failed')
 return x,y,pair,target,side
def _pair_table(pair,target,side):
 table=[]
 for p in np.unique(pair):
  ix=np.flatnonzero(pair==p);neg=int(ix[side[ix]==0][0]);pos=int(ix[side[ix]==1][0]);table.append((int(p),int(target[pos]),neg,pos))
 return table
def _role_rows(table,selected):return [row for row in table if row[0] in set(int(x) for x in selected)]
def _batch(rows,batch,seed):
 order=np.random.default_rng(seed).permutation(len(rows))
 for part in range(0,len(order),batch):yield [rows[int(i)] for i in order[part:part+batch]]
def _shuffle_neg(rows,epoch,seed):
 answer={}
 for target in TARGETS:
  local=[row for row in rows if row[1]==target]; perm=np.random.default_rng(seed+epoch*31+target).permutation(len(local));
  if any(int(perm[i])==i for i in range(len(local))):perm=np.roll(perm,1)
  for i,row in enumerate(local):answer[row[0]]=local[int(perm[i])][2]
 return answer
def _logits(model,x,index,device):
 with torch.no_grad():return model(torch.from_numpy(x[np.asarray(index,dtype=np.int64)]).to(device)).cpu()
def _early_loss(model,x,y,rows,device):
 ids=np.asarray([v for row in rows for v in row[2:]],dtype=np.int64); model.eval(); return float(F.binary_cross_entropy_with_logits(_logits(model,x,ids,device)[:,TARGETS],torch.from_numpy(y[ids][:,TARGETS]).float()).item())
def _fit(name,seed,state,x,y,fit,early,evaluation,args,root,device):
 root.mkdir(parents=True,exist_ok=True); receipt=root/'run_receipt.json'
 if receipt.exists() and _read(receipt).get('status')=='completed':return _read(root/'metrics.json')
 model=CounterfactualVisibilityProbe(width=args.width).to(device);model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay);hist=[];best=(float('inf'),0,None)
 for epoch in range(1,args.epochs+1):
  model.train();losses=[]; shuffled=_shuffle_neg(fit,epoch,seed+500) if name=='B1_shuffle' else None
  for rows in _batch(fit,args.batch_pairs,seed+epoch*101):
   pos=np.asarray([r[3] for r in rows],dtype=np.int64);neg=np.asarray([r[2] if shuffled is None else shuffled[r[0]] for r in rows],dtype=np.int64); ids=np.r_[neg,pos]; logits=model(torch.from_numpy(x[ids]).to(device));loss=F.binary_cross_entropy_with_logits(logits,torch.from_numpy(y[ids]).float().to(device))
   if name!='B0':
    targets=torch.tensor([r[1] for r in rows],device=device);margin=F.softplus(args.margin-(logits[len(rows):,targets]-logits[:len(rows),targets])).mean();loss=loss+args.margin_weight*margin
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach().cpu()))
  early_value=_early_loss(model,x,y,early,device);hist.append({'epoch':epoch,'fit_loss':float(np.mean(losses)),'early_target_bce':early_value})
  if early_value<best[0]:best=(early_value,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
 model.load_state_dict(best[2]);model.eval();ev_ids=np.asarray([v for row in evaluation for v in row[2:]],dtype=np.int64);scores=np.zeros((len(x),6),np.float32)
 for start in range(0,len(ev_ids),args.eval_batch):scores[ev_ids[start:start+args.eval_batch]]=_logits(model,x,ev_ids[start:start+args.eval_batch],device).numpy()
 metrics={};margin_success={}
 for t in TARGETS:
  rows=[r for r in evaluation if r[1]==t];neg=np.asarray([r[2] for r in rows]);pos=np.asarray([r[3] for r in rows]);truth=np.r_[np.ones(len(pos)),np.zeros(len(neg))];score=np.r_[scores[pos,t],scores[neg,t]];alltruth=y[ev_ids,t]
  metrics[str(t)]={'weak_auroc':float(roc_auc_score(truth,score)),'weak_aupr':float(average_precision_score(truth,score)),'all_auroc':float(roc_auc_score(alltruth,scores[ev_ids,t])),'weak_positive_count':int(len(pos)),'matched_negative_count':int(len(neg))};margin_success[str(t)]=float((scores[pos,t]-scores[neg,t]>=args.margin).mean())
 result={'status':'completed','name':name,'seed':int(seed),'best_epoch':int(best[1]),'best_early_target_bce':float(best[0]),'parameter_count':int(sum(p.numel() for p in model.parameters())),'metrics':metrics,'margin_success':margin_success,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0e_used':False}
 torch.save({'state_dict':best[2],'seed':seed,'name':name,'architecture':'CounterfactualVisibilityProbe'},root/'best_model.pth');np.savez_compressed(root/'evaluation_predictions.npz',sample_indices=ev_ids,logits=scores[ev_ids],targets=y[ev_ids]);_write(root/'history.json',hist);_write(root/'metrics.json',result);_write(receipt,{k:result[k] for k in ('status','name','seed','parameter_count','locked_test_used','test_v2_used','test_v3_used','test_h0e_used')});return result
def _compare(rows,left,right):
 output=[]
 for seed in sorted({r['seed'] for r in rows}):
  a=next(r for r in rows if r['seed']==seed and r['name']==left);b=next(r for r in rows if r['seed']==seed and r['name']==right);targets={str(t):{'weak_auroc':a['metrics'][str(t)]['weak_auroc']-b['metrics'][str(t)]['weak_auroc'],'all_auroc':a['metrics'][str(t)]['all_auroc']-b['metrics'][str(t)]['all_auroc'],'margin_success':a['margin_success'][str(t)]-b['margin_success'][str(t)]} for t in TARGETS};output.append({'seed':seed,'targets':targets,'mean_weak_auroc':float(np.mean([targets[str(t)]['weak_auroc'] for t in TARGETS])),'mean_margin_success':float(np.mean([targets[str(t)]['margin_success'] for t in TARGETS]))})
 return output
def run(a):
 audit,seal=_read(a.audit),_read(a.test_v3_manifest)
 if not audit.get('identity_audit_passed') or audit.get('test_files_opened') or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('audit/seal failed')
 if _sha(a.data_path).lower()!=audit['sha256'].lower():raise ValueError('data hash changed')
 roles=np.load(a.roles,allow_pickle=False);x,y,pair,target,side=_load(a.data_path);table=_pair_table(pair,target,side);fit,early,evaluation=(_role_rows(table,roles[k]) for k in ('fit','early_stop','evaluation'))
 if a.smoke:a.seeds,a.epochs=[a.seeds[0]],a.smoke_epochs
 device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'));rows=[];initial=[]
 for seed in a.seeds:
  _seed(seed);template=CounterfactualVisibilityProbe(width=a.width);state={k:v.detach().cpu().clone() for k,v in template.state_dict().items()};initial.append({'seed':seed,'initial_state_sha256':_state_hash(template)})
  for name in ('B0','B1_shuffle','C_CVMC'):rows.append(_fit(name,seed,state,x,y,fit,early,evaluation,a,a.output_root/'runs'/f'seed_{seed}'/name,device))
 comparisons={key:_compare(rows,'C_CVMC',key) for key in ('B0','B1_shuffle')};aggregates={}
 for control,items in comparisons.items():aggregates[control]={'mean_weak_auroc':_mean_ci([r['mean_weak_auroc'] for r in items],a.bootstrap_seed),'mean_margin_success':_mean_ci([r['mean_margin_success'] for r in items],a.bootstrap_seed+17),'per_target':{str(t):_mean_ci([r['targets'][str(t)]['weak_auroc'] for r in items],a.bootstrap_seed+31+t) for t in TARGETS}}
 checks={}
 for control in ('B0','B1_shuffle'):
  g=aggregates[control];items=comparisons[control];checks[f'{control}_weak_auroc_ge_0p015']=g['mean_weak_auroc']['mean']>=.015;checks[f'{control}_weak_auroc_ci_positive']=g['mean_weak_auroc']['bootstrap_95_ci'][0]>0;checks[f'{control}_weak_auroc_positive_4of5']=sum(r['mean_weak_auroc']>0 for r in items)>=4;checks[f'{control}_every_target_nonnegative']=all(v['mean']>=0 for v in g['per_target'].values());checks[f'{control}_margin_success_ge_0p03']=g['mean_margin_success']['mean']>=.03;checks[f'{control}_margin_positive_4of5']=sum(r['mean_margin_success']>0 for r in items)>=4;checks[f'{control}_all_auroc_safe']=all(r['targets'][str(t)]['all_auroc']>=-.01 for r in items for t in TARGETS)
 checks.update({'matched_parameter_count':len({r['parameter_count'] for r in rows})==1,'matched_initialization':len({r['initial_state_sha256'] for r in initial})==len(initial),'test_v3_sealed':True});passed=bool(all(checks.values())) if not a.smoke else False
 out={'status':'completed','experiment':'route_a_h0e_counterfactual_visibility_margin','evidence_role':'smoke' if a.smoke else 'fresh_train_only_counterfactual_pair_identifiability','data':{'path':str(a.data_path),'sha256':audit['sha256'],'pair_roles':{k:len(v) for k,v in {'fit':fit,'early_stop':early,'evaluation':evaluation}.items()},'initialization':initial},'architecture':{'model':'CounterfactualVisibilityProbe','width':a.width,'parameter_count':rows[0]['parameter_count'],'inference_inputs':'single I/Q only'},'optimizer':{'name':'AdamW','learning_rate':a.learning_rate,'weight_decay':a.weight_decay,'epochs':a.epochs,'batch_pairs':a.batch_pairs,'bce':'six-label BCE','margin':'softplus(0.5-(z_plus-z_minus))','margin_weight':a.margin_weight},'rows':rows,'comparisons':comparisons,'aggregates':aggregates,'gate':{'checks':checks,'passed':passed,'decision':'permit_separate_fresh_six_label_cvmc_integration_design' if passed else 'stop_counterfactual_visibility_margin'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0e_used':False};a.output_root.mkdir(parents=True,exist_ok=True);_write(a.output_root/'route_a_h0e_counterfactual_visibility_margin_summary.json',out);_write(a.output_root/'run_receipt.json',{'status':'completed','data_sha256':audit['sha256'],'gate_passed':passed,'decision':out['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0e_used':False});print(json.dumps({'passed':passed,'checks':checks,'output':str(a.output_root)},ensure_ascii=False,indent=2));return out
def args():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_counterfactual_visibility_h0e\counterfactual_visibility_pairs_train.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/role_pairs.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output-root',type=Path,default=ROOT/'outputs/tapc/route_a_h0e_counterfactual_visibility_margin');p.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041,2042,2043]);p.add_argument('--bootstrap-seed',type=int,default=2026080220);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-pairs',type=int,default=48);p.add_argument('--eval-batch',type=int,default=256);p.add_argument('--width',type=int,default=16);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--margin',type=float,default=.5);p.add_argument('--margin-weight',type=float,default=.5);p.add_argument('--device',default=None);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-epochs',type=int,default=2);return p.parse_args()
if __name__=='__main__':run(args())
