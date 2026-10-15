"""Five-seed H0f Counterfactual Delta Prototype representation screen."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.counterfactual_delta_prototype import CounterfactualDeltaPrototypeProbe,TARGET_INDICES
from experiments.run_route_a_controlled_lt import _mean_ci,_read,_write

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def _seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def _state_hash(model):
 h=hashlib.sha256()
 for value in model.state_dict().values(): h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def _load(path):
 with h5py.File(path,'r') as h:
  raw=np.asarray(h['iq_complex']); labels=np.asarray(h['label_selected'],dtype=np.uint8); pair=np.asarray(h['pair_id']).reshape(-1).astype(np.int64); target=np.asarray(h['pair_target_index']).reshape(-1).astype(np.int64)-1; side=np.asarray(h['pair_side']).reshape(-1).astype(np.int64)
 x=np.stack([raw['real'],raw['imag']],axis=1).astype(np.float32);x/=np.sqrt(np.maximum(np.square(x).sum(1).mean(1),1e-12))[:,None,None]
 if x.shape!=(4800,2,3000) or labels.shape!=(4800,6): raise ValueError('H0f data contract')
 return x,labels,pair,target,side
def _table(pair,target,side):
 result=[]
 for identity in np.unique(pair):
  ix=np.flatnonzero(pair==identity);neg=int(ix[side[ix]==0][0]);pos=int(ix[side[ix]==1][0]);result.append((int(identity),int(target[pos]),neg,pos))
 return result
def _select(rows,identities):
 allowed=set(int(x) for x in identities);return [r for r in rows if r[0] in allowed]
def _batches(rows,size,seed):
 for start in range(0,len(rows),size):
  order=np.random.default_rng(seed).permutation(len(rows));yield [rows[int(i)] for i in order[start:start+size]]
def _predict(model,x,ids,device,batch):
 chunks=[];model.eval()
 with torch.no_grad():
  for begin in range(0,len(ids),batch): chunks.append(model(torch.from_numpy(x[ids[begin:begin+batch]]).to(device),return_features=True))
 return {key:np.concatenate([entry[key].cpu().numpy() for entry in chunks]) for key in ('logits','embedding')}
def _early_bce(model,x,y,rows,device):
 ids=np.asarray([item for row in rows for item in row[2:]],dtype=np.int64);model.eval()
 with torch.no_grad():return float(F.binary_cross_entropy_with_logits(model(torch.from_numpy(x[ids]).to(device))[:,TARGET_INDICES],torch.from_numpy(y[ids][:,TARGET_INDICES]).float().to(device)).cpu())
def _fit(name,seed,state,x,y,fit,early,evaluation,args,root,device):
 root.mkdir(parents=True,exist_ok=True);receipt=root/'run_receipt.json'
 if receipt.exists() and _read(receipt).get('status')=='completed':return _read(root/'metrics.json')
 model=CounterfactualDeltaPrototypeProbe(width=args.width,projection_dim=args.projection_dim).to(device);model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay);best=(float('inf'),0,None);history=[];binding=torch.tensor([0,1,2] if name!='B1_binding_permuted' else [1,2,0],device=device)
 for epoch in range(1,args.epochs+1):
  model.train();losses=[]
  for rows in _batches(fit,args.batch_pairs,seed+101*epoch):
   neg=np.asarray([r[2] for r in rows]);pos=np.asarray([r[3] for r in rows]);ids=np.r_[neg,pos];out=model(torch.from_numpy(x[ids]).to(device),return_features=True);loss=F.binary_cross_entropy_with_logits(out['logits'],torch.from_numpy(y[ids]).float().to(device))
   delta=F.normalize(out['embedding'][len(rows):]-out['embedding'][:len(rows)],dim=1);target_pos=torch.tensor([TARGET_INDICES.index(r[1]) for r in rows],device=device);delta_logits=delta@out['prototypes'].T/args.temperature
   if name!='B0':loss=loss+args.delta_weight*F.cross_entropy(delta_logits,binding[target_pos])
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach().cpu()))
  early_value=_early_bce(model,x,y,early,device);history.append({'epoch':epoch,'fit_loss':float(np.mean(losses)),'early_target_bce':early_value})
  if early_value<best[0]:best=(early_value,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
 model.load_state_dict(best[2]);model.eval();eval_ids=np.asarray([value for row in evaluation for value in row[2:]],dtype=np.int64);predict=_predict(model,x,eval_ids,device,args.eval_batch);pos_map={int(v):i for i,v in enumerate(eval_ids)};metrics={}
 for target in TARGET_INDICES:
  rows=[r for r in evaluation if r[1]==target];neg=np.asarray([pos_map[r[2]] for r in rows]);pos=np.asarray([pos_map[r[3]] for r in rows]);truth=np.r_[np.ones(len(pos)),np.zeros(len(neg))];score=np.r_[predict['logits'][pos,target],predict['logits'][neg,target]];metrics[str(target)]={'weak_auroc':float(roc_auc_score(truth,score)),'weak_aupr':float(average_precision_score(truth,score)),'all_auroc':float(roc_auc_score(y[eval_ids,target],predict['logits'][:,target])),'weak_positive_count':len(pos),'matched_negative_count':len(neg)}
 delta_rows=[];target_positions=[]
 for row in evaluation:
  delta_rows.append(predict['embedding'][pos_map[row[3]]]-predict['embedding'][pos_map[row[2]]]);target_positions.append(TARGET_INDICES.index(row[1]))
 delta=F.normalize(torch.from_numpy(np.asarray(delta_rows)).to(device),dim=1)
 with torch.no_grad():
  accuracy=float(((delta@F.normalize(model.prototypes,dim=1).T).argmax(1)==torch.tensor(target_positions,device=device)).float().mean().cpu())
 result={'status':'completed','name':name,'seed':seed,'best_epoch':best[1],'best_early_target_bce':best[0],'parameter_count':sum(p.numel() for p in model.parameters()),'metrics':metrics,'evaluation_true_binding_delta_accuracy':accuracy,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0f_used':False}
 torch.save({'state_dict':best[2],'seed':seed,'name':name,'architecture':'CounterfactualDeltaPrototypeProbe'},root/'best_model.pth');np.savez_compressed(root/'evaluation_predictions.npz',sample_indices=eval_ids,logits=predict['logits'],embeddings=predict['embedding'],targets=y[eval_ids]);_write(root/'history.json',history);_write(root/'metrics.json',result);_write(receipt,{k:result[k] for k in ('status','name','seed','parameter_count','locked_test_used','test_v2_used','test_v3_used','test_h0f_used')});return result
def _comparison(rows,control):
 answer=[]
 for seed in sorted({r['seed'] for r in rows}):
  c=next(r for r in rows if r['seed']==seed and r['name']=='C_CDP');b=next(r for r in rows if r['seed']==seed and r['name']==control);target={str(t):{'weak_auroc':c['metrics'][str(t)]['weak_auroc']-b['metrics'][str(t)]['weak_auroc'],'all_auroc':c['metrics'][str(t)]['all_auroc']-b['metrics'][str(t)]['all_auroc']} for t in TARGET_INDICES};answer.append({'seed':seed,'targets':target,'mean_weak_auroc':float(np.mean([target[str(t)]['weak_auroc'] for t in TARGET_INDICES])),'candidate_delta_accuracy':c['evaluation_true_binding_delta_accuracy']})
 return answer
def run(args):
 audit,seal=_read(args.audit),_read(args.test_v3_manifest)
 if not audit.get('identity_audit_passed') or audit.get('test_files_opened') or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('audit/seal failure')
 if _sha(args.data_path).lower()!=audit['sha256'].lower():raise ValueError('source hash changed')
 role=np.load(args.roles,allow_pickle=False);x,y,pair,target,side=_load(args.data_path);table=_table(pair,target,side);fit,early,evaluation=(_select(table,role[k]) for k in ('fit','early_stop','evaluation'))
 if args.smoke:args.seeds,args.epochs=[args.seeds[0]],args.smoke_epochs
 device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'));rows=[];initial=[]
 for seed in args.seeds:
  _seed(seed);template=CounterfactualDeltaPrototypeProbe(width=args.width,projection_dim=args.projection_dim);state={k:v.detach().cpu().clone() for k,v in template.state_dict().items()};initial.append({'seed':seed,'state_sha256':_state_hash(template)})
  for name in ('B0','B1_binding_permuted','C_CDP'):rows.append(_fit(name,seed,state,x,y,fit,early,evaluation,args,args.output_root/'runs'/f'seed_{seed}'/name,device))
 comparisons={control:_comparison(rows,control) for control in ('B0','B1_binding_permuted')};aggregate={}
 for control,items in comparisons.items():aggregate[control]={'mean_weak_auroc':_mean_ci([r['mean_weak_auroc'] for r in items],args.bootstrap_seed),'per_target':{str(t):_mean_ci([r['targets'][str(t)]['weak_auroc'] for r in items],args.bootstrap_seed+t+31) for t in TARGET_INDICES},'delta_accuracy':_mean_ci([r['candidate_delta_accuracy'] for r in items],args.bootstrap_seed+71)}
 checks={}
 for control,items in comparisons.items():
  value=aggregate[control];checks.update({f'{control}_weak_auroc_ge_0p015':value['mean_weak_auroc']['mean']>=.015,f'{control}_weak_auroc_ci_positive':value['mean_weak_auroc']['bootstrap_95_ci'][0]>0,f'{control}_weak_auroc_positive_4of5':sum(r['mean_weak_auroc']>0 for r in items)>=4,f'{control}_every_target_nonnegative':all(x['mean']>=0 for x in value['per_target'].values()),f'{control}_all_auroc_safe':all(r['targets'][str(t)]['all_auroc']>=-.01 for r in items for t in TARGET_INDICES)})
 checks.update({'candidate_delta_accuracy_ge_0p60':aggregate['B0']['delta_accuracy']['mean']>=.60,'matched_parameter_count':len({r['parameter_count'] for r in rows})==1,'matched_initialization':len({r['state_sha256'] for r in initial})==len(initial),'test_v3_sealed':True});passed=bool(all(checks.values())) if not args.smoke else False
 out={'status':'completed','experiment':'route_a_h0f_counterfactual_delta_prototype','evidence_role':'smoke' if args.smoke else 'fresh_train_only_counterfactual_delta_prototype_identifiability','data':{'path':str(args.data_path),'sha256':audit['sha256'],'roles':{'fit':len(fit),'early_stop':len(early),'evaluation':len(evaluation)},'initialization':initial},'architecture':{'model':'CounterfactualDeltaPrototypeProbe','width':args.width,'projection_dim':args.projection_dim,'parameter_count':rows[0]['parameter_count'],'inference_inputs':'single I/Q only'},'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,'weight_decay':args.weight_decay,'epochs':args.epochs,'batch_pairs':args.batch_pairs,'delta_weight':args.delta_weight,'temperature':args.temperature},'rows':rows,'comparisons':comparisons,'aggregates':aggregate,'gate':{'checks':checks,'passed':passed,'decision':'permit_separate_fresh_six_label_cdp_integration_design' if passed else 'stop_counterfactual_delta_prototype'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0f_used':False};args.output_root.mkdir(parents=True,exist_ok=True);_write(args.output_root/'route_a_h0f_counterfactual_delta_prototype_summary.json',out);_write(args.output_root/'run_receipt.json',{'status':'completed','data_sha256':audit['sha256'],'gate_passed':passed,'decision':out['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0f_used':False});print(json.dumps({'passed':passed,'checks':checks,'output':str(args.output_root)},ensure_ascii=False,indent=2));return out
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_counterfactual_delta_h0f\counterfactual_delta_pairs_train.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h0f_counterfactual_delta_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_h0f_counterfactual_delta_generation/role_pairs.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output-root',type=Path,default=ROOT/'outputs/tapc/route_a_h0f_counterfactual_delta_prototype');p.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041,2042,2043]);p.add_argument('--bootstrap-seed',type=int,default=2026080221);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-pairs',type=int,default=48);p.add_argument('--eval-batch',type=int,default=256);p.add_argument('--width',type=int,default=16);p.add_argument('--projection-dim',type=int,default=16);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--delta-weight',type=float,default=.2);p.add_argument('--temperature',type=float,default=.1);p.add_argument('--device',default=None);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-epochs',type=int,default=2);return p.parse_args()
if __name__=='__main__':run(parse_args())
