"""H0c: substitute frozen v16 predicted width for H0b oracle delay coordinates."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import h5py, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.delay_block_self_similarity import WidthNormalizedDelayBlockReplicaProbe
from experiments.run_route_a_controlled_lt import _mean_ci,_read,_write
from experiments.run_route_a_development_v16_isrj_width_probe import WidthProbe
from experiments.run_route_a_h0_isrj_delay_block_self_similarity import _sha,_signals
WIDTH,REPLICA=53,54

def _seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def _metrics(y,p):
 span=max(float(y.max()-y.min()),1e-12);return {'r2':float(r2_score(y,p)),'spearman':float(spearmanr(y,p).statistic),'nmae_observed_range':float(np.abs(y-p).mean()/span),'mae_count':float(np.abs(y-p).mean())}
def _file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def _predict_width(signal,seed,args,device):
 checkpoint=args.v16_root/'runs'/'W_real'/f'seed_{seed}'/'best_model.pth'; normalizer=args.v16_root/'width_normalizer.json'
 if not checkpoint.is_file() or not normalizer.is_file():raise FileNotFoundError('frozen v16 dependency missing')
 state=torch.load(checkpoint,map_location='cpu',weights_only=False); normal=json.loads(normalizer.read_text(encoding='utf8')); model=WidthProbe(width=16).to(device);model.load_state_dict(state['state_dict']);model.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(signal),args.width_batch_size):out.append(model(torch.from_numpy(signal[start:start+args.width_batch_size]).to(device)).cpu().numpy())
 value=np.concatenate(out)*float(normal['std_seconds'])+float(normal['mean_seconds']);samples=np.rint(value*1e7).astype(np.int64).clip(100,300)
 return samples,{'checkpoint':str(checkpoint),'checkpoint_sha256':_file_sha(checkpoint),'normalizer':str(normalizer),'normalizer_sha256':_file_sha(normalizer),'raw_predicted_width_range_us':[float(value.min()*1e6),float(value.max()*1e6)],'clamped_width_samples_range':[int(samples.min()),int(samples.max())]}
def _fit(name,seed,xfit,wfit,yfit,xearly,wearly,yearly,xeval,weval,yeval,mean,std,args,root,device):
 root.mkdir(parents=True,exist_ok=True);receipt=root/'run_receipt.json'
 if receipt.exists() and _read(receipt).get('status')=='completed':return _read(root/'metrics.json')
 _seed(seed);model=WidthNormalizedDelayBlockReplicaProbe().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay);loss_fn=nn.SmoothL1Loss(beta=1.0);gen=torch.Generator().manual_seed(seed+1900);loader=DataLoader(TensorDataset(torch.from_numpy(xfit),torch.from_numpy(wfit),torch.from_numpy((yfit-mean)/std)),batch_size=args.batch_size,shuffle=True,generator=gen,num_workers=0)
 early_x,early_w,early_y=torch.from_numpy(xearly).to(device),torch.from_numpy(wearly).to(device),torch.from_numpy((yearly-mean)/std).to(device);best_loss,best_epoch,best,hist=float('inf'),0,None,[]
 for epoch in range(1,args.epochs+1):
  model.train();losses=[]
  for signal,width,target in loader:
   opt.zero_grad(set_to_none=True);loss=loss_fn(model(signal.to(device),width.to(device)),target.to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach().cpu()))
  model.eval()
  with torch.no_grad():val=float(loss_fn(model(early_x,early_w),early_y).cpu())
  hist.append({'epoch':epoch,'fit_smooth_l1':float(np.mean(losses)),'early_smooth_l1':val})
  if val<best_loss:best_loss,best_epoch,best=val,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(best);model.eval()
 with torch.no_grad():prediction=model(torch.from_numpy(xeval).to(device),torch.from_numpy(weval).to(device)).cpu().numpy()*std+mean
 result=_metrics(yeval,prediction);result.update({'status':'completed','name':name,'seed':int(seed),'best_epoch':int(best_epoch),'best_early_smooth_l1':best_loss,'parameter_count':int(sum(v.numel() for v in model.parameters())),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0c_used':False});torch.save({'state_dict':best,'seed':int(seed),'name':name,'architecture':'WidthNormalizedDelayBlockReplicaProbe'},root/'best_model.pth');np.savez_compressed(root/'predictions.npz',true_replicas=yeval,prediction=prediction);_write(root/'history.json',hist);_write(root/'metrics.json',result);_write(receipt,{k:result[k] for k in ('status','name','seed','parameter_count','locked_test_used','test_v2_used','test_v3_used','test_h0c_used')});return result
def run(args):
 audit,seal=_read(args.audit),_read(args.test_v3_manifest)
 if not audit.get('identity_audit_passed') or audit.get('test_files_opened') or any(seal['test_v3'][x] for x in ('access_authorized','analysis_allowed','used')):raise ValueError('audit/test seal failed')
 if _sha(args.data_path).lower()!=audit['sha256'].lower():raise ValueError('source hash changed')
 roles=np.load(args.roles,allow_pickle=False);ids={x:np.asarray(roles[x],dtype=np.int64) for x in ('fit','early_stop','evaluation')};all_ids=np.sort(np.concatenate(tuple(ids.values())))
 with h5py.File(args.data_path,'r') as h:values=np.asarray(h['parameter_values'],dtype=np.float64).T
 signal=_signals(args.data_path,all_ids);pos={int(v):i for i,v in enumerate(all_ids)};index={name:np.asarray([pos[int(v)] for v in vals],dtype=np.int64) for name,vals in ids.items()};target=values[all_ids,REPLICA].astype(np.float32);oracle=np.rint(values[all_ids,WIDTH]*1e7).astype(np.int64);fit,early,ev=index['fit'],index['early_stop'],index['evaluation'];mean,std=float(target[fit].mean()),float(target[fit].std())
 if std<=0 or oracle.min()<100 or oracle.max()>300:raise ValueError('target/oracle contract failed')
 rng=np.random.default_rng(args.permutation_seed);permuted=target.copy();permuted[fit]=target[fit][rng.permutation(len(fit))];permuted[early]=target[early][rng.permutation(len(early))]
 if args.smoke:args.seeds,args.epochs=[args.seeds[0]],args.smoke_epochs
 device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'));rows=[];frozen=[]
 for seed in args.seeds:
  predicted,dependency=_predict_width(signal,seed,args,device);frozen.append({'seed':int(seed),**dependency});base=args.output_root/'runs'/f'seed_{seed}'
  rows.append(_fit('O_real',seed,signal[fit],oracle[fit],target[fit],signal[early],oracle[early],target[early],signal[ev],oracle[ev],target[ev],mean,std,args,base/'O_real',device))
  rows.append(_fit('P_real',seed,signal[fit],predicted[fit],target[fit],signal[early],predicted[early],target[early],signal[ev],predicted[ev],target[ev],mean,std,args,base/'P_real',device))
  rows.append(_fit('P_perm',seed,signal[fit],predicted[fit],permuted[fit],signal[early],predicted[early],permuted[early],signal[ev],predicted[ev],target[ev],mean,std,args,base/'P_perm',device))
 paired=[]
 for seed in args.seeds:
  o=next(r for r in rows if r['seed']==seed and r['name']=='O_real');p=next(r for r in rows if r['seed']==seed and r['name']=='P_real');q=next(r for r in rows if r['seed']==seed and r['name']=='P_perm');paired.append({'seed':int(seed),'pred_minus_perm_r2':p['r2']-q['r2'],'pred_minus_oracle_r2':p['r2']-o['r2'],'pred_minus_perm_spearman':p['spearman']-q['spearman']})
 agg={key:_mean_ci([r[key] for r in paired],args.permutation_seed+i) for i,key in enumerate(('pred_minus_perm_r2','pred_minus_oracle_r2','pred_minus_perm_spearman'))};p_rows=[r for r in rows if r['name']=='P_real'];checks={'matched_parameter_count':len({r['parameter_count'] for r in rows})==1,'pred_mean_r2_ge_0p20':float(np.mean([r['r2'] for r in p_rows]))>=.20,'pred_mean_spearman_ge_0p40':float(np.mean([r['spearman'] for r in p_rows]))>=.40,'pred_minus_perm_r2_ge_0p20':agg['pred_minus_perm_r2']['mean']>=.20,'pred_minus_perm_r2_ci_positive':agg['pred_minus_perm_r2']['bootstrap_95_ci'][0]>0,'pred_positive_4of5':sum(r['pred_minus_perm_r2']>0 for r in paired)>=4,'pred_minus_oracle_r2_ge_neg0p15':agg['pred_minus_oracle_r2']['mean']>=-.15,'pred_minus_oracle_r2_ci_ge_neg0p20':agg['pred_minus_oracle_r2']['bootstrap_95_ci'][0]>=-.20,'test_v3_sealed':True};passed=bool(all(checks.values())) if not args.smoke else False
 report={'status':'completed','experiment':'route_a_h0c_isrj_predicted_width_substitution','evidence_role':'smoke' if args.smoke else 'fresh_train_only_frozen_predicted_width_replica_identifiability','data':{'path':str(args.data_path),'sha256':audit['sha256'],'roles':{k:int(len(v)) for k,v in ids.items()}},'frozen_v16_dependencies':frozen,'representation':{'map':'width-normalized complex delay-block correlation','multipliers':list(range(1,9)),'blocks':12,'predicted_width_clamp_samples':[100,300]},'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,'weight_decay':args.weight_decay,'loss':'SmoothL1','batch_size':args.batch_size,'epochs':args.epochs},'rows':rows,'paired':{'per_seed':paired,'aggregate':agg},'gate':{'checks':checks,'passed':passed,'decision':'permit_separate_h1_isrj_expert_design' if passed else 'stop_predicted_width_substitution'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0c_used':False};args.output_root.mkdir(parents=True,exist_ok=True);_write(args.output_root/'route_a_h0c_isrj_predicted_width_substitution_summary.json',report);_write(args.output_root/'run_receipt.json',{'status':'completed','data_sha256':audit['sha256'],'gate_passed':passed,'decision':report['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0c_used':False});print(json.dumps({'passed':passed,'checks':checks,'output':str(args.output_root)},ensure_ascii=False,indent=2));return report
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_predicted_width_h0c\closedset_6class_train.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h0c_isrj_predicted_width_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_h0c_isrj_predicted_width_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--v16-root',type=Path,default=ROOT/'outputs/tapc/route_a_development_v16_isrj_width_probe');p.add_argument('--output-root',type=Path,default=ROOT/'outputs/tapc/route_a_h0c_isrj_predicted_width_substitution');p.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041,2042,2043]);p.add_argument('--permutation-seed',type=int,default=2026080216);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--width-batch-size',type=int,default=128);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--device',default=None);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-epochs',type=int,default=2);return p.parse_args()
if __name__=='__main__':run(parse_args())
