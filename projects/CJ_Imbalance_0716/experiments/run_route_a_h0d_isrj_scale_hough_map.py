"""H0d: fixed I/Q scale-Hough copy bank, with no width oracle or predictor."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import h5py, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.delay_block_self_similarity import ScaleHoughDelayBlockMap, ScaleHoughReplicaProbe
from experiments.run_route_a_controlled_lt import _mean_ci,_read,_write
from experiments.run_route_a_h0_isrj_delay_block_self_similarity import _sha,_signals
REPLICA=54

def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def _metrics(y,p):
    span=max(float(y.max()-y.min()),1e-12)
    return {'r2':float(r2_score(y,p)),'spearman':float(spearmanr(y,p).statistic),'nmae_observed_range':float(np.abs(y-p).mean()/span),'mae_count':float(np.abs(y-p).mean())}
def _features(signal,device,batch_size):
    transform=ScaleHoughDelayBlockMap().to(device).eval(); result=[]
    with torch.no_grad():
        for start in range(0,len(signal),batch_size): result.append(transform(torch.from_numpy(signal[start:start+batch_size]).to(device)).cpu().numpy())
    return np.concatenate(result).astype(np.float32,copy=False)
def _fit(name,seed,xfit,yfit,xearly,yearly,xeval,yeval,mean,std,args,root,device):
    root.mkdir(parents=True,exist_ok=True); receipt=root/'run_receipt.json'
    if receipt.exists() and _read(receipt).get('status')=='completed': return _read(root/'metrics.json')
    _seed(seed); model=ScaleHoughReplicaProbe().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); loss_fn=nn.SmoothL1Loss(beta=1.0)
    gen=torch.Generator().manual_seed(seed+1900); loader=DataLoader(TensorDataset(torch.from_numpy(xfit),torch.from_numpy((yfit-mean)/std)),batch_size=args.batch_size,shuffle=True,generator=gen,num_workers=0)
    early_x,early_y=torch.from_numpy(xearly).to(device),torch.from_numpy((yearly-mean)/std).to(device); best_loss,best_epoch,best,hist=float('inf'),0,None,[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for feature,target in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model.forward_map(feature.to(device)),target.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): val=float(loss_fn(model.forward_map(early_x),early_y).cpu())
        hist.append({'epoch':epoch,'fit_smooth_l1':float(np.mean(losses)),'early_smooth_l1':val})
        if val<best_loss: best_loss,best_epoch,best=val,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): prediction=model.forward_map(torch.from_numpy(xeval).to(device)).cpu().numpy()*std+mean
    result=_metrics(yeval,prediction); result.update({'status':'completed','name':name,'seed':int(seed),'best_epoch':int(best_epoch),'best_early_smooth_l1':best_loss,'parameter_count':int(sum(v.numel() for v in model.parameters())),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0d_used':False})
    torch.save({'state_dict':best,'seed':int(seed),'name':name,'architecture':'ScaleHoughReplicaProbe'},root/'best_model.pth'); np.savez_compressed(root/'predictions.npz',true_replicas=yeval,prediction=prediction); _write(root/'history.json',hist); _write(root/'metrics.json',result); _write(receipt,{k:result[k] for k in ('status','name','seed','parameter_count','locked_test_used','test_v2_used','test_v3_used','test_h0d_used')}); return result
def run(args):
    audit,seal=_read(args.audit),_read(args.test_v3_manifest)
    if not audit.get('identity_audit_passed') or audit.get('test_files_opened') or any(seal['test_v3'][x] for x in ('access_authorized','analysis_allowed','used')): raise ValueError('audit/test seal failed')
    if _sha(args.data_path).lower()!=audit['sha256'].lower(): raise ValueError('source hash changed')
    roles=np.load(args.roles,allow_pickle=False); ids={x:np.asarray(roles[x],dtype=np.int64) for x in ('fit','early_stop','evaluation')}; all_ids=np.sort(np.concatenate(tuple(ids.values())))
    with h5py.File(args.data_path,'r') as h: values=np.asarray(h['parameter_values'],dtype=np.float64).T
    signal=_signals(args.data_path,all_ids); pos={int(v):i for i,v in enumerate(all_ids)}; index={name:np.asarray([pos[int(v)] for v in vals],dtype=np.int64) for name,vals in ids.items()}; target=values[all_ids,REPLICA].astype(np.float32); fit,early,ev=index['fit'],index['early_stop'],index['evaluation']; mean,std=float(target[fit].mean()),float(target[fit].std())
    if std<=0: raise ValueError('target contract failed')
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu')); cache=args.output_root/'scale_hough_map_features.npz'
    if cache.exists(): feature=np.load(cache,allow_pickle=False)['feature']; map_sha=_sha(cache)
    else:
        feature=_features(signal,device,args.feature_batch_size); args.output_root.mkdir(parents=True,exist_ok=True); np.savez_compressed(cache,feature=feature,receiver_noise_seed=all_ids); map_sha=_sha(cache)
    if feature.shape!=(len(all_ids),2,88,12) or not np.isfinite(feature).all(): raise ValueError('fixed map contract failed')
    rng=np.random.default_rng(args.permutation_seed); permuted=target.copy(); permuted[fit]=target[fit][rng.permutation(len(fit))]; permuted[early]=target[early][rng.permutation(len(early))]
    if args.smoke: args.seeds,args.epochs=[args.seeds[0]],args.smoke_epochs
    rows=[]
    for seed in args.seeds:
        base=args.output_root/'runs'/f'seed_{seed}'
        rows.append(_fit('H_real',seed,feature[fit],target[fit],feature[early],target[early],feature[ev],target[ev],mean,std,args,base/'H_real',device))
        rows.append(_fit('H_perm',seed,feature[fit],permuted[fit],feature[early],permuted[early],feature[ev],target[ev],mean,std,args,base/'H_perm',device))
    paired=[]
    for seed in args.seeds:
        real=next(r for r in rows if r['seed']==seed and r['name']=='H_real'); perm=next(r for r in rows if r['seed']==seed and r['name']=='H_perm'); paired.append({'seed':int(seed),'real_minus_perm_r2':real['r2']-perm['r2'],'real_minus_perm_spearman':real['spearman']-perm['spearman']})
    aggregate={key:_mean_ci([r[key] for r in paired],args.permutation_seed+i) for i,key in enumerate(('real_minus_perm_r2','real_minus_perm_spearman'))}; real_rows=[r for r in rows if r['name']=='H_real']
    checks={'matched_parameter_count':len({r['parameter_count'] for r in rows})==1,'mean_r2_ge_0p20':float(np.mean([r['r2'] for r in real_rows]))>=.20,'mean_spearman_ge_0p40':float(np.mean([r['spearman'] for r in real_rows]))>=.40,'real_minus_perm_r2_ge_0p20':aggregate['real_minus_perm_r2']['mean']>=.20,'real_minus_perm_r2_ci_positive':aggregate['real_minus_perm_r2']['bootstrap_95_ci'][0]>0,'positive_4of5':sum(r['real_minus_perm_r2']>0 for r in paired)>=4,'test_v3_sealed':True}; passed=bool(all(checks.values())) if not args.smoke else False
    report={'status':'completed','experiment':'route_a_h0d_isrj_scale_hough_map','evidence_role':'smoke' if args.smoke else 'fresh_train_only_inference_ready_replica_identifiability','data':{'path':str(args.data_path),'sha256':audit['sha256'],'roles':{k:int(len(v)) for k,v in ids.items()}},'representation':{'map':'fixed complex scale-Hough delay-block correlation','width_hypotheses_samples':list(range(100,301,20)),'multipliers':list(range(1,9)),'blocks':12,'feature_shape':[2,88,12],'feature_cache':str(cache),'feature_cache_sha256':map_sha,'learned_map_parameters':0},'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,'weight_decay':args.weight_decay,'loss':'SmoothL1','batch_size':args.batch_size,'epochs':args.epochs},'rows':rows,'paired':{'per_seed':paired,'aggregate':aggregate},'gate':{'checks':checks,'passed':passed,'decision':'permit_separate_h1_isrj_expert_design' if passed else 'stop_scale_hough_map'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0d_used':False}; args.output_root.mkdir(parents=True,exist_ok=True); _write(args.output_root/'route_a_h0d_isrj_scale_hough_map_summary.json',report); _write(args.output_root/'run_receipt.json',{'status':'completed','data_sha256':audit['sha256'],'feature_cache_sha256':map_sha,'gate_passed':passed,'decision':report['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0d_used':False}); print(json.dumps({'passed':passed,'checks':checks,'output':str(args.output_root)},ensure_ascii=False,indent=2)); return report
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_scale_hough_h0d\closedset_6class_train.mat')); p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h0d_isrj_scale_hough_generation/identity_audit.json'); p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_h0d_isrj_scale_hough_generation/role_indices.npz'); p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json'); p.add_argument('--output-root',type=Path,default=ROOT/'outputs/tapc/route_a_h0d_isrj_scale_hough_map'); p.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041,2042,2043]); p.add_argument('--permutation-seed',type=int,default=2026080217); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--feature-batch-size',type=int,default=32); p.add_argument('--learning-rate',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--device',default=None); p.add_argument('--smoke',action='store_true'); p.add_argument('--smoke-epochs',type=int,default=2); return p.parse_args()
if __name__=='__main__': run(parse_args())
