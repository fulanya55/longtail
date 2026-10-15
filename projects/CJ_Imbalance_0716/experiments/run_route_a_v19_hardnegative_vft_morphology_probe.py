"""Fresh identity-isolated nonlinear VFT-vs-ISRJ morphology probe."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels,read_parameters  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci,_read,_write  # noqa:E402
from experiments.run_route_a_development_v16_isrj_width_probe import _load_signals  # noqa:E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa:E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft  # noqa:E402
from experiments.run_route_a_development_v2_h5 import _sha256  # noqa:E402

VFT,ISRJ=4,5

class MorphologyProbe(nn.Module):
    def __init__(self,dim=268,hidden=64,dropout=.1):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def _set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def _scores(model,x,device,batch=256):
    model.eval();rows=[]
    with torch.no_grad():
        for s in range(0,len(x),batch): rows.append(model(torch.from_numpy(x[s:s+batch]).to(device)).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)

def _train(name,xfit,yfit,xearly,yearly,seed,device,args,root):
    _set_seed(seed);model=MorphologyProbe(xfit.shape[1],args.hidden,args.dropout).to(device);optim=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    gen=torch.Generator();gen.manual_seed(seed+701);best={'loss':float('inf'),'epoch':0,'state':None};history=[]
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(xfit),generator=gen).numpy();losses=[]
        for s in range(0,len(order),args.batch_size):
            idx=order[s:s+args.batch_size];xb=torch.from_numpy(xfit[idx]).to(device);yb=torch.from_numpy(yfit[idx]).to(device)
            optim.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(xb),yb);loss.backward();optim.step();losses.append(float(loss.detach().cpu()))
        with torch.no_grad(): val=F.binary_cross_entropy_with_logits(model(torch.from_numpy(xearly).to(device)),torch.from_numpy(yearly).to(device)).item()
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'early_loss':float(val)})
        if val < best['loss']-1e-10: best={'loss':float(val),'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    model.load_state_dict(best['state']);out=root/name;out.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'best_epoch':best['epoch'],'best_early_loss':best['loss'],'architecture':{'input_dim':int(xfit.shape[1]),'hidden':int(args.hidden),'dropout':float(args.dropout)}},out/'best_model.pth');_write(out/'history.json',history);return model,best,history

def _metric(y,score): return {'auroc':float(roc_auc_score(y,score)),'aupr':float(average_precision_score(y,score)),'count':int(len(y))}

def _weak_metric(labels,weak,ids,score):
    positive=(labels[ids,VFT]==1)&weak[ids,VFT];negative=labels[ids,VFT]==0
    if positive.sum()<20 or negative.sum()<20:return None
    y=np.r_[np.ones(positive.sum(),dtype=np.uint8),np.zeros(negative.sum(),dtype=np.uint8)];s=np.r_[score[positive],score[negative]]
    return {'positive_count':int(positive.sum()),'negative_count':int(negative.sum()),'auroc':float(roc_auc_score(y,s)),'aupr':float(average_precision_score(y,s))}

def _load(args):
    audit=_read(args.audit);seal=_read(args.test_v3_manifest)
    if not audit.get('identity_audit_passed') or audit.get('test_files_opened'):raise ValueError('v19 audit failed')
    if any(seal['test_v3'][key] for key in ('access_authorized','analysis_allowed','used')):raise ValueError('test-v3 is not sealed')
    if _sha256(args.data_path).lower()!=audit['sha256'].lower():raise ValueError('v19 source hash changed')
    roles=np.load(args.roles,allow_pickle=False);ids={k:np.asarray(roles[k],dtype=np.int64) for k in ('fit','early_stop','evaluation')}
    labels=read_labels(args.data_path);params,_=read_parameters(args.data_path);weak=compute_weak_component_mask(labels,_component_power(params),args.weak_threshold_db)
    with h5py.File(args.data_path,'r') as h:receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64)
    if any(np.intersect1d(ids[a],ids[b]).size for a in ids for b in ids if a<b):raise ValueError('v19 role identity leakage')
    for name,index in ids.items():
        if not np.all(labels[index,ISRJ]==1):raise ValueError(f'{name} includes non-ISRJ sample')
        if len(np.unique(labels[index,VFT]))!=2:raise ValueError(f'{name} lacks target state')
    return audit,labels,weak,receiver,ids

def _one_seed(args,features,labels,weak,receiver,ids,seed,device):
    root=args.output_root/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True)
    fit,early,evaluation=(ids[name] for name in ('fit','early_stop','evaluation'));mean=features[fit].mean(0).astype(np.float32);std=np.maximum(features[fit].std(0),1e-5).astype(np.float32);x=((features-mean)/std).astype(np.float32)
    y=labels[:,VFT].astype(np.float32);rng=np.random.default_rng(20719000+seed);yperm=y.copy();yperm[fit]=y[fit][rng.permutation(len(fit))];yperm[early]=y[early][rng.permutation(len(early))]
    real,best_real,_=_train('M_real',x[fit],y[fit],x[early],y[early],seed,device,args,root)
    perm,best_perm,_=_train('M_perm',x[fit],yperm[fit],x[early],yperm[early],seed,device,args,root)
    rows={}
    for name,model,best in (('M_real',real,best_real),('M_perm',perm,best_perm)):
        score=_scores(model,x,device,args.eval_batch_size);metrics=_metric(y[evaluation],score[evaluation]);weak_metrics=_weak_metric(labels,weak,evaluation,score[evaluation]);out=root/name
        np.savez_compressed(out/'predictions.npz',receiver_noise_seed=receiver[evaluation],targets=y[evaluation].astype(np.uint8),scores=score[evaluation],weak_vft=weak[evaluation,VFT].astype(np.uint8),sample_ids=evaluation)
        _write(out/'metrics.json',{'evaluation':metrics,'weak_vft_vs_hard_negative':weak_metrics,'best_epoch':best['epoch'],'best_early_loss':best['loss']});rows[name]={'evaluation':metrics,'weak_vft_vs_hard_negative':weak_metrics,'best_epoch':int(best['epoch']),'best_early_loss':float(best['loss'])}
    np.savez_compressed(root/'normalization.npz',mean=mean,std=std);row={'seed':int(seed),'parameter_count':int(sum(p.numel() for p in real.parameters())),'feature_dim':int(x.shape[1]),'roles':{k:int(len(v)) for k,v in ids.items()},'variants':rows,'normalization_sha256':_sha256(root/'normalization.npz'),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v19_used':False};_write(root/'seed_summary.json',row);return row

def run(args):
    audit,labels,weak,receiver,ids=_load(args);device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'));signals=_load_signals(args.data_path,np.sort(np.concatenate(tuple(ids.values()))));all_ids=np.sort(np.concatenate(tuple(ids.values())))
    if not np.array_equal(all_ids,np.arange(len(labels))[labels[:,ISRJ]==1]):raise ValueError('v19 eligible identity set changed')
    features=_extract_vft(signals,device,args.feature_batch_size);full=np.empty((len(labels),features.shape[1]),dtype=np.float32);full[all_ids]=features
    args.output_root.mkdir(parents=True,exist_ok=True);rows=[_one_seed(args,full,labels,weak,receiver,ids,seed,device) for seed in args.seeds]
    fields=('auroc','aupr');delta={field:_mean_ci([r['variants']['M_real']['evaluation'][field]-r['variants']['M_perm']['evaluation'][field] for r in rows],20719100+i) for i,field in enumerate(fields)};real={field:_mean_ci([r['variants']['M_real']['evaluation'][field] for r in rows],20719200+i) for i,field in enumerate(fields)}
    weak_rows=[r['variants']['M_real']['weak_vft_vs_hard_negative'] for r in rows]
    weak_auroc=_mean_ci([r['auroc'] for r in weak_rows],20719300) if all(r is not None for r in weak_rows) else None
    auroc_delta=[r['variants']['M_real']['evaluation']['auroc']-r['variants']['M_perm']['evaluation']['auroc'] for r in rows]
    checks={'real_auroc_ge_0p65':real['auroc']['mean']>=.65,'real_aupr_ge_0p65':real['aupr']['mean']>=.65,'real_minus_perm_auroc_ge_0p10':delta['auroc']['mean']>=.10,'real_minus_perm_auroc_ci_positive':delta['auroc']['bootstrap_95_ci'][0]>0,'real_minus_perm_auroc_positive_4of5':sum(v>0 for v in auroc_delta)>=4,'weak_vft_auroc_ge_0p55':weak_auroc is not None and weak_auroc['mean']>=.55,'matched_parameter_count':len({r['parameter_count'] for r in rows})==1,'test_v3_sealed':True}
    result={'status':'completed','experiment':'route_a_v19_hardnegative_vft_morphology_probe','evidence_role':'fresh_identity_isolated_identifiability_probe','data_audit_sha256':_sha256(args.audit),'seeds':[int(x) for x in args.seeds],'architecture':{'input_dim':268,'hidden':int(args.hidden),'dropout':float(args.dropout),'parameter_count':rows[0]['parameter_count']},'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,'weight_decay':args.weight_decay,'batch_size':args.batch_size,'epochs':args.epochs},'rows':rows,'real':real,'real_minus_permutation':delta,'weak_vft_vs_hard_negative':weak_auroc,'gate':{'checks':checks,'passed':bool(all(checks.values())),'decision':'permit_fresh_q4_integration_design' if all(checks.values()) else 'stop_morphology_expert_integration'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v19_used':False};_write(args.output_root/'route_a_v19_hardnegative_vft_morphology_probe_summary.json',result);_write(args.output_root/'run_receipt.json',{'status':'completed','gate_passed':result['gate']['passed'],'decision':result['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v19_used':False});print(json.dumps({'passed':result['gate']['passed'],'checks':checks,'output':str(args.output_root)},ensure_ascii=False,indent=2));return result

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v19_vft_morphology_probe\closedset_6class_train.mat'));p.add_argument('--audit',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v19_vft_morphology_probe_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v19_vft_morphology_probe_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v19_hardnegative_vft_morphology_probe');p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]);p.add_argument('--device',default=None);p.add_argument('--hidden',type=int,default=64);p.add_argument('--dropout',type=float,default=.1);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--eval-batch-size',type=int,default=256);p.add_argument('--feature-batch-size',type=int,default=32);p.add_argument('--epochs',type=int,default=30);p.add_argument('--weak-threshold-db',type=float,default=10.);return p.parse_args()
if __name__=='__main__':run(parse_args())
