"""Five-seed fresh validation of a fixed weak-positive/hard-negative ranking loss."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.run_route_a_v22_visibility_balanced_morphology_expert import (
    PROJECT_ROOT, MorphologyProbe, _load, _mean_ci, _sha256, _load_signals,
    _extract_vft, _metric, _stratum_loss, _stratum_metric, _predict, _read,
    _set_seed, _write,
)


def _train(name, xfit, yfit, weakfit, xearly, yearly, weakearly, seed, device, args, root):
    _set_seed(seed)
    model=MorphologyProbe(xfit.shape[1],args.hidden,args.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    generator=torch.Generator();generator.manual_seed(seed+902);best={'loss':float('inf'),'epoch':0,'state':None};history=[]
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(xfit),generator=generator).numpy();losses=[];pairs=[]
        for start in range(0,len(order),args.batch_size):
            index=order[start:start+args.batch_size];xb=torch.from_numpy(xfit[index]).to(device);yb=torch.from_numpy(yfit[index]).to(device);wb=torch.from_numpy(weakfit[index]).to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(xb);bce=F.binary_cross_entropy_with_logits(logits,yb);pair=torch.zeros((),device=device)
            if name=='M_WPR':
                weak_scores=logits[(yb==1)&(wb==1)];hard_scores=logits[yb==0]
                if len(weak_scores) and len(hard_scores): pair=F.softplus(-(weak_scores[:,None]-hard_scores[None,:])).mean()
            loss=bce+args.ranking_weight*pair;loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()));pairs.append(float(pair.detach().cpu()))
        with torch.no_grad(): early_loss=float(_stratum_loss(model(torch.from_numpy(xearly).to(device)),torch.from_numpy(yearly).to(device),torch.from_numpy(weakearly).to(device)).cpu())
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'train_pair_loss':float(np.mean(pairs)),'early_three_stratum_loss':early_loss})
        if early_loss<best['loss']-1e-10:best={'loss':early_loss,'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    model.load_state_dict(best['state']);out=root/name;out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'best_epoch':best['epoch'],'best_early_three_stratum_loss':best['loss'],'ranking_weight':float(args.ranking_weight if name=='M_WPR' else 0.0),'architecture':{'input_dim':int(xfit.shape[1]),'hidden':args.hidden,'dropout':args.dropout}},out/'best_model.pth');_write(out/'history.json',history)
    return model,best


def _one_seed(args,features,labels,weak,receiver,ids,seed,device):
    root=args.output_root/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True);fit,early,evaluation=(ids[name] for name in ('fit','early_stop','evaluation'))
    mean=features[fit].mean(0).astype(np.float32);std=np.maximum(features[fit].std(0),1e-5).astype(np.float32);x=((features-mean)/std).astype(np.float32);y=labels[:,4].astype(np.float32);w=weak[:,4].astype(np.float32);variants={}
    for name in ('M_BCE','M_WPR'):
        model,best=_train(name,x[fit],y[fit],w[fit],x[early],y[early],w[early],seed,device,args,root);score=_predict(model,x,device,args.eval_batch_size);out=root/name
        metrics={'overall':_metric(y[evaluation],score[evaluation]),'weak_vft_vs_hard_negative':_stratum_metric(labels,weak,evaluation,score[evaluation],'weak'),'strong_vft_vs_hard_negative':_stratum_metric(labels,weak,evaluation,score[evaluation],'strong'),'best_epoch':best['epoch'],'best_early_three_stratum_loss':best['loss'],'ranking_weight':float(args.ranking_weight if name=='M_WPR' else 0.)}
        np.savez_compressed(out/'predictions.npz',sample_ids=evaluation,receiver_noise_seed=receiver[evaluation],targets=y[evaluation].astype(np.uint8),weak_vft=w[evaluation].astype(np.uint8),scores=score[evaluation]);_write(out/'metrics.json',metrics);variants[name]=metrics
    np.savez_compressed(root/'normalization.npz',mean=mean,std=std)
    return {'seed':seed,'feature_dim':int(x.shape[1]),'parameter_count':int(sum(p.numel() for p in MorphologyProbe(x.shape[1],args.hidden,args.dropout).parameters())),'roles':{k:int(len(v)) for k,v in ids.items()},'variants':variants,'normalization_sha256':_sha256(root/'normalization.npz'),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v23_used':False}


def run(args):
    audit,labels,weak,receiver,ids=_load(args);device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'));all_ids=np.sort(np.concatenate(tuple(ids.values())))
    signals=_load_signals(args.data_path,all_ids);raw=_extract_vft(signals,device,args.feature_batch_size);features=np.zeros((len(labels),raw.shape[1]),dtype=np.float32);features[all_ids]=raw;args.output_root.mkdir(parents=True,exist_ok=True);rows=[_one_seed(args,features,labels,weak,receiver,ids,seed,device) for seed in args.seeds]
    def delta(section,field,salt):return _mean_ci([row['variants']['M_WPR'][section][field]-row['variants']['M_BCE'][section][field] for row in rows],salt)
    d={'weak':{field:delta('weak_vft_vs_hard_negative',field,230100+i) for i,field in enumerate(('auroc','aupr'))},'strong':{field:delta('strong_vft_vs_hard_negative',field,230200+i) for i,field in enumerate(('auroc','aupr'))},'overall':{field:delta('overall',field,230300+i) for i,field in enumerate(('auroc','aupr'))}}
    weak_seed=[r['variants']['M_WPR']['weak_vft_vs_hard_negative']['auroc']-r['variants']['M_BCE']['weak_vft_vs_hard_negative']['auroc'] for r in rows];strong_seed=[r['variants']['M_WPR']['strong_vft_vs_hard_negative']['auroc']-r['variants']['M_BCE']['strong_vft_vs_hard_negative']['auroc'] for r in rows];all_seed=[r['variants']['M_WPR']['overall']['auroc']-r['variants']['M_BCE']['overall']['auroc'] for r in rows]
    checks={'weak_auroc_gain_ge_0p015':d['weak']['auroc']['mean']>=.015,'weak_auroc_ci_positive':d['weak']['auroc']['bootstrap_95_ci'][0]>0,'weak_auroc_positive_4of5':sum(x>0 for x in weak_seed)>=4,'weak_aupr_gain_ge_0p010':d['weak']['aupr']['mean']>=.010,'strong_auroc_safe_every_seed':all(x>=-.010 for x in strong_seed),'overall_auroc_safe_every_seed':all(x>=-.010 for x in all_seed),'matched_architecture':len({r['parameter_count'] for r in rows})==1 and len({r['feature_dim'] for r in rows})==1,'test_v3_sealed':True};passed=bool(all(checks.values()))
    result={'status':'completed','experiment':'route_a_v23_weak_pair_ranking_morphology_expert','evidence_role':'fresh_identity_isolated_weak_pair_ranking_probe','data_audit_sha256':_sha256(args.audit),'seeds':list(args.seeds),'architecture':{'input_dim':268,'hidden':args.hidden,'dropout':args.dropout,'parameter_count':rows[0]['parameter_count']},'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,'weight_decay':args.weight_decay,'batch_size':args.batch_size,'epochs':args.epochs,'ranking_weight':args.ranking_weight,'checkpoint_criterion':'unweighted_three_stratum_early_loss'},'rows':rows,'wpr_minus_bce':d,'gate':{'checks':checks,'passed':passed,'decision':'permit_fresh_fusion_design' if passed else 'stop_weak_pair_ranking_morphology'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v23_used':False};_write(args.output_root/'route_a_v23_weak_pair_ranking_morphology_expert_summary.json',result);_write(args.output_root/'run_receipt.json',{'status':'completed','gate_passed':passed,'decision':result['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v23_used':False});print(json.dumps({'passed':passed,'checks':checks,'output':str(args.output_root)},ensure_ascii=False,indent=2));return result


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v23_weak_pair_ranking_morphology\closedset_6class_train.mat'));parser.add_argument('--audit',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v23_weak_pair_ranking_morphology_generation/identity_audit.json');parser.add_argument('--roles',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v23_weak_pair_ranking_morphology_generation/role_indices.npz');parser.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');parser.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v23_weak_pair_ranking_morphology_expert');parser.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]);parser.add_argument('--device',default=None);parser.add_argument('--hidden',type=int,default=64);parser.add_argument('--dropout',type=float,default=.10);parser.add_argument('--learning-rate',type=float,default=3e-4);parser.add_argument('--weight-decay',type=float,default=1e-4);parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--eval-batch-size',type=int,default=256);parser.add_argument('--feature-batch-size',type=int,default=32);parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--weak-threshold-db',type=float,default=10.);parser.add_argument('--ranking-weight',type=float,default=.25);return parser.parse_args()
if __name__=='__main__':run(parse_args())
