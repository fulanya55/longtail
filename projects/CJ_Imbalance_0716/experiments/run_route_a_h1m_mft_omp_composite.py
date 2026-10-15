"""Fixed-score H1m composite validation for the H0l three-atom OMP cue."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels,read_parameters
from cjlt.models.mft_omp_residual import fixed_phase_permutation,lfm_reference,three_atom_residual_fraction
from experiments.run_route_a_development_v2_h5 import _component_power,_sha256
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals

MFT=3

def _scores(x,reference,device,batch):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(x),int(batch)):
            rows.append(three_atom_residual_fraction(torch.from_numpy(x[start:start+int(batch)]).to(device),reference).cpu().numpy())
    return np.concatenate(rows).astype(np.float64)

def _matched(labels,weak):
    other=[0,1,2,4,5]; pos=(labels[:,MFT]==1)&weak[:,MFT]; neg=labels[:,MFT]==0
    shared={tuple(x) for x in labels[pos][:,other].tolist()}&{tuple(x) for x in labels[neg][:,other].tolist()}
    kp=np.asarray([bool(pos[i]) and tuple(labels[i,other].tolist()) in shared for i in range(len(labels))])
    kn=np.asarray([bool(neg[i]) and tuple(labels[i,other].tolist()) in shared for i in range(len(labels))])
    keep=kp|kn
    return keep,kp[keep].astype(np.uint8),{'weak_positive_count':int(kp.sum()),'negative_count':int(kn.sum()),'matched_context_count':int(len(shared))}

def _rank(y,s): return {'count':int(len(y)),'auroc':float(roc_auc_score(y,s)),'aupr':float(average_precision_score(y,s))}

def _bootstrap(y,candidate,control,draws=10000,seed=20802141):
    y=np.asarray(y,dtype=np.uint8);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(seed);auroc=[];aupr=[]
    if len(pos)<2 or len(neg)<2: raise ValueError('need both classes for bootstrap')
    for _ in range(int(draws)):
        ix=np.r_[pos[rng.integers(0,len(pos),len(pos))],neg[rng.integers(0,len(neg),len(neg))]]
        auroc.append(roc_auc_score(y[ix],candidate[ix])-roc_auc_score(y[ix],control[ix]))
        aupr.append(average_precision_score(y[ix],candidate[ix])-average_precision_score(y[ix],control[ix]))
    return {'method':'paired_class_stratified_bootstrap_10000','auroc_delta_ci95':[float(x) for x in np.percentile(auroc,[2.5,97.5])],'aupr_delta_ci95':[float(x) for x in np.percentile(aupr,[2.5,97.5])],'auroc_delta_bootstrap_mean':float(np.mean(auroc)),'aupr_delta_bootstrap_mean':float(np.mean(aupr))}

def run(a):
    audit=json.loads(a.audit.read_text(encoding='utf8'));seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'))
    if not audit.get('identity_audit_passed') or not all(audit['contracts'].values()): raise ValueError('H1m identity audit failed')
    if any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')): raise ValueError('test-v3 is not sealed')
    if _sha256(a.data_path).lower()!=audit['sha256'].lower(): raise ValueError('H1m source hash changed')
    roles=np.load(a.roles,allow_pickle=False);evaluation=np.asarray(roles['evaluation'],dtype=np.int64)
    labels=read_labels(a.data_path);parameters,_=read_parameters(a.data_path);weak=compute_weak_component_mask(labels,_component_power(parameters),10.0)
    signals=_load_signals(a.data_path,np.arange(len(labels)),a.io_batch_size);device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ref=lfm_reference(device=device);candidate=_scores(signals,ref,device,a.score_batch_size);control=_scores(signals,fixed_phase_permutation(ref),device,a.score_batch_size)
    keep,y,counts=_matched(labels[evaluation],weak[evaluation]);ids=evaluation[keep];c=candidate[ids];b=control[ids]
    if counts['weak_positive_count']<20 or counts['negative_count']<20: raise ValueError(f'H1m primary view too small: {counts}')
    cand,ctrl=_rank(y,c),_rank(y,b);delta={'auroc':cand['auroc']-ctrl['auroc'],'aupr':cand['aupr']-ctrl['aupr']};boot=_bootstrap(y,c,b)
    rid=np.asarray(roles['receiver_noise_seed'],dtype=np.int64)[ids];folds=[]
    for fold in range(5):
        mask=((rid//10)%5)==fold
        if np.unique(y[mask]).size!=2: raise ValueError(f'subfold {fold} lacks a class')
        folds.append({'fold':fold,'count':int(mask.sum()),'weak_positive_count':int(y[mask].sum()),'negative_count':int((1-y[mask]).sum()),'auroc_delta':_rank(y[mask],c[mask])['auroc']-_rank(y[mask],b[mask])['auroc'],'aupr_delta':_rank(y[mask],c[mask])['aupr']-_rank(y[mask],b[mask])['aupr']})
    checks={'weak_auroc_delta_ge_0p015':delta['auroc']>=.015,'weak_auroc_ci_lower_gt_zero':boot['auroc_delta_ci95'][0]>0,'weak_aupr_delta_gt_zero':delta['aupr']>0,'weak_aupr_ci_lower_gt_zero':boot['aupr_delta_ci95'][0]>0,'weak_auroc_positive_subfold_4of5':sum(x['auroc_delta']>0 for x in folds)>=4,'zero_fit_parameters':True,'zero_threshold_calibration_fusion':True,'source_identity_contract':True,'test_v3_sealed':True}
    result={'status':'completed','experiment':'RouteA_H1m_MFTOMPComposite','evidence_role':'fresh_composite_mft_sparse_attribution_validation','data_sha256':_sha256(a.data_path),'source_audit_sha256':_sha256(a.audit),'role_indices_sha256':_sha256(a.roles),'statistic':{'candidate':'correct_LFM_fixed_three_atom_complex_OMP_residual_fraction','control':'fixed_phase_permuted_reference_same_dictionary','dictionary_delay_samples':[400,999],'atoms':3,'trained_parameters':0,'threshold_fit':False,'calibration_fit':False,'fusion_fit':False},'matched_view':counts,'candidate':cand,'control':ctrl,'candidate_minus_control':delta,'paired_bootstrap':boot,'receiver_subfolds':folds,'gate':{'checks':checks,'passed':bool(all(checks.values())),'decision':'permit_one_fresh_frozen_MFT_OMP_replication' if all(checks.values()) else 'stop_MFT_three_atom_OMP_residual'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h1m_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({'gate':result['gate'],'candidate_minus_control':delta,'bootstrap':boot,'output':str(a.output)},ensure_ascii=False,indent=2));return result

def args():
    p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\h1m_mft_omp_composite_20260803\closedset_6class_train.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h1m_mft_omp_composite_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_h1m_mft_omp_composite_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h1m_mft_omp/route_a_h1m_mft_omp_summary.json');p.add_argument('--device',default=None);p.add_argument('--io-batch-size',type=int,default=128);p.add_argument('--score-batch-size',type=int,default=16);return p.parse_args()
if __name__=='__main__':run(args())
