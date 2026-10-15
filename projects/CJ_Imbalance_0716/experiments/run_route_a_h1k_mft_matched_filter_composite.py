"""Frozen no-fit H1k validation for MFT matched-filter effective support."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels,read_parameters  # noqa:E402
from cjlt.models.mft_matched_filter_support import effective_delay_support,fixed_phase_permutation,lfm_reference  # noqa:E402
from experiments.run_route_a_development_v2_h5 import _component_power,_sha256  # noqa:E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa:E402

MFT=3

def _read(p:Path)->dict:return json.loads(p.read_text(encoding='utf8'))
def _scores(x:np.ndarray,reference:torch.Tensor,device:torch.device,batch:int)->np.ndarray:
    rows=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):rows.append(effective_delay_support(torch.from_numpy(x[start:start+batch]).to(device),reference).cpu().numpy())
    return np.concatenate(rows).astype(np.float64)
def _rank(y,s):return {'count':int(len(y)),'auroc':float(roc_auc_score(y,s)),'aupr':float(average_precision_score(y,s))}
def _matched_view(labels:np.ndarray,weak:np.ndarray)->tuple[np.ndarray,np.ndarray,dict]:
    other=[0,1,2,4,5];positive=(labels[:,MFT]==1)&weak[:,MFT];negative=labels[:,MFT]==0
    pctx={tuple(row) for row in labels[positive][:,other].tolist()};nctx={tuple(row) for row in labels[negative][:,other].tolist()};shared=pctx&nctx
    keep_pos=np.asarray([bool(positive[i]) and tuple(labels[i,other].tolist()) in shared for i in range(len(labels))]);keep_neg=np.asarray([bool(negative[i]) and tuple(labels[i,other].tolist()) in shared for i in range(len(labels))]);keep=keep_pos|keep_neg
    return keep,keep_pos[keep].astype(np.uint8),{'weak_positive_count':int(keep_pos.sum()),'negative_count':int(keep_neg.sum()),'matched_context_count':int(len(shared))}
def _bootstrap(y,candidate,control,seed:int)->dict:
    """10,000 paired draws, resampling fixed positives and negatives separately."""
    y=np.asarray(y,dtype=np.uint8);candidate=np.asarray(candidate);control=np.asarray(control);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if len(pos)<2 or len(neg)<2:raise ValueError('bootstrap needs both classes')
    rng=np.random.default_rng(seed);auroc=[];aupr=[]
    for _ in range(10_000):
        ix=np.r_[pos[rng.integers(0,len(pos),len(pos))],neg[rng.integers(0,len(neg),len(neg))]];draw_y=y[ix]
        auroc.append(roc_auc_score(draw_y,candidate[ix])-roc_auc_score(draw_y,control[ix]));aupr.append(average_precision_score(draw_y,candidate[ix])-average_precision_score(draw_y,control[ix]))
    return {'method':'paired_class_stratified_bootstrap_10000','auroc_delta_ci95':[float(x) for x in np.percentile(auroc,[2.5,97.5])],'aupr_delta_ci95':[float(x) for x in np.percentile(aupr,[2.5,97.5])],'auroc_delta_bootstrap_mean':float(np.mean(auroc)),'aupr_delta_bootstrap_mean':float(np.mean(aupr))}
def run(a):
    audit,seal=_read(a.audit),_read(a.test_v3_manifest)
    if not audit.get('identity_audit_passed') or not all(audit['contracts'].values()):raise ValueError('H1k identity audit failed')
    if any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('test-v3 is not sealed')
    if _sha256(a.data_path).lower()!=audit['sha256'].lower():raise ValueError('H1k source hash changed')
    roles=np.load(a.roles,allow_pickle=False);evaluation=np.asarray(roles['evaluation'],dtype=np.int64)
    labels=read_labels(a.data_path);signals=_load_signals(a.data_path,np.arange(len(labels)),a.io_batch_size);device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ref=lfm_reference(device=device);candidate=_scores(signals,ref,device,a.score_batch_size);control=_scores(signals,fixed_phase_permutation(ref),device,a.score_batch_size)
    # Evaluation-only metadata is read only after the two deployable scores are fixed.
    params,_=read_parameters(a.data_path);weak=compute_weak_component_mask(labels,_component_power(params),10.0)
    keep,y,counts=_matched_view(labels[evaluation],weak[evaluation]);ids=evaluation[keep];c=candidate[ids];b=control[ids]
    if counts['weak_positive_count']<20 or counts['negative_count']<20:raise ValueError(f'H1k primary view too small: {counts}')
    cand=_rank(y,c);ctrl=_rank(y,b);delta={'auroc':cand['auroc']-ctrl['auroc'],'aupr':cand['aupr']-ctrl['aupr']};boot=_bootstrap(y,c,b,20802121)
    rid=np.asarray(roles['receiver_noise_seed'],dtype=np.int64)[ids];folds=[]
    for fold in range(5):
        mask=(rid//10)%5==fold
        if np.unique(y[mask]).size!=2:raise ValueError(f'subfold {fold} lacks a class')
        folds.append({'fold':fold,'count':int(mask.sum()),'weak_positive_count':int(y[mask].sum()),'negative_count':int((1-y[mask]).sum()),'auroc_delta':_rank(y[mask],c[mask])['auroc']-_rank(y[mask],b[mask])['auroc'],'aupr_delta':_rank(y[mask],c[mask])['aupr']-_rank(y[mask],b[mask])['aupr']})
    checks={'weak_auroc_delta_ge_0p015':delta['auroc']>=.015,'weak_auroc_ci_lower_gt_zero':boot['auroc_delta_ci95'][0]>0,'weak_aupr_delta_gt_zero':delta['aupr']>0,'weak_aupr_ci_lower_gt_zero':boot['aupr_delta_ci95'][0]>0,'weak_auroc_positive_subfold_4of5':sum(row['auroc_delta']>0 for row in folds)>=4,'zero_fit_parameters':True,'zero_threshold_calibration_fusion':True,'source_identity_contract':True,'test_v3_sealed':True}
    result={'status':'completed','experiment':'RouteA_H1k_MFTMatchedFilterComposite','evidence_role':'fresh_composite_mft_fixed_statistic_validation','data_sha256':_sha256(a.data_path),'source_audit_sha256':_sha256(a.audit),'role_indices_sha256':_sha256(a.roles),'statistic':{'candidate':'correct_LFM_matched_filter_Renyi2_effective_delay_support','control':'fixed_phase_permuted_unit_modulus_reference','roi_samples':[400,1000],'trained_parameters':0,'threshold_fit':False,'calibration_fit':False,'fusion_fit':False},'matched_view':counts,'candidate':cand,'control':ctrl,'candidate_minus_control':delta,'paired_bootstrap':boot,'receiver_subfolds':folds,'gate':{'checks':checks,'passed':bool(all(checks.values())),'decision':'permit_one_fresh_frozen_MFT_replication' if all(checks.values()) else 'stop_MFT_effective_support_statistic'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h1k_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({'gate':result['gate'],'candidate_minus_control':delta,'bootstrap':boot,'output':str(a.output)},ensure_ascii=False,indent=2));return result
def args():
    p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h1k_mft_support_composite\closedset_6class_train.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h1k_mft_support_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=ROOT/'outputs/tapc/route_a_h1k_mft_support_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h1k_mft_support/route_a_h1k_mft_support_summary.json');p.add_argument('--device',default=None);p.add_argument('--io-batch-size',type=int,default=128);p.add_argument('--score-batch-size',type=int,default=128);return p.parse_args()
if __name__=='__main__':run(args())
