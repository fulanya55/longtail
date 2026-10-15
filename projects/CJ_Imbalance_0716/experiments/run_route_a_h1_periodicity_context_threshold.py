"""FMB-predicted context threshold for frozen H1+B2."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels,read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256,_write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_h1_periodicity_subgroup_safe_threshold import _worst_fpr
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration,_metrics

ISRJ=5; FMB=0

def _select_context(target,weak,p_isrj,risk,base_threshold,overall_cap=.20,risk_cap=.22):
 candidates=np.unique(np.r_[p_isrj,base_threshold,1.,np.nextafter(1.,2.)]); best=None
 for threshold in candidates[candidates>=base_threshold]:
  pred=p_isrj>=base_threshold; pred[risk]=p_isrj[risk]>=threshold; neg=target==0; risk_neg=neg&risk
  overall=float(pred[neg].mean()); flagged=float(pred[risk_neg].mean()) if risk_neg.any() else 0.
  if overall<=overall_cap+1e-12 and flagged<=risk_cap+1e-12:
   wp=(target==1)&weak; recall=float(pred[wp].mean()); key=(recall,-float(threshold))
   if best is None or key>best[0]: best=(key,float(threshold),overall,flagged)
 if best is None: raise RuntimeError("no feasible context threshold")
 return {"risk_threshold":best[1],"threshold_weak_recall":best[0][0],"overall_fpr":best[2],"predicted_fmb_positive_fpr":best[3],"risk_fraction":float(risk.mean())}

def _value(metrics,key):
 if key in ('macro_aupr','macro_f1'): return metrics[key]
 mapping={'isrj_aupr':'aupr','isrj_weak_recall':'weak_recall','isrj_strong_recall':'strong_recall','isrj_weak_auroc':'weak_vs_negative_auroc','isrj_fpr':'negative_fpr'}
 return metrics['per_label']['ISRJ'][mapping[key]]

def _run_seed(args,labels,weak,seed):
 root=args.output_root/f"seed_{seed}"; root.mkdir(parents=True,exist_ok=True); roles=np.load(args.c6_root/f"seed_{seed}"/'role_indices.npz'); logits=np.load(args.increment_root/f"seed_{seed}"/'all_role_logits.npz'); src=args.increment_root/f"seed_{seed}"/'B2_H1AlignedPeriodicity'; cal=json.loads((src/'calibration.json').read_text(encoding='utf-8')); base_thr=np.asarray(json.loads((src/'thresholds.json').read_text(encoding='utf-8'))['thresholds']); tids=roles['threshold'].astype(np.int64); eids=roles['early'].astype(np.int64); tp=_apply_frozen_calibration(logits['B2_H1AlignedPeriodicity__threshold'],cal); trisk=tp[:,FMB]>=base_thr[FMB]; selection=_select_context(labels[tids,ISRJ],weak[tids,ISRJ],tp[:,ISRJ],trisk,float(base_thr[ISRJ]),args.overall_fpr_cap,args.risk_fpr_cap)
 elogits=logits['B2_H1AlignedPeriodicity__early']; ep=_apply_frozen_calibration(elogits,cal); erisk=ep[:,FMB]>=base_thr[FMB]; base_pred=ep>=base_thr[None,:]; safe_pred=base_pred.copy(); safe_pred[:,ISRJ]=ep[:,ISRJ]>=base_thr[ISRJ]; safe_pred[erisk,ISRJ]=ep[erisk,ISRJ]>=selection['risk_threshold']; safe=_metrics(labels[eids],weak[eids],ep,safe_pred); base=_metrics(labels[eids],weak[eids],ep,base_pred); h1=json.loads((args.increment_root/f"seed_{seed}"/'B0_H1'/'metrics.json').read_text(encoding='utf-8')); worst,all_fpr=_worst_fpr(labels[eids],safe_pred,args.min_negative); out=root/'C1_FMBPredictedContextThreshold'; out.mkdir(parents=True,exist_ok=True); _write_json(out/'selection.json',{"source_role":"threshold","base_isrj_threshold":float(base_thr[ISRJ]),"fmb_threshold":float(base_thr[FMB]),**selection}); _write_json(out/'metrics.json',safe); _write_json(out/'subgroup_fpr.json',all_fpr); np.savez_compressed(out/'predictions.npz',sample_ids=eids,targets=labels[eids],weak_mask=weak[eids].astype(np.uint8),logits=elogits,probabilities=ep,predictions=safe_pred.astype(np.uint8),risk_flag=erisk.astype(np.uint8)); return {"seed":seed,"selection":selection,"worst_early_subgroup":{"name":worst[0],"fpr":worst[1]},"safe_metrics":safe,"base_b2_metrics":base,"h1_metrics":h1,"probability_sha256":_sha256(src/'predictions.npz'),"test_v3_used":False}

def run(args):
 seal=json.loads(args.test_v3_manifest.read_text(encoding='utf-8'));
 if seal['test_v3']['access_authorized'] or seal['test_v3']['analysis_allowed'] or seal['test_v3']['used']: raise ValueError('test-v3 is not sealed')
 labels=read_labels(args.data_path); values,_=read_parameters(args.data_path); weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db); rows=[_run_seed(args,labels,weak,s) for s in args.seeds]; keys=('macro_aupr','macro_f1','isrj_aupr','isrj_weak_recall','isrj_strong_recall','isrj_weak_auroc','isrj_fpr'); aggregate={k:{"safe":_mean_ci([_value(r['safe_metrics'],k) for r in rows],20336000+i),"safe_vs_b2":_mean_ci([_value(r['safe_metrics'],k)-_value(r['base_b2_metrics'],k) for r in rows],20336100+i),"safe_vs_h1":_mean_ci([_value(r['safe_metrics'],k)-_value(r['h1_metrics'],k) for r in rows],20336200+i)} for i,k in enumerate(keys)}; b2gain=np.mean([_value(r['base_b2_metrics'],'isrj_weak_recall')-_value(r['h1_metrics'],'isrj_weak_recall') for r in rows]); retention=float(aggregate['isrj_weak_recall']['safe_vs_h1']['mean']/b2gain); checks={"worst_subgroup_fpr_le_0p25":max(r['worst_early_subgroup']['fpr'] for r in rows)<=.25,"mean_isrj_fpr_le_0p20":aggregate['isrj_fpr']['safe']['mean']<=.20,"retain_80pct_b2_weak_gain":retention>=.80,"strong_recall_drop_le_0p03":aggregate['isrj_strong_recall']['safe_vs_b2']['mean']>=-.03,"macro_f1_drop_le_0p005":aggregate['macro_f1']['safe_vs_b2']['mean']>=-.005,"aupr_unchanged":abs(aggregate['isrj_aupr']['safe_vs_b2']['mean'])<1e-10,"weak_auroc_unchanged":abs(aggregate['isrj_weak_auroc']['safe_vs_b2']['mean'])<1e-10,"four_of_five_risk_threshold_ge_base":sum(r['selection']['risk_threshold']>=json.loads((args.increment_root/f"seed_{r['seed']}"/'B2_H1AlignedPeriodicity'/'thresholds.json').read_text())['thresholds'][ISRJ] for r in rows)>=4,"test_boundary_safe":all(not r['test_v3_used'] for r in rows)}; checks={k:bool(v) for k,v in checks.items()}; result={"status":"completed","experiment":"RouteA_H1_Periodicity_FMBPredictedContextThreshold","seeds":args.seeds,"rows":rows,"aggregate":aggregate,"weak_recall_gain_retention":retention,"development_gate":{"passed":bool(all(checks.values())),"checks":checks},"locked_test_used":False,"test_v3_used":False}; args.output_root.mkdir(parents=True,exist_ok=True); out=args.output_root/'route_a_h1_periodicity_context_threshold_summary.json'; _write_json(out,result); print(json.dumps({"status":"completed","retention":retention,"gate":result['development_gate'],"summary":str(out)},ensure_ascii=False,indent=2)); return result

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat')); p.add_argument('--c6-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_label_selective_projection'); p.add_argument('--increment-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_h1_periodicity_increment'); p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json'); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_h1_periodicity_context_threshold'); p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]); p.add_argument('--overall-fpr-cap',type=float,default=.20); p.add_argument('--risk-fpr-cap',type=float,default=.22); p.add_argument('--min-negative',type=int,default=50); p.add_argument('--weak-threshold-db',type=float,default=10.); run(p.parse_args())
