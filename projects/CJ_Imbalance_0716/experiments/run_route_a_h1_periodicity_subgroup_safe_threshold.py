"""Select a global ISRJ threshold under worst subgroup FPR constraints."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256,_write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration,_metrics
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import CLASS_NAMES
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels,read_parameters

ISRJ=5

def _groups(labels,min_negative=50):
 y=labels[:,ISRJ]; groups={"overall":np.ones(len(labels),bool)}; order=labels.sum(1)
 for value in range(1,7): groups[f"order_{value}"]=order==value
 for index,name in enumerate(CLASS_NAMES):
  if index!=ISRJ:
   groups[f"{name}_present"]=labels[:,index]==1; groups[f"{name}_absent"]=labels[:,index]==0
 return {n:m for n,m in groups.items() if np.sum(m&(y==0))>=min_negative}

def _select_threshold(target,weak,probability,labels,overall_cap=.20,subgroup_cap=.22,min_negative=50):
 groups=_groups(labels,min_negative); candidates=np.unique(np.r_[probability,0.,1.,np.nextafter(1.,2.)]); best=None
 for threshold in candidates:
  pred=probability>=threshold; feasible=True; fprs={}
  for name,mask in groups.items():
   negative=mask&(target==0); fpr=float(pred[negative].mean()); fprs[name]=fpr; cap=overall_cap if name=="overall" else subgroup_cap
   if fpr>cap+1e-12: feasible=False; break
  if feasible:
   wp=(target==1)&weak; recall=float(pred[wp].mean()) if wp.any() else 0.; key=(recall,-float(threshold))
   if best is None or key>best[0]: best=(key,float(threshold),fprs)
 if best is None: raise RuntimeError("no feasible subgroup-safe threshold")
 return {"threshold":best[1],"threshold_weak_recall":best[0][0],"subgroup_fpr":best[2],"groups":sorted(groups)}

def _worst_fpr(labels,pred,min_negative=50):
 values={}
 for n,m in _groups(labels,min_negative).items():
  neg=m&(labels[:,ISRJ]==0); values[n]=float(pred[neg,ISRJ].mean())
 return max(values.items(),key=lambda x:x[1]),values

def _run_seed(args,labels,weak,seed):
 root=args.output_root/f"seed_{seed}"; root.mkdir(parents=True,exist_ok=True); roles=np.load(args.c6_root/f"seed_{seed}"/"role_indices.npz"); logits=np.load(args.increment_root/f"seed_{seed}"/"all_role_logits.npz"); src=args.increment_root/f"seed_{seed}"/"B2_H1AlignedPeriodicity"; cal=json.loads((src/"calibration.json").read_text(encoding="utf-8")); base_thr=np.asarray(json.loads((src/"thresholds.json").read_text(encoding="utf-8"))["thresholds"])
 t_ids=roles["threshold"].astype(np.int64); e_ids=roles["early"].astype(np.int64); t_logits=logits["B2_H1AlignedPeriodicity__threshold"]; t_prob=_apply_frozen_calibration(t_logits,cal); selection=_select_threshold(labels[t_ids,ISRJ],weak[t_ids,ISRJ],t_prob[:,ISRJ],labels[t_ids],args.overall_fpr_cap,args.subgroup_fpr_cap,args.min_negative)
 e_logits=logits["B2_H1AlignedPeriodicity__early"]; e_prob=_apply_frozen_calibration(e_logits,cal); safe_thr=base_thr.copy(); safe_thr[ISRJ]=selection["threshold"]; safe_pred=e_prob>=safe_thr[None,:]; base_pred=e_prob>=base_thr[None,:]; safe_metrics=_metrics(labels[e_ids],weak[e_ids],e_prob,safe_pred); base_metrics=_metrics(labels[e_ids],weak[e_ids],e_prob,base_pred); worst,all_fpr=_worst_fpr(labels[e_ids],safe_pred,args.min_negative)
 b0=json.loads((args.increment_root/f"seed_{seed}"/"B0_H1"/"metrics.json").read_text(encoding="utf-8")); out=root/"S1_SubgroupSafeThreshold"; out.mkdir(parents=True,exist_ok=True); _write_json(out/"selection.json",{"source_role":"threshold","base_threshold":float(base_thr[ISRJ]),**selection}); _write_json(out/"thresholds.json",{"thresholds":safe_thr.tolist(),"source_role":"threshold_subgroup_safe"}); _write_json(out/"metrics.json",safe_metrics); _write_json(out/"subgroup_fpr.json",all_fpr); np.savez_compressed(out/"predictions.npz",sample_ids=e_ids,targets=labels[e_ids],weak_mask=weak[e_ids].astype(np.uint8),logits=e_logits,probabilities=e_prob,predictions=safe_pred.astype(np.uint8))
 return {"seed":seed,"selection":selection,"worst_early_subgroup":{"name":worst[0],"fpr":worst[1]},"safe_metrics":safe_metrics,"base_b2_metrics":base_metrics,"b0_metrics":b0,"logits_sha256":_sha256(args.increment_root/f"seed_{seed}"/"all_role_logits.npz"),"test_v3_used":False}

def run(args):
 seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"));
 if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]: raise ValueError("test-v3 is not sealed")
 labels=read_labels(args.data_path); values,_=read_parameters(args.data_path); weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db); rows=[_run_seed(args,labels,weak,s) for s in args.seeds]
 def val(m,key):
  if key in ('macro_aupr','macro_f1'): return m[key]
  name=key.replace('isrj_',''); mapping={'weak_recall':'weak_recall','strong_recall':'strong_recall','fpr':'negative_fpr','aupr':'aupr','weak_auroc':'weak_vs_negative_auroc'}; return m['per_label']['ISRJ'][mapping[name]]
 keys=('macro_aupr','macro_f1','isrj_aupr','isrj_weak_recall','isrj_strong_recall','isrj_weak_auroc','isrj_fpr'); aggregate={k:{"safe":_mean_ci([val(r['safe_metrics'],k) for r in rows],20335000+i),"safe_vs_b2":_mean_ci([val(r['safe_metrics'],k)-val(r['base_b2_metrics'],k) for r in rows],20335100+i),"safe_vs_h1":_mean_ci([val(r['safe_metrics'],k)-val(r['b0_metrics'],k) for r in rows],20335200+i)} for i,k in enumerate(keys)}
 b2_gain=np.mean([val(r['base_b2_metrics'],'isrj_weak_recall')-val(r['b0_metrics'],'isrj_weak_recall') for r in rows]); safe_gain=aggregate['isrj_weak_recall']['safe_vs_h1']['mean']; retention=float(safe_gain/b2_gain) if b2_gain>0 else 0.; checks={"worst_subgroup_fpr_le_0p25":max(r['worst_early_subgroup']['fpr'] for r in rows)<=.25,"mean_isrj_fpr_le_0p20":aggregate['isrj_fpr']['safe']['mean']<=.20,"retain_80pct_b2_weak_gain":retention>=.80,"strong_recall_drop_le_0p03":aggregate['isrj_strong_recall']['safe_vs_b2']['mean']>=-.03,"macro_f1_drop_le_0p005":aggregate['macro_f1']['safe_vs_b2']['mean']>=-.005,"aupr_unchanged":abs(aggregate['isrj_aupr']['safe_vs_b2']['mean'])<1e-10,"weak_auroc_unchanged":abs(aggregate['isrj_weak_auroc']['safe_vs_b2']['mean'])<1e-10,"threshold_role_only":True,"test_boundary_safe":all(not r['test_v3_used'] for r in rows)}; checks={k:bool(v) for k,v in checks.items()}
 result={"status":"completed","experiment":"RouteA_H1_Periodicity_SubgroupSafeThreshold","seeds":args.seeds,"rows":rows,"aggregate":aggregate,"weak_recall_gain_retention":retention,"development_gate":{"passed":bool(all(checks.values())),"checks":checks},"locked_test_used":False,"test_v3_used":False}; args.output_root.mkdir(parents=True,exist_ok=True); out=args.output_root/"route_a_h1_periodicity_subgroup_safe_threshold_summary.json"; _write_json(out,result); print(json.dumps({"status":"completed","retention":retention,"gate":result['development_gate'],"summary":str(out)},ensure_ascii=False,indent=2)); return result

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat')); p.add_argument('--c6-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_label_selective_projection'); p.add_argument('--increment-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_h1_periodicity_increment'); p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json'); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_h1_periodicity_subgroup_safe_threshold'); p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]); p.add_argument('--overall-fpr-cap',type=float,default=.20); p.add_argument('--subgroup-fpr-cap',type=float,default=.22); p.add_argument('--min-negative',type=int,default=50); p.add_argument('--weak-threshold-db',type=float,default=10.); run(p.parse_args())
