"""Frozen B0/B2 evaluation on train-only ISRJ parameter-OOD external data."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import h5py, joblib, numpy as np, torch
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch.utils.data import DataLoader

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import FrozenFeatureHead
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_model_from_state, _load_rawsaw
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _head_scores, _replace_isrj
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, _clipped_logit, _load_expert, _standardize_scores
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import CLASS_NAMES, SAW_INDEX, TARGET_NAMES, _per_label_metrics, _raw_scores

DATA_SHA="a7e318ba3eeaedb4fea67c894238e4674b133059bc405342ea40c9dfb767e081"
VARIANTS=("B0_H1_Frozen","B2_H1AlignedPeriodicity_Frozen")
F2_MODEL_NAME="F2_DFMS_Full"

def _load_signals(path,indices,batch_size):
 with h5py.File(path,"r") as handle:
  has_legacy="data_legacy" in handle
 if has_legacy:
  ds=V73SignalDataset(path,indices,class_mode="six",normalization="complex_rms"); loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0); rows=[]
  try:
   for batch in loader: rows.append(batch["signal"].numpy().astype(np.float32))
  finally: ds.close()
  return np.concatenate(rows)
 rows=[]; indices=np.asarray(indices,dtype=np.int64)
 with h5py.File(path,"r") as handle:
  dataset=handle["iq_complex"]
  for start in range(0,len(indices),batch_size):
   selected=indices[start:start+batch_size]
   stored=np.asarray(dataset[:,selected])
   complex_values=(stored["real"]+1j*stored["imag"]).T
   iq=np.stack([complex_values.real,complex_values.imag],axis=1).astype(np.float32)
   rms=np.sqrt(np.maximum(np.mean(iq[:,0]**2+iq[:,1]**2,axis=1,keepdims=True),1e-12))
   rows.append(iq/rms[:,:,None])
 return np.concatenate(rows).astype(np.float32)

def _features(model,signals,device,batch_size):
 rows=[]; model.eval()
 with torch.no_grad():
  for start in range(0,len(signals),batch_size):
   x=torch.from_numpy(signals[start:start+batch_size]).to(device)
   with torch.amp.autocast("cuda",enabled=device.type=="cuda"): rows.append(model.forward_features(x).float().cpu().numpy())
 return np.concatenate(rows).astype(np.float32)

def _head_logits(head,features,device,batch_size):
 rows=[]; head.eval()
 with torch.no_grad():
  for start in range(0,len(features),batch_size): rows.append(head(torch.from_numpy(features[start:start+batch_size]).to(device)).cpu().numpy())
 return np.concatenate(rows).astype(np.float32)

def _apply_frozen_calibration(logits,payload):
 params=payload["parameters"]
 values=np.column_stack([1/(1+np.exp(-np.clip(float(row["coefficient"])*logits[:,i]+float(row["intercept"]),-40,40))) for i,row in enumerate(params)])
 return values.astype(np.float32)

def _metrics(labels,weak,prob,pred):
 per=_per_label_metrics(labels,prob,pred,weak); target=[per[n] for n in TARGET_NAMES]; saw=labels[:,SAW_INDEX].astype(bool)
 return {"macro_aupr":float(average_precision_score(labels,prob,average="macro")),"macro_f1":float(f1_score(labels,pred,average="macro",zero_division=0)),"hamming_loss":float(hamming_loss(labels,pred)),"per_label":per,
  "target_mean_weak_recall":float(np.mean([x["weak_recall"] for x in target])),"target_mean_strong_recall":float(np.mean([x["strong_recall"] for x in target])),"target_mean_negative_fpr":float(np.mean([x["negative_fpr"] for x in target])),"target_weak_vs_negative_auroc":float(np.mean([x["weak_vs_negative_auroc"] for x in target])),
  "subgroups":{name:{"macro_aupr":float(average_precision_score(labels[mask],prob[mask],average="macro")),"macro_f1":float(f1_score(labels[mask],pred[mask],average="macro",zero_division=0))} for name,mask in (("sawfm_present",saw),("sawfm_absent",~saw))}}

def _isrj_group_metrics(labels,weak,prob,pred,mask):
 y=labels[mask,ISRJ_INDEX]; w=weak[mask,ISRJ_INDEX]; p=prob[mask,ISRJ_INDEX]; q=pred[mask,ISRJ_INDEX]; pos=y==1; neg=~pos; wp=pos&w; sp=pos&~w
 if wp.sum()<10 or neg.sum()<10: return None
 from sklearn.metrics import roc_auc_score
 return {"count":int(mask.sum()),"weak_positive_count":int(wp.sum()),"negative_count":int(neg.sum()),"aupr":float(average_precision_score(y,p)),"weak_recall":float(q[wp].mean()),"strong_recall":float(q[sp].mean()) if sp.any() else float('nan'),"fpr":float(q[neg].mean()),"weak_auroc":float(roc_auc_score(np.r_[np.ones(wp.sum()),np.zeros(neg.sum())],np.r_[p[wp],p[neg]]))}

def _strata(labels,values,receiver):
 width=values[:,53]*1e6; replicas=values[:,54]; jnr=values[:,52]; order=labels.sum(1); present=labels[:,ISRJ_INDEX]==1; groups={}
 for name,mask in (("width_22_26",present&(width<26)),("width_26_30",present&(width>=26)),("jnr_10_20",present&(jnr<20)),("jnr_20_30",present&(jnr>=20))): groups[name]=mask|(~present)
 for value in (6,7,8): groups[f"replicas_{value}"]=(present&(replicas==value))|(~present)
 for value in range(1,7): groups[f"order_{value}"]=order==value
 for index,name in enumerate(CLASS_NAMES):
  if index!=ISRJ_INDEX: groups[f"colabel_{name}_present"]=labels[:,index]==1
 for value in range(5): groups[f"receiver_cohort_{value}"]=receiver%5==value
 return groups

def _run_seed(args,labels,values,weak,signals,receiver,seed,device):
 root=args.output_root/f"seed_{seed}"; receipt=root/"seed_summary.json"
 if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
 focal=args.convergence_root/f"seed_{seed}"/"E2_Focal_36"/"best_model.pth"; state=torch.load(focal,map_location="cpu",weights_only=False)["model_state_dict"]; f2_dir=args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}"; backbone=_load_model_from_state(f2_dir,device,state)
 role=np.load(args.c6_root/f"seed_{seed}"/"role_indices.npz"); fit_ids=role["fit"]; dev_fit=_load_signals(args.development_path,fit_ids,args.io_batch_size); fit_features=_features(backbone,dev_fit,device,args.model_batch_size); ext_features=_features(backbone,signals,device,args.model_batch_size); del backbone,dev_fit
 mean=fit_features.mean(0,keepdims=True); std=np.maximum(fit_features.std(0,keepdims=True),1e-5); ext_features=((ext_features-mean)/std).astype(np.float32)
 head_path=args.head_root/f"seed_{seed}"/"H1_Linear_BCE"/"best_model.pth"; hp=torch.load(head_path,map_location="cpu",weights_only=False); head=FrozenFeatureHead(128,"linear").to(device); head.load_state_dict(hp["model_state_dict"]); base=_head_logits(head,ext_features,device,args.model_batch_size); del head,fit_features,ext_features
 raw,raw_path=_load_rawsaw(args,seed,device); base[:,SAW_INDEX]=_raw_scores(raw,signals,device,args.raw_batch_size); del raw
 expert,em,es,expert_hash=_load_expert(args.periodicity_root,seed,"H_AlignedHGB"); px=_extract(signals,device,args.periodicity_batch_size); eraw=_clipped_logit(expert.predict_proba(((px-em)/np.maximum(es,1e-5)).astype(np.float32))[:,1]); old=json.loads((args.old_f3_root/f"seed_{seed}"/"seed_summary.json").read_text(encoding="utf-8")); z=_standardize_scores(eraw,old["score_stats"]["aligned"])
 fusion_root=args.increment_root/f"seed_{seed}"/"head_aligned"; fusion=joblib.load(fusion_root/"model.joblib"); norm=np.load(fusion_root/"normalization.npz"); fused=_replace_isrj(base,_head_scores(fusion,_fusion_features(base,z),norm["mean"],norm["std"]))
 rows=[]; groups=_strata(labels,values,receiver)
 for name,logits in zip(VARIANTS,(base,fused)):
  src=args.increment_root/f"seed_{seed}"/("B0_H1" if name==VARIANTS[0] else "B2_H1AlignedPeriodicity"); cal=json.loads((src/"calibration.json").read_text(encoding="utf-8")); thr=json.loads((src/"thresholds.json").read_text(encoding="utf-8")); prob=_apply_frozen_calibration(logits,cal); thresholds=np.asarray(thr["thresholds"]); pred=prob>=thresholds[None,:]; metrics=_metrics(labels,weak,prob,pred); strata={g:_isrj_group_metrics(labels,weak,prob,pred,m) for g,m in groups.items()}; out=root/name; out.mkdir(parents=True,exist_ok=True); _write_json(out/"metrics.json",metrics); _write_json(out/"strata.json",strata); np.savez_compressed(out/"predictions.npz",sample_ids=np.arange(len(labels)),targets=labels,weak_mask=weak.astype(np.uint8),logits=logits,probabilities=prob,predictions=pred.astype(np.uint8)); rows.append({"variant":name,"metrics":metrics,"strata":strata,"calibration_sha256":_sha256(src/"calibration.json"),"thresholds_sha256":_sha256(src/"thresholds.json")})
 row={"seed":seed,"status":"completed","external_data_sha256":DATA_SHA,"focal_checkpoint_sha256":_sha256(focal),"h1_checkpoint_sha256":_sha256(head_path),"rawsaw_checkpoint_sha256":_sha256(raw_path),"periodicity_expert_sha256":expert_hash,"fusion_model_sha256":_sha256(fusion_root/"model.joblib"),"variants":rows,"external_refit":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}; _write_json(receipt,row); return row

def _vm(row): return {x["variant"]:x for x in row["variants"]}
METRICS={"macro_aupr":lambda m:m["macro_aupr"],"macro_f1":lambda m:m["macro_f1"],"isrj_aupr":lambda m:m["per_label"]["ISRJ"]["aupr"],"isrj_weak_recall":lambda m:m["per_label"]["ISRJ"]["weak_recall"],"isrj_strong_recall":lambda m:m["per_label"]["ISRJ"]["strong_recall"],"isrj_weak_auroc":lambda m:m["per_label"]["ISRJ"]["weak_vs_negative_auroc"],"isrj_fpr":lambda m:m["per_label"]["ISRJ"]["negative_fpr"],"sawfm_aupr":lambda m:m["per_label"]["SawFM"]["aupr"],"sawfm_weak_recall":lambda m:m["per_label"]["SawFM"]["weak_recall"]}

def run(args):
 audit=json.loads(args.audit.read_text(encoding="utf-8")); seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"));
 if not audit["identity_audit_passed"] or not audit["parameter_ood_passed"] or audit["sha256"].lower()!=DATA_SHA: raise ValueError("external audit failed")
 if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]: raise ValueError("test-v3 is not sealed")
 labels=read_labels(args.data_path); values,_=read_parameters(args.data_path); weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db); signals=_load_signals(args.data_path,np.arange(len(labels)),args.io_batch_size)
 with h5py.File(args.data_path,'r') as f: receiver=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
 device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); args.output_root.mkdir(parents=True,exist_ok=True); rows=[_run_seed(args,labels,values,weak,signals,receiver,s,device) for s in args.seeds]
 aggregate={v:{m:_mean_ci([fn(_vm(r)[v]["metrics"]) for r in rows],20333000+vi*100+mi) for mi,(m,fn) in enumerate(METRICS.items())} for vi,v in enumerate(VARIANTS)}; per_seed=[]
 for row in rows:
  a=_vm(row)[VARIANTS[0]]["metrics"]; b=_vm(row)[VARIANTS[1]]["metrics"]; per_seed.append({"seed":row["seed"],**{m:fn(b)-fn(a) for m,fn in METRICS.items()}})
 delta={m:_mean_ci([r[m] for r in per_seed],20334000+i) for i,m in enumerate(METRICS)}; delta["positive_weak_recall_seeds"]=sum(r["isrj_weak_recall"]>0 for r in per_seed); delta["positive_weak_auroc_seeds"]=sum(r["isrj_weak_auroc"]>0 for r in per_seed)
 checks={"isrj_aupr_positive":delta["isrj_aupr"]["mean"]>0,"weak_recall_ge_0p030":delta["isrj_weak_recall"]["mean"]>=.03,"weak_auroc_positive":delta["isrj_weak_auroc"]["mean"]>0,"four_of_five_recall":delta["positive_weak_recall_seeds"]>=4,"four_of_five_auroc":delta["positive_weak_auroc_seeds"]>=4,"b2_fpr_le_0p25":aggregate[VARIANTS[1]]["isrj_fpr"]["mean"]<=.25,"macro_aupr_nonnegative":delta["macro_aupr"]["mean"]>=0,"sawfm_exact":abs(delta["sawfm_aupr"]["mean"])<1e-10 and abs(delta["sawfm_weak_recall"]["mean"])<1e-10,"no_external_refit":all(not r["external_refit"] for r in rows),"test_boundary_safe":all(not r["test_v3_used"] for r in rows)}
 result={"status":"completed","experiment":"RouteA_ISRJ_ParameterOOD_v1_Frozen","seeds":args.seeds,"external_data_sha256":DATA_SHA,"aggregate":aggregate,"b2_vs_b0":{"per_seed":per_seed,"aggregate":delta},"external_gate":{"passed":bool(all(checks.values())),"checks":checks},"external_refit":False,"locked_test_used":False,"test_v3_used":False}; out=args.output_root/"route_a_isrj_parameter_ood_v1_frozen_summary.json"; _write_json(out,result); print(json.dumps({"status":"completed","gate":result["external_gate"],"summary":str(out)},ensure_ascii=False,indent=2)); return result

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0801_isrj_ood_v1\closedset_6class_train.mat')); p.add_argument('--development-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat')); p.add_argument('--audit',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_isrj_parameter_ood_v1_generation/identity_parameter_audit.json'); p.add_argument('--f2-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_dfms_iq_structure_screen'); p.add_argument('--c6-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_label_selective_projection'); p.add_argument('--unified-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3'); p.add_argument('--convergence-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_loss_convergence_audit'); p.add_argument('--head-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines'); p.add_argument('--increment-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_h1_periodicity_increment'); p.add_argument('--old-f3-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3'); p.add_argument('--periodicity-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_isrj_joint_periodicity_expert_v3'); p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json'); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_isrj_parameter_ood_v1_frozen'); p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]); p.add_argument('--io-batch-size',type=int,default=128); p.add_argument('--model-batch-size',type=int,default=128); p.add_argument('--periodicity-batch-size',type=int,default=64); p.add_argument('--raw-batch-size',type=int,default=64); p.add_argument('--raw-hidden-channels',type=int,default=16); p.add_argument('--raw-topk',type=int,default=16); p.add_argument('--raw-dropout',type=float,default=.1); p.add_argument('--weak-threshold-db',type=float,default=10.); p.add_argument('--device',default=None); p.add_argument('--resume',action='store_true'); run(p.parse_args())
