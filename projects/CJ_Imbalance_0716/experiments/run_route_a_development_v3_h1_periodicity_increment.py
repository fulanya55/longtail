"""Test aligned periodicity increment on the strongest frozen H1 linear-BCE baseline."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import joblib, numpy as np, torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_development_v3_frozen_head_capacity_baselines import FrozenFeatureHead, _extract as _extract_backbone_features, _predict_head
from experiments.run_route_a_development_v3_mainstream_loss_baselines import EXPECTED_V3_SHA256, _load_model_from_state, _load_rawsaw, _route_rawsaw
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _fit_head, _head_scores, _replace_isrj, _sample_weights
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _block_permutation, _block_shuffle, _extract
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, METRIC_PATHS, _clipped_logit, _compare, _fit_score_stats, _load_expert, _metric, _standardize_scores, _variant_map
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _evaluate, _load_roles, _raw_scores, _validate_config
from experiments.run_shared import _data_path, load_config

VARIANTS=("B0_H1","B1_H1OnlyWeightedHead","B2_H1AlignedPeriodicity","B3_H1BlockShuffled")
F2_MODEL_NAME="F2_DFMS_Full"

def _run_seed(args,config,weak6,seed,device,data_sha):
 root=args.output_root/f"seed_{seed}"; receipt=root/"seed_summary.json"
 if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
 root.mkdir(parents=True,exist_ok=True); roles,role_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
 focal_path=args.convergence_root/f"seed_{seed}"/"E2_Focal_36"/"best_model.pth"; state=torch.load(focal_path,map_location="cpu",weights_only=False)["model_state_dict"]
 f2_dir=args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}"; backbone=_load_model_from_state(f2_dir,device,state)
 features={role:_extract_backbone_features(backbone,v["signals"],device,args.backbone_batch_size) for role,v in roles.items()}; del backbone
 mean=features["fit"].mean(0,keepdims=True); std=np.maximum(features["fit"].std(0,keepdims=True),1e-5); features={r:((x-mean)/std).astype(np.float32) for r,x in features.items()}
 head_path=args.head_root/f"seed_{seed}"/"H1_Linear_BCE"/"best_model.pth"; hp=torch.load(head_path,map_location="cpu",weights_only=False); head=FrozenFeatureHead(128,"linear").to(device); head.load_state_dict(hp["model_state_dict"]); head.eval()
 base={role:_predict_head(head,x,device,args.eval_batch_size) for role,x in features.items()}; del head
 raw,raw_path=_load_rawsaw(args,seed,device); raw_scores={role:_raw_scores(raw,v["signals"],device,args.raw_batch_size) for role,v in roles.items()}; del raw; base=_route_rawsaw(base,raw_scores)
 saved=np.load(args.head_root/f"seed_{seed}"/"H1_Linear_BCE"/"predictions.npz"); reconstruction=float(np.max(np.abs(base["early"]-saved["logits"])))
 aligned_model,aligned_mean,aligned_std,aligned_hashes=_load_expert(args.periodicity_root,seed,"H_AlignedHGB"); shuffled_model,shuffled_mean,shuffled_std,shuffled_hashes=_load_expert(args.periodicity_root,seed,"S_BlockShuffledHGB")
 permutation=_block_permutation(120,20301000+seed)[0]; aligned_features={r:_extract(roles[r]["signals"],device,args.feature_batch_size) for r in ALL_ROLES}; shuffled_features={r:_extract(_block_shuffle(roles[r]["signals"],permutation,args.block_size),device,args.feature_batch_size) for r in ALL_ROLES}
 ar,sr={},{}
 for r in ALL_ROLES:
  ar[r]=_clipped_logit(aligned_model.predict_proba(((aligned_features[r]-aligned_mean)/np.maximum(aligned_std,1e-5)).astype(np.float32))[:,1]); sr[r]=_clipped_logit(shuffled_model.predict_proba(((shuffled_features[r]-shuffled_mean)/np.maximum(shuffled_std,1e-5)).astype(np.float32))[:,1])
 selected,_,matched=_matched_view(roles["fit"]["targets6"],weak6[roles["fit"]["sample_ids"]],ISRJ_INDEX); astats=_fit_score_stats(ar["fit"][selected]); sstats=_fit_score_stats(sr["fit"][selected]); az={r:_standardize_scores(ar[r],astats) for r in ALL_ROLES}; sz={r:_standardize_scores(sr[r],sstats) for r in ALL_ROLES}
 sets={"h1":{r:_fusion_features(base[r]) for r in ALL_ROLES},"aligned":{r:_fusion_features(base[r],az[r]) for r in ALL_ROLES},"shuffled":{r:_fusion_features(base[r],sz[r]) for r in ALL_ROLES}}
 labels=roles["fit"]["targets6"][:,ISRJ_INDEX].astype(np.uint8); weights=_sample_weights(roles["fit"]["targets6"],weak6[roles["fit"]["sample_ids"]],args.weak_positive_weight); heads={}
 for offset,name in enumerate(("h1","aligned","shuffled")):
  model,m,s=_fit_head(sets[name]["fit"],labels,weights,20328000+seed+offset,args.c_value); out=root/f"head_{name}"; out.mkdir(parents=True,exist_ok=True); joblib.dump(model,out/"model.joblib"); np.savez_compressed(out/"normalization.npz",mean=m,std=s); heads[name]=(model,m,s)
 logits={VARIANTS[0]:base,
         VARIANTS[1]:{r:_replace_isrj(base[r],_head_scores(heads["h1"][0],sets["h1"][r],heads["h1"][1],heads["h1"][2])) for r in ALL_ROLES},
         VARIANTS[2]:{r:_replace_isrj(base[r],_head_scores(heads["aligned"][0],sets["aligned"][r],heads["aligned"][1],heads["aligned"][2])) for r in ALL_ROLES},
         VARIANTS[3]:{r:_replace_isrj(base[r],_head_scores(heads["shuffled"][0],sets["shuffled"][r],heads["shuffled"][1],heads["shuffled"][2])) for r in ALL_ROLES}}
 np.savez_compressed(root/"all_role_logits.npz",**{
  f"{variant}__{role}":logits[variant][role]
  for variant in VARIANTS for role in ALL_ROLES})
 results=[_evaluate(name,logits[name],roles,weak6,root/name,args) for name in VARIANTS]
 row={"seed":seed,"status":"completed","data_sha256":data_sha,"role_indices_sha256":_sha256(role_path),"focal_checkpoint_sha256":_sha256(focal_path),"h1_checkpoint_sha256":_sha256(head_path),"rawsaw_checkpoint_sha256":_sha256(raw_path),"h1_reconstruction_max_abs_error":reconstruction,"periodicity_expert_sha256":{"aligned":aligned_hashes,"shuffled":shuffled_hashes},"matched_receipt":matched,"variants":results,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}; _write_json(receipt,row); return row

def run(args):
 config=load_config(args.config); _validate_config(config); path=_data_path(config); sha=_sha256(path)
 if sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
 seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"));
 if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]: raise ValueError("test-v3 is not sealed")
 labels=read_labels(path); params,_=read_parameters(path); weak=compute_weak_component_mask(labels,_component_power(params),args.weak_threshold_db); device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); args.output_root.mkdir(parents=True,exist_ok=True); rows=[_run_seed(args,config,weak,s,device,sha) for s in args.seeds]
 aggregate={v:{name:_mean_ci([_metric(_variant_map(r)[v],p) for r in rows],20329000+vi*100+i) for i,(name,p) in enumerate(METRIC_PATHS.items())} for vi,v in enumerate(VARIANTS)}; comps={"B2_vs_B0":_compare(rows,VARIANTS[2],VARIANTS[0],20330000),"B2_vs_B1":_compare(rows,VARIANTS[2],VARIANTS[1],20331000),"B2_vs_B3":_compare(rows,VARIANTS[2],VARIANTS[3],20332000)}; b0=comps["B2_vs_B0"]["aggregate"]; b1=comps["B2_vs_B1"]["aggregate"]; b3=comps["B2_vs_B3"]["aggregate"]
 checks={"weak_recall_vs_h1_ge_0p010":b0["isrj_weak_recall"]["mean"]>=.01,"weak_auroc_vs_h1_ge_0p005":b0["isrj_weak_auroc"]["mean"]>=.005,"aupr_vs_h1_nonnegative":b0["isrj_aupr"]["mean"]>=0,"weak_auroc_vs_h1only_ge_0p005":b1["isrj_weak_auroc"]["mean"]>=.005,"weak_auroc_vs_shuffle_ge_0p005":b3["isrj_weak_auroc"]["mean"]>=.005,"four_of_five_positive_recall":b0["positive_isrj_weak_recall_seeds"]>=4,"four_of_five_positive_auroc":b0["positive_isrj_weak_auroc_seeds"]>=4,"fpr_le_0p20":aggregate[VARIANTS[2]]["isrj_negative_fpr"]["mean"]<=.2,"boundary_safe":all(r["h1_reconstruction_max_abs_error"]<=1e-5 and not r["test_v3_used"] for r in rows)}
 result={"status":"completed","experiment":"RouteA_DevelopmentV3_H1PeriodicityIncrement","seeds":args.seeds,"data_sha256":sha,"aggregate":aggregate,"comparisons":comps,"promotion_gate":{"passed":bool(all(checks.values())),"checks":checks},"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}; out=args.output_root/"route_a_development_v3_h1_periodicity_increment_summary.json"; _write_json(out,result); print(json.dumps({"status":"completed","gate":result["promotion_gate"],"summary":str(out)},ensure_ascii=False,indent=2)); return result

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml"); p.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen"); p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--unified-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); p.add_argument("--convergence-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_loss_convergence_audit"); p.add_argument("--head-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines"); p.add_argument("--periodicity-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_h1_periodicity_increment"); p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]); p.add_argument("--device",default=None); p.add_argument("--resume",action="store_true"); p.add_argument("--io-batch-size",type=int,default=256); p.add_argument("--backbone-batch-size",type=int,default=128); p.add_argument("--feature-batch-size",type=int,default=64); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--raw-batch-size",type=int,default=64); p.add_argument("--raw-hidden-channels",type=int,default=16); p.add_argument("--raw-topk",type=int,default=16); p.add_argument("--raw-dropout",type=float,default=.1); p.add_argument("--block-size",type=int,default=25); p.add_argument("--c-value",type=float,default=.1); p.add_argument("--weak-positive-weight",type=float,default=4.); p.add_argument("--max-negative-fpr",type=float,default=.2); p.add_argument("--weak-threshold-db",type=float,default=10.); return p.parse_args()
if __name__=="__main__": run(parse_args())
