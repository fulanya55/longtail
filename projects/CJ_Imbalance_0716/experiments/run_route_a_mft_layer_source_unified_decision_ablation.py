"""Frozen Global/final/stage2/stage3 MFT source ablation in current Q4 decisions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa:E402
from experiments.run_route_a_development_v2_h5 import _sha256,_write_json  # noqa:E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME,_component_power  # noqa:E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa:E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import REPRESENTATIONS,_extract_representations,_matched_view,_scores  # noqa:E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import _bounded_mft_residual  # noqa:E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256,_evaluate,_load_roles,_validate_config  # noqa:E402
from experiments.run_route_a_vft_strict_complementary_score_fusion_v3 import _fit_score_stats,_load_probe,_normalize_features,_standardize_scores  # noqa:E402
from experiments.run_shared import _data_path,load_config  # noqa:E402

MFT=3; NON_MFT=(0,1,2,4,5); ROLES=("fit","validation","early","calibration","threshold")
ARMS=("P0_Q4_NoLayerExpert","P1_Global","P2_FinalTemporal","P3_Stage2","P4_Stage3")
SOURCE={"P1_Global":"G_Global","P2_FinalTemporal":"T_FinalTemporal","P3_Stage2":"S2_Physics","P4_Stage3":"S3_Physics"}

def _variant(row,arm):return next(item for item in row["variants"] if item["variant"]==arm)
def _metric(item,key):return float(item["metrics"]["per_label"]["MFT"][key])

def _run_seed(args,config,weak,seed,device,data_sha):
    root=args.output_root/f"seed_{seed}";receipt=root/"seed_summary.json"
    if args.resume and receipt.is_file():return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True,exist_ok=True);roles,role_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    stored=np.load(args.construction_root/f"seed_{seed}"/"all_role_logits.npz");q4={role:np.asarray(stored[f"Q4_VFTAlignedRidgeFusion__{role}"],dtype=np.float32) for role in ROLES}
    ckpt=args.c6_root/f"seed_{seed}"/"C6_Joint"/"best_model.pth";state=torch.load(ckpt,map_location="cpu",weights_only=False)["model_state_dict"]
    model=_load_model_from_state(args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}",device,state);features={role:_extract_representations(model,roles[role],device,args.feature_batch_size,args.topk) for role in ROLES};del model
    if device.type=="cuda":torch.cuda.empty_cache()
    selected,_,match=_matched_view(roles["fit"]["targets6"],weak[roles["fit"]["sample_ids"]],MFT)
    logits={ARMS[0]:q4};expert={}
    for arm,rep in SOURCE.items():
        path=args.layerwise_root/f"seed_{seed}"/rep/"MFT"/"best_probe.pth";probe,payload=_load_probe(path,device)
        raw={role:_scores(probe,_normalize_features(features[role][rep],payload),device) for role in ROLES};stats=_fit_score_stats(raw["fit"][selected]);z={role:_standardize_scores(raw[role],stats) for role in ROLES}
        values={role:_bounded_mft_residual(q4[role],z[role],args.residual_cap,args.uncertainty_power)[0] for role in ROLES}
        if any(np.max(np.abs(values[role][:,NON_MFT]-q4[role][:,NON_MFT]))!=0. for role in ROLES):raise AssertionError("layer expert changed non-MFT logit")
        logits[arm]=values;expert[arm]={"representation":rep,"probe_sha256":_sha256(path),"fit_score_stats":stats}
    variants=[_evaluate(arm,logits[arm],roles,weak,root/arm,args) for arm in ARMS]
    row={"status":"completed","seed":int(seed),"data_sha256":data_sha,"role_indices_sha256":_sha256(role_path),"c6_checkpoint_sha256":_sha256(ckpt),"construction_logits_sha256":_sha256(args.construction_root/f"seed_{seed}"/"all_role_logits.npz"),"experts":expert,"matched_fit_receipt":match,"residual":{"cap":args.residual_cap,"uncertainty_power":args.uncertainty_power,"new_trainable_parameters":0},"non_mft_logits_exact":True,"variants":variants,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False};_write_json(receipt,row);return row

def _comparison(rows,arm,offset):
    out={}
    for i,key in enumerate(("weak_vs_negative_auroc","aupr","weak_recall","strong_recall","negative_fpr")):
        values=[_metric(_variant(row,arm),key)-_metric(_variant(row,ARMS[0]),key) for row in rows];out[key]={"summary":_mean_ci(values,offset+i),"values":values,"positive_seeds":int(sum(v>0 for v in values))}
    return out

def run(args):
    config=load_config(args.config);_validate_config(config);path=_data_path(config);data_sha=_sha256(path)
    if data_sha.lower()!=EXPECTED_V3_SHA256:raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[k] for k in ("access_authorized","analysis_allowed","used")):raise ValueError("test-v3 is not sealed")
    labels=read_labels(path);values,_=read_parameters(path);weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db);device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"));seeds=args.seeds[:1] if args.smoke else args.seeds;rows=[_run_seed(args,config,weak,seed,device,data_sha) for seed in seeds]
    comparisons={} if args.smoke else {arm:_comparison(rows,arm,2026084700+100*i) for i,arm in enumerate(SOURCE)}
    sources={} if args.smoke else {arm:{"checks":{"weak_auroc_ge_0p005":value["weak_vs_negative_auroc"]["summary"]["mean"]>=.005,"aupr_nonnegative":value["aupr"]["summary"]["mean"]>=0,"four_positive_weak_auroc":value["weak_vs_negative_auroc"]["positive_seeds"]>=4,"mean_early_fpr_le_0p20":float(np.mean([_metric(_variant(row,arm),"negative_fpr") for row in rows]))<=.20,"non_mft_logits_exact":all(row["non_mft_logits_exact"] for row in rows)}} for arm,value in comparisons.items()}
    for value in sources.values():value["passed"] = bool(all(value["checks"].values()))
    payload={"status":"completed","experiment":"RouteA_MFTLayerSourceUnifiedDecisionAblation","smoke":bool(args.smoke),"seeds":list(seeds),"data_sha256":data_sha,"rows":rows,"comparisons_vs_q4":comparisons,"source_gate":{"sources":sources,"passed_sources":[arm for arm,value in sources.items() if value["passed"]]},"scope":"frozen_layer_source_factor_ablation_no_candidate_selection","locked_test_used":False,"test_v2_used":False,"test_v3_used":False};args.output_root.mkdir(parents=True,exist_ok=True);_write_json(args.output_root/"route_a_mft_layer_source_unified_decision_ablation_summary.json",payload);print(json.dumps(payload,ensure_ascii=False,indent=2));return payload

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml");p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection");p.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen");p.add_argument("--construction-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_h1_b2_mft_vft_unified_v3");p.add_argument("--layerwise-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_layerwise_matched_negative_separability_v3_strict");p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json");p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_mft_layer_source_unified_decision_ablation");p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--device",default=None);p.add_argument("--io-batch-size",type=int,default=256);p.add_argument("--feature-batch-size",type=int,default=64);p.add_argument("--topk",type=int,default=8);p.add_argument("--residual-cap",type=float,default=.5);p.add_argument("--uncertainty-power",type=float,default=.5);p.add_argument("--max-negative-fpr",type=float,default=.2);p.add_argument("--weak-threshold-db",type=float,default=10.);return p.parse_args()
if __name__=="__main__":run(parse_args())
