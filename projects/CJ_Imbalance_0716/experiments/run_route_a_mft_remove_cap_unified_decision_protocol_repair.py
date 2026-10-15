"""Role-correct implementation repair for the frozen MFT cap ablation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import torch  # Preload DLLs on Windows.
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.engine.thresholding import select_per_label_weak_recall_fpr_thresholds  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_mft_remove_cap_unified_decision_ablation import EVIDENCE_KEY, MFT, NON_MFT, SOURCE_Q1, SOURCE_Q4, _comparison, _with_hard_metrics  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import guarded_logits  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _apply_platt, _evaluate, _fit_platt, _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402

ARMS=("Q4_Cap075","Q4_NaturalCap100")

def _threshold_role_record(logits, roles, weak, args):
    calibrator=_fit_platt(logits["calibration"],roles["calibration"]["targets6"])
    probability=_apply_platt(calibrator,logits["threshold"])
    selection=select_per_label_weak_recall_fpr_thresholds(roles["threshold"]["targets6"],probability,weak_component_mask=weak[roles["threshold"]["sample_ids"]],max_negative_fpr=args.max_negative_fpr,source_role="threshold_selection")
    threshold=float(np.asarray(selection.thresholds)[MFT]); negative=roles["threshold"]["targets6"][:,MFT]==0
    fpr=float((probability[negative,MFT]>=threshold).mean())
    return {"mft_threshold":threshold,"mft_threshold_role_fpr":fpr,"threshold_negative_count":int(negative.sum()),"source_role":"threshold_selection"}

def _run_seed(args,config,weak,seed,data_sha):
    root=args.output_root/f"seed_{seed}"; receipt=root/"seed_summary.json"
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True,exist_ok=True)
    roles,role_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    source_path=args.construction_root/f"seed_{seed}"/"all_role_logits.npz"; evidence_path=args.mft_evidence_root/f"seed_{seed}"/"experts"/"all_role_expert_evidence.npz"
    source,evidence=np.load(source_path),np.load(evidence_path)
    q1={role:np.asarray(source[f"{SOURCE_Q1}__{role}"],dtype=np.float32) for role in ALL_ROLES}; q4={role:np.asarray(source[f"{SOURCE_Q4}__{role}"],dtype=np.float32) for role in ALL_ROLES}
    cap075={};cap100={};recon={};guard={}
    for role in ALL_ROLES:
        aligned=np.asarray(evidence[f"{EVIDENCE_KEY}__{role}"],dtype=np.float32);cap075[role],_=guarded_logits(q1[role],aligned,.75);cap100[role],_=guarded_logits(q1[role],aligned,1.)
        recon[role]=float(np.max(np.abs(cap075[role][:,MFT]-q4[role][:,MFT])));guard[role]=float(np.max(np.abs(cap100[role][:,NON_MFT]-q1[role][:,NON_MFT])));cap100[role][:,NON_MFT]=q4[role][:,NON_MFT]
    if max(recon.values())>1e-6 or max(guard.values())!=0.: raise AssertionError("frozen cap construction mismatch")
    logits={ARMS[0]:q4,ARMS[1]:cap100}; variants=[]; threshold_role={}
    for arm in ARMS:
        arm_root=root/arm; variants.append(_with_hard_metrics(_evaluate(arm,logits[arm],roles,weak,arm_root,args),arm_root));threshold_role[arm]=_threshold_role_record(logits[arm],roles,weak,args)
    row={"status":"completed","seed":int(seed),"data_sha256":data_sha,"role_indices_sha256":_sha256(role_path),"source_logits_sha256":_sha256(source_path),"mft_evidence_sha256":_sha256(evidence_path),"cap075_reconstruction_max_abs":recon,"guard_non_mft_logit_max_abs":guard,"non_mft_logit_max_abs":{role:0. for role in ALL_ROLES},"threshold_role":threshold_role,"variants":variants,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(receipt,row);return row

def run(args):
    config=load_config(args.config);_validate_config(config);path=_data_path(config);data_sha=_sha256(path)
    if data_sha.lower()!=EXPECTED_V3_SHA256:raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[k] for k in ("access_authorized","analysis_allowed","used")):raise ValueError("test-v3 is not sealed")
    labels=read_labels(path);values,_=read_parameters(path);weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db);seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,weak,seed,data_sha) for seed in seeds];comp={} if args.smoke else _comparison(rows)
    checks={"cap075_reconstruction_exact":all(max(row["cap075_reconstruction_max_abs"].values())<=1e-6 for row in rows),"non_mft_logits_exact":all(max(row["non_mft_logit_max_abs"].values())==0. for row in rows)}
    for arm in ARMS: checks[f"{arm}_threshold_role_mft_fpr_le_0p20"]=all(row["threshold_role"][arm]["mft_threshold_role_fpr"]<=.20 for row in rows)
    payload={"status":"completed","experiment":"RouteA_MFTRemoveCapUnifiedDecisionProtocolRepair","smoke":bool(args.smoke),"seeds":list(seeds),"data_sha256":data_sha,"rows":rows,"comparisons":comp,"protocol_integrity_gate":{"checks":checks,"passed":bool(all(checks.values()))},"scope":"frozen_decision_role_contract_repair_no_candidate_selection","supersedes":None,"historical_failed_ablation_preserved":str(PROJECT_ROOT/"outputs/tapc/route_a_mft_remove_cap_unified_decision_ablation/route_a_mft_remove_cap_unified_decision_ablation_summary.json"),"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True);_write_json(args.output_root/"route_a_mft_remove_cap_unified_decision_protocol_repair_summary.json",payload);print(json.dumps(payload,ensure_ascii=False,indent=2));return payload

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml");p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection");p.add_argument("--construction-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_h1_b2_mft_vft_unified_v3");p.add_argument("--mft-evidence-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3");p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json");p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_mft_remove_cap_unified_decision_protocol_repair");p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--io-batch-size",type=int,default=256);p.add_argument("--max-negative-fpr",type=float,default=.2);p.add_argument("--weak-threshold-db",type=float,default=10.0);return p.parse_args()
if __name__=="__main__":run(parse_args())
