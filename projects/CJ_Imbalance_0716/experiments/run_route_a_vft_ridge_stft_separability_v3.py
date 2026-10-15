"""Validation-independent Ridge classifier on fixed VFT STFT features."""

from __future__ import annotations

import argparse,json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels,read_parameters  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa:E402
from experiments.run_route_a_development_v2_h5 import _sha256,_write_json  # noqa:E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa:E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles,_matched_view,_rank_metrics  # noqa:E402
from experiments.run_route_a_raw_iq_physics_separability_v3 import _vft_features  # noqa:E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES  # noqa:E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256,_validate_config  # noqa:E402
from experiments.run_shared import _data_path,load_config  # noqa:E402

TARGET_INDEX=int(TARGET_INDICES[1])

def _extract(signals,device,batch_size):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(signals),int(batch_size)):
            batch=torch.from_numpy(signals[start:start+int(batch_size)]).to(device)
            rows.append(_vft_features(batch,grid_size=16).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)

def _run_seed(args,config,weak6,seed,device,data_sha):
    root=args.output_root/f"seed_{seed}";summary=root/"seed_summary.json"
    if args.resume and summary.is_file():return json.loads(summary.read_text(encoding="utf-8"))
    root.mkdir(parents=True,exist_ok=True)
    roles,indices_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    features={role:_extract(roles[role]["signals"],device,args.feature_batch_size) for role in ("fit","validation","early")}
    mean=features["fit"].mean(axis=0).astype(np.float32);std=np.maximum(features["fit"].std(axis=0),1e-5).astype(np.float32)
    normalized={role:((value-mean[None,:])/std[None,:]).astype(np.float32) for role,value in features.items()}
    views={};counts={}
    for role in ("fit","validation","early"):
        selected,y,receipt=_matched_view(roles[role]["targets6"],weak6[roles[role]["sample_ids"]],TARGET_INDEX)
        views[role]=(normalized[role][selected],y.astype(np.uint8));counts[role]=receipt
    fit_y=views["fit"][1];positive=max(int(fit_y.sum()),1);negative=max(int((fit_y==0).sum()),1)
    weights=np.where(fit_y==1,0.5/positive,0.5/negative).astype(np.float64)
    model=RidgeClassifier(alpha=float(args.ridge_alpha),fit_intercept=True)
    model.fit(views["fit"][0],fit_y,sample_weight=weights)
    validation=_rank_metrics(views["validation"][1],model.decision_function(views["validation"][0]))
    early=_rank_metrics(views["early"][1],model.decision_function(views["early"][0]))
    v0=json.loads((args.raw_physics_root/f"seed_{seed}"/"seed_summary.json").read_text(encoding="utf-8"))["labels"]["VFT"]
    final=json.loads((args.layerwise_root/f"seed_{seed}"/"seed_summary.json").read_text(encoding="utf-8"))["representations"]["T_FinalTemporal"]["VFT"]
    np.savez_compressed(root/"ridge_model.npz",coef=np.asarray(model.coef_,dtype=np.float32),
                        intercept=np.asarray(model.intercept_,dtype=np.float32),mean=mean,std=std)
    result={"validation":validation,"early":early,"validation_early_auroc_gap":validation["auroc"]-early["auroc"],
            "delta_vs_v0":early["auroc"]-v0["early"]["auroc"],
            "delta_vs_final_temporal":early["auroc"]-final["early"]["auroc"],
            "feature_dim":int(normalized["fit"].shape[1]),"ridge_alpha":float(args.ridge_alpha)}
    payload={"seed":int(seed),"status":"completed","data_sha256":data_sha,"role_indices_sha256":_sha256(indices_path),
             "result":result,"matched_counts":counts,"validation_used_for_selection":False,
             "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(summary,payload);return payload

def run(args):
    config=load_config(args.config);_validate_config(config);path=_data_path(config);sha=_sha256(path)
    if sha.lower()!=EXPECTED_V3_SHA256:raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:raise ValueError("test-v3 boundary is not sealed")
    labels=read_labels(path);values,_=read_parameters(path);weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"));seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,weak,seed,device,sha) for seed in seeds]
    names=("early_auroc","early_aupr","validation_early_auroc_gap","delta_vs_v0","delta_vs_final_temporal")
    vals={"early_auroc":[r["result"]["early"]["auroc"] for r in rows],"early_aupr":[r["result"]["early"]["aupr"] for r in rows]}
    for name in names[2:]:vals[name]=[r["result"][name] for r in rows]
    aggregate={name:_mean_ci(vals[name],20275100+i) for i,name in enumerate(names)}
    positive=int(sum(v>0 for v in vals["delta_vs_final_temporal"]))
    checks={} if args.smoke else {"absolute_ge_0p620":aggregate["early_auroc"]["mean"]>=.620,
                                 "gain_vs_final_ge_0p030":aggregate["delta_vs_final_temporal"]["mean"]>=.030,
                                 "four_of_five_positive":positive>=4,
                                 "gap_lt_0p150":aggregate["validation_early_auroc_gap"]["mean"]<.150,
                                 "noninferior_vs_v0":aggregate["delta_vs_v0"]["mean"]>=-.010}
    gate={"checks":checks,"passed":bool(checks and all(checks.values())),"positive_vs_final_seeds":positive}
    payload={"status":"completed","experiment":"RouteA_VFTRidgeSTFTSeparability_v3","smoke":bool(args.smoke),
             "seeds":list(seeds),"data_sha256":sha,"ridge_alpha":float(args.ridge_alpha),"aggregate":aggregate,
             "promotion_gate":gate,"promoted":bool(gate["passed"]),"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True);_write_json(args.output_root/"route_a_vft_ridge_stft_separability_v3_summary.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2));return payload

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    p.add_argument("--raw-physics-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_raw_iq_physics_separability_v3")
    p.add_argument("--layerwise-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_layerwise_matched_negative_separability_v3")
    p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_vft_ridge_stft_separability_v3")
    p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]);p.add_argument("--device",default=None);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true")
    p.add_argument("--ridge-alpha",type=float,default=10.0);p.add_argument("--io-batch-size",type=int,default=256);p.add_argument("--feature-batch-size",type=int,default=32);p.add_argument("--weak-threshold-db",type=float,default=10.0);return p.parse_args()

if __name__=="__main__":run(parse_args())
