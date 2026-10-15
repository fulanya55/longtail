"""Coarse 8x8 VFT STFT separability correction on development-v3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels,read_parameters  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa:E402
from experiments.run_route_a_development_v2_h5 import _sha256,_write_json  # noqa:E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa:E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _load_roles,_matched_view,_rank_metrics,_scores,_train_probe  # noqa:E402
from experiments.run_route_a_raw_iq_physics_separability_v3 import _vft_features  # noqa:E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES  # noqa:E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256,_validate_config  # noqa:E402
from experiments.run_shared import _data_path,load_config  # noqa:E402


TARGET_INDEX=int(TARGET_INDICES[1]);TARGET_NAME="VFT"


def _extract(signals,device,batch_size,grid):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(signals),int(batch_size)):
            batch=torch.from_numpy(signals[start:start+int(batch_size)]).to(device)
            rows.append(_vft_features(batch,grid_size=grid).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def _write_history(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _run_seed(args,config,weak6,seed,device,data_sha):
    seed_root=args.output_root/f"seed_{seed}";summary_path=seed_root/"seed_summary.json"
    if args.resume and summary_path.is_file():return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True,exist_ok=True)
    roles,role_indices_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    features={role:_extract(roles[role]["signals"],device,args.feature_batch_size,args.grid_size) for role in ("fit","validation","early")}
    mean=features["fit"].mean(axis=0).astype(np.float32);std=np.maximum(features["fit"].std(axis=0),1e-5).astype(np.float32)
    normalized={role:((value-mean[None,:])/std[None,:]).astype(np.float32) for role,value in features.items()}
    views={};counts={}
    for role in ("fit","validation","early"):
        selected,y,receipt=_matched_view(roles[role]["targets6"],weak6[roles[role]["sample_ids"]],TARGET_INDEX)
        views[role]=(normalized[role][selected],y);counts[role]=receipt
    train_seed=995000+int(seed)
    probe,history,info=_train_probe(*views["fit"],*views["validation"],train_seed,device,args)
    validation=_rank_metrics(views["validation"][1],_scores(probe,views["validation"][0],device))
    early=_rank_metrics(views["early"][1],_scores(probe,views["early"][0],device))
    prior=json.loads((args.raw_physics_root/f"seed_{seed}"/"seed_summary.json").read_text(encoding="utf-8"))["labels"][TARGET_NAME]
    layer=json.loads((args.layerwise_root/f"seed_{seed}"/"seed_summary.json").read_text(encoding="utf-8"))["representations"]["T_FinalTemporal"][TARGET_NAME]
    result={"validation":validation,"early":early,"feature_dim":int(normalized["fit"].shape[1]),
            "validation_early_auroc_gap":validation["auroc"]-early["auroc"],
            "v0_16x16_early_auroc":prior["early"]["auroc"],
            "v0_16x16_gap":prior["validation_early_auroc_gap"],
            "delta_vs_v0":early["auroc"]-prior["early"]["auroc"],
            "gap_reduction_vs_v0":prior["validation_early_auroc_gap"]-(validation["auroc"]-early["auroc"]),
            "final_temporal_early_auroc":layer["early"]["auroc"],
            "delta_vs_final_temporal":early["auroc"]-layer["early"]["auroc"],**info}
    torch.save({"model_state_dict":probe.state_dict(),"mean":mean,"std":std,
                "grid_size":int(args.grid_size),**info},seed_root/"best_probe.pth")
    _write_history(seed_root/"history.csv",history)
    payload={"seed":int(seed),"status":"completed","smoke":bool(args.smoke),"data_sha256":data_sha,
             "role_indices_sha256":_sha256(role_indices_path),"result":result,"matched_counts":counts,
             "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(summary_path,payload);return payload


def run(args):
    config=load_config(args.config);_validate_config(config);data_path=_data_path(config);data_sha=_sha256(data_path)
    if data_sha.lower()!=EXPECTED_V3_SHA256:raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:raise ValueError("test-v3 boundary is not sealed")
    labels6=read_labels(data_path);values,_=read_parameters(data_path)
    weak6=compute_weak_component_mask(labels6,_component_power(values),args.weak_threshold_db)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"));seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,weak6,seed,device,data_sha) for seed in seeds]
    names=("early_auroc","early_aupr","validation_early_auroc_gap","delta_vs_v0","gap_reduction_vs_v0","delta_vs_final_temporal")
    values_by_name={"early_auroc":[row["result"]["early"]["auroc"] for row in rows],
                    "early_aupr":[row["result"]["early"]["aupr"] for row in rows]}
    for name in names[2:]:values_by_name[name]=[row["result"][name] for row in rows]
    aggregate={name:_mean_ci(values_by_name[name],20274100+i) for i,name in enumerate(names)}
    positive=int(sum(value>0 for value in values_by_name["delta_vs_final_temporal"]))
    checks={} if args.smoke else {"absolute_ge_0p620":aggregate["early_auroc"]["mean"]>=.620,
                                 "gain_vs_final_ge_0p030":aggregate["delta_vs_final_temporal"]["mean"]>=.030,
                                 "four_of_five_positive":positive>=4,
                                 "noninferior_vs_v0":aggregate["delta_vs_v0"]["mean"]>=-.010,
                                 "gap_lt_0p150":aggregate["validation_early_auroc_gap"]["mean"]<.150,
                                 "gap_reduction_ge_0p040":aggregate["gap_reduction_vs_v0"]["mean"]>=.040}
    gate={"checks":checks,"passed":bool(checks and all(checks.values())),"positive_vs_final_seeds":positive}
    payload={"status":"completed","experiment":"RouteA_VFTCoarseSTFTSeparability_v3","smoke":bool(args.smoke),
             "seeds":list(seeds),"data_sha256":data_sha,"grid_size":int(args.grid_size),"aggregate":aggregate,
             "promotion_gate":gate,"promoted":bool(gate["passed"]),"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True);_write_json(args.output_root/"route_a_vft_coarse_stft_separability_v3_summary.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2));return payload


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--raw-physics-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_raw_iq_physics_separability_v3")
    parser.add_argument("--layerwise-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_layerwise_matched_negative_separability_v3")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_vft_coarse_stft_separability_v3")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]);parser.add_argument("--device",default=None)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--smoke",action="store_true");parser.add_argument("--smoke-epochs",type=int,default=2)
    parser.add_argument("--grid-size",type=int,default=8);parser.add_argument("--io-batch-size",type=int,default=256);parser.add_argument("--feature-batch-size",type=int,default=32)
    parser.add_argument("--batch-size",type=int,default=256);parser.add_argument("--learning-rate",type=float,default=1e-3)
    parser.add_argument("--epochs",type=int,default=80);parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--weight-decay",type=float,default=1e-4);parser.add_argument("--gradient-clip",type=float,default=5.0)
    parser.add_argument("--weak-threshold-db",type=float,default=10.0);return parser.parse_args()


if __name__=="__main__":run(parse_args())
