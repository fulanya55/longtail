"""Preregistered RAL-only fair baseline on the sealed-role development-v3 protocol."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power, _predict
from experiments.run_route_a_development_v3_loss_convergence_audit import _load_existing_loss
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _comparison, _load_model_from_state, _load_rawsaw, _route_rawsaw, _train_loss_model
from experiments.run_route_a_development_v3_label_selective_projection import EXPECTED_V3_SHA256
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _evaluate, _load_roles, _raw_scores, _validate_config
from experiments.run_shared import _data_path, load_config


RAL, FOCAL = "L8_RAL_RawSaw", "E2_Focal_36"
OFFICIAL = {"paper_doi":"10.1109/iccvw60793.2023.00286", "repository":"WongiPark0628/RAL", "commit":"0d7e7712c82ba9a36a08c888bb529885a871bc1f", "path":"models/get_optimizer.py::Ralloss", "parameters":{"gamma_neg":4.0,"gamma_pos":0.0,"clip":0.05,"eps":1e-8,"lamb":1.5,"epsilon_neg":0.0,"epsilon_pos":1.0,"epsilon_pos_pow":-2.5}, "reduction":"official_sum_divided_by_fixed_batch_label_count"}


def one_seed(args, config, weak, seed, device, data_sha):
    root=args.output_root/f"seed_{seed}"; receipt=root/"seed_summary.json"
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True,exist_ok=True); roles,role_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    if args.smoke: roles={role:{key:value[:args.smoke_train_samples if role=="fit" else args.smoke_eval_samples] for key,value in values.items()} for role,values in roles.items()}
    f2_dir=args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}"; template,f2_path=_load_backbone(f2_dir,F2_MODEL_NAME,device); state=copy.deepcopy({k:v.detach().cpu() for k,v in template.state_dict().items()}); del template
    raw,raw_path=_load_rawsaw(args,seed,device); raw_scores={r:_raw_scores(raw,x["signals"],device,args.raw_batch_size) for r,x in roles.items()}; del raw
    model=_load_model_from_state(f2_dir,device,state); model,training=_train_loss_model(model,RAL,roles,weak,seed,device,args,root/RAL)
    logits={r:_predict(model,x["signals"],device,args.eval_batch_size) for r,x in roles.items()}; result=_evaluate(RAL,_route_rawsaw(logits,raw_scores),roles,weak,root/RAL,args); checkpoint=root/RAL/"best_model.pth"; del model
    variants=[result] if args.smoke else [_load_existing_loss(args.focal_root,seed,FOCAL,FOCAL),result]
    payload={"seed":seed,"status":"completed","smoke":bool(args.smoke),"data_sha256":data_sha,"role_indices_sha256":_sha256(role_path),"f2_checkpoint_sha256":_sha256(f2_path),"rawsaw_checkpoint_sha256":_sha256(raw_path),"ral_checkpoint_sha256":_sha256(checkpoint),"official_formula":OFFICIAL,"training":training,"variants":variants,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}; _write_json(receipt,payload); return payload


def run(args):
    cfg=load_config(args.config); _validate_config(cfg); data=_data_path(cfg); sha=_sha256(data)
    if sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[k] for k in ("access_authorized","analysis_allowed","used")): raise ValueError("test-v3 is not sealed")
    labels=read_labels(data); parameters,_=read_parameters(data); weak=compute_weak_component_mask(labels,_component_power(parameters),args.weak_threshold_db); device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds=args.seeds[:1] if args.smoke else args.seeds; rows=[one_seed(args,cfg,weak,s,device,sha) for s in seeds]
    payload={"status":"completed","experiment":"RouteA_DevelopmentV3_RALBaseline","smoke":bool(args.smoke),"seeds":seeds,"data_sha256":sha,"official_formula":OFFICIAL,"ral_minus_focal36":{} if args.smoke else _comparison(rows,RAL,FOCAL,20804000),"scope":"fair_latest_multilabel_longtail_loss_baseline_no_candidate_selection","locked_test_used":False,"test_v2_used":False,"test_v3_used":False}; args.output_root.mkdir(parents=True,exist_ok=True); _write_json(args.output_root/"route_a_development_v3_ral_loss_baseline_summary.json",payload); print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=ROOT/"configs/student_a/six_class_development_v3.yaml"); p.add_argument("--f2-root",type=Path,default=ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen"); p.add_argument("--c6-root",type=Path,default=ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--unified-root",type=Path,default=ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); p.add_argument("--focal-root",type=Path,default=ROOT/"outputs/tapc/route_a_development_v3_loss_convergence_audit"); p.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=ROOT/"outputs/tapc/route_a_development_v3_ral_loss_baseline"); p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]); p.add_argument("--epochs",type=int,default=36); p.add_argument("--patience",type=int,default=8); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--io-batch-size",type=int,default=128); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--raw-batch-size",type=int,default=64); p.add_argument("--learning-rate",type=float,default=5e-4); p.add_argument("--representation-lr-scale",type=float,default=.1); p.add_argument("--saw-loss-weight",type=float,default=.2); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--gradient-clip",type=float,default=5.); p.add_argument("--max-negative-fpr",type=float,default=.2); p.add_argument("--weak-threshold-db",type=float,default=10.); p.add_argument("--raw-hidden-channels",type=int,default=16); p.add_argument("--raw-topk",type=int,default=16); p.add_argument("--raw-dropout",type=float,default=.1); p.add_argument("--smoke",action="store_true"); p.add_argument("--smoke-epochs",type=int,default=2); p.add_argument("--smoke-train-samples",type=int,default=128); p.add_argument("--smoke-eval-samples",type=int,default=64); p.add_argument("--device",default=None); p.add_argument("--resume",action="store_true"); run(p.parse_args())
