"""Frozen Focal representation: linear decoupling and generic MLP head baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import COMMON_INDICES, _sha256, _target_weak_rank, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power, _macro_aupr  # noqa: E402
from experiments.run_route_a_development_v3_loss_convergence_audit import _load_existing_loss  # noqa: E402
from experiments.run_route_a_development_v3_mainstream_loss_baselines import (  # noqa: E402
    EXPECTED_V3_SHA256, METRICS, _comparison, _elementwise_loss, _fit_statistics,
    _load_model_from_state, _load_rawsaw, _reference_result, _route_rawsaw, _value,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _evaluate, _load_roles, _raw_scores, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


F2_MODEL_NAME = "F2_DFMS_Full"
F0_NAME = "H0_Focal36"
F3_NAME = "R1_F3_Reference"
HEADS = {
    "H1_Linear_BCE": ("linear", "bce"),
    "H2_Linear_DB": ("linear", "db"),
    "H3_MLP4_DB": ("mlp", "db"),
}


class FrozenFeatureHead(nn.Module):
    def __init__(self, input_dim, head_type, hidden_dim=4, labels=6):
        super().__init__()
        self.head_type = head_type
        if head_type == "linear":
            self.head = nn.Linear(input_dim, labels)
        elif head_type == "mlp":
            self.head = nn.ModuleList([
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
                for _ in range(labels)
            ])
        else:
            raise ValueError("head_type must be linear or mlp")

    def forward(self, features):
        if self.head_type == "linear":
            return self.head(features)
        return torch.cat([head(features) for head in self.head], dim=1)


def _extract(model, signals, device, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            x = torch.from_numpy(np.asarray(signals[start:start + batch_size], dtype=np.float32)).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs.append(model.forward_features(x).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _train_head(features, roles, weak6, head_type, loss_mode, seed, device, args, output_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = FrozenFeatureHead(features["fit"].shape[1], head_type, args.hidden_dim).to(device)
    stats = _fit_statistics(roles["fit"]["targets6"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda index:
        args.minimum_lr_ratio + .5 * (1 - args.minimum_lr_ratio) *
        (1 + math.cos(math.pi * min(index / args.epochs, 1.0))))
    dataset = TensorDataset(torch.from_numpy(features["fit"]),
                            torch.from_numpy(roles["fit"]["targets6"].astype(np.float32)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_x = torch.from_numpy(features["validation"]).to(device)
    val_y = roles["validation"]["targets6"]
    val_weak = weak6[roles["validation"]["sample_ids"]]
    best_state, best_key, best_epoch, history = None, None, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            element = _elementwise_loss(loss_mode, logits, y, stats, device)
            loss = element[:, torch.as_tensor(COMMON_INDICES, device=device)].mean() + args.saw_loss_weight * element[:, 1].mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): logits = model(val_x).cpu().numpy()
        prob = 1 / (1 + np.exp(-np.clip(logits[:, COMMON_INDICES], -40, 40)))
        macro = _macro_aupr(val_y[:, COMMON_INDICES], prob)
        weak_rank = _target_weak_rank(val_y[:, COMMON_INDICES], prob, val_weak[:, COMMON_INDICES])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_common_macro_aupr": macro,
                        "validation_target_weak_auroc": weak_rank,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        key = (macro, weak_rank)
        if best_key is None or key[0] > best_key[0] + 1e-8 or (np.isclose(key[0], best_key[0]) and key[1] > best_key[1]):
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        scheduler.step()
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "head_type": head_type, "loss_mode": loss_mode,
                "best_epoch": best_epoch}, output_dir / "best_model.pth")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    return model, {"best_epoch": best_epoch, "best_common_macro_aupr": best_key[0],
                   "best_target_weak_auroc": best_key[1], "train_seconds": time.time() - started,
                   "trainable_parameters": sum(p.numel() for p in model.parameters()),
                   "head_type": head_type, "loss_mode": loss_mode}


def _predict_head(model, features, device, batch_size):
    output=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(features),batch_size):
            output.append(model(torch.from_numpy(features[start:start+batch_size]).to(device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _run_seed(args, config, weak6, seed, device, data_sha):
    root=args.output_root/f"seed_{seed}"; root.mkdir(parents=True,exist_ok=True)
    receipt=root/"seed_summary.json"
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
    roles, role_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    source=args.convergence_root/f"seed_{seed}"/"E2_Focal_36"/"best_model.pth"
    payload=torch.load(source,map_location="cpu",weights_only=False)
    f2_dir=args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}"
    backbone=_load_model_from_state(f2_dir,device,payload["model_state_dict"])
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    features={role:_extract(backbone,values["signals"],device,args.feature_batch_size) for role,values in roles.items()}
    del backbone
    mean=features["fit"].mean(0,keepdims=True); std=np.maximum(features["fit"].std(0,keepdims=True),1e-5)
    features={role:((value-mean)/std).astype(np.float32) for role,value in features.items()}
    raw,raw_path=_load_rawsaw(args,seed,device)
    raw_scores={role:_raw_scores(raw,values["signals"],device,args.raw_batch_size) for role,values in roles.items()}; del raw
    results=[_load_existing_loss(args.convergence_root,seed,"E2_Focal_36",F0_NAME),
             _reference_result(args.f3_root,seed,"F3_AlignedPeriodicityFusionHead",F3_NAME)]
    training={}; hashes={}
    for index,(name,(head_type,loss_mode)) in enumerate(HEADS.items()):
        head,training[name]=_train_head(features,roles,weak6,head_type,loss_mode,seed+91000+index*1000,device,args,root/name)
        logits={role:_predict_head(head,value,device,args.eval_batch_size) for role,value in features.items()}
        results.append(_evaluate(name,_route_rawsaw(logits,raw_scores),roles,weak6,root/name,args))
        hashes[name]=_sha256(root/name/"best_model.pth")
    row={"seed":seed,"status":"completed","data_sha256":data_sha,"role_indices_sha256":_sha256(role_path),
         "focal36_checkpoint_sha256":_sha256(source),"rawsaw_checkpoint_sha256":_sha256(raw_path),
         "feature_mean_sha256":__import__("hashlib").sha256(mean.tobytes()).hexdigest(),
         "feature_std_sha256":__import__("hashlib").sha256(std.tobytes()).hexdigest(),
         "training":training,"model_sha256":hashes,"variants":results,
         "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(receipt,row); return row


def _variant_map(row): return {item["variant"]:item for item in row["variants"]}


def run(args):
    config=load_config(args.config); _validate_config(config); data_path=_data_path(config); data_sha=_sha256(data_path)
    if data_sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]: raise ValueError("test-v3 is not sealed")
    labels=read_labels(data_path); values,_=read_parameters(data_path); weak=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); args.output_root.mkdir(parents=True,exist_ok=True)
    _write_json(args.output_root/"experiment_config.json",{"experiment":"RouteA_DevelopmentV3_FrozenHeadCapacityBaselines","heads":HEADS,"seeds":args.seeds,"epochs":args.epochs,"locked_test_used":False,"test_v3_used":False})
    rows=[_run_seed(args,config,weak,seed,device,data_sha) for seed in args.seeds]
    names=(F0_NAME,)+tuple(HEADS)+(F3_NAME,)
    aggregate={name:{metric:_mean_ci([_value(_variant_map(row)[name],metric) for row in rows],20326000+ni*100+mi) for mi,metric in enumerate(METRICS)} for ni,name in enumerate(names)}
    strongest=max((F0_NAME,)+tuple(HEADS),key=lambda name:(aggregate[name]["macro_aupr"]["mean"],aggregate[name]["target_weak_auroc"]["mean"]))
    comparison=_comparison(rows,F3_NAME,strongest,20327000); d=comparison["aggregate"]
    checks={"f3_isrj_aupr_positive":d["isrj_aupr"]["mean"]>0,"f3_isrj_weak_recall_ge_0p030":d["isrj_weak_recall"]["mean"]>=.03,
            "f3_isrj_weak_auroc_ge_0p010":d["isrj_weak_auroc"]["mean"]>=.01,"four_of_five_positive_weak_recall":d["positive_isrj_weak_recall_seeds"]>=4,
            "four_of_five_positive_weak_auroc":d["positive_isrj_weak_auroc_seeds"]>=4,"f3_isrj_fpr_le_0p20":aggregate[F3_NAME]["isrj_fpr"]["mean"]<=.2,
            "test_boundary_safe":all(not row["test_v3_used"] for row in rows)}
    result={"status":"completed","experiment":"RouteA_DevelopmentV3_FrozenHeadCapacityBaselines","seeds":args.seeds,"data_sha256":data_sha,
            "aggregate":aggregate,"strongest_generic_head_baseline":strongest,"f3_vs_strongest":comparison,
            "evidence_gate":{"passed":bool(all(checks.values())),"checks":checks},"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    path=args.output_root/"route_a_development_v3_frozen_head_capacity_baselines_summary.json"; _write_json(path,result)
    print(json.dumps({"status":"completed","strongest":strongest,"gate":result["evidence_gate"],"summary":str(path)},ensure_ascii=False,indent=2)); return result


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml")
    p.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen"); p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection")
    p.add_argument("--unified-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); p.add_argument("--convergence-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_loss_convergence_audit")
    p.add_argument("--f3-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines"); p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043])
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--io-batch-size",type=int,default=128); p.add_argument("--feature-batch-size",type=int,default=128); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--raw-batch-size",type=int,default=64)
    p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--minimum-lr-ratio",type=float,default=.01); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--gradient-clip",type=float,default=5.); p.add_argument("--saw-loss-weight",type=float,default=.2); p.add_argument("--hidden-dim",type=int,default=4); p.add_argument("--max-negative-fpr",type=float,default=.2); p.add_argument("--weak-threshold-db",type=float,default=10.); p.add_argument("--raw-hidden-channels",type=int,default=16); p.add_argument("--raw-topk",type=int,default=16); p.add_argument("--raw-dropout",type=float,default=.1); p.add_argument("--device",default=None); p.add_argument("--resume",action="store_true"); return p.parse_args()


if __name__=="__main__": run(parse_args())
