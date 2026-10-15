"""Frozen layerwise weak-positive versus composition-matched-negative probes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_dfms_iq_feature_sufficiency import _stage2_summary, _temporal_summary  # noqa: E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES, TARGET_NAMES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _load_roles,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


REPRESENTATIONS = ("G_Global", "T_FinalTemporal", "S2_Physics", "S3_Physics")


def _matched_view(targets6, weak6, target_index):
    targets = np.asarray(targets6, dtype=np.uint8)
    weak = np.asarray(weak6, dtype=bool)
    index = int(target_index)
    other = [value for value in range(targets.shape[1]) if value != index]
    original_weak_positive = (targets[:, index] == 1) & weak[:, index]
    negative = targets[:, index] == 0
    positive_compositions = {tuple(row) for row in targets[original_weak_positive][:, other].tolist()}
    negative_compositions = {tuple(row) for row in targets[negative][:, other].tolist()}
    matched_compositions = positive_compositions & negative_compositions
    weak_positive = np.asarray([
        bool(original_weak_positive[row])
        and tuple(targets[row, other].tolist()) in matched_compositions
        for row in range(len(targets))
    ], dtype=bool)
    matched_negative = np.asarray([
        bool(negative[row]) and tuple(targets[row, other].tolist()) in matched_compositions
        for row in range(len(targets))
    ], dtype=bool)
    selected = weak_positive | matched_negative
    labels = weak_positive[selected].astype(np.float32)
    return selected, labels, {
        "original_weak_positive_count": int(original_weak_positive.sum()),
        "weak_positive_count": int(weak_positive.sum()),
        "matched_negative_count": int(matched_negative.sum()),
        "positive_composition_count": int(len(positive_compositions)),
        "matched_composition_count": int(len(matched_compositions)),
    }


def _extract_representations(model, role, device, batch_size, topk):
    output = {name: [] for name in REPRESENTATIONS}
    signals = role["signals"]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start+int(batch_size)]).to(device)
            maps = model.forward_structure_maps(batch)
            output["G_Global"].append(maps["global"].float().cpu().numpy().astype(np.float32))
            output["T_FinalTemporal"].append(_temporal_summary(
                maps["temporal"].float().cpu().numpy(), chunk_size=int(batch_size)
            ))
            for name, prefix in (("S2_Physics", "stage2"), ("S3_Physics", "stage3")):
                output[name].append(_stage2_summary(
                    maps[f"{prefix}_real"].float().cpu().numpy(),
                    maps[f"{prefix}_imag"].float().cpu().numpy(),
                    chunk_size=max(1, int(batch_size)//2),
                ))
    return {name: np.concatenate(values).astype(np.float32) for name, values in output.items()}


def _rank_metrics(labels, scores):
    if len(np.unique(labels)) != 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores))}


def _train_probe(fit_x, fit_y, validation_x, validation_y, train_seed, device, args):
    torch.manual_seed(int(train_seed)); np.random.seed(int(train_seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(train_seed))
    model = nn.Linear(fit_x.shape[1], 1).to(device)
    positives = max(float(fit_y.sum()), 1.0); negatives = max(float(len(fit_y)-fit_y.sum()), 1.0)
    pos_weight = torch.tensor(negatives/positives, dtype=torch.float32, device=device)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(int(train_seed)), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_value, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.epochs
    for epoch in range(1, int(epochs)+1):
        model.train(); losses=[]
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device)).squeeze(1); y=y.to(device)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): score=model(validation_tensor).squeeze(1).cpu().numpy()
        metrics=_rank_metrics(validation_y, score)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),
                        "validation_auroc":metrics["auroc"],"validation_aupr":metrics["aupr"]})
        if metrics["auroc"] > best_value + 1e-8:
            best_value,best_epoch,stale=metrics["auroc"],epoch,0
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(args.patience): break
    if best_state is None: raise RuntimeError("probe did not produce checkpoint")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch":best_epoch,"train_seed":int(train_seed),
                            "validation_auroc":best_value}


def _scores(model, values, device, batch_size=512):
    rows=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(values),int(batch_size)):
            rows.append(model(torch.from_numpy(values[start:start+int(batch_size)]).to(device))
                        .squeeze(1).cpu().numpy())
    return np.concatenate(rows)


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
    seed_root=args.output_root/f"seed_{seed}"; summary_path=seed_root/"seed_summary.json"
    if args.resume and summary_path.is_file(): return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True,exist_ok=True)
    roles, role_indices_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    checkpoint_path=args.c6_root/f"seed_{seed}"/"C6_Joint"/"best_model.pth"
    state=torch.load(checkpoint_path,map_location="cpu",weights_only=False)["model_state_dict"]
    model=_load_model_from_state(args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}",device,state)
    features={role:_extract_representations(model,roles[role],device,args.feature_batch_size,args.topk)
              for role in ("fit","validation","early")}
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    results={}; counts={}
    for representation_offset,representation in enumerate(REPRESENTATIONS):
        fit_all=features["fit"][representation]
        mean=fit_all.mean(axis=0).astype(np.float32); std=np.maximum(fit_all.std(axis=0),1e-5).astype(np.float32)
        normalized={role:((features[role][representation]-mean[None,:])/std[None,:]).astype(np.float32)
                    for role in features}
        rep_results={}
        for label_offset,(target_index,target_name) in enumerate(zip(TARGET_INDICES,TARGET_NAMES)):
            views={}; counts.setdefault(target_name,{})
            for role in ("fit","validation","early"):
                selected,y,receipt=_matched_view(roles[role]["targets6"],weak6[roles[role]["sample_ids"]],target_index)
                if len(np.unique(y))!=2: raise ValueError(f"matched view lacks two classes: {role} {target_name}")
                views[role]=(normalized[role][selected],y)
                counts[target_name][role]=receipt
            train_seed=981000+int(seed)+representation_offset*100+label_offset
            probe,history,info=_train_probe(*views["fit"],*views["validation"],train_seed,device,args)
            validation_metrics=_rank_metrics(views["validation"][1],_scores(probe,views["validation"][0],device))
            early_metrics=_rank_metrics(views["early"][1],_scores(probe,views["early"][0],device))
            out=seed_root/representation/target_name; out.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state_dict":probe.state_dict(),"mean":mean,"std":std,
                        "representation":representation,"target":target_name,**info},out/"best_probe.pth")
            _write_history(out/"history.csv",history)
            _write_json(out/"metrics.json",{"validation":validation_metrics,"early":early_metrics,
                                             "feature_dim":int(normalized["fit"].shape[1]),**info})
            rep_results[target_name]={"validation":validation_metrics,"early":early_metrics,
                                      "feature_dim":int(normalized["fit"].shape[1]),**info}
        rep_results["mean"]={role:{metric:float(np.mean([rep_results[name][role][metric] for name in TARGET_NAMES]))
                                   for metric in ("auroc","aupr")} for role in ("validation","early")}
        rep_results["validation_early_auroc_gap"]=rep_results["mean"]["validation"]["auroc"]-rep_results["mean"]["early"]["auroc"]
        results[representation]=rep_results
    payload={"seed":int(seed),"status":"completed","smoke":bool(args.smoke),"data_sha256":data_sha,
             "role_indices_sha256":_sha256(role_indices_path),"c6_checkpoint_sha256":_sha256(checkpoint_path),
             "representations":results,"matched_counts":counts,"locked_test_used":False,
             "test_v2_used":False,"test_v3_used":False}
    _write_json(summary_path,payload); return payload


def run(args):
    config=load_config(args.config); _validate_config(config); data_path=_data_path(config); data_sha=_sha256(data_path)
    if data_sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]: raise ValueError("test-v3 boundary is not sealed")
    labels6=read_labels(data_path); values,_=read_parameters(data_path)
    weak6=compute_weak_component_mask(labels6,_component_power(values),args.weak_threshold_db)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,labels6,weak6,seed,device,data_sha) for seed in seeds]
    aggregate={}
    for ri,representation in enumerate(REPRESENTATIONS):
        aggregate[representation]={
            "early_mean_auroc":_mean_ci([row["representations"][representation]["mean"]["early"]["auroc"] for row in rows],20272100+ri*100),
            "early_mean_aupr":_mean_ci([row["representations"][representation]["mean"]["early"]["aupr"] for row in rows],20272101+ri*100),
            "validation_early_auroc_gap":_mean_ci([row["representations"][representation]["validation_early_auroc_gap"] for row in rows],20272102+ri*100),
            "per_label_early_auroc":{name:_mean_ci([row["representations"][representation][name]["early"]["auroc"] for row in rows],20272110+ri*100+i)
                                      for i,name in enumerate(TARGET_NAMES)},
            "feature_dim":int(rows[0]["representations"][representation][TARGET_NAMES[0]]["feature_dim"]),
        }
    comparisons={}
    baseline="T_FinalTemporal"
    for representation in ("G_Global","S2_Physics","S3_Physics"):
        deltas=[row["representations"][representation]["mean"]["early"]["auroc"]-
                row["representations"][baseline]["mean"]["early"]["auroc"] for row in rows]
        comparisons[f"{representation}_minus_{baseline}"]={"early_mean_auroc":_mean_ci(deltas,20272900+REPRESENTATIONS.index(representation)),
                                                           "positive_seeds":int(sum(value>0 for value in deltas))}
    if args.smoke: gate={"candidates":{},"any_passed":False}
    else:
        candidates={}
        for representation in ("G_Global","S2_Physics","S3_Physics"):
            delta=comparisons[f"{representation}_minus_{baseline}"]
            checks={"gain_ge_0p030":delta["early_mean_auroc"]["mean"]>=.030,
                    "absolute_ge_0p620":aggregate[representation]["early_mean_auroc"]["mean"]>=.620,
                    "four_of_five_positive":delta["positive_seeds"]>=4,
                    "all_labels_ge_0p580":all(aggregate[representation]["per_label_early_auroc"][name]["mean"]>=.580 for name in TARGET_NAMES),
                    "generalization_gap_lt_0p150":aggregate[representation]["validation_early_auroc_gap"]["mean"]<.150}
            candidates[representation]={"checks":checks,"passed":bool(all(checks.values()))}
        gate={"candidates":candidates,"any_passed":bool(any(row["passed"] for row in candidates.values()))}
    payload={"status":"completed","experiment":"RouteA_LayerwiseMatchedNegativeSeparability_v3","smoke":bool(args.smoke),
             "seeds":list(seeds),"data_sha256":data_sha,"aggregate":aggregate,"comparisons":comparisons,
             "promotion_gate":gate,"promoted_layers":[name for name,row in gate.get("candidates",{}).items() if row["passed"]],
             "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True)
    _write_json(args.output_root/"route_a_layerwise_matched_negative_separability_v3_summary.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_layerwise_matched_negative_separability_v3")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]); parser.add_argument("--device",default=None)
    parser.add_argument("--resume",action="store_true"); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--smoke-epochs",type=int,default=2)
    parser.add_argument("--io-batch-size",type=int,default=256); parser.add_argument("--feature-batch-size",type=int,default=64); parser.add_argument("--topk",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=256); parser.add_argument("--learning-rate",type=float,default=1e-3)
    parser.add_argument("--epochs",type=int,default=80); parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--weight-decay",type=float,default=1e-4); parser.add_argument("--gradient-clip",type=float,default=5.0)
    parser.add_argument("--weak-threshold-db",type=float,default=10.0)
    return parser.parse_args()


if __name__=="__main__": run(parse_args())
