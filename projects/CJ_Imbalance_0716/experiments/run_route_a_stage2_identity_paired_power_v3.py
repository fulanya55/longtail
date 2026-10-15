"""Stage2 identity-paired power consistency screen on development-v3."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.engine.thresholding import select_per_label_weak_recall_fpr_thresholds  # noqa: E402
from cjlt.models.evidence_readout import MultiTargetStage2LowRankEvidence  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import _extract_level  # noqa: E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES, TARGET_NAMES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = ("P0_SameView", "P1_NoPair", "P2_IdentityPair")
TRAIN_LEVELS = (10, 30, 50)
VALIDATION_LEVEL = 20
HELDOUT_LEVELS = (15, 20, 25, 35, 40, 45)
LOW_LEVELS = (15, 20, 25)
WEAK_LEVELS = (15, 20)
FIXED_INDICES = (0, 1, 2)


def _power_weak_mask(labels6, level_db, weak_threshold_db=10.0):
    labels = np.asarray(labels6, dtype=np.uint8)
    targets = labels[:, TARGET_INDICES].astype(bool)
    fixed_present = labels[:, FIXED_INDICES].sum(axis=1) > 0
    weak_level = float(level_db) - 30.0 <= -float(weak_threshold_db)
    return targets & fixed_present[:, None] & bool(weak_level)


def _model_scores(model, features, device, batch_size, return_maps=False):
    model.eval(); scores, maps = [], []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            values = torch.from_numpy(features[start:start+int(batch_size)]).to(
                device=device, dtype=torch.float32)
            if return_maps:
                logits, evidence, _ = model(values, return_map=True); maps.append(evidence.cpu().numpy())
            else:
                logits = model(values)
            scores.append(logits.cpu().numpy())
    logits = np.concatenate(scores).astype(np.float32)
    return (logits, np.concatenate(maps).astype(np.float32)) if return_maps else logits


def _pair_losses(student_logits, student_maps, teacher_logits, teacher_maps, positive_mask):
    mask = positive_mask.bool()
    if not bool(mask.any()):
        zero = student_logits.new_zeros(())
        return zero, zero
    score = F.smooth_l1_loss(student_logits[mask], teacher_logits.detach()[mask])
    selected_student = student_maps[mask]
    selected_teacher = teacher_maps.detach()[mask]
    map_loss = (1.0 - F.cosine_similarity(selected_student, selected_teacher, dim=-1)).mean()
    return score, map_loss


def _train_variant(variant, train_features, validation_features, fit_targets6,
                   validation_targets6, seed, device, args):
    train_seed = 981000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = MultiTargetStage2LowRankEvidence(
        channels=64, targets=3, rank=args.adapter_rank,
        hidden_channels=args.hidden_channels, topk=args.topk,
        adapter_limit=args.adapter_limit,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    targets = fit_targets6[:, TARGET_INDICES].astype(np.float32)
    positives = targets.sum(axis=0); negatives = len(targets)-positives
    pos_weight = torch.as_tensor(negatives/np.maximum(positives,1.0),
                                 dtype=torch.float32, device=device)
    rng = np.random.default_rng(train_seed)
    best_state, best_epoch, best_value, stale, history = None, 0, -float("inf"), 0, []
    epochs = args.smoke_epochs if args.smoke else args.epochs
    for epoch in range(1, int(epochs)+1):
        model.train(); order = rng.permutation(len(targets))
        totals = {"total":0.0,"classification":0.0,"score":0.0,"map":0.0}; count_total = 0
        for start in range(0, len(order), int(args.batch_size)):
            indices = order[start:start+int(args.batch_size)]
            level_values = {
                level: torch.from_numpy(train_features[level][indices]).to(device=device,dtype=torch.float32)
                for level in TRAIN_LEVELS
            }
            target = torch.from_numpy(targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = {level: model(values, return_map=True) for level,values in level_values.items()}
            classification = torch.stack([
                F.binary_cross_entropy_with_logits(outputs[level][0], target, pos_weight=pos_weight)
                for level in TRAIN_LEVELS
            ]).mean()
            score_loss = classification.new_zeros(()); map_loss = classification.new_zeros(())
            if variant == "P2_IdentityPair":
                pairs = []
                for level in (10,30):
                    pairs.append(_pair_losses(outputs[level][0],outputs[level][1],
                                              outputs[50][0],outputs[50][1],target>0.5))
                score_loss = torch.stack([row[0] for row in pairs]).mean()
                map_loss = torch.stack([row[1] for row in pairs]).mean()
            elif variant == "P0_SameView":
                teacher = level_values[50]
                scale = teacher.std(dim=-1,keepdim=True).clamp_min(1e-4)
                augmented = teacher + args.same_view_noise * scale * torch.randn_like(teacher)
                augmented_output = model(augmented, return_map=True)
                score_loss,map_loss = _pair_losses(augmented_output[0],augmented_output[1],
                                                   outputs[50][0],outputs[50][1],target>0.5)
            loss = classification + args.score_weight*score_loss + args.map_weight*map_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip)
            optimizer.step(); count=len(indices); count_total+=count
            for name,value in (("total",loss),("classification",classification),
                               ("score",score_loss),("map",map_loss)):
                totals[name]+=float(value.detach().cpu())*count
        validation_logits = _model_scores(model, validation_features, device, args.eval_batch_size)
        validation_targets = validation_targets6[:,TARGET_INDICES]
        validation_value = float(np.mean([
            average_precision_score(validation_targets[:,local],validation_logits[:,local])
            for local in range(3)
        ]))
        history.append({"epoch":epoch,"train_loss":totals["total"]/count_total,
                        "classification_loss":totals["classification"]/count_total,
                        "score_loss":totals["score"]/count_total,
                        "map_loss":totals["map"]/count_total,
                        "validation_20db_macro_aupr":validation_value})
        if validation_value > best_value+1e-8:
            best_value,best_epoch,stale=validation_value,epoch,0
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(args.patience): break
    model.load_state_dict(best_state)
    return model,history,{"best_epoch":best_epoch,"train_seed":train_seed,
                          "validation_20db_macro_aupr":best_value,
                          "pos_weight":pos_weight.detach().cpu().numpy().tolist()}


def _select_thresholds(logits, labels6, level_db, max_fpr, weak_threshold_db):
    probabilities = 1.0/(1.0+np.exp(-np.clip(logits,-40,40)))
    targets = labels6[:,TARGET_INDICES]
    weak = _power_weak_mask(labels6,level_db,weak_threshold_db)
    selected = select_per_label_weak_recall_fpr_thresholds(
        targets,probabilities,weak,max_fpr,"threshold_selection")
    return np.asarray(selected.thresholds,dtype=np.float64),probabilities


def _level_metrics(logits, labels6, level_db, thresholds, weak_threshold_db):
    target = labels6[:,TARGET_INDICES].astype(np.uint8)
    probability = 1.0/(1.0+np.exp(-np.clip(logits,-40,40)))
    prediction = probability>=thresholds[None,:]
    weak = _power_weak_mask(labels6,level_db,weak_threshold_db)
    per_label={}; weak_aurocs=[]
    for local,name in enumerate(TARGET_NAMES):
        positive=target[:,local]==1; negative=~positive; weak_positive=positive&weak[:,local]
        y=np.concatenate([np.ones(weak_positive.sum()),np.zeros(negative.sum())])
        score=np.concatenate([probability[weak_positive,local],probability[negative,local]])
        weak_auroc=float(roc_auc_score(y,score)) if weak_positive.any() else float("nan")
        if np.isfinite(weak_auroc): weak_aurocs.append(weak_auroc)
        per_label[name]={"aupr":float(average_precision_score(target[:,local],probability[:,local])),
                         "auroc":float(roc_auc_score(target[:,local],probability[:,local])),
                         "recall":float(prediction[positive,local].mean()),
                         "weak_recall":float(prediction[weak_positive,local].mean()) if weak_positive.any() else float("nan"),
                         "negative_fpr":float(prediction[negative,local].mean()),
                         "weak_vs_negative_auroc":weak_auroc,
                         "weak_positive_count":int(weak_positive.sum())}
    weak_recalls=[row["weak_recall"] for row in per_label.values() if np.isfinite(row["weak_recall"])]
    return {"level_db":int(level_db),"macro_aupr":float(np.mean([row["aupr"] for row in per_label.values()])),
            "macro_auroc":float(np.mean([row["auroc"] for row in per_label.values()])),
            "macro_recall":float(np.mean([row["recall"] for row in per_label.values()])),
            "macro_weak_recall":float(np.mean(weak_recalls)) if weak_recalls else float("nan"),
            "macro_weak_vs_negative_auroc":float(np.mean(weak_aurocs)) if weak_aurocs else float("nan"),
            "macro_negative_fpr":float(np.mean([row["negative_fpr"] for row in per_label.values()])),
            "per_label":per_label}


def _summarize_levels(level_metrics):
    by_level={int(row["level_db"]):row for row in level_metrics}
    return {"low_power_macro_aupr":float(np.mean([by_level[level]["macro_aupr"] for level in LOW_LEVELS])),
            "weak_level_macro_recall":float(np.mean([by_level[level]["macro_weak_recall"] for level in WEAK_LEVELS])),
            "weak_level_macro_auroc":float(np.mean([by_level[level]["macro_weak_vs_negative_auroc"] for level in WEAK_LEVELS])),
            "low_power_macro_fpr":float(np.mean([by_level[level]["macro_negative_fpr"] for level in LOW_LEVELS])),
            "all_heldout_macro_aupr":float(np.mean([row["macro_aupr"] for row in level_metrics])),
            "all_heldout_macro_auroc":float(np.mean([row["macro_auroc"] for row in level_metrics]))}


def _write_history(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _run_seed(args,config,labels6,power_files,seed,device,data_sha):
    seed_root=args.output_root/f"seed_{seed}"; summary_path=seed_root/"seed_summary.json"
    if args.resume and summary_path.is_file(): return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True,exist_ok=True)
    saved=np.load(args.c6_root/f"seed_{seed}"/"role_indices.npz")
    ids={name:saved[name].astype(np.int64) for name in ("fit","validation","early","threshold")}
    state=torch.load(args.c6_root/f"seed_{seed}"/"C6_Joint"/"best_model.pth",
                     map_location="cpu",weights_only=False)["model_state_dict"]
    backbone=_load_model_from_state(args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}",device,state)
    train_features={level:_extract_level(backbone,power_files[level],ids["fit"],
                         config["normalization"],device,args.extract_batch_size)["features"]
                    for level in TRAIN_LEVELS}
    validation_features=_extract_level(backbone,power_files[VALIDATION_LEVEL],ids["validation"],
                         config["normalization"],device,args.extract_batch_size)["features"]
    models={}; variant_info={}; histories={}
    for variant in VARIANTS:
        model,history,info=_train_variant(variant,train_features,validation_features,
            labels6[ids["fit"]],labels6[ids["validation"]],seed,device,args)
        models[variant]=model; histories[variant]=history; variant_info[variant]=info
        checkpoint=seed_root/variant/"best_stage2_identity_pair.pth"; checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"model_state_dict":model.state_dict(),"variant":variant,**info},checkpoint)
        info["checkpoint_sha256"]=_sha256(checkpoint); _write_history(seed_root/variant/"history.csv",history)
    del train_features,validation_features
    threshold_features=_extract_level(backbone,power_files[VALIDATION_LEVEL],ids["threshold"],
                         config["normalization"],device,args.extract_batch_size)["features"]
    thresholds={}
    for variant,model in models.items():
        threshold_logits=_model_scores(model,threshold_features,device,args.eval_batch_size)
        thresholds[variant],_= _select_thresholds(threshold_logits,labels6[ids["threshold"]],
            VALIDATION_LEVEL,args.max_negative_fpr,args.weak_threshold_db)
    del threshold_features
    level_results={variant:[] for variant in VARIANTS}; prediction_store={variant:{} for variant in VARIANTS}
    for level in HELDOUT_LEVELS:
        features=_extract_level(backbone,power_files[level],ids["early"],config["normalization"],
                                device,args.extract_batch_size)["features"]
        for variant,model in models.items():
            logits=_model_scores(model,features,device,args.eval_batch_size)
            level_results[variant].append(_level_metrics(logits,labels6[ids["early"]],level,
                                                         thresholds[variant],args.weak_threshold_db))
            prediction_store[variant][str(level)]=logits.astype(np.float16)
        del features
    del backbone
    variants=[]
    for variant in VARIANTS:
        summary=_summarize_levels(level_results[variant])
        variant_dir=seed_root/variant
        _write_json(variant_dir/"metrics.json",{"summary":summary,"levels":level_results[variant],
                                                "thresholds":thresholds[variant].tolist()})
        np.savez_compressed(variant_dir/"predictions.npz",sample_ids=ids["early"],
                            targets=labels6[ids["early"]][:,TARGET_INDICES],
                            **{f"logits_{level}db":values for level,values in prediction_store[variant].items()})
        variants.append({"variant":variant,"summary":summary,"levels":level_results[variant],
                         "thresholds":thresholds[variant].tolist()})
    payload={"seed":int(seed),"status":"completed","data_sha256":data_sha,
             "role_indices_sha256":_sha256(args.c6_root/f"seed_{seed}"/"role_indices.npz"),
             "power_file_sha256":{str(level):_sha256(power_files[level]) for level in sorted(power_files)},
             "variant_info":variant_info,"variants":variants,
             "parameter_count":int(sum(p.numel() for p in models[VARIANTS[0]].parameters())),
             "same_capacity":len({sum(p.numel() for p in model.parameters()) for model in models.values()})==1,
             "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(summary_path,payload); return payload


def _vmap(row): return {item["variant"]:item["summary"] for item in row["variants"]}


def _aggregate(rows):
    result={}
    for vi,variant in enumerate(VARIANTS):
        values=[_vmap(row)[variant] for row in rows]
        result[variant]={name:_mean_ci([value[name] for value in values],20270100+vi*100+i)
                         for i,name in enumerate(values[0])}
    return result


def _compare(rows,baseline):
    paired=[]
    for row in rows:
        candidate=_vmap(row)["P2_IdentityPair"]; base=_vmap(row)[baseline]
        paired.append({"seed":row["seed"],**{name:candidate[name]-base[name] for name in candidate}})
    aggregate={name:_mean_ci([row[name] for row in paired],20270600+i)
               for i,name in enumerate(_vmap(rows[0])["P2_IdentityPair"])}
    aggregate["positive_weak_auroc_seeds"]=int(sum(row["weak_level_macro_auroc"]>0 for row in paired))
    return {"baseline":baseline,"candidate":"P2_IdentityPair","per_seed":paired,"aggregate":aggregate}


def run(args):
    config=load_config(args.config); _validate_config(config)
    data_path=_data_path(config); data_sha=_sha256(data_path)
    if data_sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]: raise ValueError("test-v3 boundary is not sealed")
    audit=json.loads(args.power_audit.read_text(encoding="utf-8"))
    if not audit.get("passed"): raise ValueError("power-sweep identity audit must pass")
    power_files=discover_power_sweep_train_files(args.power_root); labels6=read_labels(data_path)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,labels6,power_files,seed,device,data_sha) for seed in seeds]
    aggregate=_aggregate(rows); comparisons={"vs_same_view":_compare(rows,"P0_SameView"),
                                             "vs_no_pair":_compare(rows,"P1_NoPair")}
    if args.smoke: gate={"checks":{},"passed":False}
    else:
        same=comparisons["vs_same_view"]["aggregate"]; no_pair=comparisons["vs_no_pair"]["aggregate"]
        candidate=aggregate["P2_IdentityPair"]
        checks={"weak_auroc_gain_0p015_vs_same_view":same["weak_level_macro_auroc"]["mean"]>=.015,
                "weak_recall_gain_0p030_vs_same_view":same["weak_level_macro_recall"]["mean"]>=.030,
                "low_power_aupr_gain_0p010_vs_same_view":same["low_power_macro_aupr"]["mean"]>=.010,
                "weak_auroc_gain_0p010_vs_no_pair":no_pair["weak_level_macro_auroc"]["mean"]>=.010,
                "low_power_fpr_cap":candidate["low_power_macro_fpr"]["mean"]<=.20,
                "four_of_five_positive_auroc_vs_same_view":same["positive_weak_auroc_seeds"]>=4,
                "identity_capacity_test_safe":all(row["same_capacity"] and not row["test_v3_used"] for row in rows)}
        gate={"checks":checks,"passed":bool(all(checks.values()))}
    payload={"status":"completed","experiment":"RouteA_Stage2IdentityPairedPower_v3",
             "smoke":bool(args.smoke),"seeds":list(seeds),"data_sha256":data_sha,
             "protocol":{"train_levels":list(TRAIN_LEVELS),"validation_level":VALIDATION_LEVEL,
                         "heldout_levels":list(HELDOUT_LEVELS),"variants":list(VARIANTS),
                         "adapter_rank":args.adapter_rank,"epochs":args.epochs},
             "aggregate":aggregate,"comparisons":comparisons,"promotion_gate":gate,
             "promoted":bool(gate["passed"]),"locked_test_used":False,
             "test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True)
    _write_json(args.output_root/"route_a_stage2_identity_paired_power_v3_summary.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--power-root",type=Path,default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\power_sweep_9level_v3"))
    parser.add_argument("--power-audit",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_v3_power_sweep_audit"/"route_a_v3_power_sweep_identity_audit.json")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_stage2_identity_paired_power_v3")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device",default=None); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true"); parser.add_argument("--smoke-epochs",type=int,default=2)
    parser.add_argument("--adapter-rank",type=int,default=8); parser.add_argument("--hidden-channels",type=int,default=32)
    parser.add_argument("--topk",type=int,default=16); parser.add_argument("--adapter-limit",type=float,default=.20)
    parser.add_argument("--epochs",type=int,default=30); parser.add_argument("--patience",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=1e-3); parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--batch-size",type=int,default=64); parser.add_argument("--extract-batch-size",type=int,default=128)
    parser.add_argument("--eval-batch-size",type=int,default=256); parser.add_argument("--gradient-clip",type=float,default=5.0)
    parser.add_argument("--score-weight",type=float,default=.25); parser.add_argument("--map-weight",type=float,default=.10)
    parser.add_argument("--same-view-noise",type=float,default=.02); parser.add_argument("--max-negative-fpr",type=float,default=.20)
    parser.add_argument("--weak-threshold-db",type=float,default=10.0)
    return parser.parse_args()


if __name__=="__main__": run(parse_args())

