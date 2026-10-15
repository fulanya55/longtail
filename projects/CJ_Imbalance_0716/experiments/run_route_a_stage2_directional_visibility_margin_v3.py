"""Directional stage2 visibility margin with composition-matched hard negatives."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import discover_power_sweep_train_files  # noqa: E402
from cjlt.data.v73_dataset import read_labels  # noqa: E402
from cjlt.models.evidence_readout import MultiTargetStage2LowRankEvidence  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import _extract_level  # noqa: E402
from experiments.run_route_a_stage2_identity_paired_power_v3 import (  # noqa: E402
    HELDOUT_LEVELS, LOW_LEVELS, TRAIN_LEVELS, VALIDATION_LEVEL, VARIANTS as PRIOR_VARIANTS,
    _level_metrics, _model_scores, _select_thresholds, _summarize_levels, _write_history,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


CANDIDATE = "P3_DirectionalMargin"
ALL_VARIANTS = PRIOR_VARIANTS + (CANDIDATE,)


def _matched_hard_negative_pairs(labels6, seed):
    labels = np.asarray(labels6, dtype=np.uint8)
    rng = np.random.default_rng(991000 + int(seed))
    positive_all, negative_all, local_all, counts = [], [], [], {}
    for local, target_index in enumerate(TARGET_INDICES):
        other = [index for index in range(labels.shape[1]) if index != target_index]
        positives, negatives = {}, {}
        for row, label in enumerate(labels):
            key = tuple(int(label[index]) for index in other)
            if label[target_index]: positives.setdefault(key, []).append(row)
            else: negatives.setdefault(key, []).append(row)
        shared = sorted(set(positives).intersection(negatives))
        label_count = 0
        for key in shared:
            pos = np.asarray(positives[key], dtype=np.int64); neg = np.asarray(negatives[key], dtype=np.int64)
            rng.shuffle(pos); rng.shuffle(neg); count = min(len(pos), len(neg))
            if not count: continue
            positive_all.extend(pos[:count].tolist()); negative_all.extend(neg[:count].tolist())
            local_all.extend([local]*count); label_count += count
        counts[str(target_index)] = {"shared_combinations":len(shared),"pair_count":int(label_count)}
    if not positive_all: raise ValueError("no composition-matched hard-negative pairs")
    order = rng.permutation(len(positive_all))
    return {"positive":np.asarray(positive_all,dtype=np.int64)[order],
            "negative":np.asarray(negative_all,dtype=np.int64)[order],
            "local":np.asarray(local_all,dtype=np.int64)[order],"counts":counts}


def _directional_losses(model, features, pairs, pair_indices, device, args):
    positive = pairs["positive"][pair_indices]; negative = pairs["negative"][pair_indices]
    local = torch.from_numpy(pairs["local"][pair_indices]).to(device=device,dtype=torch.long)
    row = torch.arange(len(pair_indices),device=device)
    def values(level, indices):
        return torch.from_numpy(features[level][indices]).to(device=device,dtype=torch.float32)
    pos10=model(values(10,positive))[row,local]; pos30=model(values(30,positive))[row,local]
    with torch.no_grad(): pos50=model(values(50,positive))[row,local]
    neg10=model(values(10,negative))[row,local]; neg30=model(values(30,negative))[row,local]
    lower=0.5*(F.relu(pos50-pos10-args.teacher_margin).mean()
               +F.relu(pos50-pos30-args.teacher_margin).mean())
    ranking=0.5*(F.softplus(args.ranking_margin-pos10+neg10).mean()
                 +F.softplus(args.ranking_margin-pos30+neg30).mean())
    return lower,ranking


def _train_directional(train_features,validation_features,fit_targets6,validation_targets6,
                       pairs,seed,device,args):
    train_seed=981000+int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model=MultiTargetStage2LowRankEvidence(channels=64,targets=3,rank=args.adapter_rank,
        hidden_channels=args.hidden_channels,topk=args.topk,adapter_limit=args.adapter_limit).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    targets=fit_targets6[:,TARGET_INDICES].astype(np.float32)
    positives=targets.sum(axis=0); pos_weight=torch.as_tensor((len(targets)-positives)/np.maximum(positives,1.0),
                                                              dtype=torch.float32,device=device)
    rng=np.random.default_rng(train_seed)
    best_state,best_epoch,best_value,stale,history=None,0,-float("inf"),0,[]
    epochs=args.smoke_epochs if args.smoke else args.epochs
    for epoch in range(1,int(epochs)+1):
        model.train(); order=rng.permutation(len(targets)); totals={"total":0.,"classification":0.,"lower":0.,"ranking":0.}; n=0
        for start in range(0,len(order),int(args.batch_size)):
            indices=order[start:start+int(args.batch_size)]
            target=torch.from_numpy(targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            classification=[]
            for level in TRAIN_LEVELS:
                values=torch.from_numpy(train_features[level][indices]).to(device=device,dtype=torch.float32)
                classification.append(F.binary_cross_entropy_with_logits(model(values),target,pos_weight=pos_weight))
            classification=torch.stack(classification).mean()
            pair_indices=rng.choice(len(pairs["positive"]),size=len(indices),replace=len(pairs["positive"])<len(indices))
            lower,ranking=_directional_losses(model,train_features,pairs,pair_indices,device,args)
            loss=classification+args.lower_weight*lower+args.ranking_weight*ranking
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip); optimizer.step()
            count=len(indices); n+=count
            for name,value in (("total",loss),("classification",classification),("lower",lower),("ranking",ranking)):
                totals[name]+=float(value.detach().cpu())*count
        validation_logits=_model_scores(model,validation_features,device,args.eval_batch_size)
        validation_targets=validation_targets6[:,TARGET_INDICES]
        value=float(np.mean([average_precision_score(validation_targets[:,i],validation_logits[:,i]) for i in range(3)]))
        history.append({"epoch":epoch,"train_loss":totals["total"]/n,
                        "classification_loss":totals["classification"]/n,
                        "lower_loss":totals["lower"]/n,"ranking_loss":totals["ranking"]/n,
                        "validation_20db_macro_aupr":value})
        if value>best_value+1e-8:
            best_value,best_epoch,stale=value,epoch,0
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(args.patience): break
    model.load_state_dict(best_state)
    return model,history,{"best_epoch":best_epoch,"train_seed":train_seed,
                          "validation_20db_macro_aupr":best_value,
                          "pair_counts":pairs["counts"],"pair_count":int(len(pairs["positive"]))}


def _read_prior(seed,variant,prior_root):
    data=json.loads((prior_root/f"seed_{seed}"/variant/"metrics.json").read_text(encoding="utf-8"))
    return {"variant":variant,"summary":data["summary"],"levels":data["levels"],"thresholds":data["thresholds"]}


def _run_seed(args,config,labels6,power_files,seed,device,data_sha):
    seed_root=args.output_root/f"seed_{seed}"; summary_path=seed_root/"seed_summary.json"
    if args.resume and summary_path.is_file(): return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True,exist_ok=True)
    saved=np.load(args.c6_root/f"seed_{seed}"/"role_indices.npz")
    ids={name:saved[name].astype(np.int64) for name in ("fit","validation","early","threshold")}
    state=torch.load(args.c6_root/f"seed_{seed}"/"C6_Joint"/"best_model.pth",map_location="cpu",weights_only=False)["model_state_dict"]
    backbone=_load_model_from_state(args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}",device,state)
    train_features={level:_extract_level(backbone,power_files[level],ids["fit"],config["normalization"],
                                         device,args.extract_batch_size)["features"] for level in TRAIN_LEVELS}
    validation_features=_extract_level(backbone,power_files[VALIDATION_LEVEL],ids["validation"],
                                       config["normalization"],device,args.extract_batch_size)["features"]
    pairs=_matched_hard_negative_pairs(labels6[ids["fit"]],seed)
    model,history,info=_train_directional(train_features,validation_features,labels6[ids["fit"]],
                                          labels6[ids["validation"]],pairs,seed,device,args)
    del train_features,validation_features
    candidate_dir=seed_root/CANDIDATE; candidate_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=candidate_dir/"best_stage2_directional_margin.pth"
    torch.save({"model_state_dict":model.state_dict(),"variant":CANDIDATE,**info},checkpoint)
    info["checkpoint_sha256"]=_sha256(checkpoint); _write_history(candidate_dir/"history.csv",history)
    threshold_features=_extract_level(backbone,power_files[VALIDATION_LEVEL],ids["threshold"],
                                      config["normalization"],device,args.extract_batch_size)["features"]
    threshold_logits=_model_scores(model,threshold_features,device,args.eval_batch_size)
    thresholds,_=_select_thresholds(threshold_logits,labels6[ids["threshold"]],VALIDATION_LEVEL,
                                    args.max_negative_fpr,args.weak_threshold_db)
    del threshold_features
    level_results=[]; predictions={}
    for level in HELDOUT_LEVELS:
        features=_extract_level(backbone,power_files[level],ids["early"],config["normalization"],device,args.extract_batch_size)["features"]
        logits=_model_scores(model,features,device,args.eval_batch_size); del features
        level_results.append(_level_metrics(logits,labels6[ids["early"]],level,thresholds,args.weak_threshold_db))
        predictions[str(level)]=logits.astype(np.float16)
    del backbone
    summary=_summarize_levels(level_results)
    _write_json(candidate_dir/"metrics.json",{"summary":summary,"levels":level_results,"thresholds":thresholds.tolist()})
    np.savez_compressed(candidate_dir/"predictions.npz",sample_ids=ids["early"],
                        targets=labels6[ids["early"]][:,TARGET_INDICES],
                        **{f"logits_{level}db":value for level,value in predictions.items()})
    variants=[_read_prior(seed,variant,args.prior_root) for variant in PRIOR_VARIANTS]
    variants.append({"variant":CANDIDATE,"summary":summary,"levels":level_results,"thresholds":thresholds.tolist()})
    payload={"seed":int(seed),"status":"completed","data_sha256":data_sha,
             "role_indices_sha256":_sha256(args.c6_root/f"seed_{seed}"/"role_indices.npz"),
             "candidate_info":info,"variants":variants,
             "parameter_count":int(sum(p.numel() for p in model.parameters())),"same_capacity":True,
             "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(summary_path,payload); return payload


def _vmap(row): return {item["variant"]:item["summary"] for item in row["variants"]}


def _aggregate(rows):
    result={}
    for vi,variant in enumerate(ALL_VARIANTS):
        values=[_vmap(row)[variant] for row in rows]
        result[variant]={name:_mean_ci([value[name] for value in values],20271100+vi*100+i)
                         for i,name in enumerate(values[0])}
    return result


def _compare(rows,baseline):
    paired=[]
    for row in rows:
        candidate=_vmap(row)[CANDIDATE]; base=_vmap(row)[baseline]
        paired.append({"seed":row["seed"],**{name:candidate[name]-base[name] for name in candidate}})
    aggregate={name:_mean_ci([row[name] for row in paired],20271600+i)
               for i,name in enumerate(_vmap(rows[0])[CANDIDATE])}
    aggregate["positive_weak_auroc_seeds"]=int(sum(row["weak_level_macro_auroc"]>0 for row in paired))
    return {"baseline":baseline,"candidate":CANDIDATE,"per_seed":paired,"aggregate":aggregate}


def run(args):
    config=load_config(args.config); _validate_config(config); data_path=_data_path(config); data_sha=_sha256(data_path)
    if data_sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]: raise ValueError("test-v3 boundary is not sealed")
    audit=json.loads(args.power_audit.read_text(encoding="utf-8"))
    if not audit.get("passed"): raise ValueError("power-sweep identity audit must pass")
    labels6=read_labels(data_path); power_files=discover_power_sweep_train_files(args.power_root)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,labels6,power_files,seed,device,data_sha) for seed in seeds]
    aggregate=_aggregate(rows); comparisons={"vs_same_view":_compare(rows,"P0_SameView"),
                                             "vs_no_pair":_compare(rows,"P1_NoPair"),
                                             "vs_symmetric":_compare(rows,"P2_IdentityPair")}
    if args.smoke: gate={"checks":{},"passed":False}
    else:
        same=comparisons["vs_same_view"]["aggregate"]; no_pair=comparisons["vs_no_pair"]["aggregate"]
        candidate=aggregate[CANDIDATE]
        checks={"weak_auroc_gain_0p015_vs_same_view":same["weak_level_macro_auroc"]["mean"]>=.015,
                "weak_recall_gain_0p030_vs_same_view":same["weak_level_macro_recall"]["mean"]>=.030,
                "low_power_aupr_gain_0p010_vs_same_view":same["low_power_macro_aupr"]["mean"]>=.010,
                "weak_auroc_gain_0p010_vs_no_pair":no_pair["weak_level_macro_auroc"]["mean"]>=.010,
                "low_power_fpr_cap":candidate["low_power_macro_fpr"]["mean"]<=.20,
                "four_of_five_positive_auroc_vs_same_view":same["positive_weak_auroc_seeds"]>=4,
                "identity_capacity_test_safe":all(row["same_capacity"] and not row["test_v3_used"] for row in rows)}
        gate={"checks":checks,"passed":bool(all(checks.values()))}
    payload={"status":"completed","experiment":"RouteA_Stage2DirectionalVisibilityMargin_v3",
             "smoke":bool(args.smoke),"seeds":list(seeds),"data_sha256":data_sha,
             "protocol":{"teacher_margin":args.teacher_margin,"ranking_margin":args.ranking_margin,
                         "lower_weight":args.lower_weight,"ranking_weight":args.ranking_weight},
             "aggregate":aggregate,"comparisons":comparisons,"promotion_gate":gate,
             "promoted":bool(gate["passed"]),"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True)
    _write_json(args.output_root/"route_a_stage2_directional_visibility_margin_v3_summary.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--prior-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_stage2_identity_paired_power_v3")
    parser.add_argument("--power-root",type=Path,default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3\power_sweep_9level_v3"))
    parser.add_argument("--power-audit",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_v3_power_sweep_audit"/"route_a_v3_power_sweep_identity_audit.json")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_stage2_directional_visibility_margin_v3")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device",default=None); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true"); parser.add_argument("--smoke-epochs",type=int,default=2)
    parser.add_argument("--adapter-rank",type=int,default=8); parser.add_argument("--hidden-channels",type=int,default=32)
    parser.add_argument("--topk",type=int,default=16); parser.add_argument("--adapter-limit",type=float,default=.20)
    parser.add_argument("--epochs",type=int,default=30); parser.add_argument("--patience",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=1e-3); parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--batch-size",type=int,default=64); parser.add_argument("--extract-batch-size",type=int,default=128)
    parser.add_argument("--eval-batch-size",type=int,default=256); parser.add_argument("--gradient-clip",type=float,default=5.0)
    parser.add_argument("--teacher-margin",type=float,default=.5); parser.add_argument("--ranking-margin",type=float,default=.5)
    parser.add_argument("--lower-weight",type=float,default=.25); parser.add_argument("--ranking-weight",type=float,default=.25)
    parser.add_argument("--max-negative-fpr",type=float,default=.20); parser.add_argument("--weak-threshold-db",type=float,default=10.0)
    return parser.parse_args()


if __name__=="__main__": run(parse_args())

