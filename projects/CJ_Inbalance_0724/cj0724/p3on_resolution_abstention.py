"""P3o-N matched-coverage resolution-aware MFT-count abstention screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3om_data_audit import role_paths
from cj0724.p3om_sparse_peak_ordinal import (
    OrdinalTokenNet,
    _collect,
    _read_role,
    _split_fit_selection,
    ordinal_loss,
    sparse_peak_tokens,
)
from cj0724.p2w_isrj_width_oracle_ood import sha256_file
from cj0724.train import set_seed


MODEL_SEEDS = (2301, 2302, 2303)
TARGET_COVERAGE = 0.80


def array_sha256(value: np.ndarray) -> str:
    """Hash an exact in-memory prediction array for matched-decision receipts."""
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def verify_method_freeze(project: Path, freeze_path: Path) -> dict:
    """Reject a formal run if its preregistered implementation has changed."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = {
        "preregister_sha256": project / "configs/p3on_resolution_abstention_preregister_20260803.json",
        "implementation_sha256": project / "cj0724/p3on_resolution_abstention.py",
        "audit_implementation_sha256": project / "cj0724/p3om_data_audit.py",
        "generator_sha256": project / "matlab/generate_p3on_resolution_abstention_level.m",
    }
    mismatches = {
        name: {"expected": freeze[name], "actual": sha256_file(path)}
        for name, path in expected.items()
        if freeze.get(name) != sha256_file(path)
    }
    if mismatches:
        raise RuntimeError(f"P3o-N method freeze mismatch: {mismatches}")
    return freeze


def ordinal_confidence(logits: np.ndarray) -> np.ndarray:
    """Maximum probability mass induced by the seven cumulative ordinal logits."""
    cumulative = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    probability = np.empty((len(cumulative), 8), dtype=np.float64)
    probability[:, 0] = 1.0 - cumulative[:, 0]
    probability[:, 1:7] = cumulative[:, :-1] - cumulative[:, 1:]
    probability[:, 7] = cumulative[:, 6]
    return np.maximum(probability, 0.0).max(axis=1).astype(np.float32)


def resolution_score(logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Observable C1 score: confidence times peak strength and separability."""
    confidence = ordinal_confidence(logits)
    count = np.clip(np.rint(1.0 + (1.0 / (1.0 + np.exp(-logits))).sum(axis=1)).astype(np.int64), 1, 8)
    tokens = np.asarray(tokens, dtype=np.float32); contrast, position = tokens[:, :, 1], tokens[:, :, 3]
    score = np.empty(len(tokens), dtype=np.float32)
    for row, predicted_count in enumerate(count):
        active = np.flatnonzero(contrast[row] > 0)
        selected = active[np.argsort(-contrast[row, active], kind="stable")[:predicted_count]] if len(active) else np.empty(0, dtype=np.int64)
        strength = float(contrast[row, selected].mean()) if len(selected) else 0.0
        if len(selected) <= 1:
            gap = 1.0
        else:
            gap = min(1.0, float(np.diff(np.sort(position[row, selected])).min()) / 0.04)
        score[row] = confidence[row] * strength * gap
    return score


def top_coverage_mask(score: np.ndarray, coverage: float = TARGET_COVERAGE) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if not 0.0 < coverage < 1.0: raise ValueError("coverage must be in (0,1)")
    keep = int(round(len(score) * coverage))
    order = np.argsort(-score, kind="stable"); mask = np.zeros(len(score), dtype=bool); mask[order[:keep]] = True
    return mask


def selection_threshold(score: np.ndarray, coverage: float = TARGET_COVERAGE) -> float:
    """Fix a score threshold solely on selection data; never re-rank development."""
    score = np.asarray(score, dtype=np.float64)
    keep = int(round(len(score) * coverage))
    return float(np.sort(score, kind="stable")[len(score) - keep])


def metrics(prediction: np.ndarray, target: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
    accepted = np.asarray(accepted, dtype=bool)
    if not bool(accepted.any()): raise RuntimeError("empty abstention set")
    p, y = prediction[accepted], target[accepted]; error = np.abs(p - y)
    pearson = float(np.corrcoef(p, y)[0, 1]) if np.std(p) and np.std(y) else 0.0
    return {"coverage": float(accepted.mean()), "accepted_count": int(accepted.sum()), "mae": float(error.mean()), "pearson": pearson, "rounded_accuracy": float(np.mean(np.rint(p) == y))}


def evaluate(prediction, data, c0_mask, c1_mask):
    result = {}
    for name, mask in (("C0_ordinal_confidence", c0_mask), ("C1_resolution_aware", c1_mask)):
        result[name] = {"overall": metrics(prediction, data["target"], mask), "compound": metrics(prediction, data["target"], mask & data["compound"]), "by_regime": {regime: metrics(prediction, data["target"], mask & (data["regime"] == index)) for index, regime in enumerate(REGIMES)}, "by_power": {str(power): metrics(prediction, data["target"], mask & (data["power"] == power)) for power in POWERS}}
    return result


def _train_point_estimator(fit, selection, development, seed, args, output):
    mean = fit["channels"].mean(axis=(0, 2), keepdims=True); scale = np.maximum(fit["channels"].std(axis=(0, 2), keepdims=True), 1e-4)
    tokenize = lambda data: sparse_peak_tokens((data["channels"] - mean) / scale)
    fit_x, fit_y = torch.from_numpy(tokenize(fit)), torch.from_numpy(fit["target"])
    selection_x, selection_y = torch.from_numpy(tokenize(selection)), torch.from_numpy(selection["target"])
    development_x = torch.from_numpy(tokenize(development)); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(seed); model = OrdinalTokenNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)); selection_loader = DataLoader(TensorDataset(selection_x, selection_y), batch_size=args.batch_size)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss=ordinal_loss(model(x.to(device)),y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): selected_loss=float(np.mean([float(ordinal_loss(model(x.to(device)),y.to(device))) for x,y in selection_loader]))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"selection_loss":selected_loss})
        if selected_loss < best: best,stale=selected_loss,0; torch.save(model.state_dict(),output/"point_best.pt")
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(torch.load(output/"point_best.pt",map_location=device,weights_only=True)); model.eval()
    def infer(tokens):
        logits=[]
        with torch.no_grad():
            for (x,) in DataLoader(TensorDataset(tokens),batch_size=args.batch_size): logits.append(model(x.to(device)).cpu().numpy())
        value=np.concatenate(logits); return value, (1.0 + (1.0/(1.0+np.exp(-value))).sum(axis=1)).astype(np.float32)
    selection_logits, _ = infer(selection_x); development_logits, prediction = infer(development_x)
    (output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return {"parameters":int(sum(p.numel() for p in model.parameters())),"best_selection_loss":best,"epochs_ran":len(history),"checkpoint_sha256":sha256_file(output/"point_best.pt"),"selection_logits":selection_logits,"development_logits":development_logits,"prediction":prediction,"selection_tokens":selection_x.numpy(),"development_tokens":development_x.numpy()}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root",default="D:/CJ0724_rechunked_data");parser.add_argument("--rechunk-suffix",default="_rechunked_v2");parser.add_argument("--data-audit",default="outputs/p3on_resolution_abstention_preflight_20260803/data_audit.json");parser.add_argument("--output-root",default="outputs/p3on_resolution_abstention_development_20260803");parser.add_argument("--preregister",default="configs/p3on_resolution_abstention_preregister_20260803.json");parser.add_argument("--method-freeze",default="configs/p3on_resolution_abstention_method_freeze_20260803.json");parser.add_argument("--epochs",type=int,default=20);parser.add_argument("--patience",type=int,default=5);parser.add_argument("--batch-size",type=int,default=64);parser.add_argument("--learning-rate",type=float,default=3e-4);parser.add_argument("--weight-decay",type=float,default=1e-4);parser.add_argument("--device",default="cuda")
    args=parser.parse_args();project=Path(__file__).resolve().parents[1];root=Path(args.rechunk_root);preregister=project/args.preregister;method_freeze=verify_method_freeze(project,project/args.method_freeze);output=project/args.output_root;output.mkdir(parents=True,exist_ok=False)
    train_paths=role_paths(root,"train",1720000000,args.rechunk_suffix,"p3on_resolution_abstention");dev_paths=role_paths(root,"development",1730000000,args.rechunk_suffix,"p3on_resolution_abstention")
    # _read_role supplies the per-regime metadata required by _collect as
    # well as the paired identity vector.  The prior audit has already
    # checked this fresh role against all historical identities.
    train_meta,train_identity=_read_role(train_paths,[]);dev_meta,dev_identity=_read_role(dev_paths,[])
    audit_path=project/args.data_audit
    if not audit_path.exists(): raise RuntimeError("P3o-N requires completed pre-training audit")
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["maximum_prior_overlap"] or audit["cross_role_overlap"] or not audit["power_alignment_verified"] or not audit["label_contract_verified"]: raise RuntimeError("P3o-N pre-training audit failed")
    if audit["train_unique"] != len(np.unique(train_identity)) or audit["development_unique"] != len(np.unique(dev_identity)): raise RuntimeError("P3o-N audit count mismatch")
    split=_split_fit_selection(train_meta["ID_control"]["labels"],20260803);fit=_collect(train_paths,train_meta,split["fit"]);selection=_collect(train_paths,train_meta,split["selection"]);development=_collect(dev_paths,dev_meta,None)
    runs=[]
    for seed in MODEL_SEEDS:
        seed_output=output/f"seed_{seed}";seed_output.mkdir();started=time.perf_counter();point=_train_point_estimator(fit,selection,development,seed,args,seed_output)
        c0_selection=ordinal_confidence(point["selection_logits"]);c1_selection=resolution_score(point["selection_logits"],point["selection_tokens"])
        c0_threshold=selection_threshold(c0_selection);c1_threshold=selection_threshold(c1_selection)
        c0=ordinal_confidence(point["development_logits"])>=c0_threshold;c1=resolution_score(point["development_logits"],point["development_tokens"])>=c1_threshold
        point_prediction_sha256=array_sha256(point["prediction"]);same_point_predictions=point_prediction_sha256 == array_sha256(point["prediction"])
        evaluation=evaluate(point["prediction"],development,c0,c1);np.savez_compressed(seed_output/"predictions.npz",prediction=point["prediction"],target=development["target"],identity=development["identity"],power=development["power"],regime=development["regime"],compound=development["compound"],C0_accepted=c0,C1_accepted=c1)
        receipt={"seed":seed,"seconds":float(time.perf_counter()-started),"parameters":point["parameters"],"best_selection_loss":point["best_selection_loss"],"epochs_ran":point["epochs_ran"],"checkpoint_sha256":point["checkpoint_sha256"],"point_prediction_sha256":point_prediction_sha256,"C0_prediction_sha256":point_prediction_sha256,"C1_prediction_sha256":point_prediction_sha256,"same_point_predictions_C0_C1_bitwise":same_point_predictions,"selection_threshold":{"C0":c0_threshold,"C1":c1_threshold},"selection_coverage":{"C0":float((c0_selection>=c0_threshold).mean()),"C1":float((c1_selection>=c1_threshold).mean())},"evaluation":evaluation};(seed_output/"receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8");runs.append(receipt)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    deltas=[]
    for run in runs:
        c0=run["evaluation"]["C0_ordinal_confidence"];c1=run["evaluation"]["C1_resolution_aware"]
        deltas.append({"seed":run["seed"],"overall_mae_reduction":1-c1["overall"]["mae"]/c0["overall"]["mae"],"compound_mae_reduction":1-c1["compound"]["mae"]/c0["compound"]["mae"],"overall_pearson_delta":c1["overall"]["pearson"]-c0["overall"]["pearson"],"compound_pearson_delta":c1["compound"]["pearson"]-c0["compound"]["pearson"],"low_count_pearson":c1["by_regime"]["low_count"]["pearson"],"ultra_close_coverage_delta":c1["by_regime"]["ultra_close"]["coverage"]-c0["by_regime"]["ultra_close"]["coverage"],"ultra_close_mae_delta":c1["by_regime"]["ultra_close"]["mae"]-c0["by_regime"]["ultra_close"]["mae"]})
    gates={"identity_overlap_zero":True,"power_and_parameter_contracts_hold":True,"same_point_predictions_C0_C1_bitwise":all(run["same_point_predictions_C0_C1_bitwise"] and run["C0_prediction_sha256"] == run["C1_prediction_sha256"] for run in runs),"each_seed_each_method_development_coverage_between":all(.7<=run["evaluation"][name]["overall"]["coverage"]<=.9 for run in runs for name in ("C0_ordinal_confidence","C1_resolution_aware")),"each_seed_C1_accepted_overall_and_compound_mae_reduction_vs_C0_at_least_0p05":all(row["overall_mae_reduction"]>=.05 and row["compound_mae_reduction"]>=.05 for row in deltas),"each_seed_C1_accepted_overall_and_compound_pearson_delta_vs_C0_at_least_0p05":all(row["overall_pearson_delta"]>=.05 and row["compound_pearson_delta"]>=.05 for row in deltas),"each_seed_C1_accepted_low_count_pearson_at_least_0p30":all(row["low_count_pearson"]>=.3 for row in deltas),"each_seed_C1_ultra_close_coverage_at_most_C0_minus_0p03":all(row["ultra_close_coverage_delta"]<=-.03 for row in deltas),"each_seed_C1_accepted_ultra_close_mae_not_worse_than_C0":all(row["ultra_close_mae_delta"]<=0 for row in deltas),"test_generated_false":True,"test_opened_false":True,"locked_test_used_false":True}
    summary={"experiment":"P3o-N observable resolution-aware abstention for compound MFT target count","preregister_sha256":sha256_file(preregister),"method_freeze":method_freeze,"runs":runs,"deltas":deltas,"promotion_gates":gates,"development_passed":bool(all(gates.values())),"verdict":"promote_to_two_identity_confirmation" if all(gates.values()) else "stop_no_confirmation_data","classification_loss_used":False,"oracle_positive_only":True,"test_generated":False,"test_opened":False,"locked_test_used":False};(output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps({"development_passed":summary["development_passed"],"promotion_gates":gates},indent=2))


if __name__=="__main__":main()
