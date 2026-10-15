"""Fixed five-seed C0--C4 H2 development summary and promotion gates."""
from __future__ import annotations

import argparse, json
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _load(root: Path) -> dict:
    value=json.loads((root/"run_receipt.json").read_text(encoding="utf-8"))
    if value["status"] != "completed_development_unaggregated" or value["smoke"]:
        raise ValueError(f"not a formal development receipt: {root}")
    if any(value[key] for key in ("locked_test_used","test_v2_used","test_v3_used")):
        raise ValueError(f"test boundary violated: {root}")
    if len(value["results"]) != 1: raise ValueError("one mode/seed per root required")
    return value


def _derived_prediction_metrics(root: Path, result: dict) -> dict[str, float]:
    """Recompute weak-VFT ranking from immutable development predictions.

    The runner stores all raw probabilities and physical labels before any
    temperature or threshold operation.  Recomputing the weak-versus-negative
    slice here makes the promotion audit cover the AUPR criterion fixed in the
    H2 protocol, rather than relying on an all-positive VFT AUPR surrogate.
    """
    prediction_path = root / f"{result['mode']}_seed{result['seed']}" / "development_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as data:
        probability = np.asarray(data["probabilities"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=bool)
        power = np.asarray(data["component_jnr_db"], dtype=np.float64)
    weak_vft = labels[:, 4] & np.isfinite(power[:, 4]) & (power[:, 4] <= 8.001)
    negative_vft = ~labels[:, 4]
    selected = weak_vft | negative_vft
    if not weak_vft.any() or not negative_vft.any():
        raise ValueError(f"weak VFT/negative support missing: {prediction_path}")
    weak_target = labels[selected, 4]
    vft_weak_auroc = float(roc_auc_score(weak_target, probability[selected, 4]))
    recorded = float(result["metrics"]["vft_weak_auroc"])
    if not np.isclose(vft_weak_auroc, recorded, atol=1e-12, rtol=0.0):
        raise ValueError(f"stored weak-VFT AUROC mismatch: {prediction_path}")
    return {
        "vft_weak_auroc": vft_weak_auroc,
        "vft_weak_aupr": float(average_precision_score(weak_target, probability[selected, 4])),
        "non_vft_macro_aupr": float(np.mean([result["metrics"]["per_label"][index]["aupr"] for index in (0, 1, 2, 3, 5)])),
        "vft_weak_recall": float(result["threshold_metrics"]["per_label"][4]["weak_recall"]),
        "vft_fpr": float(result["threshold_metrics"]["per_label"][4]["fpr"]),
        "sawfm_aupr": float(result["metrics"]["per_label"][1]["aupr"]),
        "sawfm_weak_recall": float(result["threshold_metrics"]["per_label"][1]["weak_recall"]),
        "macro_aupr": float(result["metrics"]["macro_aupr"]),
    }


def _bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, object]:
    rng=np.random.default_rng(seed); samples=np.asarray([values[rng.integers(0,values.size,values.size)].mean() for _ in range(draws)])
    nonzero=values[values != 0]; positive=int((nonzero > 0).sum()); negative=int((nonzero < 0).sum())
    if nonzero.size:
        tail=sum(comb(nonzero.size,k) for k in range(0,min(positive,negative)+1)) / (2 ** nonzero.size)
        sign_p=float(min(1.0,2.0*tail))
    else:
        sign_p=1.0
    std=float(values.std(ddof=1)) if values.size > 1 else float("nan")
    dz=float(values.mean()/std) if np.isfinite(std) and std > 0 else (float("inf") if values.mean() > 0 else float("-inf") if values.mean() < 0 else 0.0)
    return {"mean":float(values.mean()),"per_seed":values.tolist(),"bootstrap_95_ci":[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],"positive_seeds":int((values>0).sum()),"negative_seeds":int((values<0).sum()),"paired_effect_size_dz":dz,"exact_two_sided_sign_test_p":sign_p}


def _metric(entry: dict, key: str) -> float:
    return float(entry["derived"][key])


def run(args: argparse.Namespace) -> dict[str, object]:
    table={}
    for mode in ("C0-mp","C1-mp","C2-mp","C3-mp","C4-mp"):
        table[mode]={}
        for seed in args.seeds:
            root=Path(str(args.root_template).format(mode=mode.lower().replace("-mp",""),seed=seed)); receipt=_load(root); result=receipt["results"][0]
            if result["mode"] != mode or int(result["seed"]) != seed: raise ValueError(f"receipt identity mismatch {root}")
            table[mode][seed]={"root":str(root),"result":result,"derived":_derived_prediction_metrics(root,result)}
    comparison={}
    for name, left, right, metric in (
        ("C2_minus_C0_vft_weak_auroc","C2-mp","C0-mp","vft_weak_auroc"),
        ("C2_minus_C0_vft_weak_aupr","C2-mp","C0-mp","vft_weak_aupr"),
        ("C3_minus_C0_vft_weak_auroc","C3-mp","C0-mp","vft_weak_auroc"),
        ("C3_minus_C4_vft_weak_auroc","C3-mp","C4-mp","vft_weak_auroc"),
        ("C3_minus_C0_vft_weak_aupr","C3-mp","C0-mp","vft_weak_aupr"),
        ("C3_minus_C4_vft_weak_aupr","C3-mp","C4-mp","vft_weak_aupr"),
        ("C3_minus_C0_vft_weak_recall","C3-mp","C0-mp","vft_weak_recall"),
        ("C3_minus_C0_vft_fpr","C3-mp","C0-mp","vft_fpr"),
        ("C3_minus_C0_macro_aupr","C3-mp","C0-mp","macro_aupr"),
        ("C3_minus_C0_non_vft_macro_aupr","C3-mp","C0-mp","non_vft_macro_aupr"),
        ("C3_minus_C1_sawfm_aupr","C3-mp","C1-mp","sawfm_aupr"),
        ("C3_minus_C1_sawfm_weak_recall","C3-mp","C1-mp","sawfm_weak_recall"),
    ):
        values=np.asarray([_metric(table[left][seed],metric)-_metric(table[right][seed],metric) for seed in args.seeds]); comparison[name]=_bootstrap(values,draws=args.bootstrap_draws,seed=args.bootstrap_seed)
    target=(comparison["C3_minus_C0_vft_weak_auroc"],comparison["C3_minus_C4_vft_weak_auroc"])
    target_aupr=(comparison["C3_minus_C0_vft_weak_aupr"],comparison["C3_minus_C4_vft_weak_aupr"])
    macro=comparison["C3_minus_C0_macro_aupr"]; non_vft=comparison["C3_minus_C0_non_vft_macro_aupr"]
    saw=comparison["C3_minus_C1_sawfm_aupr"]; saw_recall=comparison["C3_minus_C1_sawfm_weak_recall"]
    recall=comparison["C3_minus_C0_vft_weak_recall"]; fpr=comparison["C3_minus_C0_vft_fpr"]
    capacity_equal=all(table["C3-mp"][seed]["result"]["parameter_count"] == table["C4-mp"][seed]["result"]["parameter_count"] for seed in args.seeds)
    gates={
        "c3_beats_c0_vft_weak_auroc_mean_ge_0p030":target[0]["mean"]>=.030,
        "c3_beats_c0_vft_weak_auroc_5of5_positive":target[0]["positive_seeds"]==5,
        "c3_beats_c4_vft_weak_auroc_mean_ge_0p030":target[1]["mean"]>=.030,
        "c3_beats_c4_vft_weak_auroc_5of5_positive":target[1]["positive_seeds"]==5,
        "c3_beats_c0_vft_weak_aupr_mean_ge_0p025":target_aupr[0]["mean"]>=.025,
        "c3_beats_c0_vft_weak_aupr_ci_lower_gt_0":target_aupr[0]["bootstrap_95_ci"][0]>0,
        "c3_beats_c4_vft_weak_aupr_mean_ge_0p025":target_aupr[1]["mean"]>=.025,
        "c3_beats_c4_vft_weak_aupr_ci_lower_gt_0":target_aupr[1]["bootstrap_95_ci"][0]>0,
        "c3_vft_weak_recall_gain_ge_0p030":recall["mean"]>=.030,
        "vft_fpr_increase_le_0p015":fpr["mean"]<=.015,
        "c3_macro_aupr_gain_ge_0p005":macro["mean"]>=.005,
        "non_vft_macro_aupr_drop_le_0p010":non_vft["mean"]>=-.010,
        "sawfm_aupr_noninferior_to_c1":saw["mean"]>=-.010,
        "sawfm_weak_recall_noninferior_to_c1":saw_recall["mean"]>=-.010,
        "c3_c4_equal_parameter_count":capacity_equal,
        "all_receipts_test_free":True,
    }
    result={"status":"completed_development_summary","experiment":"RouteA_H2_MultiPulseSixLabel","seeds":args.seeds,"per_mode_per_seed":table,"comparison":comparison,"diagnostic":{"c2_minus_c0_true_vft_expert":{key:comparison[key] for key in ("C2_minus_C0_vft_weak_auroc","C2_minus_C0_vft_weak_aupr")},"threshold_metric_evidence_boundary":"Thresholds are fitted on the odd development-pair half and reported across development; they are descriptive development evidence, not an independent test claim."},"gate":{"passed":bool(all(gates.values())),"checks":gates},"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root-template",default="outputs/tapc/route_a_h2_multipulse_{mode}_seed{seed}");p.add_argument("--seeds",nargs="+",type=int,default=[20260803,20260811,20260819,20260827,20260904]);p.add_argument("--bootstrap-draws",type=int,default=10000);p.add_argument("--bootstrap-seed",type=int,default=20260803);p.add_argument("--output",type=Path,default=Path("outputs/tapc/route_a_h2_multipulse_development_summary.json"));a=p.parse_args();x=run(a);print(json.dumps({"status":x["status"],"gate":x["gate"]},ensure_ascii=False))
