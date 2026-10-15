"""Read-only stratified diagnosis of the already closed P1/P2 ISRJ H0 runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_route_a_p1_observability_adequate_h0 as p1
from experiments import run_route_a_p2_full_target_h0 as p2

ISRJ = 5


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))}


def _analyze(path: Path, name: str) -> dict:
    scorer = p1._isrj_score if name == "P1" else p2._isrj_joint_periodicity_score
    controller = (lambda x: p1._isrj_score(x[p1.SAMPLE_PERMUTATION])) if name == "P1" else (lambda x: p2._isrj_joint_periodicity_score(p2._block_shuffle(x)))
    with h5py.File(path, "r") as handle:
        target = np.asarray(handle["target_index"], dtype=np.int64)
        rows = np.flatnonzero(target == ISRJ)
        real = np.asarray(handle["iq_real"][rows], dtype=np.float64)
        imag = np.asarray(handle["iq_imag"][rows], dtype=np.float64)
        labels = np.asarray(handle["labels"][rows], dtype=np.uint8)
        pair = np.asarray(handle["pair_seed"][rows], dtype=np.uint64)
        arm = np.asarray(handle["visibility_arm"][rows], dtype=np.int64)
    signal = real + 1j * imag
    score = np.asarray([scorer(value) for value in signal], dtype=np.float64)
    control = np.asarray([controller(value) for value in signal], dtype=np.float64)
    summary, rows_by_order = {}, {}
    by_pair = []
    for identity in np.unique(pair):
        index = np.flatnonzero(pair == identity)
        ordered = index[np.argsort(arm[index])]
        if ordered.size != 3 or not np.array_equal(arm[ordered], [0, 1, 2]):
            raise ValueError(f"invalid {name} triplet")
        absent, weak, strong = ordered
        order = int(labels[absent].sum())
        by_pair.append((order, float(score[weak] - score[absent]), float(score[strong] - score[absent]), float((score[weak] - control[weak]) - (score[absent] - control[absent]))))
    for local_arm, label in enumerate(("absent", "weak", "strong")):
        index = arm == local_arm
        summary[label] = {"score": _quantiles(score[index]), "control": _quantiles(control[index]), "score_minus_control": _quantiles(score[index] - control[index])}
    values = np.asarray(by_pair, dtype=np.float64)
    for order in sorted(set(values[:, 0].astype(int))):
        selection = values[:, 0] == order
        rows_by_order[str(order)] = {"identity_count": int(selection.sum()), "weak_minus_absent": _quantiles(values[selection, 1]), "strong_minus_absent": _quantiles(values[selection, 2]), "weak_true_minus_control_pair_delta": _quantiles(values[selection, 3])}
    return {"source": str(path), "isrj_identity_count": int(len(np.unique(pair))), "arms": summary, "by_nuisance_label_count": rows_by_order}


def run(args: argparse.Namespace) -> dict:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 must remain sealed")
    result = {"status":"completed_read_only_failure_diagnosis","experiment":"RouteA_P1P2_ISRJH0FailureDiagnosis","scope":"closed_P1_P2_sources_only_no_selection_or_new_candidate","p1":_analyze(args.p1_path,"P1"),"p2":_analyze(args.p2_path,"P2"),"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-path",type=Path,default=Path(r"C:\LJQ_code\干扰数据\p1_observability_adequate_20260803_r1\closedset_6class.mat")); parser.add_argument("--p2-path",type=Path,default=Path(r"C:\LJQ_code\干扰数据\p2_full_target_observability_20260803_r1\closedset_6class.mat")); parser.add_argument("--output",type=Path,default=ROOT/"outputs/tapc/route_a_p1_p2_isrj_h0_failure_diagnosis/summary.json"); parser.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    result=run(parser.parse_args()); print(json.dumps({"status":result["status"],"test_v3_used":result["test_v3_used"]},ensure_ascii=False))
