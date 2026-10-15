"""Stage-1 sampler comparison for Route-A V3 (development only).

Uniform-Focal is read from the completed V3 baseline; only MaxPositive and
MeanPositive are trained.  All roles, seeds, B0 and optimization settings are
identical.  No test/confirmation file is opened.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path("E:/LJQ_code/CJ_Imbalance_0716")
OUT = ROOT / "outputs" / "route_a_five_class_frequency_lt_v3"
STAGE_OUT = OUT / "stage1_sampler"
AUDIT = Path("E:/LJQ_code/干扰数据/route_a_five_class_frequency_lt_v3")
DEV_TRAIN = Path("E:/LJQ_code/干扰数据/复合干扰数据0731_v3/closedset_6class_train.mat")


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")


def main() -> None:
    v3spec = importlib.util.spec_from_file_location("route_a_v3", ROOT / "experiments" / "run_route_a_five_class_frequency_lt_v3.py")
    v3 = importlib.util.module_from_spec(v3spec); assert v3spec.loader is not None; v3spec.loader.exec_module(v3)
    v2spec = importlib.util.spec_from_file_location("route_a_v2", ROOT / "experiments" / "run_route_a_five_class_frequency_lt_v2.py")
    v2 = importlib.util.module_from_spec(v2spec); assert v2spec.loader is not None; v2spec.loader.exec_module(v2)
    baseline = json.loads((OUT / "stage1_summary.json").read_text(encoding="utf-8"))
    baseline_by_seed = {int(r["seed"]): r for r in baseline["rows"] if r.get("status") == "completed"}
    y6 = v3.read_labels(DEV_TRAIN); keep, y5 = v3.filter_five(y6)
    v2.TRAIN_PATH = DEV_TRAIN; v2.OUT_ROOT = STAGE_OUT
    data = {"train5": y5, "test5": y5, "train_valid": keep, "test_valid": keep, "manifest_indices": np.asarray(json.loads((AUDIT / "development_manifest.json").read_text(encoding="utf-8"))["selected_indices"], dtype=np.int64), "audit": {}}
    class Args:
        max_seeds = 5; epochs = 18; patience = 6; batch_size = 64; eval_batch_size = 256; draws_per_epoch = 9920
        head_epochs = 12; head_patience = 4; head_batch_size = 128; head_draws_per_epoch = 9920
    args = Args(); rows = []
    arm_map = (("MaxPositive-Focal", "max_positive"), ("MeanPositive-Focal", "mean_positive"))
    for seed in v2.SEEDS[:5]:
        seed = int(seed)
        roles_path = OUT / "splits" / f"seed_{seed}.json"
        roles = {k: np.asarray(v, dtype=np.int64) for k, v in json.loads(roles_path.read_text(encoding="utf-8")).items()}
        for arm, sampler in arm_map:
            print(f"START v3 stage1 sampler seed={seed} arm={arm}", flush=True)
            rows.append(v2.train_joint(arm, seed, roles, data, args, sampler, "focal", {}, "stage1"))
            print(f"DONE v3 stage1 sampler seed={seed} arm={arm}", flush=True)
    details = {}
    for arm, _ in arm_map:
        paired = []
        for r in rows:
            b = baseline_by_seed.get(int(r["seed"]))
            if b is None or r["arm"] != arm: continue
            paired.append({"seed": int(r["seed"]), "macro_delta": float(r["early"]["macro_aupr"] - b["early"]["macro_aupr"]), "tail_delta": float(r["early"]["tail_mean_aupr"] - b["early"]["tail_mean_aupr"]), "head_deltas": {n: float(r["early"]["per_label"][n]["aupr"] - b["early"]["per_label"][n]["aupr"]) for n in ("FMB", "SinFM", "MFT")}})
        details[arm] = {"paired": paired, "mean_macro_delta": float(np.mean([x["macro_delta"] for x in paired])), "mean_tail_delta": float(np.mean([x["tail_delta"] for x in paired])), "positive_seed_count": int(sum(x["macro_delta"] > 0 for x in paired)), "worst_head_delta": float(min(x["head_deltas"][n] for x in paired for n in x["head_deltas"])), "passed": False}
    summary = {"status": "completed_development_only", "stage": "stage1_sampler", "baseline": str(OUT / "stage1_summary.json"), "rows": rows, "sampler_gate": details, "stage1_gate_pass": False, "test_used": False, "confirmation_used": False, "hard_gate": {"macro_delta_min": 0.010, "tail_delta_min": 0.020, "positive_seeds_min": 4, "worst_head_delta_min": -0.010}}
    dump(STAGE_OUT / "stage1_sampler_summary.json", summary)
    dump(OUT / "stage1_sampler_receipt.json", {"status": "completed_development_only", "summary": str(STAGE_OUT / "stage1_sampler_summary.json"), "test_evaluated": False, "confirmation_evaluated": False})
    print(json.dumps({"status": summary["status"], "gate": False, "details": {k: {m: v for m, v in d.items() if m != "paired"} for k, d in details.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
