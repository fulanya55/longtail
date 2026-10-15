"""B3 representative-power equivalence and speed screen for the GPU shared front-end."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from cj0724.b1_physics_sufficiency import extract_physics_features
from cj0724.route_b_mechanism_router import baseline_root
from cj0724.shared_physics_frontend import extract_shared_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b3_shared_frontend_screen_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    seed = 20260724
    baseline = baseline_root(seed)
    route = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
    audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
    indices = np.sort(np.load(baseline / "split_indices.npz")["development"].astype(np.int64))
    labels = np.load(Path("outputs/b2_power_boundary_20260725") /
                     f"seed_{seed}/power_10dB.npz")["labels"]
    mft = joblib.load(route / "mft_physics_classifier.joblib")
    isrj = joblib.load(route / "isrj_physics_classifier.joblib")
    rows = []
    for power in (10, 30, 50):
        path = Path(audit["levels"][str(power)]["path"])
        started = time.perf_counter()
        reference = extract_physics_features(path, indices, args.batch_size)
        reference_seconds = time.perf_counter() - started
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        candidate = extract_shared_features(path, indices, device, args.batch_size)
        if device.type == "cuda": torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - started
        mft_reference = mft.predict_proba(reference["combined"])[:, 1]
        mft_candidate = mft.predict_proba(candidate["combined"])[:, 1]
        isrj_reference = isrj.predict_proba(reference["isrj_periodicity"])[:, 1]
        isrj_candidate = isrj.predict_proba(candidate["isrj_periodicity"])[:, 1]
        rows.append({
            "power_dB": power, "samples": int(len(indices)),
            "reference_seconds": reference_seconds, "shared_seconds": candidate_seconds,
            "speedup": reference_seconds / candidate_seconds,
            "max_abs_combined_feature_difference": float(np.max(np.abs(reference["combined"] - candidate["combined"]))),
            "max_abs_isrj_feature_difference": float(np.max(np.abs(reference["isrj_periodicity"] - candidate["isrj_periodicity"]))),
            "MFT_probability_max_abs_difference": float(np.max(np.abs(mft_reference - mft_candidate))),
            "ISRJ_probability_max_abs_difference": float(np.max(np.abs(isrj_reference - isrj_candidate))),
            "MFT_aupr_delta": float(average_precision_score(labels[:, 3], mft_candidate) -
                                    average_precision_score(labels[:, 3], mft_reference)),
            "ISRJ_aupr_delta": float(average_precision_score(labels[:, 5], isrj_candidate) -
                                     average_precision_score(labels[:, 5], isrj_reference)),
        })
        print(json.dumps(rows[-1]), flush=True)
    mean_shared = float(np.mean([row["shared_seconds"] for row in rows]))
    e5_seconds = float(json.loads(Path("outputs/vft_physics_e5_frozen_candidate_20260725/audit.json").read_text())["benchmark"]["e5_total_seconds"])
    result = {
        "rows": rows,
        "gates": {
            "MFT_aupr_drop_within_0p003": min(row["MFT_aupr_delta"] for row in rows) >= -0.003,
            "ISRJ_aupr_drop_within_0p003": min(row["ISRJ_aupr_delta"] for row in rows) >= -0.003,
            "total_latency_vs_E5_at_most_1p5": (e5_seconds + mean_shared) / e5_seconds <= 1.5,
        },
        "cost": {"e5_seconds": e5_seconds, "shared_seconds_mean": mean_shared,
                 "estimated_total_seconds": e5_seconds + mean_shared,
                 "latency_multiplier_vs_E5": (e5_seconds + mean_shared) / e5_seconds},
        "locked_test_used": False,
    }
    result["B3_screen_passed"] = bool(all(result["gates"].values()))
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
