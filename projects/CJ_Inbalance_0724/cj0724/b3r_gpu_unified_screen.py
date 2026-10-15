"""Final B3-R screen: single read/transfer and all branches on GPU."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.b2r_mft_reliability import bounded_three_state, reliability_features
from cj0724.b3r_unified_pipeline import gpu_all_branches, read_and_normalize
from cj0724.models import build_benchmark_model
from cj0724.route_b_mechanism_router import baseline_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b3r_gpu_unified_screen_20260725"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    seed, device = int(args.seed), torch.device(args.device)
    baseline = baseline_root(seed)
    route = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
    repair = Path(f"outputs/b2rc_recall_constrained_20260725/seed_{seed}")
    audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
    indices = np.sort(np.load(baseline / "split_indices.npz")["development"].astype(np.int64))
    model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
    model.load_state_dict(torch.load(baseline / "best.pt", map_location=device, weights_only=False)["model_state"])
    model.eval()
    frozen = Path("outputs/vft_physics_e5_frozen_candidate_20260725")
    vft = joblib.load(frozen / f"seed_{seed}/vft_physics_classifier.joblib")
    mft = joblib.load(route / "mft_physics_classifier.joblib")
    isrj = joblib.load(route / "isrj_physics_classifier.joblib")
    delay = joblib.load(repair / "mft_delay_auxiliary.joblib")
    router = joblib.load(repair / "mft_three_state_router.joblib")
    stats = np.load(repair / "reference_stats.npz")
    protocol = json.loads((repair / "protocol.json").read_text(encoding="utf-8"))
    expected = {
        int(row["power_dB"]): row for row in csv.DictReader(
            Path("outputs/b2rc_recall_constrained_20260725/power_runs.csv").open(encoding="utf-8-sig"))
        if int(row["seed"]) == seed and row["variant"] == "three_state"
    }
    rows = []
    for power in (10, 30, 50):
        path = Path(audit["levels"][str(power)]["path"])
        iq, _, _ = read_and_normalize(path, indices)
        tick = time.perf_counter()
        e5_probability, _ = __import__("cj0724.b3r_unified_pipeline", fromlist=["gpu_e5_probabilities"]).gpu_e5_probabilities(
            iq, model, vft, device, args.batch_size)
        e5_compute = time.perf_counter() - tick
        total = time.perf_counter()
        iq, read_seconds, normalize_seconds = read_and_normalize(path, indices)
        e5, features, gpu = gpu_all_branches(iq, model, vft, device, args.batch_size)
        tree_tick = time.perf_counter()
        mft_probability = mft.predict_proba(features["combined"])[:, 1]
        isrj_probability = isrj.predict_proba(features["isrj_periodicity"])[:, 1]
        delay_probability = delay.predict_proba(features["mft_delay"])[:, 1]
        router_feature = reliability_features(e5[:, 3], mft_probability, delay_probability,
                                              features["combined"], stats["median"], stats["scale"])
        raw = router.predict_proba(router_feature)
        action = np.zeros((len(raw), 3), dtype=np.float64)
        action[:, router.classes_.astype(int)] = raw
        final = e5.copy()
        final[:, 3] = bounded_three_state(e5[:, 3], mft_probability, action,
                                          protocol["alpha"], protocol["confidence"])
        final[:, 5] = isrj_probability
        tree_seconds = time.perf_counter() - tree_tick
        total_seconds = time.perf_counter() - total
        saved = np.load(Path(f"outputs/b2_power_boundary_20260725/seed_{seed}/power_{power}dB.npz"))
        thresholds = saved["thresholds_E5"].astype(np.float64)
        thresholds[5] = saved["thresholds_B2-Hard"][5]
        metric = metrics_from_probabilities(saved["labels"], final, thresholds, saved["weak_mask"])
        reference = expected[power]
        rows.append({
            "power_dB": power, "E5_compute_only_seconds": e5_compute,
            "read_seconds": read_seconds, "normalize_seconds": normalize_seconds,
            **gpu, "tree_and_router_seconds": tree_seconds, "total_seconds": total_seconds,
            "compute_multiplier_vs_E5_compute": (total_seconds - read_seconds - normalize_seconds) / e5_compute,
            "macro_aupr_difference": float(metric["macro_aupr"] - float(reference["macro_aupr"])),
            "hamming_difference": float(metric["hamming_loss"] - float(reference["hamming"])),
        })
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "mean_compute_multiplier_vs_E5": float(np.mean([row["compute_multiplier_vs_E5_compute"] for row in rows])),
        "worst_compute_multiplier_vs_E5": float(max(row["compute_multiplier_vs_E5_compute"] for row in rows)),
        "mean_total_seconds": float(np.mean([row["total_seconds"] for row in rows])),
        "max_abs_macro_aupr_difference": float(max(abs(row["macro_aupr_difference"]) for row in rows)),
        "max_abs_hamming_difference": float(max(abs(row["hamming_difference"]) for row in rows)),
    }
    gates = {
        "mean_compute_at_most_1p5x_E5": summary["mean_compute_multiplier_vs_E5"] <= 1.5,
        "worst_compute_at_most_1p6x_E5": summary["worst_compute_multiplier_vs_E5"] <= 1.6,
        "macro_aupr_equivalent": summary["max_abs_macro_aupr_difference"] <= 0.003,
        "hamming_equivalent": summary["max_abs_hamming_difference"] <= 0.003,
    }
    result = {"seed": seed, "rows": rows, "summary": summary, "gates": gates,
              "B3R_screen_passed": bool(all(gates.values())), "locked_test_used": False}
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
