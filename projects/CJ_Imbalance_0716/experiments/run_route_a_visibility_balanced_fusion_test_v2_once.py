"""One-shot independent test-v2 evaluation of the frozen Route-A candidate."""

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _component_power, _load_backbone  # noqa: E402
from experiments.run_route_a_h5_background_ablation_v2 import (  # noqa: E402
    COMMON_INDICES, _metrics, _sha256, _sigmoid, _write,
)
from experiments.run_route_a_bounded_local_contrast_v2 import (  # noqa: E402
    _load_frozen_c6, _target_weak_auroc,
)
from experiments.run_route_a_visibility_temporal_expert_v2 import (  # noqa: E402
    _extract, _load_probe, _strong_recall,
)
from experiments.run_route_a_visibility_balanced_fusion_v2 import (  # noqa: E402
    VisibilityBalancedFusion, _apply,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_bundle(bundle, manifest):
    for record in manifest["files"]:
        path = Path(record["frozen"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError("frozen bundle hash mismatch: {}".format(path))


def _exclusive_guard(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _load_bundle_models(bundle, seed, device):
    seed_root = bundle / f"seed_{seed}"
    backbone, _ = _load_backbone(seed_root / "backbone", "F2_DFMS_Full", device)
    base_checkpoint = torch.load(seed_root / "base" / "best_head.pth", map_location="cpu", weights_only=False)
    base_head = torch.nn.Linear(128, 6)
    base_head.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
    base_head.to(device).eval()
    probe_checkpoint = torch.load(seed_root / "expert" / "best_probe.pth", map_location="cpu", weights_only=False)
    probe = torch.nn.Linear(512, 3)
    probe.load_state_dict(probe_checkpoint["model_state_dict"], strict=True)
    probe.to(device).eval()
    fusion_checkpoint = torch.load(seed_root / "fusion" / "best_fusion.pth", map_location="cpu", weights_only=False)
    fusion = VisibilityBalancedFusion(fusion_checkpoint["residual_cap"],
                                      fusion_checkpoint["uncertainty_power"])
    fusion.load_state_dict(fusion_checkpoint["model_state_dict"], strict=True)
    fusion.to(device).eval()
    return backbone, base_head, base_checkpoint, probe, probe_checkpoint, fusion, fusion_checkpoint


def _variant_metrics(name, logits, residual, targets6, weak6, calibration_path,
                     thresholds_path):
    calibration = _read(calibration_path); thresholds = _read(thresholds_path)["thresholds"]
    probabilities6 = _sigmoid(apply_temperature(logits, calibration["temperature"]))
    targets5 = targets6[:, COMMON_INDICES]
    weak5 = weak6[:, COMMON_INDICES]
    probabilities5 = probabilities6[:, COMMON_INDICES]
    result = _metrics(targets5, probabilities5, weak5, thresholds,
                      targets6[:, 1].astype(bool))
    weak_auroc, per_auroc = _target_weak_auroc(targets5, probabilities5, weak5)
    strong_recall, per_strong = _strong_recall(targets5, probabilities5, weak5, thresholds)
    result.update({"variant": name, "target_weak_auroc": weak_auroc,
                   "per_label_target_weak_auroc": per_auroc,
                   "target_strong_recall": strong_recall,
                   "per_label_target_strong_recall": per_strong,
                   "residual_abs_mean": float(np.abs(residual).mean()),
                   "residual_abs_max": float(np.abs(residual).max()),
                   "calibration_temperature": calibration["temperature"],
                   "thresholds": thresholds, "evidence_role": "independent_test_v2",
                   "test_v2_used": True, "historic_locked_test_used": False})
    return result, probabilities6


def _aggregate(rows):
    metrics = ["macro_aupr", "macro_f1", "target_weak_recall", "target_weak_auroc",
               "target_negative_fpr", "target_strong_recall", "saw_present_macro_f1",
               "saw_absent_macro_f1"]
    aggregate = {}
    for vi, variant in enumerate(("B0_FrozenC6", "E1_VTSE_Bounded")):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {metric: _mean_ci([row[metric] for row in selected],
                                                181000 + vi * 100 + mi)
                              for mi, metric in enumerate(metrics)}
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        c = next(row for row in rows if row["seed"] == seed and row["variant"] == "E1_VTSE_Bounded")
        b = next(row for row in rows if row["seed"] == seed and row["variant"] == "B0_FrozenC6")
        paired.append({"seed": seed, **{metric: c[metric] - b[metric] for metric in metrics}})
    mean = {metric: float(np.mean([row[metric] for row in paired])) for metric in metrics}
    gates = {"macro_aupr_noninferior": mean["macro_aupr"] >= -0.005,
             "macro_f1_positive": mean["macro_f1"] > 0.0,
             "target_weak_recall_positive": mean["target_weak_recall"] > 0.0,
             "target_weak_auroc_positive": mean["target_weak_auroc"] > 0.0,
             "target_fpr_cap": aggregate["E1_VTSE_Bounded"]["target_negative_fpr"]["mean"] <= 0.20,
             "strong_recall_noninferior": mean["target_strong_recall"] >= -0.010,
             "four_of_five_positive_weak_gain": sum(row["target_weak_recall"] > 0 for row in paired) >= 4,
             "saw_present_noninferior": mean["saw_present_macro_f1"] >= -0.005}
    return aggregate, paired, mean, gates


def run(args):
    bundle = args.bundle_root.resolve()
    started = bundle / "test_v2_execution_started.json"
    completed = bundle / "test_v2_execution_receipt.json"
    if started.exists() or completed.exists():
        raise RuntimeError("test-v2 one-shot guard already exists; rerun is forbidden")
    manifest = _read(bundle / "bundle_manifest.json")
    authorization = _read(bundle / "test_v2_authorization.json")
    if not authorization.get("authorized") or not authorization.get("one_shot"):
        raise ValueError("test-v2 execution is not authorized")
    _verify_bundle(bundle, manifest)
    observed_hash = _sha256(args.test_v2)
    if observed_hash.lower() != authorization["test_v2_sha256"].lower():
        raise ValueError("test-v2 SHA-256 mismatch")
    _exclusive_guard(started, {"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                               "test_v2_sha256": observed_hash, "rerun_allowed": False})
    try:
        labels6 = read_labels(args.test_v2); values, _ = read_parameters(args.test_v2)
        weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
        indices = np.arange(len(labels6), dtype=np.int64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = []
        output_root = bundle / "test_v2_results"; output_root.mkdir(parents=True, exist_ok=True)
        for seed in manifest["seeds"]:
            print(f"test-v2 frozen seed={seed}", flush=True)
            seed_root = bundle / f"seed_{seed}"
            backbone, base_head, base_checkpoint, probe, probe_checkpoint, fusion, fusion_checkpoint = _load_bundle_models(bundle, seed, device)
            arrays = _extract(backbone, base_head, probe,
                              base_checkpoint["feature_mean"], base_checkpoint["feature_std"],
                              probe_checkpoint["mean"], probe_checkpoint["std"],
                              args.test_v2, indices, "complex_rms", device,
                              args.feature_batch_size, args.topk)
            candidate_logits, candidate_residual = _apply(
                fusion, arrays, fusion_checkpoint["center"], fusion_checkpoint["scale"],
                fusion_checkpoint["orientation"], device)
            baseline_residual = np.zeros_like(arrays["base_logits"])
            baseline, baseline_prob = _variant_metrics(
                "B0_FrozenC6", arrays["base_logits"], baseline_residual,
                labels6, weak6, seed_root / "baseline" / "calibration.json",
                seed_root / "baseline" / "thresholds.json")
            candidate, candidate_prob = _variant_metrics(
                "E1_VTSE_Bounded", candidate_logits, candidate_residual,
                labels6, weak6, seed_root / "candidate" / "calibration.json",
                seed_root / "candidate" / "thresholds.json")
            baseline["seed"] = seed; candidate["seed"] = seed
            seed_out = output_root / f"seed_{seed}"; seed_out.mkdir(parents=True, exist_ok=True)
            _write(seed_out / "baseline_metrics.json", baseline)
            _write(seed_out / "candidate_metrics.json", candidate)
            np.savez_compressed(seed_out / "predictions.npz", sample_ids=indices,
                                targets=labels6, baseline_probabilities=baseline_prob,
                                candidate_probabilities=candidate_prob,
                                candidate_residual=candidate_residual)
            rows.extend([baseline, candidate])
            del backbone, base_head, probe, fusion, arrays
            if device.type == "cuda": torch.cuda.empty_cache()
        aggregate, paired, mean, gates = _aggregate(rows)
        summary = {"status": "completed", "experiment": "RouteA_VisibilityBalancedFusion_test_v2_once",
                   "test_v2_path": str(args.test_v2.resolve()), "test_v2_sha256": observed_hash,
                   "sample_count": len(labels6), "seeds": manifest["seeds"],
                   "aggregate": aggregate, "paired_candidate_minus_baseline": paired,
                   "mean_deltas": mean, "independent_replication_gates": gates,
                   "all_gates_passed": bool(all(gates.values())),
                   "test_v2_used": True, "rerun_allowed": False,
                   "historic_locked_test_used": False, "fit_or_selection_on_test_v2": False}
        _write(output_root / "route_a_visibility_balanced_fusion_test_v2_summary.json", summary)
        _write(completed, {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                           "summary_path": str((output_root / "route_a_visibility_balanced_fusion_test_v2_summary.json").resolve()),
                           "test_v2_sha256": observed_hash, "all_gates_passed": bool(all(gates.values())),
                           "rerun_allowed": False})
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return summary
    except Exception as error:
        _write(bundle / "test_v2_execution_failure.json", {
            "status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(error), "rerun_allowed": False})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--test-v2", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v2\closedset_6class_test.mat"))
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
