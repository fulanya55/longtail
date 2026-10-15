"""Audit whether frozen Focal-36 scores support asymmetric power teaching.

This is a development-only, read-only feasibility screen.  It deliberately
does not train a student, fit a calibrator, choose thresholds, or access a
test view.  It asks whether a frozen 30-dB model is more informative on the
same receiver identities at 50 dB than at 10 dB for MFT/VFT/ISRJ.  Only if
that prerequisite holds can a future *asymmetric posterior transfer* be
considered.  It is not representation-invariance or pseudo-label training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.v73_dataset import V73SignalDataset  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    _load_model_from_state,
)
from experiments.run_route_a_development_v3_mainstream_loss_baselines import (  # noqa: E402
    EXPECTED_V3_SHA256,
)
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402


TARGETS = (("MFT", 3), ("VFT", 4), ("ISRJ", 5))
SEEDS = (2039, 2040, 2041, 2042, 2043)
DEFAULT_DATA_ROOT = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v3")
DEFAULT_POWER_ROOT = DEFAULT_DATA_ROOT / "power_sweep_9level_v3"
DEFAULT_ROLE_ROOT = ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection"
DEFAULT_F2_ROOT = ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen"
DEFAULT_CONVERGENCE_ROOT = ROOT / "outputs/tapc/route_a_development_v3_loss_convergence_audit"
DEFAULT_SEAL = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/tapc/route_a_v3_asymmetric_power_teacher_prerequisite"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _rank_metrics(targets: np.ndarray, scores: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(targets).size != 2:
        raise ValueError("both positive and negative labels are required")
    count = max(1, int(np.ceil(0.10 * len(targets))))
    selected = np.argsort(scores, kind="mergesort")[-count:]
    prevalence = float(targets.mean())
    top_precision = float(targets[selected].mean())
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "aupr": float(average_precision_score(targets, scores)),
        "top10_precision": top_precision,
        "top10_lift_over_prevalence": float(top_precision / max(prevalence, 1e-12)),
        "prevalence": prevalence,
    }


def summarize_pair(high_logits: np.ndarray, low_logits: np.ndarray, targets: np.ndarray) -> dict:
    """Return threshold-free and fixed-top-decile diagnostics for one label."""

    high_prob, low_prob = _sigmoid(high_logits), _sigmoid(low_logits)
    target = np.asarray(targets, dtype=np.uint8)
    high, low = _rank_metrics(target, high_prob), _rank_metrics(target, low_prob)
    positive = target == 1
    negative = ~positive
    return {
        "high_50db": high,
        "low_10db": low,
        "high_minus_low": {
            "auroc": float(high["auroc"] - low["auroc"]),
            "aupr": float(high["aupr"] - low["aupr"]),
            "top10_precision": float(high["top10_precision"] - low["top10_precision"]),
            "positive_mean_probability": float(high_prob[positive].mean() - low_prob[positive].mean()),
            "positive_median_probability": float(np.median(high_prob[positive]) - np.median(low_prob[positive])),
            "negative_mean_probability": float(high_prob[negative].mean() - low_prob[negative].mean()),
        },
        "paired_positive_score_increase_fraction": float((high_prob[positive] > low_prob[positive]).mean()),
        "high_confidence_rescue_diagnostic": float(
            ((high_prob[positive] >= 0.75) & (low_prob[positive] < 0.50)).mean()
        ),
        "note": (
            "The 0.75/0.50 rescue quantity is descriptive only; no calibrated threshold or "
            "training target is selected by this audit."
        ),
    }


def _predict_targets(model: torch.nn.Module, path: Path, indices: np.ndarray,
                     normalization: str, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = V73SignalDataset(path=path, indices=indices, class_mode="six", normalization=normalization)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    logits, labels, sample_ids = [], [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                values = model(batch["signal"].to(device)).detach().cpu().numpy()
                logits.append(values[:, [index for _, index in TARGETS]].astype(np.float32))
                labels.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(sample_ids)


def _validate_seal(seal_path: Path) -> dict:
    seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
    fields = seal["test_v3"]
    if any(fields[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    return fields


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_seed_results(seed_rows: list[dict]) -> tuple[dict, dict]:
    aggregate = {}
    for name, _ in TARGETS:
        rows = [row["per_target"][name] for row in seed_rows]
        aggregate[name] = {
            "high_50db_auroc_mean": _mean([row["high_50db"]["auroc"] for row in rows]),
            "low_10db_auroc_mean": _mean([row["low_10db"]["auroc"] for row in rows]),
            "high_minus_low_auroc_mean": _mean([row["high_minus_low"]["auroc"] for row in rows]),
            "high_minus_low_aupr_mean": _mean([row["high_minus_low"]["aupr"] for row in rows]),
            "high_top10_lift_mean": _mean([row["high_50db"]["top10_lift_over_prevalence"] for row in rows]),
            "positive_score_increase_fraction_mean": _mean([row["paired_positive_score_increase_fraction"] for row in rows]),
            "positive_high_minus_low_auroc_seeds": int(sum(row["high_minus_low"]["auroc"] > 0.0 for row in rows)),
        }
    checks = {
        "each_target_high_50db_auroc_mean_ge_0p60": all(
            row["high_50db_auroc_mean"] >= 0.60 for row in aggregate.values()
        ),
        "each_target_high_minus_low_auroc_mean_ge_0p030": all(
            row["high_minus_low_auroc_mean"] >= 0.030 for row in aggregate.values()
        ),
        "each_target_high_minus_low_auroc_positive_in_at_least_4of5": all(
            row["positive_high_minus_low_auroc_seeds"] >= 4 for row in aggregate.values()
        ),
        "each_target_high_top10_lift_mean_ge_1p50": all(
            row["high_top10_lift_mean"] >= 1.50 for row in aggregate.values()
        ),
    }
    return aggregate, checks


def run(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite prior prerequisite audit: {args.output_root}")
    seal = _validate_seal(args.test_v3_manifest)
    original_train = args.data_root / "closedset_6class_train.mat"
    if _sha256(original_train).lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 base SHA mismatch")
    high_path = args.power_root / "level_09__jnr_50dB" / "closedset_6class_train.mat"
    low_path = args.power_root / "level_01__jnr_10dB" / "closedset_6class_train.mat"
    for path in (high_path, low_path):
        if not path.is_file() or path.name != "closedset_6class_train.mat" or "test" in str(path).lower():
            raise ValueError(f"invalid development power-view path: {path}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_rows = []
    args.output_root.mkdir(parents=True)
    source_hashes = {"development_v3_train": _sha256(original_train), "high_50db_train": _sha256(high_path), "low_10db_train": _sha256(low_path)}
    for seed in args.seeds:
        role_path = args.role_root / f"seed_{seed}" / "role_indices.npz"
        focal_path = args.convergence_root / f"seed_{seed}" / "E2_Focal_36" / "best_model.pth"
        f2_dir = args.f2_root / "runs" / "F2_DFMS_Full" / f"seed_{seed}"
        saved = np.load(role_path)
        indices = saved["early"].astype(np.int64)
        payload = torch.load(focal_path, map_location="cpu", weights_only=False)
        model = _load_model_from_state(f2_dir, device, payload["model_state_dict"])
        high_logits, high_labels, high_ids = _predict_targets(model, high_path, indices, args.normalization, device, args.batch_size)
        low_logits, low_labels, low_ids = _predict_targets(model, low_path, indices, args.normalization, device, args.batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not (np.array_equal(high_ids, low_ids) and np.array_equal(high_ids, indices)):
            raise AssertionError("power views do not preserve early-role identity order")
        if not np.array_equal(high_labels, low_labels):
            raise AssertionError("power views do not preserve labels")
        per_target = {
            name: summarize_pair(high_logits[:, position], low_logits[:, position], high_labels[:, index])
            for position, (name, index) in enumerate(TARGETS)
        }
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "early_target_logits.npz", sample_ids=high_ids, labels6=high_labels,
                            high_50db_logits=high_logits, low_10db_logits=low_logits)
        row = {
            "seed": int(seed), "status": "completed_frozen_inference_only", "early_count": int(len(indices)),
            "role_indices_sha256": _sha256(role_path), "focal36_checkpoint_sha256": _sha256(focal_path),
            "per_target": per_target, "prediction_path": str(seed_dir / "early_target_logits.npz"),
            "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        }
        _write_json(seed_dir / "seed_summary.json", row)
        seed_rows.append(row)
    aggregate, checks = aggregate_seed_results(seed_rows)
    result = {
        "status": "completed_frozen_prerequisite_audit", "experiment": "RouteA_V3_AsymmetricPowerTeacherPrerequisite",
        "evidence_role": "development_v3_feasibility_screen_not_new_model_evidence", "seeds": list(args.seeds),
        "power_views": {"student_low": str(low_path), "teacher_high": str(high_path), "fixed_jnr_reference": "30 dB"},
        "source_sha256": source_hashes, "test_v3_seal_sha256": _sha256(args.test_v3_manifest),
        "test_v3_seal_status": {key: seal[key] for key in ("access_authorized", "analysis_allowed", "used")},
        "aggregate": aggregate,
        "preregistered_checks": checks,
        "candidate_admitted_for_fresh_source_design": bool(all(checks.values())),
        "interpretation": {
            "passes_mean": "Only a frozen-score precondition holds; a new identity-isolated source and a separately preregistered multi-seed training test are still required.",
            "fails_mean": "Do not train asymmetric posterior transfer or tune a gate, temperature, loss weight, or view pair on this source.",
            "not_performed": ["student training", "teacher updating", "calibration", "threshold selection", "test-v2 access", "test-v3 access"],
        },
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(args.output_root / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--role-root", type=Path, default=DEFAULT_ROLE_ROOT)
    parser.add_argument("--f2-root", type=Path, default=DEFAULT_F2_ROOT)
    parser.add_argument("--convergence-root", type=Path, default=DEFAULT_CONVERGENCE_ROOT)
    parser.add_argument("--test-v3-manifest", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--normalization", default="complex_rms")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
