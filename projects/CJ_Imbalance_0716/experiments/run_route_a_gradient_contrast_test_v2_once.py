"""One-time frozen test-v2 evaluation for GradientContrastFixed1p5."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.engine.calibration import apply_temperature  # noqa: E402
from cjlt.engine.evaluator import compute_multilabel_metrics  # noqa: E402
from cjlt.engine.thresholding import apply_thresholds  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import (  # noqa: E402
    COMMON_INDICES,
    COMMON_NAMES,
    TARGET_NAMES,
    _comparison,
    _per_label_diagnostics,
    _read_json,
    _sha256,
    _target_weak_rank,
    _write_json,
)
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402


SEEDS = (2039, 2040, 2041, 2042, 2043)
VARIANTS = ("C6_Joint", "GradientContrastFixed1p5")
EXPECTED_TEST_SHA256 = "902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947"


def _artifact_map(bundle):
    return {item["role"]: item for item in bundle["artifacts"]}


def _validate_bundle(bundle_path, bundle):
    if bundle.get("status") != "frozen":
        raise ValueError("test bundle is not frozen")
    if bundle.get("candidate") != "GradientContrastFixed1p5":
        raise ValueError("frozen candidate drifted")
    if bundle.get("seeds") != list(SEEDS):
        raise ValueError("frozen seed list drifted")
    if bundle.get("test_v2", {}).get("expected_sha256") != EXPECTED_TEST_SHA256:
        raise ValueError("expected test-v2 SHA256 drifted")
    if bundle.get("locked_test_used") or bundle.get("test_v2_used"):
        raise ValueError("bundle evidence boundary is invalid")
    for artifact in bundle["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file():
            raise FileNotFoundError("frozen artifact missing: {}".format(path))
        if _sha256(path) != artifact["sha256"]:
            raise ValueError("frozen artifact hash mismatch: {}".format(path))
    artifact_roles = _artifact_map(bundle)
    runner_role = artifact_roles["frozen_test_runner"]
    if Path(runner_role["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("bundle runner path differs from executing runner")
    return artifact_roles


def _load_model(path, device):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("frozen checkpoint missing model_state_dict")
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _predict_file(model, data_path, device, batch_size):
    labels = read_labels(data_path)
    indices = np.arange(len(labels), dtype=np.int64)
    dataset = V73SignalDataset(
        path=data_path,
        indices=indices,
        class_mode="six",
        normalization="complex_rms",
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    logits, targets, sample_ids = [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(signal)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("frozen test inference produced non-finite logits")
                logits.append(output.detach().cpu().float().numpy())
                targets.append(batch["label"].numpy().astype(np.uint8))
                sample_ids.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    return (
        np.concatenate(logits, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(sample_ids, axis=0),
    )


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _evaluate_frozen(variant, logits6, targets6, sample_ids, weak6, temperature, thresholds):
    logits = logits6[:, COMMON_INDICES]
    targets = targets6[:, COMMON_INDICES]
    weak = weak6[sample_ids][:, COMMON_INDICES]
    probability = _sigmoid(apply_temperature(logits, float(temperature)))
    prediction = apply_thresholds(probability, np.asarray(thresholds, dtype=np.float64))
    metrics = compute_multilabel_metrics(
        targets,
        prediction,
        probabilities=probability,
        class_names=COMMON_NAMES,
        weak_component_mask=weak,
    )
    diagnostics = _per_label_diagnostics(targets, prediction, probability, weak)
    metrics["per_label_diagnostics"] = diagnostics
    metrics["target_mean_weak_recall"] = float(np.mean([
        diagnostics[name]["weak_recall"] for name in TARGET_NAMES
    ]))
    metrics["target_mean_negative_fpr"] = float(np.mean([
        diagnostics[name]["negative_fpr"] for name in TARGET_NAMES
    ]))
    metrics["target_weak_vs_negative_auroc"] = _target_weak_rank(targets, probability, weak)
    saw_present = targets6[:, 1].astype(bool)
    metrics["subgroups"] = {}
    for name, mask in (("sawfm_present", saw_present), ("sawfm_absent", ~saw_present)):
        metrics["subgroups"][name] = compute_multilabel_metrics(
            targets[mask],
            prediction[mask],
            probabilities=probability[mask],
            class_names=COMMON_NAMES,
            weak_component_mask=weak[mask],
        )
    return {
        "variant": variant,
        "temperature": float(temperature),
        "thresholds": [float(value) for value in thresholds],
        "metrics": metrics,
        "predictions": {
            "logits": logits,
            "probabilities": probability,
            "predictions": prediction,
            "targets": targets,
            "weak_mask": weak.astype(np.uint8),
            "sawfm_present": saw_present.astype(np.uint8),
            "sample_ids": sample_ids,
        },
    }


def _metric(row, path):
    value = row
    for key in path:
        value = value[key]
    return float(value)


def _variant_map(seed_row):
    return {row["variant"]: row for row in seed_row["variants"]}


def _aggregate(seed_rows):
    paths = {
        "macro_aupr": ("metrics", "macro_aupr"),
        "macro_f1": ("metrics", "macro_f1"),
        "hamming_loss": ("metrics", "hamming_loss"),
        "target_mean_weak_recall": ("metrics", "target_mean_weak_recall"),
        "target_mean_negative_fpr": ("metrics", "target_mean_negative_fpr"),
        "target_weak_vs_negative_auroc": ("metrics", "target_weak_vs_negative_auroc"),
        "sawfm_present_macro_f1": ("metrics", "subgroups", "sawfm_present", "macro_f1"),
        "sawfm_absent_macro_f1": ("metrics", "subgroups", "sawfm_absent", "macro_f1"),
    }
    result = {}
    for variant_index, variant in enumerate(VARIANTS):
        rows = [_variant_map(seed_row)[variant] for seed_row in seed_rows]
        result[variant] = {
            name: _mean_ci([_metric(row, path) for row in rows], 20261700 + 100 * variant_index + offset)
            for offset, (name, path) in enumerate(paths.items())
        }
        result[variant]["per_label_aupr"] = {
            label_name: _mean_ci(
                [row["metrics"]["per_label_diagnostics"][label_name]["aupr"] for row in rows],
                20261800 + 100 * variant_index + label_index,
            )
            for label_index, label_name in enumerate(COMMON_NAMES)
        }
    return result


def _gate(comparison, aggregate, hashes_match, finite):
    delta = comparison["aggregate"]
    candidate = aggregate["GradientContrastFixed1p5"]
    checks = {
        "macro_aupr_noninferior_minus_0p005": delta["macro_aupr"]["mean"] >= -0.005,
        "macro_f1_gain_0p010": delta["macro_f1"]["mean"] >= 0.010,
        "target_weak_recall_gain_0p030": delta["target_mean_weak_recall"]["mean"] >= 0.030,
        "target_label_aupr_noninferior_minus_0p010": all(
            delta["per_label_aupr"][name]["mean"] >= -0.010 for name in TARGET_NAMES
        ),
        "target_mean_fpr_0p20": candidate["target_mean_negative_fpr"]["mean"] <= 0.20,
        "positive_target_weak_recall_at_least_four": delta["positive_target_weak_recall_seeds"] >= 4,
        "sawfm_present_macro_f1_positive": delta["sawfm_present_macro_f1"]["mean"] > 0.0,
        "sawfm_absent_macro_f1_noninferior_minus_0p005": delta["sawfm_absent_macro_f1"]["mean"] >= -0.005,
        "frozen_hashes_match_and_outputs_finite": bool(hashes_match and finite),
        "fit_or_selection_on_test_v2_false": True,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _preflight_models(artifacts, device):
    for seed in SEEDS:
        for variant in VARIANTS:
            role = "{}_seed_{}_model".format(variant, seed)
            model = _load_model(artifacts[role]["path"], device)
            with torch.no_grad():
                output = model(torch.zeros(1, 2, 3000, device=device))
            if output.shape != (1, 6) or not torch.isfinite(output).all():
                raise ValueError("frozen model contract failed: {}".format(role))
            calibration = _read_json(artifacts["{}_seed_{}_calibration".format(variant, seed)]["path"])
            thresholds = _read_json(artifacts["{}_seed_{}_thresholds".format(variant, seed)]["path"])
            if not np.isfinite(float(calibration["temperature"])):
                raise ValueError("non-finite frozen temperature")
            values = np.asarray(thresholds["thresholds"], dtype=np.float64)
            if values.shape != (5,) or not np.isfinite(values).all():
                raise ValueError("invalid frozen thresholds")


def run(args):
    bundle = _read_json(args.bundle)
    artifacts = _validate_bundle(args.bundle, bundle)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _preflight_models(artifacts, device)
    if args.preflight:
        result = {
            "status": "preflight_passed",
            "artifact_count": len(bundle["artifacts"]),
            "model_count": len(SEEDS) * len(VARIANTS),
            "test_v2_opened": False,
            "locked_test_used": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    marker = Path(args.output_root) / "test_v2_access_started.json"
    if marker.exists():
        raise RuntimeError("test-v2 access marker already exists; rerun is forbidden")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    _write_json(marker, {
        "status": "access_started",
        "bundle": str(Path(args.bundle).resolve()),
        "candidate": "GradientContrastFixed1p5",
        "fit_or_selection_on_test_v2": False,
        "rerun_allowed": False,
    })
    test_path = Path(bundle["test_v2"]["path"])
    test_sha256 = _sha256(test_path)
    if test_sha256 != EXPECTED_TEST_SHA256:
        raise ValueError("test-v2 SHA256 mismatch after access marker")
    labels6 = read_labels(test_path)
    parameter_values, _ = read_parameters(test_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameter_values), 10.0)
    seed_rows = []
    finite = True
    for seed in SEEDS:
        variants = []
        for variant in VARIANTS:
            model = _load_model(artifacts["{}_seed_{}_model".format(variant, seed)]["path"], device)
            logits6, targets6, sample_ids = _predict_file(model, test_path, device, args.batch_size)
            finite = finite and bool(np.isfinite(logits6).all())
            if not np.array_equal(targets6, labels6[sample_ids]):
                raise AssertionError("test-v2 target/sample alignment failed")
            calibration = _read_json(artifacts["{}_seed_{}_calibration".format(variant, seed)]["path"])
            thresholds = _read_json(artifacts["{}_seed_{}_thresholds".format(variant, seed)]["path"])
            result = _evaluate_frozen(
                variant,
                logits6,
                targets6,
                sample_ids,
                weak6,
                calibration["temperature"],
                thresholds["thresholds"],
            )
            output = Path(args.output_root) / "seed_{}".format(seed) / variant
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "metrics.json", result["metrics"])
            np.savez_compressed(str(output / "predictions.npz"), **result.pop("predictions"))
            variants.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        seed_row = {
            "seed": int(seed),
            "variants": variants,
            "test_v2_sha256": test_sha256,
            "fit_or_selection_on_test_v2": False,
            "locked_test_used": False,
            "test_v2_used": True,
        }
        _write_json(Path(args.output_root) / "seed_{}".format(seed) / "seed_summary.json", seed_row)
        seed_rows.append(seed_row)
    aggregate = _aggregate(seed_rows)
    comparison = _comparison(seed_rows, "GradientContrastFixed1p5", "C6_Joint")
    gate = _gate(comparison, aggregate, hashes_match=True, finite=finite)
    payload = {
        "status": "completed",
        "experiment": "route_a_gradient_contrast_test_v2_once",
        "evidence_level": "C_independent_test" if gate["passed"] else "frozen_independent_negative",
        "test_v2_sha256": test_sha256,
        "seeds": list(SEEDS),
        "per_seed": seed_rows,
        "aggregate": aggregate,
        "comparison_candidate_minus_baseline": comparison,
        "promotion_gate": gate,
        "fit_or_selection_on_test_v2": False,
        "locked_test_used": False,
        "test_v2_used": True,
        "rerun_or_retuning_allowed": False,
    }
    output_root = Path(args.output_root)
    _write_json(output_root / "route_a_gradient_contrast_test_v2_summary.json", payload)
    _write_json(output_root / "run_receipt.json", {
        "status": "completed",
        "promotion_gate_passed": gate["passed"],
        "evidence_level": payload["evidence_level"],
        "fit_or_selection_on_test_v2": False,
        "locked_test_used": False,
        "test_v2_used": True,
        "rerun_or_retuning_allowed": False,
    })
    print(json.dumps({
        "status": payload["status"],
        "evidence_level": payload["evidence_level"],
        "comparison": comparison["aggregate"],
        "promotion_gate": gate,
        "fit_or_selection_on_test_v2": False,
        "rerun_or_retuning_allowed": False,
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_gradient_contrast_test_v2" / "frozen_bundle.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_gradient_contrast_test_v2_once_20260731")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
