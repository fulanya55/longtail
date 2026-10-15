"""One-shot locked-test evaluation for the frozen Route-A RawSaw router."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels, read_parameters  # noqa: E402
from cjlt.engine.combination_metrics import (  # noqa: E402
    compute_cardinality_metrics,
    compute_combination_metrics,
)
from cjlt.engine.locked_test import sha256_file, validate_locked_test_access  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import FrozenF2Head  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_raw_platt_five_seed_label_routing import (  # noqa: E402
    CLASS_NAMES,
    F2_MODEL_NAME,
    SAW_INDEX,
    _component_power,
)


ACCESS_MARKER = "locked_test_access_started.json"
FINAL_RECEIPT = "run_receipt.json"
PRIMARY_NAME = "RawSawCrossSeedFPR15"
BASELINE_NAME = "H2_Baseline"


def _sigmoid(values):
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))


def _verify_file(path, expected_sha256, label):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError(f"{label} hash drift: expected={expected_sha256} actual={actual}")
    return resolved


def _validate_bundle(bundle, bundle_path, expected_bundle_sha256):
    _verify_file(bundle_path, expected_bundle_sha256, "inference bundle")
    if bundle.get("status") != "ready_for_authorization":
        raise ValueError("bundle status drifted")
    if bundle.get("candidate") != "RawSawCrossSeedFPR15+H2Others":
        raise ValueError("candidate drifted")
    if bundle.get("ensemble") is not False:
        raise ValueError("ensemble must remain disabled")
    if bundle.get("class_order") != list(CLASS_NAMES):
        raise ValueError("class order drifted")
    if int(bundle.get("sawfm_index", -1)) != SAW_INDEX:
        raise ValueError("SawFM index drifted")
    seeds = [int(row["seed"]) for row in bundle["seeds"]]
    if seeds != [2039, 2040, 2041, 2042, 2043]:
        raise ValueError("seed list drifted")
    expected_thresholds = {2039: 0.48, 2040: 0.45, 2041: 0.48, 2042: 0.48, 2043: 0.48}
    for row in bundle["seeds"]:
        seed = int(row["seed"])
        if abs(float(row["raw_sawfm"]["threshold"]) - expected_thresholds[seed]) > 1e-12:
            raise ValueError(f"seed {seed} threshold drifted")
        if row.get("locked_test_used"):
            raise ValueError(f"seed {seed} bundle already used locked test")
    return seeds


def _validate_authorization(payload, bundle_path, bundle_sha256):
    if payload.get("status") != "locked_test_configuration_frozen":
        raise ValueError("authorization freeze status drifted")
    if payload.get("locked_test_authorized") is not True:
        raise ValueError("authorization freeze does not record approval")
    protocol = payload["protocol"]
    if protocol.get("scope") != "closedset_6class_test.mat only":
        raise ValueError("locked-test scope drifted")
    if protocol.get("fit_or_selection_on_locked_test") is not False:
        raise ValueError("locked-test fitting or selection must be false")
    if protocol.get("ensemble") is not False:
        raise ValueError("ensemble must remain false")
    frozen = payload["frozen_candidate"]
    if Path(frozen["path"]).resolve() != Path(bundle_path).resolve():
        raise ValueError("authorization bundle path drifted")
    if frozen["sha256"].lower() != str(bundle_sha256).lower():
        raise ValueError("authorization bundle hash drifted")
    runner = payload["runner"]
    if Path(runner["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("authorization runner path drifted")
    if runner["sha256"].lower() != sha256_file(Path(__file__).resolve()).lower():
        raise ValueError("authorization runner hash drifted")
    if len(payload.get("promotion_gate", {})) != 10:
        raise ValueError("promotion gate must contain ten frozen checks")


def _verify_bundle_sources(bundle):
    verified = {}
    for name, row in bundle["source_files"].items():
        verified[f"source.{name}"] = str(_verify_file(row["path"], row["sha256"], name))
    for seed_row in bundle["seeds"]:
        seed = int(seed_row["seed"])
        for name, row in (("f2", seed_row["f2"]), ("h2", seed_row["h2"]), ("raw", seed_row["raw_sawfm"])):
            verified[f"seed_{seed}.{name}"] = str(_verify_file(row["path"], row["sha256"], f"seed {seed} {name}"))
        h2 = seed_row["h2"]
        raw = seed_row["raw_sawfm"]
        for name, path_key, hash_key in (
            ("h2_temperature", "temperature_file", "temperature_file_sha256"),
            ("h2_threshold", "threshold_file", "threshold_file_sha256"),
            ("raw_calibration", "calibration_file", "calibration_file_sha256"),
            ("raw_threshold", "threshold_file", "threshold_file_sha256"),
        ):
            verified[f"seed_{seed}.{name}"] = str(_verify_file(
                (h2 if name.startswith("h2") else raw)[path_key],
                (h2 if name.startswith("h2") else raw)[hash_key],
                f"seed {seed} {name}",
            ))
    return verified


def _load_models(seed_row, device):
    f2_path = Path(seed_row["f2"]["path"])
    backbone, loaded_f2 = _load_backbone(f2_path.parent, F2_MODEL_NAME, device)
    if loaded_f2.resolve() != f2_path.resolve():
        raise ValueError("F2 loader path drifted")
    h2_checkpoint = torch.load(seed_row["h2"]["path"], map_location="cpu", weights_only=False)
    h2 = FrozenF2Head(feature_dim=128, num_labels=6, saw_expert=False, expert_weight=0.20)
    h2.load_state_dict(h2_checkpoint["model_state_dict"], strict=True)
    h2.to(device).eval()
    raw_checkpoint = torch.load(seed_row["raw_sawfm"]["path"], map_location="cpu", weights_only=False)
    raw = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=16, topk=16, dropout=0.1, descriptor_mode="full"
    )
    raw.load_state_dict(raw_checkpoint["model_state_dict"], strict=True)
    raw.to(device).eval()
    for model in (backbone, h2, raw):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return backbone, h2, raw, h2_checkpoint


def _model_preflight(bundle):
    device = torch.device("cpu")
    for seed_row in bundle["seeds"]:
        models = _load_models(seed_row, device)
        del models


def _predict_seed(seed_row, test_path, sample_count, normalization, device, batch_size):
    backbone, h2, raw, h2_checkpoint = _load_models(seed_row, device)
    mean = torch.as_tensor(h2_checkpoint["main_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(h2_checkpoint["main_std"], dtype=torch.float32, device=device)
    temperature = float(seed_row["h2"]["temperature"])
    coefficient = float(seed_row["raw_sawfm"]["platt_coefficient"])
    intercept = float(seed_row["raw_sawfm"]["platt_intercept"])
    dataset = V73SignalDataset(
        path=test_path,
        indices=np.arange(int(sample_count), dtype=np.int64),
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    h2_parts, raw_parts, target_parts, id_parts = [], [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    features = backbone.forward_features(signal).float()
                    normalized = (features - mean) / std
                    h2_logits = h2(normalized, normalized)
                    raw_logits = raw(signal[:, 0:1], signal[:, 1:2], None, None)
                h2_parts.append(h2_logits.detach().cpu().float().numpy())
                raw_parts.append(raw_logits.detach().cpu().float().numpy())
                target_parts.append(batch["label"].numpy().astype(np.uint8))
                id_parts.append(np.asarray(batch["sample_index"], dtype=np.int64))
    finally:
        dataset.close()
    h2_logits = np.concatenate(h2_parts, axis=0).astype(np.float64)
    raw_logits = np.concatenate(raw_parts, axis=0).reshape(-1).astype(np.float64)
    targets = np.concatenate(target_parts, axis=0).astype(np.uint8)
    sample_ids = np.concatenate(id_parts, axis=0).astype(np.int64)
    h2_probability = _sigmoid(h2_logits / temperature)
    raw_probability = _sigmoid(coefficient * raw_logits + intercept)
    routed_probability = h2_probability.copy()
    routed_probability[:, SAW_INDEX] = raw_probability
    h2_thresholds = np.asarray(seed_row["h2"]["thresholds"], dtype=np.float64)
    routed_thresholds = h2_thresholds.copy()
    routed_thresholds[SAW_INDEX] = float(seed_row["raw_sawfm"]["threshold"])
    return {
        "sample_ids": sample_ids,
        "targets": targets,
        "h2_logits": h2_logits,
        "raw_logits": raw_logits,
        "h2_probability": h2_probability,
        "raw_probability": raw_probability,
        "routed_probability": routed_probability,
        "h2_thresholds": h2_thresholds,
        "routed_thresholds": routed_thresholds,
    }


def _ece(probability, targets, bins=10):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(float(probability[mask].mean()) - float(targets[mask].mean()))
    return float(result)


def _metrics(probability, targets, thresholds, weak_mask):
    probability = np.asarray(probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.uint8)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    prediction = probability >= thresholds.reshape(1, -1)
    per_label = {}
    for index, name in enumerate(CLASS_NAMES):
        positive = targets[:, index].astype(bool)
        negative = ~positive
        weak = weak_mask[:, index]
        strong = positive & ~weak
        per_label[name] = {
            "aupr": float(average_precision_score(targets[:, index], probability[:, index])),
            "recall": float(prediction[positive, index].mean()),
            "fpr": float(prediction[negative, index].mean()),
            "weak_recall": float(prediction[weak, index].mean()) if weak.any() else None,
            "weak_count": int(weak.sum()),
            "strong_recall": float(prediction[strong, index].mean()) if strong.any() else None,
            "strong_count": int(strong.sum()),
            "ece_10bin": _ece(probability[:, index], targets[:, index]),
        }
    weak_prediction = prediction[weak_mask]
    return {
        "macro_aupr": float(average_precision_score(targets, probability, average="macro")),
        "micro_aupr": float(average_precision_score(targets, probability, average="micro")),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, prediction)),
        "exact_match": float(np.all(prediction == targets.astype(bool), axis=1).mean()),
        "weak_component_recall": float(weak_prediction.mean()) if weak_prediction.size else None,
        "per_label": per_label,
    }, prediction.astype(np.uint8)


def _seed_metrics(seed, arrays, weak_mask):
    baseline, baseline_prediction = _metrics(
        arrays["h2_probability"], arrays["targets"], arrays["h2_thresholds"], weak_mask
    )
    candidate, candidate_prediction = _metrics(
        arrays["routed_probability"], arrays["targets"], arrays["routed_thresholds"], weak_mask
    )
    other = [index for index in range(len(CLASS_NAMES)) if index != SAW_INDEX]
    probability_delta = float(np.max(np.abs(
        arrays["routed_probability"][:, other] - arrays["h2_probability"][:, other]
    )))
    threshold_delta = float(np.max(np.abs(
        arrays["routed_thresholds"][other] - arrays["h2_thresholds"][other]
    )))
    prediction_equal = bool(np.array_equal(
        candidate_prediction[:, other], baseline_prediction[:, other]
    ))
    result = {
        "seed": int(seed),
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            "macro_aupr": candidate["macro_aupr"] - baseline["macro_aupr"],
            "macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
            "exact_match": candidate["exact_match"] - baseline["exact_match"],
            "saw_recall": candidate["per_label"]["SawFM"]["recall"] - baseline["per_label"]["SawFM"]["recall"],
            "saw_fpr": candidate["per_label"]["SawFM"]["fpr"] - baseline["per_label"]["SawFM"]["fpr"],
            "saw_weak_recall": candidate["per_label"]["SawFM"]["weak_recall"] - baseline["per_label"]["SawFM"]["weak_recall"],
        },
        "other_label_invariance": {
            "max_probability_delta": probability_delta,
            "max_threshold_delta": threshold_delta,
            "predictions_equal": prediction_equal,
            "passed": probability_delta == 0.0 and threshold_delta == 0.0 and prediction_equal,
        },
        "combination": {
            BASELINE_NAME: compute_combination_metrics(arrays["targets"], baseline_prediction, CLASS_NAMES),
            PRIMARY_NAME: compute_combination_metrics(arrays["targets"], candidate_prediction, CLASS_NAMES),
        },
        "cardinality": {
            BASELINE_NAME: compute_cardinality_metrics(arrays["targets"], baseline_prediction),
            PRIMARY_NAME: compute_cardinality_metrics(arrays["targets"], candidate_prediction),
        },
        "locked_test_used": True,
        "fit_or_selection_on_locked_test": False,
    }
    return result, baseline_prediction, candidate_prediction


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "count": int(len(array))}


def _aggregate(rows):
    metrics = ("macro_aupr", "micro_aupr", "macro_f1", "hamming_loss", "exact_match", "weak_component_recall")
    result = {}
    for role in ("baseline", "candidate"):
        result[role] = {metric: _mean_std([row[role][metric] for row in rows]) for metric in metrics}
        result[role]["SawFM"] = {
            metric: _mean_std([row[role]["per_label"]["SawFM"][metric] for row in rows])
            for metric in ("aupr", "recall", "fpr", "weak_recall", "strong_recall", "ece_10bin")
        }
    result["deltas"] = {
        metric: _mean_std([row["deltas"][metric] for row in rows])
        for metric in ("macro_aupr", "macro_f1", "exact_match", "saw_recall", "saw_fpr", "saw_weak_recall")
    }
    return result


def _promotion_gate(rows, aggregate, source_audit):
    checks = {
        "five_seed_and_hash_audit": len(rows) == 5 and bool(source_audit),
        "other_five_labels_exactly_invariant": all(row["other_label_invariance"]["passed"] for row in rows),
        "each_seed_saw_fpr_0p20": all(row["candidate"]["per_label"]["SawFM"]["fpr"] <= 0.20 + 1e-12 for row in rows),
        "mean_macro_aupr_not_down": aggregate["candidate"]["macro_aupr"]["mean"] >= aggregate["baseline"]["macro_aupr"]["mean"] - 1e-12,
        "mean_macro_f1_not_down": aggregate["candidate"]["macro_f1"]["mean"] >= aggregate["baseline"]["macro_f1"]["mean"] - 1e-12,
        "mean_exact_match_delta_minus_0p01": aggregate["deltas"]["exact_match"]["mean"] >= -0.01 - 1e-12,
        "mean_saw_recall_gain_0p20": aggregate["deltas"]["saw_recall"]["mean"] >= 0.20 - 1e-12,
        "mean_saw_weak_recall_gain_0p20": aggregate["deltas"]["saw_weak_recall"]["mean"] >= 0.20 - 1e-12,
        "no_fit_or_selection_on_locked_test": all(not row["fit_or_selection_on_locked_test"] for row in rows),
        "one_shot_artifacts_complete": True,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _exclusive_marker(path, payload):
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(args):
    bundle_path = Path(args.bundle).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    seeds = _validate_bundle(bundle, bundle_path, args.expected_bundle_sha256)
    authorization_path = validate_locked_test_access(
        args.authorize_locked_test, args.configuration_freeze, args.expected_freeze_sha256
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    _validate_authorization(authorization, bundle_path, args.expected_bundle_sha256)
    source_audit = _verify_bundle_sources(bundle)
    _model_preflight(bundle)
    if args.preflight_only:
        payload = {
            "status": "preflight_passed", "seeds": seeds,
            "bundle_sha256": sha256_file(bundle_path),
            "configuration_freeze_sha256": sha256_file(authorization_path),
            "source_count": len(source_audit), "locked_test_accessed": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"one-shot output directory already exists: {output_root}")
    marker_payload = {
        "status": "access_started", "one_shot": True,
        "candidate": bundle["candidate"],
        "bundle_sha256": sha256_file(bundle_path),
        "configuration_freeze_sha256": sha256_file(authorization_path),
        "fit_or_selection_on_locked_test": False,
    }
    _exclusive_marker(output_root / ACCESS_MARKER, marker_payload)

    test_path = Path(args.data_root) / "closedset_6class_test.mat"
    locked_test_sha256 = sha256_file(test_path)
    labels = select_labels(read_labels(test_path), "six").astype(np.uint8)
    values79, parameter_mask = read_parameters(test_path)
    if len(labels) != len(values79) or len(labels) != len(parameter_mask):
        raise ValueError("locked-test labels and parameters length mismatch")
    weak_mask = compute_weak_component_mask(labels, _component_power(values79), 10.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed_row in bundle["seeds"]:
        seed = int(seed_row["seed"])
        print(f"locked test seed={seed}", flush=True)
        arrays = _predict_seed(
            seed_row, test_path, len(labels), "complex_rms", device, args.batch_size
        )
        if not np.array_equal(arrays["targets"], labels):
            raise ValueError(f"seed {seed} target order drifted")
        metrics, baseline_prediction, candidate_prediction = _seed_metrics(seed, arrays, weak_mask)
        rows.append(metrics)
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            seed_root / "predictions.npz",
            sample_ids=arrays["sample_ids"], targets=arrays["targets"],
            h2_probability=arrays["h2_probability"].astype(np.float32),
            raw_saw_probability=arrays["raw_probability"].astype(np.float32),
            routed_probability=arrays["routed_probability"].astype(np.float32),
            h2_thresholds=arrays["h2_thresholds"].astype(np.float32),
            routed_thresholds=arrays["routed_thresholds"].astype(np.float32),
            h2_prediction=baseline_prediction, routed_prediction=candidate_prediction,
            weak_mask=weak_mask.astype(np.uint8),
        )
        _write(seed_root / "metrics.json", metrics)
        _write(seed_root / "run_receipt.json", {
            "status": "completed", "seed": seed,
            "prediction_sha256": sha256_file(seed_root / "predictions.npz"),
            "metrics_sha256": sha256_file(seed_root / "metrics.json"),
            "locked_test_used": True, "fit_or_selection_on_locked_test": False,
        })

    aggregate = _aggregate(rows)
    gate = _promotion_gate(rows, aggregate, source_audit)
    summary = {
        "status": "completed", "candidate": bundle["candidate"],
        "device": str(device), "locked_test_used": True,
        "locked_test_path": str(test_path.resolve()),
        "locked_test_sha256": locked_test_sha256,
        "sample_count": int(len(labels)), "class_order": list(CLASS_NAMES),
        "weak_threshold_db": 10.0, "weak_component_count": int(weak_mask.sum()),
        "protocol": {
            "one_shot": True, "ensemble": False,
            "fit_or_selection_on_locked_test": False,
            "network_calibrator_threshold_frozen": True,
        },
        "source_audit": source_audit,
        "per_seed": rows, "aggregate": aggregate,
        "promotion_gate": gate,
        "final_decision": "locked_test_passed" if gate["passed"] else "locked_test_failed_no_retuning_allowed",
    }
    summary_path = output_root / "route_a_raw_cross_seed_locked_test_summary.json"
    _write(summary_path, summary)
    receipt = {
        "status": "completed", "action": "one_shot_locked_test",
        "candidate": bundle["candidate"],
        "summary_sha256": sha256_file(summary_path),
        "bundle_sha256": sha256_file(bundle_path),
        "configuration_freeze_sha256": sha256_file(authorization_path),
        "locked_test_sha256": locked_test_sha256,
        "locked_test_used": True, "fit_or_selection_on_locked_test": False,
        "promotion_gate_passed": bool(gate["passed"]),
        "rerun_or_retuning_allowed": False,
    }
    _write(output_root / FINAL_RECEIPT, receipt)
    print(json.dumps({"aggregate": aggregate, "promotion_gate": gate, "receipt": receipt}, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-locked-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--configuration-freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_saw_cross_seed_locked_test_preflight" / "frozen_inference_bundle.json")
    parser.add_argument("--expected-bundle-sha256", default="14d3667dd545f5eaa8d1d69881050fa05da529ddec7b2a48d7af724c65fb3cea")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_raw_cross_seed_locked_test_once_20260726")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

