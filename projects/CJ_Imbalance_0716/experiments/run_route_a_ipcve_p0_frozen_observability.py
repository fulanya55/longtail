"""Five-B0-seed frozen observability gate for Route-A IP-CVE P0."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGET_INDICES = (3, 4, 5)
B0_SEEDS = (2039, 2040, 2041, 2042, 2043)
SOURCE_PROJECTION_SEEDS = {
    "global_128": 2026080711,
    "temporal_bin8": 2026080712,
    "stage3_mag_bin8": 2026080713,
    "stage4_mag_bin8": 2026080714,
    "raw_iq_random_projection": 2026080715,
}
PROJECTION_DIM = 64
BOOTSTRAP_SEED = 2026080719


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complex(dataset) -> np.ndarray:
    raw = np.asarray(dataset)
    if raw.dtype.fields and {"real", "imag"}.issubset(raw.dtype.fields):
        return raw["real"].astype(np.float32) + 1j * raw["imag"].astype(np.float32)
    if np.iscomplexobj(raw):
        return raw.astype(np.complex64)
    raise TypeError("iq_complex is not complex")


def _load_data(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        complex_iq = _complex(handle["iq_complex"])
        result = {
            "labels": np.asarray(handle["label_selected"], dtype=np.uint8),
            "triple": np.asarray(handle["triple_id"]).reshape(-1).astype(np.int64),
            "target": np.asarray(handle["triple_target_index"]).reshape(-1).astype(np.int64) - 1,
            "level": np.asarray(handle["triple_level"]).reshape(-1).astype(np.int64),
            "nuisance": np.asarray(handle["nuisance_index"]).reshape(-1).astype(np.int64),
            "role": np.asarray(handle["role_code"]).reshape(-1).astype(np.int64),
        }
    iq = np.stack([complex_iq.real, complex_iq.imag], axis=1).astype(np.float32)
    del complex_iq
    rms = np.sqrt(np.mean(np.sum(iq * iq, axis=1), axis=1, keepdims=True))
    result["iq"] = iq / np.maximum(rms[:, :, None], 1e-8)
    lookup = np.full((int(result["triple"].max()) + 1, 3), -1, dtype=np.int64)
    lookup[result["triple"], result["level"]] = np.arange(len(result["triple"]))
    if np.any(lookup[1:] < 0):
        raise ValueError("every triple must contain absent/low/high views")
    result["lookup"] = lookup
    return result


def _projection(source_name: str, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(SOURCE_PROJECTION_SEEDS[source_name])
    matrix = rng.standard_normal((dimension, PROJECTION_DIM))
    q, _ = np.linalg.qr(matrix)
    return q[:, :PROJECTION_DIM].astype(np.float32)


def _project(source_name: str, values: np.ndarray) -> np.ndarray:
    return np.asarray(values @ _projection(source_name, values.shape[1]), dtype=np.float32)


def _raw_control(iq: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    rows = []
    for start in range(0, len(iq), batch_size):
        values = torch.from_numpy(iq[start : start + batch_size]).to(device)
        pooled = F.adaptive_avg_pool1d(values, 64).flatten(1)
        rows.append(pooled.detach().cpu().numpy().astype(np.float32))
    return _project("raw_iq_random_projection", np.concatenate(rows))


def _source_from_maps(name: str, maps: dict) -> torch.Tensor:
    if name == "global_128":
        return maps["global"]
    if name == "temporal_bin8":
        return F.adaptive_avg_pool1d(maps["temporal"], 8).flatten(1)
    if name == "stage3_mag_bin8":
        magnitude = torch.sqrt(
            maps["stage3_real"].square() + maps["stage3_imag"].square() + 1e-8
        )
        return F.adaptive_avg_pool1d(magnitude, 8).flatten(1)
    if name == "stage4_mag_bin8":
        magnitude = torch.sqrt(
            maps["stage4_real"].square() + maps["stage4_imag"].square() + 1e-8
        )
        return F.adaptive_avg_pool1d(magnitude, 8).flatten(1)
    raise ValueError("unknown source {}".format(name))


def _extract_sources(
    model,
    iq: np.ndarray,
    source_names: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    time_permutation: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    collected = {name: [] for name in source_names}
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(iq), batch_size):
            values = torch.from_numpy(iq[start : start + batch_size]).to(device)
            if time_permutation is not None:
                values = values[:, :, torch.as_tensor(time_permutation, device=device)]
            maps = model.forward_structure_maps(values)
            logits.append(model.forward_classifier(maps["global"]).cpu().numpy().astype(np.float32))
            for name in source_names:
                collected[name].append(
                    _source_from_maps(name, maps).cpu().numpy().astype(np.float32)
                )
    projected = {
        name: _project(name, np.concatenate(chunks))
        for name, chunks in collected.items()
    }
    return projected, np.concatenate(logits)


def _view_lookup(data: dict) -> np.ndarray:
    return data["lookup"]


def _role_triples(data: dict, role_code: int) -> np.ndarray:
    return np.unique(data["triple"][data["role"] == role_code])


def _residual_dataset(
    features: np.ndarray,
    data: dict,
    triples: np.ndarray,
    unmatched: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = _view_lookup(data)
    low_rows = np.array([lookup[int(value)][1] for value in triples], dtype=np.int64)
    absent_rows = np.array([lookup[int(value)][0] for value in triples], dtype=np.int64)
    targets = data["target"][low_rows] - TARGET_INDICES[0]
    if unmatched:
        rematched = absent_rows.copy()
        for target in range(3):
            for nuisance in range(1, 6):
                group = np.flatnonzero(
                    (targets == target) & (data["nuisance"][low_rows] == nuisance)
                )
                if len(group) < 2:
                    raise ValueError("unmatched group has fewer than two triples")
                rematched[group] = np.roll(absent_rows[group], 1)
        absent_rows = rematched
    return features[low_rows] - features[absent_rows], targets.astype(np.int64)


def _fit_binding(
    features: np.ndarray,
    data: dict,
    fit_triples: np.ndarray,
    seed: int,
    mode: str,
):
    residual, targets = _residual_dataset(
        features,
        data,
        fit_triples,
        unmatched=mode == "unmatched_background",
    )
    if mode == "component_permuted":
        targets = (targets + 1) % 3
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=int(seed),
        ),
    )
    model.fit(residual, targets)
    return model


def _scores(model, features: np.ndarray) -> np.ndarray:
    values = model.decision_function(features)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("binding readout must return three scores")
    return np.asarray(values, dtype=np.float64)


def _metrics(
    model,
    features: np.ndarray,
    data: dict,
    triples: np.ndarray,
) -> dict:
    lookup = _view_lookup(data)
    residual, binding_target = _residual_dataset(features, data, triples)
    binding_prediction = model.predict(residual)
    view_scores = _scores(model, features)
    per_label = {}
    aurocs, auprs, orders = [], [], []
    for local_target, label_index in enumerate(TARGET_INDICES):
        selected = triples[
            np.array(
                [data["target"][lookup[int(value)][0]] == label_index for value in triples]
            )
        ]
        absent = np.array([lookup[int(value)][0] for value in selected], dtype=np.int64)
        low = np.array([lookup[int(value)][1] for value in selected], dtype=np.int64)
        high = np.array([lookup[int(value)][2] for value in selected], dtype=np.int64)
        truth = np.r_[np.zeros(len(absent)), np.ones(len(low))]
        score = np.r_[view_scores[absent, local_target], view_scores[low, local_target]]
        auroc = float(roc_auc_score(truth, score))
        aupr = float(average_precision_score(truth, score))
        order = float(
            np.mean(
                (view_scores[high, local_target] > view_scores[low, local_target])
                & (view_scores[low, local_target] > view_scores[absent, local_target])
            )
        )
        per_label[CLASS_NAMES[label_index]] = {
            "weak_vs_absent_auroc": auroc,
            "weak_vs_absent_aupr": aupr,
            "high_low_absent_order_fraction": order,
            "triple_count": int(len(selected)),
        }
        aurocs.append(auroc)
        auprs.append(aupr)
        orders.append(order)
    return {
        "binding_accuracy": float(np.mean(binding_prediction == binding_target)),
        "mean_weak_vs_absent_auroc": float(np.mean(aurocs)),
        "mean_weak_vs_absent_aupr": float(np.mean(auprs)),
        "mean_order_fraction": float(np.mean(orders)),
        "per_label": per_label,
    }


def _bootstrap_mean_ci(values: np.ndarray, draws: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def _non_target_drift(logits: np.ndarray, data: dict, triples: np.ndarray) -> dict:
    lookup = _view_lookup(data)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    low_absent, high_low = [], []
    for value in triples:
        rows = lookup[int(value)]
        target = int(data["target"][rows[0]])
        other = [index for index in range(6) if index != target]
        low_absent.extend(np.abs(probability[rows[1], other] - probability[rows[0], other]))
        high_low.extend(np.abs(probability[rows[2], other] - probability[rows[1], other]))
    return {
        "mean_abs_low_absent": float(np.mean(low_absent)),
        "mean_abs_high_low": float(np.mean(high_low)),
        "candidate_induced_drift": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("E:/LJQ_code")
        / "干扰数据"
        / "route_a_ipcve_development_cf_v1_20260807"
        / "ipcve_triples_train.mat",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/route_a_ipcve/p0_frozen_observability",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data = _load_data(args.data_path)
    fit_triples = _role_triples(data, 0)
    early_triples = _role_triples(data, 1)
    evaluation_triples = _role_triples(data, 2)
    source_names = (
        "global_128",
        "temporal_bin8",
        "stage3_mag_bin8",
        "stage4_mag_bin8",
    )
    raw_feature = _raw_control(data["iq"], args.batch_size, device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    normal_features = {}
    normal_logits = {}
    early_rows = []
    for seed in B0_SEEDS:
        checkpoint_path = (
            PROJECT_ROOT
            / "outputs/tapc/route_a_backbone_b0_development_v1"
            / "seed_{}".format(seed)
            / "best_model.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_backbone("dfms_iq", 6)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        features, logits = _extract_sources(
            model, data["iq"], source_names, args.batch_size, device
        )
        normal_features[seed] = features
        normal_logits[seed] = logits
        np.savez_compressed(
            args.output_root / "normal_feature_cache_seed_{}.npz".format(seed),
            **features,
            logits=logits,
        )
        raw_model = _fit_binding(raw_feature, data, fit_triples, seed, "matched_true")
        raw_early = _metrics(raw_model, raw_feature, data, early_triples)
        for source_name in source_names:
            source = features[source_name]
            arms = {}
            for mode in ("matched_true", "unmatched_background", "component_permuted"):
                readout = _fit_binding(source, data, fit_triples, seed, mode)
                arms[mode] = _metrics(readout, source, data, early_triples)
            strongest = max(
                arms["unmatched_background"]["mean_weak_vs_absent_auroc"],
                arms["component_permuted"]["mean_weak_vs_absent_auroc"],
                raw_early["mean_weak_vs_absent_auroc"],
            )
            early_rows.append(
                {
                    "seed": seed,
                    "source": source_name,
                    "matched_true": arms["matched_true"],
                    "unmatched_background": arms["unmatched_background"],
                    "component_permuted": arms["component_permuted"],
                    "raw_iq_random_projection": raw_early,
                    "selection_delta_vs_strongest_control": float(
                        arms["matched_true"]["mean_weak_vs_absent_auroc"] - strongest
                    ),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    source_selection = []
    for source_name in source_names:
        selected = [row for row in early_rows if row["source"] == source_name]
        source_selection.append(
            {
                "source": source_name,
                "mean_selection_delta": float(
                    np.mean([row["selection_delta_vs_strongest_control"] for row in selected])
                ),
                "mean_true_weak_auroc": float(
                    np.mean(
                        [row["matched_true"]["mean_weak_vs_absent_auroc"] for row in selected]
                    )
                ),
                "mean_binding_accuracy": float(
                    np.mean([row["matched_true"]["binding_accuracy"] for row in selected])
                ),
            }
        )
    source_selection.sort(
        key=lambda row: (
            row["mean_selection_delta"],
            row["mean_binding_accuracy"],
            row["mean_true_weak_auroc"],
        ),
        reverse=True,
    )
    selected_source = source_selection[0]["source"]
    selection_receipt = {
        "status": "completed_early_source_selection_before_evaluation",
        "selected_source": selected_source,
        "ranking": source_selection,
        "evaluation_metrics_computed_before_selection": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    _write_json(args.output_root / "source_selection.json", selection_receipt)

    permutation = np.random.default_rng(2026080721).permutation(data["iq"].shape[-1])
    evaluation_rows = []
    for seed in B0_SEEDS:
        checkpoint_path = (
            PROJECT_ROOT
            / "outputs/tapc/route_a_backbone_b0_development_v1"
            / "seed_{}".format(seed)
            / "best_model.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_backbone("dfms_iq", 6)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        time_features, _ = _extract_sources(
            model,
            data["iq"],
            (selected_source,),
            args.batch_size,
            device,
            time_permutation=permutation,
        )
        source = normal_features[seed][selected_source]
        arms = {}
        for mode in ("matched_true", "unmatched_background", "component_permuted"):
            readout = _fit_binding(source, data, fit_triples, seed, mode)
            arms[mode] = {
                "early": _metrics(readout, source, data, early_triples),
                "evaluation": _metrics(readout, source, data, evaluation_triples),
            }
        time_model = _fit_binding(
            time_features[selected_source], data, fit_triples, seed, "matched_true"
        )
        arms["time_permuted"] = {
            "evaluation": _metrics(
                time_model, time_features[selected_source], data, evaluation_triples
            )
        }
        raw_model = _fit_binding(raw_feature, data, fit_triples, seed, "matched_true")
        arms["raw_iq_random_projection"] = {
            "evaluation": _metrics(raw_model, raw_feature, data, evaluation_triples)
        }
        evaluation_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "arms": arms,
                "non_target_b0_drift": _non_target_drift(
                    normal_logits[seed], data, evaluation_triples
                ),
            }
        )
        np.savez_compressed(
            args.output_root / "time_permuted_feature_cache_seed_{}.npz".format(seed),
            **time_features,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    deltas, binding_deltas = [], []
    per_label_deltas = {name: [] for name in CLASS_NAMES[3:6]}
    per_label_order_deltas = {name: [] for name in CLASS_NAMES[3:6]}
    enriched_rows = []
    for row in evaluation_rows:
        true = row["arms"]["matched_true"]["evaluation"]
        controls = {
            name: payload["evaluation"]
            for name, payload in row["arms"].items()
            if name != "matched_true"
        }
        strongest_name, strongest = max(
            controls.items(), key=lambda item: item[1]["mean_weak_vs_absent_auroc"]
        )
        delta = true["mean_weak_vs_absent_auroc"] - strongest["mean_weak_vs_absent_auroc"]
        strongest_binding = max(value["binding_accuracy"] for value in controls.values())
        binding_delta = true["binding_accuracy"] - strongest_binding
        deltas.append(delta)
        binding_deltas.append(binding_delta)
        label_delta = {}
        order_delta = {}
        for name in CLASS_NAMES[3:6]:
            strongest_label = max(
                value["per_label"][name]["weak_vs_absent_auroc"]
                for value in controls.values()
            )
            strongest_order = max(
                value["per_label"][name]["high_low_absent_order_fraction"]
                for value in controls.values()
            )
            label_delta[name] = true["per_label"][name]["weak_vs_absent_auroc"] - strongest_label
            order_delta[name] = (
                true["per_label"][name]["high_low_absent_order_fraction"]
                - strongest_order
            )
            per_label_deltas[name].append(label_delta[name])
            per_label_order_deltas[name].append(order_delta[name])
        enriched_rows.append(
            {
                **row,
                "strongest_weak_auroc_control": strongest_name,
                "weak_auroc_delta_vs_strongest": float(delta),
                "binding_delta_vs_strongest": float(binding_delta),
                "per_label_weak_auroc_delta_vs_strongest": label_delta,
                "per_label_order_delta_vs_strongest": order_delta,
            }
        )

    deltas_array = np.asarray(deltas)
    binding_array = np.asarray(binding_deltas)
    target_gate = {}
    for name in CLASS_NAMES[3:6]:
        label_values = np.asarray(per_label_deltas[name])
        order_values = np.asarray(per_label_order_deltas[name])
        ci = _bootstrap_mean_ci(label_values)
        checks = {
            "mean_weak_auroc_delta_ge_0p020": float(label_values.mean()) >= 0.020,
            "ci_lower_gt_zero": ci[0] > 0.0,
            "positive_seed_count_ge_4": int(np.sum(label_values > 0)) >= 4,
            "mean_order_delta_gt_zero": float(order_values.mean()) > 0.0,
        }
        target_gate[name] = {
            "mean_weak_auroc_delta": float(label_values.mean()),
            "weak_auroc_delta_95_ci": ci,
            "positive_seed_count": int(np.sum(label_values > 0)),
            "mean_order_delta": float(order_values.mean()),
            "checks": checks,
            "passed": all(checks.values()),
        }

    true_early = np.mean(
        [
            row["arms"]["matched_true"]["early"]["mean_weak_vs_absent_auroc"]
            for row in enriched_rows
        ]
    )
    true_evaluation = np.mean(
        [
            row["arms"]["matched_true"]["evaluation"]["mean_weak_vs_absent_auroc"]
            for row in enriched_rows
        ]
    )
    aggregate_checks = {
        "mean_weak_auroc_delta_ge_0p030": float(deltas_array.mean()) >= 0.030,
        "weak_auroc_delta_ci_lower_gt_zero": _bootstrap_mean_ci(deltas_array)[0] > 0.0,
        "positive_seed_count_ge_4": int(np.sum(deltas_array > 0)) >= 4,
        "no_label_mean_below_minus_0p010": all(
            np.mean(values) >= -0.010 for values in per_label_deltas.values()
        ),
        "mean_true_binding_accuracy_ge_0p60": float(
            np.mean(
                [
                    row["arms"]["matched_true"]["evaluation"]["binding_accuracy"]
                    for row in enriched_rows
                ]
            )
        )
        >= 0.60,
        "binding_delta_ci_lower_gt_zero": _bootstrap_mean_ci(binding_array)[0] > 0.0,
        "all_target_order_deltas_positive": all(
            np.mean(values) > 0.0 for values in per_label_order_deltas.values()
        ),
        "evaluation_to_early_gap_lt_0p15": abs(float(true_evaluation - true_early)) < 0.15,
        "candidate_induced_non_target_drift_zero": True,
    }
    aggregate_passed = all(aggregate_checks.values())
    passed_targets = [name for name, value in target_gate.items() if value["passed"]]
    if aggregate_passed and len(passed_targets) == 3:
        decision = "permit_unified_p1_overlap_reaudit"
    elif passed_targets:
        decision = "permit_narrow_label_reaudit_only_{}".format("_".join(passed_targets))
    else:
        decision = "stop_ipcve_and_move_to_extended_observation_protocol"

    result = {
        "status": "completed_five_b0_seed_frozen_p0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "RouteA_IPCVE_P0_FrozenObservability",
        "evidence_role": "fresh_development_only_frozen_representation_gate",
        "data": {
            "path": str(args.data_path),
            "sha256": _sha256(args.data_path),
            "fit_triples": int(len(fit_triples)),
            "early_triples": int(len(early_triples)),
            "evaluation_triples": int(len(evaluation_triples)),
        },
        "device": str(device),
        "b0_seeds": list(B0_SEEDS),
        "projection_dim": PROJECTION_DIM,
        "source_selection": selection_receipt,
        "early_source_rows": early_rows,
        "evaluation_rows": enriched_rows,
        "aggregate": {
            "selected_source": selected_source,
            "mean_true_early_weak_auroc": float(true_early),
            "mean_true_evaluation_weak_auroc": float(true_evaluation),
            "mean_weak_auroc_delta_vs_strongest": float(deltas_array.mean()),
            "weak_auroc_delta_95_ci": _bootstrap_mean_ci(deltas_array),
            "positive_seed_count": int(np.sum(deltas_array > 0)),
            "mean_binding_delta_vs_strongest": float(binding_array.mean()),
            "binding_delta_95_ci": _bootstrap_mean_ci(binding_array),
            "target_gates": target_gate,
            "checks": aggregate_checks,
            "passed": aggregate_passed and len(passed_targets) == 3,
            "passed_targets": passed_targets,
            "decision": decision,
        },
        "trained_backbone_parameters": 0,
        "calibration_used": False,
        "threshold_selection_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    summary_path = args.output_root / "p0_summary.json"
    _write_json(summary_path, result)
    _write_json(
        args.output_root / "run_receipt.json",
        {
            "status": "completed",
            "selected_source": selected_source,
            "gate_passed": result["aggregate"]["passed"],
            "decision": decision,
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        },
    )
    print(
        json.dumps(
            {
                "selected_source": selected_source,
                "passed": result["aggregate"]["passed"],
                "decision": decision,
                "mean_delta": result["aggregate"]["mean_weak_auroc_delta_vs_strongest"],
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
