"""Post-lock exploratory cross-seed expected-gain guard for Route-B C2.

The experiment reads only saved development predictions.  For each held seed,
the guard is trained on the other seeds after excluding every receiver identity
present in the held evaluation.  It never reads any locked-test file.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = tuple(range(2041, 2047))
TARGETS = ("FMB", "SinFM")
VARIANTS = {
    "E0_FrozenGate": {"loss": None, "quantile": None},
    "E1_ExpectedGain": {"loss": "squared_error", "quantile": None},
    "Q30_LowerGain": {"loss": "quantile", "quantile": 0.30},
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _source_roots(seed):
    if int(seed) <= 2043:
        routing = PROJECT_ROOT / "outputs/student_b/route_b_c2_visibility_safe_routing_confirmatory_20260725"
    else:
        routing = PROJECT_ROOT / "outputs/student_b/route_b_c2_visibility_safe_routing_replication_2044_2046_20260725"
    if int(seed) == 2041:
        c2 = PROJECT_ROOT / "outputs/student_b/supervised_joint_parameter_confirm_seed2041/C2/seed_2041"
    elif int(seed) <= 2043:
        c2 = PROJECT_ROOT / "outputs/student_b/supervised_joint_parameter_confirm_final/C2" / "seed_{}".format(seed)
    else:
        c2 = PROJECT_ROOT / "outputs/student_b/supervised_joint_parameter_replication_seeds_2044_2046_20260725/C2" / "seed_{}".format(seed)
    return routing / "seed_{}".format(seed), c2


def _read_labels(path):
    with h5py.File(str(path), "r") as handle:
        return np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T


def _feature_matrix(q, c2, local, mean, scale, class_probability, raw_rms, level):
    safe_scale = max(float(scale), 1.0e-12)
    return np.stack(
        [
            q,
            np.abs(local - c2) / safe_scale,
            np.abs(c2 - float(mean)) / safe_scale,
            np.abs(local - float(mean)) / safe_scale,
            class_probability,
            np.log(np.maximum(raw_rms, 1.0e-12)),
            np.asarray(level, dtype=np.float64) / 50.0,
        ],
        axis=1,
    ).astype(np.float32)


def _identity_exclusion_mask(training_identities, held_identities, candidate_gate):
    identities = np.asarray(training_identities, dtype=np.int64)
    held = np.asarray(held_identities, dtype=np.int64)
    gate = np.asarray(candidate_gate, dtype=bool)
    return gate & ~np.isin(identities, held)


def _load_seed(seed):
    routing_root, c2_root = _source_roots(seed)
    receipt_path = routing_root / "run_receipt.json"
    prediction_path = routing_root / "routing_predictions.npz"
    normalizer_path = c2_root / "parameter_normalizer.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("locked_test_used") or receipt.get("status") != "completed":
        raise ValueError("development routing receipt is not reusable")
    normalizers = {
        (row["class"], row["parameter"]): row
        for row in json.loads(normalizer_path.read_text(encoding="utf-8"))
    }
    output = {
        "source_hashes": {
            "receipt": _sha256(receipt_path),
            "predictions": _sha256(prediction_path),
            "normalizer": _sha256(normalizer_path),
        }
    }
    with np.load(prediction_path) as saved:
        for target in receipt["targets"]:
            class_name = target["class"]
            if class_name not in TARGETS:
                continue
            parameter_name = target["parameter"]
            prefix = "{}_{}".format(class_name, parameter_name)
            c2 = saved[prefix + "_c2_predictions"].astype(np.float64)
            local = saved[prefix + "_local_predictions"].astype(np.float64)
            q = saved[prefix + "_routing_probabilities"].astype(np.float64)
            expected = saved[prefix + "_targets"].astype(np.float64)
            identities = saved[prefix + "_identities"].astype(np.int64)
            base_indices = saved[prefix + "_base_indices"].astype(np.int64)
            level = saved[prefix + "_levels"].astype(np.int16)
            raw_rms = saved[prefix + "_raw_rms"].astype(np.float64)
            class_probability = saved[prefix + "_class_probabilities"].astype(np.float64)
            normalizer = normalizers[(class_name, parameter_name)]
            scale = float(normalizer["scale"])
            local_weight = float(target["chosen_local_weight"])
            candidate_gate = q >= float(target["chosen_threshold"])
            blended = c2 + local_weight * (local - c2)
            realized_gain = (
                np.abs(c2 - expected) - np.abs(blended - expected)
            ) / scale
            output[class_name] = {
                "parameter": parameter_name,
                "features": _feature_matrix(
                    q,
                    c2,
                    local,
                    float(normalizer["mean"]),
                    scale,
                    class_probability,
                    raw_rms,
                    level,
                ),
                "realized_gain": realized_gain.astype(np.float32),
                "candidate_gate": candidate_gate,
                "c2": c2,
                "blended": blended,
                "expected": expected,
                "identities": identities,
                "base_indices": base_indices,
                "levels": level,
                "scale": scale,
            }
    return output


def _fit_guard(features, gains, variant, seed):
    config = VARIANTS[variant]
    if config["loss"] is None:
        return None
    arguments = {
        "loss": config["loss"],
        "learning_rate": 0.06,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": int(seed),
    }
    if config["quantile"] is not None:
        arguments["quantile"] = float(config["quantile"])
    model = HistGradientBoostingRegressor(**arguments)
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(gains, dtype=np.float32))
    return model


def _combination_rows(labels, base_indices, c2_error, final_error, final_gate):
    keys = np.asarray(
        ["".join(str(int(value)) for value in labels[index]) for index in base_indices]
    )
    rows = {}
    for key in sorted(np.unique(keys)):
        selected = keys == key
        base = float(c2_error[selected].mean())
        final = float(final_error[selected].mean())
        rows[str(key)] = {
            "count": int(selected.sum()),
            "c2_nmae": base,
            "final_nmae": final,
            "relative_reduction": float((base - final) / max(base, 1.0e-12)),
            "route_coverage": float(final_gate[selected].mean()),
        }
    return rows


def _evaluate_target(payload, predicted_gain, labels):
    base_gate = payload["candidate_gate"]
    final_gate = base_gate if predicted_gain is None else base_gate & (predicted_gain > 0.0)
    final = np.where(final_gate, payload["blended"], payload["c2"])
    c2_error = np.abs(payload["c2"] - payload["expected"]) / payload["scale"]
    final_error = np.abs(final - payload["expected"]) / payload["scale"]
    combination = _combination_rows(
        labels,
        payload["base_indices"],
        c2_error,
        final_error,
        final_gate,
    )
    c2_nmae = float(c2_error.mean())
    final_nmae = float(final_error.mean())
    return {
        "c2_nmae": c2_nmae,
        "final_nmae": final_nmae,
        "relative_reduction": float(
            (c2_nmae - final_nmae) / max(c2_nmae, 1.0e-12)
        ),
        "route_coverage": float(final_gate.mean()),
        "candidate_route_coverage": float(base_gate.mean()),
        "guard_retention": float(final_gate.sum() / max(base_gate.sum(), 1)),
        "negative_combination_count": int(
            sum(row["relative_reduction"] < 0.0 for row in combination.values())
        ),
        "minimum_combination_reduction": float(
            min(row["relative_reduction"] for row in combination.values())
        ),
        "by_combination": combination,
        "final_gate": final_gate,
        "final_prediction": final.astype(np.float32),
    }


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "values": array.tolist(),
    }


def _aggregate(rows, variant):
    targets = []
    for class_name in TARGETS:
        selected = [row["targets"][class_name] for row in rows]
        targets.append(
            {
                "class": class_name,
                "relative_reduction": _mean_std(
                    [row["relative_reduction"] for row in selected]
                ),
                "positive_seed_count": int(
                    sum(row["relative_reduction"] > 0.0 for row in selected)
                ),
                "route_coverage": _mean_std(
                    [row["route_coverage"] for row in selected]
                ),
                "negative_combination_count_total": int(
                    sum(row["negative_combination_count"] for row in selected)
                ),
                "minimum_combination_reduction": float(
                    min(row["minimum_combination_reduction"] for row in selected)
                ),
            }
        )
    combined = []
    for row in rows:
        c2 = np.mean([row["targets"][name]["c2_nmae"] for name in TARGETS])
        final = np.mean([row["targets"][name]["final_nmae"] for name in TARGETS])
        combined.append(1.0 - final / c2)
    return {
        "variant": variant,
        "targets": targets,
        "mean_seed_target_relative_reduction": float(
            np.mean(
                [
                    row["targets"][name]["relative_reduction"]
                    for row in rows
                    for name in TARGETS
                ]
            )
        ),
        "minimum_seed_target_relative_reduction": float(
            min(
                row["targets"][name]["relative_reduction"]
                for row in rows
                for name in TARGETS
            )
        ),
        "two_target_combined_relative_reduction": _mean_std(combined),
        "two_target_positive_seed_count": int(sum(value > 0.0 for value in combined)),
        "negative_combination_count_total": int(
            sum(
                row["targets"][name]["negative_combination_count"]
                for row in rows
                for name in TARGETS
            )
        ),
        "minimum_combination_reduction": float(
            min(
                row["targets"][name]["minimum_combination_reduction"]
                for row in rows
                for name in TARGETS
            )
        ),
        "identity_overlap_max": int(max(row["identity_overlap"] for row in rows)),
    }


def run(args):
    if "test" in str(args.data_path).lower():
        raise ValueError("this exploratory runner must never read a test file")
    labels = _read_labels(args.data_path)
    data = {seed: _load_seed(seed) for seed in SEEDS}
    all_variants = {}
    for variant_offset, variant in enumerate(VARIANTS):
        rows = []
        for seed in SEEDS:
            targets = {}
            held_all = np.unique(
                np.concatenate(
                    [data[seed][name]["identities"] for name in TARGETS]
                )
            )
            held_identity = set(held_all.tolist())
            training_identity = set()
            prediction_payload = {}
            for target_offset, class_name in enumerate(TARGETS):
                held = data[seed][class_name]
                held_values = np.unique(held["identities"])
                predicted_gain = None
                training_count = 0
                if variant != "E0_FrozenGate":
                    train_x = []
                    train_y = []
                    for other_seed in SEEDS:
                        if other_seed == seed:
                            continue
                        source = data[other_seed][class_name]
                        keep = _identity_exclusion_mask(
                            source["identities"],
                            held_all,
                            source["candidate_gate"],
                        )
                        training_identity.update(source["identities"][keep].tolist())
                        train_x.append(source["features"][keep])
                        train_y.append(source["realized_gain"][keep])
                    train_x = np.concatenate(train_x, axis=0)
                    train_y = np.concatenate(train_y, axis=0)
                    training_count = int(len(train_y))
                    model = _fit_guard(
                        train_x,
                        train_y,
                        variant,
                        910000 + variant_offset * 10000 + seed + target_offset * 100,
                    )
                    predicted_gain = model.predict(held["features"]).astype(np.float32)
                metrics = _evaluate_target(held, predicted_gain, labels)
                prediction_payload[class_name + "_predicted_gain"] = (
                    np.full(len(held["expected"]), np.nan, dtype=np.float32)
                    if predicted_gain is None
                    else predicted_gain
                )
                prediction_payload[class_name + "_final_gate"] = metrics.pop(
                    "final_gate"
                ).astype(np.uint8)
                prediction_payload[class_name + "_final_prediction"] = metrics.pop(
                    "final_prediction"
                )
                prediction_payload[class_name + "_base_indices"] = held["base_indices"]
                prediction_payload[class_name + "_identities"] = held["identities"]
                targets[class_name] = {
                    **metrics,
                    "guard_training_rows": training_count,
                    "held_identity_count": int(len(held_values)),
                }
            overlap = len(held_identity & training_identity)
            if overlap:
                raise ValueError("cross-seed guard leaked held receiver identities")
            seed_root = args.output_root / variant / "seed_{}".format(seed)
            seed_root.mkdir(parents=True, exist_ok=True)
            prediction_path = seed_root / "guard_predictions.npz"
            np.savez_compressed(str(prediction_path), **prediction_payload)
            row = {
                "status": "completed",
                "variant": variant,
                "seed": int(seed),
                "targets": targets,
                "held_identity_count": int(len(held_identity)),
                "training_identity_count": int(len(training_identity)),
                "identity_overlap": int(overlap),
                "predictions_sha256": _sha256(prediction_path),
                "source_hashes": data[seed]["source_hashes"],
                "locked_test_used": False,
                "post_locked_test_exploratory": True,
            }
            _write_json(seed_root / "run_receipt.json", row)
            rows.append(row)
        all_variants[variant] = {
            "aggregate": _aggregate(rows, variant),
            "runs": rows,
        }
    baseline = all_variants["E0_FrozenGate"]["aggregate"]
    expected_gain = all_variants["E1_ExpectedGain"]["aggregate"]
    promotion = {
        "mean_gain_improved": bool(
            expected_gain["mean_seed_target_relative_reduction"]
            > baseline["mean_seed_target_relative_reduction"]
        ),
        "minimum_seed_target_improved": bool(
            expected_gain["minimum_seed_target_relative_reduction"]
            > baseline["minimum_seed_target_relative_reduction"]
        ),
        "negative_combinations_reduced": bool(
            expected_gain["negative_combination_count_total"]
            < baseline["negative_combination_count_total"]
        ),
        "identity_safe": expected_gain["identity_overlap_max"] == 0,
    }
    promotion["passed"] = bool(all(promotion.values()))
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "post_locked_test_exploratory": True,
        "eligible_for_existing_locked_test_claim": False,
        "protocol": {
            "seeds": list(SEEDS),
            "primary_variant": "E1_ExpectedGain",
            "features": [
                "routing_probability",
                "normalized_local_c2_disagreement",
                "normalized_c2_distance_from_training_mean",
                "normalized_local_distance_from_training_mean",
                "class_probability",
                "log_raw_rms",
                "normalized_power_level",
            ],
            "training": "leave-one-seed-out with held receiver identities excluded",
            "input_source": "saved development confirmatory predictions only",
        },
        "variants": {
            name: value["aggregate"] for name, value in all_variants.items()
        },
        "primary_promotion_gate": promotion,
        "runs": all_variants,
    }
    _write_json(
        args.output_root / "route_b_c2_cross_seed_expected_gain_guard_summary.json",
        payload,
    )
    print(json.dumps({"variants": payload["variants"], "promotion": promotion}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/student_b/route_b_c2_cross_seed_expected_gain_guard_exploratory_20260726",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
