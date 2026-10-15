"""Audit new paired frequency-development data and freeze IF10/20/50 manifests."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import build_combo_longtail, build_label_longtail
from cjlt.data.split_builder import combo_key
from cjlt.data.v73_dataset import inspect_v73_layout, read_labels, read_parameters


DATA_ROOT = Path(r"E:\LJQ_code\干扰数据\dual-axis-frequency-development-v2")
LOW_PATH = DATA_ROOT / "sawfm_level_03__jnr_20dB/closedset_6class_train.mat"
HIGH_PATH = DATA_ROOT / "sawfm_level_05__jnr_30dB/closedset_6class_train.mat"
SWEEP_MANIFEST = DATA_ROOT / "power_sweep_manifest.csv"
FORMAL_RECEIPT = (
    ROOT
    / "outputs/route_a_frequency_reconstruction_v2/stage1_data_generation/formal_generation_receipt.json"
)
STAGE0_AUDIT = (
    ROOT / "outputs/route_a_frequency_reconstruction_v2/stage0/stage0_audit.json"
)
OUTPUT_DIR = ROOT / "outputs/route_a_frequency_reconstruction_v2/stage1_data_audit"

CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
ROLE_SEED = 3930100001
ROLE_COUNTS_PER_COMBO = {
    "optimization": 240,
    "early": 40,
    "calibration": 40,
    "threshold": 40,
    "evaluation": 40,
}
FB_SEED = 3930100010
FB_TARGET = np.asarray([1200] * 6, dtype=np.int64)
LABEL_SPECS = {
    "if10": (3930100011, np.asarray([1800, 180, 1136, 717, 452, 285], dtype=np.int64), 10.0),
    "if20": (3930100012, np.asarray([1500, 75, 824, 453, 249, 137], dtype=np.int64), 20.0),
    "if50": (3930100013, np.asarray([800, 16, 366, 167, 77, 35], dtype=np.int64), 50.0),
}
COMBO_SPECS = {
    "if10": (3930100021, 10.0),
    "if20": (3930100022, 20.0),
    "if50": (3930100023, 50.0),
}
UINT32_MODULUS = 4294967296
RECEIVER_SEED_SALT = 2 * 2246822519
EXPECTED_SAMPLE_SEED_FIRST = 3930000001
EXPECTED_SAMPLE_SEED_LAST = 3930025200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def indices_sha256(values: np.ndarray) -> str:
    return array_sha256(values, "<i8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_cell_vectors(path: Path, keys: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for key in keys:
            refs = handle[key][0, :]
            result[key] = [np.asarray(handle[ref]).reshape(-1) for ref in refs]
    return result


def build_roles(labels: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(np.asarray(labels, dtype=np.uint8)):
        grouped.setdefault(combo_key(row), []).append(index)
    expected = sum(ROLE_COUNTS_PER_COMBO.values())
    rng = np.random.RandomState(ROLE_SEED)
    roles = {name: [] for name in ROLE_COUNTS_PER_COMBO}
    for name in sorted(grouped):
        values = np.asarray(grouped[name], dtype=np.int64)
        if len(values) != expected:
            raise ValueError(f"combination {name} has {len(values)}, expected {expected}")
        rng.shuffle(values)
        start = 0
        for role, count in ROLE_COUNTS_PER_COMBO.items():
            roles[role].extend(values[start : start + count].tolist())
            start += count
    return {name: np.asarray(values, dtype=np.int64) for name, values in roles.items()}


def audit_roles(roles: dict[str, np.ndarray], sample_count: int) -> dict[str, Any]:
    names = list(roles)
    intersections: dict[str, int] = {}
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            intersections[f"{left}__{right}"] = int(
                len(np.intersect1d(roles[left], roles[right]))
            )
    merged = np.concatenate(list(roles.values()))
    return {
        "seed": ROLE_SEED,
        "counts": {name: int(len(values)) for name, values in roles.items()},
        "sha256": {name: indices_sha256(values) for name, values in roles.items()},
        "pairwise_intersection_counts": intersections,
        "union_count": int(len(np.unique(merged))),
        "covers_all_identities": int(len(np.unique(merged))) == int(sample_count),
        "all_pairwise_disjoint": all(value == 0 for value in intersections.values()),
    }


def power_contract(labels: np.ndarray) -> dict[str, Any]:
    keys = (
        "component_target_power_dB",
        "relative_power_dB",
        "weak_component_mask",
        "component_seeds",
    )
    low = read_cell_vectors(LOW_PATH, keys)
    high = read_cell_vectors(HIGH_PATH, keys)
    target_ok = relative_ok = weak_ok = component_seed_ok = True
    low_weak_saw_count = 0
    for index, label in enumerate(labels):
        active = np.flatnonzero(label)
        low_target = np.asarray(
            [20.0 if class_index == 1 else 30.0 for class_index in active],
            dtype=np.float64,
        )
        high_target = np.full(len(active), 30.0, dtype=np.float64)
        low_relative = np.zeros(len(active), dtype=np.float64)
        low_weak = np.zeros(len(active), dtype=bool)
        if len(active) > 1 and 1 in active:
            saw_position = int(np.flatnonzero(active == 1)[0])
            low_relative[saw_position] = -10.0
            low_weak[saw_position] = True
        target_ok &= np.array_equal(low[keys[0]][index], low_target)
        target_ok &= np.array_equal(high[keys[0]][index], high_target)
        relative_ok &= np.array_equal(low[keys[1]][index], low_relative)
        relative_ok &= np.array_equal(high[keys[1]][index], np.zeros(len(active)))
        weak_ok &= np.array_equal(low[keys[2]][index].astype(bool), low_weak)
        weak_ok &= not np.any(high[keys[2]][index].astype(bool))
        component_seed_ok &= np.array_equal(low[keys[3]][index], high[keys[3]][index])
        low_weak_saw_count += int(low_weak.sum())
    return {
        "only_sawfm_target_power_changes_as_preregistered": bool(target_ok),
        "relative_power_truth_matches_0_and_minus10db": bool(relative_ok),
        "weak_mask_matches_sawfm_compound_only": bool(weak_ok),
        "component_seeds_equal_across_views": bool(component_seed_ok),
        "low_view_weak_sawfm_identity_count": int(low_weak_saw_count),
        "high_view_weak_component_count": 0,
    }


def signal_probe(path: Path, indices: tuple[int, ...]) -> dict[str, Any]:
    rows = []
    with h5py.File(path, "r") as handle:
        for index in indices:
            signal = np.asarray(handle["data_legacy"][:, index], dtype=np.float64)
            power = float(np.mean(signal[:3000] ** 2 + signal[3000:] ** 2))
            rows.append(
                {
                    "index": int(index),
                    "all_finite": bool(np.isfinite(signal).all()),
                    "nonzero": bool(np.any(signal != 0.0)),
                    "rms": float(np.sqrt(max(power, 0.0))),
                }
            )
    return {"rows": rows, "passed": all(x["all_finite"] and x["nonzero"] for x in rows)}


def manifest_if(payload: dict[str, Any], kind: str) -> float:
    if kind == "label":
        counts = np.asarray(payload["achieved_label_counts"], dtype=np.float64)
    else:
        counts = np.asarray(list(payload["combo_counts"].values()), dtype=np.float64)
    return float(counts.max() / counts.min())


def make_manifests(labels: np.ndarray, optimization: np.ndarray, source_sha: str) -> dict[str, Any]:
    fb = build_label_longtail(
        labels,
        optimization,
        FB_TARGET,
        seed=FB_SEED,
        tolerance=0.0,
        source_split_sha256=source_sha,
        target_sample_count=2400,
        sample_count_tolerance=0.0,
    ).to_dict()
    label_manifests: dict[str, Any] = {}
    for name, (seed, target, target_if) in LABEL_SPECS.items():
        target_sample_count = 1600 if name == "if20" else None
        payload = build_label_longtail(
            labels,
            optimization,
            target,
            seed=seed,
            tolerance=0.05,
            source_split_sha256=source_sha,
            target_sample_count=target_sample_count,
            sample_count_tolerance=0.0,
        ).to_dict()
        payload["actual_imbalance_factor"] = manifest_if(payload, "label")
        payload["target_imbalance_factor"] = target_if
        label_manifests[name] = payload
    combo_manifests: dict[str, Any] = {}
    for name, (seed, target_if) in COMBO_SPECS.items():
        payload = build_combo_longtail(
            labels,
            optimization,
            imbalance_factor=target_if,
            seed=seed,
            source_split_sha256=source_sha,
            min_samples=3,
        ).to_dict()
        payload["achieved_sample_count"] = len(payload["selected_indices"])
        payload["actual_imbalance_factor"] = manifest_if(payload, "combo")
        payload["target_imbalance_factor"] = target_if
        combo_manifests[name] = payload
    return {"frequency_balanced": fb, "label": label_manifests, "combination": combo_manifests}


def build_audit() -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    formal = json.loads(FORMAL_RECEIPT.read_text(encoding="utf-8"))
    stage0 = json.loads(STAGE0_AUDIT.read_text(encoding="utf-8"))
    labels_low = read_labels(LOW_PATH)
    labels_high = read_labels(HIGH_PATH)
    with h5py.File(LOW_PATH, "r") as low, h5py.File(HIGH_PATH, "r") as high:
        receiver_low = np.asarray(low["receiver_noise_seed"][0, :], dtype=np.int64)
        receiver_high = np.asarray(high["receiver_noise_seed"][0, :], dtype=np.int64)
        target_fields_equal = True
        for key in ("target_delay_s", "target_range_m", "target_present", "unknown_present"):
            left = low[key][:]
            right = high[key][:]
            if np.issubdtype(left.dtype, np.floating):
                target_fields_equal &= np.array_equal(left, right, equal_nan=True)
            else:
                target_fields_equal &= np.array_equal(left, right)
    recovered_sample_low = (
        receiver_low.astype(np.int64) - RECEIVER_SEED_SALT
    ) % UINT32_MODULUS
    recovered_sample_high = (
        receiver_high.astype(np.int64) - RECEIVER_SEED_SALT
    ) % UINT32_MODULUS
    low_values, low_mask = read_parameters(LOW_PATH)
    high_values, high_mask = read_parameters(HIGH_PATH)
    roles = build_roles(labels_low)
    role_audit = audit_roles(roles, len(labels_low))
    manifests = make_manifests(labels_low, roles["optimization"], role_audit["sha256"]["optimization"])
    power = power_contract(labels_low)
    identity = {
        "sample_count": int(len(labels_low)),
        "class_names": list(CLASS_NAMES),
        "label_counts": dict(zip(CLASS_NAMES, labels_low.sum(axis=0).astype(int).tolist())),
        "labels_equal_across_views": bool(np.array_equal(labels_low, labels_high)),
        "labels_sha256": array_sha256(labels_low, "u1"),
        "receiver_noise_seed_equal_across_views": bool(np.array_equal(receiver_low, receiver_high)),
        "receiver_noise_seed_unique": int(len(np.unique(receiver_low))) == len(receiver_low),
        "receiver_noise_seed_sha256": array_sha256(receiver_low, "<i8"),
        "receiver_noise_seed_first": int(receiver_low.min()),
        "receiver_noise_seed_last": int(receiver_low.max()),
        "receiver_noise_seed_derivation": (
            "receiver_noise_seed = (sample_seed + 2*2246822519) mod 2^32; "
            "the mapping is bijective over uint32"
        ),
        "recovered_sample_seed_equal_across_views": bool(
            np.array_equal(recovered_sample_low, recovered_sample_high)
        ),
        "recovered_sample_seed_unique": int(len(np.unique(recovered_sample_low))) == len(recovered_sample_low),
        "recovered_sample_seed_first": int(recovered_sample_low.min()),
        "recovered_sample_seed_last": int(recovered_sample_low.max()),
        "formal_sample_seed_signature": formal.get("train_seed_signatures", []),
        "target_and_unknown_fields_equal_across_views": bool(target_fields_equal),
        "stage0_generator_collision_count": len(stage0["generator_collisions"]),
        "stage0_json_range_collision_count": len(stage0["json_range_collisions"]),
    }
    schema = {
        "parameter_masks_equal_across_views": bool(np.array_equal(low_mask, high_mask)),
        "low_active_parameters_finite": bool(np.isfinite(low_values[low_mask]).all()),
        "high_active_parameters_finite": bool(np.isfinite(high_values[high_mask]).all()),
        "low_signal_probe": signal_probe(LOW_PATH, (0, 400, 2400, 12599, 25199)),
        "high_signal_probe": signal_probe(HIGH_PATH, (0, 400, 2400, 12599, 25199)),
    }
    label_if_ok = all(
        abs(payload["actual_imbalance_factor"] / payload["target_imbalance_factor"] - 1.0) <= 0.05 + 1e-12
        and max(payload["label_count_error"]) <= 0.05 + 1e-12
        for payload in manifests["label"].values()
    )
    combo_if_ok = all(
        abs(payload["actual_imbalance_factor"] / payload["target_imbalance_factor"] - 1.0) <= 0.05
        for payload in manifests["combination"].values()
    )
    all_manifests_trainable = True
    for payload in [manifests["frequency_balanced"], *manifests["label"].values(), *manifests["combination"].values()]:
        selected = np.asarray(payload["selected_indices"], dtype=np.int64)
        counts = labels_low[selected].sum(axis=0)
        all_manifests_trainable &= bool(np.all(counts > 0) and np.all(counts < len(selected)))
    checks = {
        "formal_generation_completed": formal.get("status") == "completed" and formal.get("executed") is True,
        "two_views_have_25200_samples": len(labels_low) == len(labels_high) == 25200,
        "identity_pairing_exact": bool(
            identity["labels_equal_across_views"]
            and identity["receiver_noise_seed_equal_across_views"]
            and identity["receiver_noise_seed_unique"]
            and identity["target_and_unknown_fields_equal_across_views"]
        ),
        "new_identity_range_exact": (
            identity["recovered_sample_seed_equal_across_views"]
            and identity["recovered_sample_seed_unique"]
            and identity["recovered_sample_seed_first"] == EXPECTED_SAMPLE_SEED_FIRST
            and identity["recovered_sample_seed_last"] == EXPECTED_SAMPLE_SEED_LAST
            and all(
                int(signature["first"]) == EXPECTED_SAMPLE_SEED_FIRST
                and int(signature["last"]) == EXPECTED_SAMPLE_SEED_LAST
                and int(signature["count"]) == 25200
                for signature in formal.get("train_seed_signatures", [])
            )
        ),
        "historical_identity_collision_free": (
            identity["stage0_generator_collision_count"] == 0
            and identity["stage0_json_range_collision_count"] == 0
        ),
        "sawfm_visibility_intervention_exact": all(
            power[name]
            for name in (
                "only_sawfm_target_power_changes_as_preregistered",
                "relative_power_truth_matches_0_and_minus10db",
                "weak_mask_matches_sawfm_compound_only",
                "component_seeds_equal_across_views",
            )
        ),
        "roles_complete_disjoint_and_paired": role_audit["covers_all_identities"] and role_audit["all_pairwise_disjoint"],
        "fb_manifest_exact": (
            manifests["frequency_balanced"]["achieved_sample_count"] == 2400
            and manifests["frequency_balanced"]["achieved_label_counts"] == FB_TARGET.tolist()
        ),
        "label_if10_if20_if50_within_tolerance": label_if_ok,
        "combo_if10_if20_if50_within_tolerance": combo_if_ok,
        "every_manifest_has_positive_and_negative_per_label": all_manifests_trainable,
        "schema_and_finite_checks_pass": bool(
            schema["parameter_masks_equal_across_views"]
            and schema["low_active_parameters_finite"]
            and schema["high_active_parameters_finite"]
            and schema["low_signal_probe"]["passed"]
            and schema["high_signal_probe"]["passed"]
        ),
    }
    checks["stage2_smoke_admitted"] = all(checks.values())
    audit = {
        "status": "completed",
        "experiment": "RouteA_FrequencyReconstructionV2_Stage1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "new_identity_development_only",
        "files": {
            "low_visibility": {"path": str(LOW_PATH), "bytes": LOW_PATH.stat().st_size, "sha256": sha256(LOW_PATH), "layout": inspect_v73_layout(LOW_PATH)},
            "high_visibility": {"path": str(HIGH_PATH), "bytes": HIGH_PATH.stat().st_size, "sha256": sha256(HIGH_PATH), "layout": inspect_v73_layout(HIGH_PATH)},
            "power_sweep_manifest": {"path": str(SWEEP_MANIFEST), "sha256": sha256(SWEEP_MANIFEST)},
            "formal_generation_receipt": {"path": str(FORMAL_RECEIPT), "sha256": sha256(FORMAL_RECEIPT)},
            "stage0_audit": {"path": str(STAGE0_AUDIT), "sha256": sha256(STAGE0_AUDIT)},
        },
        "identity_and_label_contract": identity,
        "power_contract": power,
        "schema_and_finite_checks": schema,
        "roles": role_audit,
        "optimization_manifests": manifests,
        "checks": checks,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    return audit, roles, manifests


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    audit, roles, manifests = build_audit()
    audit_path = OUTPUT_DIR / "data_audit.json"
    write_json(audit_path, audit)
    np.savez_compressed(OUTPUT_DIR / "role_indices.npz", **roles)
    write_json(OUTPUT_DIR / "role_manifest.json", audit["roles"])
    write_json(OUTPUT_DIR / "frequency_balanced_manifest.json", manifests["frequency_balanced"])
    for axis in ("label", "combination"):
        for severity, payload in manifests[axis].items():
            write_json(OUTPUT_DIR / f"{axis}_{severity}_manifest.json", payload)
    admitted = audit["checks"]["stage2_smoke_admitted"]
    receipt = {
        "status": "completed" if admitted else "stopped",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "admit_stage2_smoke" if admitted else "stop_before_modeling",
        "data_audit": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
        "data_audit_sha256": sha256(audit_path),
        "script_sha256": sha256(Path(__file__)),
        "new_data_generated": True,
        "model_trained": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT_DIR / "run_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "decision": receipt["decision"],
        "checks": audit["checks"],
        "role_counts": audit["roles"]["counts"],
        "label_manifests": {k: {"n": v["achieved_sample_count"], "counts": v["achieved_label_counts"], "if": v["actual_imbalance_factor"]} for k, v in manifests["label"].items()},
        "combo_manifests": {k: {"n": v["achieved_sample_count"], "if": v["actual_imbalance_factor"]} for k, v in manifests["combination"].items()},
    }, indent=2, ensure_ascii=False))
    if not admitted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
