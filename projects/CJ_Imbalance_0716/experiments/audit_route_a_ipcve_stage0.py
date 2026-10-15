"""Preflight and MAT contract audit for IP-CVE absent/low/high triples."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
TARGET_INDICES = (3, 4, 5)
TARGET_PARAMETER_RANGES = {
    3: tuple(range(27, 48)),
    4: tuple(range(48, 52)),
    5: tuple(range(52, 58)),
}
SMOKE_BASE_SEED = 3_990_000_000
DEV_BASE_SEED = 4_000_000_000
CONFIRMATION_BASE_SEED = 4_050_000_000
RECEIVER_OFFSET = 198_677_742
SMOKE_TRIPLES = 15
FORMAL_TRIPLES = 2_400


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


def _range(base: int, count: int) -> tuple[int, int]:
    first = int(base + RECEIVER_OFFSET)
    return first, first + int(count) - 1


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _json_receiver_ranges(root: Path) -> list[dict]:
    rows = []
    for path in (root / "outputs").rglob("*.json"):
        if "outputs\\route_a_ipcve" in str(path).lower() or "outputs/route_a_ipcve" in path.as_posix().lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        def walk(value, key_path=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    current = "{}.{}".format(key_path, key) if key_path else key
                    if (
                        "receiver" in key.lower()
                        and "range" in key.lower()
                        and isinstance(child, list)
                        and len(child) == 2
                        and all(isinstance(item, (int, float)) for item in child)
                    ):
                        rows.append(
                            {
                                "range": [int(child[0]), int(child[1])],
                                "path": str(path),
                                "key": current,
                            }
                        )
                    walk(child, current)
            elif isinstance(value, list):
                for child in value:
                    walk(child, key_path)

        walk(payload)
    return rows


def _read_receiver(path: Path) -> np.ndarray | None:
    try:
        with h5py.File(path, "r") as handle:
            if "receiver_noise_seed" not in handle:
                return None
            return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.uint64)
    except OSError:
        payload = loadmat(
            path,
            variable_names=["receiver_noise_seed"],
            squeeze_me=True,
        )
        if "receiver_noise_seed" not in payload:
            return None
        return np.asarray(payload["receiver_noise_seed"]).reshape(-1).astype(np.uint64)


def _live_train_sources(data_root: Path) -> tuple[list[dict], list[dict]]:
    rows, errors = [], []
    for path in data_root.rglob("*.mat"):
        name = path.name.lower()
        if not any(token in name for token in ("train", "pair", "source")):
            continue
        if any(token in name for token in ("test", "locked", "confirm", "external")):
            continue
        if "route_a_ipcve" in path.as_posix().lower():
            continue
        try:
            receiver = _read_receiver(path)
            if receiver is None or receiver.size == 0:
                continue
            rows.append(
                {
                    "path": str(path),
                    "range": [int(receiver.min()), int(receiver.max())],
                    "unique_count": int(np.unique(receiver).size),
                }
            )
        except Exception as error:
            errors.append(
                {
                    "path": str(path),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    return rows, errors


def build_preflight(project_root: Path, data_root: Path) -> dict:
    recorded = _json_receiver_ranges(project_root)
    live, errors = _live_train_sources(data_root)
    proposed = {
        "smoke": _range(SMOKE_BASE_SEED, SMOKE_TRIPLES),
        "development": _range(DEV_BASE_SEED, FORMAL_TRIPLES),
        "confirmation_reserved": _range(CONFIRMATION_BASE_SEED, FORMAL_TRIPLES),
    }
    historical = [tuple(row["range"]) for row in recorded + live]
    checks = {
        name: not any(_overlap(value, old) for old in historical)
        for name, value in proposed.items()
    }
    checks["proposed_ranges_mutually_disjoint"] = not any(
        _overlap(proposed[left], proposed[right])
        for index, left in enumerate(proposed)
        for right in list(proposed)[index + 1 :]
    )
    checks["confirmation_uint32_safe"] = proposed["confirmation_reserved"][1] < 2**32
    checks["all_live_candidate_sources_readable"] = not errors
    passed = all(checks.values())
    return {
        "status": "completed_stage0_preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "identity_preflight_no_model_evidence",
        "recorded_range_count": len(recorded),
        "live_train_source_count": len(live),
        "live_read_errors": errors,
        "max_recorded_receiver_seed": max(row["range"][1] for row in recorded),
        "max_live_receiver_seed": max(row["range"][1] for row in live),
        "proposed_ranges": {key: list(value) for key, value in proposed.items()},
        "checks": checks,
        "passed": passed,
        "decision": "permit_dry_run_and_smoke" if passed else "stop_before_generation",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def _complex_values(dataset) -> np.ndarray:
    raw = np.asarray(dataset)
    if raw.dtype.fields and {"real", "imag"}.issubset(raw.dtype.fields):
        return raw["real"].astype(np.float64) + 1j * raw["imag"].astype(np.float64)
    if np.iscomplexobj(raw):
        return raw.astype(np.complex128)
    raise TypeError("iq_complex is not a MATLAB complex dataset")


def _equal_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def audit_triple_file(path: Path, expected_triples: int, formal: bool) -> dict:
    with h5py.File(path, "r") as handle:
        required = {
            "iq_complex",
            "label_selected",
            "parameter_values",
            "parameter_mask",
            "receiver_noise_seed",
            "sample_seed",
            "triple_id",
            "triple_target_index",
            "triple_level",
            "nuisance_index",
            "role_code",
            "target_offset_dB",
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError("missing fields: {}".format(missing))
        iq = _complex_values(handle["iq_complex"])
        labels = np.asarray(handle["label_selected"], dtype=np.uint8)
        values = np.asarray(handle["parameter_values"], dtype=np.float64)
        masks = np.asarray(handle["parameter_mask"], dtype=np.uint8).astype(bool)
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.uint64)
        sample = np.asarray(handle["sample_seed"]).reshape(-1).astype(np.uint64)
        triple = np.asarray(handle["triple_id"]).reshape(-1).astype(np.int64)
        target = np.asarray(handle["triple_target_index"]).reshape(-1).astype(np.int64) - 1
        level = np.asarray(handle["triple_level"]).reshape(-1).astype(np.int64)
        nuisance = np.asarray(handle["nuisance_index"]).reshape(-1).astype(np.int64)
        role = np.asarray(handle["role_code"]).reshape(-1).astype(np.int64)
        offsets = np.asarray(handle["target_offset_dB"]).reshape(-1).astype(np.float64)

    view_count = 3 * expected_triples
    shape_checks = {
        "iq": iq.shape == (view_count, 3000),
        "labels": labels.shape == (view_count, 6),
        "parameter_values": values.shape == (view_count, 79),
        "parameter_mask": masks.shape == (view_count, 79),
        "metadata": all(
            array.shape == (view_count,)
            for array in (receiver, sample, triple, target, level, nuisance, role, offsets)
        ),
    }
    identity_ok = label_ok = non_target_ok = target_parameter_ok = True
    max_scale_error = 0.0
    scale = float(10.0 ** (12.0 / 20.0))
    for triple_id in range(1, expected_triples + 1):
        indices = np.flatnonzero(triple == triple_id)
        if len(indices) != 3:
            identity_ok = False
            continue
        indices = indices[np.argsort(level[indices])]
        absent, low, high = indices.tolist()
        label_index = int(target[absent])
        if not (
            np.all(receiver[indices] == receiver[absent])
            and np.all(sample[indices] == sample[absent])
            and np.array_equal(level[indices], np.array([0, 1, 2]))
            and np.all(target[indices] == label_index)
            and np.all(nuisance[indices] == nuisance[absent])
            and np.all(role[indices] == role[absent])
        ):
            identity_ok = False
        expected_low = labels[absent].copy()
        expected_low[label_index] = 1
        if not (
            labels[absent, label_index] == 0
            and np.array_equal(labels[low], expected_low)
            and np.array_equal(labels[high], labels[low])
            and np.isnan(offsets[absent])
            and offsets[low] == -12
            and offsets[high] == 0
        ):
            label_ok = False
        target_columns = set(TARGET_PARAMETER_RANGES[label_index])
        other_columns = np.array(
            [column for column in range(values.shape[1]) if column not in target_columns]
        )
        if not (
            np.array_equal(masks[absent, other_columns], masks[low, other_columns])
            and np.array_equal(masks[low, other_columns], masks[high, other_columns])
            and _equal_nan(values[absent, other_columns], values[low, other_columns])
            and _equal_nan(values[low, other_columns], values[high, other_columns])
        ):
            non_target_ok = False
        jnr_column = int(get_parameter_specs(CLASS_NAMES[label_index])[0].column_index)
        non_jnr_target = np.array(
            [column for column in target_columns if column != jnr_column],
            dtype=np.int64,
        )
        if non_jnr_target.size and not (
            np.array_equal(masks[low, non_jnr_target], masks[high, non_jnr_target])
            and _equal_nan(values[low, non_jnr_target], values[high, non_jnr_target])
        ):
            target_parameter_ok = False
        delta_low = iq[low] - iq[absent]
        delta_high = iq[high] - iq[absent]
        denominator = max(float(np.linalg.norm(delta_high)), 1e-12)
        error = float(np.linalg.norm(delta_high - scale * delta_low) / denominator)
        max_scale_error = max(max_scale_error, error)

    target_counts = {
        CLASS_NAMES[index]: int(np.unique(triple[target == index]).size)
        for index in TARGET_INDICES
    }
    role_counts = {
        "fit": int(np.unique(triple[role == 0]).size),
        "early": int(np.unique(triple[role == 1]).size),
        "evaluation": int(np.unique(triple[role == 2]).size),
    }
    expected_role_counts = (
        {"fit": 1440, "early": 480, "evaluation": 480}
        if formal
        else role_counts
    )
    nuisance_counts_ok = True
    if formal:
        for label_index in TARGET_INDICES:
            for nuisance_index in range(1, 6):
                selected = (target == label_index) & (nuisance == nuisance_index)
                if np.unique(triple[selected]).size != 160:
                    nuisance_counts_ok = False
    checks = {
        **shape_checks,
        "identity_contract": identity_ok,
        "label_and_level_contract": label_ok,
        "non_target_parameter_contract": non_target_ok,
        "target_non_jnr_parameter_contract": target_parameter_ok,
        "waveform_scale_contract": max_scale_error < 1e-9,
        "target_counts": target_counts
        == {"MFT": expected_triples // 3, "VFT": expected_triples // 3, "ISRJ": expected_triples // 3},
        "role_counts": role_counts == expected_role_counts,
        "nuisance_counts": nuisance_counts_ok,
        "unique_receiver_per_triple": int(np.unique(receiver).size) == expected_triples,
    }
    return {
        "status": "completed_triple_contract_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(path),
        "data_sha256": _sha256(path),
        "expected_triples": expected_triples,
        "view_count": view_count,
        "receiver_noise_seed_range": [int(receiver.min()), int(receiver.max())],
        "target_triple_counts": target_counts,
        "role_triple_counts": role_counts,
        "max_waveform_scale_relative_error": max_scale_error,
        "checks": checks,
        "passed": all(checks.values()),
        "evidence_role": "smoke_contract_only" if not formal else "formal_stage0_data_contract",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }


def _write_role_triples(data_path: Path, output_path: Path) -> dict:
    with h5py.File(data_path, "r") as handle:
        triple = np.asarray(handle["triple_id"]).reshape(-1).astype(np.int64)
        role = np.asarray(handle["role_code"]).reshape(-1).astype(np.int64)
    unique_triples = np.unique(triple)
    role_by_triple = np.array(
        [int(role[np.flatnonzero(triple == value)[0]]) for value in unique_triples],
        dtype=np.int8,
    )
    payload = {
        "fit": unique_triples[role_by_triple == 0],
        "early": unique_triples[role_by_triple == 1],
        "evaluation": unique_triples[role_by_triple == 2],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "counts": {key: int(len(value)) for key, value in payload.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("E:/LJQ_code") / "干扰数据",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--expected-triples", type=int)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/route_a_ipcve/stage0/audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        result = build_preflight(args.project_root, args.data_root)
    else:
        if args.data_path is None or args.expected_triples is None:
            raise ValueError("--data-path and --expected-triples are required")
        result = audit_triple_file(
            args.data_path,
            expected_triples=args.expected_triples,
            formal=args.formal,
        )
        if args.formal and result["passed"]:
            result["role_manifest"] = _write_role_triples(
                args.data_path,
                args.output.parent / "role_triples.npz",
            )
    _write_json(args.output, result)
    receipt_path = args.output.with_name(args.output.stem + "_receipt.json")
    _write_json(
        receipt_path,
        {
            "status": "completed",
            "passed": result["passed"],
            "result_path": str(args.output),
            "result_sha256": _sha256(args.output),
            "locked_test_used": False,
            "test_v2_used": False,
            "test_v3_used": False,
        },
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "result": str(args.output),
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
