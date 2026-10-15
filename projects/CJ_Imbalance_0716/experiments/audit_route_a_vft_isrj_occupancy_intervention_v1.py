"""Audit the fresh paired ISRJ occupancy intervention data without opening tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402


REGIMES = ("W10_15_R3_4", "W22_30_R3_4", "W10_15_R6_8", "W22_30_R6_8")
EXPECTED = {
    "W10_15_R3_4": ((10e-6, 15e-6), (3, 4)),
    "W22_30_R3_4": ((22e-6, 30e-6), (3, 4)),
    "W10_15_R6_8": ((10e-6, 15e-6), (6, 8)),
    "W22_30_R6_8": ((22e-6, 30e-6), (6, 8)),
}
COUNT = 1260
ISRJ = 5
WIDTH_COLUMN = 53
REPLICA_COLUMN = 54


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def _probe_iq_hash(path: Path, columns: np.ndarray) -> str:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["iq_complex"][:, columns])
    return hashlib.sha256(values.tobytes()).hexdigest()


def _combination_code(labels: np.ndarray) -> np.ndarray:
    return (np.asarray(labels, dtype=np.uint8) * (1 << np.arange(6, dtype=np.uint8))[None, :]).sum(axis=1)


def _history_train_files(data_parent: Path, source_root: Path) -> list[Path]:
    return [
        path for path in sorted(data_parent.glob("复合干扰数据*/**/closedset_6class_train.mat"))
        if source_root.resolve() not in path.resolve().parents
    ]


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 must remain sealed")
    paths = {name: args.data_root / name / "closedset_6class_train.mat" for name in REGIMES}
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(f"missing paired intervention data: {missing}")
    labels, params, receivers = {}, {}, {}
    for name, path in paths.items():
        labels[name] = read_labels(path)
        params[name], _ = read_parameters(path)
        receivers[name] = _receiver(path)
    reference = REGIMES[0]
    non_isrj = labels[reference][:, ISRJ] == 0
    probe = np.flatnonzero(non_isrj)[:32]
    if len(probe) != 32:
        raise ValueError("insufficient ISRJ-absent records for fixed IQ control")
    hash_rows = {name: _probe_iq_hash(path, probe) for name, path in paths.items()}
    regime_rows, pair_checks = {}, {}
    for name, path in paths.items():
        present = labels[name][:, ISRJ] == 1
        width = params[name][present, WIDTH_COLUMN]
        replicas = params[name][present, REPLICA_COLUMN]
        bounds_width, bounds_replicas = EXPECTED[name]
        regime_rows[name] = {
            "path": str(path), "sha256": _sha256(path), "sample_count": int(len(labels[name])),
            "receiver_minmax": [int(receivers[name].min()), int(receivers[name].max())],
            "receiver_unique_count": int(np.unique(receivers[name]).size),
            "isrj_present_count": int(present.sum()),
            "width_observed_s": [float(width.min()), float(width.max())],
            "replicas_observed": [int(replicas.min()), int(replicas.max())],
            "width_contract": bool(width.min() >= bounds_width[0] - 1e-10 and width.max() <= bounds_width[1] + 1e-10),
            "replica_contract": bool(replicas.min() >= bounds_replicas[0] and replicas.max() <= bounds_replicas[1]),
            "non_isrj_probe_iq_sha256": hash_rows[name],
        }
        pair_checks[name] = {
            "labels_exact_reference": bool(np.array_equal(labels[name], labels[reference])),
            "receiver_exact_reference": bool(np.array_equal(receivers[name], receivers[reference])),
            "non_isrj_parameter_exact_reference": bool(np.array_equal(params[name][non_isrj], params[reference][non_isrj], equal_nan=True)),
            "non_isrj_iq_probe_exact_reference": hash_rows[name] == hash_rows[reference],
        }
    historic = []
    overlap_free = True
    for prior in _history_train_files(args.data_parent, args.data_root):
        prior_ids = _receiver(prior)
        overlap = int(np.intersect1d(receivers[reference], prior_ids).size)
        historic.append({"path": str(prior), "receiver_overlap_count": overlap})
        overlap_free = overlap_free and overlap == 0
    codes = _combination_code(labels[reference])
    code_counts = {str(int(code)): int((codes == code).sum()) for code in np.unique(codes)}
    checks = {
        "all_regimes_count_1260": all(row["sample_count"] == COUNT for row in regime_rows.values()),
        "all_63_patterns_have_20": len(code_counts) == 63 and set(code_counts.values()) == {20},
        "receiver_unique_per_regime": all(row["receiver_unique_count"] == COUNT for row in regime_rows.values()),
        "four_regimes_identity_and_label_paired": all(all(values.values()) for values in pair_checks.values()),
        "isrj_parameter_contracts": all(row["width_contract"] and row["replica_contract"] for row in regime_rows.values()),
        "all_prior_train_identity_overlap_zero": overlap_free,
        "test_v3_sealed": True,
    }
    payload = {
        "status": "completed", "experiment": "RouteA_VFTISRJ_OccupancyInterventionV1_DataAudit",
        "evidence_role": "fresh_paired_external_physics_diagnostic", "regimes": regime_rows,
        "pair_checks": pair_checks, "combination_counts": code_counts, "historical_train_identity_audit": historic,
        "checks": checks, "passed": bool(all(checks.values())), "test_files_opened": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "checks": checks, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_occupancy_v1"))
    parser.add_argument("--data-parent", type=Path, default=Path(r"C:\LJQ_code\干扰数据"))
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_isrj_occupancy_intervention_v1_generation/identity_parameter_pair_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
