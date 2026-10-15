"""Provenance audit for H1a paired weak-VFT staggered-PRI MFT+ISRJ data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from cjlt.models.vft_staggered_pri import record_samples
from experiments.find_route_a_h0o_crosspulse_identity_range import historic_receiver_ids


DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
COUNT, BASE_SEED = 640, 2_700_000_000
FIELDS = (
    "pair_seed", "receiver_noise_seed", "vft_present", "mft_component_seed", "isrj_component_seed", "vft_component_seed",
    "mft_num_targets", "mft_delays_samples", "mft_amplitudes", "isrj_width_samples", "isrj_replicas",
    "mft_effective_JNR_dB", "isrj_effective_JNR_dB", "vft_effective_JNR_dB", "vft_doppler_Hz",
    "vft_arrival_delay_samples", "pulse_start_samples",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as handle:
        if not set(FIELDS).issubset(handle.keys()) or not {"iq_real", "iq_imag"}.issubset(handle.keys()):
            raise ValueError(f"missing H1a fields in {path}")
        metadata = {name: np.asarray(handle[name]) for name in FIELDS}
        iq = np.asarray(handle["iq_real"]) + 1j * np.asarray(handle["iq_imag"])
    return metadata, iq


def _h0q_receivers(path: Path) -> set[int]:
    with h5py.File(path, "r") as handle:
        return set(map(int, np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.uint64)))


def audit(args: argparse.Namespace) -> dict[str, object]:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    selection = json.loads(args.identity_selection.read_text(encoding="utf-8"))
    if not receipt.get("executed") or selection.get("identity_overlap_count") != 0:
        raise ValueError("H1a receipt or pre-generation identity selection invalid")
    if any(seal["test_v3"][name] for name in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    absent, absent_iq = _load(args.absent_path); present, present_iq = _load(args.present_path)
    receiver = absent["receiver_noise_seed"].reshape(-1).astype(np.uint64)
    historic, sources = historic_receiver_ids(); h0q = _h0q_receivers(args.h0q_source)
    shared = ("pair_seed", "receiver_noise_seed", "mft_component_seed", "isrj_component_seed", "mft_num_targets", "mft_delays_samples", "mft_amplitudes", "isrj_width_samples", "isrj_replicas", "mft_effective_JNR_dB", "isrj_effective_JNR_dB", "pulse_start_samples")
    checks = {
        "payload_shape_exact": absent_iq.shape == present_iq.shape == (COUNT, record_samples()),
        "pair_seed_exact": np.array_equal(absent["pair_seed"].reshape(-1), np.arange(BASE_SEED, BASE_SEED + COUNT)),
        "receiver_identity_unique": np.unique(receiver).size == COUNT,
        "all_background_counterfactual_metadata_exact": all(np.array_equal(absent[name], present[name], equal_nan=True) for name in shared),
        "absent_vft_fields_nan": bool(np.all(np.isnan(absent["vft_component_seed"])) and np.all(np.isnan(absent["vft_effective_JNR_dB"])) and np.all(np.isnan(absent["vft_doppler_Hz"])) and np.all(np.isnan(absent["vft_arrival_delay_samples"]))),
        "present_vft_fields_in_preregistered_ranges": bool(np.all((present["vft_effective_JNR_dB"] >= 7.999) & (present["vft_effective_JNR_dB"] <= 8.001)) and np.all((present["vft_doppler_Hz"] >= 1000) & (present["vft_doppler_Hz"] <= 7000)) and np.all((present["vft_arrival_delay_samples"] >= 100) & (present["vft_arrival_delay_samples"] <= 500))),
        "present_absent_indicator_exact": bool(np.all(absent["vft_present"] == 0) and np.all(present["vft_present"] == 1)),
        "only_declared_vft_intervention_changes_iq": not np.array_equal(absent_iq, present_iq),
        "historical_train_identity_overlap_zero": not bool(historic.intersection(map(int, receiver))),
        "h0q_identity_overlap_zero": not bool(h0q.intersection(map(int, receiver))),
        "test_v3_still_sealed": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"H1a provenance contract failed: {checks}")
    result = {
        "status": "completed_train_only_paired_audit", "experiment": "RouteA_H1a_VFTStaggeredPRI_MFTISRJCompositeAttribution",
        "source_scope": "project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716",
        "absent_sha256": _sha256(args.absent_path), "present_sha256": _sha256(args.present_path),
        "sample_count_per_arm": COUNT, "record_samples": record_samples(), "contracts": checks,
        "pre_generation_identity_selection": str(args.identity_selection), "historic_train_sources_scanned": sources,
        "identity_audit_passed": True, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    source = DATA_ROOT / "h1a_vft_staggered_pri_mft_isrj_20260803"; root = ROOT / "outputs/tapc/route_a_h1a_vft_staggered_pri_generation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absent-path", type=Path, default=source / "vft_absent_mft_isrj.mat")
    parser.add_argument("--present-path", type=Path, default=source / "vft_present_weak_8db_mft_isrj.mat")
    parser.add_argument("--receipt", type=Path, default=root / "formal_generation_receipt.json")
    parser.add_argument("--identity-selection", type=Path, default=root / "pre_generation_identity_selection.json")
    parser.add_argument("--h0q-source", type=Path, default=DATA_ROOT / "h0q_vft_staggered_pri_20260803/vft_low_1p0_2p5kHz.mat")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=root / "paired_identity_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    value = audit(parse_args())
    print(json.dumps({"status": value["status"], "identity_audit_passed": value["identity_audit_passed"]}, ensure_ascii=False))
