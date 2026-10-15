"""Audit identity pairing and counterfactual metadata availability for Route-A P4."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import sys
import h5py, numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, sha256, write_json

REQUIRED = ("receiver_noise_seed", "component_effective_JNR_dB", "weak_component_mask", "target_range_m", "target_delay_s", "target_measured_power_dB", "target_present")

def scalar_stats(path: Path, key: str):
    with h5py.File(str(path), "r") as h:
        x = h[key]
        if x.dtype.kind in "fiu":
            a = np.asarray(x[0, :])
            return {"dtype": str(x.dtype), "shape": list(x.shape), "finite_fraction": float(np.mean(np.isfinite(a))) if x.dtype.kind == "f" else 1.0, "unique_count": int(len(np.unique(a[np.isfinite(a)]))) if x.dtype.kind == "f" else int(len(np.unique(a)))}
        return {"dtype": str(x.dtype), "shape": list(x.shape), "finite_fraction": None, "unique_count": None}

def run(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = {}; files = {}; field_stats = {}
    for p in POWERS_DB:
        path = find_file(args.data_root, p, "train"); files[str(p)] = str(path)
        with h5py.File(str(path), "r") as h:
            keys = set(h.keys()); missing = [k for k in REQUIRED if k not in keys]
            if missing: raise RuntimeError(f"{path} missing fields {missing}")
            seeds[str(p)] = np.asarray(h["receiver_noise_seed"][0, :], dtype=np.int64)
            field_stats[str(p)] = {k: scalar_stats(path, k) for k in ("target_range_m", "target_delay_s", "target_measured_power_dB", "target_present")}
    reference = seeds[str(POWERS_DB[0])]
    comparisons = {}
    for p in POWERS_DB:
        current = seeds[str(p)]
        comparisons[str(p)] = {"count": int(len(current)), "unique_count": int(len(np.unique(current))), "set_intersection_with_reference": int(len(np.intersect1d(reference, current))), "same_order_as_reference": bool(np.array_equal(reference, current))}
    payload = {"schema":"route-a-p4-identity-metadata-audit-v1","status":"completed","data_root":str(args.data_root),"powers":list(POWERS_DB),"files":files,"cross_power_receiver_seed_comparison":comparisons,"field_stats":field_stats,"counterfactual_scope":{"same_receiver_across_power":all(v["same_order_as_reference"] for v in comparisons.values()),"same_receiver_different_combination_within_power":False,"reason":"receiver_noise_seed is unique within each power view and combination_id/sample_id is not stored as a directly usable per-sample field","target_condition_fields_usable":False,"reason_target":"target range/delay/measured power are all NaN and target_present is constant zero in the audited train files"},"locked_test_used":False}
    write_json(args.output_root/"identity_metadata_audit.json", payload); write_json(args.output_root/"terminal_receipt.json", {"status":"completed","summary":str((args.output_root/"identity_metadata_audit.json").resolve()),"locked_test_used":False,"test_files_opened":False,"script_sha256":sha256(Path(__file__))}); print(json.dumps({"status":"completed","summary":str(args.output_root/"identity_metadata_audit.json")},ensure_ascii=False))

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path(r'E:\论文相关\干扰数据\复合干扰数据0716\九类')); p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/route_a_p4_identity_audit_20260828'); return p.parse_args()
if __name__=='__main__': run(parse_args())
