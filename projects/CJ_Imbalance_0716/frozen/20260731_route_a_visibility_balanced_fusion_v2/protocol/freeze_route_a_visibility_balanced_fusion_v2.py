"""Freeze the promoted development-v2 visibility-balanced candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"source": str(source.resolve()), "frozen": str(destination.resolve()),
            "sha256": _sha256(destination), "bytes": destination.stat().st_size}


def run(args):
    summary_path = args.candidate_root / "route_a_visibility_balanced_fusion_v2_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("promoted") or not all(summary.get("promotion_gates", {}).values()):
        raise ValueError("candidate did not pass all development-v2 gates")
    if summary.get("test_v2_used") or summary.get("locked_test_used"):
        raise ValueError("candidate development summary crossed a frozen evidence boundary")
    if args.bundle_root.exists():
        raise FileExistsError("freeze bundle already exists: {}".format(args.bundle_root))
    args.bundle_root.mkdir(parents=True)
    records = []
    records.append(_copy(summary_path, args.bundle_root / "development_summary.json"))
    shared = [
        PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v2.yaml",
        PROJECT_ROOT / "experiments" / "run_route_a_visibility_balanced_fusion_v2.py",
        PROJECT_ROOT / "experiments" / "run_route_a_visibility_temporal_expert_v2.py",
        PROJECT_ROOT / "experiments" / "run_route_a_stage23_identifiability_v2.py",
        PROJECT_ROOT / "experiments" / "run_route_a_visibility_balanced_fusion_test_v2_once.py",
        PROJECT_ROOT / "experiments" / "freeze_route_a_visibility_balanced_fusion_v2.py",
        PROJECT_ROOT / "docs" / "plans" / "route_a_visibility_balanced_fusion_v2_plan_20260731.md",
        PROJECT_ROOT / "docs" / "plans" / "route_a_visibility_balanced_fusion_test_v2_once_plan_20260731.md",
    ]
    for source in shared:
        records.append(_copy(source, args.bundle_root / "protocol" / source.name))
    for seed in args.seeds:
        seed_root = args.bundle_root / f"seed_{seed}"
        sources = {
            "backbone/best_model.pth": args.screen_root / "runs" / "F2_DFMS_Full" / f"seed_{seed}" / "best_model.pth",
            "base/best_head.pth": args.h5_root / f"seed_{seed}" / "C6_PosBCE" / "best_head.pth",
            "expert/best_probe.pth": args.stage23_root / f"seed_{seed}" / "T_FinalTemporal" / "best_probe.pth",
            "fusion/best_fusion.pth": args.candidate_root / f"seed_{seed}" / "best_fusion.pth",
            "fusion/history.csv": args.candidate_root / f"seed_{seed}" / "history.csv",
            "fusion/score_normalization.json": args.candidate_root / f"seed_{seed}" / "score_normalization.json",
            "baseline/calibration.json": args.candidate_root / f"seed_{seed}" / "B0_FrozenC6" / "calibration.json",
            "baseline/thresholds.json": args.candidate_root / f"seed_{seed}" / "B0_FrozenC6" / "thresholds.json",
            "candidate/calibration.json": args.candidate_root / f"seed_{seed}" / "E1_VTSE_Bounded" / "calibration.json",
            "candidate/thresholds.json": args.candidate_root / f"seed_{seed}" / "E1_VTSE_Bounded" / "thresholds.json",
            "run_context.json": args.candidate_root / f"seed_{seed}" / "run_context.json",
        }
        for relative, source in sources.items():
            records.append(_copy(source, seed_root / relative))
    manifest = {"status": "frozen", "candidate": "RouteA_VisibilityBalancedFusion_v2",
                "seeds": args.seeds, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "development_v2_sha256": summary["data_sha256"],
                "test_v2": {"path": str(args.test_v2.resolve()),
                            "expected_sha256": args.test_v2_sha256,
                            "authorized_once": True, "used": False},
                "historic_locked_test_used": False, "files": records}
    _write(args.bundle_root / "bundle_manifest.json", manifest)
    _write(args.bundle_root / "test_v2_authorization.json", {
        "authorized": True, "one_shot": True, "rerun_allowed": False,
        "test_v2_path": str(args.test_v2.resolve()),
        "test_v2_sha256": args.test_v2_sha256,
        "started_guard": "test_v2_execution_started.json",
        "completed_receipt": "test_v2_execution_receipt.json",
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2)); return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--h5-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_h5_background_ablation_v2")
    parser.add_argument("--stage23-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stage23_identifiability_v2")
    parser.add_argument("--bundle-root", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--test-v2", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v2\closedset_6class_test.mat"))
    parser.add_argument("--test-v2-sha256", default="902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
