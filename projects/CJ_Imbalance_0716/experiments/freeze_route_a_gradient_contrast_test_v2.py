"""Freeze development-v3 baseline/candidate artifacts before test-v2 access."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _read_json, _sha256, _write_json  # noqa: E402


SEEDS = (2039, 2040, 2041, 2042, 2043)
VARIANTS = ("C6_Joint", "GradientContrastFixed1p5")
EXPECTED_TEST_SHA256 = "902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947"


def _artifact(path, role):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError("missing frozen artifact: {}".format(resolved))
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def run(args):
    development_summary = _read_json(args.development_summary)
    gate = development_summary["promotion_gates"]["GradientContrastFixed1p5"]
    if not gate.get("passed"):
        raise ValueError("development-v3 candidate did not pass every gate")
    if development_summary.get("promoted_candidate") != "GradientContrastFixed1p5":
        raise ValueError("unexpected development-v3 promoted candidate")
    if development_summary.get("locked_test_used") or development_summary.get("test_v2_used"):
        raise ValueError("development-v3 evidence boundary was violated")
    artifacts = [
        _artifact(args.runner, "frozen_test_runner"),
        _artifact(args.plan, "test_preregistration"),
        _artifact(args.config, "development_v3_config"),
        _artifact(args.development_summary, "development_v3_summary"),
        _artifact(args.identity_audit, "development_v3_identity_audit"),
    ]
    for seed in SEEDS:
        seed_root = Path(args.development_root) / "seed_{}".format(seed)
        for variant in VARIANTS:
            variant_root = seed_root / variant
            artifacts.extend([
                _artifact(variant_root / "best_model.pth", "{}_seed_{}_model".format(variant, seed)),
                _artifact(variant_root / "calibration.json", "{}_seed_{}_calibration".format(variant, seed)),
                _artifact(variant_root / "thresholds.json", "{}_seed_{}_thresholds".format(variant, seed)),
            ])
            calibration = _read_json(variant_root / "calibration.json")
            thresholds = _read_json(variant_root / "thresholds.json")
            if calibration.get("source_role") != "calibration":
                raise ValueError("calibration role drift for {} seed {}".format(variant, seed))
            if thresholds.get("source_role") != "threshold_selection":
                raise ValueError("threshold role drift for {} seed {}".format(variant, seed))
            if len(thresholds.get("thresholds", [])) != 5:
                raise ValueError("expected five frozen thresholds")
    test_path = Path(args.test_v2_path).resolve()
    if not test_path.is_file():
        raise FileNotFoundError("sealed test-v2 path is missing")
    payload = {
        "status": "frozen",
        "bundle_version": 1,
        "candidate": "GradientContrastFixed1p5",
        "baseline": "C6_Joint",
        "seeds": list(SEEDS),
        "fixed_alpha": 1.5,
        "development_v3_gate_passed": True,
        "test_v2": {
            "path": str(test_path),
            "expected_sha256": EXPECTED_TEST_SHA256,
            "content_opened_during_freeze": False,
        },
        "artifacts": artifacts,
        "locked_test_used": False,
        "test_v2_used": False,
        "fit_or_selection_on_test_v2": False,
        "rerun_or_retuning_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "artifact_count": len(artifacts),
        "candidate": payload["candidate"],
        "test_v2_content_opened": False,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--development-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection" / "route_a_development_v3_label_selective_projection_summary.json")
    parser.add_argument("--identity-audit", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_generation" / "identity_audit.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "docs" / "plans" / "route_a_gradient_contrast_test_v2_once_plan_20260731.md")
    parser.add_argument("--runner", type=Path, default=PROJECT_ROOT / "experiments" / "run_route_a_gradient_contrast_test_v2_once.py")
    parser.add_argument("--test-v2-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0731_v2\closedset_6class_test.mat"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_gradient_contrast_test_v2" / "frozen_bundle.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
