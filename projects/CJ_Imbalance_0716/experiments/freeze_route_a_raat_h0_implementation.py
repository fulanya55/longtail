"""Freeze and verify the RAAT-H0/v1.2 implementation before data materialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.raat_local_map import extract_raat_map, parameter_count  # noqa: E402

OUT = ROOT / "outputs/tapc/route_a_raat_h0_implementation_freeze/implementation_manifest_v1_2.json"
FILES = (
    "cjlt/models/raat_local_map.py",
    "experiments/generate_route_a_raat_h0_trainonly_v1.m",
    "experiments/audit_route_a_raat_h0_trainonly_v1.py",
    "experiments/run_route_a_raat_h0_v1.py",
    "tests/test_raat_local_map.py",
    "docs/reports/route_a_raat_stage0_literature_overlap_audit_20260802.md",
    "docs/plans/route_a_raat_h0_preregistration_20260802.md",
    "docs/plans/route_a_raat_h0_implementation_contract_20260802.md",
    "docs/plans/route_a_raat_h0_v1_1_safety_amendment_20260802.md",
    "docs/plans/route_a_raat_h0_v1_2_matching_amendment_20260802.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_audit() -> dict:
    model = (ROOT / "cjlt/models/raat_local_map.py").read_text(encoding="utf8").lower()
    generator = (ROOT / "experiments/generate_route_a_raat_h0_trainonly_v1.m").read_text(encoding="utf8").lower()
    runner = (ROOT / "experiments/run_route_a_raat_h0_v1.py").read_text(encoding="utf8")
    model_banned = ("query", "token", "position", "logit", "parameter_values", "receiver_noise_seed", "scale_hough", "stft", "spectrogram", "prototype")
    generator_banned = ("cfg.params.", "run_closedset_splits(cfg,registry,schema,logical(executegeneration),'test'")
    return {
        "model_has_no_forbidden_metadata_or_mechanism": not any(token in model for token in model_banned),
        "generator_inherits_protocol_without_parameter_override": not any(token in generator for token in generator_banned),
        "generator_is_train_only": "'train','error'" in generator and "test_split_generated',false" in generator,
        "runner_metadata_is_post_prediction_only": runner.index("weak_mask_for_evaluation(args.data_path, labels)") > runner.index("rows.append({\"seed\": int(seed), \"variants\": variants})"),
        "runner_uses_label_only_training_view": "Match positives/negatives without using weak masks or component metadata." in runner,
        "runner_has_no_threshold_or_calibration": "--threshold" not in runner.lower() and "calibration(" not in runner.lower() and "platt" not in runner.lower(),
        "runner_has_no_early_stop": "early_stop" not in runner.lower() and "best_epoch" not in runner.lower(),
    }


def cpu_smoke() -> dict:
    generator = torch.Generator().manual_seed(20260802)
    iq = torch.randn(2, 2, 3000, generator=generator)
    expected = {"MFT": [2, 2, 24, 17], "VFT": [2, 2, 24, 17], "ISRJ": [2, 2, 12, 24]}
    values = {}
    for target, shape in expected.items():
        aligned = extract_raat_map(iq, target)
        repeated = extract_raat_map(iq, target)
        wrong = extract_raat_map(iq, target, reverse_chirp=True)
        values[target] = {
            "shape": list(aligned.shape), "shape_pass": list(aligned.shape) == shape,
            "finite": bool(torch.isfinite(aligned).all()), "deterministic": bool(torch.equal(aligned, repeated)),
            "wrong_reference_l2": float(torch.linalg.vector_norm(aligned - wrong)),
        }
    values["parameter_count"] = parameter_count()
    values["parameter_count_pass"] = parameter_count() == 7425
    values["passed"] = all(item["shape_pass"] and item["finite"] and item["deterministic"] and item["wrong_reference_l2"] > 0 for key, item in values.items() if key in expected) and values["parameter_count_pass"]
    return values


def run() -> dict:
    checks = static_audit(); smoke = cpu_smoke()
    if not all(checks.values()) or not smoke["passed"]:
        raise RuntimeError({"static": checks, "smoke": smoke})
    result = {
        "status": "frozen_implementation_before_data_generation",
        "experiment": "RouteA_RAAT_H0_v1_2",
        "candidate": "Reference-Aligned Ambiguity Topology (RAAT)",
        "source_files": {path: sha256(ROOT / path) for path in FILES},
        "python": torch.__version__, "device": "cpu", "static_audit": checks, "cpu_smoke": smoke,
        "receiver_noise_seed_range": [978677743, 978682782], "data_generated": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_opened": False, "test_v3_used": False,
        "next_required_action": "run_MATLAB_generator_once_with_executeGeneration_true_then_identity_audit",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
