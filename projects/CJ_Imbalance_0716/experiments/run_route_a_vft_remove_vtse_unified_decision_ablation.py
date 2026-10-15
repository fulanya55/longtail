"""Current development-v3 frozen-decision VTSE removal ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT = 4
NON_VFT = (0, 1, 2, 3, 5)
ARMS = ("Q2_NoVTSE", "Q3_BaseOnlyHead", "Q4_AlignedVTSE")
SOURCE = {"Q2_NoVTSE": "Q2_H1B2_MFTGuard", "Q3_BaseOnlyHead": "Q3_VFTBaseOnlyCapacity", "Q4_AlignedVTSE": "Q4_VFTAlignedRidgeFusion"}


def _metric(item: dict, name: str) -> float:
    return float(item["metrics"]["per_label"]["VFT"][name])


def _run_seed(args, config, weak6, seed, data_sha):
    root = args.output_root / f"seed_{seed}"; receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    source_path = args.construction_root / f"seed_{seed}" / "all_role_logits.npz"
    stored = np.load(source_path)
    logits = {arm: {role: np.asarray(stored[f"{SOURCE[arm]}__{role}"], dtype=np.float32) for role in ALL_ROLES} for arm in ARMS}
    reference = logits["Q2_NoVTSE"]
    if any(np.max(np.abs(logits[arm][role][:, NON_VFT] - reference[role][:, NON_VFT])) != 0.0 for arm in ARMS for role in ALL_ROLES):
        raise AssertionError("a stored VTSE arm changes a non-VFT logit")
    variants = [_evaluate(arm, logits[arm], roles, weak6, root / arm, args) for arm in ARMS]
    payload = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path), "source_logits_sha256": _sha256(source_path),
               "non_vft_logits_exact": True, "variants": variants, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(receipt, payload); return payload


def _variant_map(row: dict) -> dict:
    return {item["variant"]: item for item in row["variants"]}


def _comparison(rows: list[dict], baseline: str, seed_base: int) -> dict:
    result = {}
    for offset, metric in enumerate(("weak_vs_negative_auroc", "weak_recall", "aupr", "strong_recall", "negative_fpr")):
        values = [_metric(_variant_map(row)["Q4_AlignedVTSE"], metric) - _metric(_variant_map(row)[baseline], metric) for row in rows]
        result[metric] = {"summary": _mean_ci(values, seed_base + offset), "positive_seeds": int(sum(value > 0 for value in values)), "values": [float(value) for value in values]}
    return result


def run(args):
    config = load_config(args.config); _validate_config(config)
    path = _data_path(config); data_sha = _sha256(path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")): raise ValueError("test-v3 boundary is not sealed")
    labels = read_labels(path); values, _ = read_parameters(path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak, seed, data_sha) for seed in seeds]
    if args.smoke:
        comparisons, checks = {}, {}
    else:
        comparisons = {"q4_minus_q2": _comparison(rows, "Q2_NoVTSE", 2026083100), "q4_minus_q3": _comparison(rows, "Q3_BaseOnlyHead", 2026083200)}
        q4_fpr = float(np.mean([_metric(_variant_map(row)["Q4_AlignedVTSE"], "negative_fpr") for row in rows]))
        checks = {"q4_vs_q2_weak_auroc_ge_0p010": comparisons["q4_minus_q2"]["weak_vs_negative_auroc"]["summary"]["mean"] >= .010,
                  "q4_vs_q3_weak_auroc_ge_0p005": comparisons["q4_minus_q3"]["weak_vs_negative_auroc"]["summary"]["mean"] >= .005,
                  "q4_vs_q2_four_positive": comparisons["q4_minus_q2"]["weak_vs_negative_auroc"]["positive_seeds"] >= 4,
                  "q4_vs_q3_four_positive": comparisons["q4_minus_q3"]["weak_vs_negative_auroc"]["positive_seeds"] >= 4,
                  "q4_vft_aupr_noninferior_q2": comparisons["q4_minus_q2"]["aupr"]["summary"]["mean"] >= 0.0,
                  "q4_vft_fpr_le_0p20": q4_fpr <= .20,
                  "non_vft_logits_exact": all(row["non_vft_logits_exact"] for row in rows)}
    payload = {"status": "completed", "experiment": "RouteA_VFTRemoveVTSEUnifiedDecisionAblation", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "comparisons": comparisons, "mechanism_gate": {"checks": checks, "passed": bool(checks and all(checks.values()))},
               "scope": "current_frozen_decision_ablation_no_training_or_candidate_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_remove_vtse_unified_decision_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_remove_vtse_unified_decision_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--max-negative-fpr", type=float, default=.2); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
