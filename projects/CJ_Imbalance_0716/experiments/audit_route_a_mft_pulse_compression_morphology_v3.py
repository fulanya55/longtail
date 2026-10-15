"""Fixed shift-invariant pulse-compression morphology audit for weak MFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_mft_matched_filter_delay_profile_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    ROI_START,
    ROI_STOP,
    VARIANTS,
    build_lfm_reference,
    circular_correlation_profile,
    evaluation_view,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


SCORE_NAMES = (
    "roi_max_over_bg_q99",
    "roi_top3_over_bg_q99",
    "roi_top6_over_bg_q99",
    "nms_top3_over_bg_q99",
    "nms_top6_over_bg_q99",
    "nms_count_gt_bg_q99",
    "roi_energy_fraction",
    "roi_q99_over_bg_q99",
)


def nms_peaks(values: torch.Tensor, count: int = 12, radius: int = 8) -> torch.Tensor:
    work = values.clone()
    peaks = []
    positions = torch.arange(values.shape[1], device=values.device)[None, :]
    for _ in range(int(count)):
        peak, index = work.max(dim=1)
        peaks.append(peak)
        blocked = (positions - index[:, None]).abs() <= int(radius)
        work = work.masked_fill(blocked, -torch.inf)
    return torch.stack(peaks, dim=1)


def morphology_scores(profile: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    roi = profile[:, ROI_START:ROI_STOP]
    background = torch.cat([profile[:, :ROI_START], profile[:, ROI_STOP:]], dim=1)
    bg_q99 = torch.quantile(background, 0.99, dim=1).clamp_min(eps)
    roi_q99 = torch.quantile(roi, 0.99, dim=1)
    sample_top = torch.topk(roi, k=6, dim=1).values
    peaks = nms_peaks(roi, count=12, radius=8)
    full_energy = profile.sum(dim=1).clamp_min(eps)
    return torch.stack(
        [
            roi.amax(dim=1) / bg_q99,
            sample_top[:, :3].mean(dim=1) / bg_q99,
            sample_top.mean(dim=1) / bg_q99,
            peaks[:, :3].mean(dim=1) / bg_q99,
            peaks[:, :6].mean(dim=1) / bg_q99,
            (peaks > bg_q99[:, None]).float().sum(dim=1),
            roi.sum(dim=1) / full_energy,
            roi_q99 / bg_q99,
        ],
        dim=1,
    )


def extract_scores(signals: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    references = {
        "MF_AlignedLFM": build_lfm_reference(device, wrong_chirp=False),
        "MF_WrongChirp": build_lfm_reference(device, wrong_chirp=True),
        "Self_ACF": None,
    }
    rows = {variant: [] for variant in VARIANTS}
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start : start + int(batch_size)]).to(device)
            for variant in VARIANTS:
                profile = circular_correlation_profile(batch, references[variant])
                rows[variant].append(morphology_scores(profile).cpu().numpy().astype(np.float32))
    return {variant: np.concatenate(values) for variant, values in rows.items()}


def metric(target: np.ndarray, score: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(target, score)),
        "aupr": float(average_precision_score(target, score)),
    }


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for seed in args.seeds:
        roles, indices_path = _load_roles(
            args.c6_root, seed, data_path, config["normalization"], args.io_batch_size
        )
        role = roles["early"]
        scores = extract_scores(role["signals"], device, args.feature_batch_size)
        seed_result = {
            "seed": int(seed),
            "role_indices_sha256": _sha256(indices_path),
            "views": {},
        }
        for mode in ("high_order", "order5_negative"):
            selected, target, receipt = evaluation_view(
                role["targets6"], weak[role["sample_ids"]], mode
            )
            seed_result["views"][mode] = {"receipt": receipt, "variants": {}}
            for variant in VARIANTS:
                seed_result["views"][mode]["variants"][variant] = {
                    name: metric(target, scores[variant][selected, index])
                    for index, name in enumerate(SCORE_NAMES)
                }
        rows.append(seed_result)
    aggregate = {}
    for mode_index, mode in enumerate(("high_order", "order5_negative")):
        aggregate[mode] = {}
        for variant_index, variant in enumerate(VARIANTS):
            aggregate[mode][variant] = {}
            for score_index, name in enumerate(SCORE_NAMES):
                aggregate[mode][variant][name] = {
                    measure: _mean_ci(
                        [row["views"][mode]["variants"][variant][name][measure] for row in rows],
                        20404000 + mode_index * 10000 + variant_index * 1000 + score_index * 10 + measure_index,
                    )
                    for measure_index, measure in enumerate(("auroc", "aupr"))
                }
    aligned_rows = []
    for score_index, name in enumerate(SCORE_NAMES):
        aligned = [row["views"]["high_order"]["variants"]["MF_AlignedLFM"][name]["auroc"] for row in rows]
        wrong = [row["views"]["high_order"]["variants"]["MF_WrongChirp"][name]["auroc"] for row in rows]
        delta = [a - b for a, b in zip(aligned, wrong)]
        aligned_rows.append(
            {
                "score": name,
                "aligned_auroc": _mean_ci(aligned, 20406000 + score_index * 10),
                "aligned_positive_seeds_vs_chance": int(sum(value > 0.5 for value in aligned)),
                "aligned_minus_wrong": _mean_ci(delta, 20406001 + score_index * 10),
                "positive_seeds_vs_wrong": int(sum(value > 0 for value in delta)),
            }
        )
    candidates = [
        row for row in aligned_rows
        if row["aligned_auroc"]["mean"] >= 0.60
        and row["aligned_positive_seeds_vs_chance"] >= 4
        and row["aligned_minus_wrong"]["mean"] >= 0.03
        and row["positive_seeds_vs_wrong"] >= 4
    ]
    result = {
        "status": "completed",
        "experiment": "RouteA_MFT_PulseCompressionMorphologyAudit_v3",
        "evidence_role": "development_v3_fixed_mechanism_diagnostic",
        "data_sha256": data_sha,
        "score_names": list(SCORE_NAMES),
        "rows": rows,
        "aggregate": aggregate,
        "aligned_score_audit": aligned_rows,
        "diagnostic_gate": {
            "candidate_scores": [row["score"] for row in candidates],
            "passed": bool(candidates),
        },
        "next_action": "design_low_dimensional_frozen_probe" if candidates else "stop_pulse_compression_morphology",
        "learned_parameters": 0,
        "development_v7_used_for_selection": False,
        "development_v8_used_for_selection": False,
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_mft_pulse_compression_morphology_v3.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "gate": result["diagnostic_gate"], "summary": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_pulse_compression_morphology_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

