"""Preregistered MFT matched-filter delay-profile screen on development-v3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (  # noqa: E402
    _load_roles,
    _rank_metrics,
    _scores,
    _train_probe,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


MFT_INDEX = 3
VARIANTS = ("MF_AlignedLFM", "MF_WrongChirp", "Self_ACF")
FS_HZ = 10_000_000.0
PULSE_WIDTH_S = 100e-6
PRI_S = 300e-6
BANDWIDTH_HZ = 5_000_000.0
ROI_START = 400
ROI_STOP = 1001


def build_lfm_reference(device: torch.device, *, wrong_chirp: bool = False) -> torch.Tensor:
    pulse_samples = int(round(FS_HZ * PULSE_WIDTH_S))
    pri_samples = int(round(FS_HZ * PRI_S))
    slope = BANDWIDTH_HZ / PULSE_WIDTH_S
    if wrong_chirp:
        slope = -slope
    t = torch.arange(1, pulse_samples + 1, device=device, dtype=torch.float32) / FS_HZ
    pulse = torch.exp(1j * torch.pi * slope * t.square())
    reference = torch.zeros(pri_samples, device=device, dtype=torch.complex64)
    reference[:pulse_samples] = pulse.to(torch.complex64)
    return reference


def circular_correlation_profile(
    signal: torch.Tensor, reference: torch.Tensor | None
) -> torch.Tensor:
    z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
    spectrum = torch.fft.fft(z, dim=-1)
    if reference is None:
        product = spectrum * spectrum.conj()
    else:
        product = spectrum * torch.fft.fft(reference, dim=-1).conj()[None, :]
    return torch.fft.ifft(product, dim=-1).abs()


def _pooled(values: torch.Tensor, bins: int = 64) -> torch.Tensor:
    avg = F.adaptive_avg_pool1d(values[:, None], bins).squeeze(1)
    maximum = F.adaptive_max_pool1d(values[:, None], bins).squeeze(1)
    return torch.cat([avg, maximum], dim=1)


def summarize_profile(profile: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    scale = profile.median(dim=1, keepdim=True).values.clamp_min(eps)
    normalized = torch.log1p(profile / scale)
    roi = normalized[:, ROI_START:ROI_STOP]
    top_count = min(16, int(roi.shape[1]))
    top_values, top_indices = torch.topk(roi, k=top_count, dim=1)
    top_values, sort_order = top_values.sort(dim=1, descending=True)
    top_indices = top_indices.gather(1, sort_order).float() / float(max(roi.shape[1] - 1, 1))
    q = torch.tensor([0.25, 0.50, 0.75, 0.90, 0.95, 0.99], device=profile.device)
    full_quantiles = torch.quantile(normalized, q, dim=1).transpose(0, 1)
    roi_quantiles = torch.quantile(roi, q, dim=1).transpose(0, 1)
    full_energy = normalized.sum(dim=1).clamp_min(eps)
    roi_energy = roi.sum(dim=1)
    stats = torch.stack(
        [
            normalized.mean(dim=1),
            normalized.std(dim=1, unbiased=False),
            normalized.amax(dim=1),
            roi.mean(dim=1),
            roi.std(dim=1, unbiased=False),
            roi.amax(dim=1),
            roi_energy / full_energy,
            top_values[:, :4].mean(dim=1) / roi.mean(dim=1).clamp_min(eps),
        ],
        dim=1,
    )
    return torch.cat(
        [
            _pooled(normalized),
            _pooled(roi),
            top_values,
            top_indices,
            full_quantiles,
            roi_quantiles,
            stats,
        ],
        dim=1,
    )


def extract_features(signals: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    references = {
        "MF_AlignedLFM": build_lfm_reference(device, wrong_chirp=False),
        "MF_WrongChirp": build_lfm_reference(device, wrong_chirp=True),
        "Self_ACF": None,
    }
    rows = {name: [] for name in VARIANTS}
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start : start + int(batch_size)]).to(device)
            for name in VARIANTS:
                profile = circular_correlation_profile(batch, references[name])
                rows[name].append(summarize_profile(profile).cpu().numpy().astype(np.float32))
    return {name: np.concatenate(values) for name, values in rows.items()}


def composition_matched_view(targets: np.ndarray, weak: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    targets = np.asarray(targets, dtype=np.uint8)
    other = [index for index in range(targets.shape[1]) if index != MFT_INDEX]
    weak_positive = (targets[:, MFT_INDEX] == 1) & np.asarray(weak[:, MFT_INDEX], dtype=bool)
    negative = targets[:, MFT_INDEX] == 0
    positive_compositions = {tuple(row) for row in targets[weak_positive][:, other].tolist()}
    negative_compositions = {tuple(row) for row in targets[negative][:, other].tolist()}
    shared = positive_compositions & negative_compositions
    positive_selected = np.asarray(
        [weak_positive[i] and tuple(targets[i, other].tolist()) in shared for i in range(len(targets))],
        dtype=bool,
    )
    negative_selected = np.asarray(
        [negative[i] and tuple(targets[i, other].tolist()) in shared for i in range(len(targets))],
        dtype=bool,
    )
    selected = positive_selected | negative_selected
    return selected, positive_selected[selected].astype(np.float32), {
        "weak_positive_count": int(positive_selected.sum()),
        "matched_negative_count": int(negative_selected.sum()),
        "shared_composition_count": int(len(shared)),
    }


def evaluation_view(targets: np.ndarray, weak: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    targets = np.asarray(targets, dtype=np.uint8)
    order = targets.sum(axis=1)
    positive = (targets[:, MFT_INDEX] == 1) & np.asarray(weak[:, MFT_INDEX], dtype=bool)
    negative = targets[:, MFT_INDEX] == 0
    if mode == "high_order":
        positive &= order >= 4
        negative &= order >= 4
    elif mode == "order5_negative":
        positive &= order >= 4
        negative &= order == 5
    elif mode != "all_matched":
        raise ValueError(f"unknown evaluation mode: {mode}")
    if mode == "all_matched":
        return composition_matched_view(targets, weak)
    selected = positive | negative
    return selected, positive[selected].astype(np.float32), {
        "weak_positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
    }


def _write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(args, config, labels6, weak6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    features = {
        role: extract_features(roles[role]["signals"], device, args.feature_batch_size)
        for role in ("fit", "validation", "early")
    }
    results = {}
    receipts = {}
    for variant_index, variant in enumerate(VARIANTS):
        mean = features["fit"][variant].mean(axis=0).astype(np.float32)
        std = np.maximum(features["fit"][variant].std(axis=0), 1e-5).astype(np.float32)
        normalized = {
            role: ((features[role][variant] - mean[None, :]) / std[None, :]).astype(np.float32)
            for role in features
        }
        train_views = {}
        selected, target, receipt = composition_matched_view(
            roles["fit"]["targets6"], weak6[roles["fit"]["sample_ids"]]
        )
        train_views["fit"] = (normalized["fit"][selected], target)
        receipts.setdefault("fit", receipt)
        selected, target, receipt = evaluation_view(
            roles["validation"]["targets6"],
            weak6[roles["validation"]["sample_ids"]],
            "high_order",
        )
        train_views["validation"] = (normalized["validation"][selected], target)
        receipts.setdefault("validation_selection_high_order", receipt)
        probe, history, info = _train_probe(
            *train_views["fit"],
            *train_views["validation"],
            20401000 + int(seed) + variant_index * 100,
            device,
            args,
        )
        metrics = {}
        for role in ("validation", "early"):
            metrics[role] = {}
            for mode in ("all_matched", "high_order", "order5_negative"):
                selected, target, receipt = evaluation_view(
                    roles[role]["targets6"], weak6[roles[role]["sample_ids"]], mode
                )
                score = _scores(probe, normalized[role][selected], device)
                metrics[role][mode] = _rank_metrics(target, score)
                receipts.setdefault(f"{role}_{mode}", receipt)
        out = seed_root / variant
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": probe.state_dict(),
                "mean": mean,
                "std": std,
                "variant": variant,
                "feature_dim": int(mean.shape[0]),
                **info,
            },
            out / "best_probe.pth",
        )
        _write_history(out / "history.csv", history)
        result = {
            "feature_dim": int(mean.shape[0]),
            "metrics": metrics,
            "validation_high_order_minus_early": metrics["validation"]["high_order"]["auroc"]
            - metrics["early"]["high_order"]["auroc"],
            **info,
        }
        results[variant] = result
        _write_json(out / "metrics.json", result)
    payload = {
        "seed": int(seed),
        "status": "completed",
        "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_indices_path),
        "variants": results,
        "view_receipts": receipts,
        "test_v3_used": False,
        "locked_test_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if (
        seal["test_v3"]["access_authorized"]
        or seal["test_v3"]["analysis_allowed"]
        or seal["test_v3"]["used"]
    ):
        raise ValueError("test-v3 is not sealed")
    labels6 = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [run_seed(args, config, labels6, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        aggregate[variant] = {}
        for role in ("validation", "early"):
            aggregate[variant][role] = {}
            for mode_index, mode in enumerate(("all_matched", "high_order", "order5_negative")):
                aggregate[variant][role][mode] = {
                    metric: _mean_ci(
                        [row["variants"][variant]["metrics"][role][mode][metric] for row in rows],
                        20402000 + variant_index * 1000 + (0 if role == "validation" else 500) + mode_index * 10 + metric_index,
                    )
                    for metric_index, metric in enumerate(("auroc", "aupr"))
                }
        aggregate[variant]["validation_high_order_minus_early"] = _mean_ci(
            [row["variants"][variant]["validation_high_order_minus_early"] for row in rows],
            20402900 + variant_index,
        )
    comparisons = {}
    for control_index, control in enumerate(("MF_WrongChirp", "Self_ACF")):
        deltas = [
            row["variants"]["MF_AlignedLFM"]["metrics"]["early"]["high_order"]["auroc"]
            - row["variants"][control]["metrics"]["early"]["high_order"]["auroc"]
            for row in rows
        ]
        comparisons[f"MF_AlignedLFM_minus_{control}"] = {
            "early_high_order_auroc": _mean_ci(deltas, 20403000 + control_index),
            "positive_seeds": int(sum(value > 0 for value in deltas)),
        }
    if args.smoke:
        gate = {"checks": {}, "passed": False, "smoke_only": True}
    else:
        aligned = aggregate["MF_AlignedLFM"]
        wrong = comparisons["MF_AlignedLFM_minus_MF_WrongChirp"]
        self_acf = comparisons["MF_AlignedLFM_minus_Self_ACF"]
        checks = {
            "early_high_order_auroc_ge_0p620": aligned["early"]["high_order"]["auroc"]["mean"] >= 0.620,
            "gain_vs_wrong_ge_0p030": wrong["early_high_order_auroc"]["mean"] >= 0.030,
            "gain_vs_self_acf_ge_0p030": self_acf["early_high_order_auroc"]["mean"] >= 0.030,
            "four_of_five_vs_wrong": wrong["positive_seeds"] >= 4,
            "four_of_five_vs_self_acf": self_acf["positive_seeds"] >= 4,
            "validation_early_gap_lt_0p100": aligned["validation_high_order_minus_early"]["mean"] < 0.100,
            "early_order5_negative_auroc_ge_0p580": aligned["early"]["order5_negative"]["auroc"]["mean"] >= 0.580,
        }
        gate = {"checks": checks, "passed": bool(all(checks.values())), "smoke_only": False}
    payload = {
        "status": "completed",
        "experiment": "RouteA_MFT_MatchedFilterDelayProfile_v3",
        "seeds": list(seeds),
        "data_sha256": data_sha,
        "feature_contract": {
            "fs_Hz": FS_HZ,
            "pulse_width_s": PULSE_WIDTH_S,
            "pri_s": PRI_S,
            "bandwidth_Hz": BANDWIDTH_HZ,
            "delay_roi_samples": [ROI_START, ROI_STOP - 1],
            "variants": list(VARIANTS),
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "promotion_gate": gate,
        "next_action": "freeze_and_generate_development_v9" if gate["passed"] else "stop_matched_filter_mechanism",
        "development_v7_used_for_selection": False,
        "development_v8_used_for_selection": False,
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_mft_matched_filter_delay_profile_v3_summary.json"
    _write_json(output, payload)
    print(json.dumps({"status": "completed", "gate": gate, "summary": str(output)}, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_matched_filter_delay_profile_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
