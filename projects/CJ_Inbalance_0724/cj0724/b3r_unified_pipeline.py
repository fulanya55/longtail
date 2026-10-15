"""B3-R unified single-read inference screen with CPU/GPU branch overlap."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from cj0724.b1_physics_sufficiency import metrics_from_probabilities, physics_features_from_normalized_iq
from cj0724.b2r_mft_reliability import bounded_three_state, reliability_features
from cj0724.models import build_benchmark_model
from cj0724.route_b_mechanism_router import baseline_root
from cj0724.shared_physics_frontend import shared_feature_batch
from cj0724.vft_physics_screen import PULSE_SAMPLES, build_vft_filter_bank, feature_matrix


FREQUENCIES = np.arange(-8000.0, 8000.1, 500.0, dtype=np.float32)


def read_and_normalize(path: Path, indices: np.ndarray) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    with h5py.File(path, "r") as handle:
        # MATLAB v7.3 stores samples column-wise. Per-column reads follow the
        # file chunk layout and are substantially faster than h5py fancy indexing.
        raw = np.stack(
            [np.asarray(handle["data_legacy"][:, int(index)], dtype=np.float32) for index in indices]
        ).reshape(-1, 2, 3000)
    read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    rms = np.sqrt(np.mean(np.square(raw).sum(axis=1), axis=1, keepdims=True))
    normalized = np.ascontiguousarray(raw / np.maximum(rms[:, None], 1e-8), dtype=np.float32)
    return normalized, read_seconds, time.perf_counter() - started


@torch.no_grad()
def gpu_e5_probabilities(
    iq: np.ndarray, model: torch.nn.Module, vft_classifier, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    timings = {}
    started = time.perf_counter()
    tensor = torch.from_numpy(iq).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    timings["single_transfer_seconds"] = time.perf_counter() - started
    weights, _ = build_vft_filter_bank(FREQUENCIES, device)
    energy_kernel = torch.ones(1, 1, PULSE_SAMPLES, device=device)
    neural_parts = []
    feature_parts = []
    neural_seconds = 0.0
    vft_seconds = 0.0
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        if device.type == "cuda": torch.cuda.synchronize()
        tick = time.perf_counter()
        logits = model(batch)
        if device.type == "cuda": torch.cuda.synchronize()
        neural_seconds += time.perf_counter() - tick
        neural_parts.append(torch.sigmoid(logits).cpu().numpy())
        tick = time.perf_counter()
        correlation = F.conv1d(batch, weights)
        count = len(FREQUENCIES)
        real, imag = correlation[:, :count], correlation[:, count:]
        local_energy = F.conv1d(batch.square().sum(dim=1, keepdim=True), energy_kernel).clamp_min(1e-8)
        normalized = torch.sqrt(real.square() + imag.square() + 1e-12) / torch.sqrt(
            local_energy * float(PULSE_SAMPLES)
        )
        peaks, positions = normalized.max(dim=2)
        if device.type == "cuda": torch.cuda.synchronize()
        vft_seconds += time.perf_counter() - tick
        feature_parts.append(feature_matrix(peaks.cpu().numpy(), positions.cpu().numpy(), FREQUENCIES))
    timings["neural_compute_seconds"] = neural_seconds
    timings["vft_compute_seconds"] = vft_seconds
    started = time.perf_counter()
    neural = np.concatenate(neural_parts)
    vft_probability = vft_classifier.predict_proba(np.concatenate(feature_parts))[:, 1]
    neural[:, 4] = vft_probability
    timings["vft_classifier_seconds"] = time.perf_counter() - started
    return neural, timings


def cpu_features_chunked(iq: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    output = {"mft_delay": [], "isrj_periodicity": [], "combined": []}
    for start in range(0, len(iq), batch_size):
        batch = physics_features_from_normalized_iq(iq[start:start + batch_size])
        for key in output:
            output[key].append(batch[key])
    return {key: np.concatenate(value) for key, value in output.items()}


@torch.no_grad()
def gpu_all_branches(
    iq: np.ndarray, model: torch.nn.Module, vft_classifier, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    timings = {}
    started = time.perf_counter()
    tensor = torch.from_numpy(iq).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    timings["single_transfer_seconds"] = time.perf_counter() - started
    weights, _ = build_vft_filter_bank(FREQUENCIES, device)
    energy_kernel = torch.ones(1, 1, PULSE_SAMPLES, device=device)
    neural_parts, vft_parts = [], []
    physics_parts = {"mft_delay": [], "isrj_periodicity": [], "combined": []}
    neural_seconds = vft_seconds = physics_seconds = 0.0
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        if device.type == "cuda": torch.cuda.synchronize()
        tick = time.perf_counter()
        neural_parts.append(torch.sigmoid(model(batch)).cpu().numpy())
        if device.type == "cuda": torch.cuda.synchronize()
        neural_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        correlation = F.conv1d(batch, weights)
        count = len(FREQUENCIES)
        real, imag = correlation[:, :count], correlation[:, count:]
        local_energy = F.conv1d(batch.square().sum(dim=1, keepdim=True), energy_kernel).clamp_min(1e-8)
        normalized = torch.sqrt(real.square() + imag.square() + 1e-12) / torch.sqrt(
            local_energy * float(PULSE_SAMPLES))
        peaks, positions = normalized.max(dim=2)
        if device.type == "cuda": torch.cuda.synchronize()
        vft_seconds += time.perf_counter() - tick
        vft_parts.append(feature_matrix(peaks.cpu().numpy(), positions.cpu().numpy(), FREQUENCIES))
        tick = time.perf_counter()
        batch_physics = shared_feature_batch(batch)
        if device.type == "cuda": torch.cuda.synchronize()
        physics_seconds += time.perf_counter() - tick
        for key in physics_parts:
            physics_parts[key].append(batch_physics[key].cpu().numpy())
    neural = np.concatenate(neural_parts)
    neural[:, 4] = vft_classifier.predict_proba(np.concatenate(vft_parts))[:, 1]
    timings.update({"neural_compute_seconds": neural_seconds, "vft_compute_seconds": vft_seconds,
                    "shared_physics_compute_seconds": physics_seconds})
    return neural, {key: np.concatenate(value) for key, value in physics_parts.items()}, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b3r_unified_pipeline_20260725"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--physics-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    seed = 20260724
    baseline = baseline_root(seed)
    route = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
    repair = Path(f"outputs/b2rc_recall_constrained_20260725/seed_{seed}")
    audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
    indices = np.sort(np.load(baseline / "split_indices.npz")["development"].astype(np.int64))
    model = build_benchmark_model("efficientnet_b1_1d", num_classes=6, dropout=0.2, head_dim=256).to(device)
    model.load_state_dict(torch.load(baseline / "best.pt", map_location=device, weights_only=False)["model_state"])
    model.eval()
    frozen = Path("outputs/vft_physics_e5_frozen_candidate_20260725")
    vft_classifier = joblib.load(frozen / f"seed_{seed}/vft_physics_classifier.joblib")
    mft_classifier = joblib.load(route / "mft_physics_classifier.joblib")
    isrj_classifier = joblib.load(route / "isrj_physics_classifier.joblib")
    delay_classifier = joblib.load(repair / "mft_delay_auxiliary.joblib")
    router = joblib.load(repair / "mft_three_state_router.joblib")
    reference = np.load(repair / "reference_stats.npz")
    protocol = json.loads((repair / "protocol.json").read_text(encoding="utf-8"))
    expected_rows = {
        int(row["power_dB"]): row for row in __import__("csv").DictReader(
            Path("outputs/b2rc_recall_constrained_20260725/power_runs.csv").open(encoding="utf-8-sig"))
        if int(row["seed"]) == seed and row["variant"] == "three_state"
    }
    rows = []
    frozen_e5_seconds = float(json.loads(
        Path("outputs/vft_physics_e5_frozen_candidate_20260725/audit.json").read_text()
    )["benchmark"]["e5_total_seconds"])
    for power in (10, 30, 50):
        path = Path(audit["levels"][str(power)]["path"])
        # Standalone E5 reference using exactly one read and transfer.
        total_started = time.perf_counter()
        iq, read_seconds, normalize_seconds = read_and_normalize(path, indices)
        e5_probability, e5_gpu = gpu_e5_probabilities(iq, model, vft_classifier, device, args.batch_size)
        e5_total = time.perf_counter() - total_started

        # Unified B3-R: one read/normalization/transfer; CPU physical features overlap GPU E5/VFT.
        total_started = time.perf_counter()
        iq, unified_read, unified_normalize = read_and_normalize(path, indices)
        with ThreadPoolExecutor(max_workers=1) as executor:
            cpu_started = time.perf_counter()
            future = executor.submit(cpu_features_chunked, iq, args.physics_batch_size)
            unified_e5, unified_gpu = gpu_e5_probabilities(iq, model, vft_classifier, device, args.batch_size)
            features = future.result()
            cpu_seconds = time.perf_counter() - cpu_started
        classifier_started = time.perf_counter()
        mft_probability = mft_classifier.predict_proba(features["combined"])[:, 1]
        isrj_probability = isrj_classifier.predict_proba(features["isrj_periodicity"])[:, 1]
        delay_probability = delay_classifier.predict_proba(features["mft_delay"])[:, 1]
        router_features = reliability_features(
            unified_e5[:, 3], mft_probability, delay_probability, features["combined"],
            reference["median"], reference["scale"],
        )
        raw = router.predict_proba(router_features)
        action = np.zeros((len(raw), 3), dtype=np.float64)
        action[:, router.classes_.astype(int)] = raw
        repaired_mft = bounded_three_state(
            unified_e5[:, 3], mft_probability, action,
            float(protocol["alpha"]), float(protocol["confidence"]),
        )
        final_probability = unified_e5.copy()
        final_probability[:, 3] = repaired_mft
        final_probability[:, 5] = isrj_probability
        classifier_seconds = time.perf_counter() - classifier_started
        unified_total = time.perf_counter() - total_started
        saved = np.load(Path(f"outputs/b2_power_boundary_20260725/seed_{seed}/power_{power}dB.npz"))
        thresholds = saved["thresholds_E5"].astype(np.float64)
        thresholds[5] = saved["thresholds_B2-Hard"][5]
        metrics = metrics_from_probabilities(saved["labels"], final_probability, thresholds, saved["weak_mask"])
        expected = expected_rows[power]
        row = {
            "power_dB": power, "samples": int(len(indices)),
            "standalone_E5_total_seconds": e5_total,
            "unified_total_seconds": unified_total,
            "latency_multiplier_vs_same_run_E5": unified_total / e5_total,
            "latency_multiplier_vs_frozen_E5_audit": unified_total / frozen_e5_seconds,
            "read_seconds": unified_read, "normalize_seconds": unified_normalize,
            "single_transfer_seconds": unified_gpu["single_transfer_seconds"],
            "neural_compute_seconds": unified_gpu["neural_compute_seconds"],
            "vft_compute_seconds": unified_gpu["vft_compute_seconds"],
            "cpu_physics_wall_seconds_overlapped": cpu_seconds,
            "tree_and_router_seconds": classifier_seconds,
            "max_abs_E5_probability_difference_between_runs": float(np.max(np.abs(e5_probability - unified_e5))),
            "macro_aupr": float(metrics["macro_aupr"]),
            "expected_macro_aupr": float(expected["macro_aupr"]),
            "macro_aupr_difference": float(metrics["macro_aupr"] - float(expected["macro_aupr"])),
            "hamming": float(metrics["hamming_loss"]),
            "expected_hamming": float(expected["hamming"]),
            "hamming_difference": float(metrics["hamming_loss"] - float(expected["hamming"])),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "protocol": {"seed": seed, "powers_dB": [10, 30, 50], "single_HDF5_read_per_pipeline": True,
                     "single_normalization": True, "single_CPU_to_GPU_transfer": True,
                     "CPU_GPU_overlap": True, "math_or_threshold_changes": False,
                     "locked_test_used": False},
        "rows": rows,
        "summary": {
            "mean_latency_multiplier_vs_same_run_E5": float(np.mean([row["latency_multiplier_vs_same_run_E5"] for row in rows])),
            "worst_latency_multiplier_vs_same_run_E5": float(max(row["latency_multiplier_vs_same_run_E5"] for row in rows)),
            "mean_latency_multiplier_vs_frozen_E5_audit": float(np.mean([row["latency_multiplier_vs_frozen_E5_audit"] for row in rows])),
            "worst_latency_multiplier_vs_frozen_E5_audit": float(max(row["latency_multiplier_vs_frozen_E5_audit"] for row in rows)),
            "max_abs_macro_aupr_difference": float(max(abs(row["macro_aupr_difference"]) for row in rows)),
            "max_abs_hamming_difference": float(max(abs(row["hamming_difference"]) for row in rows)),
        },
        "gates": {}, "locked_test_used": False,
    }
    result["gates"] = {
        "mean_latency_at_most_1p5x_E5": result["summary"]["mean_latency_multiplier_vs_same_run_E5"] <= 1.5,
        "worst_latency_at_most_1p6x_E5": result["summary"]["worst_latency_multiplier_vs_same_run_E5"] <= 1.6,
        "mean_absolute_latency_at_most_1p5x_frozen_E5": result["summary"]["mean_latency_multiplier_vs_frozen_E5_audit"] <= 1.5,
        "macro_aupr_equivalent_within_0p003": result["summary"]["max_abs_macro_aupr_difference"] <= 0.003,
        "hamming_equivalent_within_0p003": result["summary"]["max_abs_hamming_difference"] <= 0.003,
    }
    result["B3R_screen_passed"] = bool(all(result["gates"].values()))
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
