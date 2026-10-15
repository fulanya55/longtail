"""Benchmark frozen always-on, sequential-lazy and parallel-lazy P2c scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.integrate_swept_experts_with_six_class import _load_main, _load_unet
from cj0724.p1k_joint_level3_system import lazy_candidate_masks
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.shared_physics_frontend import shared_feature_batch
from cj0724.vft_doppler_aligned_expert import estimate_aligned_doppler
from cj0724.vft_separator_frequency_head import instantaneous_frequency_features


REGIMES = ("ID_control", "low_count", "close_spacing", "ultra_close", "weak_target")
POWERS = (25, 35, 50)


def load_real_batches(project: Path, batch_size: int, samples_per_cell: int) -> list[torch.Tensor]:
    parts = []
    chosen = np.linspace(0, 629, samples_per_cell, dtype=np.int64)
    for regime in REGIMES:
        for power in POWERS:
            level = {25: 4, 35: 6, 50: 9}[power]
            path = project / f"generated_data/p3d_mft_ordinal_dev_seed470000000_{regime}_rechunked/level_{level:02d}__jnr_{power}dB/closedset_6class_train.mat"
            dataset = V73IQDataset(path, chosen, [28])
            try:
                parts.extend(batch["iq"] for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False))
            finally:
                dataset.close()
    if batch_size == 64:
        return parts
    return [part[offset:offset + 1] for part in parts for offset in range(len(part))]


def expert_outputs(unet, saw, iq: torch.Tensor) -> tuple[torch.Tensor, ...]:
    components = unet.forward_components(iq, None)
    separated = components["normalized_estimate"].float()
    physics = shared_feature_batch(separated)["isrj_periodicity"]
    trajectory = instantaneous_frequency_features(separated)["if_trajectory"]
    vft = estimate_aligned_doppler(iq)["prediction_hz"]
    saw_prediction = saw(iq)["prediction"]
    return components["logits"], physics, trajectory, vft, saw_prediction


def branch_outputs(unet, saw, iq: torch.Tensor, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor], parallel: bool) -> tuple[torch.Tensor, ...]:
    isrj, saw_keep, vft_keep = masks
    empty = iq.new_zeros((0,))
    if not parallel:
        if bool(isrj.any()):
            components = unet.forward_components(iq[isrj], None)
            separated = components["normalized_estimate"].float()
            unet_value = components["logits"].sum() + shared_feature_batch(separated)["isrj_periodicity"].sum() + instantaneous_frequency_features(separated)["if_trajectory"].sum()
        else:
            unet_value = empty.sum()
        saw_value = saw(iq[saw_keep])["prediction"].sum() if bool(saw_keep.any()) else empty.sum()
        vft_value = estimate_aligned_doppler(iq[vft_keep])["prediction_hz"].sum() if bool(vft_keep.any()) else empty.sum()
        return unet_value, saw_value, vft_value
    current = torch.cuda.current_stream()
    streams = [torch.cuda.Stream() for _ in range(3)]
    for stream in streams:
        stream.wait_stream(current)
    with torch.cuda.stream(streams[0]):
        if bool(isrj.any()):
            components = unet.forward_components(iq[isrj], None)
            separated = components["normalized_estimate"].float()
            unet_value = components["logits"].sum() + shared_feature_batch(separated)["isrj_periodicity"].sum() + instantaneous_frequency_features(separated)["if_trajectory"].sum()
        else:
            unet_value = empty.sum()
    with torch.cuda.stream(streams[1]):
        saw_value = saw(iq[saw_keep])["prediction"].sum() if bool(saw_keep.any()) else empty.sum()
    with torch.cuda.stream(streams[2]):
        vft_value = estimate_aligned_doppler(iq[vft_keep])["prediction_hz"].sum() if bool(vft_keep.any()) else empty.sum()
    for stream in streams:
        current.wait_stream(stream)
    return unet_value, saw_value, vft_value


def run_variant(name: str, main, unet, saw, iq: torch.Tensor, thresholds: torch.Tensor):
    logits, parameter = main(iq)
    if name == "main_only":
        return logits.sum() + parameter.sum(), (torch.zeros(len(iq), dtype=torch.bool, device=iq.device),) * 3
    probabilities = torch.sigmoid(logits)
    masks_np = lazy_candidate_masks(probabilities.detach().cpu().numpy(), thresholds.cpu().numpy())
    masks = tuple(torch.from_numpy(value).to(iq.device) for value in masks_np)
    if name == "always_on":
        values = expert_outputs(unet, saw, iq)
        return logits.sum() + parameter.sum() + sum(value.sum() for value in values), tuple(torch.ones(len(iq), dtype=torch.bool, device=iq.device) for _ in range(3))
    values = branch_outputs(unet, saw, iq, masks, parallel=name == "parallel_lazy")
    return logits.sum() + parameter.sum() + sum(values), masks


def benchmark(name: str, main, unet, saw, batches: list[torch.Tensor], thresholds: torch.Tensor, device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    gpu_batches = [batch.to(device) for batch in batches]
    with torch.inference_mode():
        for _ in range(warmup):
            run_variant(name, main, unet, saw, gpu_batches[0], thresholds)
        torch.cuda.synchronize()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        values, coverage = [], []
        for repeat in range(repeats):
            for batch in gpu_batches:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); _, masks = run_variant(name, main, unet, saw, batch, thresholds); end.record(); torch.cuda.synchronize()
                values.append(float(start.elapsed_time(end)))
                coverage.append([float(mask.float().mean()) for mask in masks])
        sample_count = sum(len(batch) for batch in gpu_batches) * repeats
    return {
        "batch_size": int(len(gpu_batches[0])), "batch_count_per_repeat": int(len(gpu_batches)),
        "latency_batch_ms_median": float(np.median(values)), "latency_batch_ms_p95": float(np.quantile(values, 0.95)), "latency_batch_ms_max": float(np.max(values)),
        "latency_per_sample_ms_median": float(np.median(values) / len(gpu_batches[0])),
        "throughput_samples_per_second": float(1000.0 * sample_count / np.sum(values)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "mean_candidate_coverage": {"isrj": float(np.mean([row[0] for row in coverage])), "sawfm": float(np.mean([row[1] for row in coverage])), "vft": float(np.mean([row[2] for row in coverage]))},
    }


def equivalence_check(unet, saw, iq: torch.Tensor, thresholds: torch.Tensor) -> dict[str, float]:
    with torch.inference_mode():
        logits, _ = main_for_equivalence(iq)
        probabilities = torch.sigmoid(logits)
        masks_np = lazy_candidate_masks(probabilities.cpu().numpy(), thresholds.cpu().numpy())
        masks = tuple(torch.from_numpy(value).to(iq.device) for value in masks_np)
        full = expert_outputs(unet, saw, iq)
        differences = {}
        if bool(masks[0].any()):
            subset = expert_outputs(unet, saw, iq[masks[0]])
            differences["unet_logits"] = float((subset[0] - full[0][masks[0]]).abs().max())
            differences["isrj_physics"] = float((subset[1] - full[1][masks[0]]).abs().max())
            differences["trajectory"] = float((subset[2] - full[2][masks[0]]).abs().max())
        differences["vft"] = float((estimate_aligned_doppler(iq[masks[2]])["prediction_hz"] - full[3][masks[2]]).abs().max()) if bool(masks[2].any()) else 0.0
        differences["sawfm"] = float((saw(iq[masks[1]])["prediction"] - full[4][masks[1]]).abs().max()) if bool(masks[1].any()) else 0.0
    return differences


main_for_equivalence = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3e_parallel_lazy_scheduler_preregister_20260801.json")
    parser.add_argument("--output", default="outputs/p3e_parallel_lazy_scheduler_benchmark_20260801.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    project = Path(__file__).resolve().parents[1]; device = torch.device("cuda")
    run_root, p2c_root = project / args.run_root, project / args.p2c_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    row = next(row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter" and int(row["seed"]) == 2041)
    freeze = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8"))["seed_freezes"][0]
    main_model, _ = _load_main(row, protocol, project, device)
    unet = _load_unet(Path(freeze["unet_checkpoint"]), device)
    saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device).eval(); saw.load_state_dict(torch.load(freeze["saw_checkpoint"], map_location=device, weights_only=True))
    thresholds = torch.tensor(freeze["routed_thresholds"], device=device)
    global main_for_equivalence; main_for_equivalence = main_model
    results = {}
    for batch_size in (1, 64):
        batches = load_real_batches(project, batch_size, 64)
        if batch_size == 1: batches = batches[::24][:40]
        results[str(batch_size)] = {name: benchmark(name, main_model, unet, saw, batches, thresholds, device, args.warmup, args.repeats) for name in ("main_only", "always_on", "sequential_lazy", "parallel_lazy")}
    check_batch = load_real_batches(project, 64, 64)[0].to(device)
    differences = equivalence_check(unet, saw, check_batch, thresholds)
    batch64 = results["64"]
    parallel_ratio = batch64["parallel_lazy"]["latency_batch_ms_median"] / batch64["main_only"]["latency_batch_ms_median"]
    parallel_reduction = 1.0 - batch64["parallel_lazy"]["latency_batch_ms_median"] / batch64["sequential_lazy"]["latency_batch_ms_median"]
    gates = {
        "candidate_subset_outputs_max_abs_difference_at_most_1e_5": bool(max(differences.values(), default=0.0) <= 1.0e-5),
        "batch64_parallel_lazy_to_main_at_most_1p5": bool(parallel_ratio <= 1.5),
        "batch64_parallel_at_least_10pct_faster_than_sequential": bool(parallel_reduction >= 0.10),
        "parallel_peak_memory_not_above_always_on": bool(all(results[str(size)]["parallel_lazy"]["peak_cuda_memory_bytes"] <= results[str(size)]["always_on"]["peak_cuda_memory_bytes"] for size in (1, 64))),
    }
    payload = {
        "experiment": "P3e frozen parallel lazy scheduler benchmark", "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "preregister_sha256": __import__("hashlib").sha256((project / args.preregister).read_bytes()).hexdigest(),
        "results": results, "equivalence_max_abs_differences": differences,
        "batch64_parallel_to_main_latency_multiplier": parallel_ratio, "batch64_parallel_reduction_vs_sequential": parallel_reduction,
        "promotion_gates": gates, "development_passed": bool(all(gates.values())),
        "locked_test_used": False,
    }
    (project / args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
