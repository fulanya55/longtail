"""Benchmark the frozen P1n main and complete physics-assisted pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cj0724.integrate_swept_experts_with_six_class import _load_main, _load_unet
from cj0724.physics_parameter_specialists import build_physics_specialist
from cj0724.shared_physics_frontend import shared_feature_batch
from cj0724.vft_doppler_aligned_expert import estimate_aligned_doppler
from cj0724.vft_separator_frequency_head import instantaneous_frequency_features


class JointPipeline(nn.Module):
    def __init__(self, main, unet, saw):
        super().__init__()
        self.main, self.unet, self.saw = main, unet, saw

    def forward(self, iq):
        logits, parameter = self.main(iq)
        components = self.unet.forward_components(iq, None)
        separated = components["normalized_estimate"].float()
        physics = shared_feature_batch(separated)
        trajectory = instantaneous_frequency_features(separated)["if_trajectory"]
        vft = estimate_aligned_doppler(iq)["prediction_hz"]
        saw = self.saw(iq)["prediction"]
        return logits, parameter, components["logits"], physics["isrj_periodicity"], trajectory, vft, saw


def latency(model, sample, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); model(sample); end.record(); torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return float(np.median(values)), float(np.quantile(values, 0.95))


def profiler_flops(model, sample):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
        model(sample)
        torch.cuda.synchronize()
    return int(sum(event.flops for event in profile.key_averages() if event.flops is not None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p1n-root", default="outputs/p1n_full_blend_joint_20260801")
    parser.add_argument("--output", default="outputs/p1n_full_blend_joint_20260801/efficiency_benchmark.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    project = Path(__file__).resolve().parents[1]
    run_root, p1n_root = project / args.run_root, project / args.p1n_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    row = next(row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed"
               and row.get("checkpoint_objective") == "parameter" and int(row["seed"]) == 2041)
    freeze = json.loads((p1n_root / "method_freeze.json").read_text(encoding="utf-8"))["seed_freezes"][0]
    device = torch.device("cuda")
    main_model, _ = _load_main(row, protocol, project, device)
    unet = _load_unet(Path(freeze["unet_checkpoint"]), device)
    saw = build_physics_specialist("sawfm_period_sequence", width=16).to(device).eval()
    saw.load_state_dict(torch.load(freeze["saw_checkpoint"], map_location=device, weights_only=True))
    joint = JointPipeline(main_model, unet, saw).to(device).eval()
    rows = []
    for name, model in (("main_only", main_model), ("p1n_joint", joint)):
        parameters = sum(value.numel() for value in model.parameters())
        sample_one = torch.randn(1, 2, 3000, device=device)
        supported_flops = profiler_flops(model, sample_one)
        for batch_size in (1, 64):
            sample = torch.randn(batch_size, 2, 3000, device=device)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(): model(sample)
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated(device))
            median, p95 = latency(model, sample, args.warmup, args.repeats)
            rows.append({
                "model": name, "batch_size": batch_size, "parameters": int(parameters),
                "profiler_supported_flops_per_sample": supported_flops,
                "peak_cuda_memory_bytes": peak, "latency_batch_ms_median": median,
                "latency_batch_ms_p95": p95, "latency_per_sample_ms_median": median / batch_size,
                "throughput_samples_per_second": 1000.0 * batch_size / median,
            })
    result = {
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "locked_test_used": False},
        "flops_boundary": "torch profiler counts supported operators; FFT/STFT and custom feature arithmetic may be omitted, so joint FLOPs are a documented lower bound",
        "rows": rows,
    }
    output = project / args.output
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
