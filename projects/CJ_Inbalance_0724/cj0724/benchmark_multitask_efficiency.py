"""Measure parameters, approximate MACs, CUDA memory, and inference latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cj0724.multitask import build_multitask_model


def count_macs(model: nn.Module, sample: torch.Tensor) -> int:
    total = 0
    hooks = []

    def convolution_hook(module, inputs, output):
        nonlocal total
        output_tensor = output[0] if isinstance(output, tuple) else output
        kernel = int(np.prod(module.kernel_size))
        per_output = kernel * (module.in_channels // module.groups)
        total += int(output_tensor.numel() * per_output)

    def linear_hook(module, inputs, output):
        nonlocal total
        total += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            hooks.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(sample)
    for hook in hooks:
        hook.remove()
    return total


def latency_ms(model: nn.Module, sample: torch.Tensor, warmup: int, repeats: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(sample)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return float(np.median(values)), float(np.quantile(values, 0.95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper efficiency benchmark")
    device = torch.device("cuda")
    rows = []
    for model_name in args.models:
        torch.manual_seed(2041)
        model = build_multitask_model(model_name, width=24, dropout=0.2, head_dim=256, parameter_hidden_dim=64).to(device).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        macs = count_macs(model, torch.randn(1, 2, 3000, device=device))
        for batch_size in args.batch_sizes:
            sample = torch.randn(batch_size, 2, 3000, device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                model(sample)
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated(device))
            median, p95 = latency_ms(model, sample, args.warmup, args.repeats)
            rows.append({
                "model": model_name, "batch_size": batch_size,
                "parameters": parameters, "trainable_parameters": trainable,
                "macs_per_sample": macs, "approx_flops_per_sample": 2 * macs,
                "peak_cuda_memory_bytes": peak,
                "latency_batch_ms_median": median, "latency_batch_ms_p95": p95,
                "latency_per_sample_ms_median": median / batch_size,
                "throughput_samples_per_second": 1000.0 * batch_size / median,
            })
        del model
        torch.cuda.empty_cache()
    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "dtype": "float32", "amp": False,
            "mac_definition": "Conv/Linear multiply-accumulate; FLOPs approximated as 2x MACs",
            "locked_test_used": False,
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
