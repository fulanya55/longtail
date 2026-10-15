"""Eager CUDA latency audit for the frozen P6a-r1 C0/C1 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.multitask import build_multitask_model


def _measure(model: torch.nn.Module, batch_size: int, repeats: int, warmup: int) -> dict[str, float]:
    sample = torch.randn(batch_size, 2, 3000, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(sample)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "repeats": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int, default=16041)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P6a latency audit requires CUDA")
    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root) if Path(args.run_root).is_absolute() else project / args.run_root
    model_names = ("mlcnn_iq_grouped_task_routed", "mlcnn_iq_grouped_physical_film_task_routed")
    results = {}
    for model_name in model_names:
        path = run_root / f"{model_name}__cos_lr1e4_wd1e4__seed{args.seed}" / "best_parameter.pt"
        checkpoint = torch.load(path, map_location="cuda", weights_only=False)
        model = build_multitask_model(model_name, 24, 0.2, 256, 64).cuda().eval()
        model.load_state_dict(checkpoint["model_state"])
        results[model_name] = {str(batch): _measure(model, batch, args.repeats, args.warmup) for batch in (1, 64)}
        del model
        torch.cuda.empty_cache()
    baseline, candidate = (results[name] for name in model_names)
    b64_ratio = candidate["64"]["median_ms"] / baseline["64"]["median_ms"]
    b1_ratio = candidate["1"]["p95_ms"] / baseline["1"]["p95_ms"]
    result = {
        "run_root": str(run_root), "seed": args.seed, "mode": "eager_fp32", "results": results,
        "ratios": {"batch64_median": b64_ratio, "batch1_p95": b1_ratio},
        "gates": {"batch64_median_ratio_passed": b64_ratio <= 1.25, "batch1_p95_ratio_passed": b1_ratio <= 1.35},
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    output = Path(args.output) if Path(args.output).is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
