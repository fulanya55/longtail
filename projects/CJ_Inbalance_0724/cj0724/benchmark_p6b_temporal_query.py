"""Eager CUDA cost audit for P6b C0/C1/C2 frozen development checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.multitask import build_multitask_model


def _measure(model: torch.nn.Module, batch_size: int, warmup: int, repeats: int) -> dict[str, float]:
    iq = torch.randn(batch_size, 2, 3000, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            model(iq)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); model(iq); end.record(); torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "repeats": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int, default=17041)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P6b latency audit requires CUDA")
    project = Path(__file__).resolve().parents[1]
    root = Path(args.run_root) if Path(args.run_root).is_absolute() else project / args.run_root
    names = ("mlcnn_iq_grouped_task_routed", "mlcnn_iq_grouped_temporal_query_task_routed", "mlcnn_iq_grouped_temporal_global_control_task_routed")
    results = {}
    for name in names:
        model = build_multitask_model(name, 24, 0.2, 256, 64).cuda().eval()
        checkpoint = torch.load(root / f"{name}__cos_lr1e4_wd1e4__seed{args.seed}" / "best_parameter.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        results[name] = {str(batch): _measure(model, batch, args.warmup, args.repeats) for batch in (1, 64)}
        del model; torch.cuda.empty_cache()
    c0, c1, c2 = (results[name] for name in names)
    ratios = {
        "C1_C2_batch64_median": c1["64"]["median_ms"] / c2["64"]["median_ms"],
        "C1_C0_batch64_median": c1["64"]["median_ms"] / c0["64"]["median_ms"],
        "C1_C0_batch1_p95": c1["1"]["p95_ms"] / c0["1"]["p95_ms"],
    }
    result = {"run_root": str(root), "seed": args.seed, "mode": "eager_fp32", "results": results, "ratios": ratios,
              "gates": {"C1_C2_batch64_median_passed": ratios["C1_C2_batch64_median"] <= 1.05,
                        "C1_C0_batch64_median_passed": ratios["C1_C0_batch64_median"] <= 1.25,
                        "C1_C0_batch1_p95_passed": ratios["C1_C0_batch1_p95"] <= 1.35},
              "test_generated": False, "test_opened": False, "locked_test_used": False}
    output = Path(args.output) if Path(args.output).is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
