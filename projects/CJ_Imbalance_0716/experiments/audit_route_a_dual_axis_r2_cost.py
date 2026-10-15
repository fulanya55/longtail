"""Reproducible architecture and CUDA inference-cost audit for Route-A R2."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "outputs" / "route_a_dual_axis_next" / "r2_formal_primary"
OUTPUT_ROOT = ROOT / "outputs" / "route_a_dual_axis_next" / "r2_cost_audit"
SCRIPT_PATH = Path(__file__).resolve()
CHECKPOINT_ROOT = FORMAL_ROOT / "seed_260811" / "LT-VL"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from cjlt.models.dual_axis_sawfm import RoutedSawFMModel  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_macs(model: nn.Module, signal: torch.Tensor) -> int:
    total = [0]
    handles = []

    def hook(module: nn.Module, _inputs, output):
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        if isinstance(module, nn.Conv1d):
            total[0] += int(
                tensor.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
            )
        elif isinstance(module, nn.Linear):
            total[0] += int(tensor.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(signal)
    finally:
        for handle in handles:
            handle.remove()
    return int(total[0])


def benchmark(model: nn.Module, batch_size: int, device: torch.device) -> dict:
    signal = torch.randn(batch_size, 2, 3000, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(50):
            model(signal)
    torch.cuda.synchronize(device)
    repeats = []
    iterations = 200 if batch_size == 1 else 100
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(iterations):
                model(signal)
        end.record()
        torch.cuda.synchronize(device)
        repeats.append(float(start.elapsed_time(end) / iterations))
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    with torch.inference_mode():
        model(signal)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    median_ms = float(statistics.median(repeats))
    return {
        "batch_size": int(batch_size),
        "iterations_per_repeat": int(iterations),
        "repeat_mean_latency_ms": repeats,
        "median_batch_latency_ms": median_ms,
        "median_per_sample_latency_ms": median_ms / batch_size,
        "median_throughput_samples_per_second": 1000.0 * batch_size / median_ms,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_bytes": peak - baseline,
    }


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen cost audit")
    device = torch.device("cuda")
    torch.manual_seed(26082501)
    torch.cuda.manual_seed_all(26082501)
    torch.backends.cudnn.benchmark = False

    core = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    raw = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=16, topk=16, dropout=0.1, descriptor_mode="full"
    )
    core_path = CHECKPOINT_ROOT / "m0_focal_core.pt"
    raw_path = CHECKPOINT_ROOT / "rawsaw.pt"
    core.load_state_dict(load_state(core_path), strict=True)
    raw.load_state_dict(load_state(raw_path), strict=True)
    routed = RoutedSawFMModel(core, raw)

    cpu_probe = torch.randn(1, 2, 3000)
    core_macs = count_macs(core.eval(), cpu_probe)
    routed_macs = count_macs(routed.eval(), cpu_probe)
    raw_increment_macs = routed_macs - core_macs
    if core_macs != 211_857_536 or raw_increment_macs != 7_442_848:
        raise ValueError(f"unexpected MAC count: core={core_macs}, raw={raw_increment_macs}")

    core = core.to(device)
    raw = raw.to(device)
    routed = routed.to(device)
    models = {
        "M0_or_MF_core": core,
        "MV_or_MFV_routed": routed,
    }
    latency = {
        name: {str(batch): benchmark(model, batch, device) for batch in (1, 64)}
        for name, model in models.items()
    }

    formal_summary_path = FORMAL_ROOT / "formal_primary_summary.json"
    terminal_path = FORMAL_ROOT / "terminal_receipt.json"
    formal = json.loads(formal_summary_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if formal["status"] != "completed" or terminal["status"] != "completed":
        raise ValueError("formal run is not complete")
    if any(bool(terminal.get(key)) for key in ("locked_test_used", "test_v2_used", "test_v3_used")):
        raise ValueError("formal receipt reports forbidden test use")
    rows = formal["rows"]
    training_cost = {
        "formal_matrix_cell_count": len(rows),
        "terminal_wall_seconds": float(terminal["elapsed_seconds"]),
        "sum_cell_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
        "mean_cell_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
        "max_peak_cuda_memory_bytes": int(max(row["peak_cuda_memory_bytes"] for row in rows)),
        "best_epoch_mean": {
            name: float(np.mean([row["best_epochs"][name] for row in rows]))
            for name in ("M0_B0_Focal", "MF_B0_DB", "RawSaw")
        },
        "best_epoch_range": {
            name: [
                int(min(row["best_epochs"][name] for row in rows)),
                int(max(row["best_epochs"][name] for row in rows)),
            ]
            for name in ("M0_B0_Focal", "MF_B0_DB", "RawSaw")
        },
    }
    summary = {
        "status": "completed",
        "evidence_level": "frozen_architecture_cost_audit",
        "representative_seed_cell": "260811/LT-VL",
        "input_shape": [1, 2, 3000],
        "parameters": {
            "M0_or_MF_core": parameter_count(core),
            "RawSaw_increment": parameter_count(raw),
            "MV_or_MFV_routed": parameter_count(routed),
            "increment_percent": 100.0 * parameter_count(raw) / parameter_count(core),
        },
        "macs_per_sample": {
            "M0_or_MF_core": core_macs,
            "RawSaw_increment": raw_increment_macs,
            "MV_or_MFV_routed": routed_macs,
            "increment_percent": 100.0 * raw_increment_macs / core_macs,
            "counted_modules": "Conv1d and Linear multiply-accumulate operations",
        },
        "latency": latency,
        "training_cost": training_cost,
        "checkpoint_hashes": {
            "m0_focal_core": sha256(core_path),
            "rawsaw": sha256(raw_path),
        },
        "formal_summary_sha256": sha256(formal_summary_path),
        "terminal_receipt_sha256": sha256(terminal_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device),
        },
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "cost_summary.json"
    write_json(summary_path, summary)
    receipt = {
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    write_json(OUTPUT_ROOT / "run_receipt.json", receipt)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
