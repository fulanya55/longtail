"""Engineering cost audit for the frozen B0 + RB-C linear-head candidate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.backbone_adapter import build_backbone
from cjlt.models.frequency_reconstruction import FrozenLinearHead
from experiments.run_route_a_dual_axis_sawfm_r2_formal import sha256, write_json


SOURCE_ROOT = ROOT / "outputs/route_a_frequency_reconstruction_v2/stage2_formal/seed_260821/LabelIF20"
OUTPUT_ROOT = ROOT / "outputs/route_a_frequency_reconstruction_v2/stage2_cost_audit"


class Candidate(nn.Module):
    def __init__(self, core, head):
        super().__init__()
        self.core = core
        self.head = head

    def forward(self, values):
        return self.head(self.core.forward_features(values))


def macs_with_hooks(model, sample):
    total = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total
        batch, out_channels, out_length = output.shape
        kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
        total += int(batch * out_channels * out_length * kernel_ops)

    def linear_hook(module, inputs, output):
        nonlocal total
        batch = int(np.prod(output.shape[:-1]))
        total += int(batch * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return total


def benchmark(model, device, batch_size, warmup, repeats):
    values = torch.randn(batch_size, 2, 3000, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(values)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            model(values)
        end.record()
        torch.cuda.synchronize(device)
    elapsed_ms = float(start.elapsed_time(end))
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "mean_batch_latency_ms": elapsed_ms / repeats,
        "mean_sample_latency_ms": elapsed_ms / (repeats * batch_size),
        "throughput_samples_per_second": repeats * batch_size * 1000.0 / elapsed_ms,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    core_checkpoint = torch.load(SOURCE_ROOT / "F0_B0_Focal.pt", map_location="cpu", weights_only=False)
    head_checkpoint = torch.load(SOURCE_ROOT / "F3_RBC_cRT.pt", map_location="cpu", weights_only=False)
    core = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    core.load_state_dict(core_checkpoint["model_state_dict"], strict=True)
    head_state = head_checkpoint["model_state_dict"]
    head = FrozenLinearHead(head_state["linear.weight"], head_state["linear.bias"])
    head.load_state_dict(head_state, strict=True)
    core.classifier = nn.Identity()
    model = Candidate(core, head).to(device).eval()
    sample = torch.randn(1, 2, 3000, device=device)
    macs = macs_with_hooks(model, sample)
    payload = {
        "status": "completed",
        "experiment": "RouteA_FrequencyReconstructionV2_CostAudit",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "input_shape": [1, 2, 3000],
        "parameter_count_total": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_count_trainable_head_only": int(sum(parameter.numel() for parameter in head.parameters())),
        "macs_per_sample_conv_linear_only": int(macs),
        "macs_scope": "Conv1d and Linear multiply-accumulates; normalization, activation, pooling, elementwise complex operations, and memory traffic excluded",
        "batch_1": benchmark(model, device, batch_size=1, warmup=50, repeats=300),
        "batch_64": benchmark(model, device, batch_size=64, warmup=20, repeats=100),
        "source_checkpoints": {
            "core": {"path": str(SOURCE_ROOT / "F0_B0_Focal.pt"), "sha256": sha256(SOURCE_ROOT / "F0_B0_Focal.pt")},
            "head": {"path": str(SOURCE_ROOT / "F3_RBC_cRT.pt"), "sha256": sha256(SOURCE_ROOT / "F3_RBC_cRT.pt")},
        },
        "inference_structure_note": "F3 replaces the original Linear(128,6) decision head; it does not add inference parameters or MACs versus B0.",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    summary = OUTPUT_ROOT / "cost_summary.json"
    write_json(summary, payload)
    write_json(OUTPUT_ROOT / "run_receipt.json", {
        "status": "completed",
        "summary_sha256": sha256(summary),
        "script_sha256": sha256(Path(__file__)),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    })
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
