"""Pre-data static selection of the generic equal-MAC control for P7a."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cj0724.benchmark_multitask_efficiency import count_macs
from cj0724.multitask import ComputeMatchedGlobalControl, build_multitask_model
from cj0724.models import build_benchmark_model
from cj0724.p3n_cross_interference_backbone import sha256_file


def control_model(channels: int):
    classifier = build_benchmark_model("legacy_mlcnn", num_classes=6, width=24, dropout=.2, head_dim=256)
    return ComputeMatchedGlobalControl(classifier, parameter_hidden_dim=64, branch_channels=int(channels)).eval()


def static_rows(channels: list[int]) -> dict:
    sample = torch.zeros(1, 2, 3000)
    route = build_multitask_model("mlcnn_iq_grouped_task_routed", width=24, dropout=.2, head_dim=256, parameter_hidden_dim=64).eval()
    reference_macs = count_macs(route, sample); reference_parameters = sum(p.numel() for p in route.parameters())
    rows = []
    for channel in channels:
        model = control_model(channel); macs = count_macs(model, sample); parameters = sum(p.numel() for p in model.parameters())
        rows.append({"channel": int(channel), "macs": int(macs), "parameters": int(parameters), "mac_ratio": float(macs / reference_macs), "parameter_ratio": float(parameters / reference_parameters)})
    eligible = [row for row in rows if .98 <= row["mac_ratio"] <= 1.02 and .98 <= row["parameter_ratio"] <= 1.02]
    selected = min(eligible, key=lambda row: (abs(row["mac_ratio"] - 1.0), row["channel"])) if eligible else None
    return {"reference": {"model": "mlcnn_iq_grouped_task_routed", "macs": int(reference_macs), "parameters": int(reference_parameters)}, "rows": rows, "selected": selected, "static_passed": selected is not None, "test_generated": False, "test_opened": False, "locked_test_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", default="configs/p7a_heterogeneous_equal_mac_static_screen_preregister_20260802.json")
    parser.add_argument("--output", default="outputs/p7a_heterogeneous_equal_mac_static_screen_20260802/result.json")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    preregistration = json.loads((project / args.preregister).read_text(encoding="utf-8"))
    result = static_rows([int(value) for value in preregistration["static_only"]["channel_grid"]])
    result["preregister_sha256"] = sha256_file(project / args.preregister)
    output = project / args.output; output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
