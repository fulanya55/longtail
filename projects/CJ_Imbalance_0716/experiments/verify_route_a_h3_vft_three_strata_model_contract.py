"""Record the CPU-only T1/T2 capacity and target-logit contract after H3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.multipulse_vft_expert import MultiPulseVFTExpertClassifier
from cjlt.models.vft_staggered_pri import record_samples


def run(output: Path) -> dict[str, object]:
    true = MultiPulseVFTExpertClassifier("T1-h3", width=8, expert_dim=4).eval()
    permuted = MultiPulseVFTExpertClassifier("T2-h3", width=8, expert_dim=4).eval()
    equal = sum(item.numel() for item in true.parameters()) == sum(item.numel() for item in permuted.parameters())
    value = torch.randn(2, 2, record_samples())
    with torch.no_grad():
        pieces = true(value, return_components=True)
    non_target = torch.allclose(pieces["logits"][:, [0, 1, 2, 3, 5]], pieces["global_logits"][:, [0, 1, 2, 3, 5]])
    target_additive = torch.allclose(pieces["logits"][:, 4], pieces["global_logits"][:, 4] + pieces["vft_expert_logit"])
    if not all((equal, non_target, target_additive)):
        raise RuntimeError("H3 target-only capacity contract failed")
    result = {
        "status": "completed_model_contract_verification",
        "experiment": "RouteA_H3_VFTThreeStrataIdentifiability",
        "checks": {
            "t1_t2_equal_parameter_count": equal,
            "t1_non_vft_logits_equal_global_path": non_target,
            "t1_vft_logit_is_additive_expert_residual": target_additive,
        },
        "execution": "post-training CPU-only model contract verification; no source data, calibration, thresholds, or predictions opened",
        "external_used": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_queue/model_contract_recovery_receipt.json"))
    value = run(parser.parse_args().output)
    print(json.dumps({"status": value["status"], "checks": value["checks"]}, ensure_ascii=False))
