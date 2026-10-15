"""Preflight the preregistered Route-B R4 ablation matrix.

This checks only that the comparison paths and data roles are available.  It
does not run an ablation or create performance evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cj0724 import DEFAULT_TRAIN_MAT
from cj0724.multitask import build_multitask_model


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    config_path = project / "configs" / "route_b_r4_preregister.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dummy = torch.randn(2, 2, 3000)
    model_paths = {
        "B1": "legacy_mlcnn_global",
        "B2": "legacy_mlcnn_class_conditioned",
        "B3": "legacy_mlcnn_class_conditioned_detached",
        "B4": "legacy_mlcnn_class_conditioned",
        "R2_spectral_film": "mlcnn_iq_grouped_spectral_film_task_routed",
        "R2_visibility_safe": "mlcnn_iq_grouped_task_routed_saw_safe",
        "R2_uniform": "mlcnn_iq_grouped_task_routed",
    }
    smoke = {}
    for label, model_name in model_paths.items():
        model = build_multitask_model(model_name, width=4, head_dim=32, parameter_hidden_dim=16).eval()
        with torch.no_grad():
            logits, parameters = model(dummy)
        smoke[label] = {
            "model": model_name,
            "logit_shape": list(logits.shape),
            "parameter_shape": list(parameters.shape),
            "finite": bool(torch.isfinite(logits).all() and torch.isfinite(parameters).all()),
        }
    payload = {
        "stage": "R4",
        "status": "verified_preflight_only",
        "active_train_file": str(Path(DEFAULT_TRAIN_MAT).resolve()),
        "ablation_count": len(config["ablation_matrix"]),
        "model_path_smoke": smoke,
        "test_mat_opened": False,
        "locked_test_used": False,
        "performance_claim": "not_validated",
        "next": "run each ablation only after complete R1 aggregation and with unchanged invariants",
    }
    (project / "docs" / "receipts").mkdir(parents=True, exist_ok=True)
    (project / "docs" / "reports").mkdir(parents=True, exist_ok=True)
    (project / "docs" / "receipts" / "route_b_r4_preflight_receipt.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (project / "docs" / "reports" / "route_b_r4_preflight.md").write_text(
        "# Route B R4 preflight\n\n" + f"- Status: **{payload['status']}**\n- Ablations: `{payload['ablation_count']}`\n- Test MAT opened: `false`\n- Locked test used: `false`\n\nThe matrix is preregistered but not performance-validated.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
