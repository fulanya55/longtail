"""P3o-H: independently wrapped finite-valid implementation of P3o-G's risk test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import cj0724.p3og_bounded_combo_cvar as core
from cj0724.p3n_cross_interference_backbone import sha256_file
from cj0724.p3o_low_power_attribution import audit_identities as audited_identity_roles


def finite_valid_per_sample_parameter_loss(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute each sample's SmoothL1 only after indexing finite valid targets."""
    valid = mask.bool() & (labels[:, :, None] > 0.5)
    if bool((valid & ~torch.isfinite(target)).any()):
        raise ValueError("valid parameter targets must be finite")
    rows = []
    for index in range(prediction.shape[0]):
        selected = valid[index]
        if bool(selected.any()):
            rows.append(F.smooth_l1_loss(prediction[index][selected], target[index][selected], reduction="mean"))
        else:
            rows.append(prediction[index].sum() * 0.0)
    return torch.stack(rows)


def _p3oh_identity_audit(project, train_identity, development_identity, ignored_seeds, original_data_file):
    return audited_identity_roles(project, train_identity, development_identity, {1270000000, 1280000000}, original_data_file)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    freeze = json.loads((project / "configs/p3oh_finite_masked_combo_cvar_method_freeze_20260802.json").read_text(encoding="utf-8"))
    if freeze["wrapper_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P3o-H wrapper hash mismatch")
    if freeze["implementation_sha256"] != sha256_file(Path(core.__file__).resolve()):
        raise RuntimeError("P3o-H core hash mismatch")
    core._per_sample_parameter_loss = finite_valid_per_sample_parameter_loss
    core.audit_identities = _p3oh_identity_audit
    core.MODEL_SEEDS = (2071, 2072, 2073)
    sys.argv.extend([
        "--train-root-template", "generated_data/p3oh_finite_masked_combo_cvar_train_seed1270000000_{regime}_rechunked",
        "--development-root-template", "generated_data/p3oh_finite_masked_combo_cvar_dev_seed1280000000_{regime}_rechunked",
        "--preregister", "configs/p3oh_finite_masked_combo_cvar_preregister_20260802.json",
        "--method-freeze", "configs/p3oh_finite_masked_combo_cvar_method_freeze_20260802.json",
        "--output-root", "outputs/p3oh_finite_masked_combo_cvar_development_20260802",
    ])
    core.main()


if __name__ == "__main__":
    main()
