"""Route-B R2/R3 train-only preflight.

This is an audit and forward smoke, not a performance run.  It validates the
physical-feature conditioned branch and the availability/alignment of the nine
train-only power views.  It deliberately does not enumerate or open any test
MAT file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cj0724 import CLASS_NAMES, DEFAULT_TRAIN_MAT
from cj0724.data import read_labels
from cj0724.multitask import raw_iq_physical_features, raw_iq_spectral_features, build_multitask_model


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def train_contract(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        receiver_seed = np.asarray(handle["receiver_noise_seed"][:, :]).reshape(-1)
        shape = tuple(int(v) for v in handle["data_legacy"].shape)
    return {
        "path": str(path.resolve()),
        "data_legacy_shape": list(shape),
        "labels_shape": list(labels.shape),
        "label_sha256": sha_array(labels),
        "receiver_noise_seed_sha256": sha_array(receiver_seed),
        "class_counts": labels.sum(axis=0).astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    parser.add_argument("--power-root", type=Path, default=Path(r"E:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("docs"))
    args = parser.parse_args()
    data_file = Path(args.data_file).resolve()
    power_root = args.power_root.resolve()
    device = torch.device("cpu")

    dummy = torch.randn(2, 2, 3000)
    branch_smoke = {}
    with torch.no_grad():
        physical_features = raw_iq_physical_features(dummy)
        spectral_features = raw_iq_spectral_features(dummy)
    for name, features in (("physical_film", physical_features), ("spectral_film", spectral_features)):
        model_name = "mlcnn_iq_grouped_physical_film_task_routed" if name == "physical_film" else "mlcnn_iq_grouped_spectral_film_task_routed"
        model = build_multitask_model(model_name, width=4, head_dim=32, parameter_hidden_dim=16).to(device).eval()
        with torch.no_grad():
            logits, parameter = model(dummy)
        branch_smoke[name] = {
            "feature_shape": list(features.shape),
            "logit_shape": list(logits.shape),
            "parameter_shape": list(parameter.shape),
            "finite": bool(torch.isfinite(features).all() and torch.isfinite(logits).all() and torch.isfinite(parameter).all()),
            "parameter_input_has_no_label_or_target": True,
        }

    levels = []
    reference_labels = None
    reference_receiver_seed = None
    for level in range(10, 51, 5):
        path = power_root / f"level_{(level - 5) // 5:02d}__jnr_{level}dB" / "closedset_6class_train.mat"
        if not path.is_file():
            levels.append({"jnr_dB": level, "path": str(path), "exists": False})
            continue
        labels = read_labels(path)
        with h5py.File(path, "r") as handle:
            receiver_seed = np.asarray(handle["receiver_noise_seed"][:, :]).reshape(-1)
        same_as_reference = reference_labels is None or np.array_equal(reference_labels, labels)
        same_receiver_seed = reference_receiver_seed is None or np.array_equal(reference_receiver_seed, receiver_seed)
        if reference_labels is None:
            reference_labels = labels
            reference_receiver_seed = receiver_seed
        levels.append({"jnr_dB": level, "path": str(path), "exists": True, "labels_shape": list(labels.shape), "labels_equal_to_first_train_view": bool(same_as_reference), "receiver_noise_seed_equal_to_first_train_view": bool(same_receiver_seed), "label_sha256": sha_array(labels), "receiver_noise_seed_sha256": sha_array(receiver_seed)})

    payload = {
        "stage": "R2_R3_preflight",
        "status": "verified_audit_and_smoke" if all(item.get("exists") and item.get("labels_equal_to_first_train_view") and item.get("receiver_noise_seed_equal_to_first_train_view") for item in levels) and all(item["finite"] for item in branch_smoke.values()) else "partial_or_failed",
        "active_train_contract": train_contract(data_file),
        "class_names": list(CLASS_NAMES),
        "auxiliary_branches": branch_smoke,
        "power_train_views": levels,
        "test_mat_opened": False,
        "locked_test_used": False,
        "performance_claim": "not_validated",
        "next": "R2 multi-seed branch comparison and R3 per-parameter/visibility/power evaluation remain pending",
    }
    output_root = args.output_root.resolve()
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "receipts").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "receipts" / "route_b_r2_r3_preflight_receipt.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_root / "results" / "route_b_r2_r3_preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = ["# Route B R2/R3 preflight", "", f"- Status: **{payload['status']}**", "- Test MAT opened: `false`", "- Locked test used: `false`", "", "## Auxiliary branches", "", f"`{json.dumps(branch_smoke, ensure_ascii=False)}`", "", "## Train-only paired power views", "", "| JNR (dB) | Exists | Labels aligned | receiver_noise_seed aligned |", "|---:|:---:|:---:|:---:|"]
    report.extend(f"| {item['jnr_dB']} | {item.get('exists', False)} | {item.get('labels_equal_to_first_train_view', False)} | {item.get('receiver_noise_seed_equal_to_first_train_view', False)} |" for item in levels)
    report += ["", "This is an audit/forward smoke only; no accuracy, NMAE, visibility, or power-robustness claim is made."]
    (output_root / "reports" / "route_b_r2_r3_preflight.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
