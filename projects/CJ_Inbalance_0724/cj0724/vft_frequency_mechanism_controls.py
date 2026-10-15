"""Mechanism controls for the frozen VFT instantaneous-frequency expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import discover_power_train_files, read_identity_metadata
from cj0724.isrj_separator_physics_head import target_metrics, threshold_at_fpr
from cj0724.oracle_swept_isolation_experiment import SweptIsolationDataset, make_loader, set_seed, sha256_indices
from cj0724.vft_separator_frequency_head import VFT_INDEX, instantaneous_frequency_features


def global_summary(trajectory: np.ndarray, bins: int = 60) -> np.ndarray:
    channels = trajectory.reshape(len(trajectory), -1, bins)
    return np.concatenate(
        (channels.mean(2), channels.std(2), channels.min(2), channels.max(2)), axis=1
    ).astype(np.float32)


@torch.no_grad()
def extract_pair(
    model: ConditionalComplexUNetSeparator,
    dataset: SweptIsolationDataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    raw_parts, separated_parts, labels, indices = [], [], [], []
    for batch in make_loader(dataset, batch_size, False, seed):
        raw = batch["iq"].to(device, non_blocking=True)
        separated = model.forward_components(raw, None)["normalized_estimate"].float()
        raw_parts.append(instantaneous_frequency_features(raw)["if_trajectory"].cpu().numpy())
        separated_parts.append(instantaneous_frequency_features(separated)["if_trajectory"].cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["sample_index"].numpy())
    return {
        "raw": np.concatenate(raw_parts).astype(np.float32),
        "separated": np.concatenate(separated_parts).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.uint8),
        "indices": np.concatenate(indices).astype(np.int64),
    }


def fit_score(
    roles: dict[str, dict[str, np.ndarray]], view: str, seed: int, cap: float, shuffle_labels: bool = False
) -> dict[str, object]:
    matrices = {}
    for role, payload in roles.items():
        matrix = payload[view]
        matrices[role] = global_summary(matrix) if view.endswith("_global") else matrix
    target = roles["train"]["labels"][:, VFT_INDEX].copy()
    if shuffle_labels:
        target = np.random.default_rng(seed + 1771).permutation(target)
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(matrices["train"], target)
    probability = {role: model.predict_proba(matrix)[:, 1] for role, matrix in matrices.items()}
    threshold = threshold_at_fpr(
        roles["calibration"]["labels"][:, VFT_INDEX], probability["calibration"], cap
    )
    return {
        "threshold": threshold,
        "calibration": target_metrics(
            roles["calibration"]["labels"][:, VFT_INDEX], probability["calibration"], threshold
        ),
        "tune": target_metrics(roles["tune"]["labels"][:, VFT_INDEX], probability["tune"], threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/vft_frequency_mechanism_controls_20260726")
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--fpr-cap", type=float, default=0.10)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    role_file = np.load(Path(args.run_root) / "role_indices.npz")
    role_names = ("train", "calibration", "tune")
    role_indices = {name: role_file[name].astype(np.int64) for name in role_names}
    files = discover_power_train_files(args.power_root)
    _, labels = read_identity_metadata(files[10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.separator_root) / "unet_wave" / f"seed_{args.seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)

    roles, datasets = {}, {}
    for role in role_names:
        dataset = SweptIsolationDataset(files[10], files[50], labels, role_indices[role], "raw_10db")
        datasets[role] = dataset
        roles[role] = extract_pair(model, dataset, args.batch_size, args.seed, device)
        roles[role]["raw_global"] = roles[role]["raw"]
        roles[role]["separated_global"] = roles[role]["separated"]
        if not np.array_equal(roles[role]["indices"], role_indices[role]):
            raise RuntimeError(f"sample order changed for {role}")

    runs = {
        "raw_trajectory": fit_score(roles, "raw", args.seed, args.fpr_cap),
        "separated_trajectory": fit_score(roles, "separated", args.seed, args.fpr_cap),
        "raw_global": fit_score(roles, "raw_global", args.seed, args.fpr_cap),
        "separated_global": fit_score(roles, "separated_global", args.seed, args.fpr_cap),
        "shuffled_label_separated": fit_score(
            roles, "separated", args.seed, args.fpr_cap, shuffle_labels=True
        ),
    }
    result = {
        "protocol": {
            "seed": args.seed,
            "roles": {name: {"count": len(value), "sha256": sha256_indices(value)} for name, value in role_indices.items()},
            "classifier": "fixed HistGradientBoosting leaf15",
            "global_control": "mean/std/min/max over each IF summary channel",
            "confirm_used": False, "locked_test_used": False,
        },
        "runs": runs,
        "checks": {
            "no_label_leakage": bool(runs["separated_trajectory"]["tune"]["aupr"] - runs["shuffled_label_separated"]["tune"]["aupr"] >= 0.20),
            "separation_adds_ranking_value": bool(runs["separated_trajectory"]["tune"]["aupr"] - runs["raw_trajectory"]["tune"]["aupr"] >= 0.01),
            "time_resolved_beats_global": bool(runs["separated_trajectory"]["tune"]["aupr"] - runs["separated_global"]["tune"]["aupr"] >= 0.01),
        },
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    for dataset in datasets.values(): dataset.close()


if __name__ == "__main__":
    main()
