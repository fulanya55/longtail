"""Frozen U-Net plus a VFT instantaneous-frequency sequence expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import discover_power_train_files, read_identity_metadata
from cj0724.isrj_separator_physics_head import clipped_logit, target_metrics, threshold_at_fpr
from cj0724.oracle_swept_isolation_experiment import (
    SweptIsolationDataset,
    make_loader,
    set_seed,
    sha256_indices,
)


VFT_INDEX = 1


def _binned_stats(sequence: torch.Tensor, bins: int, extremes: bool = False) -> torch.Tensor:
    values = sequence[:, None].float()
    mean = F.adaptive_avg_pool1d(values, bins).squeeze(1)
    square_mean = F.adaptive_avg_pool1d(values.square(), bins).squeeze(1)
    std = (square_mean - mean.square()).clamp_min(0.0).sqrt()
    parts = [mean, std]
    if extremes:
        parts.extend([
            F.adaptive_max_pool1d(values, bins).squeeze(1),
            -F.adaptive_max_pool1d(-values, bins).squeeze(1),
        ])
    return torch.cat(parts, dim=1)


@torch.no_grad()
def instantaneous_frequency_features(iq: torch.Tensor, bins: int = 60) -> dict[str, torch.Tensor]:
    """Summarize time-resolved phase increments without a global FFT."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 3000):
        raise ValueError("iq must have shape [B,2,3000]")
    signal = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    signal = signal / signal.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    increment = torch.angle(signal[:, 1:] * signal[:, :-1].conj()) / torch.pi
    magnitude = 0.5 * (signal[:, 1:].abs() + signal[:, :-1].abs())
    reliability = magnitude / magnitude.mean(dim=1, keepdim=True).clamp_min(1.0e-8)
    weighted = increment * reliability.clamp(max=3.0)
    smooth9 = F.avg_pool1d(weighted[:, None], 9, stride=1, padding=4).squeeze(1)
    smooth33 = F.avg_pool1d(weighted[:, None], 33, stride=1, padding=16).squeeze(1)
    derivative = F.pad(smooth9[:, 1:] - smooth9[:, :-1], (1, 0))

    trajectory = torch.cat([
        _binned_stats(weighted, bins),
        _binned_stats(smooth9, bins),
        _binned_stats(smooth33, bins),
    ], dim=1)
    dynamics = torch.cat([
        _binned_stats(smooth9, bins, extremes=True),
        _binned_stats(derivative, bins),
        F.adaptive_max_pool1d(derivative.abs()[:, None], bins).squeeze(1),
        _binned_stats(magnitude, bins),
    ], dim=1)
    return {
        "if_trajectory": trajectory.float(),
        "if_dynamics": dynamics.float(),
        "if_combined": torch.cat((trajectory, dynamics), dim=1).float(),
    }


@torch.no_grad()
def extract_role(
    model: ConditionalComplexUNetSeparator,
    dataset: SweptIsolationDataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = make_loader(dataset, batch_size, False, seed)
    output = {
        "labels": [], "probabilities": [], "logits": [], "sample_indices": [],
        "if_trajectory": [], "if_dynamics": [], "if_combined": [],
    }
    model.eval()
    for batch in loader:
        components = model.forward_components(batch["iq"].to(device, non_blocking=True), None)
        logits = components["logits"].float()
        features = instantaneous_frequency_features(components["normalized_estimate"].float())
        output["labels"].append(batch["label"].numpy())
        output["probabilities"].append(torch.sigmoid(logits).cpu().numpy())
        output["logits"].append(logits.cpu().numpy())
        output["sample_indices"].append(batch["sample_index"].numpy())
        for name in features:
            output[name].append(features[name].cpu().numpy())
    return {
        name: np.concatenate(parts).astype(
            np.int64 if name == "sample_indices" else np.uint8 if name == "labels" else np.float32
        )
        for name, parts in output.items()
    }


def fit_models(train: dict[str, np.ndarray], seed: int) -> dict[str, tuple[object, str]]:
    target = train["labels"][:, VFT_INDEX]
    configurations = {
        "trajectory_leaf7": ("if_trajectory", 7),
        "trajectory_leaf15": ("if_trajectory", 15),
        "dynamics_leaf15": ("if_dynamics", 15),
        "combined_leaf15": ("if_combined", 15),
    }
    models: dict[str, tuple[object, str]] = {}
    for name, (view, leaves) in configurations.items():
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(train[view], target)
        models[name] = (model, view)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    logistic.fit(train["if_combined"], target)
    models["combined_logistic"] = (logistic, "if_combined")
    return models


def build_candidates(
    roles: dict[str, dict[str, np.ndarray]], seed: int, fpr_cap: float
) -> dict[str, dict[str, object]]:
    models = fit_models(roles["train"], seed)
    base = {role: payload["probabilities"][:, VFT_INDEX] for role, payload in roles.items()}
    physics = {
        role: {name: model.predict_proba(payload[view])[:, 1] for name, (model, view) in models.items()}
        for role, payload in roles.items()
    }
    candidates: dict[str, dict[str, object]] = {}

    def register(name: str, probability: dict[str, np.ndarray]) -> None:
        threshold = threshold_at_fpr(
            roles["calibration"]["labels"][:, VFT_INDEX], probability["calibration"], fpr_cap
        )
        candidates[name] = {
            "threshold": threshold,
            "calibration": target_metrics(
                roles["calibration"]["labels"][:, VFT_INDEX], probability["calibration"], threshold
            ),
            "tune": target_metrics(roles["tune"]["labels"][:, VFT_INDEX], probability["tune"], threshold),
        }

    register("unet_classifier", base)
    for name in models:
        register(name, {role: physics[role][name] for role in roles})
        stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
        stacker.fit(
            np.column_stack((clipped_logit(base["early"]), clipped_logit(physics["early"][name]))),
            roles["early"]["labels"][:, VFT_INDEX],
        )
        stacked = {}
        for role in roles:
            matrix = np.column_stack((clipped_logit(base[role]), clipped_logit(physics[role][name])))
            stacked[role] = stacker.predict_proba(matrix)[:, 1]
        register(f"stack_{name}", stacked)

    names = sorted(models)
    stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    stacker.fit(
        np.column_stack([clipped_logit(base["early"])] + [clipped_logit(physics["early"][name]) for name in names]),
        roles["early"]["labels"][:, VFT_INDEX],
    )
    stacked = {}
    for role in roles:
        matrix = np.column_stack([clipped_logit(base[role])] + [clipped_logit(physics[role][name]) for name in names])
        stacked[role] = stacker.predict_proba(matrix)[:, 1]
    register("stack_all_frequency", stacked)
    return candidates


def select_candidate(results: list[dict[str, object]], cap: float) -> tuple[str, list[dict[str, object]]]:
    rows = []
    for name in sorted(results[0]["candidates"]):
        values = [result["candidates"][name]["tune"] for result in results]
        baseline = [result["candidates"]["unet_classifier"]["tune"] for result in results]
        row = {
            "candidate": name,
            "aupr_mean": float(np.mean([value["aupr"] for value in values])),
            "recall_mean": float(np.mean([value["recall"] for value in values])),
            "recall_std": float(np.std([value["recall"] for value in values])),
            "fpr_mean": float(np.mean([value["fpr"] for value in values])),
            "max_seed_fpr": float(max(value["fpr"] for value in values)),
            "delta_aupr_mean": float(np.mean([value["aupr"] - base["aupr"] for value, base in zip(values, baseline)])),
            "delta_recall_mean": float(np.mean([value["recall"] - base["recall"] for value, base in zip(values, baseline)])),
            "improved_seeds": int(sum(value["recall"] > base["recall"] for value, base in zip(values, baseline))),
        }
        row["eligible"] = bool(
            row["fpr_mean"] <= cap + 1.0e-12
            and row["max_seed_fpr"] <= cap + 0.02
            and row["delta_aupr_mean"] >= -0.01
        )
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["recall_mean"], row["aupr_mean"], -row["fpr_mean"]))
    return str(selected["candidate"]), rows


def build_summary(
    results: list[dict[str, object]], cap: float, separator_root: Path, isrj_root: Path
) -> dict[str, object]:
    selected, aggregate = select_candidate(results, cap)
    selected_row = next(row for row in aggregate if row["candidate"] == selected)
    isrj_results = {
        int(row["seed"]): row for row in json.loads((isrj_root / "results.json").read_text(encoding="utf-8"))
    }
    system = []
    for result in results:
        seed = int(result["seed"])
        unet = json.loads(
            (separator_root / "unet_wave" / f"seed_{seed}" / "result.json").read_text(encoding="utf-8")
        )["tune"]
        saw = unet["per_class"][0]
        vft = result["candidates"][selected]["tune"]
        isrj = isrj_results[seed]["candidates"]["stack_all_periodicity"]["tune"]
        system.append({
            "seed": seed,
            "macro_aupr": float(np.mean([saw["aupr"], vft["aupr"], isrj["aupr"]])),
            "macro_recall": float(np.mean([saw["recall"], vft["recall"], isrj["recall"]])),
            "macro_fpr": float(np.mean([saw["fpr"], vft["fpr"], isrj["fpr"]])),
            "per_class_recall": {"SawFM": saw["recall"], "VFT": vft["recall"], "ISRJ": isrj["recall"]},
        })
    means = {
        name: float(np.mean([row[name] for row in system]))
        for name in ("macro_aupr", "macro_recall", "macro_fpr")
    }
    means["SawFM_recall"] = float(np.mean([row["per_class_recall"]["SawFM"] for row in system]))
    means["VFT_recall"] = float(np.mean([row["per_class_recall"]["VFT"] for row in system]))
    means["ISRJ_recall"] = float(np.mean([row["per_class_recall"]["ISRJ"] for row in system]))
    expert_gate = bool(
        selected_row["recall_mean"] >= 0.70 and selected_row["fpr_mean"] <= cap
        and selected_row["delta_recall_mean"] >= 0.05 and selected_row["improved_seeds"] == len(results)
    )
    system_gate = bool(
        means["macro_recall"] >= 0.70 and means["macro_fpr"] <= cap
        and all(means[f"{name}_recall"] >= 0.70 for name in ("SawFM", "VFT", "ISRJ"))
    )
    return {
        "selected_candidate": selected,
        "aggregate": aggregate,
        "expert_promotion_gate": {
            "requirements": "VFT recall >=0.70, FPR <=0.10, recall gain >=0.05, 3/3 seeds improve",
            "passed": expert_gate,
        },
        "combined_system": {"per_seed": system, "mean": means, "promotion_passed": system_gate},
        "selection_role": "tune",
        "confirm_used": False,
        "locked_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--isrj-root", default="outputs/isrj_separator_physics_head_v2_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/vft_separator_frequency_head_20260726")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--fpr-cap", type=float, default=0.10)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.aggregate_only:
        results = [json.loads((output_root / f"seed_{seed}" / "result.json").read_text(encoding="utf-8")) for seed in args.seeds]
        summary = build_summary(results, args.fpr_cap, Path(args.separator_root), Path(args.isrj_root))
        (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=False)
    role_file = np.load(Path(args.run_root) / "role_indices.npz")
    role_names = ("train", "early", "calibration", "tune")
    indices = {name: role_file[name].astype(np.int64) for name in role_names}
    files = discover_power_train_files(args.power_root)
    _, labels = read_identity_metadata(files[10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = {
        "purpose": "freeze U-Net and ISRJ head; add VFT instantaneous-frequency sequence expert",
        "roles": {name: {"count": len(value), "sha256": sha256_indices(value)} for name, value in indices.items()},
        "fit_roles": {"frequency_models": "train", "stacker": "early", "threshold": "calibration", "selection": "tune"},
        "feature_rule": "phase increment plus multi-scale temporal pooling; no global FFT",
        "confirm_used": False, "locked_test_used": False, "device": str(device),
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    results = []
    for seed in args.seeds:
        set_seed(seed)
        checkpoint_path = Path(args.separator_root) / "unet_wave" / f"seed_{seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        payloads, datasets = {}, {}
        for role in role_names:
            dataset = SweptIsolationDataset(files[10], files[50], labels, indices[role], "raw_10db")
            datasets[role] = dataset
            payloads[role] = extract_role(model, dataset, args.batch_size, seed, device)
            if not np.array_equal(payloads[role]["sample_indices"], indices[role]):
                raise RuntimeError(f"sample order changed for role {role}")
        candidates = build_candidates(payloads, seed, args.fpr_cap)
        result = {"seed": seed, "separator_checkpoint": str(checkpoint_path), "candidates": candidates}
        results.append(result)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, "tune": {name: row["tune"] for name, row in candidates.items()}}), flush=True)
        for dataset in datasets.values():
            dataset.close()
        del model, payloads
        if device.type == "cuda": torch.cuda.empty_cache()

    summary = build_summary(results, args.fpr_cap, Path(args.separator_root), Path(args.isrj_root))
    (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
