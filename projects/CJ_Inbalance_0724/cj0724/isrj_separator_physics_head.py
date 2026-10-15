"""Frozen U-Net plus ISRJ periodicity expert, evaluated on development roles only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import discover_power_train_files, read_identity_metadata
from cj0724.oracle_swept_isolation_experiment import (
    SweptIsolationDataset,
    make_loader,
    select_fpr_thresholds,
    set_seed,
    sha256_indices,
)
from cj0724.shared_physics_frontend import shared_feature_batch


ISRJ_INDEX = 2


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def target_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    truth = np.asarray(labels, dtype=bool)
    score = np.asarray(probability, dtype=np.float64)
    predicted = score >= float(threshold)
    positive, negative = truth, ~truth
    tp = int((predicted & positive).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    return {
        "aupr": float(sk_metrics.average_precision_score(truth, score)),
        "recall": float(predicted[positive].mean()) if positive.any() else float("nan"),
        "fpr": float(predicted[negative].mean()) if negative.any() else float("nan"),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "threshold": float(threshold),
    }


def threshold_at_fpr(labels: np.ndarray, probability: np.ndarray, cap: float) -> float:
    labels_2d = np.asarray(labels, dtype=np.uint8)[:, None]
    probability_2d = np.asarray(probability, dtype=np.float64)[:, None]
    return float(select_fpr_thresholds(labels_2d, probability_2d, cap)[0])


@torch.no_grad()
def extract_role(
    model: ConditionalComplexUNetSeparator,
    dataset: SweptIsolationDataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = make_loader(dataset, batch_size, False, seed)
    labels, probabilities, logits, features, sample_indices = [], [], [], [], []
    model.eval()
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        components = model.forward_components(iq, None)
        separated = components["normalized_estimate"].float()
        batch_logits = components["logits"].float()
        logits.append(batch_logits.cpu().numpy())
        probabilities.append(torch.sigmoid(batch_logits).cpu().numpy())
        features.append(shared_feature_batch(separated)["isrj_periodicity"].cpu().numpy())
        labels.append(batch["label"].numpy())
        sample_indices.append(batch["sample_index"].numpy())
    return {
        "labels": np.concatenate(labels).astype(np.uint8),
        "probabilities": np.concatenate(probabilities).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
        "features": np.concatenate(features).astype(np.float32),
        "sample_indices": np.concatenate(sample_indices).astype(np.int64),
    }


def feature_view(features: np.ndarray, name: str) -> np.ndarray:
    if name == "periodicity_full":
        return features
    if name == "envelope_periodicity":
        return np.concatenate((features[:, 120:240], features[:, 300:310]), axis=1)
    raise KeyError(name)


def fit_physics_models(train: dict[str, np.ndarray], seed: int) -> dict[str, object]:
    target = train["labels"][:, ISRJ_INDEX]
    configurations = {
        "periodicity_leaf7": ("periodicity_full", 7),
        "periodicity_leaf15": ("periodicity_full", 15),
        "envelope_leaf7": ("envelope_periodicity", 7),
    }
    models = {}
    for name, (view, leaves) in configurations.items():
        classifier = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=leaves,
            l2_regularization=1.0,
            random_state=seed,
        )
        classifier.fit(feature_view(train["features"], view), target)
        models[name] = (classifier, view)
    return models


def physics_probabilities(models: dict[str, object], role: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: model.predict_proba(feature_view(role["features"], view))[:, 1]
        for name, (model, view) in models.items()
    }


def build_candidates(
    roles: dict[str, dict[str, np.ndarray]], seed: int, fpr_cap: float
) -> dict[str, dict[str, object]]:
    models = fit_physics_models(roles["train"], seed)
    physics = {role: physics_probabilities(models, payload) for role, payload in roles.items()}
    candidates: dict[str, dict[str, object]] = {}

    def register(name: str, probabilities: dict[str, np.ndarray]) -> None:
        threshold = threshold_at_fpr(
            roles["calibration"]["labels"][:, ISRJ_INDEX], probabilities["calibration"], fpr_cap
        )
        candidates[name] = {
            "threshold": threshold,
            "calibration": target_metrics(
                roles["calibration"]["labels"][:, ISRJ_INDEX], probabilities["calibration"], threshold
            ),
            "tune": target_metrics(roles["tune"]["labels"][:, ISRJ_INDEX], probabilities["tune"], threshold),
        }

    base = {role: payload["probabilities"][:, ISRJ_INDEX] for role, payload in roles.items()}
    register("unet_classifier", base)
    for name in models:
        register(name, {role: physics[role][name] for role in roles})

    early_target = roles["early"]["labels"][:, ISRJ_INDEX]
    for name in models:
        stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
        early_matrix = np.column_stack((clipped_logit(base["early"]), clipped_logit(physics["early"][name])))
        stacker.fit(early_matrix, early_target)
        stacked = {}
        for role in roles:
            matrix = np.column_stack((clipped_logit(base[role]), clipped_logit(physics[role][name])))
            stacked[role] = stacker.predict_proba(matrix)[:, 1]
        register(f"stack_{name}", stacked)

    all_names = sorted(models)
    multi_stacker = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    early_matrix = np.column_stack(
        [clipped_logit(base["early"])] + [clipped_logit(physics["early"][name]) for name in all_names]
    )
    multi_stacker.fit(early_matrix, early_target)
    stacked = {}
    for role in roles:
        matrix = np.column_stack(
            [clipped_logit(base[role])] + [clipped_logit(physics[role][name]) for name in all_names]
        )
        stacked[role] = multi_stacker.predict_proba(matrix)[:, 1]
    register("stack_all_periodicity", stacked)
    return candidates


def select_family(seed_results: list[dict[str, object]], fpr_cap: float) -> tuple[str, list[dict[str, object]]]:
    names = sorted(seed_results[0]["candidates"])
    rows = []
    for name in names:
        metrics = [result["candidates"][name]["tune"] for result in seed_results]
        baseline = [result["candidates"]["unet_classifier"]["tune"] for result in seed_results]
        row = {
            "candidate": name,
            "recall_mean": float(np.mean([value["recall"] for value in metrics])),
            "recall_std": float(np.std([value["recall"] for value in metrics])),
            "fpr_mean": float(np.mean([value["fpr"] for value in metrics])),
            "aupr_mean": float(np.mean([value["aupr"] for value in metrics])),
            "delta_recall_mean": float(np.mean([value["recall"] - base["recall"] for value, base in zip(metrics, baseline)])),
            "delta_aupr_mean": float(np.mean([value["aupr"] - base["aupr"] for value, base in zip(metrics, baseline)])),
            "improved_seeds": int(sum(value["recall"] > base["recall"] for value, base in zip(metrics, baseline))),
            "max_seed_fpr": float(max(value["fpr"] for value in metrics)),
        }
        row["eligible"] = bool(
            row["fpr_mean"] <= fpr_cap + 1e-12
            and row["delta_aupr_mean"] >= -0.01
            and row["max_seed_fpr"] <= fpr_cap + 0.02
        )
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["recall_mean"], row["aupr_mean"], -row["fpr_mean"]))
    return str(selected["candidate"]), rows


def summarize_results(results: list[dict[str, object]], fpr_cap: float) -> dict[str, object]:
    selected, aggregate = select_family(results, fpr_cap)
    selected_row = next(row for row in aggregate if row["candidate"] == selected)
    promotion = bool(
        selected_row["recall_mean"] >= 0.70
        and selected_row["fpr_mean"] <= fpr_cap
        and selected_row["delta_recall_mean"] >= 0.05
        and selected_row["improved_seeds"] == len(results)
    )
    return {
        "selected_candidate": selected,
        "selection_role": "tune",
        "aggregate": aggregate,
        "promotion_gate": {
            "requirements": "mean recall >=0.70, mean FPR <=0.10, recall gain >=0.05, 3/3 seeds improve",
            "passed": promotion,
        },
        "confirm_used": False,
        "locked_test_used": False,
    }


def combined_system_metrics(
    results: list[dict[str, object]], separator_root: Path, selected: str
) -> dict[str, object]:
    per_seed = []
    for result in results:
        seed = int(result["seed"])
        separator = json.loads(
            (separator_root / "unet_wave" / f"seed_{seed}" / "result.json").read_text(encoding="utf-8")
        )["tune"]
        expert = result["candidates"][selected]["tune"]
        per_class = separator["per_class"]
        per_seed.append({
            "seed": seed,
            "macro_aupr": float(np.mean([per_class[0]["aupr"], per_class[1]["aupr"], expert["aupr"]])),
            "macro_recall": float(np.mean([per_class[0]["recall"], per_class[1]["recall"], expert["recall"]])),
            "macro_fpr": float(np.mean([per_class[0]["fpr"], per_class[1]["fpr"], expert["fpr"]])),
            "baseline_macro_aupr": float(separator["macro_aupr"]),
            "baseline_macro_recall": float(separator["macro_recall"]),
            "baseline_macro_fpr": float(separator["macro_fpr"]),
        })
    keys = (
        "macro_aupr", "macro_recall", "macro_fpr",
        "baseline_macro_aupr", "baseline_macro_recall", "baseline_macro_fpr",
    )
    return {
        "per_seed": per_seed,
        "mean": {key: float(np.mean([row[key] for row in per_seed])) for key in keys},
        "metric_role": "tune",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--output-root", default="outputs/isrj_separator_physics_head_20260725")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--fpr-cap", type=float, default=0.10)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.aggregate_only:
        results = [
            json.loads((output_root / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))
            for seed in args.seeds
        ]
        summary = summarize_results(results, args.fpr_cap)
        summary["combined_system"] = combined_system_metrics(
            results, Path(args.separator_root), summary["selected_candidate"]
        )
        (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=False)
    run_root = Path(args.run_root)
    role_file = np.load(run_root / "role_indices.npz")
    role_names = ("train", "early", "calibration", "tune")
    indices = {name: role_file[name].astype(np.int64) for name in role_names}
    files = discover_power_train_files(args.power_root)
    _, labels = read_identity_metadata(files[10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    receipt = {
        "purpose": "freeze the selected unet_wave separator and add an ISRJ periodicity expert",
        "roles": {name: {"count": len(value), "sha256": sha256_indices(value)} for name, value in indices.items()},
        "fit_roles": {"physics": "train", "stacker": "early", "threshold": "calibration", "selection": "tune"},
        "confirm_used": False,
        "locked_test_used": False,
        "governance": "adaptive development after confirm was previously inspected; tune-only evidence",
        "device": str(device),
    }
    (output_root / "protocol.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    results = []
    for seed in args.seeds:
        set_seed(seed)
        model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
        checkpoint_path = Path(args.separator_root) / "unet_wave" / f"seed_{seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        role_payloads = {}
        datasets = {}
        for role in role_names:
            dataset = SweptIsolationDataset(files[10], files[50], labels, indices[role], "raw_10db")
            datasets[role] = dataset
            role_payloads[role] = extract_role(model, dataset, args.batch_size, seed, device)
            if not np.array_equal(role_payloads[role]["sample_indices"], indices[role]):
                raise RuntimeError(f"sample order changed for role {role}")
        candidates = build_candidates(role_payloads, seed, args.fpr_cap)
        result = {
            "seed": seed,
            "separator_checkpoint": str(checkpoint_path),
            "separator_best_epoch": int(checkpoint["epoch"]),
            "candidates": candidates,
        }
        results.append(result)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, "tune": {name: row["tune"] for name, row in candidates.items()}}), flush=True)
        for dataset in datasets.values():
            dataset.close()
        del model, role_payloads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_results(results, args.fpr_cap)
    summary["combined_system"] = combined_system_metrics(
        results, Path(args.separator_root), summary["selected_candidate"]
    )
    (output_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
