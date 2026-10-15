"""Post-hoc diagnostics for the preregistered PAM-IQ V2 structure screen.

This script only reads development-role predictions and checkpoints.  It never
opens the locked test split.  The output is intended to explain whether a model
fails because its residual gates grow excessively or because weak-positive
scores remain entangled with negative scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VARIANTS = (
    "B1_WideCapacity",
    "P0_PAMIQ",
    "D1_DualPool",
    "Q1_DualQuery",
)
SEEDS = (2041, 2042, 2043)


def _component_power(values79: np.ndarray, class_names: list[str]) -> np.ndarray:
    values = np.asarray(values79, dtype=np.float32)
    return np.stack(
        [values[:, get_parameter_specs(name)[0].column_index] for name in class_names],
        axis=1,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
    }


def _checkpoint_diagnostics(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get(
        "model_state_dict", checkpoint.get("state_dict", checkpoint)
    )
    result: dict[str, object] = {}
    for key, tensor in state.items():
        if "attention.scale" not in key and "residual_scale" not in key:
            continue
        raw = tensor.detach().float().reshape(-1)
        result[key] = {
            "raw": raw.tolist(),
            "effective_tanh": torch.tanh(raw).tolist(),
        }
    if "backbone.classifier.query_weight" in state:
        weight = state["backbone.classifier.query_weight"].detach().float()
        scale = torch.tanh(
            state["backbone.classifier.residual_scale"].detach().float()
        )
        norms = torch.linalg.vector_norm(weight, dim=1)
        result["query_weight_l2_per_label"] = norms.tolist()
        result["effective_query_l2_per_label"] = (norms * scale).tolist()
    return result


def _score_diagnostics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    weak_mask: np.ndarray,
    primary_thresholds: np.ndarray,
    f2_thresholds: np.ndarray,
    class_names: list[str],
) -> dict:
    positive = targets.astype(bool)
    weak_positive = positive & weak_mask
    strong_positive = positive & ~weak_mask
    negative = ~positive

    weak_scores = probabilities[weak_positive]
    negative_scores = probabilities[negative]
    weak_negative_scores = np.concatenate([weak_scores, negative_scores])
    weak_negative_targets = np.concatenate(
        [np.ones(weak_scores.size), np.zeros(negative_scores.size)]
    )
    result = {
        "all_labels": {
            "weak_positive": _distribution(weak_scores),
            "strong_positive": _distribution(probabilities[strong_positive]),
            "negative": _distribution(negative_scores),
            "weak_vs_negative_auroc": float(
                roc_auc_score(weak_negative_targets, weak_negative_scores)
            ),
            "weak_vs_negative_aupr": float(
                average_precision_score(weak_negative_targets, weak_negative_scores)
            ),
            "primary_weak_recall": float(
                (probabilities >= primary_thresholds)[weak_positive].mean()
            ),
            "f2_weak_recall": float(
                (probabilities >= f2_thresholds)[weak_positive].mean()
            ),
            "primary_negative_fpr": float(
                (probabilities >= primary_thresholds)[negative].mean()
            ),
            "f2_negative_fpr": float(
                (probabilities >= f2_thresholds)[negative].mean()
            ),
        },
        "per_label": {},
    }
    for label_index, class_name in enumerate(class_names):
        label_weak = weak_positive[:, label_index]
        label_strong = strong_positive[:, label_index]
        label_negative = negative[:, label_index]
        label_weak_scores = probabilities[label_weak, label_index]
        label_negative_scores = probabilities[label_negative, label_index]
        label_binary_scores = np.concatenate(
            [label_weak_scores, label_negative_scores]
        )
        label_binary_targets = np.concatenate(
            [np.ones(label_weak_scores.size), np.zeros(label_negative_scores.size)]
        )
        result["per_label"][class_name] = {
            "weak_positive": _distribution(label_weak_scores),
            "strong_positive": _distribution(
                probabilities[label_strong, label_index]
            ),
            "negative": _distribution(label_negative_scores),
            "weak_vs_negative_auroc": (
                float(roc_auc_score(label_binary_targets, label_binary_scores))
                if label_weak.any() and label_negative.any()
                else float("nan")
            ),
            "weak_vs_negative_aupr": (
                float(
                    average_precision_score(label_binary_targets, label_binary_scores)
                )
                if label_weak.any() and label_negative.any()
                else float("nan")
            ),
            "primary_threshold": float(primary_thresholds[label_index]),
            "f2_threshold": float(f2_thresholds[label_index]),
            "primary_weak_recall": (
                float(
                    (probabilities[label_weak, label_index]
                    >= primary_thresholds[label_index]).mean()
                )
                if label_weak.any()
                else float("nan")
            ),
            "f2_weak_recall": (
                float(
                    (probabilities[label_weak, label_index]
                    >= f2_thresholds[label_index]).mean()
                )
                if label_weak.any()
                else float("nan")
            ),
            "primary_negative_fpr": float(
                (label_negative_scores >= primary_thresholds[label_index]).mean()
            ),
            "f2_negative_fpr": float(
                (label_negative_scores >= f2_thresholds[label_index]).mean()
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_pam_iq_v2_structure_screen",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = _data_path(config)
    class_names = ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]
    labels = select_labels(read_labels(data_path), "six")
    values79, _ = read_parameters(data_path)
    component_power = _component_power(values79, class_names)
    full_weak_mask = compute_weak_component_mask(
        labels, component_power, args.weak_threshold_db
    )

    output: dict[str, object] = {
        "analysis_role": "development_post_hoc_diagnostic",
        "locked_test_used": False,
        "weak_threshold_db": args.weak_threshold_db,
        "variants": {},
    }
    for variant in VARIANTS:
        variant_result: dict[str, object] = {}
        for seed in SEEDS:
            run_dir = args.screen_root / "runs" / variant / f"seed_{seed}"
            predictions = np.load(run_dir / "predictions.npz")
            sample_ids = predictions["sample_ids"].astype(np.int64)
            targets = predictions["targets"].astype(np.uint8)
            if not np.array_equal(targets, labels[sample_ids]):
                raise RuntimeError(f"target/sample alignment failed: {variant} {seed}")
            thresholds = json.loads(
                (run_dir / "thresholds.json").read_text(encoding="utf-8")
            )
            variant_result[str(seed)] = {
                "checkpoint": _checkpoint_diagnostics(run_dir / "best_model.pth"),
                "scores": _score_diagnostics(
                    predictions["probabilities"],
                    targets,
                    full_weak_mask[sample_ids],
                    np.asarray(thresholds["per_label"], dtype=np.float64),
                    np.asarray(thresholds["modes"]["per_label_f2"], dtype=np.float64),
                    class_names,
                ),
            }
        output["variants"][variant] = variant_result

    output_path = args.screen_root / "route_a_pam_iq_v2_score_diagnostics.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
