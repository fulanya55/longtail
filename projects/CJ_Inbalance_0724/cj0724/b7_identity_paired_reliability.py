"""B7: identity-paired multi-power reliability evidence screen for ISRJ."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from cj0724.b1_physics_sufficiency import metrics_from_probabilities
from cj0724.b2r_mft_reliability import bounded_three_state, reliability_features
from cj0724.dual_axis import POWER_LEVELS, audit_power_alignment
from cj0724.route_b_mechanism_router import SEEDS, baseline_root
from cj0724.shared_physics_frontend import extract_shared_features


ISRJ_INDEX = 5
MFT_INDEX = 3
VFT_INDEX = 4
TRAIN_POWERS = (10, 15, 20)
TEACHER_POWER = 30
VARIANTS = (
    "low_only_supervised",
    "unpaired_views",
    "all_pair_invariance",
    "pair_type_aware_reliability",
    "explicit_JNR_shortcut_control",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def logit(probability: torch.Tensor) -> torch.Tensor:
    clipped = probability.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(clipped) - torch.log1p(-clipped)


class BoundedReliabilityResidual(nn.Module):
    """Small bounded correction; high-power views are never inference inputs."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, residual_bound: float = 1.5,
                 include_power: bool = False) -> None:
        super().__init__()
        if residual_bound <= 0:
            raise ValueError("residual_bound must be positive")
        self.include_power = bool(include_power)
        self.residual_bound = float(residual_bound)
        input_dim = int(feature_dim) + 1 + int(self.include_power)
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, reference_probability: torch.Tensor,
                power: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        reference_logit = logit(reference_probability.reshape(-1))
        columns = [features, reference_logit[:, None]]
        if self.include_power:
            if power is None:
                raise ValueError("power is required when include_power=True")
            columns.append(((power.reshape(-1, 1) - 15.0) / 5.0).to(features.dtype))
        residual = self.residual_bound * torch.tanh(self.network(torch.cat(columns, dim=1)).reshape(-1))
        return reference_logit.detach() + residual, residual


def label_preserving_unpaired_teacher(
    teacher: np.ndarray, labels: np.ndarray, seed: int,
) -> np.ndarray:
    """Break identity pairing while preserving positive/negative teacher distributions."""
    values = np.asarray(teacher, dtype=np.float32)
    target = np.asarray(labels).astype(bool)
    output = values.copy()
    rng = np.random.default_rng(int(seed))
    for state in (False, True):
        indices = np.flatnonzero(target == state)
        if len(indices) > 1:
            shift = int(rng.integers(1, len(indices)))
            output[indices] = values[np.roll(indices, shift)]
    return output


def reliability_loss(
    variant: str,
    output_logit: torch.Tensor,
    residual: torch.Tensor,
    reference_probability: torch.Tensor,
    teacher_probability: torch.Tensor,
    labels: torch.Tensor,
    hard_negative: torch.Tensor,
    teacher_threshold: float,
    distillation_weight: float = 0.35,
    teacher_margin: float = 0.05,
    max_teacher_delta: float = 1.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = labels.float().reshape(-1)
    reference_logit = logit(reference_probability.reshape(-1)).detach()
    teacher_logit = logit(teacher_probability.reshape(-1)).detach()
    positive_count = target.sum().clamp_min(1.0)
    negative_count = (1.0 - target).sum().clamp_min(1.0)
    positive_weight = (negative_count / positive_count).clamp(1.0, 5.0)
    bce = F.binary_cross_entropy_with_logits(output_logit, target, reduction="none",
                                             pos_weight=positive_weight)
    sample_weight = torch.where(hard_negative.bool(), torch.full_like(bce, 1.5), torch.ones_like(bce))
    base = (bce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
    zero = output_logit.sum() * 0.0
    distillation = zero
    eligible_count = 0
    if variant == "all_pair_invariance":
        bounded_teacher = torch.maximum(
            torch.minimum(teacher_logit, reference_logit + max_teacher_delta),
            reference_logit - max_teacher_delta,
        )
        distillation = F.smooth_l1_loss(output_logit, bounded_teacher)
        eligible_count = int(len(target))
    elif variant in ("unpaired_views", "pair_type_aware_reliability", "explicit_JNR_shortcut_control"):
        eligible = (
            (target > 0.5)
            & (teacher_probability.reshape(-1) >= float(teacher_threshold))
            & (teacher_probability.reshape(-1) >= reference_probability.reshape(-1) + float(teacher_margin))
        )
        eligible_count = int(eligible.sum().item())
        if eligible.any():
            bounded_teacher = torch.minimum(
                teacher_logit[eligible], reference_logit[eligible] + max_teacher_delta
            )
            distillation = F.smooth_l1_loss(output_logit[eligible], bounded_teacher)
    elif variant != "low_only_supervised":
        raise ValueError(f"Unknown variant: {variant}")
    penalty = residual.square().mean()
    total = base + float(distillation_weight) * distillation + 0.01 * penalty
    return total, {
        "base": base.detach(), "distillation": distillation.detach(),
        "residual": penalty.detach(), "eligible_count": torch.tensor(eligible_count),
    }


def binary_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    target = np.asarray(labels).astype(bool)
    prediction = np.asarray(probability) >= float(threshold)
    return {
        "aupr": float(average_precision_score(target, probability)),
        "recall": float(prediction[target].mean()),
        "fpr": float(prediction[~target].mean()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    possible = []
    for mask in range(1 << len(values)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(values))])
        possible.append(abs(float((values * signs).mean())))
    return float(np.mean(np.asarray(possible) >= observed - 1e-15))


def train_variant(
    variant: str, seed: int, train: dict[str, np.ndarray], early: dict[str, np.ndarray],
    teacher_threshold: float, args: argparse.Namespace, device: torch.device,
) -> tuple[BoundedReliabilityResidual, list[dict[str, float]], int]:
    include_power = variant == "explicit_JNR_shortcut_control"
    model = BoundedReliabilityResidual(
        train["features"].shape[1], args.hidden_dim, args.residual_bound, include_power
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    teacher = train["teacher"].copy()
    if variant == "unpaired_views":
        teacher = label_preserving_unpaired_teacher(teacher, train["target"], seed + 7001)
    dataset = TensorDataset(
        torch.from_numpy(train["features"]), torch.from_numpy(train["reference"]),
        torch.from_numpy(teacher), torch.from_numpy(train["target"]),
        torch.from_numpy(train["hard_negative"]), torch.from_numpy(train["power"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    early_features = torch.from_numpy(early["features"]).to(device)
    early_reference = torch.from_numpy(early["reference"]).to(device)
    early_power = torch.from_numpy(early["power"]).to(device)
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "base": 0.0, "distillation": 0.0, "residual": 0.0}
        eligible = samples = 0
        for features, reference, batch_teacher, target, hard_negative, power in loader:
            features, reference = features.to(device), reference.to(device)
            batch_teacher, target = batch_teacher.to(device), target.to(device)
            hard_negative, power = hard_negative.to(device), power.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_logit, residual = model(features, reference, power if include_power else None)
            loss, parts = reliability_loss(
                variant, output_logit, residual, reference, batch_teacher, target, hard_negative,
                teacher_threshold, args.distillation_weight, args.teacher_margin,
                args.residual_bound,
            )
            loss.backward()
            optimizer.step()
            count = len(target)
            totals["loss"] += float(loss.detach()) * count
            for key in ("base", "distillation", "residual"):
                totals[key] += float(parts[key]) * count
            eligible += int(parts["eligible_count"])
            samples += count
        model.eval()
        with torch.no_grad():
            early_logit, early_residual = model(
                early_features, early_reference, early_power if include_power else None
            )
            probability = torch.sigmoid(early_logit).cpu().numpy()
        metric = binary_metrics(early["target"], probability, teacher_threshold)
        hard_fpr = float((probability[early["hard_negative"].astype(bool)] >= teacher_threshold).mean())
        row = {
            "epoch": float(epoch), "loss": totals["loss"] / samples,
            "base": totals["base"] / samples,
            "distillation": totals["distillation"] / samples,
            "residual": totals["residual"] / samples,
            "eligible_fraction": float(eligible / samples),
            "early_aupr": metric["aupr"], "early_fpr": metric["fpr"],
            "early_hard_negative_fpr": hard_fpr,
            "early_mean_abs_residual": float(np.abs(early_residual.cpu().numpy()).mean()),
        }
        history.append(row)
        key = (metric["aupr"], -hard_fpr, -row["loss"])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    return model, history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/b7_identity_paired_reliability_20260726"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--residual-bound", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--teacher-margin", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    seed_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    identity_receipts: list[dict[str, object]] = []
    for seed in SEEDS:
        tick_seed = time.perf_counter()
        set_seed(seed)
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir()
        baseline = baseline_root(seed)
        route_root = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
        repair_root = Path(f"outputs/b2rc_recall_constrained_20260725/seed_{seed}")
        b2_root = Path(f"outputs/b2_power_boundary_20260725/seed_{seed}")
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        files = {power: Path(audit["levels"][str(power)]["path"]) for power in POWER_LEVELS}
        alignment = audit_power_alignment(files)
        split = np.load(baseline / "split_indices.npz")
        manifest = json.loads((baseline / "longtail_manifest.json").read_text(encoding="utf-8"))
        roles = {
            "train": np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64)),
            "early": np.sort(split["early_stop"].astype(np.int64)),
            "development": np.sort(split["development"].astype(np.int64)),
        }
        if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
            raise RuntimeError("identity roles overlap")
        with h5py.File(files[10], "r") as handle:
            labels_all = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
            identity_all = np.rint(np.asarray(handle["receiver_noise_seed"][:, :]).reshape(-1)).astype(np.int64)
        identity_receipts.append({
            "seed": seed, "alignment_sha256": alignment["audit_sha256"],
            "train_indices_sha256": array_sha256(roles["train"]),
            "early_indices_sha256": array_sha256(roles["early"]),
            "development_indices_sha256": array_sha256(roles["development"]),
            "train_identity_sha256": array_sha256(identity_all[roles["train"]]),
            "early_identity_sha256": array_sha256(identity_all[roles["early"]]),
            "development_identity_sha256": array_sha256(identity_all[roles["development"]]),
        })
        all_indices = np.sort(np.concatenate(list(roles.values())))
        lookup = {int(index): position for position, index in enumerate(all_indices)}
        positions = {role: np.asarray([lookup[int(index)] for index in indices], dtype=np.int64)
                     for role, indices in roles.items()}
        feature_started = time.perf_counter()
        by_power: dict[int, dict[str, np.ndarray]] = {}
        for power in (*TRAIN_POWERS, TEACHER_POWER):
            extracted = extract_shared_features(files[power], all_indices, device, args.feature_batch_size)
            by_power[power] = extracted
        feature_seconds = time.perf_counter() - feature_started
        reference_model = joblib.load(route_root / "isrj_physics_classifier.joblib")
        teacher_threshold = float(np.load(b2_root / "power_10dB.npz")["thresholds_B2-Hard"][ISRJ_INDEX])
        teacher_by_role = {}
        for role in roles:
            teacher_features = by_power[TEACHER_POWER]["isrj_periodicity"][positions[role]]
            teacher_by_role[role] = reference_model.predict_proba(teacher_features)[:, 1].astype(np.float32)
        train_parts = {key: [] for key in ("features", "reference", "teacher", "target", "hard_negative", "power")}
        for power in TRAIN_POWERS:
            features = by_power[power]["isrj_periodicity"][positions["train"]]
            target = labels_all[roles["train"], ISRJ_INDEX].astype(np.float32)
            train_parts["features"].append(features)
            train_parts["reference"].append(reference_model.predict_proba(features)[:, 1].astype(np.float32))
            train_parts["teacher"].append(teacher_by_role["train"])
            train_parts["target"].append(target)
            train_parts["hard_negative"].append(
                ((target < 0.5) & (labels_all[roles["train"], MFT_INDEX].astype(bool)
                                  | labels_all[roles["train"], VFT_INDEX].astype(bool))).astype(np.float32)
            )
            train_parts["power"].append(np.full(len(target), power, dtype=np.float32))
        train_data = {key: np.concatenate(value) for key, value in train_parts.items()}
        early_features = by_power[10]["isrj_periodicity"][positions["early"]]
        early_target = labels_all[roles["early"], ISRJ_INDEX].astype(np.float32)
        early_data = {
            "features": early_features,
            "reference": reference_model.predict_proba(early_features)[:, 1].astype(np.float32),
            "teacher": teacher_by_role["early"], "target": early_target,
            "hard_negative": ((early_target < 0.5)
                              & (labels_all[roles["early"], MFT_INDEX].astype(bool)
                                 | labels_all[roles["early"], VFT_INDEX].astype(bool))).astype(np.float32),
            "power": np.full(len(early_target), 10, dtype=np.float32),
        }
        mean = train_data["features"].mean(axis=0, keepdims=True)
        scale = train_data["features"].std(axis=0, keepdims=True)
        scale = np.maximum(scale, 1e-5)
        train_data["features"] = np.ascontiguousarray((train_data["features"] - mean) / scale, dtype=np.float32)
        early_data["features"] = np.ascontiguousarray((early_data["features"] - mean) / scale, dtype=np.float32)
        development_features_raw = by_power[10]["isrj_periodicity"][positions["development"]]
        development_features = np.ascontiguousarray((development_features_raw - mean) / scale, dtype=np.float32)
        development_reference = reference_model.predict_proba(development_features_raw)[:, 1].astype(np.float32)
        saved = np.load(b2_root / "power_10dB.npz")
        labels = saved["labels"].astype(np.uint8)
        if not np.array_equal(labels, labels_all[roles["development"]]):
            raise RuntimeError("saved development labels do not match identity role")
        max_reference_difference = float(np.max(np.abs(
            development_reference - saved["probabilities_B2-Hard"][:, ISRJ_INDEX]
        )))
        combined_model = joblib.load(route_root / "mft_physics_classifier.joblib")
        delay_model = joblib.load(repair_root / "mft_delay_auxiliary.joblib")
        mft_router = joblib.load(repair_root / "mft_three_state_router.joblib")
        reference_stats = np.load(repair_root / "reference_stats.npz")
        repair_protocol = json.loads((repair_root / "protocol.json").read_text(encoding="utf-8"))
        combined = by_power[10]["combined"][positions["development"]]
        delay = by_power[10]["mft_delay"][positions["development"]]
        old_mft = combined_model.predict_proba(combined)[:, 1]
        delay_probability = delay_model.predict_proba(delay)[:, 1]
        router_feature = reliability_features(
            saved["probabilities_E5"][:, MFT_INDEX], old_mft, delay_probability, combined,
            reference_stats["median"], reference_stats["scale"],
        )
        raw_action = mft_router.predict_proba(router_feature)
        actions = np.zeros((len(raw_action), 3), dtype=np.float64)
        actions[:, mft_router.classes_.astype(int)] = raw_action
        repaired_mft = bounded_three_state(
            saved["probabilities_E5"][:, MFT_INDEX], old_mft, actions,
            float(repair_protocol["alpha"]), float(repair_protocol["confidence"]),
        )
        reference_full = saved["probabilities_B2-Hard"].astype(np.float64).copy()
        reference_full[:, MFT_INDEX] = repaired_mft
        reference_full[:, ISRJ_INDEX] = development_reference
        thresholds = saved["thresholds_E5"].astype(np.float64)
        thresholds[ISRJ_INDEX] = teacher_threshold
        reference_metrics = metrics_from_probabilities(labels, reference_full, thresholds, saved["weak_mask"])
        reference_binary = binary_metrics(labels[:, ISRJ_INDEX], development_reference, teacher_threshold)
        hard_negative_mask = (~labels[:, ISRJ_INDEX].astype(bool)) & (
            labels[:, MFT_INDEX].astype(bool) | labels[:, VFT_INDEX].astype(bool)
        )
        fit_started = time.perf_counter()
        seed_variant_rows = []
        for variant in VARIANTS:
            model, history, best_epoch = train_variant(
                variant, seed, train_data, early_data, teacher_threshold, args, device
            )
            include_power = variant == "explicit_JNR_shortcut_control"
            model.eval()
            with torch.no_grad():
                output_logit, residual = model(
                    torch.from_numpy(development_features).to(device),
                    torch.from_numpy(development_reference).to(device),
                    torch.full((len(labels),), 10.0, device=device) if include_power else None,
                )
                candidate_isrj = torch.sigmoid(output_logit).cpu().numpy()
                residual_np = residual.cpu().numpy()
            candidate_full = reference_full.copy()
            candidate_full[:, ISRJ_INDEX] = candidate_isrj
            candidate_metrics = metrics_from_probabilities(labels, candidate_full, thresholds, saved["weak_mask"])
            target_metric = binary_metrics(labels[:, ISRJ_INDEX], candidate_isrj, teacher_threshold)
            row = {
                "seed": seed, "variant": variant, "best_epoch": best_epoch,
                "reference_ISRJ_aupr": reference_binary["aupr"],
                "candidate_ISRJ_aupr": target_metric["aupr"],
                "delta_ISRJ_aupr": target_metric["aupr"] - reference_binary["aupr"],
                "reference_ISRJ_fpr": reference_binary["fpr"],
                "candidate_ISRJ_fpr": target_metric["fpr"],
                "delta_ISRJ_fpr": target_metric["fpr"] - reference_binary["fpr"],
                "reference_ISRJ_recall": reference_binary["recall"],
                "candidate_ISRJ_recall": target_metric["recall"],
                "ISRJ_recall_drop": reference_binary["recall"] - target_metric["recall"],
                "reference_macro_aupr": reference_metrics["macro_aupr"],
                "candidate_macro_aupr": candidate_metrics["macro_aupr"],
                "delta_macro_aupr": candidate_metrics["macro_aupr"] - reference_metrics["macro_aupr"],
                "reference_hamming": reference_metrics["hamming_loss"],
                "candidate_hamming": candidate_metrics["hamming_loss"],
                "delta_hamming": candidate_metrics["hamming_loss"] - reference_metrics["hamming_loss"],
                "reference_hard_negative_fpr": float((development_reference[hard_negative_mask] >= teacher_threshold).mean()),
                "candidate_hard_negative_fpr": float((candidate_isrj[hard_negative_mask] >= teacher_threshold).mean()),
                "mean_positive_score_shift": float((candidate_isrj - development_reference)[labels[:, ISRJ_INDEX].astype(bool)].mean()),
                "mean_hard_negative_score_shift": float((candidate_isrj - development_reference)[hard_negative_mask].mean()),
                "score_separation_delta": float(
                    (candidate_isrj[labels[:, ISRJ_INDEX].astype(bool)].mean() - candidate_isrj[hard_negative_mask].mean())
                    - (development_reference[labels[:, ISRJ_INDEX].astype(bool)].mean()
                       - development_reference[hard_negative_mask].mean())
                ),
                "mean_abs_logit_residual": float(np.abs(residual_np).mean()),
                "max_abs_logit_residual": float(np.abs(residual_np).max()),
                "explicit_JNR_input": include_power,
            }
            run_rows.append(row)
            seed_variant_rows.append(row)
            variant_root = seed_root / variant
            variant_root.mkdir()
            torch.save({
                "model_state": model.state_dict(), "mean": mean.astype(np.float32),
                "scale": scale.astype(np.float32), "best_epoch": best_epoch,
                "variant": variant, "teacher_power_dB": TEACHER_POWER,
                "student_train_powers_dB": TRAIN_POWERS,
            }, variant_root / "best.pt")
            (variant_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({"seed": seed, "variant": variant,
                              "delta_ISRJ_aupr": row["delta_ISRJ_aupr"],
                              "delta_hamming": row["delta_hamming"],
                              "delta_ISRJ_fpr": row["delta_ISRJ_fpr"],
                              "recall_drop": row["ISRJ_recall_drop"]}), flush=True)
        fit_seconds = time.perf_counter() - fit_started
        primary = next(row for row in seed_variant_rows if row["variant"] == "pair_type_aware_reliability")
        unpaired = next(row for row in seed_variant_rows if row["variant"] == "unpaired_views")
        low_only = next(row for row in seed_variant_rows if row["variant"] == "low_only_supervised")
        seed_rows.append({
            "seed": seed, **{key: primary[key] for key in (
                "delta_ISRJ_aupr", "delta_macro_aupr", "delta_hamming", "delta_ISRJ_fpr",
                "ISRJ_recall_drop", "score_separation_delta", "mean_positive_score_shift",
                "mean_hard_negative_score_shift",
            )},
            "paired_advantage_vs_unpaired": primary["candidate_ISRJ_aupr"] - unpaired["candidate_ISRJ_aupr"],
            "paired_advantage_vs_low_only": primary["candidate_ISRJ_aupr"] - low_only["candidate_ISRJ_aupr"],
            "max_reference_probability_difference": max_reference_difference,
        })
        cost_rows.append({
            "seed": seed, "feature_seconds": feature_seconds, "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - tick_seed,
            "primary_model_bytes": (seed_root / "pair_type_aware_reliability" / "best.pt").stat().st_size,
        })
        (seed_root / "protocol.json").write_text(json.dumps({
            "seed": seed, "teacher_power_dB": TEACHER_POWER,
            "student_train_powers_dB": list(TRAIN_POWERS), "evaluation_power_dB": 10,
            "teacher_threshold": teacher_threshold, "identity_key": "receiver_noise_seed",
            "thresholds_changed": False, "development_used_for_selection": False,
            "explicit_JNR_used_by_primary": False, "locked_test_used": False,
        }, indent=2), encoding="utf-8")
    means = {key: float(np.mean([row[key] for row in seed_rows])) for key in (
        "delta_ISRJ_aupr", "delta_macro_aupr", "delta_hamming", "delta_ISRJ_fpr",
        "ISRJ_recall_drop", "score_separation_delta", "mean_positive_score_shift",
        "mean_hard_negative_score_shift", "paired_advantage_vs_unpaired",
        "paired_advantage_vs_low_only",
    )}
    same_direction = sum(
        row["delta_ISRJ_aupr"] > 0 and row["delta_hamming"] <= 0 and row["delta_ISRJ_fpr"] <= 0
        for row in seed_rows
    )
    deltas = np.asarray([row["delta_ISRJ_aupr"] for row in seed_rows], dtype=np.float64)
    summary = {
        **means, "same_direction_seeds": int(same_direction), "seed_rows": seed_rows,
        "ISRJ_AUPR_exact_sign_flip_p_two_sided": _sign_flip_p(deltas),
    }
    gates = {
        "mean_10dB_ISRJ_AUPR_gain_at_least_0p015": means["delta_ISRJ_aupr"] >= 0.015,
        "mean_10dB_macro_AUPR_gain_at_least_0p0025": means["delta_macro_aupr"] >= 0.0025,
        "mean_10dB_hamming_not_increased": means["delta_hamming"] <= 0.0,
        "mean_10dB_ISRJ_FPR_not_increased": means["delta_ISRJ_fpr"] <= 0.0,
        "mean_10dB_ISRJ_recall_drop_at_most_0p005": means["ISRJ_recall_drop"] <= 0.005,
        "paired_AUPR_advantage_vs_unpaired_at_least_0p005": means["paired_advantage_vs_unpaired"] >= 0.005,
        "paired_AUPR_advantage_vs_low_only_at_least_0p005": means["paired_advantage_vs_low_only"] >= 0.005,
        "same_direction_at_least_4_seeds": same_direction >= 4,
        "identity_alignment_verified_all_seeds": len(identity_receipts) == len(SEEDS),
    }
    passed = bool(all(gates.values()))
    result = {
        "protocol": {
            "stage": "B7", "reference": "frozen B2-RC/B3-R", "seeds": list(SEEDS),
            "student_train_powers_dB": list(TRAIN_POWERS), "teacher_power_dB": TEACHER_POWER,
            "evaluation_power_dB": 10, "primary_candidate": "pair_type_aware_reliability",
            "inference_uses_high_power_view": False, "explicit_JNR_used_by_primary": False,
            "thresholds_changed": False, "development_used_for_selection": False,
            "locked_test_used": False,
        },
        "summary": summary, "gates": gates, "B7_passed": passed,
        "decision": "enter_B8" if passed else "stop_route_B_at_B2RC_B3R",
        "cost": {
            "mean_feature_seconds_per_seed": float(np.mean([row["feature_seconds"] for row in cost_rows])),
            "mean_fit_seconds_per_seed": float(np.mean([row["fit_seconds"] for row in cost_rows])),
            "mean_total_seconds_per_seed": float(np.mean([row["total_seconds"] for row in cost_rows])),
            "mean_primary_model_bytes": float(np.mean([row["primary_model_bytes"] for row in cost_rows])),
        },
        "locked_test_used": False,
    }
    _write_csv(args.output_root / "variant_runs.csv", run_rows)
    _write_csv(args.output_root / "seed_summary.csv", seed_rows)
    _write_csv(args.output_root / "cost.csv", cost_rows)
    (args.output_root / "identity_receipts.json").write_text(
        json.dumps(identity_receipts, indent=2), encoding="utf-8"
    )
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
