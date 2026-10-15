"""P5b1 hierarchy of in-range JNR physical-sufficiency probes.

This is a diagnostic-only study.  Oracle composition is deliberately supplied
to every non-constant probe, while nominal sweep power is retained exclusively
for balanced sampling and post-hoc reporting.  No probe prediction may be used
as a deployable route.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cj0724 import CLASS_NAMES
from cj0724.data import SIGNAL_LENGTH, complex_rms_normalize
from cj0724.dual_axis import combo_bits, read_identity_metadata
from cj0724.models import IQGroupedConvClassifier
from cj0724.multitask import build_multitask_model


POWERS = (10, 15, 20, 25, 30)
TARGET_CLASS_INDICES = (3, 4, 5)
TARGET_NAMES = ("MFT.JNR", "VFT.JNR", "ISRJ.JNR")
TARGET_COLUMNS = (27, 48, 52)
GLOBAL = "legacy_mlcnn_global"
ROUTED = "mlcnn_iq_grouped_task_routed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _power_files(root: Path, powers: tuple[int, ...] = POWERS) -> dict[int, Path]:
    files = {
        int(power): root / f"level_{index:02d}__jnr_{int(power)}dB" / "closedset_6class_train.mat"
        for index, power in enumerate(powers, start=1)
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P5b1 power files are missing: {missing}")
    return files


class PairedPowerOracleDataset(Dataset):
    """A role-specific, power-balanced view that never returns nominal power as input."""

    def __init__(self, files: dict[int, Path], role_indices: np.ndarray) -> None:
        self.files = files
        self.powers = tuple(sorted(files))
        self.indices = np.asarray(role_indices, dtype=np.int64)
        reference_seeds, reference_labels = read_identity_metadata(files[30])
        self.identities = reference_seeds
        self.labels = reference_labels.astype(np.float32)
        self.combination_ids = np.asarray(
            [int(combo_bits(row), 2) for row in reference_labels], dtype=np.int16
        )
        self.values: dict[int, np.ndarray] = {}
        self.masks: dict[int, np.ndarray] = {}
        for power, path in files.items():
            identities, labels = read_identity_metadata(path)
            if not np.array_equal(identities, self.identities) or not np.array_equal(labels, reference_labels):
                raise RuntimeError(f"Identity/label pairing changed at {power} dB")
            with h5py.File(path, "r") as handle:
                values = np.asarray(handle["parameter_values"][list(TARGET_COLUMNS), :], dtype=np.float32).T
                masks = np.asarray(handle["parameter_mask"][list(TARGET_COLUMNS), :], dtype=np.uint8).T.astype(bool)
            self.values[power] = values
            self.masks[power] = masks
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.powers) * self.indices.size

    def _handle(self, power: int) -> h5py.File:
        if power not in self._handles:
            self._handles[power] = h5py.File(self.files[power], "r")
        return self._handles[power]

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        power_index, local = divmod(int(position), self.indices.size)
        power = self.powers[power_index]
        sample_index = int(self.indices[local])
        raw = np.asarray(self._handle(power)["data_legacy"][:, sample_index], dtype=np.float32)
        iq = complex_rms_normalize(raw.reshape(2, SIGNAL_LENGTH))
        labels = self.labels[sample_index]
        valid = self.masks[power][sample_index] & labels[list(TARGET_CLASS_INDICES)].astype(bool)
        return {
            "iq": torch.from_numpy(iq),
            "composition": torch.from_numpy(labels.copy()),
            "target": torch.from_numpy(self.values[power][sample_index].copy()),
            "valid": torch.from_numpy(valid.astype(np.bool_)),
            "identity": int(self.identities[sample_index]),
            "power": int(power),
            "combination": int(self.combination_ids[sample_index]),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __del__(self) -> None:
        self.close()


def analytic_features(iq: torch.Tensor) -> torch.Tensor:
    """Fixed time, FFT and autocorrelation features; no nominal-power feature."""
    real, imag = iq[:, 0], iq[:, 1]
    magnitude = torch.sqrt(real.square() + imag.square() + 1.0e-8)
    time = torch.stack([
        real.mean(dim=1), imag.mean(dim=1), real.std(dim=1), imag.std(dim=1),
        magnitude.mean(dim=1), magnitude.std(dim=1), magnitude.amax(dim=1),
        (real * imag).mean(dim=1),
    ], dim=1)
    # The I/Q observation is complex, so this must be the full complex FFT.
    # `rfft` only accepts real tensors and would silently be conceptually wrong
    # even if the implementation coerced I/Q to a real channel.
    spectrum = torch.fft.fft(torch.complex(real, imag), dim=1).abs().square()
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    bins = spectrum.shape[1]
    edges = torch.linspace(0, bins, 9, device=iq.device).round().long()
    bands = torch.stack([
        spectrum[:, edges[index]:edges[index + 1]].sum(dim=1)
        for index in range(8)
    ], dim=1)
    frequency = torch.arange(bins, device=iq.device, dtype=iq.dtype)[None, :]
    centroid = (spectrum * frequency).sum(dim=1) / max(bins - 1, 1)
    entropy = -(spectrum * spectrum.clamp_min(1.0e-12).log()).sum(dim=1) / math.log(max(bins, 2))
    ac = []
    for lag in (1, 2, 4, 8, 16, 32, 64, 128):
        ac.append((real[:, :-lag] * real[:, lag:] + imag[:, :-lag] * imag[:, lag:]).mean(dim=1))
    return torch.cat([time, bands, centroid[:, None], entropy[:, None], torch.stack(ac, dim=1)], dim=1)


@dataclass
class FeatureRole:
    feature: np.ndarray
    composition: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    identities: np.ndarray
    power: np.ndarray
    combination: np.ndarray


def collect_role(
    dataset: PairedPowerOracleDataset, device: torch.device, extractor: nn.Module | None,
    extractor_kind: str | None, batch_size: int,
) -> FeatureRole:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    features, compositions, targets, valids, identities, powers, combinations = [], [], [], [], [], [], []
    if extractor is not None:
        extractor.eval()
    with torch.inference_mode():
        for batch in loader:
            iq = batch["iq"].to(device)
            if extractor_kind == "analytic":
                feature = analytic_features(iq)
            elif extractor_kind == "mlcnn":
                feature = extractor.classification_adapter(extractor.classification_backbone(iq))
            elif extractor_kind == "iqgrouped":
                feature = extractor.parameter_adapter(extractor.parameter_backbone(iq))
            else:
                raise ValueError(f"Unknown extractor kind {extractor_kind!r}")
            features.append(feature.float().cpu().numpy())
            compositions.append(batch["composition"].numpy())
            targets.append(batch["target"].numpy())
            valids.append(batch["valid"].numpy().astype(bool))
            identities.append(batch["identity"].numpy())
            powers.append(batch["power"].numpy())
            combinations.append(batch["combination"].numpy())
    return FeatureRole(
        feature=np.concatenate(features), composition=np.concatenate(compositions),
        target=np.concatenate(targets), valid=np.concatenate(valids),
        identities=np.concatenate(identities), power=np.concatenate(powers),
        combination=np.concatenate(combinations),
    )


def feature_matrix(role: FeatureRole, mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.concatenate([role.feature, role.composition], axis=1).astype(np.float32)
    if mean is None:
        mean = combined.mean(axis=0)
        scale = combined.std(axis=0).clip(min=1.0e-6)
    return ((combined - mean) / scale).astype(np.float32), mean, scale


def target_statistics(role: FeatureRole) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median, mean, scale = [], [], []
    for target_index in range(3):
        values = role.target[role.valid[:, target_index], target_index]
        median.append(float(np.median(values)))
        mean.append(float(np.mean(values)))
        scale.append(float(np.std(values).clip(min=1.0e-6)))
    return np.asarray(median, np.float32), np.asarray(mean, np.float32), np.asarray(scale, np.float32)


def metrics(
    prediction: np.ndarray, role: FeatureRole, scale: np.ndarray, baseline: np.ndarray | None = None,
    powers: tuple[int, ...] = POWERS,
) -> dict[str, object]:
    by_target: dict[str, object] = {}
    for target_index, name in enumerate(TARGET_NAMES):
        valid = role.valid[:, target_index]
        error = np.abs(prediction[valid, target_index] - role.target[valid, target_index]) / scale[target_index]
        values = role.target[valid, target_index]
        predicted = prediction[valid, target_index]
        correlation = float(pearsonr(values, predicted).statistic) if len(values) > 2 and np.std(predicted) > 0 else 0.0
        per_power = {}
        per_combo = {}
        for power in powers:
            selected = valid & (role.power == power)
            if selected.any():
                per_power[str(power)] = float(
                    np.abs(prediction[selected, target_index] - role.target[selected, target_index]).mean() / scale[target_index]
                )
        for combination in np.unique(role.combination):
            selected = valid & (role.combination == combination)
            if selected.any():
                per_combo[str(int(combination))] = float(
                    np.abs(prediction[selected, target_index] - role.target[selected, target_index]).mean() / scale[target_index]
                )
        by_target[name] = {
            "count": int(valid.sum()), "nmae": float(error.mean()), "pearson": correlation,
            "per_power_nmae": per_power, "per_combination_nmae": per_combo,
        }
    return {"by_target": by_target}


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.bool()
    # Invalid parameter entries are intentionally NaN.  Index before calling
    # the loss: calculating loss over the full tensor would propagate NaNs
    # into the autograd graph even though a later mask hides them numerically.
    return F.smooth_l1_loss(prediction[mask], target[mask]) if mask.any() else prediction.sum() * 0.0


class FeatureProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class TargetSpecificIQOracle(nn.Module):
    """Width-24 grouped-I/Q extractor and three composition-conditioned scalar heads."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        encoder = IQGroupedConvClassifier(num_classes=6, width=width, dropout=0.0)
        self.iq_paths = encoder.iq_paths
        self.fuse_iq = encoder.fuse_iq
        self.pool = encoder.pool
        dimension = width * 4 + len(CLASS_NAMES)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in TARGET_NAMES
        ])

    def forward(self, iq: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
        feature = self.pool(self.fuse_iq(self.iq_paths(iq))).squeeze(-1)
        conditioned = torch.cat([feature, composition], dim=1)
        return torch.cat([head(conditioned) for head in self.heads], dim=1)


def _feature_loader(role: FeatureRole, matrix: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(matrix), torch.from_numpy(role.target), torch.from_numpy(role.valid),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_feature_probe(
    train: FeatureRole, early: FeatureRole, confirm: FeatureRole, seed: int,
    target_mean: np.ndarray, target_scale: np.ndarray, epochs: int, patience: int,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_x, feature_mean, feature_scale = feature_matrix(train)
    early_x, _, _ = feature_matrix(early, feature_mean, feature_scale)
    confirm_x, _, _ = feature_matrix(confirm, feature_mean, feature_scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = FeatureProbe(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    train_loader = _feature_loader(train, train_x, 256, True)
    early_tensor = torch.from_numpy(early_x).to(device)
    early_target = torch.from_numpy((early.target - target_mean) / target_scale).to(device)
    early_valid = torch.from_numpy(early.valid).to(device)
    best_score, remaining, best_state = float("inf"), patience, None
    for epoch in range(1, epochs + 1):
        model.train()
        for feature, target, valid in train_loader:
            feature, target, valid = feature.to(device), target.to(device), valid.to(device)
            normalized = (target - torch.as_tensor(target_mean, device=device)) / torch.as_tensor(target_scale, device=device)
            loss = masked_smooth_l1(model(feature), normalized, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = float(masked_smooth_l1(model(early_tensor), early_target, early_valid).item())
        if score < best_score - 1.0e-7:
            best_score, remaining, best_state = score, patience, copy.deepcopy(model.state_dict())
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_dim": int(train_x.shape[1]), "feature_mean": feature_mean,
            "feature_scale": feature_scale, "target_mean": target_mean,
            "target_scale": target_scale, "seed": int(seed),
        }, checkpoint_path)
    model.eval()
    with torch.inference_mode():
        early_prediction = model(early_tensor).cpu().numpy() * target_scale + target_mean
        prediction = model(torch.from_numpy(confirm_x).to(device)).cpu().numpy() * target_scale + target_mean
    return early_prediction, prediction, {"early_normalized_smooth_l1": best_score, "epochs_run": epoch,
                                       "checkpoint": str(checkpoint_path) if checkpoint_path else None}


def ridge_prediction(train: FeatureRole, query: FeatureRole, target_mean: np.ndarray, target_scale: np.ndarray) -> np.ndarray:
    train_x, feature_mean, feature_scale = feature_matrix(train)
    query_x, _, _ = feature_matrix(query, feature_mean, feature_scale)
    result = np.zeros((query_x.shape[0], 3), dtype=np.float32)
    design = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float32)], axis=1)
    query_design = np.concatenate([query_x, np.ones((query_x.shape[0], 1), dtype=np.float32)], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float32) * 1.0
    regularizer[-1, -1] = 0.0
    for target_index in range(3):
        valid = train.valid[:, target_index]
        y = (train.target[valid, target_index] - target_mean[target_index]) / target_scale[target_index]
        coefficient = np.linalg.solve(
            design[valid].T @ design[valid] + regularizer,
            design[valid].T @ y,
        )
        result[:, target_index] = query_design @ coefficient * target_scale[target_index] + target_mean[target_index]
    return result


def train_iq_oracle(
    train: PairedPowerOracleDataset, early: PairedPowerOracleDataset, confirm: PairedPowerOracleDataset,
    seed: int, target_mean: np.ndarray, target_scale: np.ndarray, epochs: int, patience: int,
) -> tuple[np.ndarray, FeatureRole, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = TargetSpecificIQOracle(width=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    train_loader = DataLoader(train, batch_size=126, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    early_loader = DataLoader(early, batch_size=126, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    mean_tensor = torch.as_tensor(target_mean, device=device)
    scale_tensor = torch.as_tensor(target_scale, device=device)
    best_score, remaining, best_state = float("inf"), patience, None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            prediction = model(batch["iq"].to(device), batch["composition"].to(device))
            target = (batch["target"].to(device) - mean_tensor) / scale_tensor
            loss = masked_smooth_l1(prediction, target, batch["valid"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        scores = []
        with torch.inference_mode():
            for batch in early_loader:
                prediction = model(batch["iq"].to(device), batch["composition"].to(device))
                target = (batch["target"].to(device) - mean_tensor) / scale_tensor
                scores.append(float(masked_smooth_l1(prediction, target, batch["valid"].to(device)).item()))
        score = float(np.mean(scores))
        if score < best_score - 1.0e-7:
            best_score, remaining, best_state = score, patience, copy.deepcopy(model.state_dict())
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    loader = DataLoader(confirm, batch_size=126, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    prediction_parts = []
    role = collect_role(confirm, device, None, "analytic", 126)
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch["iq"].to(device), batch["composition"].to(device))
            prediction_parts.append((prediction.cpu().numpy() * target_scale + target_mean).astype(np.float32))
    return np.concatenate(prediction_parts), role, {"early_normalized_smooth_l1": best_score, "epochs_run": epoch}


def frozen_source_model(project: Path, source_seed: int) -> tuple[nn.Module, dict[str, object]]:
    run_root = project / "outputs/p4c_heterogeneous_route_variable_jnr_20260801/tuned_multitask_20260801_063924"
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    row = next(
        item for item in summary
        if item["model"] == ROUTED and int(item["seed"]) == int(source_seed)
        and item.get("checkpoint_objective") == "parameter"
    )
    model = build_multitask_model(
        ROUTED, protocol["width"], protocol["dropout"], protocol["head_dim"],
        protocol["parameter_hidden_dim"], protocol["legacy_attention"],
    )
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = project / checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"])
    return model, {"source_seed": int(source_seed), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}


def bootstrap_gain(
    baseline: np.ndarray, prediction: np.ndarray, role: FeatureRole, target_index: int,
    scale: np.ndarray, seed: int,
) -> dict[str, float | int]:
    valid = role.valid[:, target_index]
    gain = (
        np.abs(baseline[valid, target_index] - role.target[valid, target_index])
        - np.abs(prediction[valid, target_index] - role.target[valid, target_index])
    ) / scale[target_index]
    identities = role.identities[valid]
    unique = np.unique(identities)
    per_identity = np.asarray([gain[identities == identity].mean() for identity in unique])
    rng = np.random.default_rng(seed + target_index)
    bootstrap = np.asarray([
        per_identity[rng.integers(0, per_identity.size, per_identity.size)].mean()
        for _ in range(2000)
    ])
    return {
        "mean_nmae_gain": float(per_identity.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "identity_clusters": int(unique.size), "replicates": 2000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--role-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    project = Path.cwd()
    preregister_path = project / args.preregister
    protocol = json.loads(preregister_path.read_text(encoding="utf-8"))
    role_receipt = json.loads((project / args.role_receipt).read_text(encoding="utf-8"))
    if not role_receipt["all_roles_disjoint"] or not role_receipt["same_roles_required_at_all_powers"]:
        raise RuntimeError("P5b1 role receipt is invalid")
    roles = np.load(project / args.roles)
    files = _power_files(project / args.data_root)
    datasets = {role: PairedPowerOracleDataset(files, roles[role]) for role in ("train", "early", "confirm")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analytic_roles = {role: collect_role(dataset, device, None, "analytic", 126) for role, dataset in datasets.items()}
    constant, target_mean, target_scale = target_statistics(analytic_roles["train"])
    constant_prediction = np.broadcast_to(constant, analytic_roles["confirm"].target.shape).copy()
    constant_metrics = metrics(constant_prediction, analytic_roles["confirm"], target_scale)
    results: dict[str, object] = {
        "O0_constant": {"prediction": constant.tolist(), "confirm": constant_metrics},
    }
    probe_seeds = [int(value) for value in protocol["training"]["probe_seeds"]]

    # O1: a predeclared early-only ridge versus one-hidden-layer MLP selection.
    o1_rows = []
    for seed in probe_seeds:
        ridge_early = ridge_prediction(analytic_roles["train"], analytic_roles["early"], target_mean, target_scale)
        ridge_score = float(np.mean([
            row["nmae"] for row in metrics(ridge_early, analytic_roles["early"], target_scale)["by_target"].values()
        ]))
        mlp_early, mlp_prediction, receipt = train_feature_probe(
            analytic_roles["train"], analytic_roles["early"], analytic_roles["confirm"], seed,
            target_mean, target_scale, int(protocol["training"]["maximum_epochs"]),
            int(protocol["training"]["early_stopping_patience"]),
        )
        mlp_score = float(np.mean([
            row["nmae"] for row in metrics(mlp_early, analytic_roles["early"], target_scale)["by_target"].values()
        ]))
        # Both O1 candidates are compared using the identical early NMAE metric;
        # the confirmation role is not read during model selection.
        choose_mlp = mlp_score <= ridge_score
        if choose_mlp:
            prediction, selected = mlp_prediction, "one_hidden_mlp"
        else:
            prediction, selected = ridge_prediction(analytic_roles["train"], analytic_roles["confirm"], target_mean, target_scale), "ridge"
        o1_rows.append({
            "probe_seed": seed, "selected_by_early": selected,
            "ridge_early_nmae": ridge_score, "mlp_early_nmae": mlp_score, **receipt,
            "confirm": metrics(prediction, analytic_roles["confirm"], target_scale),
        })
    results["O1_analytic_composition"] = {"rows": o1_rows}

    # O2/O3 use a deterministic corresponding frozen P4c source seed and a new scalar-head seed.
    for kind, label in (("mlcnn", "O2_frozen_mlcnn_composition"), ("iqgrouped", "O3_frozen_iqgrouped_composition")):
        rows = []
        for source_seed, probe_seed in zip((3041, 3042, 3043), probe_seeds):
            frozen, source_receipt = frozen_source_model(project, source_seed)
            frozen = frozen.to(device).eval()
            role_features = {
                role: collect_role(dataset, device, frozen, kind, 126)
                for role, dataset in datasets.items()
            }
            _early_prediction, prediction, receipt = train_feature_probe(
                role_features["train"], role_features["early"], role_features["confirm"], probe_seed,
                target_mean, target_scale, int(protocol["training"]["maximum_epochs"]),
                int(protocol["training"]["early_stopping_patience"]),
            )
            rows.append({
                "probe_seed": probe_seed, "frozen_source": source_receipt, **receipt,
                "confirm": metrics(prediction, role_features["confirm"], target_scale),
            })
            del frozen
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[label] = {"rows": rows}

    # O4: the only trainable raw-I/Q upper-bound probe.  It has no nominal-power input.
    o4_rows = []
    for seed in probe_seeds:
        prediction, role, receipt = train_iq_oracle(
            datasets["train"], datasets["early"], datasets["confirm"], seed,
            target_mean, target_scale, int(protocol["training"]["maximum_epochs"]),
            int(protocol["training"]["early_stopping_patience"]),
        )
        o4_rows.append({
            "probe_seed": seed, **receipt,
            "confirm": metrics(prediction, role, target_scale),
            "prediction": prediction,
            "role": role,
        })
    o4_serializable = []
    for row in o4_rows:
        o4_serializable.append({key: value for key, value in row.items() if key not in ("prediction", "role")})
    results["O4_target_specific_iqgrouped_oracle"] = {"rows": o4_serializable}

    baseline_by_target = constant_metrics["by_target"]
    target_summary = {}
    worst_combo_failures = []
    bootstrap = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        rows = [entry["confirm"]["by_target"][target_name] for entry in o4_serializable]
        baseline_nmae = float(baseline_by_target[target_name]["nmae"])
        reductions = [1.0 - float(row["nmae"]) / baseline_nmae for row in rows]
        pearsons = [float(row["pearson"]) for row in rows]
        low_power_reductions = [
            1.0 - float(row["per_power_nmae"]["10"]) / float(baseline_by_target[target_name]["per_power_nmae"]["10"])
            for row in rows
        ]
        for row in rows:
            for combination, nmae in row["per_combination_nmae"].items():
                baseline_combo = float(baseline_by_target[target_name]["per_combination_nmae"][combination])
                if float(nmae) - baseline_combo > protocol["sufficiency_gates_for_O4"]["no_target_has_confirm_worst_combination_nmae_worse_than_O0_by_more_than"]:
                    worst_combo_failures.append({
                        "target": target_name, "combination": combination,
                        "delta": float(nmae) - baseline_combo,
                    })
        per_seed_bootstrap = [
            bootstrap_gain(constant_prediction, entry["prediction"], entry["role"], target_index, target_scale, entry["probe_seed"])
            for entry in o4_rows
        ]
        bootstrap[target_name] = per_seed_bootstrap
        target_summary[target_name] = {
            "constant_confirm_nmae": baseline_nmae,
            "o4_confirm_nmae_by_seed": [float(row["nmae"]) for row in rows],
            "mean_nmae_reduction_vs_constant": float(np.mean(reductions)),
            "nmae_reduction_by_seed": reductions,
            "mean_confirm_pearson": float(np.mean(pearsons)),
            "confirm_pearson_by_seed": pearsons,
            "mean_10db_nmae_reduction_vs_constant": float(np.mean(low_power_reductions)),
            "10db_nmae_reduction_by_seed": low_power_reductions,
        }
    gates = protocol["sufficiency_gates_for_O4"]
    mft, vft, isrj = (target_summary[name] for name in TARGET_NAMES)
    gate_results = {
        "mft_mean_nmae_reduction": mft["mean_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_nmae_reduction_vs_O0_at_least"],
        "isrj_mean_nmae_reduction": isrj["mean_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_nmae_reduction_vs_O0_at_least"],
        "mft_three_seed_positive": all(value > 0 for value in mft["nmae_reduction_by_seed"]),
        "isrj_three_seed_positive": all(value > 0 for value in isrj["nmae_reduction_by_seed"]),
        "mft_pearson": mft["mean_confirm_pearson"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_pearson_at_least"],
        "isrj_pearson": isrj["mean_confirm_pearson"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_pearson_at_least"],
        "mft_10db": mft["mean_10db_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_10db_confirm_nmae_reduction_vs_O0_at_least"],
        "isrj_10db": isrj["mean_10db_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_10db_confirm_nmae_reduction_vs_O0_at_least"],
        "vft_control": vft["mean_nmae_reduction_vs_constant"] >= 0.0,
        "worst_combination": not worst_combo_failures,
        "bootstrap_ci": all(
            all(row["ci95_low"] > gates["identity_clustered_bootstrap_gain_ci95_low_greater_than"] for row in bootstrap[name])
            for name in ("MFT.JNR", "ISRJ.JNR")
        ),
    }
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "o4_confirm_predictions.npz"
    np.savez_compressed(
        prediction_path,
        predictions=np.stack([row["prediction"] for row in o4_rows]),
        target=o4_rows[0]["role"].target, valid=o4_rows[0]["role"].valid.astype(np.uint8),
        identities=o4_rows[0]["role"].identities, powers=o4_rows[0]["role"].power,
        combinations=o4_rows[0]["role"].combination,
    )
    payload = {
        "experiment": protocol["experiment"],
        "preregister": str(preregister_path), "preregister_sha256": sha256_file(preregister_path),
        "role_receipt": role_receipt,
        "probe_input_contract": {
            "oracle_composition_provided": True,
            "nominal_power_provided_to_any_probe": False,
            "powers_used": list(POWERS),
            "raw_iq_input_normalization": "per-sample complex RMS, matching controlled I/Q model interface",
            "diagnostic_only": True,
        },
        "target_normalization_train_only": {"mean": target_mean.tolist(), "scale": target_scale.tolist()},
        "results": results,
        "o4_target_summary": target_summary,
        "o4_identity_clustered_bootstrap": bootstrap,
        "worst_combination_failures": worst_combo_failures,
        "gate_results": gate_results,
        "physical_sufficiency_demonstrated": bool(all(gate_results.values())),
        "o4_confirm_predictions": str(prediction_path),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    output = output_root / "result.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "o4_target_summary": target_summary, "gate_results": gate_results,
        "physical_sufficiency_demonstrated": payload["physical_sufficiency_demonstrated"],
    }, indent=2))
    for dataset in datasets.values():
        dataset.close()


if __name__ == "__main__":
    main()
