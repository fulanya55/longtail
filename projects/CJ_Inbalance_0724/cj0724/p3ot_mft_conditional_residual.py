"""P3o-T: deployable MFT-support-conditioned ISRJ residual on frozen MLCNN.

The oracle P3o-S subtraction result is a causal diagnosis only.  P3o-T keeps
the raw compound I/Q and every frozen base-model logit, then learns a scalar
ISRJ residual from an observable inverse-delay support summary.  No labels,
parameters, or generator components enter inference features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main
from cj0724.multitask import PARAMETER_COLUMNS, reshape_selected_parameters
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import worst_combo_predicted_gate_nmae
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import MFT_CLASS_INDEX, REGIMES, reference_signal
from cj0724.p3os_mft_oracle_subtraction import paired_bootstrap_source_fpr_reduction
from cj0724.paper_core_evidence import gated_metrics, load_normalizer
from cj0724.train import set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr


BASE_MODEL_SEEDS = (2041, 2042, 2043)
RESIDUAL_SEEDS = (2401, 2402, 2403)
ISRJ_CLASS_INDEX = 5
SUPPORT_START, SUPPORT_STOP, SUPPORT_BINS = 350, 1101, 16


def role_paths(root: Path, role: str, seed: int, suffix: str = "") -> dict[str, dict[int, Path]]:
    stem = f"p3ot_mft_conditional_residual_{role}_seed{int(seed)}"
    return {regime: transition_files(root / f"{stem}_{regime}{suffix}", "train", POWERS) for regime in REGIMES}


def ordered_paths(paths: dict[str, dict[int, Path]]) -> list[Path]:
    return [paths[regime][power] for regime in REGIMES for power in POWERS]


def identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.int64)).tobytes()).hexdigest()


def _role_metadata(paths: dict[str, dict[int, Path]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    identities = labels = None
    rows = []
    for regime in REGIMES:
        files = paths[regime]
        alignment = audit_power_alignment(files)
        current_identity, current_labels = read_identity_metadata(files[25])
        if len(current_identity) != 2079:
            raise RuntimeError(f"P3o-T expects 2079 identities in {regime}, got {len(current_identity)}")
        if identities is None:
            identities, labels = current_identity, current_labels
        elif not np.array_equal(identities, current_identity) or not np.array_equal(labels, current_labels):
            raise RuntimeError(f"P3o-T identities/labels differ across regimes at {regime}")
        rows.append({"regime": regime, "identity_count": int(len(current_identity)), "alignment": alignment})
    assert identities is not None and labels is not None
    return np.asarray(identities, dtype=np.int64), np.asarray(labels, dtype=np.uint8), rows


def audit_roles(
    project: Path,
    root: Path,
    suffix: str,
    role_seeds: dict[str, int],
    prior_roots: list[Path],
) -> dict[str, object]:
    paths = {role: role_paths(root, role, seed, suffix) for role, seed in role_seeds.items()}
    role_identities: dict[str, np.ndarray] = {}
    reference_labels = None
    contracts = {}
    for role, current in paths.items():
        identity, labels, rows = _role_metadata(current)
        role_identities[role] = identity
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError("P3o-T label contract differs between roles")
        contracts[role] = rows
    names = list(role_identities)
    cross = {f"{left}_{right}": int(np.intersect1d(role_identities[left], role_identities[right]).size) for index, left in enumerate(names) for right in names[index + 1:]}
    if any(cross.values()):
        raise RuntimeError(f"P3o-T cross-role identity overlap: {cross}")
    current_files = {path.resolve() for by_role in paths.values() for by_regime in by_role.values() for path in by_regime.values()}
    history_roots = [project / "generated_data", Path("D:/CJ0724_generated_data"), Path("D:/CJ0724_rechunked_data")] + prior_roots
    previous = set()
    for history in history_roots:
        if history.is_dir():
            previous.update(path.resolve() for path in history.rglob("closedset_6class_train.mat"))
    protocol = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json").read_text(encoding="utf-8"))
    original = Path(protocol["data_file"])
    if original.is_file():
        previous.add(original.resolve())
    prior_rows, maximum = [], 0
    for path in sorted(previous):
        if path in current_files:
            continue
        old_identity, _ = read_identity_metadata(path)
        overlaps = {role: int(np.intersect1d(identity, old_identity).size) for role, identity in role_identities.items()}
        maximum = max(maximum, *overlaps.values())
        prior_rows.append({"path": str(path), "overlaps": overlaps})
    if maximum:
        raise RuntimeError("P3o-T identity overlap with a historical dataset")
    return {
        "roles": {role: {"seed": int(role_seeds[role]), "identity_count": int(len(identity)), "identity_sha256": identity_hash(identity), "contract": contracts[role]} for role, identity in role_identities.items()},
        "cross_role_overlap": cross,
        "maximum_prior_overlap": maximum,
        "prior_files_checked": len(prior_rows),
        "prior_rows": prior_rows,
        "power_alignment_verified": True,
        "label_contract_verified": True,
        "rechunk_root": str(root.resolve()),
        "suffix": suffix,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }


class FixedMFTSupport(nn.Module):
    """Observable complex-LFM inverse-delay summary and a matched phase control."""

    def __init__(self, bins: int = SUPPORT_BINS) -> None:
        super().__init__()
        reference = reference_signal()
        spectrum = np.fft.fft(reference).astype(np.complex64)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        phase = np.exp(1j * np.linspace(0.0, 13.0 * np.pi, len(spectrum), endpoint=False)).astype(np.complex64)
        self.bins = int(bins)
        self.register_buffer("reference_conjugate", torch.from_numpy(spectrum.conj()))
        self.register_buffer("denominator", torch.from_numpy(power + 1.0e-6 * float(power.max())))
        self.register_buffer("control_phase", torch.from_numpy(phase))

    def _impulse(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1:] != (2, 3000):
            raise ValueError(f"Expected [B,2,3000], got {tuple(iq.shape)}")
        signal = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        return torch.fft.ifft(torch.fft.fft(signal, dim=1) * self.reference_conjugate[None, :] / self.denominator[None, :], dim=1)[:, SUPPORT_START:SUPPORT_STOP]

    def _summary(self, impulse: torch.Tensor) -> torch.Tensor:
        magnitude = impulse.abs()
        normalized = magnitude / magnitude.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
        pooled = F.adaptive_avg_pool1d(normalized[:, None, :], self.bins).squeeze(1)
        maximum = F.adaptive_max_pool1d(normalized[:, None, :], self.bins).squeeze(1)
        return torch.cat((pooled, maximum), dim=1)

    def forward(self, iq: torch.Tensor, mode: str) -> torch.Tensor:
        impulse = self._impulse(iq)
        if mode == "support":
            return self._summary(impulse)
        if mode == "control":
            scrambled = torch.fft.ifft(torch.fft.fft(impulse, dim=1) * self.control_phase[None, :SUPPORT_STOP - SUPPORT_START], dim=1)
            return self._summary(scrambled)
        if mode == "zero":
            return torch.zeros((len(iq), 2 * self.bins), dtype=iq.dtype, device=iq.device)
        raise ValueError(f"Unknown support mode {mode!r}")


class ISRJConditionalResidual(nn.Module):
    """Equal-capacity scalar ISRJ correction; base logits remain untouched."""

    def __init__(self, feature_dim: int = 256, support_dim: int = 2 * SUPPORT_BINS, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim + support_dim + 1),
            nn.Linear(feature_dim + support_dim + 1, int(hidden_dim)), nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, frozen_feature: torch.Tensor, frozen_logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        mft_gate = torch.sigmoid(frozen_logits[:, MFT_CLASS_INDEX]).detach().unsqueeze(1)
        values = torch.cat((frozen_feature.detach(), support.detach(), mft_gate), dim=1)
        return self.network(values).squeeze(1) * mft_gate.squeeze(1)


def corrected_logits(base_logits: torch.Tensor, delta_isrj: torch.Tensor) -> torch.Tensor:
    if base_logits.ndim != 2 or base_logits.shape[1] != 6 or delta_isrj.shape != (len(base_logits),):
        raise ValueError("Incompatible P3o-T logits/residual")
    result = base_logits.clone()
    result[:, ISRJ_CLASS_INDEX] = result[:, ISRJ_CLASS_INDEX] + delta_isrj
    return result


def conditional_residual_loss(corrected: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor, weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(corrected[:, ISRJ_CLASS_INDEX], labels[:, ISRJ_CLASS_INDEX])
    source = (labels[:, MFT_CLASS_INDEX] > 0.5) & (labels[:, ISRJ_CLASS_INDEX] < 0.5)
    if bool(source.any()):
        conditional = F.softplus(corrected[source, ISRJ_CLASS_INDEX] - base_logits[source, ISRJ_CLASS_INDEX]).mean()
    else:
        conditional = corrected.sum() * 0.0
    return bce + float(weight) * conditional, bce, conditional


def _freeze(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.inference_mode()
def frozen_base_outputs(model: nn.Module, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, prediction = model(iq)
    feature = model.classification_adapter(model.classification_backbone(iq))
    replicated = model.classifier(feature)
    if not torch.equal(logits, replicated):
        raise RuntimeError("P3o-T frozen classification logit reproduction failed")
    return logits.detach(), feature.detach(), prediction.detach()


def _dataset(paths: dict[str, dict[int, Path]]) -> ConcatDataset:
    indices = np.arange(2079, dtype=np.int64)
    return ConcatDataset([V73IQDataset(path, indices, PARAMETER_COLUMNS) for path in ordered_paths(paths)])


def _close(dataset: ConcatDataset) -> None:
    for part in dataset.datasets:
        if isinstance(part, V73IQDataset):
            part.close()


def _metadata(paths: dict[str, dict[int, Path]]) -> dict[str, np.ndarray]:
    labels = []
    identity = []
    regime = []
    power = []
    sample_index = []
    for regime_index, regime_name in enumerate(REGIMES):
        current_identity, current_labels = read_identity_metadata(paths[regime_name][25])
        for power_value in POWERS:
            labels.append(current_labels)
            identity.append(current_identity)
            regime.append(np.full(len(current_identity), regime_index, dtype=np.int8))
            power.append(np.full(len(current_identity), power_value, dtype=np.int16))
            sample_index.append(np.arange(len(current_identity), dtype=np.int64))
    return {"labels": np.concatenate(labels), "identity": np.concatenate(identity), "regime": np.concatenate(regime), "power": np.concatenate(power), "sample_index": np.concatenate(sample_index)}


def _forward_residual(base: nn.Module, branch: ISRJConditionalResidual | None, frontend: FixedMFTSupport, iq: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    base_logits, feature, prediction = frozen_base_outputs(base, iq)
    if branch is None:
        return base_logits, prediction
    support = frontend(iq, mode)
    return corrected_logits(base_logits, branch(feature, base_logits, support)), prediction


def _selection_loss(base: nn.Module, branch: ISRJConditionalResidual, frontend: FixedMFTSupport, mode: str, loader: DataLoader, device: torch.device) -> float:
    branch.eval(); values = []
    with torch.inference_mode():
        for batch in loader:
            iq, labels = batch["iq"].to(device), batch["label"].to(device)
            logits, _ = _forward_residual(base, branch, frontend, iq, mode)
            values.append(float(F.binary_cross_entropy_with_logits(logits[:, ISRJ_CLASS_INDEX], labels[:, ISRJ_CLASS_INDEX])))
    return float(np.mean(values))


def train_residual(
    base: nn.Module,
    frontend: FixedMFTSupport,
    mode: str,
    fit_loader: DataLoader,
    selection_loader: DataLoader,
    seed: int,
    output: Path,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int,
    conditional_weight: float,
    gradient_clip_norm: float,
    device: torch.device,
) -> tuple[ISRJConditionalResidual, dict[str, object]]:
    set_seed(seed)
    branch = ISRJConditionalResidual().to(device)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    best, best_epoch, history = float("inf"), 0, []
    checkpoint = output / "best.pt"
    for epoch in range(1, int(epochs) + 1):
        branch.train(); losses = []
        lr = _set_warmup_cosine_lr(optimizer, learning_rate, epoch, epochs, warmup_epochs)
        for batch in fit_loader:
            iq, labels = batch["iq"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            base_logits, feature, _ = frozen_base_outputs(base, iq)
            support = frontend(iq, mode)
            logits = corrected_logits(base_logits, branch(feature, base_logits, support))
            loss, bce, conditional = conditional_residual_loss(logits, base_logits, labels, conditional_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(), float(gradient_clip_norm)); optimizer.step()
            losses.append((float(loss.detach()), float(bce.detach()), float(conditional.detach())))
        selection = _selection_loss(base, branch, frontend, mode, selection_loader, device)
        row = {"epoch": epoch, "lr": float(lr), "train_loss": float(np.mean([v[0] for v in losses])), "train_bce": float(np.mean([v[1] for v in losses])), "train_conditional": float(np.mean([v[2] for v in losses])), "selection_isrj_bce": selection}
        history.append(row)
        if selection < best:
            best, best_epoch = selection, epoch
            torch.save(branch.state_dict(), checkpoint)
    branch.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); branch.eval()
    return branch, {"best_selection_isrj_bce": best, "best_epoch": best_epoch, "history": history, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "parameters": int(sum(p.numel() for p in branch.parameters()))}


@torch.inference_mode()
def collect_record(
    base: nn.Module,
    branch: ISRJConditionalResidual | None,
    frontend: FixedMFTSupport,
    mode: str,
    loader: DataLoader,
    device: torch.device,
    normalizer,
    metadata: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    parts = {name: [] for name in ("logits", "labels", "prediction", "target", "mask")}
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        logits, normalized_prediction = _forward_residual(base, branch, frontend, iq, mode)
        target = reshape_selected_parameters(batch["parameter_values"].to(device))
        mask = reshape_selected_parameters(batch["parameter_mask"].to(device)).bool()
        parts["logits"].append(logits.float().cpu())
        parts["labels"].append(batch["label"].to(torch.uint8).cpu())
        parts["prediction"].append(normalizer.inverse_transform(normalized_prediction.float().cpu()))
        parts["target"].append(target.float().cpu())
        parts["mask"].append(mask.cpu())
    labels = torch.cat(parts["labels"]).numpy().astype(np.uint8)
    if not np.array_equal(labels, metadata["labels"]):
        raise RuntimeError("P3o-T loader metadata order mismatch")
    logits = torch.cat(parts["logits"]).numpy()
    return {
        "identity": metadata["identity"], "regime": metadata["regime"], "power": metadata["power"], "sample_index": metadata["sample_index"],
        "labels": labels, "logits": logits, "probabilities": 1.0 / (1.0 + np.exp(-logits)),
        "prediction": torch.cat(parts["prediction"]).numpy(), "target": torch.cat(parts["target"]).numpy(), "mask": torch.cat(parts["mask"]).numpy().astype(np.uint8),
        "thresholds": np.asarray(thresholds, dtype=np.float64), "scale": normalizer.scale.numpy(),
    }


def cp_upper(false_positives: int, negatives: int, alpha: float = 0.05) -> float:
    if negatives <= 0 or false_positives >= negatives:
        return 1.0
    return float(beta.ppf(1.0 - alpha, false_positives + 1, negatives - false_positives))


def choose_selection_isrj_threshold(record: dict[str, np.ndarray]) -> dict[str, object]:
    probability, labels, regime, power = record["probabilities"][:, ISRJ_CLASS_INDEX], record["labels"][:, ISRJ_CLASS_INDEX].astype(bool), record["regime"], record["power"]
    feasible = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        cells, valid = [], True
        for regime_index, regime_name in enumerate(REGIMES):
            for power_value in POWERS:
                rows = (regime == regime_index) & (power == power_value); truth = labels[rows]; prediction = probability[rows] >= threshold
                fp, negatives = int(prediction[~truth].sum()), int((~truth).sum())
                observed, upper = fp / max(negatives, 1), cp_upper(fp, negatives)
                if observed > 0.08 + 1e-12 or upper > 0.15 + 1e-12:
                    valid = False
                cells.append({"regime": regime_name, "power": int(power_value), "fpr": observed, "cp95_upper": upper, "recall": float(prediction[truth].mean())})
        if valid:
            feasible.append({"threshold": float(threshold), "mean_recall": float(np.mean([row["recall"] for row in cells])), "cells": cells})
    if not feasible:
        return {"threshold": 1.0, "feasible": False, "mean_recall": 0.0, "cells": []}
    selected = max(feasible, key=lambda row: (row["mean_recall"], -row["threshold"])); selected["feasible"] = True; selected["feasible_count"] = len(feasible)
    return selected


def classification_summary(record: dict[str, np.ndarray]) -> dict[str, object]:
    labels, probability, thresholds = record["labels"], record["probabilities"], record["thresholds"]
    prediction = probability >= thresholds[None, :]
    source = (labels[:, MFT_CLASS_INDEX] > 0) & (labels[:, ISRJ_CLASS_INDEX] == 0)
    truth = labels[:, ISRJ_CLASS_INDEX].astype(bool)
    ap = [float(average_precision_score(labels[:, index], probability[:, index])) for index in range(6)]
    cells = []
    for regime_index, regime_name in enumerate(REGIMES):
        for power_value in POWERS:
            cell = (record["regime"] == regime_index) & (record["power"] == power_value); source_cell = cell & source; truth_cell = cell & truth
            cells.append({"regime": regime_name, "power": int(power_value), "source_isrj_fpr": float(prediction[source_cell, ISRJ_CLASS_INDEX].mean()), "isrj_recall": float(prediction[truth_cell, ISRJ_CLASS_INDEX].mean()), "isrj_fpr": float(prediction[cell & ~truth, ISRJ_CLASS_INDEX].mean())})
    return {
        "source_isrj_fpr": float(prediction[source, ISRJ_CLASS_INDEX].mean()), "worst_source_cell_fpr": float(max(row["source_isrj_fpr"] for row in cells)),
        "isrj_recall": float(prediction[truth, ISRJ_CLASS_INDEX].mean()), "isrj_fpr": float(prediction[~truth, ISRJ_CLASS_INDEX].mean()),
        "macro_aupr": float(np.mean(ap)), "macro_f1": float(f1_score(labels, prediction.astype(np.uint8), average="macro", zero_division=0)),
        "mft_ap": ap[MFT_CLASS_INDEX], "isrj_ap": ap[ISRJ_CLASS_INDEX], "cells": cells,
    }


def _difference(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    return {"source_fpr_reduction": float(baseline["source_isrj_fpr"] - candidate["source_isrj_fpr"]), "worst_source_fpr_reduction": float(baseline["worst_source_cell_fpr"] - candidate["worst_source_cell_fpr"]), "isrj_recall_delta": float(candidate["isrj_recall"] - baseline["isrj_recall"]), "macro_aupr_delta": float(candidate["macro_aupr"] - baseline["macro_aupr"]), "macro_f1_delta": float(candidate["macro_f1"] - baseline["macro_f1"]), "mft_ap_delta": float(candidate["mft_ap"] - baseline["mft_ap"])}


def _regime_direction(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    b = {(row["regime"], row["power"]): row["source_isrj_fpr"] for row in baseline["cells"]}
    c = {(row["regime"], row["power"]): row["source_isrj_fpr"] for row in candidate["cells"]}
    return {f"{name}_{power}": float(b[(name, power)] - c[(name, power)]) for name in REGIMES for power in POWERS}


def _apply_selection_threshold(record: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    result = dict(record); values = np.asarray(record["thresholds"], dtype=np.float64).copy(); values[ISRJ_CLASS_INDEX] = float(threshold); result["thresholds"] = values
    return result


def _verify_freeze(project: Path, freeze_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    files = {"preregister_sha256": project / "configs/p3ot_mft_conditional_residual_preregister_20260803.json", "implementation_sha256": Path(__file__), "generator_sha256": project / "matlab/generate_p3ot_mft_conditional_residual_level.m", "support_dependency_sha256": project / "cj0724/p3c_mft_sparse_deconvolution.py"}
    mismatch = {key: {"expected": freeze.get(key), "actual": sha256_file(path)} for key, path in files.items() if freeze.get(key) != sha256_file(path)}
    if mismatch:
        raise RuntimeError(f"P3o-T method freeze mismatch: {mismatch}")
    return freeze


def _base_receipts(project: Path) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]], dict[str, object]]:
    run_root = project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156"
    p2c_root = project / "outputs/p2c_lazy_joint_confirmation_20260801"
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    thresholds = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8"))
    selected = {int(row["seed"]): row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    freezes = {int(row["seed"]): row for row in thresholds["seed_freezes"]}
    if set(selected) != set(BASE_MODEL_SEEDS) or set(freezes) != set(BASE_MODEL_SEEDS):
        raise RuntimeError("P3o-T frozen base checkpoint or safe-threshold receipt is incomplete")
    return selected, freezes, protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rechunk-root", required=True)
    parser.add_argument("--rechunk-suffix", default="")
    parser.add_argument("--method-freeze", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--preflight-output")
    parser.add_argument("--prior-identity-root", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--conditional-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    if bool(args.output_root) == bool(args.preflight_output):
        raise ValueError("Specify exactly one of --output-root or --preflight-output")
    _verify_freeze(project, project / args.method_freeze)
    root = Path(args.rechunk_root)
    roles = {"train": 1960000000, "selection": 1970000000, "development": 1980000000}
    audit = audit_roles(project, root, args.rechunk_suffix, roles, [Path(value) for value in args.prior_identity_root])
    if args.preflight_output:
        destination = project / args.preflight_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
        print(json.dumps({"preflight_passed": True, "identity_overlap_zero": True, "prior_files_checked": audit["prior_files_checked"]}, indent=2)); return
    output = project / args.output_root
    output.mkdir(parents=True, exist_ok=False)
    selected, threshold_freezes, protocol = _base_receipts(project)
    normalizer = load_normalizer(protocol["normalizer"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    paths = {role: role_paths(root, role, seed, args.rechunk_suffix) for role, seed in roles.items()}
    datasets = {role: _dataset(paths[role]) for role in paths}
    metadata = {role: _metadata(paths[role]) for role in paths}
    loaders = {"fit": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(20260803), num_workers=0, pin_memory=device.type == "cuda"), "selection": DataLoader(datasets["selection"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"), "development": DataLoader(datasets["development"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")}
    frontend = FixedMFTSupport().to(device).eval()
    runs, saved = [], {}
    try:
        for base_seed, residual_seed in zip(BASE_MODEL_SEEDS, RESIDUAL_SEEDS):
            base, checkpoint = _load_main(selected[base_seed], protocol, project, device); _freeze(base)
            base_thresholds = np.asarray(threshold_freezes[base_seed]["safe_thresholds"], dtype=np.float64)
            condition_records: dict[str, dict[str, np.ndarray]] = {}
            selection_records: dict[str, dict[str, np.ndarray]] = {}
            condition_records["T0_base"] = collect_record(base, None, frontend, "zero", loaders["development"], device, normalizer, metadata["development"], base_thresholds)
            selection_records["T0_base"] = collect_record(base, None, frontend, "zero", loaders["selection"], device, normalizer, metadata["selection"], base_thresholds)
            branch_receipts = {}
            for name, mode in (("T1_zero_side_residual", "zero"), ("T2_support_conditional_residual", "support"), ("T3_matched_noninformative_control", "control")):
                run_dir = output / f"base{base_seed}_residual{residual_seed}" / name; run_dir.mkdir(parents=True)
                branch, receipt = train_residual(base, frontend, mode, loaders["fit"], loaders["selection"], residual_seed, run_dir, args.learning_rate, args.weight_decay, args.epochs, args.warmup_epochs, args.conditional_weight, args.gradient_clip_norm, device)
                branch_receipts[name] = receipt
                condition_records[name] = collect_record(base, branch, frontend, mode, loaders["development"], device, normalizer, metadata["development"], base_thresholds)
                selection_records[name] = collect_record(base, branch, frontend, mode, loaders["selection"], device, normalizer, metadata["selection"], base_thresholds)
                del branch
            selection_thresholds = {name: choose_selection_isrj_threshold(record) for name, record in selection_records.items()}
            primary = {name: classification_summary(record) for name, record in condition_records.items()}
            calibrated = {name: classification_summary(_apply_selection_threshold(record, selection_thresholds[name]["threshold"])) for name, record in condition_records.items()}
            for name, record in condition_records.items():
                saved[f"base{base_seed}_{name}_identity"] = record["identity"]; saved[f"base{base_seed}_{name}_labels"] = record["labels"]; saved[f"base{base_seed}_{name}_regime"] = record["regime"]; saved[f"base{base_seed}_{name}_power"] = record["power"]; saved[f"base{base_seed}_{name}_probabilities"] = record["probabilities"]
            comparisons = {}
            for candidate, baseline, tag in (("T2_support_conditional_residual", "T0_base", "T2_minus_T0"), ("T2_support_conditional_residual", "T1_zero_side_residual", "T2_minus_T1"), ("T2_support_conditional_residual", "T3_matched_noninformative_control", "T2_minus_T3")):
                bootstrap = paired_bootstrap_source_fpr_reduction(condition_records[baseline]["probabilities"][:, ISRJ_CLASS_INDEX], condition_records[candidate]["probabilities"][:, ISRJ_CLASS_INDEX], condition_records[candidate]["labels"], condition_records[candidate]["identity"], base_thresholds[ISRJ_CLASS_INDEX], 910000 + base_seed + len(comparisons), args.bootstrap_resamples)
                comparisons[tag] = {"difference": _difference(primary[candidate], primary[baseline]), "bootstrap_source_fpr_reduction": bootstrap, "regime_power_source_fpr_reduction": _regime_direction(primary[candidate], primary[baseline])}
            parameter = {name: gated_metrics(record) | {"worst_combo_predicted_gate_nmae": worst_combo_predicted_gate_nmae(record)} for name, record in condition_records.items()}
            run = {"base_model_seed": base_seed, "residual_seed": residual_seed, "base_checkpoint": str(checkpoint), "base_safe_thresholds": base_thresholds, "branch_receipts": branch_receipts, "primary_frozen_threshold_metrics": primary, "secondary_selection_calibration": selection_thresholds, "secondary_calibrated_metrics": calibrated, "comparisons": comparisons, "predicted_gate_parameter_metrics": parameter}
            (output / f"base{base_seed}_result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
            runs.append(run); del base
            if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        for dataset in datasets.values(): _close(dataset)
    gates = {
        "identity_overlap_zero": audit["maximum_prior_overlap"] == 0 and not any(audit["cross_role_overlap"].values()),
        "T2_source_fpr_reduction_vs_T0_at_least_0p02_each_seed": all(row["comparisons"]["T2_minus_T0"]["difference"]["source_fpr_reduction"] >= .02 for row in runs),
        "T2_source_fpr_reduction_vs_T1_at_least_0p01_each_seed": all(row["comparisons"]["T2_minus_T1"]["difference"]["source_fpr_reduction"] >= .01 for row in runs),
        "T2_source_fpr_reduction_vs_T3_at_least_0p01_each_seed": all(row["comparisons"]["T2_minus_T3"]["difference"]["source_fpr_reduction"] >= .01 for row in runs),
        "all_paired_bootstrap_ci_lows_positive": all(run["comparisons"][tag]["bootstrap_source_fpr_reduction"]["ci95_low"] > 0 for run in runs for tag in ("T2_minus_T0", "T2_minus_T1", "T2_minus_T3")),
        "T2_isrj_recall_drop_at_most_0p02_each_seed": all(row["comparisons"]["T2_minus_T0"]["difference"]["isrj_recall_delta"] >= -.02 for row in runs),
        "T2_worst_source_cell_not_worse_each_seed": all(row["comparisons"]["T2_minus_T0"]["difference"]["worst_source_fpr_reduction"] >= 0 for row in runs),
        "T2_mft_ap_bitwise_equal_each_seed": all(abs(row["comparisons"]["T2_minus_T0"]["difference"]["mft_ap_delta"]) <= 1e-12 for row in runs),
        "T2_macro_aupr_drop_at_most_0p005_each_seed": all(row["comparisons"]["T2_minus_T0"]["difference"]["macro_aupr_delta"] >= -.005 for row in runs),
        "T2_macro_f1_drop_at_most_0p005_each_seed": all(row["comparisons"]["T2_minus_T0"]["difference"]["macro_f1_delta"] >= -.005 for row in runs),
        "T2_predicted_gate_nmae_ratio_at_most_1p01_each_seed": all(row["predicted_gate_parameter_metrics"]["T2_support_conditional_residual"]["predicted_gate_parameter_nmae"] <= 1.01 * row["predicted_gate_parameter_metrics"]["T0_base"]["predicted_gate_parameter_nmae"] for row in runs),
        "T2_miss_penalty_nonworse_each_seed": all(row["predicted_gate_parameter_metrics"]["T2_support_conditional_residual"]["miss_penalty_parameter_nmae"] <= row["predicted_gate_parameter_metrics"]["T0_base"]["miss_penalty_parameter_nmae"] for row in runs),
        "T2_worst_combo_nonworse_each_seed": all(row["predicted_gate_parameter_metrics"]["T2_support_conditional_residual"]["worst_combo_predicted_gate_nmae"] <= row["predicted_gate_parameter_metrics"]["T0_base"]["worst_combo_predicted_gate_nmae"] for row in runs),
        "T2_at_least_four_regimes_nonnegative_each_seed": all(sum(all(value >= 0 for key, value in row["comparisons"]["T2_minus_T0"]["regime_power_source_fpr_reduction"].items() if key.startswith(regime + "_")) for regime in REGIMES) >= 4 for row in runs),
        "T2_power25_source_fpr_direction_nonnegative_each_seed": all(all(value >= 0 for key, value in row["comparisons"]["T2_minus_T0"]["regime_power_source_fpr_reduction"].items() if key.endswith("_25")) for row in runs),
        "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True,
    }
    summary = {"experiment": "P3o-T deployable MFT-support conditional ISRJ residual", "development_passed": bool(all(gates.values())), "verdict": "promote_to_fresh_confirmation_without_retuning" if all(gates.values()) else "stop_no_confirmation_data", "promotion_gates": gates, "data_audit": audit, "runs": runs, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    np.savez_compressed(output / "development_predictions.npz", **saved)
    print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
