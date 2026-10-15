"""Fresh-source H1t test of paired asymmetric target-local posterior transfer."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.losses.classification import MultiLabelClassificationLoss  # noqa: E402
from experiments.audit_route_a_h1t_source import TARGET_COLUMNS, _view_path  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    _component_power, _configure_partial_scope, _load_signal_array,
)
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    _load_model_from_state,
)


TARGET_NAMES = ("MFT", "VFT", "ISRJ")
VARIANTS = ("B0_SupervisedFocal10dB", "T1_PairedTargetLocalTransfer", "P1_CompositionPermutedTeacher")
SOURCE_SEEDS = (2101, 2102, 2103, 2104, 2105)
FOCAL_SEEDS = (2039, 2040, 2041, 2042, 2043)
DEFAULT_SOURCE = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h1t_asymmetric_power_v1")
DEFAULT_SOURCE_AUDIT = ROOT / "outputs/tapc/route_a_h1t_source_audit/summary.json"
DEFAULT_F2_ROOT = ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen"
DEFAULT_FOCAL_ROOT = ROOT / "outputs/tapc/route_a_development_v3_loss_convergence_audit"
DEFAULT_SEAL = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/tapc/route_a_h1t_asymmetric_visibility_transfer"


def binary_posterior_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                        weak_positive: torch.Tensor, temperature: float) -> torch.Tensor:
    """Target-local Bernoulli KL with a detached privileged teacher."""

    if student_logits.shape != teacher_logits.shape or student_logits.shape != weak_positive.shape:
        raise ValueError("target-local KL shapes must match")
    temperature = float(temperature)
    teacher = torch.sigmoid(teacher_logits.detach() / temperature)
    student_log_p = F.logsigmoid(student_logits / temperature)
    student_log_q = F.logsigmoid(-student_logits / temperature)
    teacher_log_p = torch.log(teacher.clamp(1e-7, 1.0 - 1e-7))
    teacher_log_q = torch.log((1.0 - teacher).clamp(1e-7, 1.0 - 1e-7))
    kl = (teacher * (teacher_log_p - student_log_p) + (1.0 - teacher) * (teacher_log_q - student_log_q)).clamp_min(0.0)
    selected = weak_positive.to(dtype=kl.dtype)
    return kl.mul(selected).sum() / selected.sum().clamp_min(1.0)


def composition_permute_teacher(teacher_logits: np.ndarray, labels: np.ndarray,
                                weak: np.ndarray) -> tuple[np.ndarray, dict]:
    """Cyclically destroy identity pairing while retaining per-target score marginals."""

    source = np.asarray(teacher_logits, dtype=np.float32)
    output = source.copy()
    labels, weak = np.asarray(labels, dtype=np.uint8), np.asarray(weak, dtype=bool)
    packed = np.packbits(labels, axis=1, bitorder="little").reshape(-1)
    details = {}
    for position, (name, column) in enumerate(zip(TARGET_NAMES, TARGET_COLUMNS)):
        affected, strata = 0, []
        mask = (labels[:, column] == 1) & weak[:, column]
        for key in sorted(np.unique(packed[mask]).tolist()):
            positions = np.flatnonzero(mask & (packed == key))
            if len(positions) >= 2:
                output[positions, position] = source[np.roll(positions, 1), position]
                affected += len(positions); strata.append(int(key))
        details[name] = {"permuted_count": int(affected), "permuted_composition_count": int(len(strata)),
                         "score_marginal_exact": bool(np.array_equal(np.sort(output[mask, position]), np.sort(source[mask, position])))}
    return output, details


def _weak_rank(labels: np.ndarray, weak: np.ndarray, logits: np.ndarray, column: int, kind: str) -> float:
    negative = labels[:, column] == 0
    if kind == "weak":
        positive = (labels[:, column] == 1) & weak[:, column]
    elif kind == "strong":
        positive = (labels[:, column] == 1) & ~weak[:, column]
    else:
        positive = labels[:, column] == 1
    choose = positive | negative
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    return float(roc_auc_score(positive[choose].astype(np.uint8), logits[choose, column]))


def _metrics(labels: np.ndarray, weak: np.ndarray, logits: np.ndarray) -> dict:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    target = {}
    for name, column in zip(TARGET_NAMES, TARGET_COLUMNS):
        positive = labels[:, column] == 1
        target[name] = {
            "all_auroc": _weak_rank(labels, weak, logits, column, "all"),
            "all_aupr": float(average_precision_score(labels[:, column], probability[:, column])),
            "weak_auroc": _weak_rank(labels, weak, logits, column, "weak"),
            "weak_aupr": float(average_precision_score(((labels[:, column] == 1) & weak[:, column]).astype(np.uint8), probability[:, column])),
            "strong_auroc": _weak_rank(labels, weak, logits, column, "strong"),
            "weak_count": int(((labels[:, column] == 1) & weak[:, column]).sum()),
            "strong_count": int(((labels[:, column] == 1) & ~weak[:, column]).sum()),
            "negative_count": int((labels[:, column] == 0).sum()),
        }
    non_target = {}
    for name, column in (("FMB", 0), ("SawFM", 1), ("SinFM", 2)):
        non_target[name] = {"auroc": _weak_rank(labels, weak, logits, column, "all"),
                            "aupr": float(average_precision_score(labels[:, column], probability[:, column]))}
    return {"per_target": target, "non_target": non_target,
            "target_mean_weak_auroc": float(np.nanmean([value["weak_auroc"] for value in target.values()])),
            "target_mean_weak_aupr": float(np.nanmean([value["weak_aupr"] for value in target.values()])),
            "target_mean_strong_auroc": float(np.nanmean([value["strong_auroc"] for value in target.values()]))}


def _predict(model: torch.nn.Module, signals: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            values.append(model(torch.from_numpy(signals[start:start + int(batch_size)]).to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _train(model: torch.nn.Module, variant: str, role: dict, teacher_logits: np.ndarray,
           permuted_logits: np.ndarray, seed: int, device: torch.device, args: argparse.Namespace,
           output_dir: Path) -> tuple[torch.nn.Module, dict]:
    scope = _configure_partial_scope(model)
    optimizer = torch.optim.AdamW([
        {"params": scope["shared"], "lr": args.learning_rate * args.representation_lr_scale},
        {"params": scope["classifier"], "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    train_seed = 330000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    dataset = TensorDataset(torch.from_numpy(role["fit"]["signals"]), torch.from_numpy(role["fit"]["labels"].astype(np.float32)),
                            torch.from_numpy(role["fit"]["weak"][:, list(TARGET_COLUMNS)]),
                            torch.from_numpy(teacher_logits), torch.from_numpy(permuted_logits))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(train_seed))
    focal = MultiLabelClassificationLoss("focal", gamma_pos=0.0, gamma_neg=3.0).to(device)
    best_state, best_key, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; count = 0
        for signal, labels, weak, high, permuted in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(signal.to(device)); labels = labels.to(device)
            supervised = focal.elementwise(logits, labels).mean()
            if variant == "B0_SupervisedFocal10dB":
                transfer = torch.zeros((), device=device)
            else:
                source = high if variant == "T1_PairedTargetLocalTransfer" else permuted
                transfer = binary_posterior_kl(logits[:, list(TARGET_COLUMNS)], source.to(device), weak.to(device), args.temperature)
            loss = supervised + (args.transfer_weight * args.temperature * args.temperature * transfer)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip); optimizer.step()
            loss_sum += float(loss.detach().cpu()); count += 1
        validation_logits = _predict(model, role["validation"]["signals"], device, args.eval_batch_size)
        validation = _metrics(role["validation"]["labels"], role["validation"]["weak"], validation_logits)
        key = validation["target_mean_weak_auroc"]
        history.append({"epoch": epoch, "train_loss": loss_sum / max(count, 1), "validation_target_mean_weak_auroc": key})
        if key > best_key + 1e-8:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True)
    torch.save({"model_state_dict": best_state, "variant": variant, "best_epoch": best_epoch,
                "train_seed": train_seed, "trainable_parameter_count": int(sum(value.numel() for value in scope["shared"] + scope["classifier"]))}, output_dir / "best_model.pth")
    _write_history(output_dir / "history.csv", history)
    return model, {"best_epoch": int(best_epoch), "validation_target_mean_weak_auroc": float(best_key),
                   "train_seconds": float(time.time() - started), "train_seed": int(train_seed),
                   "trainable_parameter_count": int(sum(value.numel() for value in scope["shared"] + scope["classifier"]))}


def _load_roles(source_path: Path, high_path: Path, role_path: Path, normalization: str, io_batch_size: int) -> tuple[dict, dict]:
    saved = np.load(role_path)
    roles = {}
    for name in ("fit", "validation", "early"):
        indices = saved[name].astype(np.int64)
        low = _load_signal_array(source_path, indices, normalization, io_batch_size)
        roles[name] = {"signals": low["signals"], "labels": low["targets6"].astype(np.uint8), "sample_ids": low["sample_ids"]}
    high = _load_signal_array(high_path, saved["fit"].astype(np.int64), normalization, io_batch_size)
    if not (np.array_equal(high["sample_ids"], roles["fit"]["sample_ids"]) and np.array_equal(high["targets6"], roles["fit"]["labels"])):
        raise AssertionError("high and low fit views do not preserve identity/labels")
    values, _ = read_parameters(source_path)
    all_labels = read_labels(source_path).astype(np.uint8)
    weak_all = compute_weak_component_mask(all_labels, _component_power(values), 10.0)
    for name, values_role in roles.items(): values_role["weak"] = weak_all[values_role["sample_ids"]]
    return roles, {"signals": high["signals"], "labels": high["targets6"], "sample_ids": high["sample_ids"]}


def _comparison(rows: list[dict], candidate: str, baseline: str, offset: int) -> dict:
    metrics = ("target_mean_weak_auroc", "target_mean_weak_aupr", "target_mean_strong_auroc")
    by_seed = {row["source_seed"]: row for row in rows if row["variant"] == candidate}
    base = {row["source_seed"]: row for row in rows if row["variant"] == baseline}
    per_seed = []
    for seed in sorted(by_seed):
        record = {"source_seed": seed}
        for metric in metrics: record[metric] = by_seed[seed]["metrics"][metric] - base[seed]["metrics"][metric]
        record["per_target_weak_auroc"] = {name: by_seed[seed]["metrics"]["per_target"][name]["weak_auroc"] - base[seed]["metrics"]["per_target"][name]["weak_auroc"] for name in TARGET_NAMES}
        per_seed.append(record)
    aggregate = {metric: _mean_ci([row[metric] for row in per_seed], offset + index) for index, metric in enumerate(metrics)}
    aggregate["positive_target_mean_weak_auroc_seeds"] = int(sum(row["target_mean_weak_auroc"] > 0 for row in per_seed))
    aggregate["per_target_weak_auroc"] = {name: _mean_ci([row["per_target_weak_auroc"][name] for row in per_seed], offset + 20 + index) for index, name in enumerate(TARGET_NAMES)}
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def run(args: argparse.Namespace) -> dict:
    if args.output_root.exists(): raise FileExistsError(f"refusing to overwrite H1t: {args.output_root}")
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if not all(source_audit["checks"].values()): raise ValueError("source audit did not pass")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8-sig"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")): raise ValueError("test-v3 seal is not intact")
    low_path, high_path = _view_path(args.source_root, 10), _view_path(args.source_root, 50)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True)
    _write_json(args.output_root / "experiment_config.json", {"experiment": "RouteA_H1t_AsymmetricVisibilityTransfer", "variants": VARIANTS,
        "source_seeds": args.source_seeds, "focal_seeds": args.focal_seeds, "epochs": args.epochs, "patience": args.patience,
        "learning_rate": args.learning_rate, "representation_lr_scale": args.representation_lr_scale, "transfer_weight": args.transfer_weight,
        "temperature": args.temperature, "ranking_only": True, "calibration_performed": False, "threshold_selection_performed": False,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False})
    rows = []
    for source_seed, focal_seed in zip(args.source_seeds, args.focal_seeds):
        role_path = args.source_audit.parent / "roles" / f"seed_{source_seed}_roles.npz"
        roles, high_fit = _load_roles(low_path, high_path, role_path, args.normalization, args.io_batch_size)
        focal_path = args.focal_root / f"seed_{focal_seed}" / "E2_Focal_36" / "best_model.pth"
        state = torch.load(focal_path, map_location="cpu", weights_only=False)["model_state_dict"]
        f2_dir = args.f2_root / "runs" / "F2_DFMS_Full" / f"seed_{focal_seed}"
        teacher = _load_model_from_state(f2_dir, device, state)
        high_logits = _predict(teacher, high_fit["signals"], device, args.eval_batch_size)[:, list(TARGET_COLUMNS)]
        del teacher
        if device.type == "cuda": torch.cuda.empty_cache()
        permuted, permutation = composition_permute_teacher(high_logits, roles["fit"]["labels"], roles["fit"]["weak"])
        seed_root = args.output_root / f"source_seed_{source_seed}__focal_seed_{focal_seed}"
        seed_root.mkdir()
        for variant in VARIANTS:
            model = _load_model_from_state(f2_dir, device, state)
            model, train = _train(model, variant, roles, high_logits, permuted, source_seed, device, args, seed_root / variant)
            early_logits = _predict(model, roles["early"]["signals"], device, args.eval_batch_size)
            metrics = _metrics(roles["early"]["labels"], roles["early"]["weak"], early_logits)
            np.savez_compressed(seed_root / variant / "early_predictions.npz", sample_ids=roles["early"]["sample_ids"], labels=roles["early"]["labels"], weak=roles["early"]["weak"].astype(np.uint8), logits=early_logits)
            row = {"source_seed": int(source_seed), "focal_seed": int(focal_seed), "variant": variant, "training": train, "metrics": metrics,
                   "role_sha256": _sha256(role_path), "focal_checkpoint_sha256": _sha256(focal_path), "high_teacher_prediction_sha256": hashlib.sha256(high_logits.tobytes()).hexdigest(),
                   "permutation": permutation, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
            _write_json(seed_root / variant / "seed_summary.json", row); rows.append(row)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
    t1_b0, t1_p1 = _comparison(rows, "T1_PairedTargetLocalTransfer", "B0_SupervisedFocal10dB", 20340000), _comparison(rows, "T1_PairedTargetLocalTransfer", "P1_CompositionPermutedTeacher", 20341000)
    target_no_drop = all(value["mean"] >= -0.010 for value in t1_b0["aggregate"]["per_target_weak_auroc"].values())
    checks = {"t1_b0_mean_weak_auroc_ge_0p030": t1_b0["aggregate"]["target_mean_weak_auroc"]["mean"] >= .030,
              "t1_b0_positive_weak_auroc_4of5": t1_b0["aggregate"]["positive_target_mean_weak_auroc_seeds"] >= 4,
              "t1_b0_weak_auroc_ci_lower_gt_0": t1_b0["aggregate"]["target_mean_weak_auroc"]["bootstrap_95_ci"][0] > 0,
              "t1_p1_mean_weak_auroc_ge_0p020": t1_p1["aggregate"]["target_mean_weak_auroc"]["mean"] >= .020,
              "t1_p1_positive_weak_auroc_4of5": t1_p1["aggregate"]["positive_target_mean_weak_auroc_seeds"] >= 4,
              "t1_p1_weak_auroc_ci_lower_gt_0": t1_p1["aggregate"]["target_mean_weak_auroc"]["bootstrap_95_ci"][0] > 0,
              "no_target_mean_weak_auroc_drop_below_minus_0p010": target_no_drop,
              "strong_auroc_drop_no_worse_than_minus_0p010": t1_b0["aggregate"]["target_mean_strong_auroc"]["mean"] >= -.010,
              "test_boundary_safe": True}
    result = {"status": "completed", "experiment": "RouteA_H1t_AsymmetricVisibilityTransfer", "ranking_only": True, "rows": rows,
              "t1_vs_b0": t1_b0, "t1_vs_p1": t1_p1, "evidence_gate": {"passed": bool(all(checks.values())), "checks": checks},
              "source_audit_sha256": _sha256(args.source_audit), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(args.output_root / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--f2-root", type=Path, default=DEFAULT_F2_ROOT); parser.add_argument("--focal-root", type=Path, default=DEFAULT_FOCAL_ROOT)
    parser.add_argument("--test-v3-manifest", type=Path, default=DEFAULT_SEAL); parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-seeds", type=int, nargs="+", default=list(SOURCE_SEEDS)); parser.add_argument("--focal-seeds", type=int, nargs="+", default=list(FOCAL_SEEDS))
    parser.add_argument("--epochs", type=int, default=36); parser.add_argument("--patience", type=int, default=8); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--io-batch-size", type=int, default=128); parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4); parser.add_argument("--representation-lr-scale", type=float, default=.1); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip", type=float, default=5.0); parser.add_argument("--transfer-weight", type=float, default=.25); parser.add_argument("--temperature", type=float, default=2.0); parser.add_argument("--normalization", default="complex_rms"); parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
