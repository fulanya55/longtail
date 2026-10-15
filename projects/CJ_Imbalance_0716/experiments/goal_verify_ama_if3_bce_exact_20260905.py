"""Governed BCE/exact-match verification on the new six-class AMA IF3 data.

This experiment deliberately keeps all tuning on the training MAT.  The test
MAT is loaded only after all checkpoint and threshold parameters have been
frozen.  It audits evaluator equivalence, compares threshold rules, and
compares Macro-AUPR checkpoint selection with exact-first selection.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.engine.combination_metrics import (  # noqa: E402
    compute_cardinality_metrics,
    compute_combination_metrics,
)
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_c1_baseline import initialize_f2  # noqa: E402


DATA_ROOT = Path(
    r"E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA"
)
TRAIN_PATH = DATA_ROOT / "closedset_6class_train.mat"
TEST_PATH = DATA_ROOT / "closedset_6class_test.mat"
AUDIT_PATH = DATA_ROOT / "audit_summary.json"
INIT = Path(
    r"E:\LJQ_code\CJ_Imbalance_0716\frozen\20260731_route_a_visibility_balanced_fusion_v2\seed_2039\backbone\best_model.pth"
)
OUT = ROOT / "outputs" / "goal_ama_if3_bce_exact_20260905"
CLASS_NAMES = ("FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ")
SEEDS = (3050, 3051, 3052, 3053, 3054)
SPLIT_SEED = 20260905
CHECKPOINT_EPSILON = 0.002
EPOCHS = 8
BATCH_SIZE = 32

GOAL_PROMPT = r"""在新六类 AMA IF3/CF8/N8000 数据上验证 BCE 优化目标与组合全正确指标的一致性：严格分离 fit、early-stop、threshold_selection、check 和最终 test；完成评估器一致性、固定 logits 阈值比较、Macro-AUPR 与 exact-first checkpoint 比较；报告 sample exact、frequency-weighted exact、macro-combination exact、cardinality exact、seen/unseen、逐类指标和五种子不确定性；禁止用 test 选阈值或 checkpoint，保留所有哈希、manifest、logits、checkpoint、receipt、失败证据和中文报告。"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40.0, 40.0)))


class ArrayDataset(Dataset):
    def __init__(self, signal: np.ndarray, labels: np.ndarray):
        self.signal = signal
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return torch.from_numpy(self.signal[index]), torch.from_numpy(self.labels[index])


def load_mat(path: Path, indices: np.ndarray | None = None):
    with h5py.File(str(path), "r") as handle:
        n = int(handle["label_selected"].shape[1])
        raw_indices = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        order = np.argsort(raw_indices)
        sorted_indices = raw_indices[order]
        labels = np.asarray(handle["label_selected"][:, sorted_indices], dtype=np.uint8).T
        stored = np.asarray(handle["iq_complex"][:, sorted_indices])
        receiver = np.asarray(handle["receiver_noise_seed"][:, sorted_indices]).reshape(-1)
    signal = np.stack(
        [np.asarray(stored["real"], dtype=np.float32).T,
         np.asarray(stored["imag"], dtype=np.float32).T], axis=1
    )
    rms = np.sqrt(np.mean(signal[:, 0] ** 2 + signal[:, 1] ** 2, axis=1))
    signal = signal / np.sqrt(rms[:, None, None] ** 2 + 1.0e-12)
    inverse = np.argsort(order)
    return (
        np.ascontiguousarray(signal[inverse]),
        labels[inverse].astype(np.float32),
        receiver[inverse].astype(np.int64),
    )


def combo_codes(labels: np.ndarray) -> np.ndarray:
    return labels.astype(np.uint8) @ (1 << np.arange(labels.shape[1], dtype=np.uint8))


def distribution(labels: np.ndarray) -> dict:
    y = labels.astype(np.uint8)
    counts = np.bincount(combo_codes(y).astype(np.int64), minlength=64)[1:]
    active = counts[counts > 0]
    return {
        "n": int(len(y)),
        "label_counts": y.sum(axis=0).astype(int).tolist(),
        "effective_if": float(y.sum(axis=0).max() / y.sum(axis=0).min()),
        "unique_combinations": int(np.count_nonzero(counts)),
        "combo_min": int(active.min()),
        "combo_max": int(active.max()),
        "effective_cf": float(active.max() / active.min()),
        "combo_counts": {str(i + 1): int(v) for i, v in enumerate(counts) if v > 0},
    }


def named_combo_counts(labels: np.ndarray) -> dict:
    counts = {}
    for row in labels.astype(np.uint8):
        key = "+".join(CLASS_NAMES[index] for index in np.flatnonzero(row))
        counts[key] = counts.get(key, 0) + 1
    return counts


def split_roles(labels: np.ndarray) -> dict:
    """Stratified roles; every non-empty combo is present in every role."""
    rng = np.random.default_rng(SPLIT_SEED)
    codes = combo_codes(labels)
    roles = {name: [] for name in ("fit", "early_stop", "threshold_selection", "check")}
    fractions = (0.50, 0.20, 0.15, 0.15)
    for code in sorted(np.unique(codes).tolist()):
        group = np.flatnonzero(codes == code)
        rng.shuffle(group)
        n = len(group)
        cuts = [int(round(n * fractions[0])), int(round(n * (fractions[0] + fractions[1]))), int(round(n * (fractions[0] + fractions[1] + fractions[2])))]
        cuts[0] = max(1, min(cuts[0], n - 3))
        cuts[1] = max(cuts[0] + 1, min(cuts[1], n - 2))
        cuts[2] = max(cuts[1] + 1, min(cuts[2], n - 1))
        pieces = (group[:cuts[0]], group[cuts[0]:cuts[1]], group[cuts[1]:cuts[2]], group[cuts[2]:])
        for name, piece in zip(roles, pieces):
            roles[name].extend(piece.tolist())
    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in roles.items()}


def role_manifest(labels: np.ndarray, receiver: np.ndarray, roles: dict) -> dict:
    seed_sets = {name: set(receiver[idx].tolist()) for name, idx in roles.items()}
    overlap = {}
    names = list(roles)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap[f"{left}__{right}"] = int(len(seed_sets[left] & seed_sets[right]))
    return {
        "schema": "goal-ama-if3-bce-exact-role-manifest-v1",
        "data_root": str(DATA_ROOT),
        "classes": list(CLASS_NAMES),
        "train_sha256": sha256(TRAIN_PATH),
        "selection_seed": SPLIT_SEED,
        "roles": {
            name: {
                "count": int(len(index)),
                "indices": index.tolist(),
                "indices_sha256": hashlib.sha256(index.tobytes()).hexdigest(),
                "distribution": distribution(labels[index]),
                "receiver_noise_seed_count": int(len(seed_sets[name])),
            }
            for name, index in roles.items()
        },
        "receiver_noise_seed_overlaps": overlap,
        "test_role": "not_loaded_until_all_checkpoint_and_threshold_parameters_are_frozen",
        "locked_test_used": False,
    }


def _per_label_thresholds(y: np.ndarray, logits: np.ndarray, objective: str) -> np.ndarray:
    p = sigmoid(logits)
    out = []
    for col in range(y.shape[1]):
        candidates = np.unique(np.concatenate(([0.0], p[:, col], [1.0])))
        scores = []
        for threshold in candidates:
            pred = (p[:, col] >= threshold).astype(np.uint8)
            scores.append(float(f1_score(y[:, col], pred, zero_division=0)))
        scores = np.asarray(scores)
        best = np.flatnonzero(scores == scores.max())
        out.append(float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]]))
    return np.asarray(out, dtype=np.float64)


def _exact_score(y: np.ndarray, logits: np.ndarray, thresholds: np.ndarray, temperature: float = 1.0, bias: np.ndarray | None = None) -> float:
    z = logits.astype(np.float64) / float(temperature)
    if bias is not None:
        z = z + np.asarray(bias, dtype=np.float64)[None, :]
    pred = (sigmoid(z) >= np.asarray(thresholds)[None, :]).astype(np.uint8)
    return float(np.mean(np.all(pred == y.astype(np.uint8), axis=1)))


def global_exact_threshold(y: np.ndarray, logits: np.ndarray) -> np.ndarray:
    p = sigmoid(logits)
    candidates = np.unique(np.concatenate(([0.05, 0.5, 0.95], p.reshape(-1))))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    scores = np.asarray([_exact_score(y, logits, np.full(y.shape[1], t)) for t in candidates])
    best = np.flatnonzero(scores == scores.max())
    return np.full(y.shape[1], float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]]))


def coordinate_exact_thresholds(y: np.ndarray, logits: np.ndarray, passes: int = 3) -> np.ndarray:
    p = sigmoid(logits)
    thresholds = np.full(y.shape[1], 0.5, dtype=np.float64)
    candidates = [np.unique(np.concatenate(([0.05, 0.5, 0.95], p[:, col]))) for col in range(y.shape[1])]
    for _ in range(passes):
        changed = False
        for col in range(y.shape[1]):
            scores = []
            for threshold in candidates[col]:
                trial = thresholds.copy(); trial[col] = float(threshold)
                scores.append(_exact_score(y, logits, trial))
            scores = np.asarray(scores)
            best = np.flatnonzero(scores == scores.max())
            chosen = float(candidates[col][best[np.argmin(np.abs(candidates[col][best] - 0.5))]])
            changed = changed or chosen != thresholds[col]
            thresholds[col] = chosen
        if not changed:
            break
    return thresholds


def bce_loss(y: np.ndarray, logits: np.ndarray) -> float:
    z = np.clip(logits.astype(np.float64), -40.0, 40.0)
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(y: np.ndarray, logits: np.ndarray) -> float:
    opt = minimize_scalar(lambda log_t: bce_loss(y, logits / math.exp(float(log_t))), bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(float(opt.x)))


def fit_logit_bias(y: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    base = logits / float(temperature)
    result = []
    for col in range(y.shape[1]):
        opt = minimize_scalar(lambda bias: bce_loss(y[:, col:col + 1], base[:, col:col + 1] + float(bias)), bounds=(-8.0, 8.0), method="bounded")
        result.append(float(opt.x))
    return np.asarray(result, dtype=np.float64)


def evaluate(y: np.ndarray, logits: np.ndarray, thresholds: np.ndarray | None = None, temperature: float = 1.0, bias: np.ndarray | None = None, train_combo_counts: dict | None = None) -> dict:
    y_uint = y.astype(np.uint8)
    z = logits.astype(np.float64) / float(temperature)
    if bias is not None:
        z = z + np.asarray(bias, dtype=np.float64)[None, :]
    probabilities = sigmoid(z)
    th = np.full(y.shape[1], 0.5, dtype=np.float64) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    predictions = (probabilities >= th[None, :]).astype(np.uint8)
    exact_vector = np.all(predictions == y_uint, axis=1)
    powers = (1 << np.arange(y.shape[1], dtype=np.int64))[None, :]
    true_codes = (y_uint.astype(np.int64) * powers).sum(axis=1)
    pred_codes = (predictions.astype(np.int64) * powers).sum(axis=1)
    per_label = {}
    aupr, auroc = [], []
    for col, name in enumerate(CLASS_NAMES):
        truth = y_uint[:, col]
        pos, neg = truth == 1, truth == 0
        aupr_value = float(average_precision_score(truth, probabilities[:, col]))
        auroc_value = float(roc_auc_score(truth, probabilities[:, col])) if len(np.unique(truth)) == 2 else float("nan")
        per_label[name] = {
            "aupr": aupr_value,
            "auroc": auroc_value,
            "recall": float(predictions[pos, col].mean()) if pos.any() else float("nan"),
            "negative_fpr": float(predictions[neg, col].mean()) if neg.any() else float("nan"),
            "f1": float(f1_score(truth, predictions[:, col], zero_division=0)),
            "positive_count": int(pos.sum()),
        }
        aupr.append(aupr_value); auroc.append(auroc_value)
    combination = compute_combination_metrics(y_uint, predictions, CLASS_NAMES, train_combo_counts=train_combo_counts)
    cardinality = compute_cardinality_metrics(y_uint, predictions)
    rows = combination["per_combination"]
    frequency_weighted = float(sum(row["exact_match_accuracy"] * row["evaluation_count"] for row in rows) / len(y_uint)) if rows else float("nan")
    consistency = {
        "subset_accuracy": float(np.mean(exact_vector)),
        "combination_exact_accuracy": float(np.mean(true_codes == pred_codes)),
        "numpy_all_equal": float(np.mean(np.all(predictions == y_uint, axis=1))),
        "absolute_max_difference": float(max(abs(float(np.mean(exact_vector)) - float(np.mean(true_codes == pred_codes))), abs(float(np.mean(exact_vector)) - float(np.mean(np.all(predictions == y_uint, axis=1))))))
    }
    if consistency["absolute_max_difference"] > 0.0:
        raise AssertionError(f"evaluator inconsistency: {consistency}")
    return {
        "macro_aupr": float(np.nanmean(aupr)),
        "macro_auroc": float(np.nanmean(auroc)),
        "macro_f1": float(f1_score(y_uint, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(np.mean(predictions != y_uint)),
        "sample_exact_accuracy": consistency["subset_accuracy"],
        "frequency_weighted_combination_accuracy": frequency_weighted,
        "macro_combination_accuracy": float(combination["summary"]["macro_combination_accuracy"]),
        "cardinality_stratified_exact": {str(row["cardinality"]): row["subset_accuracy"] for row in cardinality["per_cardinality"]},
        "seen_combination_accuracy": combination["summary"].get("seen_combination_accuracy"),
        "unseen_combination_accuracy": combination["summary"].get("unseen_combination_accuracy"),
        "consistency": consistency,
        "combination_count": combination["summary"]["combination_count"],
        "per_label": per_label,
    }


def predict(model: nn.Module, signal: np.ndarray, labels: np.ndarray, device: torch.device):
    loader = DataLoader(ArrayDataset(signal, labels), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    logits = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            logits.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return labels.astype(np.uint8), np.concatenate(logits).astype(np.float32)


def checkpoint_choice(history: list[dict], strategy: str) -> int:
    aupr = np.asarray([row["validation"]["macro_aupr"] for row in history], dtype=np.float64)
    exact = np.asarray([row["validation"]["sample_exact_accuracy"] for row in history], dtype=np.float64)
    if strategy == "macro_aupr":
        return int(np.flatnonzero(aupr == aupr.max())[0] + 1)
    if strategy == "exact_first_constrained":
        eligible = np.flatnonzero(aupr >= aupr.max() - CHECKPOINT_EPSILON)
        best_exact = exact[eligible].max()
        tied = eligible[exact[eligible] == best_exact]
        return int(tied[np.argmin(-aupr[tied])] + 1)
    raise ValueError(strategy)


def threshold_parameters(y: np.ndarray, logits: np.ndarray) -> dict:
    f1 = _per_label_thresholds(y, logits, "f1")
    global_th = global_exact_threshold(y, logits)
    combo = coordinate_exact_thresholds(y, logits)
    temp = fit_temperature(y, logits)
    bias = fit_logit_bias(y, logits)
    temp_bias = fit_logit_bias(y, logits, temp)
    return {
        "fixed_0.5": {"thresholds": np.full(6, 0.5), "temperature": 1.0, "bias": np.zeros(6)},
        "global_exact_threshold": {"thresholds": global_th, "temperature": 1.0, "bias": np.zeros(6)},
        "per_label_f1_threshold": {"thresholds": f1, "temperature": 1.0, "bias": np.zeros(6)},
        "combo_exact_coordinate_threshold": {"thresholds": combo, "temperature": 1.0, "bias": np.zeros(6)},
        "temperature_scaling": {"thresholds": np.full(6, 0.5), "temperature": temp, "bias": np.zeros(6)},
        "logit_bias_bce": {"thresholds": np.full(6, 0.5), "temperature": 1.0, "bias": bias},
        "temperature_then_logit_bias_bce": {"thresholds": np.full(6, 0.5), "temperature": temp, "bias": temp_bias},
    }


def serialize_parameters(parameters: dict) -> dict:
    return {name: {"thresholds": np.asarray(value["thresholds"]).tolist(), "temperature": float(value["temperature"]), "bias": np.asarray(value["bias"]).tolist()} for name, value in parameters.items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "goal_prompt.md").write_text("# Goal prompt\n\n" + GOAL_PROMPT + "\n", encoding="utf-8")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    train_signal, train_labels, train_receiver = load_mat(TRAIN_PATH)
    if train_labels.shape != (8000, 6) or len(np.unique(combo_codes(train_labels))) != 63:
        raise AssertionError("new six-class train schema/coverage check failed")
    roles = split_roles(train_labels)
    manifest = role_manifest(train_labels, train_receiver, roles)
    manifest["expected_test_sha256"] = audit["files"]["closedset_6class_test.mat"]["sha256"]
    manifest["expected_train_sha256"] = audit["files"]["closedset_6class_train.mat"]["sha256"]
    (OUT / "role_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "schema": "goal-ama-if3-bce-exact-config-v1",
        "goal_prompt": GOAL_PROMPT,
        "data_root": str(DATA_ROOT),
        "classes": list(CLASS_NAMES),
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "checkpoint_epsilon_macro_aupr": CHECKPOINT_EPSILON,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss": "BCEWithLogitsLoss",
        "checkpoint_strategies": ["macro_aupr", "exact_first_constrained"],
        "test_policy": "load only after all checkpoint and threshold parameters are frozen",
        "locked_test_used": False,
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device("cuda:0")
    fit_idx, early_idx, threshold_idx, check_idx = (roles[k] for k in ("fit", "early_stop", "threshold_selection", "check"))
    fx, fy, _ = train_signal[fit_idx], train_labels[fit_idx], train_receiver[fit_idx]
    ex, ey, _ = train_signal[early_idx], train_labels[early_idx], train_receiver[early_idx]
    qx, qy, _ = train_signal[threshold_idx], train_labels[threshold_idx], train_receiver[threshold_idx]
    cx, cy, _ = train_signal[check_idx], train_labels[check_idx], train_receiver[check_idx]
    train_combo_counts = named_combo_counts(fy)
    all_rows = []
    frozen = []
    for seed in SEEDS:
        set_seed(seed)
        model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
        init_meta = initialize_f2(model, INIT)
        loader = DataLoader(ArrayDataset(fx, fy), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        seed_dir = OUT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        history = []
        epoch_logits = {}
        started = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train(); losses = []
            for x, y in loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            ey_uint, early_logits = predict(model, ex, ey, device)
            early_metric = evaluate(ey_uint, early_logits, train_combo_counts=train_combo_counts)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": early_metric})
            epoch_logits[str(epoch)] = early_logits
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation": early_metric}, seed_dir / f"epoch_{epoch:02d}.pth")
            print(json.dumps({"seed": seed, "epoch": epoch, "macro_aupr": early_metric["macro_aupr"], "exact": early_metric["sample_exact_accuracy"]}, ensure_ascii=False), flush=True)
        np.savez_compressed(seed_dir / "early_stop_logits.npz", targets=ey_uint, **{f"epoch_{key}": value for key, value in epoch_logits.items()})
        choices = {strategy: checkpoint_choice(history, strategy) for strategy in ("macro_aupr", "exact_first_constrained")}
        for strategy, epoch in choices.items():
            payload = torch.load(seed_dir / f"epoch_{epoch:02d}.pth", map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            qy_uint, threshold_logits = predict(model, qx, qy, device)
            cy_uint, check_logits = predict(model, cx, cy, device)
            params = threshold_parameters(qy_uint, threshold_logits)
            method_rows = {}
            for method, pars in params.items():
                method_rows[method] = {
                    "parameters": {"thresholds": np.asarray(pars["thresholds"]).tolist(), "temperature": float(pars["temperature"]), "bias": np.asarray(pars["bias"]).tolist()},
                    "threshold_selection": evaluate(qy_uint, threshold_logits, **pars, train_combo_counts=train_combo_counts),
                    "check": evaluate(cy_uint, check_logits, **pars, train_combo_counts=train_combo_counts),
                }
            frozen.append({"seed": int(seed), "strategy": strategy, "checkpoint_epoch": int(epoch), "checkpoint_sha256": sha256(seed_dir / f"epoch_{epoch:02d}.pth"), "threshold_methods": method_rows, "elapsed_seconds": float(time.time() - started), "initialization": init_meta})
        all_rows.append({"seed": int(seed), "history": history, "checkpoint_choices": choices, "fit_distribution": distribution(fy), "early_distribution": distribution(ey), "threshold_selection_distribution": distribution(qy), "check_distribution": distribution(cy)})
        print(json.dumps({"seed": seed, "checkpoint_choices": choices}, ensure_ascii=False), flush=True)
    # No test MAT has been loaded before this point. Parameters are now frozen.
    test_signal, test_labels, test_receiver = load_mat(TEST_PATH)
    if sha256(TEST_PATH) != manifest["expected_test_sha256"]:
        raise AssertionError("test SHA-256 differs from generation audit")
    if len(set(train_receiver.tolist()) & set(test_receiver.tolist())) != 0:
        raise AssertionError("train/test receiver_noise_seed overlap")
    test_rows = []
    for frozen_row in frozen:
        seed = int(frozen_row["seed"]); strategy = frozen_row["strategy"]; epoch = int(frozen_row["checkpoint_epoch"])
        seed_dir = OUT / f"seed_{seed}"
        model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
        model.load_state_dict(torch.load(seed_dir / f"epoch_{epoch:02d}.pth", map_location=device, weights_only=False)["model_state_dict"])
        ty, test_logits = predict(model, test_signal, test_labels, device)
        np.savez_compressed(seed_dir / f"test_logits_{strategy}.npz", targets=ty, logits=test_logits)
        for method, method_row in frozen_row["threshold_methods"].items():
            pars = method_row["parameters"]
            test_rows.append({"seed": seed, "strategy": strategy, "checkpoint_epoch": epoch, "method": method, "test": evaluate(ty, test_logits, np.asarray(pars["thresholds"]), float(pars["temperature"]), np.asarray(pars["bias"]), train_combo_counts=train_combo_counts)})
    summary = {
        "schema": "goal-ama-if3-bce-exact-summary-v1",
        "status": "completed",
        "data_root": str(DATA_ROOT),
        "train_sha256": sha256(TRAIN_PATH),
        "test_sha256": sha256(TEST_PATH),
        "classes": list(CLASS_NAMES),
        "seeds": list(SEEDS),
        "checkpoint_epsilon_macro_aupr": CHECKPOINT_EPSILON,
        "role_manifest": str((OUT / "role_manifest.json").resolve()),
        "training_rows": all_rows,
        "frozen_rows": frozen,
        "test_rows": test_rows,
        "test_loaded_after_freeze": True,
        "locked_test_used": False,
        "evidence_boundary": "new-v2-six-class-development; final test loaded once after freeze; no threshold/checkpoint tuning on test",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "terminal_receipt.json").write_text(json.dumps({"status": "completed", "summary": str((OUT / "summary.json").resolve()), "locked_test_used": False, "test_loaded_after_freeze": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": str((OUT / "summary.json").resolve()), "frozen_models": len(frozen), "test_rows": len(test_rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
