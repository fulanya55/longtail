"""Read-only full 465-combination error map for the Route-A nine-class screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CJ0724_ROOT = Path(r"E:\LJQ_code\CJ_Inbalance_0724")
if str(CJ0724_ROOT) not in sys.path:
    sys.path.insert(0, str(CJ0724_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from cj0724.models import build_model
from experiments.run_route_a_nineclass_feature_sufficiency import (
    CLASS_NAMES, POWERS_DB, F2HeadWrapper, NineClassDataset, find_file, sha256, write_json,
)


def pattern_code(values: np.ndarray) -> np.ndarray:
    powers = (2 ** np.arange(values.shape[1], dtype=np.int64))[None, :]
    return (np.asarray(values, dtype=np.int64) * powers).sum(axis=1).astype(np.int64)


def model_for(name: str, checkpoint: Path, f2_root: Path, seed: int, device: torch.device):
    if name == "A0_F2_frozen_linear":
        backbone = build_backbone("dfms_iq", num_classes=9, width=16, dropout=0.2)
        f2 = torch.load(str(f2_root / f"seed_{seed}" / "backbone" / "best_model.pth"), map_location="cpu", weights_only=False)
        state = {k[len("backbone."):]: v for k, v in f2["model_state_dict"].items() if k.startswith("backbone.") and not k.startswith("backbone.classifier.")}
        backbone.load_state_dict(state, strict=False)
        head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(backbone.feature_dim, 9))
        model = F2HeadWrapper(backbone, head)
    elif name == "A0b_F2_frozen_mlp":
        backbone = build_backbone("dfms_iq", num_classes=9, width=16, dropout=0.2)
        f2 = torch.load(str(f2_root / f"seed_{seed}" / "backbone" / "best_model.pth"), map_location="cpu", weights_only=False)
        state = {k[len("backbone."):]: v for k, v in f2["model_state_dict"].items() if k.startswith("backbone.") and not k.startswith("backbone.classifier.")}
        backbone.load_state_dict(state, strict=False)
        head = torch.nn.Sequential(torch.nn.Linear(backbone.feature_dim, 128), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 9))
        model = F2HeadWrapper(backbone, head)
    elif name == "A1_F2_finetune":
        model = build_backbone("dfms_iq", num_classes=9, width=16, dropout=0.2)
    elif name == "A2_EfficientNetB1":
        model = build_model("efficientnet_b1_1d", num_classes=9, dropout=0.2)
    elif name == "A3_F2_CSFP":
        model = build_backbone("dfms_iq_csfp", num_classes=9, width=16, dropout=0.2)
    elif name == "A4_F2_LocalStatsK16":
        model = build_backbone("dfms_iq_local_topk16", num_classes=9, width=16, dropout=0.2)
    else:
        raise ValueError(name)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def select_all_combinations(path: Path) -> np.ndarray:
    with h5py.File(str(path), "r") as handle:
        labels = np.asarray(handle["known_label"][:, :], dtype=np.uint8).T
    codes = pattern_code(labels)
    first = {}
    for index, code in enumerate(codes.tolist()):
        first.setdefault(int(code), int(index))
    indices = np.sort(np.asarray([first[code] for code in sorted(first)], dtype=np.int64))
    if len(indices) != 465:
        raise RuntimeError(f"expected 465 covered combinations, got {len(indices)} in {path}")
    return indices


def evaluate(model, path: Path, indices: np.ndarray, device: torch.device):
    dataset = NineClassDataset(path, indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    logits, targets = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                logits.append(model(batch["signal"].to(device, non_blocking=True)).cpu().numpy())
                targets.append(batch["label"].numpy())
    finally:
        dataset.close()
    targets = np.concatenate(targets).astype(np.uint8)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(np.concatenate(logits), -40, 40)))
    predictions = (probabilities >= 0.5).astype(np.uint8)
    true_codes = pattern_code(targets)
    pred_codes = pattern_code(predictions)
    rows = []
    for true_code, pred_code in zip(true_codes.tolist(), pred_codes.tolist()):
        rows.append({"true_code": int(true_code), "predicted_code": int(pred_code), "correct": bool(true_code == pred_code)})
    transitions = {}
    for row in rows:
        key = f"{row['true_code']:03d}->{row['predicted_code']:03d}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "covered_target_combinations": int(len(np.unique(true_codes))),
        "covered_prediction_combinations": int(len(np.unique(pred_codes))),
        "combination_exact_accuracy": float(np.mean(true_codes == pred_codes)),
        "rows": rows,
        "transition_counts": transitions,
    }


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("combination map requires CUDA")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    models = ("A0_F2_frozen_linear", "A0b_F2_frozen_mlp", "A1_F2_finetune", "A2_EfficientNetB1", "A3_F2_CSFP", "A4_F2_LocalStatsK16")
    result = {"schema": "route-a-nineclass-465-combination-map-v1", "locked_test_used": False, "test_files_opened": False, "seed": int(args.seed), "powers_db": list(POWERS_DB), "models": {}}
    for name in models:
        checkpoint = args.model_root / f"{name}__seed_{args.seed}.pth"
        model = model_for(name, checkpoint, args.f2_checkpoint_root, args.seed, device)
        result["models"][name] = {}
        for power in POWERS_DB:
            path = find_file(args.data_root, power, "test")
            indices = select_all_combinations(path)
            result["models"][name][str(power)] = evaluate(model, path, indices, device)
        del model
        torch.cuda.empty_cache()
    write_json(output / "combination_error_map.json", result)
    write_json(output / "terminal_receipt.json", {"status": "completed", "locked_test_used": False, "test_files_opened": False, "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"status": "completed", "output": str((output / 'combination_error_map.json').resolve())}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"))
    parser.add_argument("--f2-checkpoint-root", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_visibility_balanced_fusion_v2")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_nineclass_feature_sufficiency_20260826/model_screen_f2_init_dev_audit")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_nineclass_feature_sufficiency_20260826/combination_audit_seed2039")
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
