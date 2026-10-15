"""P3g: low-power visibility auxiliary objective on the fixed Route-A F2 backbone.

The classifier is trained on the fixed 30 dB anchor.  A shared auxiliary head
is trained only from the paired 10 dB view and the generator-provided
weak-component visibility mask.  This is a bounded observability probe, not an
expert or a power-dependent inference gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import (
    CLASS_NAMES,
    SELECTED_INDICES,
    NineClassDataset,
    metrics,
    pure_six_indices,
)
from experiments.run_route_a_nineclass_feature_sufficiency import (
    POWERS_DB,
    find_file,
    profile_cost,
    receiver_seeds,
    sha256,
    write_json,
)
from experiments.run_route_a_p3_dual_axis_longtail import load_shared_f2, set_seed

VARIANTS = ("F2_BCE_30dB", "F2_BCE_low_visibility")
SEEDS = (2039, 2040, 2041)


def visibility_targets(path: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return active-label visibility targets and masks for a MAT/HDF5 view."""
    indices = np.asarray(indices, dtype=np.int64)
    with h5py.File(str(path), "r") as h:
        labels = np.asarray(
            h["known_label"][list(SELECTED_INDICES), :][:, indices],
            dtype=np.float32,
        ).T
        refs = h["weak_component_mask"][0, indices]
        target = np.zeros_like(labels, dtype=np.float32)
        mask = np.zeros_like(labels, dtype=np.float32)
        for row, ref in enumerate(refs):
            weak = np.asarray(h[ref][:], dtype=np.float32).reshape(-1)
            active = np.flatnonzero(labels[row] > 0)
            if len(weak) != len(active):
                raise RuntimeError(
                    f"weak mask alignment mismatch at {indices[row]}: "
                    f"{len(weak)} vs {len(active)}"
                )
            # Existing P2c convention: weak < 0.5 means the active component
            # is visible enough for the auxiliary target.
            target[row, active] = (weak < 0.5).astype(np.float32)
            mask[row, active] = 1.0
    return target, mask


class PairedLowVisibilityDataset(Dataset):
    def __init__(self, path30: Path, path10: Path, indices: np.ndarray):
        selected = np.asarray(SELECTED_INDICES, dtype=np.int64)
        self.base30 = NineClassDataset(path30, indices, label_indices=selected)
        self.low = NineClassDataset(path10, indices, label_indices=selected)
        self.visibility_target, self.visibility_mask = visibility_targets(path10, indices)

    def __len__(self) -> int:
        return len(self.base30)

    def __getitem__(self, index: int) -> dict[str, object]:
        a = self.base30[int(index)]
        b = self.low[int(index)]
        return {
            "signal_30": a["signal"],
            "signal_low": b["signal"],
            "label": a["label"],
            "visibility_target": torch.from_numpy(self.visibility_target[index]),
            "visibility_mask": torch.from_numpy(self.visibility_mask[index]),
        }

    def close(self) -> None:
        self.base30.close()
        self.low.close()


class AuxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone(
            "dfms_iq", num_classes=6, width=16, dropout=0.2
        )
        self.visibility_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 6))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        logits, features = self.backbone(x, return_features=True)
        aux = self.visibility_head(features)
        return (logits, aux) if return_aux else logits


def predict(model, path, indices, device, batch_size):
    ds = NineClassDataset(
        path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64)
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    ys, logits = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in dl:
                logits.append(
                    model(batch["signal"].to(device, non_blocking=True))
                    .cpu()
                    .numpy()
                )
                ys.append(batch["label"].numpy())
    finally:
        ds.close()
    return np.concatenate(ys).astype(np.uint8), np.concatenate(logits).astype(np.float32)


def run_one(args, name, seed, paths, fit, val, power_dev, out):
    set_seed(seed)
    device = args.device
    model = AuxModel().to(device)
    init = load_shared_f2(model.backbone, args.f2_checkpoint)
    ds = PairedLowVisibilityDataset(paths[30], paths[10], fit)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    params = [p for p in model.parameters() if p.requires_grad]
    cls_loss = nn.BCEWithLogitsLoss()
    history, best, best_epoch = [], -1.0, 0
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, cls_losses, aux_losses = [], [], []
            epoch_cosines, epoch_conflicts = [], []
            for batch in dl:
                x30 = batch["signal_30"].to(device, non_blocking=True)
                xlow = batch["signal_low"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits30, _ = model(x30, return_aux=True)
                lc = cls_loss(logits30, y)
                if name == "F2_BCE_low_visibility":
                    _, aux_low = model(xlow, return_aux=True)
                    mask = batch["visibility_mask"].to(device, non_blocking=True).bool()
                    target = batch["visibility_target"].to(device, non_blocking=True)
                    la = (
                        nn.functional.binary_cross_entropy_with_logits(
                            aux_low[mask], target[mask]
                        )
                        if bool(mask.any())
                        else aux_low.sum() * 0.0
                    )
                    loss = lc + args.aux_weight * la
                else:
                    la = logits30.sum() * 0.0
                    loss = lc
                if name == "F2_BCE_low_visibility" and args.aux_mode == "anchor_projected":
                    # Inject the auxiliary gradient only after projecting away
                    # any component that opposes the 30 dB classification
                    # gradient. This protects the fixed anchor without adding
                    # an inference-time expert or threshold.
                    g_cls = torch.autograd.grad(
                        lc, params, retain_graph=True, allow_unused=True
                    )
                    g_aux = torch.autograd.grad(
                        args.aux_weight * la,
                        params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    def nz(g, p):
                        return torch.zeros_like(p) if g is None else g
                    dot = sum(
                        torch.sum(nz(a, p) * nz(b, p))
                        for p, a, b in zip(params, g_cls, g_aux)
                    )
                    n_cls = sum(torch.sum(nz(a, p) ** 2) for p, a in zip(params, g_cls))
                    n_aux = sum(torch.sum(nz(a, p) ** 2) for p, a in zip(params, g_aux))
                    cosine = dot / (torch.sqrt(n_cls * n_aux) + 1e-12)
                    conflict = bool((dot < 0).detach().cpu())
                    epoch_cosines.append(float(cosine.detach().cpu()))
                    epoch_conflicts.append(float(conflict))
                    if conflict and float(n_cls.detach().cpu()) > 0:
                        g_aux = [
                            nz(a, p) - dot / n_cls * nz(c, p)
                            for p, a, c in zip(params, g_aux, g_cls)
                        ]
                    for p, a, c in zip(params, g_aux, g_cls):
                        p.grad = nz(c, p) + nz(a, p)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                cls_losses.append(float(lc.detach().cpu()))
                aux_losses.append(float(la.detach().cpu()))

            yv, lv = predict(model, paths[30], val, device, args.eval_batch_size)
            vm = metrics(yv, lv)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "train_cls_loss": float(np.mean(cls_losses)),
                    "train_aux_loss": float(np.mean(aux_losses)),
                    "anchor_gradient_cosine_mean": float(np.mean(epoch_cosines))
                    if epoch_cosines
                    else None,
                    "anchor_gradient_conflict_fraction": float(np.mean(epoch_conflicts))
                    if epoch_conflicts
                    else None,
                    "validation": vm,
                }
            )
            if vm["macro_aupr"] > best:
                best = vm["macro_aupr"]
                best_epoch = epoch
                torch.save(model.state_dict(), out / f"{name}__seed_{seed}.pth")
    finally:
        ds.close()

    model.load_state_dict(
        torch.load(
            str(out / f"{name}__seed_{seed}.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    result = {
        "variant": name,
        "seed": seed,
        "history": history,
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1]["validation"],
        "power_dev": {},
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "cost_profile": profile_cost(model.backbone, device),
        "initialization": init,
        "aux_weight": args.aux_weight,
        "aux_mode": args.aux_mode,
        "low_visibility_power_db": args.low_power,
        "locked_test_used": False,
        "elapsed_seconds": time.time() - started,
    }
    for power, (path, indices) in power_dev.items():
        yp, lp = predict(model, path, indices, device, args.eval_batch_size)
        result["power_dev"][str(power)] = metrics(yp, lp)
    result["checkpoint_sha256"] = sha256(out / f"{name}__seed_{seed}.pth")
    write_json(out / f"{name}__seed_{seed}.json", result)
    print(
        json.dumps(
            {
                "variant": name,
                "seed": seed,
                "val_macro_aupr": result["validation"]["macro_aupr"],
                "val_combo": result["validation"]["combination_exact_accuracy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("P3g requires CUDA")
    args.device = device
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {p: find_file(args.data_root, p, "train") for p in POWERS_DB}
    base = paths[30]
    pure = pure_six_indices(base)
    perm = np.random.default_rng(args.split_seed).permutation(pure)
    fit = np.sort(perm[: args.fit_samples])
    val = np.sort(perm[args.fit_samples : args.fit_samples + args.val_samples])
    used = set(receiver_seeds(base, np.sort(np.concatenate([fit, val]))))
    power_dev = {}
    for power in POWERS_DB:
        ppath = paths[power]
        pidx = pure_six_indices(ppath)
        rs = np.asarray(receiver_seeds(ppath, pidx), dtype=np.int64)
        candidates = pidx[~np.isin(rs, np.fromiter(used, dtype=np.int64))]
        rng = np.random.default_rng(args.split_seed + power)
        indices = np.sort(
            rng.choice(
                candidates,
                size=min(args.power_dev_samples, len(candidates)),
                replace=False,
            )
        )
        power_dev[power] = (ppath, indices)

    sample_id_sha256 = __import__(
        "experiments.run_route_a_p3_dual_axis_longtail",
        fromlist=["sample_id_sha256"],
    ).sample_id_sha256
    role = {
        "schema": "route-a-p3g-low-visibility-aux-role-manifest-v1",
        "selected_classes": list(CLASS_NAMES),
        "backbone": "F2_DFMS_Full/dfms_iq",
        "paired_power_views": [10, 30, 50],
        "pairing_key": "receiver_noise_seed and identical sample order across train views",
        "low_visibility_source": {
            "power_db": args.low_power,
            "field": "weak_component_mask",
            "target_rule": "active selected label and weak_component_mask < 0.5",
        },
        "fit": {
            "path": str(base),
            "power_db": 30,
            "count": int(len(fit)),
            "sample_indices_sha256": sample_id_sha256(fit),
            "receiver_noise_seed": sorted(set(receiver_seeds(base, fit))),
        },
        "validation": {
            "path": str(base),
            "power_db": 30,
            "count": int(len(val)),
            "sample_indices_sha256": sample_id_sha256(val),
            "receiver_noise_seed": sorted(set(receiver_seeds(base, val))),
        },
        "power_development": {
            "source_role": "train_only_unseen_identity",
            "count_by_power": {str(p): len(i) for p, (_, i) in power_dev.items()},
        },
        "test": {"used": False, "locked_test_used": False},
    }
    write_json(args.output_root / "role_manifest.json", role)

    rows = []
    for seed in args.seeds:
        for name in VARIANTS:
            out = args.output_root / name
            out.mkdir(parents=True, exist_ok=True)
            rows.append(run_one(args, name, int(seed), paths, fit, val, power_dev, out))

    summary = {
        "schema": "route-a-p3g-low-visibility-aux-summary-v1",
        "status": "completed",
        "locked_test_used": False,
        "protocol": {
            "train_power_db": 30,
            "low_visibility_power_db": args.low_power,
            "epochs": args.epochs,
            "aux_weight": args.aux_weight,
            "aux_mode": args.aux_mode,
            "seeds": list(args.seeds),
        },
        "variants": {},
    }
    for name in VARIANTS:
        records = [r for r in rows if r["variant"] == name]
        summary["variants"][name] = {
            "parameters": records[0]["total_parameters"],
            "val_macro_aupr_mean": float(
                np.mean([r["validation"]["macro_aupr"] for r in records])
            ),
            "val_macro_aupr_std": float(
                np.std([r["validation"]["macro_aupr"] for r in records], ddof=1)
            ),
            "val_macro_f1_mean": float(
                np.mean([r["validation"]["macro_f1"] for r in records])
            ),
            "val_combo_mean": float(
                np.mean(
                    [r["validation"]["combination_exact_accuracy"] for r in records]
                )
            ),
            "power_dev_macro_aupr_mean_by_power": {
                str(power): float(
                    np.mean([r["power_dev"][str(power)]["macro_aupr"] for r in records])
                )
                for power in POWERS_DB
            },
            "power_dev_combo_mean_by_power": {
                str(power): float(
                    np.mean(
                        [
                            r["power_dev"][str(power)][
                                "combination_exact_accuracy"
                            ]
                            for r in records
                        ]
                    )
                )
                for power in POWERS_DB
            },
        }
    write_json(args.output_root / "p3g_low_visibility_aux_summary.json", summary)
    write_json(
        args.output_root / "terminal_receipt.json",
        {
            "status": "completed",
            "summary": str(
                (args.output_root / "p3g_low_visibility_aux_summary.json").resolve()
            ),
            "locked_test_used": False,
            "test_files_opened": False,
            "script_sha256": sha256(Path(__file__)),
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "summary": str(
                    args.output_root / "p3g_low_visibility_aux_summary.json"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/route_a_p3g_low_visibility_aux_20260905",
    )
    parser.add_argument(
        "--f2-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/route_a_c1_baseline_full_20260826/best_model.pth",
    )
    parser.add_argument("--low-power", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--split-seed", type=int, default=260905)
    parser.add_argument("--fit-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--power-dev-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--aux-mode", choices=("direct", "anchor_projected"), default="direct"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
