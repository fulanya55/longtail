"""P3 development screen: induced dual-axis long-tail and bounded losses.

This runner never opens test or locked-test files.  It uses the audited pure
six-class 30 dB train view, creates one deterministic mixed label/combination
long-tail manifest per seed, and evaluates BCE, class-balanced BCE, and a
bounded dual-axis (class + weak-component) weighted BCE on exactly the same
selected samples.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from cjlt.data.longtail_manifest import build_mixed_longtail, compute_weak_component_mask, profile_multi_axis_imbalance, sample_id_sha256
from experiments.run_route_a_c1_baseline import (
    CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
)
from experiments.run_route_a_nineclass_feature_sufficiency import (
    POWERS_DB, find_file, profile_cost, receiver_seeds, sha256, write_json
)

VARIANTS = ("F2_BCE", "F2_class_balanced", "F2_dual_axis")
SEEDS = (2039, 2040, 2041)


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def component_power_matrix(path: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode per-active-component effective JNR into six aligned columns."""
    indices = np.asarray(indices, dtype=np.int64)
    with h5py.File(str(path), "r") as h:
        labels9 = np.asarray(h["known_label"][:, indices], dtype=np.uint8).T
        refs = h["component_effective_JNR_dB"][0, indices]
        labels6 = labels9[:, list(SELECTED_INDICES)]
        power6 = np.zeros_like(labels6, dtype=np.float64)
        for row, ref in enumerate(refs):
            values = np.asarray(h[ref][:], dtype=np.float64).reshape(-1)
            active = np.flatnonzero(labels9[row])
            if len(values) != len(active):
                raise RuntimeError(f"component power alignment mismatch at {indices[row]}: {len(values)} vs {len(active)}")
            for pos, global_index in enumerate(active.tolist()):
                if global_index in SELECTED_INDICES:
                    power6[row, list(SELECTED_INDICES).index(global_index)] = values[pos]
    return labels6, power6


def predict(model, path: Path, indices: np.ndarray, device: torch.device, batch_size: int):
    ds = NineClassDataset(path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    labels, logits = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in dl:
                logits.append(model(batch["signal"].to(device, non_blocking=True)).cpu().numpy())
                labels.append(batch["label"].numpy())
    finally:
        ds.close()
    return np.concatenate(labels).astype(np.uint8), np.concatenate(logits).astype(np.float32)


def load_shared_f2(model: nn.Module, checkpoint: Path) -> dict:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    shared = {}
    for key, value in state.items():
        if key.startswith("backbone."): key = key[len("backbone."):]
        if key.startswith("classifier."): continue
        shared[key] = value
    missing, unexpected = model.load_state_dict(shared, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    unexpected = [k for k in unexpected if not k.startswith(("label_feature_bank.", "prototype_memory."))]
    if missing or unexpected:
        raise RuntimeError(f"F2 initialization mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "checkpoint_best_epoch": payload.get("best_epoch")}


def loss_for(name: str, logits: torch.Tensor, targets: torch.Tensor, class_weight: torch.Tensor | None, weak: torch.Tensor | None):
    if name == "F2_BCE":
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)
    weight = class_weight.to(logits.device).view(1, -1).expand_as(logits)
    if name == "F2_dual_axis" and weak is not None:
        weight = weight * (1.0 + 0.5 * weak.to(logits.device))
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight)


def run_one(args, name: str, seed: int, train_path: Path, train_indices: np.ndarray,
            weak_mask: np.ndarray, val_indices: np.ndarray, power_dev: dict[int, tuple[Path, np.ndarray]],
            output: Path, manifest_meta: dict) -> dict:
    set_seed(seed)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(args.device)
    init = load_shared_f2(model, args.f2_checkpoint)
    ds = NineClassDataset(train_path, train_indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = ds.labels.sum(axis=0).astype(np.float64)
    if name == "F2_class_balanced":
        cw = np.sqrt(np.maximum(counts.max(), 1.0) / np.maximum(counts, 1.0))
        cw = np.clip(cw, 0.5, 4.0); cw = cw / cw.mean()
    elif name == "F2_dual_axis":
        cw = np.sqrt(np.maximum(counts.max(), 1.0) / np.maximum(counts, 1.0))
        cw = np.clip(cw, 0.5, 4.0); cw = cw / cw.mean()
    else: cw = np.ones(6, dtype=np.float32)
    cw_t = torch.from_numpy(cw.astype(np.float32))
    weak_t = torch.from_numpy(weak_mask.astype(np.float32))
    history, best, best_epoch = [], -1.0, 0
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []; grad_norms = []
            order = np.random.permutation(len(train_indices))
            # DataLoader shuffling is retained; weak masks are indexed by local batch sample_index.
            for batch in dl:
                x = batch["signal"].to(args.device, non_blocking=True); y = batch["label"].to(args.device, non_blocking=True)
                local = np.asarray([int(np.flatnonzero(train_indices == int(v))[0]) for v in batch["sample_index"]], dtype=np.int64)
                weak = weak_t[local]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_for(name, logits, y, cw_t, weak if name == "F2_dual_axis" else None)
                loss.backward(); grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))); optimizer.step(); losses.append(float(loss.detach().cpu()))
            yt, yl = predict(model, train_path, val_indices, args.device, args.eval_batch_size)
            vm = metrics(yt, yl)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "grad_norm_mean": float(np.mean(grad_norms)), "validation": vm}
            history.append(row)
            if vm["macro_aupr"] > best:
                best, best_epoch = vm["macro_aupr"], epoch
                torch.save(model.state_dict(), output / f"{name}__seed_{seed}.pth")
    finally: ds.close()
    model.load_state_dict(torch.load(str(output / f"{name}__seed_{seed}.pth"), map_location=args.device, weights_only=True))
    result = {"variant": name, "seed": seed, "manifest": manifest_meta, "initialization": init,
              "history": history, "best_epoch": best_epoch, "validation": history[best_epoch-1]["validation"],
              "label_counts": counts.astype(int).tolist(), "class_weight": cw.astype(float).tolist(),
              "weak_counts": weak_mask.sum(axis=0).astype(int).tolist(), "cost_profile": profile_cost(model, args.device),
              "total_parameters": int(sum(p.numel() for p in model.parameters())), "power_dev": {},
              "locked_test_used": False, "elapsed_seconds": time.time()-started}
    for power, (path, idx) in power_dev.items():
        yt, yl = predict(model, path, idx, args.device, args.eval_batch_size); result["power_dev"][str(power)] = metrics(yt, yl)
    result["checkpoint_sha256"] = sha256(output / f"{name}__seed_{seed}.pth")
    write_json(output / f"{name}__seed_{seed}.json", result)
    print(json.dumps({"variant": name, "seed": seed, "val_macro_aupr": result["validation"]["macro_aupr"], "val_combo": result["validation"]["combination_exact_accuracy"]}, ensure_ascii=False), flush=True)
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda": raise RuntimeError("P3 requires CUDA")
    args.device = device
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_path = find_file(args.data_root, 30, "train")
    pure = pure_six_indices(train_path)
    perm = np.random.default_rng(args.split_seed).permutation(pure)
    fit = np.sort(perm[:args.fit_samples]); val = np.sort(perm[args.fit_samples:args.fit_samples+args.val_samples])
    labels6, power6 = component_power_matrix(train_path, fit)
    source_sha = sample_id_sha256(fit)
    used = set(receiver_seeds(train_path, np.sort(np.concatenate([fit, val]))))
    power_dev = {}
    for power in POWERS_DB:
        path = find_file(args.data_root, power, "train"); pp = pure_six_indices(path); seeds = np.asarray(receiver_seeds(path, pp), dtype=np.int64)
        cand = pp[~np.isin(seeds, np.fromiter(used, dtype=np.int64))]
        idx = np.sort(np.random.default_rng(args.split_seed + power).choice(cand, size=min(args.power_dev_samples, len(cand)), replace=False)); power_dev[power] = (path, idx)
    profile = profile_multi_axis_imbalance(labels6, np.arange(len(fit)), power6, args.weak_threshold_db, CLASS_NAMES)
    role = {"schema":"route-a-p3-dual-axis-role-manifest-v1", "selected_classes":list(CLASS_NAMES), "excluded_classes":["SN","VFT","RangeFakeTarget"], "filter":"at_least_one_selected_and_zero_excluded", "fit":{"path":str(train_path),"power_db":30,"count":int(len(fit)),"sample_indices_sha256":sample_id_sha256(fit),"receiver_noise_seed":sorted(set(receiver_seeds(train_path, fit)))}, "validation":{"path":str(train_path),"count":int(len(val)),"sample_indices_sha256":sample_id_sha256(val),"receiver_noise_seed":sorted(set(receiver_seeds(train_path, val)))}, "power_development":{"source_role":"train_only_unseen_identity","count_by_power":{str(p):int(len(i)) for p,(path,i) in power_dev.items()}}, "test":{"used":False,"locked_test_used":False}}
    write_json(args.output_root / "role_manifest.json", role); write_json(args.output_root / "multi_axis_profile.json", profile)
    rows = []
    for seed in args.seeds:
        manifest = build_mixed_longtail(labels6, np.arange(len(fit)), power6, args.label_if, args.combo_if, args.weak_threshold_db, int(seed), source_sha)
        selected_local = manifest.selected_indices.astype(np.int64); selected_orig = fit[selected_local]; selected_weak = compute_weak_component_mask(labels6, power6, args.weak_threshold_db)[selected_local]
        order = np.argsort(selected_orig); selected_orig = selected_orig[order]; selected_weak = selected_weak[order]
        meta = manifest.to_dict(); meta.update({"source_fit_count":int(len(fit)),"selected_original_indices_sha256":sample_id_sha256(selected_orig),"selected_count":int(len(selected_orig))})
        write_json(args.output_root / f"longtail_manifest_seed_{seed}.json", meta)
        for name in VARIANTS:
            out = args.output_root / name; out.mkdir(parents=True, exist_ok=True)
            rows.append(run_one(args, name, int(seed), train_path, selected_orig, selected_weak, val, power_dev, out, meta))
    summary = {"schema":"route-a-p3-dual-axis-summary-v1","status":"completed","locked_test_used":False,"protocol":{"fit_power_db":30,"label_imbalance_factor":args.label_if,"combo_imbalance_factor":args.combo_if,"weak_threshold_db":args.weak_threshold_db,"epochs":args.epochs,"seeds":list(args.seeds)},"variants":{}}
    for name in VARIANTS:
        rr = [r for r in rows if r["variant"] == name]
        summary["variants"][name] = {"parameters":rr[0]["total_parameters"],"val_macro_aupr_mean":float(np.mean([r["validation"]["macro_aupr"] for r in rr])),"val_macro_aupr_std":float(np.std([r["validation"]["macro_aupr"] for r in rr],ddof=1)),"val_macro_f1_mean":float(np.mean([r["validation"]["macro_f1"] for r in rr])),"val_combo_mean":float(np.mean([r["validation"]["combination_exact_accuracy"] for r in rr])),"power_dev_macro_aupr_mean_by_power":{str(p):float(np.mean([r["power_dev"][str(p)]["macro_aupr"] for r in rr])) for p in POWERS_DB},"power_dev_combo_mean_by_power":{str(p):float(np.mean([r["power_dev"][str(p)]["combination_exact_accuracy"] for r in rr])) for p in POWERS_DB}}
    write_json(args.output_root / "p3_dual_axis_summary.json", summary); write_json(args.output_root / "terminal_receipt.json", {"status":"completed","summary":str((args.output_root/"p3_dual_axis_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False,"script_sha256":sha256(Path(__file__))})
    print(json.dumps({"status":"completed","summary":str(args.output_root/"p3_dual_axis_summary.json")}, ensure_ascii=False), flush=True)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path(r'E:\论文相关\干扰数据\复合干扰数据0716\九类')); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p3_dual_axis_longtail_20260828'); p.add_argument('--f2-checkpoint',type=Path,default=PROJECT_ROOT/'outputs/route_a_c1_baseline_full_20260826/best_model.pth'); p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS)); p.add_argument('--split-seed',type=int,default=260828); p.add_argument('--fit-samples',type=int,default=4096); p.add_argument('--val-samples',type=int,default=2048); p.add_argument('--power-dev-samples',type=int,default=256); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--eval-batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=2); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--label-if',type=float,default=5.0); p.add_argument('--combo-if',type=float,default=5.0); p.add_argument('--weak-threshold-db',type=float,default=10.0); p.add_argument('--device',default='cuda:0'); return p.parse_args()

if __name__=='__main__': run(parse_args())
