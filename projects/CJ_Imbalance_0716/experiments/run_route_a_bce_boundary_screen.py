"""Run the common BCE anchor on audited development-v2 long-tail manifests."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import find_file, POWERS_DB, receiver_seeds, profile_cost, sha256, write_json


def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def discover_data_root() -> Path:
    parts = [
        "".join(chr(c) for c in [0x8bba, 0x6587, 0x76f8, 0x5173]),
        "".join(chr(c) for c in [0x5e72, 0x6270, 0x6570, 0x636e]),
        "".join(chr(c) for c in [0x590d, 0x5408, 0x5e72, 0x6270, 0x6570, 0x636e]) + "0716",
        "".join(chr(c) for c in [0x4e5d, 0x7c7b]),
    ]
    return Path("E:/").joinpath(*parts)


def load_shared_f2(model: torch.nn.Module, checkpoint: Path) -> dict:
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
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "best_epoch": payload.get("best_epoch")}


def predict(model, path: Path, indices: np.ndarray, device: torch.device, batch_size: int, logit_bias: torch.Tensor | None = None):
    ds = NineClassDataset(path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ys, ls = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in dl:
                logits = model(batch["signal"].to(device, non_blocking=True))
                if logit_bias is not None:
                    logits = logits + logit_bias.view(1, -1)
                ls.append(logits.cpu().numpy())
                ys.append(batch["label"].numpy())
    finally:
        ds.close()
    return np.concatenate(ys).astype(np.uint8), np.concatenate(ls).astype(np.float32)


def compute_loss(variant: str, logits: torch.Tensor, targets: torch.Tensor, pos_counts: np.ndarray, combo_vectors: torch.Tensor | None = None) -> torch.Tensor:
    if variant == "BCE":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if variant == "CB_BCE":
        beta = 0.99
        n = torch.as_tensor(pos_counts, dtype=logits.dtype, device=logits.device).clamp_min(1.0)
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=logits.dtype, device=logits.device), n))
        w = w / w.mean()
        # Positive terms receive effective-number correction; negative terms
        # remain at unit weight to avoid changing the global prior twice.
        per = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weight = 1.0 + targets * (w.view(1, -1) - 1.0)
        return (per * weight).mean()
    if variant == "ASL":
        # Multi-label asymmetric loss with fixed, pre-registered settings.
        gamma_pos, gamma_neg, clip = 0.0, 4.0, 0.05
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if clip > 0:
            xs_neg = (xs_neg + clip).clamp(max=1.0)
        loss_pos = targets * torch.log(xs_pos.clamp_min(1e-8))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp_min(1e-8))
        if gamma_neg > 0:
            loss_neg = loss_neg * torch.pow(xs_pos, gamma_neg)
        return -(loss_pos + loss_neg).mean()
    if variant == "PositiveFocal":
        gamma = 2.0
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        return (-(1.0 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-8))).mean()
    if variant == "PFmix25":
        # Bounded interpolation: retain 75% BCE geometry and add only 25% of
        # the global Positive-Focal objective.
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        gamma = 2.0
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        focal = (-(1.0 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-8))).mean()
        return 0.75 * bce + 0.25 * focal
    if variant == "PF_HN":
        # Bounded focal interpolation plus a small top-k hinge on head-label
        # negatives.  Only the hardest 10% negative logits are constrained,
        # avoiding a global negative-gradient amplification.
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        prob = torch.sigmoid(logits); pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        focal = (-(1.0 - pt).pow(2.0) * torch.log(pt.clamp_min(1e-8))).mean()
        n = np.asarray(pos_counts, dtype=np.float64); head = np.argsort(-n)[:4]
        head_idx = torch.as_tensor(head, dtype=torch.long, device=logits.device)
        neg_logits = logits[:, head_idx][targets[:, head_idx] < 0.5]
        if neg_logits.numel():
            k = max(1, int(np.ceil(0.10 * neg_logits.numel())))
            hardest = torch.topk(neg_logits, k=k).values
            hinge = torch.relu(hardest + 0.5).mean()
        else:
            hinge = logits.sum() * 0.0
        return 0.75 * bce + 0.25 * focal + 0.05 * hinge
    if variant == "ComboMargin":
        # Combination-level margin against the hardest observed alternative.
        # The candidate set is restricted to training combinations, so no
        # validation/locked labels are used to construct the objective.
        if combo_vectors is None or combo_vectors.numel() == 0:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        signs = 2.0 * combo_vectors.to(logits.device) - 1.0
        all_scores = torch.matmul(logits, signs.t())
        true_score = torch.sum(logits * (2.0 * targets - 1.0), dim=1)
        target_keys = (targets > 0.5).to(torch.int64)
        combo_keys = (combo_vectors > 0.5).to(torch.int64)
        equal = torch.all(target_keys[:, None, :] == combo_keys[None, :, :], dim=2)
        neg_scores = all_scores.masked_fill(equal, -torch.inf)
        hard_negative = neg_scores.max(dim=1).values
        margin_loss = torch.relu(0.5 - (true_score - hard_negative)).mean()
        return bce + 0.10 * margin_loss
    if variant == "TailPositiveFocal":
        # Apply focal modulation only to the two least frequent labels.  All
        # head-label positive and negative terms stay BCE, limiting the false
        # positive tradeoff observed with global Positive-Focal.
        gamma = 2.0
        n = np.asarray(pos_counts, dtype=np.float64)
        tail = np.argsort(n)[:2]
        prob = torch.sigmoid(logits)
        per = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        factor = (1.0 - pt).pow(gamma)
        weight = torch.ones_like(per)
        weight[:, torch.as_tensor(tail, dtype=torch.long, device=logits.device)] = torch.where(
            targets[:, torch.as_tensor(tail, dtype=torch.long, device=logits.device)] > 0.5,
            factor[:, torch.as_tensor(tail, dtype=torch.long, device=logits.device)],
            torch.ones_like(factor[:, torch.as_tensor(tail, dtype=torch.long, device=logits.device)]),
        )
        return (per * weight).mean()
    raise ValueError(f"unknown variant: {variant}")


def run_one(args, condition: str, variant: str, seed: int, manifest: dict, source: Path, val: np.ndarray, power_dev: dict[int, tuple[Path, np.ndarray]], out: Path) -> dict:
    set_seed(seed)
    device = torch.device(args.device)
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    init = load_shared_f2(model, args.f2_checkpoint)
    # h5py requires monotonically increasing fancy indices; manifest order is
    # deliberately shuffled for reproducibility, so sort only for I/O.
    train_indices = np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64))
    ds = NineClassDataset(source, train_indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    pos_counts = np.asarray(manifest.get("achieved_label_counts", manifest.get("label_counts")), dtype=np.float64)
    combo_vectors = None
    if variant == "ComboMargin":
        keys = sorted((manifest.get("combo_counts") or {}).keys())
        if keys:
            combo_vectors = torch.as_tensor([[float(int(bit)) for bit in key] for key in keys], dtype=torch.float32, device=device)
    logit_bias = None
    if variant == "LogitAdj":
        prior = np.clip(pos_counts / float(len(train_indices)), 1e-4, 1.0 - 1e-4)
        # Prior-to-neutral correction: rare positive labels receive a larger
        # logit during both training and evaluation (tau=1, fixed).
        logit_bias = torch.as_tensor(-np.log(prior / (1.0 - prior)), dtype=torch.float32, device=device)
    sampler = None
    if variant == "ComboSampler":
        # Uniformize observed combination identities while keeping the same
        # number of optimizer samples (sampling with replacement).
        combo_keys = ["".join(str(int(v)) for v in row.tolist()) for row in ds.labels]
        combo_counts = {}
        for key in combo_keys: combo_counts[key] = combo_counts.get(key, 0) + 1
        weights = torch.as_tensor([1.0 / combo_counts[k] for k in combo_keys], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best, best_epoch = [], -1.0, 0
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []; grad_norms = []
            for batch in dl:
                x = batch["signal"].to(device, non_blocking=True); y = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                active_variant = "BCE" if variant in ("ComboSampler", "LogitAdj") else variant
                if variant == "DRW":
                    # Deferred reweighting: keep the representation and
                    # optimizer fixed, enable effective-number correction only
                    # in the second half of the budget.
                    active_variant = "BCE" if epoch <= max(args.epochs // 2, 1) else "CB_BCE"
                logits = model(x)
                if logit_bias is not None:
                    logits = logits + logit_bias.view(1, -1)
                loss = compute_loss(active_variant, logits, y, pos_counts, combo_vectors)
                loss.backward(); grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))); optimizer.step(); losses.append(float(loss.detach().cpu()))
            yt, yl = predict(model, source, val, device, args.eval_batch_size, logit_bias)
            vm = metrics(yt, yl)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "grad_norm_mean": float(np.mean(grad_norms)), "validation": vm})
            if vm["macro_aupr"] > best:
                best, best_epoch = vm["macro_aupr"], epoch
                torch.save(model.state_dict(), out / f"{condition}__{variant}__seed_{seed}.pth")
    finally:
        ds.close()
    model.load_state_dict(torch.load(str(out / f"{condition}__{variant}__seed_{seed}.pth"), map_location=device, weights_only=True))
    result = {
        "schema": "route-a-bce-boundary-result-v2", "condition": condition, "variant": variant, "seed": int(seed),
        "manifest": manifest, "initialization": init, "best_epoch": best_epoch,
        "validation": history[best_epoch - 1]["validation"], "history": history,
        "train_label_counts": manifest.get("achieved_label_counts", manifest.get("label_counts")),
        "effective_if": manifest.get("effective_if"), "requested_if": manifest.get("requested_if"),
        "requested_cf": manifest.get("requested_cf"), "achieved_cf": manifest.get("achieved_cf"),
        "logit_adjustment_tau": 1.0 if variant == "LogitAdj" else None,
        "logit_adjustment_bias": (logit_bias.detach().cpu().numpy().tolist() if logit_bias is not None else None),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "cost_profile": profile_cost(model, device), "power_dev": {},
        "locked_test_used": False, "test_files_opened": False,
        "elapsed_seconds": float(time.time() - started),
    }
    for power, (path, idx) in power_dev.items():
        yt, yl = predict(model, path, idx, device, args.eval_batch_size, logit_bias)
        result["power_dev"][str(power)] = metrics(yt, yl)
    result["checkpoint_sha256"] = sha256(out / f"{condition}__{variant}__seed_{seed}.pth")
    write_json(out / f"{condition}__{variant}__seed_{seed}.json", result)
    print(json.dumps({"condition": condition, "variant": variant, "seed": seed, "macro_aupr": result["validation"]["macro_aupr"], "combo": result["validation"]["combination_exact_accuracy"]}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_bce_boundary_v1_20260901")
    p.add_argument("--val-manifest", type=Path, default=None, help="Optional JSON manifest providing strict validation indices")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/route_a_bce_boundary_screen_20260901")
    p.add_argument("--f2-checkpoint", type=Path, default=PROJECT_ROOT / "outputs/route_a_c1_baseline_full_20260826/best_model.pth")
    p.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041])
    p.add_argument("--split-seed", type=int, default=260828)
    p.add_argument("--conditions", nargs="+", default=["label_IF1", "label_IF2", "label_IF3", "combo_CF1", "combo_CF4", "combo_CF8"])
    p.add_argument("--variants", nargs="+", default=["BCE"])
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("BCE boundary screen requires CUDA")
    args.output_root.mkdir(parents=True, exist_ok=True)
    data_root = discover_data_root(); source = next(x for x in data_root.rglob("closedset_9class_train.mat") if "level_05" in str(x))
    pure = pure_six_indices(source); perm = np.random.default_rng(args.split_seed).permutation(pure); fit = np.sort(perm[:4096]); val = np.sort(perm[4096:6144])
    if args.val_manifest is not None:
        val_payload = json.loads(args.val_manifest.read_text(encoding="utf-8"))
        val = np.sort(np.asarray(val_payload.get("validation_indices", val_payload.get("selected_indices")), dtype=np.int64))
    used = set(receiver_seeds(source, np.sort(np.concatenate([fit, val])))); power_dev = {}
    for power in (10, 30, 50):
        path = find_file(data_root, power, "train"); pp = pure_six_indices(path); seeds = np.asarray(receiver_seeds(path, pp), dtype=np.int64)
        cand = pp[~np.isin(seeds, np.fromiter(used, dtype=np.int64))]
        idx = np.sort(np.random.default_rng(260828 + power).choice(cand, size=min(256, len(cand)), replace=False)); power_dev[power] = (path, idx)
    rows = []
    for condition in args.conditions:
        manifests = sorted(args.manifest_root.glob(f"{condition}__seed_*.json"))
        if not manifests: raise FileNotFoundError(f"No manifests for {condition}")
        for seed in args.seeds:
            mp = args.manifest_root / f"{condition}__seed_{seed}.json"
            if not mp.exists(): raise FileNotFoundError(mp)
            manifest = json.loads(mp.read_text(encoding="utf-8")); out = args.output_root / condition; out.mkdir(parents=True, exist_ok=True)
            for variant in args.variants:
                variant_out = out / variant; variant_out.mkdir(parents=True, exist_ok=True)
                rows.append(run_one(args, condition, variant, seed, manifest, source, val, power_dev, variant_out))
    summary = {"schema":"route-a-bce-boundary-screen-summary-v2","status":"completed","locked_test_used":False,"test_files_opened":False,"conditions":{},"variants":list(args.variants),"seeds":list(args.seeds),"epochs":args.epochs}
    for condition in args.conditions:
      for variant in args.variants:
        rr = [r for r in rows if r["condition"] == condition and r["variant"] == variant]
        summary["conditions"][f"{condition}__{variant}"] = {
            "n": len(rr), "effective_if": [r.get("effective_if") for r in rr], "effective_cf": [r.get("achieved_cf") for r in rr],
            "val_macro_aupr_mean": float(np.mean([r["validation"]["macro_aupr"] for r in rr])), "val_macro_aupr_std": float(np.std([r["validation"]["macro_aupr"] for r in rr], ddof=1)),
            "val_macro_f1_mean": float(np.mean([r["validation"]["macro_f1"] for r in rr])), "val_combo_mean": float(np.mean([r["validation"]["combination_exact_accuracy"] for r in rr])),
            "val_hamming_mean": float(np.mean([r["validation"]["hamming_loss"] for r in rr])),
            "power_dev_combo_mean_by_power": {str(p): float(np.mean([r["power_dev"][str(p)]["combination_exact_accuracy"] for r in rr])) for p in (10,30,50)},
        }
    write_json(args.output_root / "summary.json", summary)
    write_json(args.output_root / "terminal_receipt.json", {"status":"completed","summary":str((args.output_root/"summary.json").resolve()),"locked_test_used":False,"test_files_opened":False})
    print(json.dumps({"status":"completed","summary":str((args.output_root/"summary.json").resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
