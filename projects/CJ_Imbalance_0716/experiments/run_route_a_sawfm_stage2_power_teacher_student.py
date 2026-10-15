"""SawFM stage2 teacher-student power stabilization with identity-safe negatives."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import (  # noqa: E402
    build_role_bound_power_pair_index,
    discover_power_sweep_train_files,
    load_level_identity,
)
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from cjlt.models.evidence_readout import SawStage2LowRankEvidence  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa: E402
from experiments.run_route_a_dfms_iq_label_local_saw import (  # noqa: E402
    F2_MODEL_NAME,
    _load_h2_head,
)
from experiments.run_shared import _data_path, build_run_plan, load_config  # noqa: E402


SAW_INDEX = 1
SWEPT_INDICES = (3, 4, 5)
OTHER_INDICES = (0, 2, 3, 4, 5)
VARIANTS = (
    "S0_StudentRandomNeg",
    "S1_TeacherRandomNeg",
    "S2_TeacherHardNeg",
)


def _sha256_values(*arrays):
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def _combo_key(label):
    return tuple(int(label[index]) for index in OTHER_INDICES)


def _stratified_triplets(
    base_indices,
    labels,
    identities,
    pairs_per_combo,
    seed,
):
    base_indices = np.asarray(base_indices, dtype=np.int64)
    role_labels = labels[base_indices]
    positive_mask = (role_labels[:, SAW_INDEX] == 1) & (
        role_labels[:, list(SWEPT_INDICES)].sum(axis=1) > 0
    )
    negative_mask = role_labels[:, SAW_INDEX] == 0
    positive_by_combo = {}
    negative_by_combo = {}
    for position, base_index in enumerate(base_indices):
        key = _combo_key(role_labels[position])
        if positive_mask[position]:
            positive_by_combo.setdefault(key, []).append(int(base_index))
        elif negative_mask[position]:
            negative_by_combo.setdefault(key, []).append(int(base_index))
    shared = sorted(set(positive_by_combo).intersection(negative_by_combo))
    if len(shared) != 28:
        raise ValueError("Expected 28 SawFM visibility combinations, got {}".format(len(shared)))
    rng = np.random.default_rng(int(seed))
    selected_positive = []
    selected_hard_negative = []
    combo_counts = {}
    for key in shared:
        positives = np.asarray(positive_by_combo[key], dtype=np.int64)
        negatives = np.asarray(negative_by_combo[key], dtype=np.int64)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        count = min(int(pairs_per_combo), len(positives), len(negatives))
        if count != int(pairs_per_combo):
            raise ValueError("Combination {} has insufficient identities".format(key))
        selected_positive.extend(positives[:count].tolist())
        selected_hard_negative.extend(negatives[:count].tolist())
        combo_counts[str(key)] = int(count)
    all_negative = base_indices[negative_mask].copy()
    rng.shuffle(all_negative)
    count = len(selected_positive)
    if len(all_negative) < count:
        raise ValueError("Insufficient random SawFM negatives")
    selected_random_negative = all_negative[:count]
    positive = np.asarray(selected_positive, dtype=np.int64)
    hard_negative = np.asarray(selected_hard_negative, dtype=np.int64)
    random_negative = np.asarray(selected_random_negative, dtype=np.int64)
    order = rng.permutation(count)
    positive = positive[order]
    hard_negative = hard_negative[order]
    random_negative = random_negative[order]
    return {
        "positive_base_indices": positive,
        "hard_negative_base_indices": hard_negative,
        "random_negative_base_indices": random_negative,
        "positive_identities": identities[positive].astype(np.int64),
        "hard_negative_identities": identities[hard_negative].astype(np.int64),
        "random_negative_identities": identities[random_negative].astype(np.int64),
        "combo_counts": combo_counts,
        "selection_sha256": _sha256_values(positive, hard_negative, random_negative),
    }


def _development_pairs(base_indices, labels, identities, seed):
    base_indices = np.asarray(base_indices, dtype=np.int64)
    role_labels = labels[base_indices]
    positive_by_combo = {}
    negative_by_combo = {}
    for position, base_index in enumerate(base_indices):
        label = role_labels[position]
        key = _combo_key(label)
        if label[SAW_INDEX] == 1 and label[list(SWEPT_INDICES)].sum() > 0:
            positive_by_combo.setdefault(key, []).append(int(base_index))
        elif label[SAW_INDEX] == 0:
            negative_by_combo.setdefault(key, []).append(int(base_index))
    rng = np.random.default_rng(int(seed))
    positive = []
    negative = []
    combo_counts = {}
    for key in sorted(set(positive_by_combo).intersection(negative_by_combo)):
        positives = np.asarray(positive_by_combo[key], dtype=np.int64)
        negatives = np.asarray(negative_by_combo[key], dtype=np.int64)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        count = min(len(positives), len(negatives))
        if count:
            positive.extend(positives[:count].tolist())
            negative.extend(negatives[:count].tolist())
            combo_counts[str(key)] = int(count)
    if not positive:
        raise ValueError("Development role contains no matched SawFM visibility pairs")
    positive = np.asarray(positive, dtype=np.int64)
    negative = np.asarray(negative, dtype=np.int64)
    return {
        "positive_base_indices": positive,
        "hard_negative_base_indices": negative,
        "positive_identities": identities[positive].astype(np.int64),
        "hard_negative_identities": identities[negative].astype(np.int64),
        "combo_counts": combo_counts,
        "selection_sha256": _sha256_values(positive, negative),
    }


def _level_lookup(power_files, levels):
    result = {}
    for level in levels:
        identities, labels = load_level_identity(power_files[int(level)])
        result[int(level)] = {
            "labels": labels,
            "identity_to_index": {
                int(identity): int(index) for index, identity in enumerate(identities)
            },
        }
    return result


def _indices_for_identities(level_info, identities):
    lookup = level_info["identity_to_index"]
    indices = []
    for identity in identities:
        value = int(identity)
        if value not in lookup:
            raise ValueError("Identity {} missing from power level".format(value))
        indices.append(lookup[value])
    return np.asarray(indices, dtype=np.int64)


def _extract_level(
    backbone,
    path,
    indices,
    normalization,
    device,
    batch_size,
    base_head=None,
    main_mean=None,
    main_std=None,
):
    dataset = V73SignalDataset(
        path=path,
        indices=np.asarray(indices, dtype=np.int64),
        class_mode="six",
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    features = []
    logits = []
    mean_tensor = (
        torch.as_tensor(main_mean, dtype=torch.float32, device=device)
        if main_mean is not None
        else None
    )
    std_tensor = (
        torch.as_tensor(main_std, dtype=torch.float32, device=device)
        if main_std is not None
        else None
    )
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    maps = backbone.forward_structure_maps(signal)
                    stage2 = torch.cat(
                        [maps["stage2_real"], maps["stage2_imag"]], dim=1
                    )
                    stage2 = F.avg_pool1d(stage2, kernel_size=4, stride=4)
                features.append(stage2.detach().cpu().to(torch.float16).numpy())
                if base_head is not None:
                    global_features = maps["global"].float()
                    normalized = (global_features - mean_tensor) / std_tensor
                    base_logits = base_head(normalized, normalized)
                    logits.append(base_logits.detach().cpu().float().numpy())
    finally:
        dataset.close()
    result = {"features": np.concatenate(features, axis=0)}
    if logits:
        result["base_logits"] = np.concatenate(logits, axis=0)
    return result


def _binary_metrics(positive_scores, negative_scores):
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative_scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    targets = np.concatenate(
        [np.ones(len(positive_scores), dtype=np.uint8), np.zeros(len(negative_scores), dtype=np.uint8)]
    )
    scores = np.concatenate([positive_scores, negative_scores])
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "aupr": float(average_precision_score(targets, scores)),
        "positive_count": int(len(positive_scores)),
        "negative_count": int(len(negative_scores)),
    }


def _model_scores(model, features, device, batch_size=128, return_maps=False):
    model.eval()
    scores = []
    maps = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            values = torch.from_numpy(features[start : start + int(batch_size)]).to(
                device=device, dtype=torch.float32
            )
            if return_maps:
                logits, evidence_map, _ = model(values, return_map=True)
                maps.append(evidence_map.cpu().numpy())
            else:
                logits = model(values)
            scores.append(logits.cpu().numpy())
    result = np.concatenate(scores, axis=0)
    if return_maps:
        return result, np.concatenate(maps, axis=0)
    return result


def _batch_values(values, indices, device):
    return torch.from_numpy(values[indices]).to(device=device, dtype=torch.float32)


def _train_variant(
    variant,
    arrays,
    fit_indices,
    validation_indices,
    seed,
    device,
    args,
    train_seed_override=None,
):
    train_seed = (
        171000 + int(seed) + VARIANTS.index(variant) * 1000
        if train_seed_override is None
        else int(train_seed_override)
    )
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    model = SawStage2LowRankEvidence(
        channels=64,
        rank=args.adapter_rank,
        hidden_channels=args.hidden_channels,
        topk=args.topk,
        adapter_limit=args.adapter_limit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(train_seed)
    best_state = None
    best_epoch = 0
    best_auroc = -float("inf")
    best_aupr = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = rng.permutation(fit_indices)
        totals = {"total": 0.0, "classification": 0.0, "teacher": 0.0, "score": 0.0, "map": 0.0, "rank": 0.0}
        sample_count = 0
        for start in range(0, len(order), int(args.batch_size)):
            indices = order[start : start + int(args.batch_size)]
            positive40 = _batch_values(arrays["positive40"], indices, device)
            positive50 = _batch_values(arrays["positive50"], indices, device)
            negative_prefix = "hard" if variant == "S2_TeacherHardNeg" else "random"
            negative40 = _batch_values(arrays[negative_prefix + "40"], indices, device)
            negative50 = _batch_values(arrays[negative_prefix + "50"], indices, device)
            optimizer.zero_grad(set_to_none=True)
            pos40_logits, pos40_map, _ = model(positive40, return_map=True)
            pos50_logits, pos50_map, _ = model(positive50, return_map=True)
            neg40_logits = model(negative40)
            neg50_logits = model(negative50)
            positive_target40 = torch.ones_like(pos40_logits)
            positive_target50 = torch.ones_like(pos50_logits)
            negative_target40 = torch.zeros_like(neg40_logits)
            negative_target50 = torch.zeros_like(neg50_logits)
            classification_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(pos40_logits, positive_target40)
                + F.binary_cross_entropy_with_logits(pos50_logits, positive_target50)
                + F.binary_cross_entropy_with_logits(neg40_logits, negative_target40)
                + F.binary_cross_entropy_with_logits(neg50_logits, negative_target50)
            )
            teacher_loss = pos40_logits.new_tensor(0.0)
            score_loss = pos40_logits.new_tensor(0.0)
            map_loss = pos40_logits.new_tensor(0.0)
            rank_loss = pos40_logits.new_tensor(0.0)
            if variant != "S0_StudentRandomNeg":
                teacher = _batch_values(arrays["teacher10"], indices, device)
                teacher_logits, teacher_map, _ = model(teacher, return_map=True)
                teacher_loss = 0.5 * F.binary_cross_entropy_with_logits(
                    teacher_logits, torch.ones_like(teacher_logits)
                )
                score_loss = 0.25 * (
                    F.smooth_l1_loss(pos40_logits, teacher_logits.detach())
                    + F.smooth_l1_loss(pos50_logits, teacher_logits.detach())
                )
                map_loss = 0.1 * (
                    (1.0 - F.cosine_similarity(pos40_map, teacher_map.detach(), dim=-1)).mean()
                    + (1.0 - F.cosine_similarity(pos50_map, teacher_map.detach(), dim=-1)).mean()
                )
            if variant == "S2_TeacherHardNeg":
                rank_loss = 0.25 * (
                    F.softplus(args.ranking_margin - pos40_logits + neg40_logits).mean()
                    + F.softplus(args.ranking_margin - pos50_logits + neg50_logits).mean()
                )
            loss = classification_loss + teacher_loss + score_loss + map_loss + rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            count = len(indices)
            sample_count += count
            for name, value in (
                ("total", loss),
                ("classification", classification_loss),
                ("teacher", teacher_loss),
                ("score", score_loss),
                ("map", map_loss),
                ("rank", rank_loss),
            ):
                totals[name] += float(value.detach().cpu()) * count
        validation_positive = _model_scores(
            model, arrays["positive45"][validation_indices], device
        )
        validation_negative = _model_scores(
            model, arrays["hard45"][validation_indices], device
        )
        validation_metrics = _binary_metrics(validation_positive, validation_negative)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": totals["total"] / max(sample_count, 1),
                "classification_loss": totals["classification"] / max(sample_count, 1),
                "teacher_loss": totals["teacher"] / max(sample_count, 1),
                "score_distill_loss": totals["score"] / max(sample_count, 1),
                "map_consistency_loss": totals["map"] / max(sample_count, 1),
                "ranking_loss": totals["rank"] / max(sample_count, 1),
                "validation_auroc": validation_metrics["auroc"],
                "validation_aupr": validation_metrics["aupr"],
            }
        )
        improved = validation_metrics["auroc"] > best_auroc + 1e-8
        tied_better = np.isclose(validation_metrics["auroc"], best_auroc) and (
            validation_metrics["aupr"] > best_aupr + 1e-8
        )
        if improved or tied_better:
            best_auroc = validation_metrics["auroc"]
            best_aupr = validation_metrics["aupr"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("SawFM teacher-student training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history, train_seed


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parameter_counts(model):
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


def _extract_seed_arrays(
    args,
    config,
    power_files,
    level_info,
    backbone,
    base_head,
    h2_checkpoint,
    selection,
    development,
    device,
):
    def extract(level, identities, with_logits=False):
        indices = _indices_for_identities(level_info[int(level)], identities)
        return _extract_level(
            backbone,
            power_files[int(level)],
            indices,
            config["normalization"],
            device,
            args.feature_batch_size,
            base_head=base_head if with_logits else None,
            main_mean=h2_checkpoint["main_mean"] if with_logits else None,
            main_std=h2_checkpoint["main_std"] if with_logits else None,
        )

    positive = selection["positive_identities"]
    hard = selection["hard_negative_identities"]
    random_negative = selection["random_negative_identities"]
    arrays = {
        "teacher10": extract(10, positive)["features"],
        "positive40": extract(40, positive)["features"],
        "positive50": extract(50, positive)["features"],
        "hard40": extract(40, hard)["features"],
        "hard50": extract(50, hard)["features"],
        "random40": extract(40, random_negative)["features"],
        "random50": extract(50, random_negative)["features"],
        "positive45": extract(45, positive)["features"],
        "hard45": extract(45, hard)["features"],
    }
    development_positive = extract(45, development["positive_identities"], with_logits=True)
    development_negative = extract(45, development["hard_negative_identities"], with_logits=True)
    arrays.update(
        {
            "development_positive45": development_positive["features"],
            "development_negative45": development_negative["features"],
            "development_positive_base_logits": development_positive["base_logits"],
            "development_negative_base_logits": development_negative["base_logits"],
        }
    )
    return arrays


def _run_seed(args, config, labels, base_identities, seed, device):
    plan = build_run_plan(
        config,
        loss_name="db",
        seed=seed,
        max_train_samples=0,
        max_eval_samples=args.max_eval_samples,
    )
    source_dir = args.screen_root / "runs" / F2_MODEL_NAME / "seed_{}".format(seed)
    source_receipt = _read(source_dir / "run_receipt.json")
    if source_receipt.get("locked_test_used"):
        raise ValueError("F2 source used locked test")
    if plan["split_sha256"] != source_receipt["split_sha256"]:
        raise ValueError("Split hash drifted for seed {}".format(seed))
    selection = _stratified_triplets(
        plan["train_indices"],
        labels,
        base_identities,
        args.pairs_per_combo,
        161000 + int(seed),
    )
    development = _development_pairs(
        plan["early_stop_indices"], labels, base_identities, 162000 + int(seed)
    )
    train_identity_set = set(
        selection["positive_identities"].tolist()
        + selection["hard_negative_identities"].tolist()
        + selection["random_negative_identities"].tolist()
    )
    development_identity_set = set(
        development["positive_identities"].tolist()
        + development["hard_negative_identities"].tolist()
    )
    overlap = sorted(train_identity_set.intersection(development_identity_set))
    if overlap:
        raise ValueError("Training and development identities overlap")
    power_root = Path(str(config["data_root"])) / str(config["power_sweep_dir"])
    power_files = discover_power_sweep_train_files(power_root)
    levels = (10, 40, 45, 50)
    level_info = _level_lookup(power_files, levels)
    selected_base = np.unique(
        np.concatenate(
            [
                selection["positive_base_indices"],
                selection["hard_negative_base_indices"],
                selection["random_negative_base_indices"],
            ]
        )
    )
    train_pair_index = build_role_bound_power_pair_index(
        _data_path(config),
        power_root,
        selected_base,
        levels=levels,
        reference_level=10,
    )
    development_pair_index = build_role_bound_power_pair_index(
        _data_path(config),
        power_root,
        np.unique(
            np.concatenate(
                [
                    development["positive_base_indices"],
                    development["hard_negative_base_indices"],
                ]
            )
        ),
        levels=(10, 45),
        reference_level=10,
    )
    backbone, f2_checkpoint_path = _load_backbone(source_dir, F2_MODEL_NAME, device)
    base_head, h2_checkpoint, h2_checkpoint_path, _ = _load_h2_head(
        seed, args.frozen_head_root, device
    )
    print("seed={} extracting identity-safe stage2 views".format(seed), flush=True)
    arrays = _extract_seed_arrays(
        args,
        config,
        power_files,
        level_info,
        backbone,
        base_head,
        h2_checkpoint,
        selection,
        development,
        device,
    )
    rng = np.random.default_rng(163000 + int(seed))
    order = rng.permutation(len(selection["positive_identities"]))
    split = int(round(len(order) * 0.8))
    fit_indices = order[:split]
    validation_indices = order[split:]
    seed_root = args.output_root / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    sampling_payload = {
        "seed": int(seed),
        "pairs_per_combo": int(args.pairs_per_combo),
        "fit_pair_count": int(len(fit_indices)),
        "validation_pair_count": int(len(validation_indices)),
        "development_positive_count": int(len(development["positive_identities"])),
        "development_negative_count": int(len(development["hard_negative_identities"])),
        "train_combo_counts": selection["combo_counts"],
        "development_combo_counts": development["combo_counts"],
        "selection_sha256": selection["selection_sha256"],
        "development_selection_sha256": development["selection_sha256"],
        "train_pair_sha256": train_pair_index.sha256,
        "development_pair_sha256": development_pair_index.sha256,
        "train_development_identity_overlap": int(len(overlap)),
        "locked_test_used": False,
    }
    _write(seed_root / "sampling_manifest.json", sampling_payload)
    rows = []
    reconnect_payloads = {}
    for variant in VARIANTS:
        print("seed={} variant={}".format(seed, variant), flush=True)
        output_dir = seed_root / variant
        model, best_epoch, history, train_seed = _train_variant(
            variant,
            arrays,
            fit_indices,
            validation_indices,
            seed,
            device,
            args,
        )
        validation_positive = _model_scores(
            model, arrays["positive45"][validation_indices], device
        )
        validation_negative = _model_scores(
            model, arrays["hard45"][validation_indices], device
        )
        development_positive = _model_scores(
            model, arrays["development_positive45"], device
        )
        development_negative = _model_scores(
            model, arrays["development_negative45"], device
        )
        validation_metrics = _binary_metrics(validation_positive, validation_negative)
        development_metrics = _binary_metrics(development_positive, development_negative)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_saw_stage2_evidence.pth"
        fit_positive = np.concatenate(
            [
                _model_scores(model, arrays["positive40"][fit_indices], device),
                _model_scores(model, arrays["positive50"][fit_indices], device),
            ]
        )
        negative_prefix = "hard" if variant == "S2_TeacherHardNeg" else "random"
        fit_negative = np.concatenate(
            [
                _model_scores(model, arrays[negative_prefix + "40"][fit_indices], device),
                _model_scores(model, arrays[negative_prefix + "50"][fit_indices], device),
            ]
        )
        fit_scores = np.concatenate([fit_positive, fit_negative])
        score_center = float(fit_scores.mean())
        score_scale = float(max(fit_scores.std(), 1e-5))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant": variant,
                "best_epoch": int(best_epoch),
                "train_seed": int(train_seed),
                "score_center": score_center,
                "score_scale": score_scale,
                "sampling_manifest": sampling_payload,
            },
            checkpoint_path,
        )
        _write_history(output_dir / "history.csv", history)
        with torch.no_grad():
            _, _, diagnostics = model(
                torch.from_numpy(arrays["development_positive45"][: min(64, len(arrays["development_positive45"]))]).to(
                    device=device, dtype=torch.float32
                ),
                return_map=True,
            )
        row = {
            "variant": variant,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "train_seed": int(train_seed),
            "validation_auroc": validation_metrics["auroc"],
            "validation_aupr": validation_metrics["aupr"],
            "development_auroc": development_metrics["auroc"],
            "development_aupr": development_metrics["aupr"],
            "generalization_gap_abs": abs(
                validation_metrics["auroc"] - development_metrics["auroc"]
            ),
            "validation_positive_count": validation_metrics["positive_count"],
            "validation_negative_count": validation_metrics["negative_count"],
            "development_positive_count": development_metrics["positive_count"],
            "development_negative_count": development_metrics["negative_count"],
            "parameter_counts": _parameter_counts(model),
            "adapter_scale": float(diagnostics["adapter_scale"].detach().cpu()),
            "logit_scale": float(diagnostics["logit_scale"].detach().cpu()),
            "score_center": score_center,
            "score_scale": score_scale,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "f2_checkpoint_sha256": _sha256(f2_checkpoint_path),
            "h2_checkpoint_sha256": _sha256(h2_checkpoint_path),
            "split_sha256": plan["split_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "train_pair_sha256": train_pair_index.sha256,
            "development_pair_sha256": development_pair_index.sha256,
            "identity_overlap": int(len(overlap)),
            "locked_test_used": False,
        }
        rows.append(row)
        _write(output_dir / "variant_summary.json", row)
        np.savez_compressed(
            str(output_dir / "predictions.npz"),
            validation_positive_scores=validation_positive,
            validation_negative_scores=validation_negative,
            development_positive_scores=development_positive,
            development_negative_scores=development_negative,
        )
        _write(
            output_dir / "run_receipt.json",
            {
                "status": "completed",
                "variant": variant,
                "seed": int(seed),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "f2_checkpoint_sha256": row["f2_checkpoint_sha256"],
                "h2_checkpoint_sha256": row["h2_checkpoint_sha256"],
                "split_sha256": row["split_sha256"],
                "selection_sha256": row["selection_sha256"],
                "train_pair_sha256": row["train_pair_sha256"],
                "development_pair_sha256": row["development_pair_sha256"],
                "identity_overlap": row["identity_overlap"],
                "locked_test_used": False,
            },
        )
        if variant == "S2_TeacherHardNeg":
            reconnect_path = output_dir / "reconnect_inputs.npz"
            np.savez_compressed(
                str(reconnect_path),
                positive_evidence=development_positive,
                negative_evidence=development_negative,
                positive_base_saw_logits=arrays["development_positive_base_logits"][:, SAW_INDEX],
                negative_base_saw_logits=arrays["development_negative_base_logits"][:, SAW_INDEX],
                score_center=np.asarray([score_center], dtype=np.float32),
                score_scale=np.asarray([score_scale], dtype=np.float32),
            )
            reconnect_payloads[int(seed)] = reconnect_path
    return rows, sampling_payload, reconnect_payloads


def _aggregate(rows):
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: _mean_ci(
                [row[metric] for row in selected],
                181000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(
                (
                    "validation_auroc",
                    "validation_aupr",
                    "development_auroc",
                    "development_aupr",
                    "generalization_gap_abs",
                )
            )
        }
    return aggregate


def _comparison(rows, candidate, baseline, seed_offset):
    paired = []
    for seed in sorted({row["seed"] for row in rows}):
        candidate_row = next(
            row for row in rows if row["seed"] == seed and row["variant"] == candidate
        )
        baseline_row = next(
            row for row in rows if row["seed"] == seed and row["variant"] == baseline
        )
        paired.append(
            {
                "seed": int(seed),
                "development_auroc_delta": candidate_row["development_auroc"]
                - baseline_row["development_auroc"],
                "development_aupr_delta": candidate_row["development_aupr"]
                - baseline_row["development_aupr"],
                "generalization_gap_delta": candidate_row["generalization_gap_abs"]
                - baseline_row["generalization_gap_abs"],
            }
        )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "per_seed": paired,
        "development_auroc_delta": _mean_ci(
            [row["development_auroc_delta"] for row in paired], seed_offset
        ),
        "development_aupr_delta": _mean_ci(
            [row["development_aupr_delta"] for row in paired], seed_offset + 1
        ),
        "generalization_gap_delta": _mean_ci(
            [row["generalization_gap_delta"] for row in paired], seed_offset + 2
        ),
        "development_auroc_positive_seeds": int(
            sum(row["development_auroc_delta"] > 0 for row in paired)
        ),
    }


def _promotion_gate(rows, aggregate, comparison, sampling_payloads):
    s2_rows = [row for row in rows if row["variant"] == "S2_TeacherHardNeg"]
    audit_passed = all(
        payload["train_development_identity_overlap"] == 0
        and not payload["locked_test_used"]
        for payload in sampling_payloads
    )
    checks = {
        "development_auroc_0p60": aggregate["S2_TeacherHardNeg"]["development_auroc"]["mean"] >= 0.60,
        "development_auroc_two_seeds": sum(row["development_auroc"] >= 0.60 for row in s2_rows) >= 2,
        "generalization_gap_0p15": aggregate["S2_TeacherHardNeg"]["generalization_gap_abs"]["mean"] <= 0.15,
        "gain_vs_s0_0p02": comparison["development_auroc_delta"]["mean"] >= 0.02,
        "gain_vs_s0_two_seeds": comparison["development_auroc_positive_seeds"] >= 2,
        "development_aupr_noninferior": comparison["development_aupr_delta"]["mean"] >= 0.0,
        "identity_and_locked_test_audit": audit_passed,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _bounded_reconnect(reconnect_payloads, output_root, max_residual):
    rows = []
    for seed, path in sorted(reconnect_payloads.items()):
        values = np.load(path)
        center = float(values["score_center"][0])
        scale = float(values["score_scale"][0])
        positive_evidence = values["positive_evidence"]
        negative_evidence = values["negative_evidence"]
        positive_base = values["positive_base_saw_logits"]
        negative_base = values["negative_base_saw_logits"]

        def connect(base, evidence):
            z = (evidence - center) / max(scale, 1e-5)
            residual = float(max_residual) * np.tanh(z) * np.exp(-np.abs(base))
            return base + residual, residual

        positive_final, positive_residual = connect(positive_base, positive_evidence)
        negative_final, negative_residual = connect(negative_base, negative_evidence)
        before = _binary_metrics(positive_base, negative_base)
        after = _binary_metrics(positive_final, negative_final)
        row = {
            "seed": int(seed),
            "before": before,
            "after": after,
            "auroc_delta": after["auroc"] - before["auroc"],
            "aupr_delta": after["aupr"] - before["aupr"],
            "max_abs_residual": float(
                max(np.abs(positive_residual).max(), np.abs(negative_residual).max())
            ),
            "mean_abs_residual": float(
                np.mean(
                    np.concatenate(
                        [np.abs(positive_residual), np.abs(negative_residual)]
                    )
                )
            ),
            "locked_test_used": False,
        }
        rows.append(row)
        _write(output_root / "seed_{}".format(seed) / "S2_TeacherHardNeg" / "bounded_reconnect_metrics.json", row)
    return {
        "status": "completed_after_oracle_gate",
        "max_residual": float(max_residual),
        "per_seed": rows,
        "auroc_delta": _mean_ci([row["auroc_delta"] for row in rows], 191000),
        "aupr_delta": _mean_ci([row["aupr_delta"] for row in rows], 191001),
        "locked_test_used": False,
    }


def run(args):
    config = load_config(args.config)
    base_path = _data_path(config)
    labels = select_labels(read_labels(base_path), "six")
    base_identities, identity_labels = load_level_identity(base_path)
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base identity labels are misaligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    sampling_payloads = []
    reconnect_payloads = {}
    for seed in args.seeds:
        seed_rows, sampling, seed_reconnect = _run_seed(
            args, config, labels, base_identities, int(seed), device
        )
        rows.extend(seed_rows)
        sampling_payloads.append(sampling)
        reconnect_payloads.update(seed_reconnect)
    aggregate = _aggregate(rows)
    comparisons = {
        "s1_vs_s0": _comparison(
            rows, "S1_TeacherRandomNeg", "S0_StudentRandomNeg", 186000
        ),
        "s2_vs_s0": _comparison(
            rows, "S2_TeacherHardNeg", "S0_StudentRandomNeg", 186100
        ),
        "s2_vs_s1": _comparison(
            rows, "S2_TeacherHardNeg", "S1_TeacherRandomNeg", 186200
        ),
    }
    gate = _promotion_gate(
        rows, aggregate, comparisons["s2_vs_s0"], sampling_payloads
    )
    if gate["passed"]:
        reconnect = _bounded_reconnect(
            reconnect_payloads, args.output_root, args.max_residual
        )
        reconnect_decision = "completed_after_oracle_gate"
    else:
        reconnect = {
            "status": "skipped",
            "reason": "oracle_promotion_gate_failed",
            "locked_test_used": False,
        }
        reconnect_decision = "skip_bounded_reconnect"
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "device": str(device),
        "protocol": {
            "seeds": list(args.seeds),
            "variants": list(VARIANTS),
            "teacher_level": 10,
            "student_levels": [40, 50],
            "heldout_development_level": 45,
            "pairs_per_combo": int(args.pairs_per_combo),
            "adapter_rank": int(args.adapter_rank),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
        },
        "sampling": sampling_payloads,
        "per_run": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "oracle_promotion_gate": gate,
        "bounded_reconnect_decision": reconnect_decision,
        "bounded_reconnect": reconnect,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(
        args.output_root / "route_a_sawfm_stage2_power_teacher_student_summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml",
    )
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen",
    )
    parser.add_argument(
        "--frozen-head-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--pairs-per-combo", type=int, default=40)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--adapter-limit", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--max-residual", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
