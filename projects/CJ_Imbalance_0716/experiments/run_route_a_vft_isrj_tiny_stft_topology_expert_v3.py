"""Tiny 2D STFT topology experts for strict VFT and ISRJ separability."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (  # noqa: E402
    _load_roles, _matched_view, _rank_metrics,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


TARGETS = ((4, "VFT"), (5, "ISRJ"))
VARIANTS = ("A_Aligned", "S_TimeShuffled")
ROLES = ("fit", "validation", "early")


class TinySTFTTopologyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, values):
        maps = self.features(values)
        pooled = torch.cat([maps.mean(dim=(2, 3)), maps.amax(dim=(2, 3))], dim=1)
        return self.classifier(pooled).squeeze(1)


def _stft_images(signal, grid_size=32):
    z = torch.complex(signal[:, 0].float(), signal[:, 1].float())
    window = torch.hann_window(128, device=z.device)
    spectrum = torch.stft(z, n_fft=128, hop_length=32, window=window, center=True,
                          return_complex=True, onesided=False)
    power = torch.fft.fftshift(spectrum.abs().square(), dim=1)
    distribution = power / power.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    image = F.adaptive_avg_pool2d(torch.log1p(distribution * 128.0)[:, None],
                                  (int(grid_size), int(grid_size)))
    return image


def _extract_images(signals, device, batch_size):
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows.append(_stft_images(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def _permutation(size, seed):
    values = np.random.default_rng(int(seed)).permutation(int(size)).astype(np.int64)
    return values, hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _apply_time_permutation(images, permutation):
    return np.asarray(images)[:, :, :, np.asarray(permutation, dtype=np.int64)]


def _scores(model, values, device, batch_size=512):
    rows = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            rows.append(model(torch.from_numpy(values[start:start + int(batch_size)]).to(device)).cpu().numpy())
    return np.concatenate(rows)


def _train(fit_x, fit_y, validation_x, validation_y, seed, device, args):
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    model = TinySTFTTopologyExpert().to(device)
    positives = max(float(fit_y.sum()), 1.0); negatives = max(float(len(fit_y) - fit_y.sum()), 1.0)
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(int(seed)), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_value, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device)); target = y.to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        metrics = _rank_metrics(validation_y, _scores(model, validation_x, device))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_auroc": metrics["auroc"], "validation_aupr": metrics["aupr"]})
        if metrics["auroc"] > best_value + 1e-8:
            best_value, best_epoch, stale = metrics["auroc"], epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.patience): break
    if best_state is None: raise RuntimeError("topology expert did not produce checkpoint")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": int(seed),
                            "validation_auroc": best_value}


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; summary_path = root / "seed_summary.json"
    if args.resume and summary_path.is_file(): return json.loads(summary_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    images = {role: _extract_images(roles[role]["signals"], device, args.stft_batch_size) for role in ROLES}
    mean = float(images["fit"].mean()); std = float(max(images["fit"].std(), 1e-6))
    images = {role: ((values - mean) / std).astype(np.float32) for role, values in images.items()}
    prior = json.loads((args.layerwise_root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    results, counts = {}, {}
    parameter_count = int(sum(parameter.numel() for parameter in TinySTFTTopologyExpert().parameters()))
    for target_offset, (target_index, target_name) in enumerate(TARGETS):
        views = {}; counts[target_name] = {}
        for role in ROLES:
            selected, labels, receipt = _matched_view(
                roles[role]["targets6"], weak6[roles[role]["sample_ids"]], target_index
            )
            views[role] = (images[role][selected], labels,
                           np.asarray(roles[role]["sample_ids"])[selected])
            counts[target_name][role] = receipt
        permutation_seed = 20297000 + int(seed) * 10 + target_offset
        permutation, permutation_sha = _permutation(32, permutation_seed)
        results[target_name] = {}
        train_seed = 20298000 + int(seed) * 10 + target_offset
        for variant in VARIANTS:
            transformed = {
                role: (views[role][0] if variant == "A_Aligned" else
                       _apply_time_permutation(views[role][0], permutation)).astype(np.float32)
                for role in ROLES
            }
            model, history, info = _train(
                transformed["fit"], views["fit"][1], transformed["validation"],
                views["validation"][1], train_seed, device, args,
            )
            validation_scores = _scores(model, transformed["validation"], device)
            early_scores = _scores(model, transformed["early"], device)
            validation = _rank_metrics(views["validation"][1], validation_scores)
            early = _rank_metrics(views["early"][1], early_scores)
            output = root / target_name / variant; output.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "image_mean": mean, "image_std": std,
                        "target": target_name, "variant": variant, "parameter_count": parameter_count,
                        "time_permutation": permutation, "time_permutation_sha256": permutation_sha, **info},
                       output / "best_model.pth")
            _write_history(output / "history.csv", history)
            np.savez_compressed(output / "predictions.npz",
                                validation_sample_ids=views["validation"][2],
                                validation_labels=views["validation"][1].astype(np.uint8),
                                validation_scores=validation_scores,
                                early_sample_ids=views["early"][2],
                                early_labels=views["early"][1].astype(np.uint8), early_scores=early_scores)
            results[target_name][variant] = {"validation": validation, "early": early,
                                              "validation_early_auroc_gap": validation["auroc"] - early["auroc"],
                                              "parameter_count": parameter_count, **info}
        baseline = prior["representations"]["T_FinalTemporal"][target_name]["early"]["auroc"]
        aligned = results[target_name]["A_Aligned"]["early"]["auroc"]
        shuffled = results[target_name]["S_TimeShuffled"]["early"]["auroc"]
        results[target_name]["comparisons"] = {
            "final_temporal_early_auroc": baseline,
            "aligned_minus_final_temporal": aligned - baseline,
            "aligned_minus_time_shuffled": aligned - shuffled,
        }
        results[target_name]["time_permutation"] = {"seed": permutation_seed, "sha256": permutation_sha}
    payload = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
               "image_normalization": {"source_role": "fit_all", "mean": mean, "std": std},
               "parameter_count": parameter_count, "labels": results, "matched_counts": counts,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload); return payload


def run(args):
    config = load_config(args.config); _validate_config(config); data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]: raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    aggregate, gates = {}, {}
    for label_offset, (_, name) in enumerate(TARGETS):
        aligned = [row["labels"][name]["A_Aligned"] for row in rows]
        comparisons = [row["labels"][name]["comparisons"] for row in rows]
        aggregate[name] = {
            "aligned_early_auroc": _mean_ci([row["early"]["auroc"] for row in aligned], 20299000 + label_offset * 100),
            "aligned_early_aupr": _mean_ci([row["early"]["aupr"] for row in aligned], 20299001 + label_offset * 100),
            "aligned_minus_final_temporal": _mean_ci([row["aligned_minus_final_temporal"] for row in comparisons], 20299002 + label_offset * 100),
            "aligned_minus_time_shuffled": _mean_ci([row["aligned_minus_time_shuffled"] for row in comparisons], 20299003 + label_offset * 100),
            "validation_early_gap": _mean_ci([row["validation_early_auroc_gap"] for row in aligned], 20299004 + label_offset * 100),
            "positive_vs_final_seeds": int(sum(row["aligned_minus_final_temporal"] > 0 for row in comparisons)),
            "positive_vs_shuffled_seeds": int(sum(row["aligned_minus_time_shuffled"] > 0 for row in comparisons)),
            "parameter_count": int(aligned[0]["parameter_count"]),
        }
        if args.smoke:
            gates[name] = {"checks": {}, "passed": False}
        else:
            checks = {"absolute_auroc_ge_0p620": aggregate[name]["aligned_early_auroc"]["mean"] >= .620,
                      "gain_vs_final_ge_0p030": aggregate[name]["aligned_minus_final_temporal"]["mean"] >= .030,
                      "gain_vs_shuffled_ge_0p020": aggregate[name]["aligned_minus_time_shuffled"]["mean"] >= .020,
                      "four_of_five_vs_final": aggregate[name]["positive_vs_final_seeds"] >= 4,
                      "four_of_five_vs_shuffled": aggregate[name]["positive_vs_shuffled_seeds"] >= 4,
                      "validation_early_gap_lt_0p150": aggregate[name]["validation_early_gap"]["mean"] < .150,
                      "same_capacity_exact": all(row["labels"][name]["A_Aligned"]["parameter_count"] == row["labels"][name]["S_TimeShuffled"]["parameter_count"] for row in rows)}
            gates[name] = {"checks": checks, "passed": bool(all(checks.values()))}
    payload = {"status": "completed", "experiment": "RouteA_VFT_ISRJ_TinySTFTTopologyExpert_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "component_promotion_gate": gates,
               "promoted_labels": [name for name, gate in gates.items() if gate["passed"]],
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_vft_isrj_tiny_stft_topology_expert_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--layerwise-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_layerwise_matched_negative_separability_v3_strict")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_vft_isrj_tiny_stft_topology_expert_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043]); parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--stft-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8); parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0); return parser.parse_args()


if __name__ == "__main__": run(parse_args())

