"""B6-S: physical sufficiency screen for two independent ISRJ evidence domains."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from cj0724.b3r_unified_pipeline import read_and_normalize
from cj0724.dual_axis import CLASS_NAMES, POWER_LEVELS
from cj0724.route_b_mechanism_router import SEEDS, baseline_root
from cj0724.shared_physics_frontend import shared_feature_batch


ISRJ_INDEX = 5
MFT_INDEX = 3
VFT_INDEX = 4


def _pool_max(values: torch.Tensor, bins: int) -> torch.Tensor:
    usable = values.shape[1] // bins * bins
    return values[:, :usable].reshape(len(values), bins, -1).amax(dim=2)


def _pool_mean(values: torch.Tensor, bins: int) -> torch.Tensor:
    usable = values.shape[1] // bins * bins
    return values[:, :usable].reshape(len(values), bins, -1).mean(dim=2)


@torch.no_grad()
def cross_domain_feature_batch(iq: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return compact envelope-periodicity and comb-spectrum evidence."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 3000):
        raise ValueError("iq must have shape [B,2,3000]")
    signal = torch.complex(iq[:, 0], iq[:, 1])
    signal = signal / signal.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)

    envelope = signal.abs()
    envelope = envelope - envelope.mean(dim=1, keepdim=True)
    envelope_fft = torch.fft.fft(envelope, n=8192, dim=1)
    envelope_acf = torch.fft.ifft(envelope_fft.abs().square(), dim=1).abs()[:, 1:1201]
    envelope_acf = envelope_acf / envelope_acf[:, :1].clamp_min(1e-8)
    periodic = envelope_acf[:, 39:600]
    envelope_bins = _pool_max(periodic, 60)
    envelope_top = periodic.topk(12, dim=1)
    envelope_positions = envelope_top.indices.sort(dim=1).values.float() / periodic.shape[1]
    envelope_strength = envelope_top.values.sort(dim=1).values
    envelope_summary = torch.stack((
        periodic.amax(dim=1), periodic.mean(dim=1), periodic.std(dim=1),
        periodic.topk(2, dim=1).values[:, 0] - periodic.topk(2, dim=1).values[:, 1],
    ), dim=1)
    envelope_features = torch.cat(
        (envelope_bins, envelope_positions, envelope_strength, envelope_summary), dim=1
    )

    spectrum = torch.fft.fftshift(torch.fft.fft(signal, n=4096, dim=1), dim=1).abs()
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1e-8)
    baseline = F.avg_pool1d(spectrum[:, None], kernel_size=33, stride=1, padding=16)[:, 0]
    residual = (spectrum - baseline).clamp_min(0.0)
    residual = residual / residual.sum(dim=1, keepdim=True).clamp_min(1e-8)
    residual_fft = torch.fft.rfft(residual, n=8192, dim=1)
    spectral_acf = torch.fft.irfft(residual_fft.abs().square(), n=8192, dim=1).abs()[:, 1:1025]
    spectral_acf = spectral_acf / spectral_acf[:, :1].clamp_min(1e-8)
    spectral_lag_bins = _pool_max(spectral_acf, 64)
    cepstrum = torch.fft.ifft(torch.log(spectrum.clamp_min(1e-9)).to(torch.complex64), dim=1).abs()[:, 1:1025]
    cepstrum = cepstrum / cepstrum[:, :1].clamp_min(1e-8)
    cepstrum_bins = _pool_mean(cepstrum, 64)
    peaks = residual.topk(12, dim=1)
    positions = peaks.indices.sort(dim=1).values.float() / residual.shape[1]
    spacings = positions[:, 1:] - positions[:, :-1]
    peak_strength = peaks.values.sort(dim=1).values
    comb_summary = torch.stack((
        spectral_acf.amax(dim=1), cepstrum.amax(dim=1),
        spacings.mean(dim=1), spacings.std(dim=1),
    ), dim=1)
    comb_features = torch.cat(
        (spectral_lag_bins, cepstrum_bins, spacings, peak_strength, comb_summary), dim=1
    )
    return {
        "envelope_periodicity": envelope_features.float(),
        "comb_spectrum": comb_features.float(),
    }


@torch.no_grad()
def extract_domains(
    path: Path, indices: np.ndarray, device: torch.device, batch_size: int,
) -> tuple[dict[str, np.ndarray], float]:
    started = time.perf_counter()
    iq, _, _ = read_and_normalize(path, indices)
    tensor = torch.from_numpy(iq).to(device)
    parts = {"envelope_periodicity": [], "comb_spectrum": [], "reference": []}
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        domains = cross_domain_feature_batch(batch)
        reference = shared_feature_batch(batch)["isrj_periodicity"]
        for key in domains:
            parts[key].append(domains[key].cpu().numpy())
        parts["reference"].append(reference.cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}, time.perf_counter() - started


def identity_balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    target = np.asarray(labels).astype(bool)
    weights = np.zeros(len(target), dtype=np.float64)
    weights[target] = 0.5 / max(int(target.sum()), 1)
    weights[~target] = 0.5 / max(int((~target).sum()), 1)
    return weights * len(target)


def binary_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    target = np.asarray(labels).astype(bool)
    predicted = np.asarray(probability) >= float(threshold)
    return {
        "aupr": float(average_precision_score(target, probability)),
        "recall": float(predicted[target].mean()),
        "fpr": float(predicted[~target].mean()),
    }


def select_safe_threshold(
    labels: np.ndarray, probability: np.ndarray, reference_probability: np.ndarray,
    reference_threshold: float, powers: np.ndarray, hard_negative: np.ndarray,
) -> tuple[float, bool, list[dict[str, float]]]:
    target = np.asarray(labels).astype(bool)
    hard_negative = np.asarray(hard_negative).astype(bool)
    candidates = np.unique(np.concatenate((
        np.linspace(0.001, 0.999, 200),
        np.quantile(probability, np.linspace(0.0, 1.0, 101)),
    )))
    rows = []
    for threshold in candidates:
        recall_drops = []
        for power in POWER_LEVELS:
            mask = target & (powers == power)
            reference_recall = float((reference_probability[mask] >= reference_threshold).mean())
            recall = float((probability[mask] >= threshold).mean())
            recall_drops.append(reference_recall - recall)
        rows.append({
            "threshold": float(threshold),
            "max_power_recall_drop": float(max(recall_drops)),
            "hard_negative_fpr": float((probability[hard_negative] >= threshold).mean()),
        })
    eligible = [row for row in rows if row["max_power_recall_drop"] <= 0.005]
    selected = min(eligible, key=lambda row: (row["hard_negative_fpr"], -row["threshold"])) if eligible else min(
        rows, key=lambda row: (row["max_power_recall_drop"], row["hard_negative_fpr"])
    )
    return float(selected["threshold"]), bool(eligible), rows


def screen_slice(
    labels: np.ndarray, reference: np.ndarray, envelope: np.ndarray, comb: np.ndarray,
    reference_threshold: float, envelope_threshold: float, comb_threshold: float,
) -> dict[str, float]:
    positive = labels[:, ISRJ_INDEX].astype(bool)
    hard_negative = (~positive) & (
        labels[:, MFT_INDEX].astype(bool) | labels[:, VFT_INDEX].astype(bool)
    )
    reference_predicted = reference >= reference_threshold
    envelope_predicted = envelope >= envelope_threshold
    comb_predicted = comb >= comb_threshold
    false_negative = positive & ~reference_predicted
    hard_false_positive = hard_negative & reference_predicted
    envelope_rescue = false_negative & envelope_predicted
    comb_rescue = false_negative & comb_predicted
    joint_negative = (~envelope_predicted) & (~comb_predicted)
    return {
        "reference_false_negative_count": int(false_negative.sum()),
        "reference_hard_false_positive_count": int(hard_false_positive.sum()),
        "union_rescue": float(((envelope_rescue | comb_rescue).sum()) / max(int(false_negative.sum()), 1)),
        "envelope_unique_rescue": float((envelope_rescue & ~comb_rescue).sum() / max(int(false_negative.sum()), 1)),
        "comb_unique_rescue": float((comb_rescue & ~envelope_rescue).sum() / max(int(false_negative.sum()), 1)),
        "joint_hard_false_positive_correction": float(
            (hard_false_positive & joint_negative).sum() / max(int(hard_false_positive.sum()), 1)
        ),
        "joint_negative_positive_harm": float((positive & reference_predicted & joint_negative).sum() / max(int(positive.sum()), 1)),
        "domain_disagreement": float((envelope_predicted != comb_predicted).mean()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/b6_cross_domain_sufficiency_20260725"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    power_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir()
        baseline = baseline_root(seed)
        route_root = Path(f"outputs/route_b_mechanism_router_20260725/seed_{seed}")
        b2_root = Path(f"outputs/b2_power_boundary_20260725/seed_{seed}")
        audit = json.loads((baseline / "data_audit.json").read_text(encoding="utf-8"))
        files = {power: Path(audit["levels"][str(power)]["path"]) for power in POWER_LEVELS}
        split = np.load(baseline / "split_indices.npz")
        manifest = json.loads((baseline / "longtail_manifest.json").read_text(encoding="utf-8"))
        roles = {
            "train": np.sort(np.asarray(manifest["selected_indices"], dtype=np.int64)),
            "early": np.sort(split["early_stop"].astype(np.int64)),
            "development": np.sort(split["development"].astype(np.int64)),
        }
        all_indices = np.sort(np.concatenate(list(roles.values())))
        lookup = {int(index): position for position, index in enumerate(all_indices)}
        positions = {role: np.asarray([lookup[int(index)] for index in indices], dtype=np.int64)
                     for role, indices in roles.items()}
        with h5py.File(files[10], "r") as handle:
            labels_all = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        data = {role: {"labels": [], "powers": [], "envelope_periodicity": [],
                       "comb_spectrum": [], "reference": []} for role in roles}
        feature_seconds = 0.0
        for power in POWER_LEVELS:
            features, elapsed = extract_domains(files[power], all_indices, device, args.batch_size)
            feature_seconds += elapsed
            for role in roles:
                data[role]["labels"].append(labels_all[roles[role]])
                data[role]["powers"].append(np.full(len(roles[role]), power, dtype=np.int16))
                for key in ("envelope_periodicity", "comb_spectrum", "reference"):
                    data[role][key].append(features[key][positions[role]])
        for role in data:
            data[role] = {key: np.concatenate(value) for key, value in data[role].items()}
        reference_model = joblib.load(route_root / "isrj_physics_classifier.joblib")
        train_reference = reference_model.predict_proba(data["train"]["reference"])[:, 1]
        del train_reference
        models = {}
        fit_seconds = 0.0
        target = data["train"]["labels"][:, ISRJ_INDEX]
        weights = identity_balanced_class_weights(target)
        for domain in ("envelope_periodicity", "comb_spectrum"):
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                l2_regularization=1.0, random_state=seed,
            )
            tick = time.perf_counter()
            model.fit(data["train"][domain], target, sample_weight=weights)
            fit_seconds += time.perf_counter() - tick
            models[domain] = model
            joblib.dump(model, seed_root / f"{domain}.joblib")
        early_count = len(roles["early"])
        repeated_identity = np.tile(roles["early"], len(POWER_LEVELS))
        threshold_identity = roles["early"][:early_count // 2]
        safety_identity = roles["early"][early_count // 2:]
        threshold_mask = np.isin(repeated_identity, threshold_identity)
        safety_mask = np.isin(repeated_identity, safety_identity)
        early_reference = reference_model.predict_proba(data["early"]["reference"])[:, 1]
        reference_threshold = float(np.load(b2_root / "power_10dB.npz")["thresholds_B2-Hard"][ISRJ_INDEX])
        thresholds = {}
        safe = {}
        early_probability = {}
        for domain, model in models.items():
            probability = model.predict_proba(data["early"][domain])[:, 1]
            early_probability[domain] = probability
            threshold, feasible, candidates = select_safe_threshold(
                data["early"]["labels"][threshold_mask, ISRJ_INDEX], probability[threshold_mask],
                early_reference[threshold_mask], reference_threshold, data["early"]["powers"][threshold_mask],
                ((~data["early"]["labels"][threshold_mask, ISRJ_INDEX].astype(bool))
                 & (data["early"]["labels"][threshold_mask, MFT_INDEX].astype(bool)
                    | data["early"]["labels"][threshold_mask, VFT_INDEX].astype(bool))),
            )
            thresholds[domain] = threshold
            safety_drops = []
            for power in POWER_LEVELS:
                mask = safety_mask & (data["early"]["powers"] == power) & data["early"]["labels"][:, ISRJ_INDEX].astype(bool)
                reference_recall = float((early_reference[mask] >= reference_threshold).mean())
                domain_recall = float((probability[mask] >= threshold).mean())
                safety_drops.append(reference_recall - domain_recall)
            safe[domain] = bool(feasible and max(safety_drops) <= 0.01)
            calibration_rows.append({
                "seed": seed, "domain": domain, "threshold": threshold,
                "threshold_fit_feasible": feasible, "max_safety_recall_drop": max(safety_drops),
                "safety_passed": safe[domain], "candidate_count": len(candidates),
            })
        (seed_root / "protocol.json").write_text(json.dumps({
            "seed": seed, "thresholds": thresholds, "safety": safe,
            "threshold_role": "early-stop first half", "safety_role": "early-stop second half",
            "development_used_for_selection": False, "locked_test_used": False,
        }, indent=2), encoding="utf-8")
        seed_primary = None
        for power_index, power in enumerate(POWER_LEVELS):
            start = power_index * len(roles["development"])
            stop = start + len(roles["development"])
            labels = data["development"]["labels"][start:stop]
            reference = reference_model.predict_proba(data["development"]["reference"][start:stop])[:, 1]
            envelope = models["envelope_periodicity"].predict_proba(
                data["development"]["envelope_periodicity"][start:stop])[:, 1]
            comb = models["comb_spectrum"].predict_proba(data["development"]["comb_spectrum"][start:stop])[:, 1]
            reference_metric = binary_metrics(labels[:, ISRJ_INDEX], reference, reference_threshold)
            envelope_metric = binary_metrics(labels[:, ISRJ_INDEX], envelope, thresholds["envelope_periodicity"])
            comb_metric = binary_metrics(labels[:, ISRJ_INDEX], comb, thresholds["comb_spectrum"])
            slice_metric = screen_slice(
                labels, reference, envelope, comb, reference_threshold,
                thresholds["envelope_periodicity"], thresholds["comb_spectrum"],
            )
            row = {
                "seed": seed, "power_dB": power,
                **{f"reference_{key}": value for key, value in reference_metric.items()},
                **{f"envelope_{key}": value for key, value in envelope_metric.items()},
                **{f"comb_{key}": value for key, value in comb_metric.items()},
                "envelope_aupr_delta": envelope_metric["aupr"] - reference_metric["aupr"],
                "comb_aupr_delta": comb_metric["aupr"] - reference_metric["aupr"],
                **slice_metric,
            }
            power_rows.append(row)
            if power == 10:
                seed_primary = row
            print(json.dumps({"seed": seed, "power": power,
                              "envelope_aupr_delta": row["envelope_aupr_delta"],
                              "comb_aupr_delta": row["comb_aupr_delta"],
                              "union_rescue": row["union_rescue"],
                              "joint_correction": row["joint_hard_false_positive_correction"],
                              "harm": row["joint_negative_positive_harm"]}), flush=True)
        assert seed_primary is not None
        seed_rows.append({
            "seed": seed, "calibration_safe": bool(all(safe.values())),
            **{key: seed_primary[key] for key in (
                "envelope_aupr_delta", "comb_aupr_delta", "union_rescue",
                "envelope_unique_rescue", "comb_unique_rescue",
                "joint_hard_false_positive_correction", "joint_negative_positive_harm",
            )},
        })
        cost_rows.append({
            "seed": seed, "feature_seconds": feature_seconds, "fit_seconds": fit_seconds,
            "model_bytes": int(sum(path.stat().st_size for path in seed_root.glob("*.joblib"))),
        })
    means = {
        key: float(np.mean([row[key] for row in seed_rows])) for key in (
            "envelope_aupr_delta", "comb_aupr_delta", "union_rescue",
            "envelope_unique_rescue", "comb_unique_rescue",
            "joint_hard_false_positive_correction", "joint_negative_positive_harm",
        )
    }
    same_direction = sum(
        row["union_rescue"] > 0 and row["joint_hard_false_positive_correction"] > 0
        and row["joint_negative_positive_harm"] <= 0.01 for row in seed_rows
    )
    calibration_safe = sum(row["calibration_safe"] for row in seed_rows)
    summary = {**means, "same_direction_seeds": int(same_direction),
               "calibration_safe_seeds": int(calibration_safe), "seed_rows": seed_rows}
    gates = {
        "envelope_10dB_aupr_delta_at_least_minus_0p03": means["envelope_aupr_delta"] >= -0.03,
        "comb_10dB_aupr_delta_at_least_minus_0p03": means["comb_aupr_delta"] >= -0.03,
        "union_rescue_at_least_0p10": means["union_rescue"] >= 0.10,
        "envelope_unique_rescue_at_least_0p02": means["envelope_unique_rescue"] >= 0.02,
        "comb_unique_rescue_at_least_0p02": means["comb_unique_rescue"] >= 0.02,
        "joint_hard_FP_correction_at_least_0p10": means["joint_hard_false_positive_correction"] >= 0.10,
        "joint_negative_positive_harm_at_most_0p005": means["joint_negative_positive_harm"] <= 0.005,
        "same_direction_at_least_4_seeds": same_direction >= 4,
        "calibration_safe_at_least_4_seeds": calibration_safe >= 4,
    }
    result = {
        "protocol": {"stage": "B6-S", "seeds": list(SEEDS), "powers_dB": list(POWER_LEVELS),
                     "router_constructed": False, "development_used_for_selection": False,
                     "explicit_JNR_used": False, "locked_test_used": False},
        "summary": summary, "gates": gates,
        "B6_physical_sufficiency_passed": bool(all(gates.values())),
        "allow_router_construction": bool(all(gates.values())),
        "cost": {"mean_feature_seconds_per_seed": float(np.mean([row["feature_seconds"] for row in cost_rows])),
                 "mean_fit_seconds_per_seed": float(np.mean([row["fit_seconds"] for row in cost_rows])),
                 "mean_model_bytes_per_seed": float(np.mean([row["model_bytes"] for row in cost_rows]))},
        "locked_test_used": False,
    }
    _write_csv(args.output_root / "power_runs.csv", power_rows)
    _write_csv(args.output_root / "seed_summary.csv", seed_rows)
    _write_csv(args.output_root / "calibration.csv", calibration_rows)
    _write_csv(args.output_root / "cost.csv", cost_rows)
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
