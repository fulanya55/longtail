"""P3a edge-preserving sample-domain ISRJ separator versus frozen P2z."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cj0724.conditional_unet_separator_experiment import ConditionalComplexUNetSeparator
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import (
    ISRJ_CLASS_INDEX, POWERS, REGIMES, WIDTH_COLUMN, sha256_file,
)
from cj0724.p2z_isrj_reset_consistent_separator import (
    ISRJSeparatorDataset, evaluate_separator, separation_objective,
    stratified_roles_10, variant_metrics,
)
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)


class TimeDomainResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        padding = 4 * int(dilation)
        self.network = nn.Sequential(
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv1d(
                width, width, kernel_size=9, padding=padding,
                dilation=int(dilation), groups=width, bias=False,
            ),
            nn.Conv1d(width, width, kernel_size=1, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class TimeDomainComplexResidualSeparator(nn.Module):
    def __init__(self, width: int = 16, dilations=(1, 2, 4, 8, 16, 32)) -> None:
        super().__init__()
        self.stem = nn.Conv1d(2, width, kernel_size=9, padding=4, bias=False)
        self.blocks = nn.Sequential(*[TimeDomainResidualBlock(width, dilation) for dilation in dilations])
        self.output = nn.Conv1d(width, 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def normalize_iq(iq: torch.Tensor) -> torch.Tensor:
        energy = iq.square().sum(1).mean(-1, keepdim=True).clamp_min(1.0e-12)
        return iq / torch.sqrt(energy)[:, None, :]

    def forward_components(self, iq: torch.Tensor, condition_logits=None) -> dict[str, torch.Tensor]:
        correction = self.output(self.blocks(self.stem(iq.float())))
        estimate = iq.float() + correction
        return {"estimated_iq": estimate, "normalized_estimate": self.normalize_iq(estimate)}


def train_candidate(
    files: dict[int, Path], roles: dict[str, np.ndarray], seed: int,
    run_dir: Path, device: torch.device, args: argparse.Namespace,
) -> tuple[TimeDomainComplexResidualSeparator, list[dict[str, object]], float]:
    set_seed(seed)
    model = TimeDomainComplexResidualSeparator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    datasets = {role: ISRJSeparatorDataset(files, roles[role]) for role in ("fit", "selection")}
    loaders = {
        "fit": DataLoader(
            datasets["fit"], batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
        ),
        "selection": DataLoader(
            datasets["selection"], batch_size=args.batch_size, shuffle=False,
            pin_memory=device.type == "cuda",
        ),
    }
    run_dir.mkdir(parents=True)
    best_loss, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_rows = []
            for batch in loaders["fit"]:
                optimizer.zero_grad(set_to_none=True)
                total, parts = separation_objective(
                    model, batch["iq"].to(device), batch["target_iq"].to(device), 0.5
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
                train_rows.append({"total": float(total.detach()), **{
                    name: float(value) for name, value in parts.items()
                }})
            model.eval()
            selection_rows = []
            with torch.inference_mode():
                for batch in loaders["selection"]:
                    total, parts = separation_objective(
                        model, batch["iq"].to(device), batch["target_iq"].to(device), 0.5
                    )
                    selection_rows.append({"total": float(total), **{
                        name: float(value) for name, value in parts.items()
                    }})
            train_mean = {name: float(np.mean([row[name] for row in train_rows])) for name in train_rows[0]}
            selection_mean = {
                name: float(np.mean([row[name] for row in selection_rows]))
                for name in selection_rows[0]
            }
            history.append({"epoch": epoch, "train": train_mean, "selection": selection_mean})
            if selection_mean["total"] < best_loss:
                best_loss, stale = selection_mean["total"], 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    finally:
        for dataset in datasets.values():
            dataset.close()
    return model, history, float(time.perf_counter() - started)


def load_p2z(seed: int, project: Path, root: str, device: torch.device):
    model = ConditionalComplexUNetSeparator(width=12, classifier_width=12, condition=False).to(device)
    checkpoint = project / root / f"seed_{seed}" / "S1_waveform_plus_reset" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        default="generated_data/p2z_isrj_reset_separator_train_seed3700000000_ID_control_rechunked",
    )
    parser.add_argument(
        "--development-root-template",
        default="generated_data/p3a_isrj_time_domain_dev_seed4000000000_{regime}_rechunked",
    )
    parser.add_argument(
        "--p2z-root", default="outputs/p2z_isrj_reset_consistent_separator_development_20260801"
    )
    parser.add_argument(
        "--training-protocol",
        default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json",
    )
    parser.add_argument(
        "--preregister", default="configs/p3a_isrj_time_domain_separator_preregister_20260801.json"
    )
    parser.add_argument(
        "--output-root", default="outputs/p3a_isrj_time_domain_separator_development_20260801"
    )
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    preregister = project / args.preregister
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = transition_files(project / args.train_root, "train", POWERS)
    train_alignment = audit_power_alignment(train_files)
    train_identities, train_labels = read_identity_metadata(train_files[POWERS[0]])
    with h5py.File(train_files[POWERS[0]], "r") as handle:
        train_mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
    all_roles = stratified_roles_10(train_labels, 20260801)
    roles = {
        name: indices[(train_labels[indices, ISRJ_CLASS_INDEX] > 0) & train_mask[indices]]
        for name, indices in all_roles.items()
    }

    protocol = json.loads((project / args.training_protocol).read_text(encoding="utf-8"))
    excluded_paths = [Path(protocol["data_file"])]
    excluded_paths.extend(
        Path(value) if Path(value).is_absolute() else project / value
        for value in args.excluded_identity_files
    )
    development = {}
    development_identity_reference = None
    for regime in REGIMES:
        files = transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS)
        alignment = audit_power_alignment(files)
        identities, labels = read_identity_metadata(files[POWERS[0]])
        with h5py.File(files[POWERS[0]], "r") as handle:
            mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
        selected = np.flatnonzero((labels[:, ISRJ_CLASS_INDEX] > 0) & mask)
        if len(selected) != 320:
            raise RuntimeError(f"P3a development count mismatch in {regime}")
        if development_identity_reference is None:
            development_identity_reference = identities
        elif not np.array_equal(development_identity_reference, identities):
            raise RuntimeError("P3a development regimes are not identity paired")
        if np.intersect1d(train_identities, identities).size:
            raise RuntimeError("P3a train/development identity overlap")
        exclusions = []
        for path in excluded_paths:
            prior, _ = read_identity_metadata(path)
            exclusions.append({
                "path": str(path.resolve()),
                "overlap_count": int(np.intersect1d(identities, prior).size),
            })
        if any(row["overlap_count"] for row in exclusions):
            raise RuntimeError("P3a development identity overlap")
        development[regime] = {
            "files": files, "identities": identities, "labels": labels,
            "selected": selected, "alignment": alignment, "exclusions": exclusions,
        }

    runs = []
    for seed in MODEL_SEEDS:
        run_dir = output_root / f"seed_{seed}"
        candidate, history, train_seconds = train_candidate(
            train_files, roles, seed, run_dir / "TD_Reset", device, args
        )
        baseline, baseline_checkpoint = load_p2z(seed, project, args.p2z_root, device)
        regimes = {}
        baseline_seconds = candidate_seconds = 0.0
        for regime, record in development.items():
            started = time.perf_counter()
            baseline_arrays = evaluate_separator(
                baseline, record["files"], record["selected"], record["labels"], device, args.batch_size
            )
            baseline_seconds += time.perf_counter() - started
            started = time.perf_counter()
            candidate_arrays = evaluate_separator(
                candidate, record["files"], record["selected"], record["labels"], device, args.batch_size
            )
            candidate_seconds += time.perf_counter() - started
            baseline_metrics = variant_metrics(baseline_arrays)
            candidate_metrics = variant_metrics(candidate_arrays)
            regime_dir = run_dir / regime
            regime_dir.mkdir(parents=True)
            np.savez_compressed(regime_dir / "baseline_predictions.npz", **baseline_arrays)
            np.savez_compressed(regime_dir / "candidate_predictions.npz", **candidate_arrays)
            (regime_dir / "result.json").write_text(json.dumps(_jsonable({
                "frozen_P2z": baseline_metrics, "TD_Reset": candidate_metrics,
            }), indent=2), encoding="utf-8")
            regimes[regime] = {"frozen_P2z": baseline_metrics, "TD_Reset": candidate_metrics}

        baseline_mae = np.mean([
            regimes[regime]["frozen_P2z"]["separated"]["overall"]["mae_s"] for regime in REGIMES
        ])
        candidate_mae = np.mean([
            regimes[regime]["TD_Reset"]["separated"]["overall"]["mae_s"] for regime in REGIMES
        ])
        run = {
            "seed": seed,
            "checkpoint_sha256": sha256_file(run_dir / "TD_Reset" / "best.pt"),
            "baseline_checkpoint": str(baseline_checkpoint.resolve()),
            "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
            "epochs_ran": len(history),
            "train_seconds": train_seconds,
            "baseline_evaluation_seconds": baseline_seconds,
            "candidate_evaluation_seconds": candidate_seconds,
            "mean_mae_reduction_vs_frozen_P2z": float(
                1.0 - candidate_mae / max(baseline_mae, 1.0e-12)
            ),
            "regimes": regimes,
        }
        (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
        (run_dir / "receipt.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8")
        runs.append(run)
        print(json.dumps({
            "seed": seed,
            "mean_mae_reduction_vs_frozen_P2z": run["mean_mae_reduction_vs_frozen_P2z"],
            "candidate": {
                regime: regimes[regime]["TD_Reset"]["separated"]["overall"] for regime in REGIMES
            },
        }), flush=True)
        del candidate, baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_rows = [
        run["regimes"][regime]["TD_Reset"] for run in runs for regime in REGIMES
    ]
    exact_rows = [runs[0]["regimes"][regime]["TD_Reset"]["exact"]["overall"] for regime in REGIMES]
    candidate_parameters = sum(parameter.numel() for parameter in TimeDomainComplexResidualSeparator().parameters())
    p2z_parameters = sum(parameter.numel() for parameter in ConditionalComplexUNetSeparator(
        width=12, classifier_width=12, condition=False
    ).parameters())
    gates = {
        "identity_overlap_zero": bool(all(
            exclusion["overlap_count"] == 0
            for record in development.values() for exclusion in record["exclusions"]
        )),
        "exact_each_regime_reduction_vs_constant_at_least_0p10": bool(all(
            row["mae_reduction_vs_constant"] >= 0.10 for row in exact_rows
        )),
        "exact_each_regime_pearson_at_least_0p30": bool(all(
            row["pearson"] >= 0.30 for row in exact_rows
        )),
        "candidate_all_15_reduction_vs_constant_at_least_0p10": bool(all(
            row["separated"]["overall"]["mae_reduction_vs_constant"] >= 0.10 for row in candidate_rows
        )),
        "candidate_all_15_pearson_at_least_0p30": bool(all(
            row["separated"]["overall"]["pearson"] >= 0.30 for row in candidate_rows
        )),
        "candidate_all_15_catastrophic_at_most_0p10": bool(all(
            row["separated"]["overall"]["catastrophic_fraction"] <= 0.10 for row in candidate_rows
        )),
        "candidate_compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(
            row["separated"]["compound"]["mae_reduction_vs_constant"] >= 0.10 for row in candidate_rows
        )),
        "candidate_compound_all_15_pearson_at_least_0p30": bool(all(
            row["separated"]["compound"]["pearson"] >= 0.30 for row in candidate_rows
        )),
        "candidate_all_15_mae_better_than_raw": bool(all(
            row["separated"]["overall"]["mae_s"] < row["raw"]["overall"]["mae_s"]
            for row in candidate_rows
        )),
        "candidate_each_seed_mean_reduction_vs_P2z_at_least_0p20": bool(all(
            run["mean_mae_reduction_vs_frozen_P2z"] >= 0.20 for run in runs
        )),
        "candidate_mean_reduction_vs_P2z_at_least_0p30": bool(np.mean([
            run["mean_mae_reduction_vs_frozen_P2z"] for run in runs
        ]) >= 0.30),
        "candidate_all_15_waveform_cosine_at_least_0p30": bool(all(
            row["waveform_cosine_mean"] >= 0.30 for row in candidate_rows
        )),
        "candidate_parameter_fraction_at_most_0p25": bool(
            candidate_parameters / p2z_parameters <= 0.25
        ),
    }
    passed = bool(all(gates.values()))
    summary = {
        "experiment": "P3a edge-preserving time-domain ISRJ separator",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "device": str(device),
        "identity_audit": {
            "train": {
                "identity_count": int(len(train_identities)),
                "identity_sha256": _hash_int64(train_identities),
                "alignment": train_alignment,
            },
            "development": {
                "identity_count": int(len(development_identity_reference)),
                "identity_sha256": _hash_int64(development_identity_reference),
                "cross_train_overlap": int(np.intersect1d(train_identities, development_identity_reference).size),
                "per_regime": {
                    regime: {"alignment": record["alignment"], "exclusions": record["exclusions"]}
                    for regime, record in development.items()
                },
            },
        },
        "role_counts": {name: int(len(indices)) for name, indices in roles.items()},
        "parameters": {
            "TD_Reset": int(candidate_parameters),
            "P2z_STFT_Reset": int(p2z_parameters),
            "fraction": float(candidate_parameters / p2z_parameters),
        },
        "runs": runs,
        "promotion_gates": gates,
        "development_passed": passed,
        "interpretation": (
            "The edge-preserving time-domain separator passed; freeze it for two fresh-seed confirmation runs."
            if passed else
            "The time-domain separator did not transfer the exact-component oracle into compound I/Q. Stop learned ISRJ waveform-separator development and retain P2x as oracle/identifiability evidence only."
        ),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "development_passed": passed,
        "promotion_gates": gates,
        "parameters": summary["parameters"],
        "mean_reduction_vs_P2z": float(np.mean([
            run["mean_mae_reduction_vs_frozen_P2z"] for run in runs
        ])),
        "interpretation": summary["interpretation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
