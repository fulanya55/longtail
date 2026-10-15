"""P3b diagnostic: regime-labelled upper bound for P3a ISRJ features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import ISRJ_CLASS_INDEX, POWERS, REGIMES, WIDTH_COLUMN, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3a_isrj_multiphysics_ordinal import (
    VARIANT_CHANNELS, collect_role, evaluate_predictions, predict_model, train_model,
)


def load_saved_global(project: Path, root: str, regime: str) -> dict[str, np.ndarray]:
    path = project / root / "seed_2041" / "A0_acf_ordinal" / regime / "predictions.npz"
    saved = np.load(path)
    return {name: saved[name] for name in saved.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root-template",
        default="generated_data/p3a_isrj_ordinal_train_seed3400000000_{regime}_rechunked",
    )
    parser.add_argument(
        "--development-root-template",
        default="generated_data/p3a_isrj_ordinal_dev_seed3500000000_{regime}_rechunked",
    )
    parser.add_argument(
        "--global-root", default="outputs/p3a_isrj_multiphysics_ordinal_development_20260801"
    )
    parser.add_argument(
        "--preregister", default="configs/p3b_isrj_regime_oracle_sufficiency_preregister_20260801.json"
    )
    parser.add_argument(
        "--output-root", default="outputs/p3b_isrj_regime_oracle_sufficiency_20260801"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    preregister = project / args.preregister
    roots = {
        "train": {
            regime: transition_files(project / args.train_root_template.format(regime=regime), "train", POWERS)
            for regime in REGIMES
        },
        "development": {
            regime: transition_files(project / args.development_root_template.format(regime=regime), "train", POWERS)
            for regime in REGIMES
        },
    }
    metadata = {"train": {}, "development": {}}
    identity_reference = {}
    for role in ("train", "development"):
        reference = None
        for regime in REGIMES:
            files = roots[role][regime]
            alignment = audit_power_alignment(files)
            identities, labels = read_identity_metadata(files[POWERS[0]])
            with h5py.File(files[POWERS[0]], "r") as handle:
                values = np.asarray(handle["parameter_values"], dtype=np.float64)
                mask = np.asarray(handle["parameter_mask"][WIDTH_COLUMN, :], dtype=np.uint8).astype(bool)
            selected = np.flatnonzero(
                (labels[:, ISRJ_CLASS_INDEX] > 0) & mask & np.isfinite(values[WIDTH_COLUMN])
            )
            if len(selected) != 320:
                raise RuntimeError(f"P3b {role} count mismatch in {regime}")
            if reference is None:
                reference = identities
            elif not np.array_equal(reference, identities):
                raise RuntimeError(f"P3b {role} regimes are not identity paired")
            metadata[role][regime] = {
                "identities": identities, "labels": labels, "values": values,
                "selected": selected, "alignment": alignment,
            }
        identity_reference[role] = reference
    if np.intersect1d(identity_reference["train"], identity_reference["development"]).size:
        raise RuntimeError("P3b train/development identity overlap")

    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801)
    fit_data = collect_role(roots["train"], metadata["train"], roles["fit"])
    selection_data = collect_role(roots["train"], metadata["train"], roles["selection"])
    development_data = collect_role(roots["development"], metadata["development"], None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for regime_index, regime in enumerate(REGIMES):
        fit_mask = fit_data["regime"] == regime_index
        selection_mask = selection_data["regime"] == regime_index
        development_mask = development_data["regime"] == regime_index
        development_subset = {
            name: values[development_mask]
            for name, values in development_data.items() if name != "features"
        }
        global_saved = load_saved_global(project, args.global_root, regime)
        if not np.array_equal(global_saved["identity"], development_subset["identity"]):
            raise RuntimeError(f"P3b saved global identity mismatch in {regime}")
        global_result = evaluate_predictions(global_saved["prediction"], development_subset)
        variants = {"G0_global_A0": global_result}
        for variant, channels in VARIANT_CHANNELS.items():
            oracle_name = "O1_regime_A0" if variant == "A0_acf_ordinal" else "O2_regime_A1"
            run_dir = output_root / regime / oracle_name
            model, history, elapsed = train_model(
                fit_data["features"][fit_mask], fit_data["target_s"][fit_mask],
                selection_data["features"][selection_mask], selection_data["target_s"][selection_mask],
                channels, 2041, run_dir, device, args,
            )
            prediction = predict_model(
                model, development_data["features"][development_mask],
                run_dir / "normalization.npz", device, args.batch_size,
            )
            result = evaluate_predictions(prediction, development_subset)
            variants[oracle_name] = result
            np.savez_compressed(
                run_dir / "predictions.npz",
                identity=development_subset["identity"], power=development_subset["power"],
                combination_size=development_subset["combination_size"],
                target_s=development_subset["target_s"], prediction=prediction,
            )
            (run_dir / "history.json").write_text(json.dumps(_jsonable(history), indent=2), encoding="utf-8")
            (run_dir / "receipt.json").write_text(json.dumps(_jsonable({
                "variant": oracle_name, "channels": channels, "epochs_ran": len(history),
                "train_seconds": elapsed, "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                "metrics": result,
            }), indent=2), encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[regime] = variants
        print(json.dumps({
            "regime": regime,
            "G0": global_result["overall"],
            "O1": variants["O1_regime_A0"]["overall"],
            "O2": variants["O2_regime_A1"]["overall"],
        }), flush=True)

    def absolute_pass(name: str) -> bool:
        return bool(all(
            results[regime][name]["overall"]["mae_reduction_vs_constant"] >= 0.10
            and results[regime][name]["overall"]["pearson"] >= 0.30
            and results[regime][name]["compound"]["mae_reduction_vs_constant"] >= 0.10
            and results[regime][name]["compound"]["pearson"] >= 0.30
            for regime in REGIMES
        ))

    oracle_candidates = ("O1_regime_A0", "O2_regime_A1")
    absolute = {name: absolute_pass(name) for name in oracle_candidates}
    low_gain = {}
    for name in oracle_candidates:
        low_gain[name] = {
            regime: float(
                results[regime][name]["overall"]["pearson"]
                - results[regime]["G0_global_A0"]["overall"]["pearson"]
            )
            for regime in ("near_low", "far_low")
        }
    passed = bool(any(
        absolute[name]
        and low_gain[name]["near_low"] >= 0.20
        and low_gain[name]["far_low"] >= 0.20
        for name in oracle_candidates
    ))
    gates = {
        "identity_overlap_zero": bool(
            np.intersect1d(identity_reference["train"], identity_reference["development"]).size == 0
        ),
        "O1_absolute_all_regimes": absolute["O1_regime_A0"],
        "O2_absolute_all_regimes": absolute["O2_regime_A1"],
        "O1_near_and_far_low_pearson_gain_at_least_0p20": bool(
            low_gain["O1_regime_A0"]["near_low"] >= 0.20
            and low_gain["O1_regime_A0"]["far_low"] >= 0.20
        ),
        "O2_near_and_far_low_pearson_gain_at_least_0p20": bool(
            low_gain["O2_regime_A1"]["near_low"] >= 0.20
            and low_gain["O2_regime_A1"]["far_low"] >= 0.20
        ),
        "at_least_one_oracle_passes_full_decision_rule": passed,
    }
    summary = {
        "experiment": "P3b ISRJ regime-conditioned feature-target sufficiency Oracle",
        "evidence_role": "post_failure_diagnostic_oracle_only",
        "protocol": vars(args),
        "preregister_sha256": sha256_file(preregister),
        "device": str(device),
        "identity_audit": {
            "train_sha256": _hash_int64(identity_reference["train"]),
            "development_sha256": _hash_int64(identity_reference["development"]),
            "cross_role_overlap": int(np.intersect1d(
                identity_reference["train"], identity_reference["development"]
            ).size),
        },
        "results": results,
        "low_width_pearson_gain_vs_global": low_gain,
        "diagnostic_gates": gates,
        "oracle_sufficient": passed,
        "interpretation": (
            "Regime-labelled upper bound passed. The raw features contain width ordering, but a future learned router requires entirely new identities and three-seed validation."
            if passed else
            "Even the regime-labelled upper bound failed the low-width and absolute gates. Stop raw periodic-sequence scalar-expert development."
        ),
        "method_promoted": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "oracle_sufficient": passed,
        "diagnostic_gates": gates,
        "low_width_pearson_gain_vs_global": low_gain,
        "interpretation": summary["interpretation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
