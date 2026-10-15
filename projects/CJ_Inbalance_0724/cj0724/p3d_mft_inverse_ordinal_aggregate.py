"""Recover the P3d summary from completed immutable seed receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import sha256_file
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3d_mft_inverse_ordinal import MODEL_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/p3d_mft_inverse_ordinal_development_20260801")
    parser.add_argument("--preregister", default="configs/p3d_mft_inverse_ordinal_preregister_20260801.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output_root = project / args.output_root
    runs = [json.loads((output_root / f"seed_{seed}" / "receipt.json").read_text(encoding="utf-8")) for seed in MODEL_SEEDS]
    feature = np.load(output_root / "feature_receipt.npz")
    gates = {
        "identity_overlap_zero": bool(np.intersect1d(feature["fit_identity"], feature["development_identity"]).size == 0 and np.intersect1d(feature["selection_identity"], feature["development_identity"]).size == 0),
        "all_15_reduction_vs_constant_at_least_0p10": bool(all(run["regimes"][regime]["overall"]["mae_reduction_vs_constant"] >= 0.10 for run in runs for regime in REGIMES)),
        "all_15_pearson_at_least_0p30": bool(all(run["regimes"][regime]["overall"]["pearson"] >= 0.30 for run in runs for regime in REGIMES)),
        "compound_all_15_reduction_vs_constant_at_least_0p10": bool(all(run["regimes"][regime]["compound"]["mae_reduction_vs_constant"] >= 0.10 for run in runs for regime in REGIMES)),
        "compound_all_15_pearson_at_least_0p30": bool(all(run["regimes"][regime]["compound"]["pearson"] >= 0.30 for run in runs for regime in REGIMES)),
    }
    summary = {
        "experiment": "P3d MFT inverse-filter delay-sequence ordinal expert",
        "recovery_note": "Summary reconstructed from completed seed receipts after a post-training Python boolean spelling error; no model was retrained and no prediction was changed.",
        "preregister_sha256": sha256_file(project / args.preregister),
        "constant_baseline": float(feature["constant"]),
        "identity_audit": {
            "fit_sha256": _hash_int64(feature["fit_identity"]),
            "selection_sha256": _hash_int64(feature["selection_identity"]),
            "development_sha256": _hash_int64(feature["development_identity"]),
            "fit_development_overlap": int(np.intersect1d(feature["fit_identity"], feature["development_identity"]).size),
            "selection_development_overlap": int(np.intersect1d(feature["selection_identity"], feature["development_identity"]).size),
        },
        "promotion_gates": gates,
        "development_passed": bool(all(gates.values())),
        "runs": runs,
        "interpretation": "Frozen MFT ordinal gates failed; do not attach and do not tune on seed470000000.",
        "classification_loss_used": False,
        "oracle_positive_only": True,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    (output_root / "summary_recovery_receipt.json").write_text(json.dumps({"source_seed_receipts": [str(output_root / f"seed_{seed}" / "receipt.json") for seed in MODEL_SEEDS], "predictions_changed": False, "models_retrained": False}, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
