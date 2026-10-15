"""Write an immutable receipt for the P2k-confirmed full-parameter candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p2j_full_parameter_attachment import COEFFICIENTS, MODEL_SEEDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_receipt(project: Path, confirmation_summary: Path) -> dict[str, object]:
    summary = json.loads(confirmation_summary.read_text(encoding="utf-8"))
    if not summary.get("confirmation_passed", False):
        raise RuntimeError("P2k confirmation did not pass")
    reference_root = project / "outputs/p2a_nine_power_generalization_20260731"
    separator_root = project / "outputs/p2f_dedicated_sawfm_separator_development_v5_20260801"
    specialist_root = project / "outputs/physics_parameter_specialists/sawfm_period_sequence"
    seed_artifacts = []
    for seed in MODEL_SEEDS:
        separator = separator_root / f"seed_{seed}" / "S1_waveform_only" / "best.pt"
        period = specialist_root / f"seed_{seed}" / "best.pt"
        p2a_seed_freeze = reference_root / f"seed_{seed}" / "seed_freeze.json"
        seed_artifacts.append({
            "seed": seed,
            "separator_checkpoint": str(separator.resolve()),
            "separator_checkpoint_sha256": sha256_file(separator),
            "period_expert_checkpoint": str(period.resolve()),
            "period_expert_checkpoint_sha256": sha256_file(period),
            "p2a_seed_freeze": str(p2a_seed_freeze.resolve()),
            "p2a_seed_freeze_sha256": sha256_file(p2a_seed_freeze),
            "affine_a": COEFFICIENTS[seed][0],
            "affine_b": COEFFICIENTS[seed][1],
        })
    evidence_paths = {
        "p2a_reference_method": reference_root / "method_freeze.json",
        "p2i_confirmation": project / "outputs/p2i_sawfm_calibrated_separator_confirmation_20260801/summary.json",
        "p2j_development": project / "outputs/p2j_full_parameter_predicted_gate_attachment_20260801/summary.json",
        "p2k_confirmation": confirmation_summary,
        "p2j_preregister": project / "configs/p2j_full_parameter_predicted_gate_attachment_preregister_20260801.json",
        "p2k_preregister": project / "configs/p2k_two_seed_full_parameter_confirmation_preregister_20260801.json",
    }
    return {
        "candidate": "P2k-confirmed calibrated separated-SawFM full-parameter attachment",
        "attachment_rule": (
            "Run the seed-matched P2f S1 separator and unchanged SawFM period expert only when the frozen "
            "P2a SawFM predicted-label gate is positive; apply the seed-matched fixed affine coefficient; "
            "replace only SawFM sweep period."
        ),
        "frozen_unchanged_components": [
            "MLCNN classification backbone and probabilities",
            "IQ-Grouped base parameter outputs",
            "P2a thresholds",
            "P2a lower-confidence ISRJ route",
            "VFT aligned Doppler expert",
            "the other eleven parameter outputs",
        ],
        "seed_artifacts": seed_artifacts,
        "evidence": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in evidence_paths.items()
        },
        "confirmation_gates": summary["confirmation_gates"],
        "confirmation_aggregate": summary["aggregate"],
        "combined_identity_bootstrap": summary["combined_identity_bootstrap"],
        "identity_audit": summary["identity_audit"],
        "method_changes_allowed_after_freeze": False,
        "threshold_changes_allowed_after_freeze": False,
        "confirmation_seed_selection_allowed": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
        "next_evidence_required": [
            "fresh template-mismatch and mechanism-OOD evaluation",
            "fresh weak-component boundary evaluation",
            "hash-recorded derived long-tail manifest evaluation",
            "end-to-end latency and parameter-count audit on deployment hardware",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-summary", default="outputs/p2k_full_parameter_confirmation_20260801/summary.json")
    parser.add_argument("--output", default="outputs/p2k_full_parameter_confirmation_20260801/method_freeze.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    confirmation_summary = project / args.confirmation_summary
    output = project / args.output
    receipt = build_receipt(project, confirmation_summary)
    output.write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "sha256": sha256_file(output),
        "candidate": receipt["candidate"],
        "locked_test_used": receipt["locked_test_used"],
    }, indent=2))


if __name__ == "__main__":
    main()
