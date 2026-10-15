"""Freeze the P2n analytic SawFM full-system candidate and its evidence hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cj0724.integrate_swept_experts_with_six_class import _jsonable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_receipt(project: Path, p2n_summary_path: Path) -> dict[str, object]:
    p2n = json.loads(p2n_summary_path.read_text(encoding="utf-8"))
    if not p2n.get("promotion_passed", False):
        raise RuntimeError("P2n did not pass")
    p2m_summary_path = project / "outputs/p2m_analytic_saw_period_physics_20260801/summary.json"
    p2m = json.loads(p2m_summary_path.read_text(encoding="utf-8"))
    if not p2m.get("promotion_passed", False):
        raise RuntimeError("P2m did not pass")
    artifacts = {
        "p2k_method_freeze": project / "outputs/p2k_full_parameter_confirmation_20260801/method_freeze.json",
        "p2m_preregister": project / "configs/p2m_analytic_saw_period_preregister_20260801.json",
        "p2m_implementation": project / "cj0724/p2m_analytic_saw_period.py",
        "p2m_summary": p2m_summary_path,
        "p2n_preregister": project / "configs/p2n_analytic_saw_full_system_preregister_20260801.json",
        "p2n_implementation": project / "cj0724/p2n_analytic_saw_full_system.py",
        "p2n_summary": p2n_summary_path,
    }
    return {
        "candidate": "P2n frozen P2f separator plus fixed FFT-ACF SawFM period attachment",
        "analytic_rule": p2m["algorithm"],
        "attachment_rule": "Use the frozen P2a SawFM predicted-label gate; run the seed-matched P2f S1 separator; replace only SawFM sweep period with the fixed P2m FFT-ACF estimate.",
        "unchanged_components": [
            "MLCNN classification probabilities",
            "IQ-Grouped base parameter outputs",
            "P2a thresholds and lower-confidence ISRJ route",
            "VFT aligned expert",
            "the other eleven parameter outputs",
        ],
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "physical_promotion_gates": p2m["promotion_gates"],
        "system_promotion_gates": p2n["promotion_gates"],
        "system_per_regime": p2n["per_regime"],
        "method_changes_allowed_after_freeze": False,
        "threshold_changes_allowed_after_freeze": False,
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
        "next_required": [
            "two new identity seeds across all five ID/near/far regimes",
            "weak-component and combination-disjoint confirmation",
            "hash-recorded long-tail derived manifest",
            "batch-1 and batch-64 latency, throughput, FLOPs and peak-memory audit",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2n-summary", default="outputs/p2n_analytic_saw_full_system_20260801/summary.json")
    parser.add_argument("--output", default="outputs/p2n_analytic_saw_full_system_20260801/method_freeze.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = project / args.output
    receipt = build_receipt(project, project / args.p2n_summary)
    output.write_text(json.dumps(_jsonable(receipt), indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()), "sha256": sha256_file(output),
        "candidate": receipt["candidate"], "locked_test_used": receipt["locked_test_used"],
    }, indent=2))


if __name__ == "__main__":
    main()
