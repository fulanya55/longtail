"""Record the permutation-equivalence audit for the v25 cyclic-source control."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = ROOT / "outputs/tapc/route_a_v25_target_specific_complex_evidence_probe/route_a_v25_target_specific_complex_evidence_probe_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "status": "completed",
        "experiment": "route_a_v25_cyclic_source_control_equivalence_audit",
        "finding": "invalid_for_target_specific_causal_interpretation",
        "reason": [
            "All three local branches have identical architecture.",
            "The cyclic control routes each target loss to a different branch but does not remove target-label supervision.",
            "A cyclic permutation of candidate local-branch parameters makes candidate and control logits exactly equal and local-only BCE updates preserve that equality.",
            "The formal v25 runner loaded independently random branch states rather than the matched permutation, so its candidate-control deltas can include branch-initialization assignment effects.",
        ],
        "historical_v25_summary_sha_source": str(summary_path),
        "historical_v25_gate": summary["gate"],
        "required_disposition": "Do not cite v25 MFT/VFT/ISRJ deltas as source-controlled target-specific evidence. Preserve the artifact as an exploratory architecture screen only; do not launch MFT confirmation based on it.",
        "validated_by": "tests/test_target_specific_complex_evidence_control_equivalence.py",
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    path = ROOT / "outputs/tapc/route_a_v25_cyclic_source_control_equivalence_audit/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
