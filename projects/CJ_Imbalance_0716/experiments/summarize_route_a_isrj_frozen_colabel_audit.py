"""Compact read-only console summary for the completed ISRJ co-label audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs/tapc/route_a_isrj_frozen_colabel_audit/route_a_isrj_frozen_colabel_audit.json"


def main() -> None:
    result = json.loads(PATH.read_text(encoding="utf-8"))
    for source in result["inputs"]:
        print(f"SOURCE={source['source']}")
        for group, aggregate in source["aggregate"].items():
            delta = aggregate["b2_minus_b0"]
            auroc = delta["weak_vs_negative_auroc"]
            recall = delta["weak_recall"]
            fpr = delta["negative_fpr"]
            print(
                f"{group:18s} auroc={auroc['mean']:+.5f} "
                f"ci=[{auroc['bootstrap_95_ci'][0]:+.5f},{auroc['bootstrap_95_ci'][1]:+.5f}] "
                f"recall={recall['mean']:+.5f} fpr={fpr['mean']:+.5f} "
                f"seeds={aggregate['supported_seed_count']}"
            )


if __name__ == "__main__":
    main()
