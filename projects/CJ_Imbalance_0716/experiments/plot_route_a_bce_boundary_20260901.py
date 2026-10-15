"""Create the preregistered BCE boundary figure from development-only receipts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/route_a_bce_boundary_screen_20260901"
OUT = ROOT / "outputs/route_a_longtail_goal_aggregate_20260901"


def load(condition: str):
    rows = []
    for path in sorted((SRC / condition).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("variant", "BCE") == "BCE":
            rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("label", [("IF=1", "label_IF1"), ("IF=2", "label_IF2"), ("IF=3", "label_IF3")]),
        ("combination", [("CF=1", "combo_CF1"), ("CF=4", "combo_CF4"), ("CF=8", "combo_CF8")]),
    ]
    metrics = [("macro_aupr", "Macro-AUPR"), ("macro_f1", "Macro-F1"),
               ("combination_exact_accuracy", "Exact-combination accuracy")]
    records = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
    colors = {"label": "#2864dc", "combination": "#d45a35"}
    for ax, (key, title) in zip(axes, metrics):
        for family, conditions in specs:
            means, stds, labels = [], [], []
            for severity, condition in conditions:
                vals = np.asarray([r["validation"][key] for r in load(condition)], dtype=float)
                means.append(float(vals.mean())); stds.append(float(vals.std(ddof=1)))
                labels.append(severity)
                records.append({"family": family, "severity": severity, "condition": condition,
                                "metric": key, "mean": means[-1], "std": stds[-1], "n": len(vals)})
            x = np.arange(len(labels))
            offset = -0.04 if family == "label" else 0.04
            ax.errorbar(x + offset, means, yerr=stds, marker="o", linewidth=1.8,
                        capsize=3, color=colors[family], label=family)
            for xi, yi, label in zip(x + offset, means, labels):
                ax.annotate(label, (xi, yi), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7, color=colors[family])
        ax.set_title(title); ax.set_xticks([]); ax.set_ylim(0, 1.03); ax.grid(alpha=.2)
    axes[0].set_ylabel("score")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.suptitle("Route-A ordinary BCE boundary (development-v2, 3 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "bce_boundary.png", bbox_inches="tight")
    plt.close(fig)
    with (OUT / "bce_boundary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print(json.dumps({"status": "completed", "figure": str((OUT/'bce_boundary.png').resolve()),
                      "table": str((OUT/'bce_boundary.csv').resolve()),
                      "locked_test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
