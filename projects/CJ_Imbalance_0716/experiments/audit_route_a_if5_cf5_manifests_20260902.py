"""Audit the original nominal IF=5/CF=5 Route-A manifests read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/route_a_p3_dual_axis_longtail_20260828"
OUT = ROOT / "outputs/route_a_if5_cf5_manifest_audit_20260902"
NAMES = ["FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ"]


def audit(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    labels = np.asarray(d["label_counts"], dtype=int)
    combos = {str(k): int(v) for k, v in d["combo_counts"].items()}
    tiers = {str(k): str(v) for k, v in d.get("combo_tiers", {}).items()}
    weak = np.asarray(d.get("weak_label_counts") or [0] * len(labels), dtype=int)
    label_tiers = {str(k): str(v) for k, v in d.get("label_tiers", {}).items()}
    head_keys = [k for k, v in tiers.items() if v == "head"]
    middle_keys = [k for k, v in tiers.items() if v == "middle"]
    tail_keys = [k for k, v in tiers.items() if v == "tail"]
    singleton = [k for k in combos if sum(int(x) for x in k) == 1]
    top10 = sorted(combos, key=lambda k: (-combos[k], k))[: max(1, int(np.ceil(len(combos) * .10)))]
    label_tier_positive_incidences = {
        tier: {NAMES[int(k)]: int(labels[int(k)]) for k, v in label_tiers.items() if v == tier}
        for tier in ("head", "middle", "tail")
    }
    return {
        "manifest": path.name,
        "seed": int(d["seed"]),
        "nominal_if": float(d.get("imbalance_factor", 5.0)),
        "nominal_cf": 5.0,
        "selected_count": int(d.get("selected_count", sum(combos.values()))),
        "label_counts": {n: int(v) for n, v in zip(NAMES, labels)},
        "label_tier_positive_incidences": label_tier_positive_incidences,
        "effective_if_max_over_min": float(labels.max() / labels.min()),
        "combo_max": int(max(combos.values())),
        "combo_min": int(min(combos.values())),
        "effective_cf_max_over_min": float(max(combos.values()) / min(combos.values())),
        "unique_combo_count": int(len(combos)),
        "combo_tier_unique": {"head": len(head_keys), "middle": len(middle_keys), "tail": len(tail_keys)},
        "combo_tier_samples": {
            "head": int(sum(combos.get(k, 0) for k in head_keys)),
            "middle": int(sum(combos.get(k, 0) for k in middle_keys)),
            "tail": int(sum(combos.get(k, 0) for k in tail_keys)),
        },
        "weak_component_counts": {n: int(v) for n, v in zip(NAMES, weak)},
        "weak_component_total": int(weak.sum()),
        "weak_component_sample_count_lower_bound": int(weak.max()) if len(weak) else 0,
        "singleton_combo_unique": int(len(singleton)),
        "singleton_combo_samples": int(sum(combos[k] for k in singleton)),
        "top10_frequency_combo_unique": int(len(top10)),
        "top10_frequency_combo_samples": int(sum(combos[k] for k in top10)),
        "top_frequency_combos": [[k, combos[k]] for k in sorted(combos, key=lambda k: (-combos[k], k))[:5]],
        "bottom_frequency_combos": [[k, combos[k]] for k in sorted(combos, key=lambda k: (combos[k], k))[:5]],
        "head_combo_keys": head_keys,
        "singleton_combo_keys": singleton,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit(p) for p in sorted(SRC.glob("longtail_manifest_seed_*.json"))]
    payload = {
        "schema": "route-a-if5-cf5-manifest-audit-v1",
        "source": str(SRC.resolve()),
        "nominal_protocol": {"label_if": 5.0, "combo_if": 5.0, "weak_threshold_db": 10.0},
        "definitions": {
            "effective_if": "max actual positive-label count / min actual positive-label count",
            "effective_cf": "max actual combination count / min actual combination count over present combinations",
            "head/middle/tail": "the combo_tiers stored in each manifest (frequency-ranked thirds)",
            "weak_component_sample_count": "manifest weak_label_counts are component incidences; union sample count is not recoverable from this manifest and is reported as a lower bound",
            "easy_combinations": "singleton combinations (one active label) and top-10%-frequency combinations; both are reported separately",
        },
        "manifests": rows,
        "locked_test_used": False,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# IF=5 / CF=5 manifest audit (read-only)", "",
        "名义协议：label IF=5、combination CF=5、弱分量阈值 10 dB。所有数值来自 manifest 的实际字段；未读取 locked test。", "",
        "| manifest | N | 每类正样本 FMB/AMA/SawFM/SinFM/MFT/ISRJ | 有效 IF | combo min/max | 有效 CF | 唯一组合 | head/middle/tail 样本 | 弱分量计数 | 单标签组合样本 | top10%组合样本 |",
        "|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in rows:
        c = r["combo_tier_samples"]
        lines.append(f"| {r['manifest']} | {r['selected_count']} | "
                     f"{r['label_counts']['FMB']}/{r['label_counts']['AMA']}/{r['label_counts']['SawFM']}/{r['label_counts']['SinFM']}/{r['label_counts']['MFT']}/{r['label_counts']['ISRJ']} | "
                     f"{r['effective_if_max_over_min']:.3f} | {r['combo_min']}/{r['combo_max']} | {r['effective_cf_max_over_min']:.3f} | {r['unique_combo_count']} | "
                     f"{c['head']}/{c['middle']}/{c['tail']} | {r['weak_component_counts']} (sum={r['weak_component_total']}) | "
                     f"{r['singleton_combo_samples']} | {r['top10_frequency_combo_samples']} |")
    lines += ["", "## 判读", "", "- manifest 中实际标签计数和组合计数均不是名义 IF=5、CF=5；有效 IF/CF 以表中计算值为准。", "- `head/middle/tail` 是按该 manifest 的组合频率排序后三等分，不是物理难度标签。", "- `weak_label_counts` 记录弱分量发生次数；由于旧 manifest 未保存逐样本 weak mask，不能把其总和直接称为独立样本数。此次三个 manifest 的弱分量计数均为 0，因此至少可以确认没有被记录的弱分量正例。", "- 单标签组合样本和 top-10% 高频组合样本占比用于衡量训练集是否仍被容易识别的组合主导；二者不能等同于模型实际难度，但可作为数据覆盖审计证据。"]
    (OUT / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "manifests": len(rows), "audit": str((OUT/'audit.json').resolve()), "locked_test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
